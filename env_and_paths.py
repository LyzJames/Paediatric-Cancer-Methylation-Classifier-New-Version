# Databricks notebook source
# Core
import sys, gc, types, os, json, math, textwrap, datetime as dt
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
import psutil
from mlflow.models.signature import infer_signature
from math import ceil
from disease_tree import DiseaseTree
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

# Data
import polars as pl
import pyarrow.feather as feather
import pyarrow as pa

# ML
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, balanced_accuracy_score, f1_score, classification_report, confusion_matrix, log_loss

# R / limma (feature selection)
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri, packages as rpackages
numpy2ri.activate(); pandas2ri.activate()
from rpy2.robjects.packages import importr

# Save model bundle (single file)
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Random seeds for reproducibility
RANDOM_STATE = 42

# COMMAND ----------

def ensure_limma(userlib="/databricks/rlibs"):
    # Just point R to the right user library and import
    ro.r(f'''
    dir.create("{userlib}", recursive = TRUE, showWarnings = FALSE)
    .libPaths(c("{userlib}", .libPaths()))
    ''')
    return importr("limma")

limma = ensure_limma()
print("limma loaded:", limma.__version__)

# COMMAND ----------

# Ensure required R packages are available: limma, stats
limma = importr("limma")
base  = importr("base")
stats = importr("stats")

# COMMAND ----------

# <<< EDIT THESE THREE PATHS >>>
TREE_JOBLIB = "./data/freeze0525/diseaseTree_mapped.joblib"
MVALUE_CSV = "/Volumes/cb_prod/comp9300-9900-f16b-donut/9900-f16b-donut/data/ mvalue_outputs_masked_subset_leukaemia/MValue_concat.csv"

# Data
DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Tasks (nodes / task lists saved as joblib)
OUT_TASKS_DIR = Path("./tasks")
OUT_TASKS_DIR.mkdir(parents=True, exist_ok=True)

# Evaluation outputs (metrics, curves, logs)
OUT_EVAL_DIR = Path("./evaluate")
OUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Sanity
assert Path(TREE_JOBLIB).exists(), "Tree JSON path not found"
assert Path(MVALUE_CSV).exists(), "MValue CSV path not found"

# COMMAND ----------

# --- Map sample IDs to child class names ---
def id_to_child(ids: list[str], node) -> dict[str, str]:
    """
    Map each sample ID in 'ids' to the child name it belongs to.
    Returns a dict {sample_id: child_name}.
    Logs warnings for IDs not found in any child.
    """
    child_map = child_membership(node)
    out, skipped = {}, []

    for sid in ids:
        found = False
        for cname, sset in child_map.items():
            if sid in sset:
                out[sid] = cname
                found = True
                break
        if not found:
            skipped.append(sid)

    if skipped:
        print(f"[WARN] {node.name}: {len(skipped)} ids not in any child.samples — skipped")

    return out

# Build child membership map for a DiseaseTree node
def child_membership(node) -> dict[str, set[str]]:
    """
    Return {child_name: set(sample_ids)} for direct children of a DiseaseTree node.
    Samples are aggregated from each child's samples lists.
    """
    out = {}
    for ch in getattr(node, "children", []) or []:
        sset = set(getattr(ch, "samples", []) or [])
        out[ch.name] = sset
    return out

# Build train/val lists for a parent node (union of children)
def node_train_val_ids(node: Dict) -> Tuple[List[str], List[str], List[str]]:
    train_ids, val_ids, all_ids = set(), set(), set()
    for ch in node.get("children", []):
        train_ids.update(ch.get("training_samples", []))
        val_ids.update(ch.get("validation_samples", []))
        all_ids.update(ch.get("samples", []))
    return list(train_ids), list(val_ids), list(all_ids)

# Labels csv (optional sanity)
def load_labels_map(labels_csv: str) -> Dict[str, str]:
    df = pl.read_csv(labels_csv, infer_schema_length=0)
    return dict(zip(df["biosample_id"].to_list(), df["ground_truth"].to_list()))

def load_matrix_rows_for_ids(
    mvalue_source: Union[str, Path, pl.DataFrame],
    wanted_ids: List[str]
) -> pl.DataFrame:

    if not wanted_ids:
        return pl.DataFrame()

    id_df = pl.DataFrame({"biosample_id": wanted_ids})

    # CASE A — mvalue_source is already an in-memory DF
    if isinstance(mvalue_source, pl.DataFrame):
        return mvalue_source.join(id_df, on="biosample_id", how="semi")

    # CASE B — mvalue_source is a file path
    scan = pl.scan_csv(mvalue_source)

    # Lazy semi-join → only loads rows we need
    filtered = scan.join(id_df.lazy(), on="biosample_id", how="semi")

    # Actually read data into memory
    return filtered.collect()


def df_to_numpy_Xy(df: pl.DataFrame, id_to_class: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert Polars DF to (X, y, feature_names) using id_to_class mapping.
    Drops rows not in id_to_class.
    """
    df = df.filter(pl.col("biosample_id").is_in(list(id_to_class.keys())))
    if df.height == 0:
        return np.empty((0,0)), np.array([]), []
    # biosample_id then probe columns
    probe_cols = [c for c in df.columns if c != "biosample_id"]
    X = df.select(probe_cols).to_numpy()
    y = np.array([id_to_class[_id] for _id in df["biosample_id"].to_list()])
    return X, y, probe_cols

# COMMAND ----------

@dataclass
class ModelBundle:
    node_name: str
    classes: List[str]                 # class names in order
    probe_names: List[str]             # selected features (column names)
    classifier: object                 # CalibratedClassifierCV
    created_at: str

def save_bundle(bundle: ModelBundle, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{name}.joblib"
    joblib.dump(bundle, file_path, compress=3)
    return file_path

def load_bundle(path: Path) -> ModelBundle:
    return joblib.load(path)