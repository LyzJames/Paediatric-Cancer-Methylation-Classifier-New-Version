# Databricks notebook source
import mlflow
import json
import numpy as np
import polars as pl
import pandas as pd

# COMMAND ----------

def predict_node_from_mlflow(
    node_name: str,
    mvalue_df: pl.DataFrame,
    eval_dir: str,
    model_stage: str = "Production",   # Or "Staging", or a version like "1"
):
    """
    Load a node classifier from MLflow and run predictions on a new methylation dataset.
    
    Inputs:
        node_name: the exact node name used during training (e.g., "Leukaemia")
        mvalue_df: Polars DataFrame containing full methylation matrix:
                   columns: [biosample_id, cg... probes]
        eval_dir: directory where <node_name>_summary.json is stored
        model_stage: MLflow model stage ("Production", "Staging", or version string like "1")
    
    Output:
        pandas DataFrame with:
          - biosample_id
          - predicted_label
          - probabilities for each class
    """

    # Load metadata (selected probes, classes, etc.)
    summary_path = f"{eval_dir}/{node_name}_summary.json"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    selected_probes = summary["selected_probes"]
    classes = summary["classes"]

    print(f"[INFO] Loaded {len(selected_probes)} selected probes.")
    print(f"[INFO] Classes: {classes}")

    # Prepare input data: filter only probe columns needed by model
    # Ensure the probes exist in the incoming DataFrame
    missing = [p for p in selected_probes if p not in mvalue_df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing probes in new data: {missing[:5]} ... TOTAL={len(missing)}")

    # Keep biosample_id separately
    biosample_ids = mvalue_df["biosample_id"].to_list()

    # Only keep selected probes in the SAME ORDER used in training
    X_new_df = (
        mvalue_df
        .select(selected_probes)
        .fill_nan(0)
        .fill_null(0)
        .to_pandas()
    )

    # convert to NumPy to avoid feature-name checks
    X_new = X_new_df.to_numpy()

    # Load model from MLflow Model Registry
    registry_name = f"peds_methylation_{node_name.replace(' ', '_')}"

    print(f"[INFO] Loading MLflow model: {registry_name} (stage={model_stage})")

    model_uri = f"models:/{registry_name}/{model_stage}"
    model = mlflow.sklearn.load_model(model_uri)

    print("[INFO] Model loaded successfully from MLflow.")

    # Predict
    proba = model.predict_proba(X_new)
    pred_idx = np.argmax(proba, axis=1)
    pred_labels = [classes[i] for i in pred_idx]

    # Build result DataFrame
    out = pd.DataFrame({
        "biosample_id": biosample_ids,
        "predicted_label": pred_labels,
    })

    # Add probability columns
    for i, cls in enumerate(classes):
        out[f"proba_{cls}"] = proba[:, i]

    return out

# COMMAND ----------

# Example new methylation rows to classify (Polars DataFrame)
new_samples = pl.read_csv("/Volumes/cb_prod/comp9300-9900-f16b-donut/9900-f16b-donut/data/ mvalue_outputs_masked_subset_leukaemia/MValue_concat.csv")

# COMMAND ----------

pred_df = predict_node_from_mlflow(
    node_name="Leukaemia",
    mvalue_df=new_samples,
    eval_dir="/Workspace/9900-f16b-donut/sprint3/evaluate",    # directory containing JSON summaries
    model_stage="Production"            # or "Staging" or "1"
)

pred_df