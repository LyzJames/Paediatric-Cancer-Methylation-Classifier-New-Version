# Databricks notebook source
class DifferentialMethylationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dm_script_path="./differentialMethylation.R", dm_top_n=50):
        self.dm_script_path = dm_script_path
        self.dm_top_n = dm_top_n
        self.selected_probes_ = None

    def fit(self, X, y):
        """
        X is a numpy array or DataFrame of probe values.
        y is a list of class labels.
        """
        # Load R script
        ro.r["source"](self.dm_script_path)
        runDM_multi = ro.globalenv["runDM_multi"]

        # Convert to numpy float64
        X_np = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        labels_r = ro.StrVector(list(y))
        probe_names = list(X.columns)
        probe_names_r = ro.StrVector(probe_names)

        # Run DM
        res = runDM_multi(X_np, labels_r, probe_names_r, self.dm_top_n, True)
        selected_union = list(res.rx2("selected_union"))

        # Save selected probes for use in transform()
        self.selected_probes_ = selected_union
        return self

    def transform(self, X):
        """
        Select only columns in selected_probes_.
        X is DataFrame or numpy array.
        """
        if isinstance(X, pd.DataFrame):
            cols = [c for c in X.columns if c in self.selected_probes_]
            return X[cols]
        else:
            # If numpy: probe names are cg0,cg1,... in order
            idx = [int(c.replace("cg","")) for c in self.selected_probes_]
            return X[:, idx]

# COMMAND ----------

def train_single_node_pipeline(
    node,  # DiseaseTree object
    target_node_name: str,
    mvalue_csv: pl.DataFrame,
    eval_dir: Path,
    *,
    random_state: int = RANDOM_STATE,
    test_size: float = 0.2,  # kept for API compatibility, not used directly
    dm_script_path: str = "./differentialMethylation.R",
    dm_top_n: int = 50,
):
    """
    Full end-to-end pipeline with:
      - Loading methylation data for a single DiseaseTree node
      - R-based differential methylation (DM) as a sklearn Pipeline step
      - RandomForest + GridSearchCV
      - Calibration
      - Save:
          * metrics + best hyperparameters + training metadata as JSON in eval_dir
          * evaluation report (.txt) in eval_dir
          * trained model in MLflow (NOT in out_models_dir)
    """
    # CHILD LABELS
    child_map = child_membership(node)
    child_names = list(child_map.keys())
    print(f"Child names: {child_names}")

    # LOAD MATRIX
    train_ids = list(node.training_samples or [])
    val_ids   = list(node.validation_samples or [])

    train_id2y = id_to_child(train_ids, node)
    val_id2y   = id_to_child(val_ids, node)

    # KEEP ONLY labeled IDs
    train_ids = list(train_id2y.keys())
    val_ids   = list(val_id2y.keys())

    train_df = load_matrix_rows_for_ids(mvalue_csv, train_ids)
    val_df   = load_matrix_rows_for_ids(mvalue_csv, val_ids)

    loaded_train_ids = set(train_df["biosample_id"].to_list())
    loaded_val_ids   = set(val_df["biosample_id"].to_list())

    train_ids_set = set(train_ids)
    val_ids_set   = set(val_ids)

    # Detect missing IDs
    missing_train = train_ids_set - loaded_train_ids
    missing_val   = val_ids_set - loaded_val_ids

    if missing_train or missing_val:
        print("\n[WARN] Data not complete for this node!")
        print(" - Missing train sample IDs:", sorted(missing_train))
        print(" - Missing val sample IDs:", sorted(missing_val))
        print(" - Loaded train:", len(loaded_train_ids),
              ", Expected:", len(train_ids))
        print(" - Loaded val:", len(loaded_val_ids),
              ", Expected:", len(val_ids))

    # === Per-child checks: each child must have >= 4 samples total ===
    for child_name in child_names:
        child_samples = child_map.get(child_name, set())

        train_child_ids = child_samples & loaded_train_ids
        val_child_ids   = child_samples & loaded_val_ids

        tr = len(train_child_ids)
        va = len(val_child_ids)

        if tr + va < 4:
            print("\n[ERROR] Requirement: each class/child must have at least 4 samples in dataset.")
            return f"[FAILED] Node {target_node_name}"

    print("[INFO] Shapes (train_df, val_df):", train_df.shape, val_df.shape)
    mem = psutil.virtual_memory()
    print(f"[MEM] After load: {mem.available/1024**3:.2f} GB")

    # CAST TO NUMERIC (Polars → pandas)
    train_num = train_df.with_columns(
        pl.all().exclude("biosample_id").cast(pl.Float64, strict=False)
    )
    val_num = val_df.with_columns(
        pl.all().exclude("biosample_id").cast(pl.Float64, strict=False)
    )

    # free original dfs
    del train_df, val_df
    gc.collect()

    mem = psutil.virtual_memory()
    print(f"[MEM] After cast: {mem.available/1024**3:.2f} GB")

    probe_cols_full = [c for c in train_num.columns if c != "biosample_id"]

    # Fill NaN/null with 0, then convert to pandas
    train_X_df = (
        train_num
        .select(probe_cols_full)
        .fill_nan(0)
        .fill_null(0)
        .to_pandas()
    )
    val_X_df = (
        val_num
        .select(probe_cols_full)
        .fill_nan(0)
        .fill_null(0)
        .to_pandas()
    )

    # Build y labels aligned with DataFrame rows
    train_bsid = train_num["biosample_id"].to_list()
    val_bsid   = val_num["biosample_id"].to_list()

    y_train = [train_id2y[sid] for sid in train_bsid]
    y_val   = [val_id2y[sid] for sid in val_bsid]

    # We no longer need Polars numeric dfs
    del train_num, val_num
    gc.collect()

    mem = psutil.virtual_memory()
    print(f"[MEM] After pandas conversion: {mem.available/1024**3:.2f} GB")
    print(f"Train samples: {len(y_train)}, Val samples: {len(y_val)}")
    print(f"Train feature cols: {train_X_df.shape[1]}")

    #              SKLEARN PIPELINE: DM → RandomForest

    dm_step = DifferentialMethylationSelector(
        dm_script_path=dm_script_path,
        dm_top_n=dm_top_n,
    )

    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=1,  # keep 1 to avoid OOM; you can tune later
    )

    pipeline = Pipeline([
        ("differentialMethylation", dm_step),
        ("modelGeneration", rf),
    ])

    param_grid = {
        "modelGeneration__n_estimators": [300, 600],
        "modelGeneration__max_depth": [None, 20, 40],
        "modelGeneration__max_features": ["sqrt", 0.3],
        "modelGeneration__min_samples_leaf": [1, 2, 5],
        "modelGeneration__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state,
    )

    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        refit=True,
        verbose=2,
        error_score="raise",
    )

    mem = psutil.virtual_memory()
    print(f"[MEM] Before GridSearchCV: {mem.available/1024**3:.2f} GB")
    print("[INFO] Running Pipeline(GridSearchCV) with DM → RF ...")
    gs.fit(train_X_df, y_train)

    best_pipeline = gs.best_estimator_
    dm_best = best_pipeline.named_steps["differentialMethylation"]
    best_rf = best_pipeline.named_steps["modelGeneration"]
    selected_probes = dm_best.selected_probes_

    print("[INFO] Best RF params:", gs.best_params_)
    print("[INFO] Selected probes:", len(selected_probes))

    #              CALIBRATION

    # Transform train/val with the fitted DM step
    X_train_dm = dm_best.transform(train_X_df)
    X_val_dm   = dm_best.transform(val_X_df)

    cal = CalibratedClassifierCV(best_rf, method="sigmoid", cv=3)
    cal.fit(X_train_dm, y_train)
    mem = psutil.virtual_memory()
    print(f"[MEM] After calibration: {mem.available/1024**3:.2f} GB")

    #              METRICS

    X_train_np = X_train_dm.to_numpy()
    X_val_np   = X_val_dm.to_numpy()
    y_train_np = np.asarray(y_train)
    y_val_np   = np.asarray(y_val)

    classes = np.array(cal.classes_)
    n_classes = len(classes)

    metrics = {}

    # ---- TRAIN ----
    train_proba = cal.predict_proba(X_train_np)
    train_pred  = classes[np.argmax(train_proba, axis=1)]

    metrics["train"] = {
        "accuracy": float(accuracy_score(y_train_np, train_pred))
    }

    if n_classes == 2:
        pos_label = classes[1]
        pos_idx = list(classes).index(pos_label)

        train_proba_pos = train_proba[:, pos_idx]
        train_y_bin = (y_train_np == pos_label).astype(int)

        metrics["train"].update({
            "roc_auc": float(roc_auc_score(train_y_bin, train_proba_pos)),
            "brier_score": float(brier_score_loss(train_y_bin, train_proba_pos)),
        })
    else:
        metrics["train"]["log_loss"] = float(
            log_loss(y_train_np, train_proba, labels=classes)
        )
        y_train_onehot = label_binarize(y_train_np, classes=classes)
        brier_train = float(np.mean((train_proba - y_train_onehot) ** 2))
        metrics["train"]["brier_score"] = brier_train

    # ---- VALIDATION ----
    val_proba = cal.predict_proba(X_val_np)
    val_pred  = classes[np.argmax(val_proba, axis=1)]

    metrics["val"] = {
        "accuracy": float(accuracy_score(y_val_np, val_pred))
    }

    if n_classes == 2:
        pos_label = classes[1]
        pos_idx = list(classes).index(pos_label)

        val_proba_pos = val_proba[:, pos_idx]
        val_y_bin = (y_val_np == pos_label).astype(int)

        metrics["val"].update({
            "roc_auc": float(roc_auc_score(val_y_bin, val_proba_pos)),
            "brier_score": float(brier_score_loss(val_y_bin, val_proba_pos)),
        })
    else:
        metrics["val"]["log_loss"] = float(
            log_loss(y_val_np, val_proba, labels=classes)
        )
        y_val_onehot = label_binarize(y_val_np, classes=classes)
        brier_val = float(np.mean((val_proba - y_val_onehot) ** 2))
        metrics["val"]["brier_score"] = brier_val

    val_report = classification_report(y_val_np, val_pred, digits=3)
    val_cm = confusion_matrix(y_val_np, val_pred, labels=child_names)
    print("\n[VAL] classification report:\n", val_report)

    # SAVE STRUCTURED SUMMARY (metrics + hyperparams + metadata)
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "node_name": target_node_name,
        "classes": child_names,
        "n_classes": len(child_names),
        "dm_top_n": dm_top_n,
        "dm_script_path": dm_script_path,
        "random_state": random_state,
        "train_size": len(y_train),
        "val_size": len(y_val),
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "best_params": gs.best_params_,
        "selected_probes": list(selected_probes),
    }

    summary_path = eval_dir / f"{target_node_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f_summary:
        json.dump(summary, f_summary, indent=2)
    print("[INFO] Saved summary JSON:", summary_path)

    run = mlflow.active_run()
    if run is not None:
        # Log metrics (flatten train/val)
        # summary["metrics"] looks like: {"train": {...}, "val": {...}}
        flat_metrics = {}
        for split_name, split_metrics in summary["metrics"].items():
            for k, v in split_metrics.items():
                if v is not None:
                    flat_metrics[f"{split_name}_{k}"] = float(v)

        if flat_metrics:
            mlflow.log_metrics(flat_metrics)

        # Log hyperparameters + training config
        # best RF params from GridSearchCV
        flat_params = {}
        for k, v in summary.get("best_params", {}).items():
            flat_params[f"rf_{k}"] = v

        # add training metadata as params too
        flat_params.update({
            "dm_top_n": summary.get("dm_top_n"),
            "dm_script_path": summary.get("dm_script_path"),
            "random_state": summary.get("random_state"),
            "train_size": summary.get("train_size"),
            "val_size": summary.get("val_size"),
            "n_selected_probes": len(summary.get("selected_probes", [])),
        })

        mlflow.log_params(flat_params)

        # Log tags (for easier filtering)
        mlflow.set_tag("node_name", summary.get("node_name", target_node_name))
        mlflow.set_tag("classes", ",".join(summary.get("classes", [])))
        mlflow.set_tag("status", "SUCCESS")

        print("[MLflow] Logged metrics, params, and tags.")
    else:
        print("[MLflow] No active run; skipping MLflow logging.")

    # SAVE EVALUATION REPORT (human-readable)
    eval_path = eval_dir / f"{target_node_name}.txt"
    with eval_path.open("w", encoding="utf-8") as f:
        f.write(f"Node: {target_node_name}\n")
        f.write(f"Classes: {child_names}\n\n")

        f.write("=== Train metrics ===\n")
        json.dump(metrics["train"], f, indent=2)
        f.write("\n\n")

        f.write("=== Validation metrics ===\n")
        json.dump(metrics["val"], f, indent=2)
        f.write("\n\n")

        f.write("=== Validation classification report ===\n")
        f.write(val_report)
        f.write("\n\n")

        f.write("=== Validation confusion matrix ===\n")
        f.write("rows/cols = child_names in order above\n")
        f.write(np.array2string(val_cm, max_line_width=120))
        f.write("\n")

    print("[INFO] Saved evaluation report:", eval_path)

    # SAVE MODEL IN MLFLOW (NOT in out_models_dir)
    run = mlflow.active_run()
    if run is not None:
        # signature based on DM-transformed features
        signature = infer_signature(X_train_np, cal.predict_proba(X_train_np))

        registered_model_name = (
            f"peds_methylation_{target_node_name.replace(' ', '_')}"
        )

        mlflow.sklearn.log_model(
            sk_model=cal,
            artifact_path="model",
            signature=signature,
            input_example=X_train_np[:3],
            registered_model_name=registered_model_name,
        )

        # also log summary JSON as an artifact for convenience
        mlflow.log_artifact(str(summary_path), artifact_path="eval")

        print(f"[MLflow] Model logged & registered as: {registered_model_name}")
    else:
        print("[MLflow] No active run; skipping MLflow model save.")

    return f"[SUCCESS] Completed node: {target_node_name}"


# COMMAND ----------

def train_node_from_tasks(
    node_name: str,
    TASK_PATH: Path = Path("/Workspace/9900-f16b-donut/sprint3/tasks/tasks.joblib"),
    EVAL_DIR: Path = Path("/Workspace/9900-f16b-donut/sprint3/evaluate"),
    MVALUE_CSV: Path = Path(MVALUE_CSV),
    RANDOM_STATE: int = RANDOM_STATE,
):
    """
    Train the model for a single node by name using tasks.joblib.
    """
    print(f"\n[INFO] Loading tasks from {TASK_PATH}")
    tasks = joblib.load(TASK_PATH)
    print(f"[INFO] Loaded {len(tasks)} nodes")

    # Ensure dirs exist
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Print memory for debugging
    mem = psutil.virtual_memory()
    print(f"[MEM] Total: {mem.total/1024**3:.2f} GB | "
          f"Available: {mem.available/1024**3:.2f} GB | "
          f"Used: {mem.used/1024**3:.2f} GB | "
          f"Percent used: {mem.percent}%")

    found = False
    node_metrics = None # added

    for node in tasks:
        if node_name == node.name:
            print("Loading DataFrame...")
            df = pl.read_csv(MVALUE_CSV)
            found = True
            print("\n" + "=" * 80)
            print(f"[NODE] Training node: {node_name}")
            print("=" * 80)

            try:
                res = train_single_node_pipeline(
                    node=node,
                    target_node_name=node_name,
                    mvalue_csv=df,
                    eval_dir=EVAL_DIR,
                    random_state=RANDOM_STATE,
                    dm_script_path="/Workspace/9900-f16b-donut/sprint3/differentialMethylation.R",
                    dm_top_n=50,
                )
                print(res)
            
            except Exception as e:
                print(f"[ERROR] Failed on node {node_name}: {e}")
                node_metrics = None # added
                # continue
            
            break # added

    if not found:
        print(f"[ERROR] Node '{node_name}' not found in tasks")
        return None
    
    return None
    

# COMMAND ----------

# MAGIC %md
# MAGIC