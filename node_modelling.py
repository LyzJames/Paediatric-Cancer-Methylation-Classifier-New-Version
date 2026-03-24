# Databricks notebook source
# MAGIC %md
# MAGIC ### Notes
# MAGIC This notebook is responsible for:
# MAGIC
# MAGIC - Loading environment settings and training utilities for node-level modelling.
# MAGIC - Training the classifier for a single disease-tree node.
# MAGIC - Logging node-specific parameters (node name, parent run ID) to MLflow.
# MAGIC - Recording evaluation metrics (accuracy, train/test accuracy) for downstream aggregation.
# MAGIC - Marking skipped or invalid nodes so the Job Manager can summarise all training outcomes.

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

# === parameters ===
# get node name from job task parameters setting
dbutils.widgets.text("node_name", "") 
dbutils.widgets.text("parent_run_id", "")
dbutils.widgets.text("experiment", "")
node_name = dbutils.widgets.get("node_name")
parent_run_id = dbutils.widgets.get("parent_run_id")
experiment = dbutils.widgets.get("experiment")

print(f"Training for node: {node_name}")
print(f"parent_run_id: {parent_run_id}")
print(f"experiment: {experiment}")

# COMMAND ----------

# Set the MLflow experiment if a path is provided.
# Databricks uses workspace paths (e.g. /Users/.../experiment).
if experiment:
    mlflow.set_experiment(experiment)

# Start an MLflow run for a node.
with mlflow.start_run(run_name=f"node = {node_name}"):

    # Load shared environment variables, paths, and utilities.
    %run "./env_and_paths.ipynb"

    # Load training logic that defines
    %run "./parallel_training.ipynb"

    # Log identifying information so that the Job Manager notebook
    mlflow.log_param("node_name", node_name)
    mlflow.log_param("parent_run_id", parent_run_id)
    mlflow.set_tag("node_name", node_name)
    mlflow.set_tag("parent_run_id", parent_run_id)

    print(f"before running node: {node_name}, parent_run_id: {parent_run_id}")

    metrics = train_node_from_tasks(node_name)

    # If the training function returns metrics, log them to MLflow
    if isinstance(metrics, dict):
        acc = metrics.get("accuracy")
        train_acc = metrics.get("train_accuracy")
        test_acc = metrics.get("test_accuracy")

        if acc is not None:
            mlflow.log_metric("accuracy", acc)
        if train_acc is not None:
            mlflow.log_metric("train_accuracy", train_acc)
        if test_acc is not None:
            mlflow.log_metric("test_accuracy", test_acc)

        # record a tag indicating that training completed successfully
        mlflow.set_tag("status", "TRAINED")
    else:
        # no metrics are returned
        mlflow.set_tag("status", "NO_METRICS")

# COMMAND ----------

run = mlflow.last_active_run()
print("current run_id:", run.info.run_id)

client = MlflowClient()
print("metrics:", client.get_run(run.info.run_id).data.metrics)