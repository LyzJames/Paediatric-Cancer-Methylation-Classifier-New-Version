# Databricks notebook source
# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC - This notebook is responsible for:
# MAGIC   - Submitting one training job per disease-tree node
# MAGIC   - Limiting the number of child jobs running in parallel
# MAGIC   - Logging all submitted runs to MLflow
# MAGIC   - Providing a small dashboard to monitor job status and summarize metrics

# COMMAND ----------

import mlflow
import time
import json
import pandas as pd
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from datetime import datetime

# COMMAND ----------

# ========= Configuration ===========

# 1. Set node names manually
NODE_LIST = [
    "Acute myeloid leukaemia",
    "Leukaemia"
]

# 2. Set Databricks job parameters and SDK client
EXPERIMENT_PATH = "/9900-f16b-donut/sprint3/experiment"
mlflow.set_experiment(EXPERIMENT_PATH)
RUN_JOB_ID = 133383140616614  # Job ID for the node training
MAX_PARALLEL = 2            # maximum number of parallel node runs
POLL_INTERVAL = 30            # Seconds to wait between polling job run statuses
TERMINAL_STATES = {"TERMINATED", "INTERNAL_ERROR", "SKIPPED"}



# COMMAND ----------

# Helper function to extract life_cycle_state from w.jobs.get_run(...) result
def get_lifecycle_state(run):
    """
    Extract life_cycle_state from w.jobs.get_run(...) result and
    normalize it to a simple upper-case string, e.g.
    RunLifeCycleState.TERMINATED -> "TERMINATED".
    """
    state = getattr(run, "state", None)
    if state is None:
        return None

    raw = getattr(state, "life_cycle_state", None)
    if raw is None:
        return None

    s = str(raw)  # e.g. "RunLifeCycleState.TERMINATED" or "TERMINATED"
    if "." in s:
        s = s.split(".")[-1]
    return s.upper()

def refresh_active_runs(active):
    """
    Check all active runs, keep only those that are not in a terminal lifecycle state.
    """
    still_active = []
    for r in active:
        run_id = r["job_run_id"]
        if not run_id:
            # No valid run_id, skip but do not keep it as active
            continue

        try:
            run = w.jobs.get_run(run_id=run_id)
            life_cycle = get_lifecycle_state(run)
            print(f"[refresh] run_id={run_id}, node={r['node']}, life_cycle_state={life_cycle}")
        except Exception as e:
            print(f"[refresh] get_run({run_id}) failed: {e}")
            # Be conservative: keep it as active to avoid over-submission
            still_active.append(r)
            continue

        if life_cycle in TERMINAL_STATES:
            print(f"[refresh] run {run_id} for node {r['node']} finished. Removing from active list.")
        else:
            still_active.append(r)

    print(f"[refresh] active_runs size after refresh: {len(still_active)}")
    return still_active

def get_child_run_status(df):
    """
    Given the df_submitted DataFrame, fetch the latest status
    for each job_run_id from Databricks Jobs API.
    Returns a DataFrame containing ONLY simple Python types.
    """
    rows = []
    for _, row in df.iterrows():
        node = row["node"]
        run_id = row["job_run_id"]

        # Handle missing run_id
        if not run_id:
            rows.append({
                "node": node,
                "job_run_id": None,
                "life_cycle_state": "UNKNOWN",
                "result_state": "UNKNOWN",
            })
            continue

        try:
            run = w.jobs.get_run(run_id=run_id)

            # ---- normalize lifecycle state ----
            raw_life = getattr(run.state, "life_cycle_state", None)
            life_str = str(raw_life)
            # Example: "RunLifeCycleState.TERMINATED" → "TERMINATED"
            if "." in life_str:
                life_str = life_str.split(".")[-1]
            life_str = life_str.upper()

            # ---- normalize result state ----
            raw_result = getattr(run.state, "result_state", None)
            result_str = str(raw_result)
            if "." in result_str:
                result_str = result_str.split(".")[-1]
            result_str = result_str.upper()

        except Exception as e:
            print(f"get_run({run_id}) failed: {e}")
            life_str = "ERROR"
            result_str = "ERROR"

        # Append ONLY simple strings & ints
        rows.append({
            "node": node,
            "job_run_id": int(run_id),
            "life_cycle_state": life_str,
            "result_state": result_str,
        })

    return pd.DataFrame(rows)

# COMMAND ----------

# ========= Main =========
w = WorkspaceClient()
session_name = f"ensemble_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
submitted = []      # All submitted runs (for logging to MLflow)
active_runs = []    # Runs that are still active (for concurrency control)

with mlflow.start_run(run_name=session_name) as parent_run:
    parent_id = parent_run.info.run_id
    print("Parent MLflow run_id:", parent_id)

    for node in NODE_LIST:
        # Wait until we have a free slot under MAX_PARALLEL
        while len(active_runs) >= MAX_PARALLEL:
            print(
                f"[orchestrator] Active run count {len(active_runs)} "
                f"reached limit {MAX_PARALLEL}. Waiting for some runs to finish..."
            )
            time.sleep(POLL_INTERVAL)
            active_runs = refresh_active_runs(active_runs)

        # Now we have capacity to submit a new run
        params = {
            "node_name": node,
            "parent_run_id": parent_id,
            "experiment": EXPERIMENT_PATH,
        }

        print(f"[orchestrator] Submitting Job run for node: {node}")
        run_now_response = w.jobs.run_now(
            job_id=RUN_JOB_ID,
            job_parameters=params,
        )

        # New SDK: Wait object exposes run_id directly
        run_id = None
        if hasattr(run_now_response, "run_id"):
            run_id = run_now_response.run_id
        else:
            try:
                result_obj = run_now_response.result()  # may block until done
                run_id = getattr(result_obj, "run_id", None)
            except Exception as e:
                print(f"[orchestrator] Warning: could not extract run_id: {e}")
                run_id = None

        record = {
            "node": node,
            "job_run_id": run_id,
        }
        submitted.append(record)

        if run_id is not None:
            active_runs.append(record)
            print(
                f"[orchestrator] Added node {node} to active_runs. "
                f"Current active size: {len(active_runs)}"
            )
        else:
            print(
                f"[orchestrator] No valid run_id for node {node}, "
                "not adding to active_runs."
            )

    # After all nodes have been submitted, wait for remaining runs to finish
    print("[orchestrator] All nodes have been submitted. Waiting for remaining runs to finish...")
    while len(active_runs) > 0:
        time.sleep(POLL_INTERVAL)
        active_runs = refresh_active_runs(active_runs)

    # Show summary and log
    df_submitted = pd.DataFrame(submitted)
    display(df_submitted)
    mlflow.log_dict(submitted, "submitted_jobs.json")

print("Orchestrator finished submitting all node training jobs.")

# COMMAND ----------

status_df = get_child_run_status(df_submitted)
display(status_df.sort_values("node").reset_index(drop=True))


# COMMAND ----------

client = MlflowClient()

# 1. Get experiment id from path
experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
experiment_id = experiment.experiment_id
print("Using experiment_id:", experiment_id)

# 2. Search all child runs with this parent_run_id
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.parent_run_id = '{parent_id}'",
)

print(f"Found {len(runs)} child runs for parent_run_id={parent_id}")


# COMMAND ----------

rows = []

for r in runs:
    tags = r.data.tags or {}
    metrics = r.data.metrics or {}

    node_name = tags.get("node_name", "UNKNOWN")
    status_tag = tags.get("status", "UNKNOWN")

    start = r.info.start_time
    end = r.info.end_time
    duration = (end - start) / 1000.0 if start and end else None

    rows.append({
        "run_id": r.info.run_id,
        "mlflow_status": r.info.status,
        "tag_status": status_tag,
        "node": node_name,
        "train_accuracy": metrics.get("train_accuracy", None),
        "val_accuracy": metrics.get("val_accuracy", None),
        "duration_sec": duration,
    })

if not rows:
    print(f"No child runs found for parent_run_id={parent_id}")
    df_runs = pd.DataFrame(columns=[
        "run_id", "mlflow_status", "tag_status", "node",
        "train_accuracy", "test_accuracy", "duration_sec",
    ])
else:
    df_runs = pd.DataFrame(rows)
    display(df_runs)
