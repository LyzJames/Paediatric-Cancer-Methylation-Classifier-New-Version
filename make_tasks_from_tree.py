# Databricks notebook source
# Databricks uses absolute or relative paths without file extension.
# If notebook name has spaces, wrap it in quotes.
%run "./env_and_paths.ipynb"

# COMMAND ----------

# ---------------------------
# Joblib loader shim
# ---------------------------
def _setup_mock_modules():
    """Map mch.core.disease_tree.DiseaseTree → our local DiseaseTree for unpickling."""
    mock_modules = {
        'mch': types.ModuleType('mch'),
        'mch.core': types.ModuleType('mch.core'),
        'mch.core.disease_tree': types.ModuleType('mch.core.disease_tree'),
    }
    for name, mod in mock_modules.items():
        sys.modules[name] = mod
    sys.modules['mch.core.disease_tree'].DiseaseTree = DiseaseTree

def load_tree_joblib(path: str) -> Optional[DiseaseTree]:
    _setup_mock_modules()
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Tree not found: {p.resolve()}")
        return None
    with open(p, "rb") as f:
        tree = joblib.load(f)
    if not isinstance(tree, DiseaseTree):
        print("[WARN] Loaded object is not a DiseaseTree; attempting to proceed.")
    return tree

# COMMAND ----------

tree = load_tree_joblib(TREE_JOBLIB)
tree = tree.propagate_samples_up()
# (optional) if your tree doesn't already have training/validation populated:
tree.split_validation_training(validation_ratio=0.2, random_seed=42)

tasks = tree.build_classification_tasks(verbose=True)

# COMMAND ----------

print(f"Found {len(tasks)} Node (classification tasks) after filtering:")
print("\nNode samples counts (excluding node's own samples):")

nodes = []
node_names = []

for t in tasks:
    node = t["node"]
    nodes.append(node)
    node_names.append(t["node_name"])

    # counts
    train_cnt = len(node.training_samples)
    val_cnt   = len(node.validation_samples)
    num_classes = len(t["classes"])

    print(
        f" - {t['node_name']}: "
        f"train={train_cnt}, val={val_cnt}, "
        f"Number of Classes={num_classes}"
    )

# COMMAND ----------

# make sure output folder exists
os.makedirs("tasks", exist_ok=True)

# save into a single joblib
output_path = "tasks/tasks.joblib"
joblib.dump(nodes, output_path)

print(f"✅ Saved {len(nodes)} nodes to {output_path}")

# COMMAND ----------

# Choose output file
out_path = Path("tasks/node_names.json")

# Save
with out_path.open("w", encoding="utf-8") as f:
    json.dump(node_names, f, indent=2)

print(f"[INFO] Saved {len(node_names)} node names to {out_path}")