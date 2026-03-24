# Paediatric Cancer Methylation Classifier — New Version (project.zip)

This repository contains the **new and improved version** of the Paediatric Cancer Methylation Classifier project.  
It is a simplified, cleaner, and more scalable version of the original codebase.  
The system focuses on **task-based model training**, **parallel execution**, **reproducibility**, and **clear model outputs**.

---

## 🌳 Overview

This project builds a machine-learning classifier for paediatric cancer using DNA methylation (M-value) data.  
Instead of training one large model, the system trains **one model per node** in a disease hierarchy (e.g., “Leukaemia”, “Lymphoma”, etc.).

The new version provides:

- Clear **disease tree handling**
- Automatic creation of **training tasks**
- Per-node **train/validation splits**
- **Feature selection** using an R script (limma/DM)
- Random Forest models with **GridSearchCV**
- **Parallel training** using Databricks Jobs
- **Standardised outputs** for every model
- Reproducible workflows through fixed seeds and stored metadata

---

## Project Files and Folders

This project contains several Python files, R scripts, and notebooks.  
Each file has a clear purpose in the training pipeline.

---

## Main Files

### `disease_tree.py`
Defines the disease tree structure and creates the training tasks for each node.  
This file controls how samples are grouped and how tasks are made.

### `differentialMethylation.R`
An R script that performs feature selection using the limma package.  
It takes the training data and returns the best probes to use for each node model.

### `env_and_paths.ipynb`
A shared notebook that sets up folder paths, helper functions, and common utilities.  
All other notebooks import this notebook for consistency.

### `make_tasks_from_tree.ipynb`
Builds the training tasks from the disease tree.  
It loads the tree, spreads samples to each node, splits train/validation sets, and saves the tasks.

### `parallel_training.ipynb`
Contains the full training pipeline for one node.  
It loads the data, runs R feature selection, trains a Random Forest model with GridSearchCV, and saves metrics.

### `node_modelling.ipynb`
This notebook is designed to be run by a Databricks Job.  
It receives the node name as a job parameter and trains the corresponding node model.

### `job_manager.ipynb`
Starts a parent MLflow run and sends out one job for each node to Databricks.  
Used to train many node models at the same time.

---

## Folders Used or Created

### `tasks/` folder
Created after running `make_tasks_from_tree.ipynb`.  
Stores task files (`tasks.joblib`) containing all nodes and their sample lists.

### `models/` folder
Created after training.  
Stores the model bundles (`.joblib`) for each node.

### `evaluate/` folder (if present)
Created during training.  
Stores accuracy reports, confusion matrices, and performance summaries for each node.


## 🧩 Step 1 — Build Tasks From Disease Tree

Run: make_tasks_from_tree.ipynb

This notebook:

- Loads the disease tree  
- Fills sample lists for every node  
- Splits samples into training and validation using a fixed seed  
- Creates one task per node  
- Saves them into `tasks/tasks.joblib`  

Each task contains the correct samples for that node and is ready for training.

---

## 🧪 Step 2 — Train One Node Model

Run: node_modelling.ipynb

This notebook is designed for Databricks Jobs.  
It takes these inputs:

- `node_name`  
- `parent_run_id`  
- `experiment`  

Inside it:

- Loads the training pipeline  
- Runs feature selection using `differentialMethylation.R`  
- Trains a Random Forest with GridSearchCV  
- Saves the final model bundle into `models/`  
- Saves accuracy and reports into `evaluate/`  

You can also run this notebook manually for testing.

---

## ⚡ Step 3 — Train All Nodes in Parallel

Run: job_manager.ipynb

This notebook:

- Starts a parent MLflow run  
- Reads the list of node names  
- For each node, triggers a Databricks Job run  
- Saves all job run IDs into `submitted_jobs.json`  

With multiple workers, many node models train at the same time.

---

## 🔬 Feature Selection (R / Limma)

The file `differentialMethylation.R` performs probe selection using limma:

- Only training samples are passed to R (safe from leakage)  
- Top probes per class are selected  
- These selected probes are returned to Python and used to train the model  

This improves accuracy and reduces memory use.

---

## 📦 Model Saving

Each trained model is saved as a **ModelBundle** in `.joblib` format.  
A bundle includes:

- The trained classifier  
- Selected features  
- Node name  
- Child class names  
- Creation timestamp  

This gives a consistent model format across all nodes.

---

## 📄 Output Files

After running the project, you will see:

- `tasks/tasks.joblib` — list of all node tasks  
- `models/*.joblib` — one model per node  
- `evaluate/*.txt` — accuracy, confusion matrix, and reports  
- `submitted_jobs.json` — list of Databricks job runs  

---

## 🔧 Improvements Over Old Version

Compared to the old project, this version provides:

- Cleaner and safer data use for each node  
- No mixing between training and validation sets  
- Ability to train many node models at once  
- Standard joblib model output  
- Reproducible results with fixed seeds  
- Simpler and easier-to-read workflow  

---

## 📌 Requirements

To run this project on Databricks, the following are needed:

### **Software**
- Python 3.9+  
- Databricks Runtime with ML  
- R installed on the cluster  
- R package: `limma` (installed through the setup script)  
- rpy2 environment setup

### **Project Setup Files**
- `install_rpy2_init.sh`  
  Shell script that installs rpy2 and required R libraries on the Databricks cluster.

- `mc_requirements_no_rpy2.txt`  
  Python dependencies for the project (excluding rpy2).  
  These should be installed using `%pip install -r mc_requirements_no_rpy2.txt`.

### **Data Inputs**
- Methylation matrix in CSV or Feather format  
- Disease tree saved as a `.joblib` file  

---