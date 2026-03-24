# Data

We periodically take a snapshot of our data to use for training updated models using a new sample set. Wach of these snapshopts is referred to as a freeze. 

A freeze directory will hold: 
- base training data in a parquet file, organized as rows/samples, columns/variables (methylation probes) as required for sklearn libraries.
- An oncotree in a joblib file. This will be provided

An Oncotree held in a python data class called DIseaseTree (in mch.core.disease_tree) that recursively defines a tree structure, with each node of the tree being an instance of the class. Each node has:
- a name, 
- a list of training samples associated with it
- a list of calibration samples
- a list of validation samples
- a list of child nodes, all of type DiseaseTree.

Directories will have to me made to hold models and embeddings. 