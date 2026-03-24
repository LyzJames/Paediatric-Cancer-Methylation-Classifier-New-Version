"""
Core disease tree functionality used across the package.
"""
from __future__ import annotations

import random
import sys
import types
from sklearn.model_selection import StratifiedShuffleSplit
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from copy import deepcopy


@dataclass
class DiseaseTree:
    name: str
    children: List["DiseaseTree"]
    samples: List[str]
    training_samples: List[str] = field(default_factory=list)
    validation_samples: List[str] = field(default_factory=list)
    selected_features: List[Any] = field(default_factory=list)

    # -------- basic utilities --------
    def is_leaf(self) -> bool:
        return len(self.children) == 0 and len(self.samples) > 0

    def get_child_names(self) -> List[str]:
        return [child.name for child in self.children]

    # bugfix: recursive search worked incorrectly & used undefined function name
    def find_sample(self, sample_id: str, path: Optional[List[str]] = None) -> Optional[List[str]]:
        """Return the path (list of node names) from root to the node that contains sample_id."""
        if path is None:
            path = []
        path.append(self.name)

        if sample_id in self.samples:
            return path

        for child in self.children:
            result = child.find_sample(sample_id, path.copy())
            if result is not None:
                return result

        return None

    # -------- split train/val directly on self (no external 'tree' arg) --------
    def split_validation_training(self, validation_ratio: float = 0.2, random_seed: int = 42) -> "DiseaseTree":
        random.seed(random_seed)

        def process(node: "DiseaseTree") -> List[str]:
            """
            Post-order traversal.

            Returns:
                A list of all sample IDs under this node's subtree (from leaves).
            """
            # Leaf: no train/val; just return its own samples for parents to use.
            if node.is_leaf():
                node.training_samples = []
                node.validation_samples = []
                return list(node.samples or [])

            # Internal node
            node.training_samples = []
            node.validation_samples = []
            all_descendant_samples: List[str] = []

            for ch in node.children:
                # Recursively process child; this also sets child's own train/val
                child_samples = process(ch)   # all leaf samples under this child
                all_descendant_samples.extend(child_samples)

                # Now, for THIS node, split this child's samples into train/val
                if child_samples:
                    local = child_samples.copy()
                    random.shuffle(local)
                    n_val = max(1, int(len(local) * validation_ratio)) if len(local) > 0 else 0
                    child_val = local[:n_val]
                    child_train = local[n_val:]
                else:
                    child_val, child_train = [], []

                # Aggregate per-child splits into this node's train/val
                node.validation_samples.extend(child_val)
                node.training_samples.extend(child_train)

            return all_descendant_samples

        process(self)
        return self
    
    def propagate_samples_up(self) -> "DiseaseTree":
        """
        Ensure that for every internal node, `node.samples` contains
        all samples from its children (recursively from leaves).

        - Leaves: keep their `samples` as-is.
        - Internal nodes: set `samples` to the union of all descendant
          leaf samples. If some child samples are missing, they are added.
        """

        def dfs(node: "DiseaseTree") -> set:
            # Leaf: just return its own samples
            if node.is_leaf():
                return set(node.samples or [])

            # Internal: collect from children first (post-order)
            union_from_children: set = set()
            for ch in node.children:
                union_from_children |= dfs(ch)

            # Current node's existing samples (if any)
            current = set(node.samples or [])

            # Ensure node.samples includes all child samples
            new_samples = current | union_from_children
            node.samples = list(new_samples)

            return new_samples

        dfs(self)
        return self

    def get_nodes_at_level(self, level: int) -> List["DiseaseTree"]:
        """1-based level counting (root is level 1)."""
        nodes: List["DiseaseTree"] = []

        def collect(node: "DiseaseTree", cur: int):
            if cur == level:
                nodes.append(node)
            elif cur < level:
                for ch in node.children:
                    collect(ch, cur + 1)

        collect(self, 1)
        return nodes
    
    def delete_node(self, target_name):
        if self.name == target_name:
            # If the current node matches the target, return None to delete it
            return None
        
        # Recursively search for the target node in children
        new_children = [child.delete_node(target_name) for child in self.children]
        # Remove None (deleted nodes) from children list
        new_children = [child for child in new_children if child is not None]

        # Update children list with modified list
        self.children = new_children
        
        return self  # Return modified node

    # bugfix: make this an instance method with proper recursion
    def filter_tree_by_depth(self, target_depth: int) -> List[str]:
        """
        Return the list of node names at exactly 'target_depth' edges below self.
        (depth 0 is self)
        """
        if target_depth == 0:
            return [self.name]
        if target_depth < 0:
            return []
        names: List[str] = []
        for ch in self.children:
            names.extend(ch.filter_tree_by_depth(target_depth - 1))
        return names

    def _node_train_val_ids(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Return (train_ids, val_ids, all_ids) for this node.
        If explicit train/val are empty, fall back to union including .samples.
        """
        tr = list(self.training_samples or [])
        va = list(self.validation_samples or [])
        if not tr and not va:
            # no explicit split yet — derive from samples (may be empty for internal nodes)
            all_ids = sorted(set(self.samples or []))
        else:
            all_ids = sorted(set(tr) | set(va) | set(self.samples or []))
        return tr, va, all_ids
    
    def _child_samples_union(self) -> set:
        """Set of all sample IDs present in direct children (any of their sample lists)."""
        u = set()
        for ch in self.children:
            # prefer explicit per-type lists, but include raw samples if present
            child_ids = set(ch.samples or [])
            child_ids.update(ch.training_samples or [])
            child_ids.update(ch.validation_samples or [])
            u.update(child_ids)
        return u

    def build_classification_tasks(self, verbose: bool = True) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []

        def dfs(node: "DiseaseTree"):
            if not node.children:
                return

            # collect children that have >= 10 samples
            eligible_children = []
            ids_n = 0
            for ch in node.children:
                child_ids = set(getattr(ch, "samples", []))

                if len(child_ids) >= 4:
                    ids_n += len(child_ids)
                    eligible_children.append(ch)

            # Node is a task only if at least 2 children have >= 10 samples
            if len(eligible_children) >= 2 and ids_n >= 10:
                # Shallow clone node and override children
                node_copy = deepcopy(node)
                node_copy.children = eligible_children
                # Collect IDs from eligible children
                keep_ids = set()
                for ch in eligible_children:
                    keep_ids.update(ch.samples)

                # Filter node-level samples to only those coming from eligible children
                node_copy.samples = [s for s in node_copy.samples if s in keep_ids]
                node_copy.training_samples = [s for s in node_copy.training_samples if s in keep_ids]
                node_copy.validation_samples = [s for s in node_copy.validation_samples if s in keep_ids]
                classes = [ch.name for ch in eligible_children]
                tasks.append({
                    "node_name": node.name,
                    "classes": classes,
                    "node": node_copy
                })

            # Recurse down the tree
            for ch in node.children:
                dfs(ch)

        dfs(self)
        return tasks

    # -------- convenience: tabular view of samples at a given level --------
    def get_samples_at_level(self, level: int) -> pl.DataFrame:
        dfs = []
        for node in self.get_nodes_at_level(level):
            samples = node.get_samples_recursive()
            df = pl.DataFrame({
                "sample_id": samples,
                "cancerType": [node.name] * len(samples)
            })
            dfs.append(df)
        return pl.concat(dfs) if dfs else pl.DataFrame({"sample_id": [], "cancerType": []})