"""
parser.py

This module parses a trained XGBoost model (JSON format) and converts it into
a padded, level-wise 2D tensor representation suitable for JAX execution.

Design Rationale
----------------

The tree is transformed into fixed-shape arrays to make it compatible with
JAX's JIT compilation and accelerator (GPU/TPU) execution.

Key benefits of this representation:

1. Coalesced Memory Access
   When executing on a GPU, all features corresponding to the same node index
   across multiple trees are stored contiguously in memory. This allows the GPU
   to fetch them in a single memory transaction, significantly improving memory
   bandwidth utilization.

2. Predictable Indexing
   Tree traversal is replaced by direct indexing into arrays. For example, to
   access the left child of node `j` in tree `i`, the kernel simply reads:

       lefts[i, j]

   This avoids pointer chasing and eliminates the need for complex offset
   computations, making execution faster and more hardware-friendly.

3. Neutral Padding
   Trees are padded to a uniform shape using sentinel values (e.g., -1).
   These act as stopping conditions during computation and ensure that all
   operations can be performed using fixed-size tensors without dynamic control
   flow.

Overall, this representation converts tree traversal into a static tensor
computation graph, enabling efficient JAX JIT compilation and parallel execution.
"""

import os
import json
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict

class TreeParser:
    """A class to parse the trained XGBoost model JSON file and extract the necessary data for JAX processing."""
    def __init__(self, trained_model_path: str):
        """Initialize the TreeParser with the path to the trained XGBoost model JSON file."""
        self.trained_model_path = trained_model_path
        self.model_tree_data = None
        # check if the provided path is valid and contains a valid JSON
        self.valid_model_json = self._is_valid_json()
        self.trees_data = self.model_tree_data['learner']['gradient_booster']['model']['trees'] if self.valid_model_json else None
        self.num_trees = len(self.trees_data) if self.trees_data is not None else 0
        self.max_node_num = max(len(idx['left_children']) for idx in self.trees_data) if self.trees_data is not None else 0

    def _is_valid_json(self) -> bool:
        """Check if the provided path is valid and contains a valid JSON file."""
        try:
            with open(self.trained_model_path, 'r') as f:
                self.model_tree_data = json.load(f)
            return True
        except ValueError as e:
            print(f"Invalid JSON error: {e}")
            return False
        except FileNotFoundError as e:
            print(f"File not found error: {e}")
            return False
        
    def preallocate_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ preallocate array size that will hold the data for each tree

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: arrays with tree data. padding for default values has been done with -1 and for threshold as 0.0
        """
        # features
        features = np.full((self.num_trees, self.max_node_num), -1, dtype=np.int32)
        # left children
        left_children = np.full((self.num_trees, self.max_node_num), -1, dtype=np.int32)
        # right children
        right_children = np.full((self.num_trees, self.max_node_num), -1, dtype=np.int32)
        # thresholds
        thresholds = np.full((self.num_trees, self.max_node_num), 0.0, dtype=np.float32)
        return features, left_children, right_children, thresholds

    def parse_2d_arrays(self):
        """Parse the tree data and store it in 2D arrays for JAX processing."""
        # arrays to hold the data we need for JAX processing
        # we need to loop over each tree, compute the nodes and edges, and store them in a format suitable for JAX processing
        features, left_children, right_children, thresholds = self.preallocate_arrays()
        for tree_id, tree in enumerate(self.trees_data):
            tree_num_nodes = len(tree['left_children'])
            features[tree_id, :tree_num_nodes] = np.array(tree['split_conditions'], dtype=np.int32)
            left_children[tree_id, :tree_num_nodes] = np.array(tree['left_children'], dtype=np.int32)
            right_children[tree_id, :tree_num_nodes] = np.array(tree['right_children'], dtype=np.int32)
            thresholds[tree_id, :tree_num_nodes] = np.array(tree['thresholds'], dtype=np.float32)
        
        # conver to jax arrays
        self.features = jnp.array(features)
        self.left_children = jnp.array(left_children)
        self.right_children = jnp.array(right_children)
        self.thresholds = jnp.array(thresholds)

    def parse(self) -> Dict[str, jnp.ndarray]:
        """ parse function will provide the features, left and right children, threshold for all the trees in 2d format

        Returns:
            Dict[str, jnp.ndarray]: dict holding key and the corresponding jax array holding data
        """
        return {
            "features": self.features,
            "left_children": self.left_children,
            "right_children": self.right_children,
            "thresholds": self.thresholds,
            "max_nodes": self.max_node_num,
            "num_trees": self.num_trees,
        }

    def get_metadata(self) -> Dict[str, float]:
        """ metadata base score 

        Returns:
            Dict[str, float]: base score of the model
        """
        base_score = float(self.model_tree_data['learner']['learner_model_param']['base_score'])
        return {"base_score": base_score}


# tree_parser = TreeParser(trained_model_path="checkpoints/xgboost_model/xgboost_model.json")
# tree_parser.get_metadata()