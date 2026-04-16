import json
import numpy as np
import jax.numpy as jnp
from typing import Optional
from src.compiler.parser import TreeParser

class SoftParser:
    def __init__(self, trained_model_path: str):
        """ Initialize the SoftParser with the path to the trained XGBoost model JSON file."""
        self.trained_model_path = trained_model_path
        self.model_tree_data = None
        # check if the provided path is valid and contains a valid JSON
        self.valid_model_json = self._is_valid_json()
        self.trees_data = self.model_tree_data['learner']['gradient_booster']['model']['trees'] if self.valid_model_json else None
        self.num_features = int(self.model_tree_data['learner']['gradient_booster']['model']['trees'][0]['tree_param']['num_feature'])
        # create tree parser and format to get 2d jax arrays
        self.tree_parser = TreeParser(trained_model_path=trained_model_path)
        self.jax_2d_arrays = self.tree_parser.parse()
        self.features = self.jax_2d_arrays['features']          # Shape: [n_trees, max_nodes]
        self.thresholds = self.jax_2d_arrays['thresholds']      # Shape: [n_trees, max_nodes]
        self.left_children = self.jax_2d_arrays['left_children']
        self.right_children = self.jax_2d_arrays['right_children']
        self.n_trees, self.max_nodes = self.features.shape        

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
        
    def build_dense_path_matrices(self):
        """
        Builds the Ancestor Matrix (A) and Leaf Weights array for Option B.
        """
        # binary tree, max leaves is (max_nodes + 1) // 2
        max_leaves = (self.max_nodes + 1) // 2
        # A matrix: [trees, max_leaves, max_nodes]
        A = np.zeros((self.n_trees, max_leaves, self.max_nodes), dtype=np.int32)
        # Leaf weights: [trees, max_leaves]
        leaf_weights = np.zeros((self.n_trees, max_leaves), dtype=np.float32)
        
        for tree_idx in range(self.n_trees):
            leaf_counter = 0
            # helper function for Depth First Search
            def dfs(node_idx, current_path):
                nonlocal leaf_counter
                # cehck if it's a leaf node (left child is -1)
                if self.left_children[tree_idx, node_idx] == -1:
                    #  found a leaf , add it weight
                    leaf_weights[tree_idx, leaf_counter] = self.thresholds[tree_idx, node_idx]
                    # records its ancestor now
                    for ancestor_node, direction in current_path:
                        A[tree_idx, leaf_counter, ancestor_node] = direction
                    leaf_counter += 1
                    return
                # if internal node found, traverse left (direction=1) and right (direction=2)
                left_child = self.left_children[tree_idx, node_idx]
                right_child = self.right_children[tree_idx, node_idx]
                
                dfs(left_child, current_path + [(node_idx, 1)])
                dfs(right_child, current_path + [(node_idx, 2)])

            # Start DFS from the root (node 0) with an empty path
            dfs(0, [])
        return jnp.array(A), jnp.array(leaf_weights)

    def routing_matrices(self, method_type: Optional[str] = "iterative"):
        # create empty matrices for dot product later
        W = np.zeros((self.n_trees, self.max_nodes, self.num_features), dtype=np.float32)
        threshold_mat = np.zeros((self.n_trees, self.max_nodes), dtype=np.float32)
        for tree_idx in range(self.n_trees):
            for node_idx in range(self.max_nodes):
                feat = self.features[tree_idx, node_idx]
                # a valid internal node (not a padded leaf)
                if feat != -1 and self.left_children[tree_idx, node_idx] != -1:
                    # we will place a 1 at the exact feature index for this node
                    W[tree_idx, node_idx, feat] = 1.0
                    # lets copy the equivalent threshold
                    threshold_mat[tree_idx, node_idx] = self.thresholds[tree_idx, node_idx]
                else:
                    # for leaf nodes, we store the leaf weight in the threshold array
                    threshold_mat[tree_idx, node_idx] = self.thresholds[tree_idx, node_idx]

        # teranspose here itself do we can do X @ W directly in jnp.dot
        W_transposed = np.transpose(W, (0, 2, 1))
        routing_data = {
            'W': jnp.array(W_transposed),
            'tau': jnp.array(threshold_mat),
            'lefts': jnp.array(self.left_children),
            'rights': jnp.array(self.right_children),
            'thresholds': jnp.array(self.thresholds),
        }
        if method_type == "dense":
            A_matrix, leaf_weights = self.build_dense_path_matrices()
            routing_data.update({'A': A_matrix, 'leaf_wts': leaf_weights})
        return routing_data

        

# soft_parser = SoftParser(trained_model_path="checkpoints/xgboost_model/xgboost_model.json")
# routing_matrices = soft_parser.routing_matrices()