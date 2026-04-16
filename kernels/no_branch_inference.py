"""
JAX-based No-Branch Tree Inference and Evaluation Module.

This module implements a fully vectorized and branch-free inference pipeline
for tree-based ensemble models (e.g., XGBoost) using JAX. It replaces traditional
control-flow-heavy tree traversal with JAX primitives (`lax`, `vmap`, `jit`)
to enable efficient execution via XLA compilation.

Core capabilities include:
    - Single-tree inference using array-based traversal
    - Parallel forest inference via vectorization
    - Wrapper abstraction for switching between single-tree and full-forest modes
    - Verification pipeline comparing JAX predictions with baseline XGBoost outputs
    - Optional reproducibility via controlled random seeding

The design emphasizes:
    - Eliminating Python-level branching
    - Maximizing hardware utilization (CPU/GPU/TPU)
    - Ensuring numerical consistency with baseline models

This module is primarily intended for:
    - Research on compiler-optimized inference (XLA)
    - Performance benchmarking of tree-based models
    - Debugging and validating custom inference kernels
"""

import jax
import numpy as np
import jax.numpy as jnp
from jax import lax
from typing import Optional
from src.data.loader import CaliforniaHousingLoader
from src.compiler.parser import TreeParser
from scripts.evaluator import BaseLineEvaluator
import argparse

def predict_single_tree(x, tree_feats, tree_thresh, tree_lefts, tree_rights, max_depth):
    """
    Perform inference for a single decision tree on a single input sample.

    This function traverses a decision tree represented in array form without
    using explicit control flow (e.g., `if` statements). Instead, it relies on
    JAX primitives (`lax.select`, `lax.fori_loop`) to enable compilation into
    efficient, branch-free computation graphs.

    The traversal starts at the root node and iteratively selects the next node
    based on feature-threshold comparisons until reaching a leaf node.

    Args:
        x (jnp.ndarray):
            Input feature vector of shape [n_features].

        tree_feats (jnp.ndarray):
            Feature indices for each node in the tree.

        tree_thresh (jnp.ndarray):
            Threshold values for splits or leaf values.

        tree_lefts (jnp.ndarray):
            Indices of left child nodes.

        tree_rights (jnp.ndarray):
            Indices of right child nodes.

        max_depth (int):
            Maximum traversal depth (used as loop bound).

    Returns:
        jnp.ndarray:
            Prediction value from the leaf node.
    """
    def body_fun(i, current_node):
        # first checking if we are already at leaf node
        is_leaf = tree_lefts[current_node] == -1
        # now fetch node metadata
        feat_idx = tree_feats[current_node]
        threshold = tree_thresh[current_node]
        # we do arithmetic comparison and not use 'if' statements
        go_left = x[feat_idx] < threshold
        #go_right = x[feat_idx] > threshold
        # select next node if not at leaf node
        next_node = lax.select(is_leaf, current_node, lax.select(go_left, tree_lefts[current_node], tree_rights[current_node]))
        return next_node
        # If we hit a leaf (-1), we stay at that leaf to avoid invalid memory access
        #return lax.select(current_node == -1, -1, next_node)

    # Initial state: root node is always 0
    final_node_idx = lax.fori_loop(0, max_depth, body_fun, 0)
    # Return the value stored in the threshold array at the leaf position
    return tree_thresh[final_node_idx] 

def jax_forest_predict(X_batch, features, thresholds, lefts, rights, max_depth):
    """
    Compute predictions for a batch of samples across an entire tree ensemble.

    This function performs fully vectorized inference over both:
        - Multiple trees (ensemble dimension)
        - Multiple input samples (batch dimension)

    It uses nested `jax.vmap` transformations to parallelize computation and
    leverages XLA for kernel fusion and optimized execution.

    Args:
        X_batch (jnp.ndarray):
            Batch of input samples with shape [batch_size, n_features].

        features (jnp.ndarray):
            Feature indices for all trees, shape [n_trees, max_nodes].

        thresholds (jnp.ndarray):
            Threshold or leaf values for all trees.

        lefts (jnp.ndarray):
            Left child indices for all trees.

        rights (jnp.ndarray):
            Right child indices for all trees.

        max_depth (int):
            Maximum traversal depth for all trees.

    Returns:
        jnp.ndarray:
            Final predictions for each sample, shape [batch_size],
            computed as the sum of outputs from all trees.
    """
    # parallelize 'predict_single_tree' across the trees for samples is constant for all trees in this map level
    vmap_trees = jax.vmap(predict_single_tree, in_axes=(None, 0, 0, 0, 0, None))
    # 2. Parallelize across the Batch Dimension (Data Samples)
    vmap_samples = jax.vmap(vmap_trees, in_axes=(0, None, None, None, None, None))
    # 3. Execute: Get raw leaf values [batch_size, n_trees]
    # This is where XLA performs 'Kernel Fusion'
    raw_tree_outputs = vmap_samples(X_batch, features, thresholds, lefts, rights, max_depth)
    # 4. Final Aggregation (Additive Model)
    # Sum across the tree axis (axis 1)
    forest_sum = jnp.sum(raw_tree_outputs, axis=1)
    return forest_sum
    #return base_score + (learning_rate * forest_sum)

def jax_wrapper(X_batch, single_tree_inference: Optional[bool] = True, **params):
    """
    Wrapper function to switch between single-tree and full-forest inference modes.

    This function provides a unified interface for invoking either:
        - Single-tree inference (for debugging and validation)
        - Full-forest inference (for actual predictions)

    It abstracts the underlying vectorization logic and parameter handling.

    Args:
        X_batch (jnp.ndarray):
            Batch of input samples.

        single_tree_inference (bool, optional):
            If True, performs inference using only the first tree.
            If False, performs full ensemble inference.
            Defaults to True.

        **params:
            Dictionary containing model parameters:
                - features
                - thresholds
                - left_children
                - right_children
                - max_depth

    Returns:
        jnp.ndarray:
            Predictions corresponding to the selected inference mode.
    """
    if single_tree_inference:
        # vmap the function to handle batched data and get predictions from single tree
        vmapped_fn = jax.vmap(predict_single_tree, in_axes=(0, None, None, None, None, None))
        return vmapped_fn(X_batch, params['features'][0], params['thresholds'][0], params['left_children'][0], params['right_children'][0], params['max_depth'])
    else:
        # full forest wrapper
        return jax_forest_predict(X_batch, params['features'], params['thresholds'], params['left_children'], params['right_children'], params['max_depth'])


def verification_and_evaluation(batch_size: int, seed: Optional[int] = None, xla_fusion_analysis: Optional[bool] = False):
    """
    Validate JAX-based inference against a baseline XGBoost model.

    This function performs consistency checks between predictions generated by
    the custom JAX inference pipeline and a reference XGBoost implementation.
    It supports both single-tree and full-forest validation.

    The evaluation pipeline includes:
        - Optional reproducibility via random seed control
        - Data loading and batch preparation
        - Model parsing into JAX-compatible arrays
        - Execution of baseline and JAX inference
        - Consistency verification and reporting

    If discrepancies are detected in full forest predictions, a detailed
    per-tree debugging routine is triggered.

    Args:
        batch_size (int):
            Number of samples to use for evaluation.

        seed (Optional[int], optional):
            Random seed for reproducible sampling.
            If provided, ensures deterministic test data selection.

    Outputs:
        Prints:
            - Status of single-tree consistency check
            - Status of full-forest consistency check
            - Debug information if inconsistencies are detected

    Notes:
        - This function is primarily intended for debugging and validation.
        - Reproducibility depends on the underlying data loader implementation.
    """
    if seed is not None:
        np.random.seed(seed)
        print('[NOBRANCH] evaluation will be done with reproducibility....')

    data_loader = CaliforniaHousingLoader()
    test_data = data_loader.get_test_samples(n=batch_size)
    print(f"[DEBUG] total entries in x test in data {len(data_loader.X_test)}")
    jax_sample_batch = jnp.array(test_data.values)
    # sample_batch = jnp.array(data_loader.get_test_samples(n=10).values)#
    # print(sample_batch.shape)
    # --- parse the tree to get 2d arrays --------
    # get jax 2d arrays
    tree_parser = TreeParser(trained_model_path="checkpoints/xgboost_model/xgboost_model.json")
    jax_2d_arrays = tree_parser.parse()
    jax_params = {'features': jax_2d_arrays['features'],
            'thresholds': jax_2d_arrays['thresholds'],
            'left_children': jax_2d_arrays['left_children'],
            'right_children': jax_2d_arrays['right_children'],
            'max_depth': jax_2d_arrays['max_nodes']
    }
    
    # evaluate now
    evaluator = BaseLineEvaluator(jax_predict_fn=jax_wrapper)
    real_base_score = evaluator.model_base_score
    print(f"Verifying Forest Logic for Batch Size: {test_data.shape[0]}!")
    print("--- Running Single Tree Check ---")
    single_results = evaluator.check(X_sample=jax_sample_batch, jax_params=jax_params)
    print(f"Single Tree Consistency: {single_results['is_consistent']}")

    print("\n--- Running Full Forest Check ---")
    # Set jax_single_tree_prediction=False to trigger the forest logic in the wrapper
    forest_results = evaluator.check(X_sample=jax_sample_batch, jax_params=jax_params, single_tree=False)
    print(f"Full Forest Consistency: {forest_results['is_consistent']}")
    if not forest_results['is_consistent']:
        BaseLineEvaluator.debug_trees(sample_batch=jax_sample_batch, jax_params=jax_params, jax_wrapper=jax_wrapper, model=evaluator.model, real_base_score=real_base_score)
    #single_tree_results = evaluator.check(X_sample=sample, jax_params=jax_params)
    #print(single_tree_results)
    if xla_fusion_analysis:
        lowered = jax.jit(jax_forest_predict).lower(jax_sample_batch, jax_params['features'], jax_params['thresholds'],jax_params['left_children'], jax_params['right_children'], jax_params['max_depth'])
        hlo_ir = lowered.compiler_ir()
        with open("hlo_fusion_analysis.txt", "w") as f:
            f.write(str(hlo_ir))
        print("HLO IR saved to hlo_fusion_analysis.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No branch infernce")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size to test inference")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()

    verification_and_evaluation(batch_size=args.batch_size, seed=args.seed, xla_fusion_analysis=True)