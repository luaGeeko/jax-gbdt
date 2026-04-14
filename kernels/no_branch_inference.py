import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional, Dict
from src.data.loader import CaliforniaHousingLoader
from src.compiler.parser import TreeParser
from scripts.evaluator import BaseLineEvaluator

def predict_single_tree(x, tree_feats, tree_thresh, tree_lefts, tree_rights, max_depth):
    """
    Inference for one sample on one tree using 2D array indexing.
    """
    def body_fun(i, current_node):
        # first checking if we are already at leaf node
        is_leaf = tree_lefts[current_node] == -1
        # now fetch node metadata
        feat_idx = tree_feats[current_node]
        threshold = tree_thresh[current_node]
        # we do arithmetic comparison and not use 'if' statements
        go_right = x[feat_idx] > threshold
        # select next node if not at leaf node
        next_node = lax.select(is_leaf, current_node, lax.select(go_right, tree_rights[current_node], tree_lefts[current_node]))
        return next_node
        # If we hit a leaf (-1), we stay at that leaf to avoid invalid memory access
        #return lax.select(current_node == -1, -1, next_node)

    # Initial state: root node is always 0
    final_node_idx = lax.fori_loop(0, max_depth, body_fun, 0)
    # Return the value stored in the threshold array at the leaf position
    return tree_thresh[final_node_idx] 

def jax_forest_predict(X_batch, features, thresholds, lefts, rights, max_depth):
    """
    Computes full forest inference in parallel.
    Shapes:
        X_batch: [batch, n_feats]
        features/thresholds/etc: [n_trees, max_nodes]
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
    if single_tree_inference:
        # vmap the function to handle batched data and get predictions from single tree
        print("single forest prediction will be done ........")
        vmapped_fn = jax.vmap(predict_single_tree, in_axes=(0, None, None, None, None, None))
        return vmapped_fn(X_batch, params['features'][0], params['thresholds'][0], params['left_children'][0], params['right_children'][0], params['max_depth'])
    else:
        # full forest wrapper
        print("full forest prediction will be done ........")
        return jax_forest_predict(X_batch, params['features'], params['thresholds'], params['left_children'], params['right_children'], params['max_depth'])


data_loader = CaliforniaHousingLoader()
get_single_sample = data_loader.get_test_samples()
sample = jnp.array(get_single_sample.values)
# sample_batch = jnp.array(data_loader.get_test_samples(n=10).values)#
# print(sample_batch.shape)

# get jax 2d arrays
tree_parser = TreeParser(trained_model_path="checkpoints/xgboost_model/xgboost_model.json")
jax_2d_arrays = tree_parser.parse()
jax_params = {'features': jax_2d_arrays['features'],
        'thresholds': jax_2d_arrays['thresholds'],
        'left_children': jax_2d_arrays['left_children'],
        'right_children': jax_2d_arrays['right_children'],
        'max_depth': jax_2d_arrays['max_nodes']
}
meta = tree_parser.get_metadata()

evaluator = BaseLineEvaluator(jax_predict_fn=jax_wrapper)

print("--- Running Single Tree Check ---")
single_results = evaluator.check(X_sample=sample, jax_params=jax_params)
print(f"Single Tree Consistency: {single_results['is_consistent']}")

print("\n--- Running Full Forest Check ---")
# Set jax_single_tree_prediction=False to trigger the forest logic in the wrapper
forest_results = evaluator.check(X_sample=sample, jax_params=jax_params, single_tree=False)
print(f"Full Forest Consistency: {forest_results['is_consistent']}")
print(f"Max Error: {forest_results['max_error']}")
#single_tree_results = evaluator.check(X_sample=sample, jax_params=jax_params)
#print(single_tree_results)

# ========== testing forest map =================
#results = evaluator.check(sample_batch, jax_params=jax_2d_arrays)
#jax_forest_predict(X_batch=sample, features=jax_params['features'], thresholds=jax_params['thresholds'], lefts=jax_params['left_children'], rights=jax_params['right_children'], max_depth=jax_params['max_depth'])
