import jax
import numpy as np
import jax.numpy as jnp
from jax import lax
from typing import Optional
from src.data.loader import CaliforniaHousingLoader
from src.compiler.soft_sigmoid_parser import SoftParser
from scripts.evaluator import BaseLineEvaluator
import argparse

def soft_node_activations(X_batch, W, tau, temperature=10.0):
    """
    Compute the split probability for every single node in the 
    entire forest simultaneously using pure matrix multiplication.
    Dense Projection (Replaces the 'gather') and we do a batched dot product. 
    jnp.einsum is the cleanest way to multiply a batch by a 3D tensor in JAX
    'bf,tfn->btn' means: (batch, features) * (trees, features, nodes) -> (batch, trees, nodes)

    Then we go for soft split by subtracting thresholds and apply Sigmoid.
    If H < tau, prob is close to 0. If H > tau, prob is close to 1.
    Temperature controls how "hard" the split acts.
    
    X_batch: [batch_size, n_features]
    W:       [n_trees, n_features, max_nodes] (The Routing Matrix)
    tau:     [n_trees, max_nodes]             (The Thresholds)
    """
    H = jnp.einsum('bf,tfn->btn', X_batch, W)
    left_probabilities = jax.nn.sigmoid(temperature * (tau - H))
    # right split probability is simply 1 - left
    right_probabilities = 1.0 - left_probabilities
    return left_probabilities, right_probabilities

def calculate_paths_iterative(left_probs, right_probs, left_children, right_children, thresholds):
    """
    Option A: Iteratively push probabilities down the tree.
    left_probs:  [batch, trees, max_nodes]
    left_children: [trees, max_nodes]
    """
    batch_size, n_trees, max_nodes = left_probs.shape
    
    # Initialize all reach probabilities to 0.0
    reach_probs = jnp.zeros((batch_size, n_trees, max_nodes))
    
    # The root node (index 0) always has a reach probability of 1.0
    reach_probs = reach_probs.at[:, :, 0].set(1.0)
    
    # We must unroll the loop so XLA can compile it statically
    # We loop up to max_nodes to push probabilities down
    for i in range(max_nodes):
        current_reach = reach_probs[:, :, i]
        # Calculate what gets pushed left and right
        push_left = current_reach * left_probs[:, :, i]
        push_right = current_reach * right_probs[:, :, i]
        # Get the destination indices for the children
        l_idx = left_children[:, i]
        r_idx = right_children[:, i]
        # Update the reach probabilities of the children in-place
        # We use a mask to ignore padded nodes/leaves (where child index is -1)
        valid_left = l_idx != -1
        valid_right = r_idx != -1
        # JAX requires vmap to handle the batch dimension during advanced indexing
        def update_child(reach, left_i, right_i, p_left, p_right, v_left, v_right):
            reach = jnp.where(v_left[:, None], reach.at[jnp.arange(n_trees), left_i].add(p_left), reach)
            reach = jnp.where(v_right[:, None], reach.at[jnp.arange(n_trees), right_i].add(p_right), reach)
            return reach
        
        reach_probs = jax.vmap(update_child, in_axes=(0, None, None, 0, 0, None, None))(
            reach_probs, l_idx, r_idx, push_left, push_right, valid_left, valid_right
        )

    # The final prediction is the sum of (reach_prob * leaf_weight) for all nodes.
    # Because internal nodes have threshold weights too, we only want to sum the leaves.
    # A node is a leaf if its left_child is -1.
    is_leaf = left_children == -1
    # Mask out internal nodes
    leaf_reach_probs = jnp.where(is_leaf, reach_probs, 0.0)
    # Multiply probabilities by weights and sum across all nodes and trees
    weighted_leaves = leaf_reach_probs * thresholds
    return jnp.sum(weighted_leaves, axis=(1, 2))

# def calculate_paths_dense(left_probs, right_probs, ancestor_matrix, leaf_weights):
#     """
#     Option B: Pure Dense Tensor Math
#     left_probs: [batch, trees, max_nodes]
#     ancestor_matrix: [trees, n_leaves, max_nodes] (1=Left, 2=Right, 0=Ignore)
#     leaf_weights: [trees, n_leaves]
#     """
#     # expand probabilities so they can broadcast against the leaves now the shape becomes: [batch, trees, 1, max_nodes]
#     lp_expanded = jnp.expand_dims(left_probs, axis=2)
#     rp_expanded = jnp.expand_dims(right_probs, axis=2)
    
#     # helps in selecting the correct probability for every node in the path simultaneously, for example
#     # if Ancestor==1, use P_left and if Ancestor==2, use P_right. If 0, use 1.0 (neutral for multiplication)
#     path_probs = jnp.where(ancestor_matrix == 1, lp_expanded, jnp.where(ancestor_matrix == 2, rp_expanded, 1.0))
    
#     # now Multiply all the probabilities along the path (axis=3 is the nodes)
#     # Shape becomes: [batch, trees, leaves]
#     final_leaf_reach = jnp.prod(path_probs, axis=3)
    
#     # multiply by leaf weights and sum, Shape becomes: [batch]
#     return jnp.sum(final_leaf_reach * leaf_weights, axis=(1, 2))

def calculate_paths_dense(left_probs, right_probs, ancestor_matrix, leaf_weights):
    """
    Option B (Optimized for TPU MXU): Pure Dense Tensor Math using Log-Space.
    left_probs: [batch, trees, max_nodes]
    ancestor_matrix: [trees, n_leaves, max_nodes] (1=Left, 2=Right, 0=Ignore)
    leaf_weights: [trees, n_leaves]
    """
    # 1. Convert to Log-Space (Add small epsilon to prevent log(0))
    # Shape: [batch, trees, nodes]
    log_lp = jnp.log(left_probs + 1e-7)
    log_rp = jnp.log(right_probs + 1e-7)
    
    # 2. Extract the binary routing masks from the Ancestor Matrix
    # Shape: [trees, leaves, nodes]
    A_left = (ancestor_matrix == 1).astype(jnp.float32)
    A_right = (ancestor_matrix == 2).astype(jnp.float32)
    
    # 3. The MXU Bottleneck-Breaker: Batched Matrix Multiplication
    # We multiply the log-probabilities by the routing masks and sum across nodes
    # 'btn,tln->btl' means (batch, trees, nodes) * (trees, leaves, nodes) -> (batch, trees, leaves)
    log_path_left = jnp.einsum('btn,tln->btl', log_lp, A_left)
    log_path_right = jnp.einsum('btn,tln->btl', log_rp, A_right)
    
    # 4. Combine and convert back from Log-Space
    total_log_path = log_path_left + log_path_right
    final_leaf_reach = jnp.exp(total_log_path)
    
    # 5. Multiply by leaf weights and sum
    return jnp.sum(final_leaf_reach * leaf_weights, axis=(1, 2))

def verification_and_evaluation(batch_size: int,  temperature: float, seed: Optional[int] = None, method_type: Optional[str] = 'iterative', xla_fusion_analysis: Optional[bool] = False):
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
    """
    if seed is not None:
        np.random.seed(seed)
        print('[SOFTSIGMOID] evaluation will be done with reproducibility....')

    data_loader = CaliforniaHousingLoader()
    test_data = data_loader.get_test_samples(n=batch_size)
    print(f"[SOFTSIGMOID] [DEBUG] total entries in x test in data {len(data_loader.X_test)}")
    jax_sample_batch = jnp.array(test_data.values)
    # sample_batch = jnp.array(data_loader.get_test_samples(n=10).values)#
    # print(sample_batch.shape)
    # --- soft parser will provide with 3d matrix  ------
    soft_parser = SoftParser(trained_model_path="checkpoints/xgboost_model/xgboost_model.json")
    routing_matrices = soft_parser.routing_matrices(method_type=method_type)
    # lets pass this through soft sigmoid activations so we get left and right activations
    left_probs, right_probs = soft_node_activations(X_batch=jax_sample_batch, W=routing_matrices['W'], tau=routing_matrices['tau'], temperature=temperature)
    # pass these probs iteratively through the each tree to get results
    if method_type == "iterative":
        print(f"[SOFTSIGMOID] Iterative method called!")
        jax_forest_results = calculate_paths_iterative(left_probs=left_probs, right_probs=right_probs, left_children=routing_matrices['lefts'], right_children=routing_matrices['rights'], thresholds=routing_matrices['thresholds'])
    elif method_type == "dense":
        print(f"[SOFTSIGMOID] Dense method called!")
        jax_forest_results = calculate_paths_dense(left_probs=left_probs, right_probs=right_probs, ancestor_matrix=routing_matrices['A'], leaf_weights=routing_matrices['leaf_wts'])
    else:
        print(f'[SOFTSIGMOID] method type is not valid! Existing now .....')
        return
    # lets evaluate now
    evaluator = BaseLineEvaluator(mode='soft')
    real_base_score = evaluator.model_base_score
    # real_base_score = evaluator.model.__dict__['base_score'][0]
    print(f"[SOFTSIGMOID] Verifying Forest Logic for Batch Size: {test_data.shape[0]}!")
    print("\n--- Running SOFT SIGMOID Forest Check ---")
    forest_results = evaluator.check(X_sample=jax_sample_batch, single_tree=False, jax_preds=jax_forest_results)
    print(f"[SOFTSIGMOID] Full Forest Consistency: {forest_results['is_consistent']}")
    if xla_fusion_analysis:
        print("\n--- Extracting XLA HLO IR for Soft Fusion Analysis ---")
        # the pure function to compile (using normal temperature)
        @jax.jit
        def jitted_soft_iterative_predict(X, W, tau, lefts, rights, thresholds):
            lp, rp = soft_node_activations(X, W, tau, temperature=temperature)
            return calculate_paths_iterative(lp, rp, lefts, rights, thresholds)
        
        @jax.jit
        def jitted_soft_dense_predict(X, W, tau, ancestor_matrix, leaf_weights):
            lp, rp = soft_node_activations(X, W, tau, temperature=temperature)
            return calculate_paths_dense(lp, rp, ancestor_matrix, leaf_weights)
        
        if method_type == "iterative":
            # Lower it to XLA using your new routing matrices
            lowered = jitted_soft_iterative_predict.lower(
                jax_sample_batch, 
                routing_matrices['W'], 
                routing_matrices['tau'], 
                routing_matrices['lefts'], 
                routing_matrices['rights'], 
                routing_matrices['thresholds']
            )
        elif method_type == "dense":
            lowered = jitted_soft_dense_predict.lower(
                jax_sample_batch, 
                routing_matrices['W'], 
                routing_matrices['tau'], 
                routing_matrices['A'],
                routing_matrices['leaf_wts']
            )
        
        # Extract and save the Intermediate Representation
        hlo_ir = lowered.compiler_ir()
        filename = "soft_hlo_fusion_analysis.txt"
        with open(filename, "w") as f:
            f.write(str(hlo_ir))
        print(f"HLO IR successfully saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="soft tree  infernce")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size to test inference")
    parser.add_argument("--temp", type=float, default=10000.0, help="temperature for sigmoid")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--method", type=str, choices=['iterative', 'dense'], default='iterative', help="Choose routing method")
    args = parser.parse_args()

    verification_and_evaluation(batch_size=args.batch_size, temperature=args.temp, seed=args.seed, xla_fusion_analysis=True, method_type=args.method)