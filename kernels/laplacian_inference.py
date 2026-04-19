import jax
import numpy as np
import jax.numpy as jnp
from jax import lax
from typing import Optional
import argparse

from src.data.loader import CaliforniaHousingLoader
from src.compiler.laplacian_parser import LaplacianParser
from scripts.evaluator import BaseLineEvaluator

def soft_node_activations(X_batch, W, tau, temperature=10.0):
    """
    Compute soft routing probabilities for decision tree splits.

    This function applies a sigmoid-based relaxation of hard decision
    boundaries, enabling differentiable routing through tree nodes.

    Args:
        X_batch (jnp.ndarray):
            Input feature batch of shape (batch_size, num_features).

        W (jnp.ndarray):
            Node weight tensor used to compute split decisions.

        tau (jnp.ndarray):
            Threshold values for each node.

        temperature (float, optional):
            Controls sharpness of the sigmoid function.
            Higher values approximate hard binary splits.
            Defaults to 10.0.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            - Left branch probabilities
            - Right branch probabilities
    """
    H = jnp.einsum('bf,tfn->btn', X_batch, W)
    left_probabilities = jax.nn.sigmoid(temperature * (tau - H))
    right_probabilities = 1.0 - left_probabilities
    return left_probabilities, right_probabilities

def calculate_laplacian_dense(left_probs, right_probs, dense_A_template, leaf_weights):
    """
    Perform Phase 3 Laplacian-based tree inference using dense matrices.

    This method models the decision tree as a directed acyclic graph (DAG)
    and propagates probability mass from root to leaves using a Neumann
    series (iterative matrix diffusion).

    The computation consists of:
    1. Constructing a probabilistic adjacency matrix A(x)
    2. Initializing a root signal vector
    3. Iteratively diffusing signal through the graph
    4. Aggregating leaf node contributions into final predictions

    Args:
        left_probs (jnp.ndarray):
            Probability of taking the left branch at each node.

        right_probs (jnp.ndarray):
            Probability of taking the right branch at each node.

        dense_A_template (jnp.ndarray):
            Encodes graph structure:
            - 1 indicates left edges
            - 2 indicates right edges

        leaf_weights (jnp.ndarray):
            Weights associated with leaf nodes for prediction.

    Returns:
        jnp.ndarray:
            Predicted outputs for each sample in the batch.
    """
    batch_size, n_trees, max_nodes = left_probs.shape
    
    # 1. Build the Adjacency Matrix A(x)
    # dense_A_template shape: [trees, nodes, nodes]
    A_left = jnp.where(dense_A_template == 1, 1.0, 0.0)
    A_right = jnp.where(dense_A_template == 2, 1.0, 0.0)
    
    # Broadcast probabilities into the Adjacency Matrix
    # Shape becomes: [batch, trees, nodes, nodes]
    A = A_left * left_probs[..., None] + A_right * right_probs[..., None]

    # 2. Initialize the Root Signal (S0)
    # Only the root node (index 0) gets a starting signal of 1.0
    S0 = jnp.zeros((batch_size, n_trees, max_nodes))
    S0 = S0.at[:, :, 0].set(1.0)
    
    total_signal = S0

    # 3. Neumann Series / DAG Diffusion: S_k = S_{k-1} @ A
    # Because it's a DAG, taking the power of A pushes the signal down the tree.
    # We loop max_nodes times to ensure the signal reaches the deepest possible leaf.
    def step_fn(i, val):
        """
        Single diffusion step in the Neumann series.

        Args:
            i (int): Iteration index (unused but required by JAX loop API).
            val (Tuple[jnp.ndarray, jnp.ndarray]):
                Current signal and accumulated signal.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                Updated signal and accumulated signal.
        """
        current_signal, tot_signal = val
        # Diffuse signal one step forward: bti (signal) * btij (adjacency) -> btj (new signal)
        current_signal = jnp.einsum('bti,btij->btj', current_signal, A)
        tot_signal += current_signal
        return current_signal, tot_signal

    # lax.fori_loop is XLA-optimized and unrolls statically for the compiler
    _, final_signal_state = lax.fori_loop(0, max_nodes, step_fn, (S0, total_signal))

    # 4. Extract final predictions by multiplying the diffused signal by leaf weights
    return jnp.sum(final_signal_state * leaf_weights, axis=(1, 2))


def verification_and_evaluation(batch_size: int, temperature: float, seed: Optional[int] = None, xla_fusion_analysis: bool = False):
    """
    Run verification and evaluation for Laplacian-based inference.

    This function:
    - Loads test data
    - Executes Laplacian-based inference
    - Verifies correctness against baseline evaluator
    - Optionally extracts XLA HLO IR for fusion analysis

    Args:
        batch_size (int):
            Number of samples used for evaluation.

        temperature (float):
            Sigmoid temperature controlling softness of splits.

        seed (Optional[int], optional):
            Random seed for reproducibility.
            Defaults to None.

        xla_fusion_analysis (bool, optional):
            If True, extracts and saves XLA HLO IR.
            Defaults to False.

    Returns:
        None:
            Prints evaluation results and verification status.
    """
    if seed is not None:
        np.random.seed(seed)
        print('[LAPLACIAN] Evaluation will be done with reproducibility....')

    data_loader = CaliforniaHousingLoader()
    test_data = data_loader.get_test_samples(n=batch_size)
    print(f"[LAPLACIAN] [DEBUG] Total entries in X_test: {len(data_loader.X_test)}")
    jax_sample_batch = jnp.array(test_data.values)

    # --- Phase 3 Laplacian Parser ---
    laplacian_parser = LaplacianParser(trained_model_path="checkpoints/xgboost_model/xgboost_model.json")
    W, tau = laplacian_parser.get_routing_matrices()
    topology = laplacian_parser.get_graph_topology()

    # --- JAX Execution ---
    left_probs, right_probs = soft_node_activations(
        X_batch=jax_sample_batch, W=W, tau=tau, temperature=temperature
    )

    print(f"[LAPLACIAN] Dense Graph Diffusion method called!")
    jax_forest_results = calculate_laplacian_dense(
        left_probs=left_probs, 
        right_probs=right_probs, 
        dense_A_template=topology['dense_A_template'], 
        leaf_weights=topology['leaf_weights']
    )

    # --- Verification ---
    evaluator = BaseLineEvaluator(mode='soft')
    print(f"[LAPLACIAN] Verifying Graph Diffusion Logic for Batch Size: {test_data.shape[0]}!")
    print("\n--- Running LAPLACIAN Neumann Check ---")
    forest_results = evaluator.check(X_sample=jax_sample_batch, single_tree=False, jax_preds=jax_forest_results)
    print(f"[LAPLACIAN] Full Forest Consistency: {forest_results['is_consistent']}")

    # --- HLO Extraction ---
    if xla_fusion_analysis:
        print("\n--- Extracting XLA HLO IR for Laplacian Fusion Analysis ---")
        
        @jax.jit
        def jitted_laplacian_dense(X, W_mat, tau_mat, A_template, leaves):
            """
            JIT-compiled Laplacian inference for HLO inspection.

            Args:
                X (jnp.ndarray): Input batch.
                W_mat (jnp.ndarray): Routing weights.
                tau_mat (jnp.ndarray): Thresholds.
                A_template (jnp.ndarray): Graph adjacency template.
                leaves (jnp.ndarray): Leaf weights.

            Returns:
                jnp.ndarray: Predicted outputs.
            """
            lp, rp = soft_node_activations(X, W_mat, tau_mat, temperature=temperature)
            return calculate_laplacian_dense(lp, rp, A_template, leaves)

        lowered = jitted_laplacian_dense.lower(
            jax_sample_batch, 
            jnp.array(W), 
            jnp.array(tau), 
            jnp.array(topology['dense_A_template']),
            jnp.array(topology['leaf_weights'])
        )
        
        filename = "laplacian_dense_hlo.txt"
        with open(filename, "w") as f:
            f.write(str(lowered.compiler_ir()))
        print(f"HLO IR successfully saved to {filename}")

if __name__ == "__main__":
    """
            JIT-compiled Laplacian inference for HLO inspection.

            Args:
                X (jnp.ndarray): Input batch.
                W_mat (jnp.ndarray): Routing weights.
                tau_mat (jnp.ndarray): Thresholds.
                A_template (jnp.ndarray): Graph adjacency template.
                leaves (jnp.ndarray): Leaf weights.

            Returns:
                jnp.ndarray: Predicted outputs.
            """
    parser = argparse.ArgumentParser(description="Laplacian tree inference")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size to test inference")
    parser.add_argument("--temp", type=float, default=1000000.0, help="temperature for sigmoid (high for hard matching)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    args = parser.parse_args()

    verification_and_evaluation(
        batch_size=args.batch_size, 
        temperature=args.temp, 
        seed=args.seed, 
        xla_fusion_analysis=True
    )