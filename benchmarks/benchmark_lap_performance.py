import time
import jax
import jax.numpy as jnp
import numpy as np
import xgboost as xgb
import argparse
import os

from src.data.loader import CaliforniaHousingLoader
from src.compiler.laplacian_parser import LaplacianParser
from kernels.laplacian_inference import soft_node_activations, calculate_laplacian_dense

def run_laplacian_benchmark(batch_size: int = 10000, n_runs: int = 100, xla_fusion_analysis: bool = False, method_type: str = 'dense'):
    """
    Run a performance benchmark comparing JAX-based Laplacian inference
    against standard XGBoost CPU inference.

    This function evaluates the throughput and latency of a compiled
    JAX/XLA Laplacian kernel against a traditional XGBoost model using
    the same dataset. It supports optional extraction of XLA HLO IR
    for fusion analysis.

    The benchmark consists of:
    1. Data loading and batching
    2. XGBoost CPU inference baseline
    3. JAX JIT-compiled Laplacian inference (dense method)
    4. Optional HLO IR extraction for kernel inspection

    Args:
        batch_size (int, optional):
            Number of samples per inference batch. If larger than the
            dataset size, data is tiled. Defaults to 10000.

        n_runs (int, optional):
            Number of repeated inference runs for benchmarking.
            Used to compute average latency and throughput.
            Defaults to 100.

        xla_fusion_analysis (bool, optional):
            If True, extracts and saves the compiled XLA HLO IR
            for inspection. Defaults to False.

        method_type (str, optional):
            Type of Laplacian inference method. Currently supported:
            - 'dense': Dense Laplacian computation
            - 'sparse': (Not yet implemented)

            Defaults to 'dense'.

    Raises:
        NotImplementedError:
            If a method other than 'dense' is selected.

    Returns:
        None:
            This function prints benchmark results including latency,
            throughput (IPS), and relative speedup.
    """
    print(f"--- Hardware Found: {jax.devices()} ---")
    print(f"Preparing Laplacian Benchmark (Batch Size: {batch_size}, Iterations: {n_runs}, Method: {method_type.upper()})...")

    # 1. Data Preparation and Tiling
    data_loader = CaliforniaHousingLoader()
    test_data = data_loader.get_test_samples(n=len(data_loader.X_test))
    raw_batch = test_data.values
    
    if batch_size > len(raw_batch):
        repeats = (batch_size // len(raw_batch)) + 1
        raw_batch = np.tile(raw_batch, (repeats, 1))[:batch_size]
    else:
        raw_batch = raw_batch[:batch_size]

    np_batch = np.array(raw_batch)
    jax_batch = jnp.array(raw_batch)

    # 2. XGBoost Baseline Setup
    model_path = "checkpoints/xgboost_model/xgboost_model.json"
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # 3. Laplacian Graph Parsing
    print(f"Parsing model into Phase 3 Graph Topology using {method_type} parser...")
    laplacian_parser = LaplacianParser(trained_model_path=model_path)
    W, tau = laplacian_parser.get_routing_matrices()
    topology = laplacian_parser.get_graph_topology()
    
    # Pre-load arrays onto device memory
    jax_W = jnp.array(W)
    jax_tau = jnp.array(tau)
    jax_A_dense = jnp.array(topology['dense_A_template'])
    jax_leaves = jnp.array(topology['leaf_weights'])
    
    # XGBoost Baseline Benchmark (CPU)
    print("\n[1/2] Running Standard XGBoost (Baseline)...")
    xgb_start = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict(np_batch)
    xgb_total_time = time.perf_counter() - xgb_start
    xgb_ips = (batch_size * n_runs) / xgb_total_time
    print(f"XGBoost Avg Latency: {(xgb_total_time/n_runs)*1000:.2f} ms | Throughput: {xgb_ips:,.0f} IPS")

    # JAX/XLA Laplacian Kernel Setup
    print(f"\n[2/2] Running JAX Laplacian Fused Kernel ({method_type})...")
    
    @jax.jit
    def jitted_laplacian_dense_predict(X, W_mat, tau_mat, A_template, leaves):
        # We use a standard temperature of 10.0 for stable execution benchmarking
        lp, rp = soft_node_activations(X, W_mat, tau_mat, temperature=10.0)
        return calculate_laplacian_dense(lp, rp, A_template, leaves)

    # --- COLD START (Compilation) ---
    print("Triggering XLA Compilation (Neumann Series Diffusion)...")
    comp_start = time.perf_counter()
    if method_type == "dense":
        _ = jitted_laplacian_dense_predict(
            jax_batch, jax_W, jax_tau, jax_A_dense, jax_leaves
        ).block_until_ready()
    else:
        raise NotImplementedError("Sparse CSR method coming soon!")
    comp_time = time.perf_counter() - comp_start
    print(f"Compilation Time: {comp_time:.3f} s")

    # --- WARM START (Execution) ---
    print("Running JAX Warm Iterations...")
    jax_start = time.perf_counter()
    for _ in range(n_runs):
        if method_type == "dense":
            _ = jitted_laplacian_dense_predict(
                jax_batch, jax_W, jax_tau, jax_A_dense, jax_leaves
            ).block_until_ready()
            
    jax_total_time = time.perf_counter() - jax_start
    jax_ips = (batch_size * n_runs) / jax_total_time
    
    print(f"JAX Laplacian Avg Latency: {(jax_total_time/n_runs)*1000:.2f} ms | Throughput: {jax_ips:,.0f} IPS")
    
    speedup = jax_ips / xgb_ips
    print(f"\nFINAL VERDICT: JAX Phase 3 ({method_type}) is {speedup:.2f}x faster than standard XGBoost.")

    # --- HLO FUSION ANALYSIS ---
    if xla_fusion_analysis:
        print(f"\n[3/3] Extracting XLA HLO IR for Phase 3 ({method_type}) Matrix Analysis...")
        
        if method_type == "dense":
            lowered = jitted_laplacian_dense_predict.lower(
                jax_batch, jax_W, jax_tau, jax_A_dense, jax_leaves
            )
            
        hlo_ir = lowered.compiler_ir()
        filename = f"laplacian_{method_type}_benchmark_hlo.txt"
        with open(filename, "w") as f:
            f.write(str(hlo_ir))
        print(f"HLO IR successfully saved to {filename}")

if __name__ == "__main__":
    """
    Command-line entry point for running the Laplacian benchmark.

    This script allows users to configure batch size, number of runs,
    inference method, and optional XLA HLO extraction via CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Laplacian benchmark")
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--xla_fusion_analysis", action="store_true", help="Extract HLO IR")
    parser.add_argument("--method", type=str, choices=['dense', 'sparse'], default='dense', help="Choose routing method")
    args = parser.parse_args()
    
    run_laplacian_benchmark(
        batch_size=args.batch_size, 
        n_runs=args.runs, 
        xla_fusion_analysis=args.xla_fusion_analysis,
        method_type=args.method
    )