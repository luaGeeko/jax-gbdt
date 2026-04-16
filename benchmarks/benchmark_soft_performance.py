import time
import jax
import jax.numpy as jnp
import numpy as np
import xgboost as xgb
import argparse
import os

from src.data.loader import CaliforniaHousingLoader
from src.compiler.soft_sigmoid_parser import SoftParser
from kernels.soft_sigmoid_inference import soft_node_activations, calculate_paths_iterative, calculate_paths_dense

def run_soft_benchmark(batch_size: int = 10000, n_runs: int = 100, xla_fusion_analysis: bool = False, method_type: str = 'iterative'):
    """
    Benchmark JAX-based Soft Tree (Dense Matrix) inference against standard XGBoost CPU inference.
    """
    print(f"--- Hardware Found: {jax.devices()} ---")
    print(f"Preparing Soft Tree Benchmark (Batch Size: {batch_size}, Iterations: {n_runs}, Method: {method_type.upper()})...")

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
    
    # 3. Soft Tree Matrix Parsing
    print(f"Parsing model into Phase 2 Routing Matrices using {method_type} parser...")
    soft_parser = SoftParser(trained_model_path=model_path)
    routing_matrices = soft_parser.routing_matrices(method_type=method_type)
    
    # XGBoost Baseline Benchmark (CPU)
    print("\n[1/2] Running Standard XGBoost (Baseline)...")
    xgb_start = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict(np_batch)
    xgb_total_time = time.perf_counter() - xgb_start
    xgb_ips = (batch_size * n_runs) / xgb_total_time
    print(f"XGBoost Avg Latency: {(xgb_total_time/n_runs)*1000:.2f} ms | Throughput: {xgb_ips:,.0f} IPS")

    # JAX/XLA Soft Kernel Setup
    print(f"\n[2/2] Running JAX Soft Tree Fused Kernel ({method_type})...")
    
    # Define BOTH JIT kernels explicitly
    @jax.jit
    def jitted_soft_iterative_predict(X, W, tau, lefts, rights, thresholds):
        lp, rp = soft_node_activations(X, W, tau, temperature=10.0)
        return calculate_paths_iterative(lp, rp, lefts, rights, thresholds)

    @jax.jit
    def jitted_soft_dense_predict(X, W, tau, ancestor_matrix, leaf_weights):
        lp, rp = soft_node_activations(X, W, tau, temperature=10.0)
        return calculate_paths_dense(lp, rp, ancestor_matrix, leaf_weights)

    # --- COLD START (Compilation) ---
    print("Triggering XLA Compilation (cuBLAS dot_general mapping)...")
    comp_start = time.perf_counter()
    if method_type == "iterative":
        _ = jitted_soft_iterative_predict(
            jax_batch, 
            routing_matrices['W'], 
            routing_matrices['tau'], 
            routing_matrices['lefts'], 
            routing_matrices['rights'], 
            routing_matrices['thresholds']
        ).block_until_ready()
    elif method_type == "dense":
        _ = jitted_soft_dense_predict(
            jax_batch, 
            routing_matrices['W'], 
            routing_matrices['tau'], 
            routing_matrices['A'],
            routing_matrices['leaf_wts']  # Ensure this matches your SoftParser dictionary key exactly!
        ).block_until_ready()
    comp_time = time.perf_counter() - comp_start
    print(f"Compilation Time: {comp_time:.3f} s")

    # --- WARM START (Execution) ---
    print("Running JAX Warm Iterations...")
    jax_start = time.perf_counter()
    for _ in range(n_runs):
        if method_type == "iterative":
            _ = jitted_soft_iterative_predict(
                jax_batch, 
                routing_matrices['W'], 
                routing_matrices['tau'], 
                routing_matrices['lefts'], 
                routing_matrices['rights'], 
                routing_matrices['thresholds']
            ).block_until_ready()
        elif method_type == "dense":
            _ = jitted_soft_dense_predict(
                jax_batch, 
                routing_matrices['W'], 
                routing_matrices['tau'], 
                routing_matrices['A'],
                routing_matrices['leaf_wts']
            ).block_until_ready()
            
    jax_total_time = time.perf_counter() - jax_start
    jax_ips = (batch_size * n_runs) / jax_total_time
    
    print(f"JAX Soft Tree Avg Latency: {(jax_total_time/n_runs)*1000:.2f} ms | Throughput: {jax_ips:,.0f} IPS")
    
    speedup = jax_ips / xgb_ips
    print(f"\nFINAL VERDICT: JAX Phase 2 ({method_type}) is {speedup:.2f}x faster than standard XGBoost.")

    # --- HLO FUSION ANALYSIS ---
    if xla_fusion_analysis:
        print(f"\n[3/3] Extracting XLA HLO IR for Phase 2 ({method_type}) Matrix Analysis...")
        
        if method_type == "iterative":
            lowered = jitted_soft_iterative_predict.lower(
                jax_batch, 
                routing_matrices['W'], 
                routing_matrices['tau'], 
                routing_matrices['lefts'], 
                routing_matrices['rights'], 
                routing_matrices['thresholds']
            )
        elif method_type == "dense":
            lowered = jitted_soft_dense_predict.lower(
                jax_batch, 
                routing_matrices['W'], 
                routing_matrices['tau'], 
                routing_matrices['A'],
                routing_matrices['leaf_wts']
            )
            
        hlo_ir = lowered.compiler_ir()
        
        # Save uniquely based on the method
        filename = f"soft_{method_type}_benchmark_hlo.txt"
        with open(filename, "w") as f:
            f.write(str(hlo_ir))
        print(f"HLO IR successfully saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--xla_fusion_analysis", action="store_true", help="Extract HLO IR")
    parser.add_argument("--method", type=str, choices=['iterative', 'dense'], default='iterative', help="Choose routing method")
    args = parser.parse_args()
    
    run_soft_benchmark(batch_size=args.batch_size,  n_runs=args.runs,  xla_fusion_analysis=args.xla_fusion_analysis, method_type=args.method)