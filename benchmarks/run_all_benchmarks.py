import time
import jax
import jax.numpy as jnp
import numpy as np
import xgboost as xgb
from typing import Optional
import argparse
import os
import csv
import traceback

from src.data.loader import CaliforniaHousingLoader
from src.compiler.parser import TreeParser
from src.compiler.soft_sigmoid_parser import SoftParser
from src.compiler.laplacian_parser import LaplacianParser

from kernels.no_branch_inference import jax_forest_predict
from kernels.soft_sigmoid_inference import soft_node_activations, calculate_paths_iterative, calculate_paths_dense
from kernels.laplacian_inference import calculate_laplacian_dense

def log_to_csv(filepath: str, data: dict):
    file_exists = os.path.isfile(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def run_unified_benchmark(batch_sizes: list, n_runs: Optional[int] = 10, temperature: Optional[float] = 100000.0, log_file: str = "results/thesis_benchmarks.csv"):
    hardware_name = str(jax.devices()[0].device_kind)
    print(f"--- Hardware Found: {hardware_name} ---")
    
    # 1. Dataset and Model
    data_loader = CaliforniaHousingLoader()
    base_raw_batch = data_loader.get_test_samples(n=len(data_loader.X_test)).values
    
    model_path = "checkpoints/xgboost_model/xgboost_model.json"
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # 2. Parse all topologies ONCE at the start to save time
    print("Parsing all mathematical topologies...")
    
    # Phase 1: No Branch
    tree_parser = TreeParser(trained_model_path=model_path)
    jax_2d_arrays = tree_parser.parse()
    
    # Phase 2: Soft Trees (Iterative & Dense)
    soft_parser = SoftParser(trained_model_path=model_path)
    p2_iter = soft_parser.routing_matrices(method_type='iterative')
    p2_dense = soft_parser.routing_matrices(method_type='dense')
    
    # Phase 3: Laplacian
    lap_parser = LaplacianParser(trained_model_path=model_path)
    p3_W, p3_tau = lap_parser.get_routing_matrices()
    p3_top = lap_parser.get_graph_topology()

    # 3. Define JIT Kernels ONCE
    @jax.jit
    def jit_no_branch(X):
        return jax_forest_predict(X, jax_2d_arrays['features'], jax_2d_arrays['thresholds'], 
                                  jax_2d_arrays['left_children'], jax_2d_arrays['right_children'], jax_2d_arrays['max_nodes'])

    @jax.jit
    def jit_soft_iterative(X):
        lp, rp = soft_node_activations(X, p2_iter['W'], p2_iter['tau'], temperature=temperature)
        return calculate_paths_iterative(lp, rp, p2_iter['lefts'], p2_iter['rights'], p2_iter['thresholds'])

    @jax.jit
    def jit_soft_dense(X):
        lp, rp = soft_node_activations(X, p2_dense['W'], p2_dense['tau'], temperature=temperature)
        return calculate_paths_dense(lp, rp, p2_dense['A'], p2_dense['leaf_wts'])

    @jax.jit
    def jit_laplacian_dense(X):
        lp, rp = soft_node_activations(X, p3_W, p3_tau, temperature=temperature)
        return calculate_laplacian_dense(lp, rp, p3_top['dense_A_template'], p3_top['leaf_weights'])

    methods_map = {
        'no_branch': jit_no_branch,
        'soft_iterative': jit_soft_iterative,
        'soft_dense': jit_soft_dense,
        'laplacian_dense': jit_laplacian_dense
    }

    # 4. Run the Master Loop
    for method_name, kernel_fn in methods_map.items():
        for batch_size in batch_sizes:
            print(f"\n{'='*60}")
            print(f"BATCH SIZE: {batch_size:,} | METHOD: {method_name.upper()}")
            print(f"{'='*60}")
            
            # Tile data
            if batch_size > len(base_raw_batch):
                repeats = (batch_size // len(base_raw_batch)) + 1
                raw_batch = np.tile(base_raw_batch, (repeats, 1))[:batch_size]
            else:
                raw_batch = base_raw_batch[:batch_size]

            np_batch = np.array(raw_batch)
            jax_batch = jnp.array(raw_batch)

            # --- Baseline CPU Math ---
            xgb_start = time.perf_counter()
            for _ in range(n_runs):
                _ = model.predict(np_batch)
            xgb_total_time = time.perf_counter() - xgb_start
            xgb_latency_ms = (xgb_total_time / n_runs) * 1000
            xgb_ips = (batch_size * n_runs) / xgb_total_time
            print(f"[Baseline] Latency: {xgb_latency_ms:.2f} ms | Throughput: {xgb_ips:,.0f} IPS")

            # --- JAX Execution with OOM Protection ---
            try:
                # Cold Start
                comp_start = time.perf_counter()
                _ = kernel_fn(jax_batch).block_until_ready()
                comp_time = time.perf_counter() - comp_start
                print(f"Compilation Time: {comp_time:.3f} s")

                # Warm Start
                jax_start = time.perf_counter()
                for _ in range(n_runs):
                    _ = kernel_fn(jax_batch).block_until_ready()
                jax_total_time = time.perf_counter() - jax_start
                
                jax_latency_ms = (jax_total_time / n_runs) * 1000
                jax_ips = (batch_size * n_runs) / jax_total_time
                speedup = jax_ips / xgb_ips
                
                print(f"[JAX] Latency: {jax_latency_ms:.2f} ms | Throughput: {jax_ips:,.0f} IPS")
                print(f"VERDICT: {speedup:.2f}x faster than XGBoost")
                status = "SUCCESS"

            except Exception as e:
                print(f"XLA COMPILER OR HARDWARE FAILED! (Likely OOM)")
                print(str(e).split('\n')[0]) # Print just the first line of the error
                comp_time = 0.0
                jax_latency_ms = 0.0
                jax_ips = 0.0
                speedup = 0.0
                status = "OOM/FAILED"

            # Log to CSV
            log_data = {
                "Hardware": hardware_name,
                "Method": method_name,
                "Batch_Size": batch_size,
                "Status": status,
                "Compilation_Time_sec": round(comp_time, 3),
                "XGBoost_Latency_ms": round(xgb_latency_ms, 3),
                "XGBoost_IPS": round(xgb_ips, 0),
                "JAX_Latency_ms": round(jax_latency_ms, 3),
                "JAX_IPS": round(jax_ips, 0),
                "Speedup_Multiplier": round(speedup, 2)
            }
            log_to_csv(log_file, log_data)
            print(f"Saved to CSV.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=int, nargs='+', default=[10000, 50000, 100000, 500000], help="List of batch sizes")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--log_file", type=str, default="results/thesis_benchmarks.csv")
    args = parser.parse_args()
    
    run_unified_benchmark(batch_sizes=args.batch_sizes, n_runs=args.runs, log_file=args.log_file)