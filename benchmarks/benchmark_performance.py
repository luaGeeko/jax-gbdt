import time
import jax
import jax.numpy as jnp
import numpy as np
import xgboost as xgb
import argparse
from src.data.loader import CaliforniaHousingLoader
from src.compiler.parser import TreeParser
from kernels.no_branch_inference import jax_forest_predict

def run_benchmark(batch_size: int = 10000, n_runs: int = 100, xla_fusion_analysis: bool = False):
    """
    [... keep your existing docstring ...]
    """
    print(f"--- Hardware Found: {jax.devices()} ---")
    print(f"Preparing Benchmark (Batch Size: {batch_size}, Iterations: {n_runs})...")

    # load data and everything needed for setup
    data_loader = CaliforniaHousingLoader()
    test_data = data_loader.get_test_samples(n=len(data_loader.X_test))
    raw_batch = test_data.values
    
    # tiling of data
    if batch_size > len(raw_batch):
        repeats = (batch_size // len(raw_batch)) + 1
        raw_batch = np.tile(raw_batch, (repeats, 1))[:batch_size]
    else:
        raw_batch = raw_batch[:batch_size]

    np_batch = np.array(raw_batch)
    jax_batch = jnp.array(raw_batch)

    # load model and parse it to get 2d jax arrays
    model_path = "checkpoints/xgboost_model/xgboost_model.json"
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    parser = TreeParser(trained_model_path=model_path)
    jax_arrays = parser.parse()
    meta = parser.get_metadata()
    
    # XGBoost Baseline Benchmark (CPU)
    print("\n[1/2] Running Standard XGBoost (Baseline)...")
    xgb_start = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict(np_batch)
    xgb_total_time = time.perf_counter() - xgb_start
    xgb_ips = (batch_size * n_runs) / xgb_total_time
    print(f"XGBoost Avg Latency: {(xgb_total_time/n_runs)*1000:.2f} ms | Throughput: {xgb_ips:,.0f} IPS")

    # JAX/XLA Benchmark
    print("\n[2/2] Running JAX Fused Kernel...")
    jitted_predict = jax.jit(jax_forest_predict)

    # --- COLD START (Compilation) ---
    print("Triggering XLA Compilation...")
    comp_start = time.perf_counter()
    _ = jitted_predict(
        jax_batch, jax_arrays['features'], jax_arrays['thresholds'],
        jax_arrays['left_children'], jax_arrays['right_children'], 
        jax_arrays['max_nodes']
    ).block_until_ready()
    comp_time = time.perf_counter() - comp_start
    print(f"Compilation Time: {comp_time:.3f} s")

    # --- WARM START (Execution) ---
    print("Running JAX Warm Iterations...")
    jax_start = time.perf_counter()
    for _ in range(n_runs):
        _ = jitted_predict(
            jax_batch, jax_arrays['features'], jax_arrays['thresholds'],
            jax_arrays['left_children'], jax_arrays['right_children'], 
            jax_arrays['max_nodes']
        ).block_until_ready()
    jax_total_time = time.perf_counter() - jax_start
    jax_ips = (batch_size * n_runs) / jax_total_time
    
    print(f"JAX Avg Latency: {(jax_total_time/n_runs)*1000:.2f} ms | Throughput: {jax_ips:,.0f} IPS")
    
    speedup = jax_ips / xgb_ips
    print(f"\nFINAL VERDICT: JAX is {speedup:.2f}x faster than standard XGBoost on this hardware.")

    # --- HLO FUSION ANALYSIS ---
    if xla_fusion_analysis:
        print("\n[3/3] Extracting XLA HLO IR for Fusion Analysis...")
        # We can call .lower() directly on the jitted function we already created
        lowered = jitted_predict.lower(
            jax_batch, jax_arrays['features'], jax_arrays['thresholds'],
            jax_arrays['left_children'], jax_arrays['right_children'], 
            jax_arrays['max_nodes']
        )
        hlo_ir = lowered.compiler_ir()
        
        # Save with a specific name so it doesn't overwrite the inference analysis
        filename = "benchmark_hlo_fusion_analysis.txt"
        with open(filename, "w") as f:
            f.write(str(hlo_ir))
        print(f"HLO IR successfully saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--runs", type=int, default=100)
    # Using action="store_true" so you just pass the flag without a value
    parser.add_argument("--xla_fusion_analysis", action="store_true", help="Extract HLO IR for fusion analysis")
    args = parser.parse_args()
    
    run_benchmark(batch_size=args.batch_size, n_runs=args.runs, xla_fusion_analysis=args.xla_fusion_analysis)