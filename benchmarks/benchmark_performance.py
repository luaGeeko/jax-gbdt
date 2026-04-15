import time
import jax
import jax.numpy as jnp
import numpy as np
import xgboost as xgb
import argparse
from src.data.loader import CaliforniaHousingLoader
from src.compiler.parser import TreeParser
from kernels.no_branch_inference import jax_forest_predict

def run_benchmark(batch_size: int = 10000, n_runs: int = 100):
    """
    Benchmark JAX-based fused tree inference against standard XGBoost inference.

    This function evaluates and compares the performance of a custom JAX/XLA-compiled
    forest inference kernel with the default XGBoost CPU implementation. It measures
    both latency and throughput under controlled batch sizes and repeated executions.

    The benchmark consists of:
    
    1. **Data Preparation**
       - Loads the full test dataset using `CaliforniaHousingLoader`.
       - Adjusts the batch size:
         - If requested batch size exceeds dataset size, the data is repeated (tiled).
         - Otherwise, a subset of the dataset is used.

    2. **Model Setup**
       - Loads a pre-trained XGBoost model from disk.
       - Parses the model into JAX-compatible array representations using `TreeParser`.

    3. **Baseline Benchmark (XGBoost)**
       - Runs inference using the standard XGBoost predictor.
       - Measures total execution time across multiple runs.
       - Reports:
         - Average latency (ms per run)
         - Throughput (inferences per second)

    4. **JAX/XLA Benchmark**
       - Compiles the JAX inference function using `jax.jit`.
       - Measures:
         - **Cold start time** (XLA compilation overhead)
         - **Warm execution time** (steady-state inference performance)
       - Reports:
         - Average latency
         - Throughput

    5. **Performance Comparison**
       - Computes relative speedup of JAX over XGBoost.

    Args:
        batch_size (int, optional):
            Number of samples per inference batch.
            Defaults to 10000.

        n_runs (int, optional):
            Number of repeated inference runs for timing stability.
            Defaults to 100.

    Outputs:
        Prints:
            - Detected hardware devices (CPU/GPU/TPU)
            - XGBoost latency and throughput
            - JAX compilation time
            - JAX latency and throughput
            - Final speedup comparison

    Notes:
        - JAX execution timing uses `.block_until_ready()` to ensure accurate measurement.
        - Compilation time is reported separately from execution time.
        - Throughput is computed as:
              (batch_size × n_runs) / total_time
        - This benchmark is designed to highlight kernel fusion and parallelism benefits in JAX.

    Use Cases:
        - Evaluating performance gains from compiler-level optimizations (XLA)
        - Comparing traditional ML inference vs compiled tensor programs
        - Stress-testing inference pipelines with large batch sizes
    """
    print(f"--- Hardware Found: {jax.devices()} ---")
    print(f"Preparing Benchmark (Batch Size: {batch_size}, Iterations: {n_runs})...")

    # load data and everything needed for setup
    data_loader = CaliforniaHousingLoader()
    # If batch_size > test set, we tile (repeat) the data to stress test the hardware
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
    # block_until_ready() is CRITICAL for accurate JAX timing
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
    
    # Final resutls as per speed and metrics
    speedup = jax_ips / xgb_ips
    print(f"\nFINAL VERDICT: JAX is {speedup:.2f}x faster than standard XGBoost on this hardware.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    
    run_benchmark(batch_size=args.batch_size, n_runs=args.runs)