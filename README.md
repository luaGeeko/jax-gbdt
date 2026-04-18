# JAX-GBDT: A High-Performance XLA Compiler for Tree Ensembles 🌲⚡

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-FF9900.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**JAX-GBDT** is a high-performance tree-to-code compiler that translates trained Gradient Boosted Decision Tree (GBDT) ensembles (like XGBoost) into pure tensor operations. By reformulating discrete, sequential `if/else` branching logic into continuous matrix mathematics, JAX-GBDT allows decision forests to natively saturate the Matrix Multiply Units (MXUs) of modern GPUs and TPUs.

## 🚀 Why JAX-GBDT?
Standard decision trees suffer from severe **branch divergence** and pointer-chasing memory bottlenecks when executed on parallel hardware accelerators. JAX-GBDT solves this using a novel **Log-Space Dense Tensor Relaxation**, converting tree topologies into static binary routing masks.

* **Ultra-High Throughput:** Achieves up to **427.0 Million Inferences Per Second (IPS)** on Google TPU v5, a massive exponential speedup over traditional CPU inference.
* **Pure Tensor Calculus:** Eliminates `$O(N^2)$` memory explosion and sparse scatter/gather bottlenecks.
* **XLA Fused:** Fully compatible with Google's Accelerated Linear Algebra (XLA) compiler for Just-In-Time (JIT) silicon execution.
* **Zero Python Overhead:** Drops directly into standard JAX pipelines with no interpreter latency.

## 📦 Installation

Clone the repository and install the dependencies. *(Note: Ensure you have the correct JAX version installed for your specific CUDA/TPU hardware).*

```bash
git clone [https://github.com/yourusername/jax-gbdt.git](https://github.com/yourusername/jax-gbdt.git)
cd jax-gbdt
pip install -r requirements.txt
```

## ⚡ Quickstart

JAX-GBDT provides a clean, Scikit-Learn style API. Just load your trained XGBoost model (saved as JSON) and compile it to silicon.
Refer notebook for quickstart under colab_notebooks

## Performance Benchmarks
| Hardware Architecture | Methodology                          | Peak Throughput       | Status                     |
|----------------------|--------------------------------------|------------------------|----------------------------|
| Mac CPU (Baseline)   | Standard XGBoost (C++)               | ~10.7 Million IPS      | ✅ Optimal for CPU         |
| NVIDIA Tesla T4      | JAX-GBDT (Phase 1: Discrete)         | ~104.6 Million IPS     | ✅ 10x Speedup             |
| Google TPU v5        | JAX-GBDT (Phase 3: Dense Laplacian)  | 0.0 IPS                | ❌ OOM Failure             |
| Google TPU v5        | JAX-GBDT (Phase 2: Log-Space Einsum) | 427.0 Million IPS      | 🚀 Hardware Saturated      |


## Citation

```
@misc{jaxgbdt2026,
  author       = {Shruti Verma},
  title        = {JAX-GBDT: A High-Performance XLA Compiler for Tree Ensembles},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/yourusername/jax-gbdt](https://github.com/luaGeeko/jax-gbdt)}},
  note         = {Technical Research Report}
}
```