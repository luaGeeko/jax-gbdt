# # JAX-GBDT: JAX Tree Compiler

A high-performance compiler that translates trained Gradient Boosted Decision Tree (GBDT) ensembles (e.g., XGBoost) into **GPU/TPU-optimized tensor programs** using JAX and XLA.

---

## 🚀 The Hardware Bottleneck

Traditional decision tree inference is fundamentally misaligned with the architecture of modern accelerators:

* ❌ **Branch Divergence:** Sequential `if/else` logic forces GPUs to serialize execution across warps.
* ❌ **Pointer-Chasing:** Dynamic tree traversal causes irregular memory access and cache misses.
* ❌ **VPU Fallback:** Hardware is forced to use slower Vector Processing Units rather than high-throughput Matrix Multiply Units (MXUs).

As a result, while GPUs/TPUs can execute trillions of dense matrix operations per second for Neural Networks, traditional tree models often run slower on accelerators than on standard CPUs.

---

## 💡 The Architectural Breakthrough

**JAX-GBDT** solves this by treating a decision forest not as a collection of branching statements, but as a series of **Directed Acyclic Graphs (DAGs)** that can be mathematically relaxed into continuous tensor space. 

Instead of traversing a tree, JAX-GBDT evaluates it using pure tensor calculus:
> **Matrix operations replace control flow.**

Through an architectural ablation study, this project evaluated multiple mathematical equivalencies to bypass hardware limits:
1. **Discrete Tensorization:** Removing Python overhead but keeping discrete branching.
2. **Sparse Continuous Relaxation:** Converting branches to differentiable probabilities using scatter/gather memory.
3. **Dense Graph Laplacian:** Treating trees as Markov chains (exposing $O(N^2)$ memory limits).
4. **Log-Space Dense Tensor Relaxation (The Solution):** Collapsing path probabilities into log-space addition, retaining $O(N)$ spatial complexity while natively saturating the hardware's systolic arrays.

---

## ⚡ Performance Reality

By translating XGBoost logic into the language of silicon accelerators, JAX-GBDT achieves exponential scaling on massively batched workloads. 

* ✅ **Zero Branch Divergence:** Fully vectorized evaluation using `jnp.einsum`.
* ✅ **XLA JIT Compilation:** Fuses the entire ensemble topology into a single silicon kernel.
* ✅ **Throughput Saturated:** Capable of achieving **427.0 Million Inferences Per Second (IPS)** on Google TPU v5 hardware.

---

## 📚 Documentation Navigation

Get started with JAX-GBDT by exploring the compiler mechanics and API:

* 👉 **[Quickstart & API](api.md):** Learn how to load and compile an XGBoost model in 3 lines of code.
* 👉 **[The Core Parsers](parser.md):** Understand how we extract tree topologies into static JAX arrays.
* 👉 **[Kernel Implementations](kernels.md):** Dive into the math behind the Log-Space Einsum optimization.
* 👉 **[Hardware Benchmarks](benchmarking.md):** View the full scaling ablation study across CPU, GPU, and TPU architectures.