# `jax-gbdt`: A High-Performance XLA Compiler for Tree Ensembles

## Core Objective

The goal of this project is to bridge the architectural gap between **Gradient Boosted Decision Trees (GBDTs)** and **modern AI accelerators (GPUs/TPUs)**.

Traditional GBDTs rely on recursive, control-flow-heavy branching logic that causes significant **thread divergence** and memory-bound bottlenecks on SIMD hardware. `jax-gbdt` transposes pre-trained tree ensembles into non-recursive, vectorized JAX primitives, enabling **XLA kernel fusion** and transforming "Logic-heavy" trees into "Arithmetic-heavy" data flows.


## System Dependencies

Some libraries (e.g. XGBoost) require the OpenMP runtime.

### macOS

```bash
brew install libomp
```

### Ubuntu / Debian

```bash
sudo apt install libomp-dev
```

### Windows

Usually installed automatically with XGBoost wheels.

After installing system dependencies, install Python packages:

```bash
pip install -r requirements.txt
```
