# JAX Tree Compiler

A system for converting tree-based models (e.g., XGBoost) into **GPU/TPU-optimized tensor programs** using JAX.

---

## 🚀 Motivation

Traditional decision tree inference is not well-suited for modern accelerators:

* ❌ Pointer-based traversal (cache misses)
* ❌ Branch divergence on GPUs
* ❌ Irregular computation graphs
* ❌ Poor utilization of SIMD/SIMT hardware

This makes tree models significantly slower than neural networks on GPUs/TPUs.

---

## 💡 Core Idea

We transform trees into **static tensor representations**:

* Flatten trees into arrays
* Pad to fixed depth
* Remove pointer traversal
* Encode decisions as tensor operations

This allows:

* ✅ Coalesced memory access
* ✅ Fully vectorized evaluation
* ✅ JAX JIT compilation
* ✅ Efficient GPU/TPU execution

---

## 🧠 Intuition

Instead of traversing a tree, during inference:
> **matrix operations instead of control flow**


## 📚 Next Steps

👉 See [Parser](parser.md) for implementation details.
