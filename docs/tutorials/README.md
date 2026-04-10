# Eshkol Tutorials

Practical, runnable tutorials covering every major feature of the Eshkol
programming language. Each tutorial is self-contained with code examples
that work in both the native compiler (`eshkol-run`) and the browser REPL
at [eshkol.ai](https://eshkol.ai) (where applicable).

---

## Start Here

| New to Eshkol? | Already know Scheme? |
|---|---|
| [First 5 Minutes](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/00_FIRST_5_MINUTES.md) | [Why Eshkol?](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/WHY_ESHKOL.md) |
| Install, hello world, 5 wow moments | Side-by-side vs Python/JS |

**Need a quick recipe?** [Cookbook](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/COOKBOOK.md) — 30 copy-paste solutions.

---

## Feature Tutorials

Start here to learn what Eshkol can do.

| # | Tutorial | What you'll learn |
|---|---|---|
| 01 | [Autodiff and ML](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/01_AUTODIFF_AND_ML.md) | Derivatives, gradients, Hessians, gradient descent from scratch |
| 02 | [Bytecode VM](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/02_BYTECODE_VM.md) | REPL, hot reload, `-B` flag, VM vs LLVM, browser REPL |
| 03 | [Weight Matrix Transformer](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/03_WEIGHT_MATRIX_TRANSFORMER.md) | Programs as neural network weights, 5-layer architecture |
| 04 | [Consciousness Engine](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/04_CONSCIOUSNESS_ENGINE.md) | Knowledge bases, unification, factor graphs, belief propagation |
| 05 | [Signal Processing](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/05_SIGNAL_PROCESSING.md) | FFT, window functions, FIR/IIR filters, Butterworth design |
| 06 | [Exact Arithmetic](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/06_EXACT_ARITHMETIC.md) | Bignums, rationals, complex numbers, automatic promotion |
| 07 | [Parallel Computing](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/07_PARALLEL_COMPUTING.md) | parallel-map, parallel-fold, futures, thread pool |
| 08 | [Continuations and Exceptions](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/08_CONTINUATIONS_AND_EXCEPTIONS.md) | call/cc, dynamic-wind, guard/raise, tail calls |
| 09 | [Module System](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/09_MODULE_SYSTEM.md) | require/provide, stdlib modules, creating libraries |
| 10 | [Macros](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/10_MACROS.md) | syntax-rules, hygiene, homoiconicity |
| 11 | [Tensors and Linear Algebra](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/11_TENSORS_AND_LINEAR_ALGEBRA.md) | Creation, reshape, matmul, GPU dispatch |
| 12 | [Lists](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/12_LISTS.md) | 60+ list operations, map/fold/filter |
| 13 | [Strings and I/O](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/13_STRINGS_AND_IO.md) | String operations, file I/O, ports |
| 14 | [Data Formats](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/14_DATA_FORMATS.md) | JSON, CSV, Base64 |
| 15 | [Pattern Matching](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/15_PATTERN_MATCHING.md) | match expressions, destructuring |
| 16 | [Hash Tables](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/16_HASH_TABLES.md) | Creation, lookup, mutation |
| 17 | [Functional Programming](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/17_FUNCTIONAL_PROGRAMMING.md) | Closures, composition, currying, memoisation |
| 18 | [Web Platform](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/18_WEB_PLATFORM.md) | WASM target, DOM API, browser apps |
| 19 | [GPU Acceleration](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/19_GPU_ACCELERATION.md) | Metal/CUDA dispatch, SIMD, cost model |
| 20 | [Bitwise and System](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/20_BITWISE_AND_SYSTEM.md) | Bitwise ops, environment, file system |

## Project Tutorials

Complete, working programs that solve real problems.

| # | Project | What you'll build |
|---|---|---|
| 21 | [Neural Network](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/21_PROJECT_NEURAL_NETWORK.md) | Train a network to learn XOR using only autodiff |
| 22 | [Expert System](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/22_PROJECT_EXPERT_SYSTEM.md) | Medical diagnosis with KB + factor graphs |
| 23 | [Data Pipeline](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/23_PROJECT_DATA_PIPELINE.md) | Statistics, filtering, normalisation on a dataset |
| 24 | [Function Optimisation](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/24_PROJECT_OPTIMISATION.md) | Rosenbrock, Newton's method, curve fitting |
| 25 | [Calculator/Interpreter](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/25_PROJECT_INTERPRETER.md) | S-expression evaluator demonstrating homoiconicity |
| 26 | [Self-Improving Programs](https://github.com/tsotchke/eshkol/blob/master/docs/tutorials/26_PROJECT_SELF_IMPROVING.md) | Differentiable self-optimisation + active inference reasoning |

---

## Running the Examples

### Native compiler (fastest)

```bash
$ eshkol-run tutorial.esk -o tutorial && ./tutorial
```

### Browser REPL (no install needed)

Visit [eshkol.ai](https://eshkol.ai) and paste code into the REPL.

### Interactive REPL (local)

```bash
$ eshkol-repl
> (+ 1 2 3)
6
```
