# GPU Dispatch — Honest Status

This page states what actually runs on the GPU today, verified on the v1.3.0
build on an **Apple M2 Ultra (Metal)**. CUDA and XLA notes are marked as such.
The GPU-campaign ledger tasks **ESH-0022** and **ESH-0023** describe an older
state ("`gpu-*` are Unknown function", "AOT runs CPU BLAS only") that is
**partly stale** on this build — see below.

---

## The `gpu-*` builtins

All five resolve as codegen builtins (`lib/backend/llvm_codegen.cpp`) in **both
`-r` (JIT) and AOT** — none are "Unknown function" on this build.

| Builtin | Signature | Result |
|---------|-----------|--------|
| `gpu-matmul` | `(gpu-matmul A B)` | ✅ `#((7 10) (15 22))` for `[[1,2],[3,4]]²`; probes Metal |
| `gpu-elementwise` | `(gpu-elementwise OP A B)` | ✅ `(gpu-elementwise + A A)` → elementwise sum. `OP` is a **bare** `+ - * /` token (also `add`/`tensor-add` spellings), not a quoted symbol |
| `gpu-softmax` | `(gpu-softmax t)` | ✅ `(gpu-softmax (tensor 1.0 2.0 3.0))` → `#(0.0900306 0.244728 0.665241)` |
| `gpu-transpose` | `(gpu-transpose A)` | ✅ 2×2 → `#((1 3) (2 4))` |
| `gpu-reduce` | `(gpu-reduce OP t)` | ✅ full reduction to a **scalar** (`(gpu-reduce + (tensor 1.0 2.0 3.0 4.0))` → `10`). `OP` is a bare `+ mean max min` token |

```scheme
(define A (reshape (tensor 1.0 2.0 3.0 4.0) (list 2 2)))
(gpu-matmul A A)             ;; => #((7 10) (15 22))
(gpu-elementwise + A A)      ;; => #((2 4) (6 8))
(gpu-reduce + (tensor 1.0 2.0 3.0 4.0))  ;; => 10
```

Note: `gpu-elementwise`/`gpu-reduce` take the operator as a **bare identifier**
(`+`, not `'+`); a quoted symbol raises the "requires (gpu-… <op> …)" error.

---

## What actually runs on the GPU

The GPU is engaged through a **cost model**, not just the `gpu-*` names. Plain
`tensor-matmul`/`matmul` and the `gpu-*` aliases share the same dispatch in
`lib/core/blas_backend.cpp`:

- Dispatch is gated by `ESHKOL_GPU_MATMUL_THRESHOLD` (default
  `1000000000` output elements). Below threshold → CPU BLAS/Accelerate; at or
  above → **Metal**.
- The Metal device is **probed and autotuned on the first matmul in both `-r`
  and AOT** (prints ~20 `[GPU] …` config lines, including the Ozaki-II GEMM
  pipeline). This happens even for a 2×2 — the probe fires, but small compute
  stays on the CPU under the default threshold.
- Forcing `ESHKOL_GPU_MATMUL_THRESHOLD=0` engages the **Metal Ozaki-II GEMM**
  and returns correct results in **both `-r` and the AOT binary** (verified: a
  256×256 AOT matmul prints the Metal banner and computes on-GPU).

```sh
# Force GPU dispatch for smaller matmuls
ESHKOL_GPU_MATMUL_THRESHOLD=0 ./build/eshkol-run -r matmul.esk
```

**Bottom line for Metal:** GPU matmul is functional and dispatches in *both*
JIT and AOT when the size threshold is met — AOT is **not** CPU-only on this
box. This contradicts the literal wording of ESH-0022/ESH-0023.

---

## Ledger status vs. observed

| Task | Ledger claim | Observed on this build (Metal) |
|------|--------------|--------------------------------|
| **ESH-0022** | `gpu-matmul`/`gpu-elementwise`/`gpu-softmax`/`gpu-reduce`/`gpu-transpose` are "Unknown function" in both paths | ✅ all five **resolve and compute correctly** in `-r` and AOT (`gpu-reduce` now returns a scalar) |
| **ESH-0023** | AOT-compiled binary runs matmul on CPU BLAS even in a GPU build | On **Metal**, AOT matmul dispatches to the GPU when the threshold is met (verified). The task was filed against **CUDA/RTX 3050**, which is not exercised here |

Treat ESH-0022/0023 as **largely resolved**: `gpu-reduce` now returns a scalar
(full reduction). What remains genuinely pending is low-level GPU-specific
low-precision dtype builtins on non-Metal backends.

---

## CUDA / XLA

- **CUDA**: the CUDA runtime and cost-model dispatch exist
  (`lib/backend/*gpu*`, GPU-campaign PRs), and lazy-init was made
  reachable-from-language in the cross-platform campaign. ESH-0023's specific
  observation (compiled binary at ~11% GPU util, CPU BLAS in the CUDA build)
  was measured on an RTX 3050 and is **not re-verified here** — treat CUDA AOT
  GPU dispatch as unconfirmed on this build.
- **XLA/StableHLO**: an optional AOT lane (`ESHKOL_LLVM_DIS`/StableHLO config in
  CMake). It is a build-time backend option, not a per-op runtime dispatch, and
  is not exercised by the examples on this page.

For end-to-end GPU tests and benchmarks see
[`tests/gpu/`](../../../tests/gpu/) (matmul/reduce/transpose/softmax/elementwise
correctness, `sf64_*` software-float kernels, CUDA host-sync regression) and
[`benchmarks/`](../../../benchmarks/) (`gpu_matmul_bench.sh`,
`gpu_vs_cpu_bench.esk`, `matmul_extreme.sh`).

---

## See also

- [operations.md](operations.md) — `tensor-matmul` and the full op surface
- [creation.md](creation.md#data-types-dtypes) — dtypes (f16/bf16/f32/f64/i8)
- [../../breakdown/AUTODIFF.md](../../breakdown/AUTODIFF.md) — GPU gradient flow (backward pass dispatch)
