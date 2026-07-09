# Eshkol Language Update Brief — Full Modernization for GPU LLM Inference

**To:** Eshkol language agent
**From:** MoThRA / Kimi-on-mesh effort (computer_mesh, mothra)
**Date:** 2026-06-17
**Mandate:** Completely update the language. Eshkol is being adopted as the
compute + control substrate for running **Kimi K2.6 as a fast, lossless,
general-purpose LLM** on modest hardware (old-donkey: RTX 3050 6 GB + CPU + disk
streaming), accelerated by MoThRA and geometric inference. This work stressed
Eshkol against a real, heavy GPU workload and surfaced concrete gaps. This brief
is the punch-list to make Eshkol a first-class GPU-LLM language.

Everything below was found empirically against **v1.2.3-scale** built with CUDA
(`-DESHKOL_GPU_ENABLED=ON`, sm_86, cuBLAS) on old-donkey. Repro snippets are real.

---

## 0. Context — why Eshkol, and what it must do

We need Eshkol to express, and run fast on GPU:
- The **MoE expert FFN** (gate/up/down GEMM + SwiGLU), batched over a verify-chunk.
- **MLA attention** (latent KV, RoPE/YaRN), batched multi-row.
- **Geometric inference**: manifold/Poincaré relevance, curvature-biased routing,
  and **learnable** routing priors (this is where Eshkol's native autodiff and
  vector calculus are uniquely valuable — keep and deepen this).
- **Quantized** weights: Kimi experts are Q4_0 (97% of the model), attention/shared
  experts Q8_0, router/norms F32.

Eshkol already nails the geometry: the KV-landmark relevance kernel
(`computer_mesh/ops/kimi/eshkol/kv_landmark_relevance.esk`) runs correctly on the
live bridge. The blockers are all in **numeric precision, GPU codegen, quantization,
and toolchain**.

---

## 1. CRITICAL — precision & dtypes (the #1 blocker)

**Problem:** Eshkol tensors are **f64 (double) only**. The GPU `matmul` therefore
runs **f64-softfloat**, which is ~10× too slow on consumer GPUs (no native f64
tensor cores). Measured: a single MoE expert FFN (M=64, K=7168, N=2048) = **3.90 ms**
at only **30% GPU util**; native fp16 cuBLAS on the same GPU does the equivalent in
~0.2 ms/GEMM. This caps us at ~34 tok/s when ~300 tok/s is on the table.

**Required — a real dtype system:**
- Add tensor element types: **`f16`, `bf16`, `f32`, `f64`, `i8`, and quantized
  `q4_0`/`q8_0`** (see §3).
- Tensor constructors/casts: `(make-tensor dims val :dtype 'f16)`,
  `(tensor-cast t 'f16)`, `f32`/`f16` literals or a `(f32-tensor ...)` form.
- `matmul` / `tensor-*` must dispatch on dtype and use the **native** path:
  f16/bf16 → cuBLAS GemmEx (tensor cores), f32 → SGEMM, only f64 → softfloat.
- Mixed-precision accumulate (f16 inputs, f32 accum) — the standard LLM combo.

**Repro:** `make-tensor (list 1024 1024) 1.0` → matmul → 30% util, slow. There is
no documented way to get an f32/f16 tensor.

---

## 2. CRITICAL — GPU in the compiled/AOT path, and explicit GPU ops

**Problem A:** `gpu-matmul`, `gpu-elementwise`, `gpu-softmax`, `gpu-reduce`,
`gpu-transpose` are documented in `docs/ESHKOL_QUICK_REFERENCE.md` but are **"Unknown
function"** in the LLVM-compiled path *and* the VM path on v1.2.3. They appear to
exist only as internal `vm_gpu_dispatch` ops, not as callable builtins.

**Problem B:** The **compiled binary** (`eshkol-run file.esk` → `a.out`) runs
`matmul` on **CPU BLAS** even in the CUDA build (peak GPU ~11%). Only **VM/JIT
mode** (`eshkol-run -r`) actually dispatches to the GPU (29–77% util). So AOT-compiled
Eshkol can't use the GPU at all today.

**Required:**
- Expose `gpu-matmul`/`gpu-elementwise`/`gpu-softmax`/`gpu-reduce`/`gpu-transpose`
  as real builtins in **both** the LLVM-compiled and VM paths.
- Make GPU dispatch work in **AOT-compiled** code (the cost-model dispatch must be
  available to compiled binaries, not VM-only), since production inference compiles.
- Add `cublasGemmStridedBatched`-backed **batched GEMM** (`matmul-batched` /
  `(matmul A B :batch n)`) — MoE batches 8 experts × N rows; one launch beats 1440.

**Repro:** `(gpu-matmul A B)` → `error: Unknown function: gpu-matmul` (both `-r`
and compiled). `(matmul (make-tensor '(1024 1024) 1.0) ...)` compiled → CPU.

---

## 3. CRITICAL — quantized tensors (Q4_0 / Q8_0) + dequant on GPU

**Why:** Kimi is a GGUF Q4_0 model (experts), Q8_0 (attention/shared/head). 543 GiB,
97% in Q4 expert banks. Inference must consume quantized weights directly.

**Required:**
- A `q4_0` / `q8_0` tensor dtype (GGUF block layout: q4_0 = 18 bytes/32 weights =
  f16 scale + 16 packed nibbles; q8_0 = 34 bytes/32 = f16 scale + 32 int8).
- GPU dequant kernels (`q4_0 → f16`) and a **quantized GEMV/GEMM** (dequant-in-kernel
  or dequant-then-cuBLAS), matching the CPU reference math bit-for-token (argmax-equal).
- Load-from-mmap’d GGUF tensor (FFI to a byte buffer + offset) without a full copy.

---

## 4. HIGH — persistent GPU runtime / FFI for per-token inference

**Problem:** Each `eshkol-run -r` reloads `stdlib.bc` (`[REPL] Discovered 384
functions ... Loaded stdlib bitcode`) — fine for batch jobs, fatal for per-token
decode where startup dwarfs the GEMM.

**Required (any of):**
- A **persistent runtime / server mode**: load stdlib + hold the CUDA context once,
  then accept many eval requests (socket or stdin loop) with no re-init.
- A clean **C ABI / FFI** so the C++ MoE worker can call an Eshkol-compiled GPU
  GEMM/FFN as a linked function (`extern "C"`), passing device or host buffers.
- Pinned-host / device-buffer handles surviving across calls (a `gpu-buffer` type),
  so resident expert weights aren't re-uploaded per token.

---

## 5. HIGH — bugs & toolchain robustness found this session

- **`norm` segfaults** on a tensor (SIGSEGV mid-program). Workaround used:
  `(sqrt (tensor-dot v v))`. Fix `norm`/`magnitude`/`normalize` for tensors.
- **CUDA + modern glibc:** building with CUDA 13.1 on glibc 2.43 fails — CUDA's
  `crt/math_functions.h` declares `rsqrt`/`rsqrtf` **without** the `noexcept` that
  glibc 2.43 now requires, at **two** sites (lines ~629/653 *and* the `__func__(...)`
  macro decls at ~6046/6072). We patched the CUDA header, but **Eshkol's CMake should
  detect this and apply `-D` shims / a compat header** so `-DESHKOL_GPU_ENABLED=ON`
  builds out-of-the-box on current distros. Also: default `CMAKE_CUDA_HOST_COMPILER`
  to a gcc ≤14 when the system gcc is 15/16 (nvcc 13.x rejects gcc-15).
- **Stale build dir:** reusing `build-cuda` across a version checkout produced
  dangling LLVM symbol link errors (`undefined reference to llvm::BasicBlock::...`)
  in the **runtime link of compiled programs**. The runtime/support lib appears to
  leak LLVM codegen symbols — **separate `libeshkol-runtime` (no LLVM) from the
  compiler lib** so compiled `a.out`s never need LLVM at link.
- **Version drift:** old-donkey shipped v1.2.1 while we needed v1.2.3 (`gpu-matmul`,
  etc.). Tag releases and make `--version` + a `eshkol features` introspection
  command authoritative so deploys can assert capability.

---

## 6. MEDIUM — keep/deepen what makes Eshkol special (autodiff on GPU)

The geometric/learnable layer is Eshkol's edge — don't lose it when adding GPU:
- **Autodiff (`gradient`, `jacobian`, ∇/∇·/∇×/∇²) over GPU/f16 tensors**, so we can
  *learn* the ORC/curvature routing priors and a self-speculative draft head from
  Kimi verifier traces, on-device.
- `tensor-apply` and elementwise activations (`silu`/`gelu`/`sigmoid`/`softmax`) as
  fused GPU kernels (we hand-rolled SwiGLU via `tensor-apply` + `tensor-mul`).
- Poincaré/hyperbolic ops as primitives (we implemented `poincare` distance from
  `tensor-dot`/`acosh`; making it native + differentiable enables geometric KV
  retrieval and curvature-aware routing as compiler-level constructs).

---

## 7. Priority order

1. **§1 dtypes (f16/bf16/f32)** + **§2 native GPU GEMM (incl. batched) in compiled path** — unblocks the whole ultra-perf engine (~10× → ~300 tok/s).
2. **§3 quantized (q4_0/q8_0) tensors + GPU dequant GEMM** — lets Eshkol consume Kimi weights directly.
3. **§4 persistent runtime / FFI** — makes per-token decode viable.
4. **§5 toolchain fixes** — out-of-the-box CUDA build on modern distros; `norm` fix; runtime/compiler lib split.
5. **§6 autodiff-on-GPU + geometric primitives** — the differentiator; powers learnable routing/draft.

## 8. How to verify (real artifacts)

- `computer_mesh/ops/kimi/eshkol/kv_landmark_relevance.esk`, `kv_landmark_select.esk`
  — geometric kernels (work today, CPU).
- `computer_mesh/ops/kimi/eshkol/moe_expert_ffn.esk` — the MoE FFN; today 3.90 ms/expert
  (f64 softfloat). **Target after §1–§2: <0.5 ms/expert (fp16 cuBLAS).**
- Bench harness pattern: `benchmarks/gpu_vs_cpu_bench.esk` (`time-it`, `random-tensor`).
- Success metric: `moe_expert_ffn.esk` in **fp16** drives a 60-layer, 8-expert
  chunk-verify at **≥150 lossless tok/s** on an RTX 3050, AOT-compiled.
