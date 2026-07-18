# GPU Acceleration in Eshkol

## 1. Architecture Overview

Eshkol employs a multi-tier dispatch hierarchy for matrix and tensor operations,
selecting the optimal compute backend at runtime based on problem size, hardware
availability, and a calibrated cost model:

```
Scalar  -->  SIMD  -->  cBLAS  -->  GPU
(tiny)      (small)   (medium)   (massive)
```

The dispatch entry point is `eshkol_matmul_f64` (`blas_backend.cpp`):

- **Tiny** (output elements <= 16): Scalar. SIMD/BLAS overhead exceeds compute.
- **Small to massive** (17 to 1B elements): cBLAS via Apple Accelerate (AMX) or
  OpenBLAS. Accelerate sustains 1100+ GFLOPS on M-series. (`blas_backend.cpp`)
- **Super-massive** (>= 1B elements, ~31600x31600+): GPU vs BLAS cost model
  comparison. GPU lazily initialized on first encounter. (`blas_backend.cpp`)

Backend selection cascades via `[[fallthrough]]` (`blas_backend.cpp`):
if the chosen backend fails, execution falls through to the next-best option.

| File | Lines | Role |
|------|-------|------|
| `lib/backend/blas_backend.cpp` | 891 | Cost model, dispatch, SIMD/BLAS |
| `lib/backend/gpu/gpu_memory.mm` | ~4000+ | Metal compute pipeline (Obj-C++) |
| `lib/backend/gpu/metal_softfloat.h` | ~800+ | SF64 IEEE 754 f64 emulation (MSL) |
| `lib/backend/gpu/gpu_memory_cuda.cpp` | 760 | CUDA backend (cuBLAS + kernels) |
| `lib/backend/gpu/gpu_cuda_kernels.cu` | 409 | CUDA kernel implementations |
| `lib/backend/gpu/gpu_memory_stub.cpp` | 341 | No-op stub for platforms without GPU |

Build-time flags select the backend: `ESHKOL_GPU_METAL_ENABLED` (macOS),
`ESHKOL_GPU_CUDA_ENABLED` (Linux/Windows), or the stub (no GPU). The stub logs
actionable errors via `eshkol_error()` (`gpu_memory_stub.cpp`).

---

## 2. Cost Model

### Estimation Formula

Defined in `estimate_matmul_cost` (`blas_backend.cpp`):

```
flops = 2 * M * K * N
efficiency_b = min(1.0, elements / efficiency_scale_b)
compute_b    = flops / (peak_gflops_b * efficiency_b * 1e9) * 1e9
cost_b       = overhead_b + compute_b
```

The efficiency term models ramp-up: small matrices cannot saturate the backend,
so throughput scales linearly until `efficiency_scale` elements are reached.

### Calibrated Parameters (`blas_backend.cpp`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `blas_overhead_ns` | 5,000 (5 us) | Fixed cBLAS dispatch overhead |
| `blas_peak_gflops` | 1,100 | Measured via Apple Accelerate AMX |
| `blas_efficiency_scale` | 10,000 | Elements for full BLAS saturation |
| `gpu_overhead_ns` | 200,000 (200 us) | Metal command buffer + data transfer |
| `gpu_peak_gflops` | 200 | Measured SF64 softfloat throughput |
| `gpu_efficiency_scale` | 100,000,000 | 100M+ elements to saturate GPU |
| `simd_overhead_ns` | 100 | Minimal SIMD dispatch overhead |
| `simd_peak_gflops` | 25 | Peak NEON/AVX throughput |
| `simd_efficiency_scale` | 1,000 | SIMD saturates quickly |

### Backend Selection (`blas_backend.cpp`)

`select_best_backend` compares estimated costs and selects the minimum. Given
the above parameters, the GPU path is only selected when output elements exceed
~1 billion (`g_gpu_matmul_threshold`, `blas_backend.cpp`). The SIMD-to-BLAS
threshold is 64 elements (`blas_backend.cpp`) -- cBLAS overhead (~5us) is
negligible versus its 100x advantage over naive SIMD.

---

## 3. Metal Backend

### Initialization (`gpu_memory.mm:736`)

1. Create `MTLDevice` via `MTLCreateSystemDefaultDevice()`
2. Create `MTLCommandQueue`; detect unified memory via `MTLGPUFamilyApple1`
3. Identify GPU family (7=M1, 8=M2, 9=M3, 10=M4+)
4. Populate `HardwareProfile` (`gpu_memory.mm:105`): `max_tg_mem`,
   `max_threads_per_tg` (1024), `thread_exec_width` (32), `device_mem`
5. Compute all kernel configs via occupancy-aware scoring
6. Compile Metal shaders with configuration `#define`s; fast-math disabled
   (`options.fastMathEnabled = NO`, `gpu_memory.mm:901`)
7. Create compute pipeline states for all kernels

The shader source is auto-generated from `metal_softfloat.h` into
`metal_sf64_embedded.inc` at build time. Configuration defines (tile sizes,
thread counts) are prepended via `build_shader_config_string()` (`gpu_memory.mm:657`).

### Kernel Configuration System (`gpu_memory.mm:88-131`)

Seven kernel types are supported: `KERNEL_SF64`, `KERNEL_DF64`,
`KERNEL_F32_SIMD`, `KERNEL_F32_SIMD_128`, `KERNEL_FP24`, `KERNEL_FP53`,
`KERNEL_OZAKI`. Each has a dedicated parameter search that enumerates candidate
tile sizes and evaluates them via `score_config()` (`gpu_memory.mm:400`):

```
score = occupancy * arithmetic_intensity * sqrt(BM * BN) * BK
```

Balancing latency hiding, compute density, tile coverage, and K-block
amortization. Shared memory cost is computed per-kernel-type to ensure
configurations fit within `maxThreadgroupMemoryLength`.

### Adaptive Row-Blocking (`gpu_memory.mm:586`)

Large matrices are partitioned into M-dimension chunks. Each kernel type has a
weight factor (SF64=8.0, FP53=4.0, Ozaki-II=3.0, FP24=2.0, DF64=1.5,
default=1.0), with chunks aligned to tile boundaries, capped at 4096 rows,
targeting ~200ms GPU time per chunk.

### Ozaki-II CRT DGEMM (`gpu_memory.mm:136-332`)

Exact f64 GEMM via Chinese Remainder Theorem. 49 pairwise coprime moduli
(`gpu_memory.mm:138`) decompose f64 elements into residues, perform f32 GEMM per
residue pair, then reconstruct via CRT. Constants are precomputed using
`__int128` for exact arithmetic (`gpu_memory.mm:188`). Adaptive N selection
(`gpu_memory.mm:260`) minimizes moduli count based on actual input magnitudes.

The default exact tier (`ESHKOL_SF64_KERNEL=ozaki`, fixed N=16 moduli) does the
per-modulus residue split and the double-double CRT reconstruction on the CPU —
`O(num_moduli · N^2)` host passes that download every per-modulus `W_l` plane and
dominate wall time.

### Ozaki-II reduced-precision FAST tier (opt-in, fully-GPU)

`ESHKOL_SF64_KERNEL=ozaki-fast` (or `ESHKOL_OZAKI_FAST=1`) selects a
reduced-precision LINEAR-CRT tier built on **near-peak MPS f32 GEMMs**, with the
residue split and CRT reconstruction fully on the GPU. It is a port of a measured
standalone prototype. The **default path is unchanged** — the tier is opt-in.

The pipeline (host does only: upload A,B once; one O(N^2) exponent pass; download C):

1. **Cap-limited prime-power moduli.** `ESHKOL_OZAKI_FAST_MODULI` (default 11,
   clamped to `[2,16]`) is the accuracy knob. The moduli are the largest
   pairwise-coprime prime powers `≤ cap`, where `cap` is chosen from K so that
   `K·(p/2)² < 2²⁴`. Under that bound a **single MPS f32 GEMM of centered residues
   is integer-exact** (the prototype verified this to f64) — so each modulus is one
   MPS GEMM at the GPU's ~20 TF f32 ceiling, no K-tiling. This is the entire
   performance story vs the exact tier's custom modular kernel.
2. **GPU residue split** (`ozaki_fast_split`): reads each operand element as an f64
   bit-pattern, extracts the 53-bit significand, applies the `E`-scaling as an
   exact right-shift-with-round (`E≤52` ⇒ shift ≤ 0), reduces `mod p`, centers to
   `[-p/2, p/2)`. Zero host f64 math.
3. **GPU df32 fractional-CRT reconstruction** (`ozaki_fast_accum` +
   `ozaki_fast_finalize`): `a_l = (G_l·q_l) mod p_l`, `S += a_l/p_l` accumulated in
   df32 (double-single) reduced mod 1 each step, then `Cs/P = frac(S) − round`,
   `C = frac·P·2^(e_i+f_j−2E)`, output as a df32 pair the host widens to f64.

All moduli run in **one command buffer with a single CPU-GPU sync** (per-modulus
`waitUntilCompleted` starves the GPU under CPU load — a prototype finding).

**df32 is the accuracy ceiling of this tier.** Metal has no f64, so reconstruction
is df32 (~48-bit → ~1e-11 floor). Two load-bearing constraints (verified on hardware):

- The library is compiled with `fastMathEnabled = NO` / `MTLMathModeSafe`
  (`gpu_memory.mm`). Fast-math annihilates the TwoSum/TwoProduct compensation and
  collapses the reconstruction to plain f32 (~1e-7). Mandatory.
- Power-of-two descale uses `ldexp` (exact), never `exp2` (a polynomial
  approximation that injects a uniform ~1e-7 error).

#### Accuracy vs moduli (measured, M2 Ultra, worst case over the four gate regimes
integer / fractional / pi-e / wide-magnitude, K up to 4096)

| `ESHKOL_OZAKI_FAST_MODULI` | worst-case rel err | tier          |
|---------------------------|--------------------|---------------|
| 10                        | ~2.2e-7 (FAILS gate on pi/e) | ~1e-8 well-conditioned only |
| **11 (default)**          | **~2.4e-8**        | ~1e-8         |
| 12                        | ~4.0e-8            | ~1e-8         |
| 14                        | ~6e-11 (df32 floor)| ~1e-11        |
| 16 (exact tier, `ozaki`)  | bit-exact          | exact f64     |

The cancellation-heavy pi/e regime is binding: its near-zero-mean products amplify
both the `2^-E` truncation and the df32 reconstruction floor. Accuracy is NOT
monotone in moduli count — beyond ~12 the df32 accumulator grows and the pi/e floor
rises again; ~14 is the practical df32 floor (~1e-11). Well-conditioned data
(fractional, wide-magnitude) reaches ~1e-11..1e-13 at far fewer moduli.

#### Throughput (measured, M2 Ultra, internal GPU pipeline, best-of-5, `ESHKOL_OZAKI_PROFILE=1`)

Effective `2·N^3` GFLOP/s of the full pipeline. Raw `cblas_dgemm` (AMX f64) was
measured in the same session and was NOT depressed (N=8192 = 1099 GF ≈ the clean
1088 peak), so the AMX ratios below are fair, not overstated.

| tier                | N=4096 (GF) | N=8192 (GF) | vs AMX dgemm (N=8192) |
|---------------------|-------------|-------------|-----------------------|
| fast, 10 moduli     | ~1080       | **~1448**   | **1.32x** (fails pi/e gate; max-speed opt-in) |
| **fast, 11 (default)** | ~1081    | **~1384**   | **1.26x** |
| fast, 12 moduli     | ~1022       | ~1278       | 1.16x     |
| exact, 16 moduli    | ~140        | ~143        | 0.13x     |
| `cblas_dgemm` (AMX f64) | ~1080   | ~1099       | 1.00x (reference) |

- **N=8192: the fast tier beats clean AMX f64 dgemm — 1.26x at the default 11
  moduli (~2.4e-8), up to 1.32x at 10 moduli (~1e-8 on well-conditioned data).**
  This matches the prototype's ~1400 GF / ~1.3x-clean-AMX at ~1e-8.
- **N=4096: a tie with AMX (~1.0x).** Smaller N is overhead-bound (MPS f32 is not
  yet at its large-N ceiling and the host readback is a larger fraction) — the
  prototype documented the same (N=4096 does not win; the crossover is N=8192).
- **vs the exact Ozaki tier: 7.7x (N=4096) to ~9.7x (N=8192) faster** — the exact
  tier at N=8192 also falls to its serial path (16-modulus buffers exceed its 4 GB
  batch cap), so the e2e gap is largest there.

For genuine full f64 (~1e-14) AMX remains the right tool: full f64 needs ~19–22
moduli (GEMM count ≈ AMX) and df32 reconstruction cannot reach 1e-14 anyway
(~1e-11 floor). The GPU wins only at reduced precision (≳1e-11) and large N.

Set `ESHKOL_OZAKI_PROFILE=1` to print the per-matmul internal pipeline ms and
effective GFLOP/s for both the exact and fast tiers.

---

## 4. SF64 Software Float64

Apple Silicon GPUs lack native f64 hardware. SF64 emulates full IEEE 754
double-precision using `uint2` (two 32-bit integers) on Metal's integer ALU.

### Representation (`metal_softfloat.h`)

```metal
typedef uint2 sf64;  // .x = high 32 bits, .y = low 32 bits
```

IEEE 754 layout: sign (1 bit, bit 63), exponent (11 bits, bias 1023),
mantissa (52 bits, implicit leading 1). 128-bit intermediates use
`struct sf128 { sf64 hi; sf64 lo; };`.

### Core Operations

**Addition** (`metal_softfloat.h`): 11 guard bits via left-shift. After
magnitude add/subtract, normalize leading 1 to bit 62, extract 10 round bits
(`& 0x3FF`), delegate to `sf64_round_pack()`.

**Multiplication** (`metal_softfloat.h`): 64x64 -> 128 bit multiply via
`mul64x64()` using Metal's `mulhi()` intrinsic for 32-bit high-word extraction.
4 partial products with branchless carry propagation.

**FMA** (`metal_softfloat.h`): True single-rounding using 128-bit
intermediate. Fast path (all operands normal) skips 13 NaN/Inf/Zero/subnormal
checks -- the hot path in matmul accumulation.

**Division** (`metal_softfloat.h`): 63-iteration long division. Remainder
preserved as sticky bit for correct rounding.

### Rounding (`metal_softfloat.h`)

IEEE 754 round-to-nearest-even with 10 rounding bits, halfway at `0x200`:

```metal
bool round_up = (round_bits > 0x200u) ||
                ((round_bits == 0x200u) && ((sig.y & 1u) != 0));
```

The mask `0x3FF` was a critical bug fix -- the earlier `0x7FF` shifted the
halfway point, causing systematic rounding errors.

### Performance

`shr128_jam` (`metal_softfloat.h`) includes a hand-optimized 1-31 bit
fast path (direct 4-word shifts, no nested calls) for FMA exponent alignment.
Each operation introduces at most 0.5 ULP error, matching hardware f64.

---

## 5. CUDA Backend

The CUDA backend (`gpu_memory_cuda.cpp`) provides native f64 GPU acceleration on
NVIDIA hardware -- no software emulation required.

### Initialization (`gpu_memory_cuda.cpp`)

Enumerate devices, select device 0, create CUDA stream and cuBLAS handle, bind
handle to stream. Thread-safe via `std::mutex` + double-check pattern.

### cuBLAS Matmul (`gpu_memory_cuda.cpp`)

cuBLAS is column-major; Eshkol tensors are row-major. The transpose trick:
```cpp
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K, &alpha, B_ptr, N, A_ptr, K, &beta, C_ptr, N);
```
Both `cublasDgemm` (f64) and `cublasSgemm` (f32) are supported.

### INT8 Tensor-Core Ozaki f64 GEMM (opt-in, `gpu_memory_cuda.cpp`)

An opt-in f64 matmul path recovers FP64-accurate `C = A*B` from the INT8 (IMMA)
tensor cores, which run ~500x faster than the deliberately crippled native FP64
pipeline on consumer/prosumer NVIDIA GPUs (GeForce Ampere f64 = 1/64 FP32).
Enable with `ESHKOL_CUDA_F64_KERNEL=ozaki-int8`; the default f64 GPU matmul
stays `cublasDgemm`.

Scheme (Ootomo-Ozaki-Yokota, per-row/col scaled 7-bit integer slicing), for
row-major `A` (MxK), `B` (KxN):

1. Scale into `[-1,1]`: `r_i = max_k|A[i,k]|`, `c_j = max_k|B[k,j]|`.
2. Slice `A/r` and `B/c` into `s = T+1` signed 7-bit int8 slices, base 128
   (`|slice| <= 127`). A-slices are stored in natural layout, B-slices are
   stored **transposed** so each pair GEMM issues as
   `cublasGemmEx(OP_T, OP_N, CUDA_R_8I -> CUDA_R_32I, COMPUTE_32I)` on the fast
   IMMA **TN** layout — mandatory on sm_86 (a 3.7x throughput cliff vs NN);
   Blackwell (sm_120) is layout-indifferent, so TN is safe everywhere.
3. `C[i,j] = r_i c_j * sum_{p+q<=T} 128^{-(p+q+2)} (A_p x B_q)[i,j]`. Each
   `A_p x B_q` is one INT8->INT32 GEMM; the int32 accumulation is **exact** (no
   rounding) — the only error source is dropping slice-pairs with `p+q > T`.
4. **Diagonal-fused reconstruction**: all slice-pairs on a diagonal `d = p+q`
   carry the same weight `128^-(d+2)`, so they accumulate into one int32 buffer
   via `cublasGemmEx` `beta=1` (staying on the tensor path), then a single fused
   FP64 kernel streams the diagonal buffers, applies the weight, and scales by
   `r_i c_j`. Buffers are capped at `floor((2^31-1)/(K*127^2))` pairs so the
   accumulation stays int32-exact at any N; the split/reconstruct kernels are in
   `gpu_cuda_kernels.cu`.

The accuracy/throughput knob is `ESHKOL_OZAKI_CUDA_T` (default 6 = full f64
~1e-15, 28 GEMMs; T=4 ~1e-11, 15 GEMMs, ~2x faster). `#GEMMs = (T+1)(T+2)/2`;
relative error `~2^{-7(T+1)}`. `K` is guarded below 133,000 (`K*127^2 < 2^31`);
out of range it falls back **loudly** to `cublasDgemm` (K-panel splitting is not
implemented). Best throughput needs M, K, N multiples of 4; any `cublasGemmEx`
`NOT_SUPPORTED` (e.g. unaligned dims) falls back quietly to `cublasDgemm`.

Measured (normwise Frobenius error is the accuracy metric — a single near-zero
output entry inflates per-element error meaninglessly):

| GPU | f64-accurate INT8-Ozaki | native `cublasDgemm` | speedup |
|-----|-------------------------|----------------------|---------|
| RTX 3090 (sm_86), N=4096 | 4.74 TFLOP/s-eq @ err 2.7e-15 | 0.54 TFLOP/s | **8.8x** |
| RTX PRO 6000 Blackwell (sm_120), N=16384 | ~30 TFLOP/s @ err <8.4e-15 (fused) | 1.50 TFLOP/s | **20x** |

Per-row/col scaling keeps wide-dynamic-range data accurate (7.5e-15 on 1e-3..1e3
inputs on the 3090). FP8-Ozaki was measured but is 2x slower than INT8 on
Blackwell (identical tensor throughput, fewer exact bits/slice), so INT8 is the
f64-accurate path. Correctness is gated by
`tests/gpu/cuda_ozaki_correctness_gate.sh` (ICC oracle
`cuda-ozaki-int8-correctness`).

### Custom Kernels (`gpu_cuda_kernels.cu`)

All kernels use 256 threads/block with grid-stride loops, grid capped at 65535.

- **Elementwise** (`gpu_cuda_kernels.cu:21`): 15 ops dispatched by int code.
- **Reduce** (`gpu_cuda_kernels.cu:76`): Two-pass -- block reduction with
  `__shfl_down_sync` warp shuffle + shared memory, then single-block finalize.
- **Transpose** (`gpu_cuda_kernels.cu:186`): Tiled with `TILE_DIM=32`,
  `BLOCK_ROWS=8`, +1 shared memory padding to avoid bank conflicts.
- **Softmax** (`gpu_cuda_kernels.cu:319`): Three-pass (max, exp+sum, normalize).
- **Normalize** (`gpu_cuda_kernels.cu:364`): Layer normalization (mean, variance,
  scale+shift).

---

## 6. Supported Operations

| Operation | Metal Kernel | CUDA Kernel | Complexity |
|-----------|-------------|-------------|------------|
| Matmul (f64) | `matmul_sf64`, `matmul_sf64_v2` | `cublasDgemm` | O(MKN) |
| Matmul (df64) | `matmul_df64`, `matmul_df64_pure` | -- | O(MKN) |
| Matmul (f32) | `matmul_f32_simd`, `matmul_f32_simd_128` | `cublasSgemm` | O(MKN) |
| Matmul (fp24/fp53) | `matmul_fp24`, `matmul_fp53` | -- | O(MKN) |
| Matmul (Ozaki-II) | `matmul_ozaki_gemm` | INT8-Ozaki `cublasGemmEx` (opt-in) | O(T^2*MKN) |
| Elementwise | `elementwise_sf64` | `elementwise_f64_kernel` | O(N) |
| Reduce (full/axis) | `reduce_sf64`, `reduce_axis_sf64` | `reduce_f64_kernel` | O(N) |
| Softmax | `softmax_sf64` | `softmax_f64_kernel` | O(N) |
| Transpose | `transpose_sf64` | `transpose_f64_kernel` | O(N) |
| Normalize | `normalize_sf64` | `normalize_f64_kernel` | O(N) |
| Convert f64<->f32/df64 | `convert_*` pipelines | -- | O(N) |

**Precision tiers** (Metal, via `ESHKOL_GPU_PRECISION`): exact=SF64 (52-bit
mantissa), high=DF64 (~100-bit), fast=f32 (23-bit). Specialized: FP24
(truncated mantissa + int64 accum), FP53 (uint2 + 128-bit accum), Ozaki-II
(exact via CRT).

---

## 7. Memory Management

### Metal Unified Memory

Apple Silicon uses `MTLResourceStorageModeShared` (`gpu_memory.mm:636`) --
unified CPU/GPU memory, no explicit transfers. Host pointers are wrapped via
`newBufferWithBytesNoCopy:` for zero-copy access.

A size-binned buffer pool (`gpu_memory.mm:611-653`) reuses `MTLBuffer` objects.
Sizes are rounded to the next power of 2; each bucket holds up to 8 buffers
(`POOL_MAX_PER_BUCKET`). Excess buffers are dropped (freed by ARC).

### CUDA Memory Types (`gpu_memory_cuda.cpp`)

- `ESHKOL_MEM_UNIFIED`: `cudaMallocManaged()` -- automatic page migration
- `ESHKOL_MEM_DEVICE`: `cudaMalloc()` -- device-only, maximum throughput
- `ESHKOL_MEM_HOST_PINNED`: `cudaMallocHost()` -- pinned, async DMA transfers

Host wrapping uses `cudaHostRegister()` with fallback to unified alloc + memcpy.

### Buffer Handle

```c
typedef struct {
    void*  host_ptr;    void*  device_ptr;    size_t size_bytes;
    EshkolMemoryType mem_type;  EshkolGPUBackend backend;
    void*  backend_data;        uint32_t flags;  // bit 0: wrapped
} EshkolGPUBuffer;
```

API: `eshkol_gpu_alloc()`, `eshkol_gpu_alloc_aligned()`, `eshkol_gpu_free()`,
`eshkol_gpu_wrap_host()`, `eshkol_gpu_sync()`, `eshkol_gpu_wait()`. The `flags`
bit 0 prevents deallocation of externally-owned memory.

---

## 8. Configuration

### Environment Variables

**Dispatch control:**

| Variable | Default | Source |
|----------|---------|--------|
| `ESHKOL_GPU_MATMUL_THRESHOLD` | 1,000,000,000 | `blas_backend.cpp` |
| `ESHKOL_BLAS_THRESHOLD` | 64 | `blas_backend.cpp` |
| `ESHKOL_FORCE_GPU=1` | -- | Force GPU for all ops |
| `ESHKOL_FORCE_CPU=1` | -- | Disable GPU |
| `ESHKOL_GPU_PRECISION` | "exact" | "exact"/"high"/"fast" (`blas_backend.cpp`) |

**Cost model tuning:**

| Variable | Default | Source |
|----------|---------|--------|
| `ESHKOL_BLAS_PEAK_GFLOPS` | 1100 | `blas_backend.cpp` |
| `ESHKOL_GPU_PEAK_GFLOPS` | 200 | `blas_backend.cpp` |

**CUDA INT8-Ozaki f64 GEMM (opt-in):**

| Variable | Values | Meaning |
|----------|--------|---------|
| `ESHKOL_CUDA_F64_KERNEL` | `ozaki-int8` | Select the INT8 tensor-core Ozaki f64 matmul (default: `cublasDgemm`) |
| `ESHKOL_OZAKI_CUDA_T` | 1..8 (default 6) | Accuracy/throughput knob: T=6 full f64 (~1e-15), T=4 ~1e-11 (~2x faster) |

**Metal kernel tuning:**

| Variable | Values | Variable | Values |
|----------|--------|----------|--------|
| `ESHKOL_SF64_TG` | 8,16,32 | `ESHKOL_SF64_TILE_K` | 4,8,16,32 |
| `ESHKOL_DF64_TG` | 8,16 | `ESHKOL_DF64_BK` | 8,16,32 |
| `ESHKOL_FP_BK` | 8,16,32 | `ESHKOL_FP53_BK` | 8,16 |
| `ESHKOL_F32S_WM` | 1,2,4 | `ESHKOL_F32S_WN` | 1,2,4 |
| `ESHKOL_F32S_BK` | 8,16,32 | `ESHKOL_F32S128_WM` | 1,2,4 |
| `ESHKOL_F32S128_WN` | 1,2,4 | `ESHKOL_F32S128_BK` | 8,16,32 |

**Ozaki-II exact/fast DGEMM (Metal, tier 0):**

| Variable | Values | Effect |
|----------|--------|--------|
| `ESHKOL_SF64_KERNEL` | `ozaki` / `ozaki-fast` / `fp53` / `legacy` | selects the exact CRT tier, the opt-in reduced-precision fast tier (MPS f32 GEMMs), or the fp53/legacy exact kernels |
| `ESHKOL_OZAKI_FAST` | `1` | enables the fast tier (equivalent to `ESHKOL_SF64_KERNEL=ozaki-fast`) |
| `ESHKOL_OZAKI_FAST_MODULI` | 2–16 (default 11) | fast-tier accuracy knob; out-of-range clamps loudly to `[2,16]` |
| `ESHKOL_OZAKI_NUM_MODULI` | 2–16 (default 16) | exact-tier moduli count; out-of-range clamps loudly |
| `ESHKOL_OZAKI_ADAPTIVE` | `1` | exact tier: adaptive (approximate) moduli minimisation |
| `ESHKOL_OZAKI_PROFILE` | `1` | prints per-matmul internal pipeline ms + effective GFLOP/s (exact and fast tiers) |

#### Exact-reference certification gate (opt-in)

`tests/gpu/ozaki_certification_gate.sh` runs a deterministic `512x512` integer-valued f64 witness. It uses an independent sampled native-`i128` dot-product oracle with a proven bound below `2^58`, then converts oracle sums to correctly rounded f64. On the identical witness, Apple Accelerate must differ on at least one sample while fixed-`N16` Metal exact Ozaki (`ozaki`) must match every sampled reference. The script exercises both default cached-JIT and AOT and requires real fixed-`N16` Metal dispatch markers, explicitly rejecting CPU fallback. ICC oracle name is `ozaki-certification`. This is a correctness-only certification witness: no default-dispatch change and no performance claim.

The default DGEMM tier is unchanged; both `ozaki` and `ozaki-fast` are opt-in.
See section 3 (Ozaki-II CRT DGEMM) for the accuracy/throughput tables.

**Diagnostic:** `ESHKOL_GPU_VERBOSE` enables dispatch logging to stderr.

### Tuning Guide

Force GPU for benchmarking:
```bash
ESHKOL_GPU_MATMUL_THRESHOLD=1 ESHKOL_FORCE_GPU=1 ./program
```

Calibrate cost model by measuring peak cBLAS throughput on a large matmul (e.g.
4096x4096) and setting `ESHKOL_BLAS_PEAK_GFLOPS` to the observed value.

Select precision for workload:
```bash
ESHKOL_GPU_PRECISION=fast ./inference    # ML: f32 suffices
ESHKOL_GPU_PRECISION=exact ./simulation  # Science: exact f64
```

Override threadgroup dimensions if auto-tuned occupancy is suboptimal:
```bash
ESHKOL_SF64_TG=16 ESHKOL_SF64_TILE_K=8 ./program
```

The Metal backend logs all kernel configurations at init (to stderr) including
occupancy scores, making it straightforward to assess auto-tuned optimality.

### SIMD Implementation Details

The SIMD tier provides architecture-specific matmul as a middle ground between
scalar and cBLAS:

- **ARM NEON** (`blas_backend.cpp`): Cache-blocked with 64x64x64 tiles
  tuned for M-series L1 (64KB data). 4x4 register micro-kernel with
  `vfmaq_f64` FMA and 8 `float64x2_t` accumulators.

- **x86 AVX** (`blas_backend.cpp`): 64x64x64 tiles for x86 L1 (~32KB).
  4x8 micro-kernel with `_mm256_fmadd_pd` (FMA3) or separate mul+add, 8
  `__m256d` accumulators.

- **x86 SSE2** (`blas_backend.cpp`): Fallback for older x86. 4x4
  micro-kernel with `__m128d`, 64x64x64 blocking.

Edge tiles that do not fill a complete micro-kernel fall back to scalar
computation via `scalar_block` / `avx_scalar_block` / `sse_scalar_block`.
All SIMD implementations zero the result matrix with `memset` before blocked
accumulation to support the `C[i][j] += sum` pattern (`blas_backend.cpp`).

---

## 9. SF64 Double-Double Arithmetic -- Algorithmic Deep Dive

Apple Silicon GPUs have no native f64 ALU. The SF64 subsystem
(`lib/backend/gpu/metal_softfloat.h`) emulates full IEEE 754 double-precision
arithmetic using Metal's 32-bit integer pipeline, based on Berkeley SoftFloat
algorithms.

### The `uint2` Representation

A `double` is 64 bits. Metal's Shading Language provides `uint2` (two 32-bit
unsigned integers), which SF64 uses as its storage type
(`metal_softfloat.h`):

```metal
typedef uint2 sf64;   // .x = high 32 bits, .y = low 32 bits
```

The IEEE 754 layout maps as follows:

| Field     | Bits     | Location in `sf64.x`                   |
|-----------|----------|----------------------------------------|
| Sign      | 1 bit    | Bit 31 of `.x`                         |
| Exponent  | 11 bits  | Bits 30-20 of `.x` (bias 1023)         |
| Mantissa high | 20 bits | Bits 19-0 of `.x`                  |
| Mantissa low  | 32 bits | All of `.y`                         |

Constants `SF64_SIGN_MASK` (0x80000000), `SF64_EXP_MASK` (0x7FF00000), and
`SF64_MANT_HI_MASK` (0x000FFFFF) extract each field
(`metal_softfloat.h`). Classification functions (`sf64_is_zero`,
`sf64_is_inf`, `sf64_is_nan`, `sf64_is_subnormal`, `sf64_is_signaling_nan`)
operate purely on these bitfields (`metal_softfloat.h`).

On little-endian ARM64, a `double` in memory stores `[low32, high32]`, but
`uint2.x` is the first word. The `native_to_sf64` / `sf64_to_native` helpers
(`metal_softfloat.h`) perform the swap inline in every kernel,
eliminating the need for a separate word-swap dispatch pass.

### 128-Bit Intermediates

For multiply and FMA, SF64 needs more than 64 bits of intermediate precision.
The `sf128` struct (`metal_softfloat.h`) holds two `sf64` values as
high and low halves:

```metal
struct sf128 { sf64 hi; sf64 lo; };   // 128-bit value
```

The `uint128_t` struct (`metal_softfloat.h`) provides a 4-word
representation (`w3..w0`, MSW first) used for multiplication output. These
are not hardware-supported -- all 128-bit operations are synthesized from
32-bit word arithmetic with explicit carry/borrow propagation.

### Core Operations

**Addition** (`sf64_add`, `metal_softfloat.h`): The algorithm uses 11
guard bits by left-shifting both significands by 11, placing the implicit
leading 1 at bit 63. After aligning exponents via `shr64_jam` (which
preserves rounding information as a sticky bit), it adds or subtracts
magnitudes. On same-sign overflow (carry out of bit 63), it shifts right by
2 and sets bit 62. On subtraction, it normalizes via `clz64` to place the
leading 1 at bit 62. Finally, 10 round bits are extracted from the low 10
bits (`& 0x3FF`), the significand is shifted right by 10, and `sf64_round_pack`
produces the final result (`metal_softfloat.h`).

**Multiplication** (`sf64_mul`, `metal_softfloat.h`): Both significands are
shifted to have the leading 1 at bit 63. The 64x64-bit multiply is performed
by `mul64x64` (`metal_softfloat.h`), which decomposes into 4 partial
products using Metal's `mulhi()` intrinsic for the high 32-bit word of each
32x32 product. Carries are propagated branchlessly via `select(0u, 1u, ...)`.
The result is a 128-bit product in `uint128_t{w3,w2,w1,w0}`. The product's
leading 1 is at bit 126 or 127; if bit 127 (overflow), the high word is
shifted right by 1 and the exponent incremented. Bits w1|w0 contribute a
sticky bit for correct rounding.

**Fused Multiply-Add** (`sf64_fma`, `metal_softfloat.h`): This is the
critical inner-loop operation for matmul (C[i][j] += A[i][k] * B[k][j]).
It computes a*b+c with a single rounding at the end, matching IEEE 754
semantics. The implementation has two paths:

1. **Fast path** (lines 707-747): When all three operands are normal
   (exponent neither 0 nor 0x7FF), it extracts components directly without
   any NaN/Inf/Zero/subnormal checks. This saves 13 branch evaluations per
   FMA -- the hot path in matmul accumulation.

2. **Slow path** (lines 716-747): Handles all special cases: NaN propagation
   via `sf64_propagate_nan3`, infinity arithmetic (inf*0=NaN, inf+(-inf)=NaN),
   zero operands, and subnormal normalization via `clz64`-based shifting.

The product a*b is computed as a full 128-bit value. The addend c is widened
to 128 bits by shifting its 53-bit significand to bit 126
(`shl64(sigC, 10)` into the high half). After aligning exponents via
`shr128_jam`, the 128-bit addition or subtraction is performed with
4-word carry propagation (`add128`/`sub128`, `metal_softfloat.h`).
On subtraction with cancellation, `clz128` finds the new leading 1 and
`shl128` normalizes. The sticky bit is computed from the low 64 bits
(`R.lo.x | R.lo.y`), 10 round bits are extracted from `R.hi.y`, and
`sf64_round_pack` produces the single-rounded result.

**Division** (`sf64_div`, `metal_softfloat.h`): Uses 63-iteration long
division. The dividend significand is placed at bit 62 (aligned); if the
dividend is less than the divisor, it is shifted left by 1 and the exponent
decremented. Each iteration compares the remainder against the divisor,
subtracts if larger, and sets the corresponding quotient bit. After the loop,
the remainder is preserved as a sticky bit for correct rounding. This gives
exact results but is the most expensive SF64 operation at 63 serial iterations.

### Rounding: Round-to-Nearest-Even

`sf64_round_pack` (`metal_softfloat.h`) implements IEEE 754
round-to-nearest-even using 10 rounding bits. The halfway point is at
`0x200` (bit 9):

```metal
bool round_up = (round_bits > 0x200u) ||
                ((round_bits == 0x200u) && ((sig.y & 1u) != 0));
```

The second clause handles the tie-breaking rule: when rounding bits are
exactly 0x200 (halfway), the result rounds toward even (LSB of significand).
If rounding causes the mantissa to overflow (bit 52 carries into bit 53),
the significand is shifted right and the exponent incremented. Overflow to
infinity and underflow to subnormal/zero are handled as edge cases.

The `0x3FF` mask was a critical bug fix (documented in `metal_softfloat.h`
and the project memory). The earlier `0x7FF` (11 bits) shifted the halfway
point, causing systematic rounding errors in matmul accumulation.

### The `shr128_jam` Fast Path

The `shr128_jam` function (`metal_softfloat.h`) is the hot path in FMA
exponent alignment. For shifts of 1-31 bits (the common case when aligning
operands with similar exponents), it uses a hand-optimized 4-word direct
shift (`metal_softfloat.h`):

```metal
if (n < 32) {
    uint w0 = (x.lo.y >> un) | (x.lo.x << inv);
    uint w1 = (x.lo.x >> un) | (x.hi.y << inv);
    uint w2 = (x.hi.y >> un) | (x.hi.x << inv);
    uint w3 = x.hi.x >> un;
    return sf128{sf64(w3, w2), sf64(w1, w0 | sticky)};
}
```

This avoids the overhead of nested function calls to `shr64_jam` and
`shr128`, reducing register pressure on Metal's GPU SIMD units. The sticky
bit preserves information about shifted-out bits for correct rounding.

### Transcendental Functions

Metal has no native f64 math functions. SF64 implements them using
FMA-chain polynomial approximations:

- **`sf64_exp`** (`metal_softfloat.h`): Cody-Waite range reduction
  (x = n*ln2 + r, |r| < ln2/2), then 13-term Horner evaluation using
  `sf64_fma` for each coefficient (1/n! for n=2..13). Final scaling by
  2^n via direct exponent adjustment.

- **`sf64_log`** (`metal_softfloat.h`): Extracts exponent and mantissa,
  reduces to [sqrt(2)/2, sqrt(2)] range, computes s=(f-1)/(f+1), then
  evaluates an odd-power series (2/(2k+1) for k=1..6) via Horner form.
  Result: n*ln2 + log(f).

- **`sf64_sin` / `sf64_cos`** (`metal_softfloat.h`): Cody-Waite
  reduction to [-pi/4, pi/4] via `sf64_trig_reduce`, then minimax polynomials
  (`sf64_sin_poly`: 5 odd terms, `sf64_cos_poly`: 5 even terms), with
  quadrant-based sign/function selection.

- **`sf64_sqrt`** (`metal_softfloat.h`): Newton-Raphson with 5 iterations.
  Initial estimate halves the exponent. Each iteration: y = (y + x/y) * 0.5.

- **`sf64_tanh`** (`metal_softfloat.h`): Computed as (e^(2x)-1)/(e^(2x)+1)
  with saturation to +/-1 for |x| >= 20.

All transcendentals use `sf64_fma` in their polynomial evaluations, maintaining
full 52-bit precision throughout.

### Precision Characteristics

SF64 provides **bit-exact IEEE 754 double-precision** (52-bit mantissa).
Each individual operation introduces at most 0.5 ULP error, matching
hardware f64. The 128-bit intermediates in FMA ensure that the product a*b
is computed exactly before the addition of c, so the only rounding occurs at
the final pack step.

The DF64 (double-float) alternative uses `float2` on Metal's native f32
FMA units, achieving approximately 100 bits of effective mantissa via
Knuth/Dekker splitting. SF64 and DF64 serve different precision tiers:
SF64 is exact but slower (integer ALU), DF64 is approximate but faster
(native f32 FMA).

---

## 10. Metal Compute Pipeline

### Shader Compilation and Caching

Metal shaders are compiled at runtime from source, not from precompiled
`.metallib` files. The shader source originates from `metal_softfloat.h`,
which CMake auto-generates into `metal_sf64_embedded.inc` at build time.
This `.inc` file is `#include`d as an NSString constant
(`g_matmul_sf64_source`) in `gpu_memory.mm:733`.

At initialization (`metal_init`, `gpu_memory.mm:736`), the full shader source
is assembled by prepending kernel-specific `#define` directives to the
embedded source. `build_shader_config_string()` (`gpu_memory.mm:657`)
generates all configuration defines (tile sizes, thread counts, BK values)
for every kernel type:

```objc
NSString* config = build_shader_config_string();
NSString* fullSource = [config stringByAppendingString:g_matmul_sf64_source];
```

Compilation uses `MTLCompileOptions` with `fastMathEnabled = NO`
(`gpu_memory.mm:901-902`) to ensure IEEE 754 compliance -- fast-math
optimizations would break the bit-exact rounding behavior that SF64 relies on.

The compiled library (`MTLLibrary`) contains all kernel functions. Individual
pipelines are created by looking up each kernel by name and calling
`newComputePipelineStateWithFunction:` (`gpu_memory.mm:924`). Pipeline
objects (`id<MTLComputePipelineState>`) are cached as global statics
(`gpu_memory.mm:61-81`) for the lifetime of the process.

### Hardware-Adaptive Recompilation

After creating each pipeline, `metal_init` queries the actual
`maxTotalThreadsPerThreadgroup` from the pipeline state
(`gpu_memory.mm:936-939`). If the hardware limit is lower than the
occupancy-tuned thread count (due to register pressure from complex
kernels), the system re-searches the configuration space with the reduced
constraint:

```cpp
HardwareProfile hw_reduced = g_hw;
hw_reduced.max_threads_per_tg = (uint32_t)maxTG;
search_fp_config(KERNEL_FP24, g_cfg_fp24, hw_reduced);
```

It then rebuilds the full shader config string and recompiles the entire
library with updated defines (`gpu_memory.mm:1082-1093`). This ensures that
tile sizes, thread counts, and shared memory usage are all consistent with
what the hardware actually supports. The fp24, fp53, df64_pure, and
f32_simd_128 kernels all have this adaptive recompilation path.

### Thread Organization and Dispatch

Each kernel type uses a specific threadgroup geometry:

- **SF64 / DF64**: 2D threadgroups of `TG x TG` threads (e.g. 8x8 = 64 or
  16x16 = 256). Grid dimensions: `ceil(N/BN) x ceil(M/BM)` threadgroups.
  Each thread computes a `TT x TT` sub-tile of the output matrix
  (`gpu_memory.mm:1586-1587`).

- **F32 SIMD**: 1D threadgroups of `WM * WN * 32` threads (multiples of
  warp size 32). For the small-tile kernel (`matmul_f32_simd_pure`), typical
  configuration is 128 threads (4 simdgroups). Grid: `ceil(N/BN) x ceil(M/BM)`
  (`gpu_memory.mm:1634-1635`).

- **Elementwise / Convert**: 1D threadgroups of 256 threads with 1D grids
  of `ceil(N/256)` groups (`gpu_memory.mm:1384-1386`).

- **Reduce**: 256 threads per group with tree reduction in threadgroup memory
  (`metal_softfloat.h`). Two-pass: first pass reduces chunks of 256
  to partial results, host launches second pass if needed.

### GPU Warmup

After pipeline creation, a 1x1 warmup dispatch is issued
(`gpu_memory.mm:1249-1271`). This primes the Metal driver's internal
caches and JIT compiler, avoiding a latency spike on the first real dispatch.
The warmup uses a tiny 64-byte scratch buffer and the SF64 pipeline with
1 threadgroup.

### Occupancy-Aware Scoring

The configuration search (`compute_all_configs`, `gpu_memory.mm:566`)
evaluates every candidate tile configuration via `score_config`
(`gpu_memory.mm:400`):

```
score = occupancy * arithmetic_intensity * sqrt(BM * BN) * BK
```

Where:
- **Occupancy**: `min(max_tg_mem / shared_bytes, max_threads / threads)`,
  capped at 4. Multiple threadgroups per compute unit hide memory latency.
- **Arithmetic intensity**: `2 * BM * BN * BK / ((BM*BK + BK*BN) * elem_bytes)`.
  Higher values mean more compute per byte loaded.
- **Tile area**: `sqrt(BM * BN)` favors larger output tiles that amortize
  shared memory loads.
- **BK amortization**: Linear in BK, because larger K-blocks mean fewer
  barrier synchronizations and loop iterations.

Shared memory cost is computed per-kernel-type by `compute_shared_bytes`
(`gpu_memory.mm:344-394`). Each kernel type has different storage needs:
SF64 uses 8 bytes per element (uint2), DF64 uses 8 bytes (float2), F32
uses 4 bytes (float), and Ozaki uses 6 A-slices + 6 B-slices + scratch.

### Adaptive Row-Blocking

Large matrices risk GPU command timeout (~2 seconds on macOS). The
`compute_chunk_m` function (`gpu_memory.mm:586`) computes the maximum
number of M-rows per dispatch:

```cpp
target = 4096 * 4096 / (K * weight);
target = align_to_tile(target, bm);
target = min(target, 4096);
```

Weight factors reflect per-element cost: SF64=8.0 (integer emulation is
expensive), FP53=4.0, Ozaki-II=3.0, FP24=2.0, DF64=1.5, F32=1.0.
Each chunk dispatches `C_chunk[chunk_M x N] = A_chunk[chunk_M x K] * B[K x N]`
using Metal buffer offsets (`gpu_memory.mm:1717-1719`):

```objc
[enc setBuffer:buf_a offset:(size_t)m_off * K * 8 atIndex:0];
[enc setBuffer:buf_c offset:(size_t)m_off * N * 8 atIndex:2];
```

This avoids allocating separate sub-buffers while keeping each GPU dispatch
within the timeout window.

### Hardware Profile Detection

`metal_init` (`gpu_memory.mm:736`) populates a `HardwareProfile`
(`gpu_memory.mm:105-111`) by querying the Metal device:

| Field | API | Typical Value |
|-------|-----|---------------|
| `max_tg_mem` | `maxThreadgroupMemoryLength` | 32KB (Apple7+) |
| `max_threads_per_tg` | Conservative 1024 | Checked per-pipeline |
| `thread_exec_width` | Always 32 | All Apple Silicon |
| `gpu_family` | `supportsFamily:MTLGPUFamilyAppleN` | 7=M1, 8=M2, 9=M3 |
| `device_mem` | `recommendedMaxWorkingSetSize` | Varies (8-192 GB) |

GPU family detection (`gpu_memory.mm:753-769`) uses `@available` checks to
test `MTLGPUFamilyApple7` (M1), `MTLGPUFamilyApple8` (M2), and
`MTLGPUFamilyApple9` (M3). The family number is logged at init and can be
used for future family-specific kernel tuning.

---

## 11. Buffer Pool Architecture

### Problem

Metal buffer allocation (`[MTLDevice newBufferWithLength:options:]`) involves
kernel transitions and VM mapping. In batched workloads (ML training loops,
repeated matmul), allocation overhead dominates for small-to-medium matrices.

### Size-Binned Recycling

The buffer pool (`gpu_memory.mm:611-653`) is a size-binned cache of
`MTLBuffer` objects. Each requested allocation is rounded up to the next
power of 2, and buffers are stored in buckets keyed by that rounded size.

The bucket function (`pool_bucket`, `gpu_memory.mm:619-625`) computes the
next power of 2 using standard bit-spreading:

```cpp
static size_t pool_bucket(size_t bytes) {
    if (bytes == 0) return 1;
    size_t v = bytes - 1;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16; v |= v >> 32;
    return v + 1;
}
```

This means a 5000-byte request uses an 8192-byte bucket. The overhead
(wasted space) is at most 2x, which is acceptable because Metal buffers
are virtual-memory-backed and do not consume physical pages for untouched
regions on Apple Silicon.

### Allocation and Release

`pool_alloc` (`gpu_memory.mm:627-637`) checks the bucket for a recycled
buffer. If available, it pops the last one (LIFO for cache warmth). If empty,
it allocates a new buffer with `MTLResourceStorageModeShared` for unified
CPU/GPU access:

```objc
return [g_metal_device newBufferWithLength:bucket
                                   options:MTLResourceStorageModeShared];
```

`pool_release` (`gpu_memory.mm:641-649`) returns a buffer to its bucket. If
the bucket already contains `POOL_MAX_PER_BUCKET` (8) buffers
(`gpu_memory.mm:639`), the excess buffer is dropped and ARC frees the
underlying MTLBuffer. This caps per-bucket memory at 8 * bucket_size bytes.

### Pool and Arena Interaction

The buffer pool is used exclusively for GPU dispatch scratch buffers (input,
output, and intermediate buffers in the matmul pipelines). It does not
interact directly with the Eshkol arena allocator, which manages tensor
and heap objects on the CPU side. The flow is:

1. Arena allocates tensor data (f64 doubles) as CPU memory
2. GPU dispatch wraps or copies tensor data into pooled `MTLBuffer`s
3. GPU kernel executes on the buffer contents
4. Results are `memcpy`d back to the arena-managed tensor
5. Pooled buffers are returned via `pool_release`

For the Metal unified memory path, `eshkol_gpu_wrap_host`
(`gpu_memory.mm:1398-1424`) can wrap arena pointers directly as MTLBuffers
using `newBufferWithBytesNoCopy:` for zero-copy GPU access when the memory
is page-aligned.

### Pool Drain on Shutdown

`metal_shutdown` (`gpu_memory.mm:1277-1301`) calls `pool_drain()`
(`gpu_memory.mm:651-653`) as its first action, clearing the entire
`unordered_map<size_t, vector<id<MTLBuffer>>>`. ARC releases all retained
MTLBuffer objects. Pipeline states, the library, command queue, and device
are then set to `nil` in order.

---

## 12. CUDA Backend Status

### Implementation Scope

The CUDA backend (`lib/backend/gpu/gpu_memory_cuda.cpp`, 760 lines) is a
fully functional GPU acceleration path for NVIDIA hardware. Unlike the Metal
backend which requires SF64 software emulation, CUDA provides native f64
hardware, so no precision emulation is needed.

### What Is Fully Implemented

**Initialization** (`gpu_memory_cuda.cpp`): Device enumeration via
`cudaGetDeviceCount`, device 0 selection, CUDA stream creation, cuBLAS
handle creation and stream binding. Thread-safe via `std::mutex` +
double-check pattern in `eshkol_gpu_init` (`gpu_memory_cuda.cpp`).

**Memory management** (`gpu_memory_cuda.cpp`): Three allocation modes:
- `ESHKOL_MEM_UNIFIED`: `cudaMallocManaged()` -- automatic page migration
  between host and device
- `ESHKOL_MEM_DEVICE`: `cudaMalloc()` -- device-only, maximum throughput
- `ESHKOL_MEM_HOST_PINNED`: `cudaMallocHost()` -- pinned host memory for
  async DMA

Host wrapping (`gpu_memory_cuda.cpp`) uses `cudaHostRegister()` to
pin existing memory. If registration fails (e.g., already-registered or
non-page-aligned), it falls back to unified allocation + memcpy.

Sync (`gpu_memory_cuda.cpp`) handles bidirectional transfers:
unified memory uses `cudaStreamSynchronize`, pinned memory uses
`cudaMemcpyAsync` with explicit direction.

**Matmul via cuBLAS** (`gpu_memory_cuda.cpp`): Both f64 (`cublasDgemm`)
and f32 (`cublasSgemm`) are implemented. The row-major to column-major
conversion uses the standard transpose trick:

```cpp
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K, &alpha, B_ptr, N, A_ptr, K, &beta, C_ptr, N);
```

This computes C^T = B^T * A^T in column-major, which is equivalent to
C = A * B in row-major.

**Custom kernels** (`gpu_cuda_kernels.cu`, 409 lines, launched via extern "C"
declarations in `gpu_memory_cuda.cpp`):
- Elementwise: 15 operations dispatched by integer op code, 256 threads/block
- Reduce: Two-pass block reduction with `__shfl_down_sync` warp shuffle
- Reduce axis: N-dimensional axis reduction
- Transpose: Tiled with `TILE_DIM=32`, `BLOCK_ROWS=8`, shared memory +1 padding
- Softmax: Three-pass (max, exp+sum, normalize)
- Normalize: Layer normalization (mean, variance, scale+shift)

**Full public API**: All `eshkol_gpu_*` functions
(`gpu_memory_cuda.cpp`) are implemented: `init`, `shutdown`, `alloc`,
`alloc_aligned`, `free`, `wrap_host`, `sync`, `sync_async`, `wait`,
`matmul_f64`, `matmul_f32`, `elementwise_f64`, `reduce_f64`,
`reduce_axis_f64`, `transpose_f64`, `softmax_f64`, `normalize_f64`,
`set_threshold`, `get_threshold`, `should_use`, and `matmul_dispatch`.

Every GPU function has a CPU fallback path: if the CUDA backend is not
active or the GPU operation fails, the function falls through to a scalar
CPU implementation in the same file (`gpu_memory_cuda.cpp`).

### What Is Not Implemented

- **SF64/DF64/FP24/FP53/Ozaki-II precision tiers**: These are Metal-only.
  CUDA has native f64, so software emulation is unnecessary.
- **Buffer pool**: CUDA uses `cudaMalloc`/`cudaFree` directly. No recycling
  pool is implemented (CUDA's allocator is already efficient for this pattern).
- **Multi-stream dispatch**: A single CUDA stream is used for all operations.
  Concurrent kernel execution would require stream pools.
- **Tensor core dispatch**: cuBLAS may internally use tensor cores when
  available, but there is no explicit `cublasGemmEx` / `CUBLAS_COMPUTE_64F`
  path for Ampere/Hopper tensor core f64.

### cuBLAS Integration Path

cuBLAS is the primary compute backend. The integration is minimal:

1. `cublasCreate` + `cublasSetStream` at init (`gpu_memory_cuda.cpp`)
2. `cublasDgemm` / `cublasSgemm` for matmul (`gpu_memory_cuda.cpp`)
3. `cublasDestroy` at shutdown (`gpu_memory_cuda.cpp`)

Custom CUDA kernels handle all non-matmul operations (elementwise, reduce,
transpose, softmax, normalize) because cuBLAS only provides BLAS-level
routines.

### Runtime Backend Selection

Metal and CUDA coexist via compile-time guards and runtime detection:

- **Compile-time**: `ESHKOL_GPU_METAL_ENABLED` (macOS builds) and
  `ESHKOL_GPU_CUDA_ENABLED` (Linux/Windows builds) are set by CMake.
  `gpu_memory.mm` is compiled as Objective-C++ on macOS;
  `gpu_memory_cuda.cpp` is compiled as standard C++ on Linux/Windows.
  A stub file (`gpu_memory_stub.cpp`, 341 lines) provides no-op
  implementations when neither backend is available.

- **Runtime**: `eshkol_gpu_init` (`gpu_memory_cuda.cpp`) tries CUDA
  first (on builds with `ESHKOL_GPU_CUDA_AVAILABLE`). The Metal backend
  (`gpu_memory.mm`) follows the same pattern. The first backend that
  successfully initializes sets `g_active_backend`; subsequent API calls
  dispatch to it. Only one backend is active at a time.

- **Backend query**: `eshkol_gpu_get_backend()` returns the active enum
  (`ESHKOL_GPU_METAL`, `ESHKOL_GPU_CUDA`, or `ESHKOL_GPU_NONE`).
  `eshkol_gpu_supports_f64()` returns true for CUDA (native) and false
  for "no GPU" (`gpu_memory_cuda.cpp`). The Metal backend reports
  f64 support via SF64 emulation.

## 9. CI Testing: Compilation vs. Execution

Two separate things are gated, and it matters which one you're looking at:

- **Compilation** (`.github/workflows/ci.yml`, the `linux-x64-cuda`,
  `linux-arm64-cuda`, and `windows-x64-cuda` jobs): installs a pinned NVIDIA
  CUDA toolkit, configures with both `ESHKOL_GPU_ENABLED=ON` and
  `ESHKOL_REQUIRE_GPU_BACKEND=ON`, and runs `scripts/verify_gpu_backend.py`
  before `scripts/run_gpu_tests.sh`. The verifier requires `nvcc`, CUDA runtime,
  cuBLAS, and both real CUDA sources and rejects the CPU stub. NVIDIA does not
  provide a native Windows ARM64 CUDA toolkit, so no such artifact is labeled
  or shipped. CUDA 12 builds on newer GNU hosts must select a supported
  compiler for the whole build (for example `CC=gcc-13 CXX=g++-13 cmake ...`);
  overriding only `CMAKE_CUDA_HOST_COMPILER` can mix libstdc++ ABI and search
  paths at final link time. These jobs run on GitHub-hosted runners, which have no GPU. The
  real backend compiles, but at runtime `eshkol_gpu_init()` finds zero devices, so
  `tests/gpu/*.esk` all execute their CPU fallback path. These lanes
  prove the GPU backend *builds*; they do not exercise a single GPU
  kernel.

- **Execution** (`.github/workflows/gpu-execution-gate.yml`,
  `tests/gpu/gpu_correctness_gate.sh`): builds Eshkol twice — once with
  GPU acceleration on, once off — and diffs GPU-vs-CPU output on the
  same differentiable workload within a numeric tolerance. This is the
  gate that actually proves a Metal or CUDA kernel ran and produced a
  correct answer. It exits 0 with a SKIP message on any host without a
  real GPU device (which includes essentially all GitHub-hosted
  runners), so it only ever produces a verdict where one is possible.
  It needs a self-hosted runner carrying `[self-hosted, gpu]` labels to
  run in CI; until one is registered it stays queued rather than
  failing anything. It can also be run by hand on any workstation with
  a real GPU (Metal on any Mac, CUDA with an NVIDIA driver present —
  including the Jetson AGX Xavier setup in `nix/jetson/`).
