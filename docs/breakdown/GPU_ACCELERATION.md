# GPU Acceleration in Eshkol

## 1. Architecture Overview

Eshkol employs a multi-tier dispatch hierarchy for matrix and tensor operations,
selecting the optimal compute backend at runtime based on problem size, hardware
availability, and a calibrated cost model:

```
Scalar  -->  SIMD  -->  cBLAS  -->  GPU
(tiny)      (small)   (medium)   (massive)
```

The dispatch entry point is `eshkol_matmul_f64` (`blas_backend.cpp:748`):

- **Tiny** (output elements <= 16): Scalar. SIMD/BLAS overhead exceeds compute.
- **Small to massive** (17 to 1B elements): cBLAS via Apple Accelerate (AMX) or
  OpenBLAS. Accelerate sustains 1100+ GFLOPS on M-series. (`blas_backend.cpp:767`)
- **Super-massive** (>= 1B elements, ~31600x31600+): GPU vs BLAS cost model
  comparison. GPU lazily initialized on first encounter. (`blas_backend.cpp:780`)

Backend selection cascades via `[[fallthrough]]` (`blas_backend.cpp:798-845`):
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
actionable errors via `eshkol_error()` (`gpu_memory_stub.cpp:56`).

---

## 2. Cost Model

### Estimation Formula

Defined in `estimate_matmul_cost` (`blas_backend.cpp:99`):

```
flops = 2 * M * K * N
efficiency_b = min(1.0, elements / efficiency_scale_b)
compute_b    = flops / (peak_gflops_b * efficiency_b * 1e9) * 1e9
cost_b       = overhead_b + compute_b
```

The efficiency term models ramp-up: small matrices cannot saturate the backend,
so throughput scales linearly until `efficiency_scale` elements are reached.

### Calibrated Parameters (`blas_backend.cpp:44-65`)

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

### Backend Selection (`blas_backend.cpp:134`)

`select_best_backend` compares estimated costs and selects the minimum. Given
the above parameters, the GPU path is only selected when output elements exceed
~1 billion (`g_gpu_matmul_threshold`, `blas_backend.cpp:71`). The SIMD-to-BLAS
threshold is 64 elements (`blas_backend.cpp:183`) -- cBLAS overhead (~5us) is
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

---

## 4. SF64 Software Float64

Apple Silicon GPUs lack native f64 hardware. SF64 emulates full IEEE 754
double-precision using `uint2` (two 32-bit integers) on Metal's integer ALU.

### Representation (`metal_softfloat.h:28`)

```metal
typedef uint2 sf64;  // .x = high 32 bits, .y = low 32 bits
```

IEEE 754 layout: sign (1 bit, bit 63), exponent (11 bits, bias 1023),
mantissa (52 bits, implicit leading 1). 128-bit intermediates use
`struct sf128 { sf64 hi; sf64 lo; };`.

### Core Operations

**Addition** (`metal_softfloat.h:418`): 11 guard bits via left-shift. After
magnitude add/subtract, normalize leading 1 to bit 62, extract 10 round bits
(`& 0x3FF`), delegate to `sf64_round_pack()`.

**Multiplication** (`metal_softfloat.h:537`): 64x64 -> 128 bit multiply via
`mul64x64()` using Metal's `mulhi()` intrinsic for 32-bit high-word extraction.
4 partial products with branchless carry propagation.

**FMA** (`metal_softfloat.h:696`): True single-rounding using 128-bit
intermediate. Fast path (all operands normal) skips 13 NaN/Inf/Zero/subnormal
checks -- the hot path in matmul accumulation.

**Division** (`metal_softfloat.h:606`): 63-iteration long division. Remainder
preserved as sticky bit for correct rounding.

### Rounding (`metal_softfloat.h:378`)

IEEE 754 round-to-nearest-even with 10 rounding bits, halfway at `0x200`:

```metal
bool round_up = (round_bits > 0x200u) ||
                ((round_bits == 0x200u) && ((sig.y & 1u) != 0));
```

The mask `0x3FF` was a critical bug fix -- the earlier `0x7FF` shifted the
halfway point, causing systematic rounding errors.

### Performance

`shr128_jam` (`metal_softfloat.h:239`) includes a hand-optimized 1-31 bit
fast path (direct 4-word shifts, no nested calls) for FMA exponent alignment.
Each operation introduces at most 0.5 ULP error, matching hardware f64.

---

## 5. CUDA Backend

The CUDA backend (`gpu_memory_cuda.cpp`) provides native f64 GPU acceleration on
NVIDIA hardware -- no software emulation required.

### Initialization (`gpu_memory_cuda.cpp:69`)

Enumerate devices, select device 0, create CUDA stream and cuBLAS handle, bind
handle to stream. Thread-safe via `std::mutex` + double-check pattern.

### cuBLAS Matmul (`gpu_memory_cuda.cpp:183`)

cuBLAS is column-major; Eshkol tensors are row-major. The transpose trick:
```cpp
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K, &alpha, B_ptr, N, A_ptr, K, &beta, C_ptr, N);
```
Both `cublasDgemm` (f64) and `cublasSgemm` (f32) are supported.

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
| Matmul (Ozaki-II) | `matmul_ozaki_gemm` | -- | O(N_mod*MKN) |
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

### CUDA Memory Types (`gpu_memory_cuda.cpp:102-143`)

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
| `ESHKOL_GPU_MATMUL_THRESHOLD` | 1,000,000,000 | `blas_backend.cpp:71` |
| `ESHKOL_BLAS_THRESHOLD` | 64 | `blas_backend.cpp:183` |
| `ESHKOL_FORCE_GPU=1` | -- | Force GPU for all ops |
| `ESHKOL_FORCE_CPU=1` | -- | Disable GPU |
| `ESHKOL_GPU_PRECISION` | "exact" | "exact"/"high"/"fast" (`blas_backend.cpp:82`) |

**Cost model tuning:**

| Variable | Default | Source |
|----------|---------|--------|
| `ESHKOL_BLAS_PEAK_GFLOPS` | 1100 | `blas_backend.cpp:163` |
| `ESHKOL_GPU_PEAK_GFLOPS` | 200 | `blas_backend.cpp:166` |

**Metal kernel tuning:**

| Variable | Values | Variable | Values |
|----------|--------|----------|--------|
| `ESHKOL_SF64_TG` | 8,16,32 | `ESHKOL_SF64_TILE_K` | 4,8,16,32 |
| `ESHKOL_DF64_TG` | 8,16 | `ESHKOL_DF64_BK` | 8,16,32 |
| `ESHKOL_FP_BK` | 8,16,32 | `ESHKOL_FP53_BK` | 8,16 |
| `ESHKOL_F32S_WM` | 1,2,4 | `ESHKOL_F32S_WN` | 1,2,4 |
| `ESHKOL_F32S_BK` | 8,16,32 | `ESHKOL_F32S128_WM` | 1,2,4 |
| `ESHKOL_F32S128_WN` | 1,2,4 | `ESHKOL_F32S128_BK` | 8,16,32 |

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

- **ARM NEON** (`blas_backend.cpp:377`): Cache-blocked with 64x64x64 tiles
  tuned for M-series L1 (64KB data). 4x4 register micro-kernel with
  `vfmaq_f64` FMA and 8 `float64x2_t` accumulators.

- **x86 AVX** (`blas_backend.cpp:495`): 64x64x64 tiles for x86 L1 (~32KB).
  4x8 micro-kernel with `_mm256_fmadd_pd` (FMA3) or separate mul+add, 8
  `__m256d` accumulators.

- **x86 SSE2** (`blas_backend.cpp:617`): Fallback for older x86. 4x4
  micro-kernel with `__m128d`, 64x64x64 blocking.

Edge tiles that do not fill a complete micro-kernel fall back to scalar
computation via `scalar_block` / `avx_scalar_block` / `sse_scalar_block`.
All SIMD implementations zero the result matrix with `memset` before blocked
accumulation to support the `C[i][j] += sum` pattern (`blas_backend.cpp:448`).

---

## 9. SF64 Double-Double Arithmetic -- Algorithmic Deep Dive

Apple Silicon GPUs have no native f64 ALU. The SF64 subsystem
(`lib/backend/gpu/metal_softfloat.h`) emulates full IEEE 754 double-precision
arithmetic using Metal's 32-bit integer pipeline, based on Berkeley SoftFloat
algorithms.

### The `uint2` Representation

A `double` is 64 bits. Metal's Shading Language provides `uint2` (two 32-bit
unsigned integers), which SF64 uses as its storage type
(`metal_softfloat.h:28`):

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
(`metal_softfloat.h:43-46`). Classification functions (`sf64_is_zero`,
`sf64_is_inf`, `sf64_is_nan`, `sf64_is_subnormal`, `sf64_is_signaling_nan`)
operate purely on these bitfields (`metal_softfloat.h:73-92`).

On little-endian ARM64, a `double` in memory stores `[low32, high32]`, but
`uint2.x` is the first word. The `native_to_sf64` / `sf64_to_native` helpers
(`metal_softfloat.h:1267-1268`) perform the swap inline in every kernel,
eliminating the need for a separate word-swap dispatch pass.

### 128-Bit Intermediates

For multiply and FMA, SF64 needs more than 64 bits of intermediate precision.
The `sf128` struct (`metal_softfloat.h:221`) holds two `sf64` values as
high and low halves:

```metal
struct sf128 { sf64 hi; sf64 lo; };   // 128-bit value
```

The `uint128_t` struct (`metal_softfloat.h:329-331`) provides a 4-word
representation (`w3..w0`, MSW first) used for multiplication output. These
are not hardware-supported -- all 128-bit operations are synthesized from
32-bit word arithmetic with explicit carry/borrow propagation.

### Core Operations

**Addition** (`sf64_add`, `metal_softfloat.h:418`): The algorithm uses 11
guard bits by left-shifting both significands by 11, placing the implicit
leading 1 at bit 63. After aligning exponents via `shr64_jam` (which
preserves rounding information as a sticky bit), it adds or subtracts
magnitudes. On same-sign overflow (carry out of bit 63), it shifts right by
2 and sets bit 62. On subtraction, it normalizes via `clz64` to place the
leading 1 at bit 62. Finally, 10 round bits are extracted from the low 10
bits (`& 0x3FF`), the significand is shifted right by 10, and `sf64_round_pack`
produces the final result (`metal_softfloat.h:516-519`).

**Multiplication** (`sf64_mul`, `metal_softfloat.h:537`): Both significands are
shifted to have the leading 1 at bit 63. The 64x64-bit multiply is performed
by `mul64x64` (`metal_softfloat.h:336`), which decomposes into 4 partial
products using Metal's `mulhi()` intrinsic for the high 32-bit word of each
32x32 product. Carries are propagated branchlessly via `select(0u, 1u, ...)`.
The result is a 128-bit product in `uint128_t{w3,w2,w1,w0}`. The product's
leading 1 is at bit 126 or 127; if bit 127 (overflow), the high word is
shifted right by 1 and the exponent incremented. Bits w1|w0 contribute a
sticky bit for correct rounding.

**Fused Multiply-Add** (`sf64_fma`, `metal_softfloat.h:696`): This is the
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
4-word carry propagation (`add128`/`sub128`, `metal_softfloat.h:285-306`).
On subtraction with cancellation, `clz128` finds the new leading 1 and
`shl128` normalizes. The sticky bit is computed from the low 64 bits
(`R.lo.x | R.lo.y`), 10 round bits are extracted from `R.hi.y`, and
`sf64_round_pack` produces the single-rounded result.

**Division** (`sf64_div`, `metal_softfloat.h:606`): Uses 63-iteration long
division. The dividend significand is placed at bit 62 (aligned); if the
dividend is less than the divisor, it is shifted left by 1 and the exponent
decremented. Each iteration compares the remainder against the divisor,
subtracts if larger, and sets the corresponding quotient bit. After the loop,
the remainder is preserved as a sticky bit for correct rounding. This gives
exact results but is the most expensive SF64 operation at 63 serial iterations.

### Rounding: Round-to-Nearest-Even

`sf64_round_pack` (`metal_softfloat.h:378`) implements IEEE 754
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

The `0x3FF` mask was a critical bug fix (documented in `metal_softfloat.h:376`
and the project memory). The earlier `0x7FF` (11 bits) shifted the halfway
point, causing systematic rounding errors in matmul accumulation.

### The `shr128_jam` Fast Path

The `shr128_jam` function (`metal_softfloat.h:239`) is the hot path in FMA
exponent alignment. For shifts of 1-31 bits (the common case when aligning
operands with similar exponents), it uses a hand-optimized 4-word direct
shift (`metal_softfloat.h:252-260`):

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

- **`sf64_exp`** (`metal_softfloat.h:952`): Cody-Waite range reduction
  (x = n*ln2 + r, |r| < ln2/2), then 13-term Horner evaluation using
  `sf64_fma` for each coefficient (1/n! for n=2..13). Final scaling by
  2^n via direct exponent adjustment.

- **`sf64_log`** (`metal_softfloat.h:1054`): Extracts exponent and mantissa,
  reduces to [sqrt(2)/2, sqrt(2)] range, computes s=(f-1)/(f+1), then
  evaluates an odd-power series (2/(2k+1) for k=1..6) via Horner form.
  Result: n*ln2 + log(f).

- **`sf64_sin` / `sf64_cos`** (`metal_softfloat.h:1188-1222`): Cody-Waite
  reduction to [-pi/4, pi/4] via `sf64_trig_reduce`, then minimax polynomials
  (`sf64_sin_poly`: 5 odd terms, `sf64_cos_poly`: 5 even terms), with
  quadrant-based sign/function selection.

- **`sf64_sqrt`** (`metal_softfloat.h:930`): Newton-Raphson with 5 iterations.
  Initial estimate halves the exponent. Each iteration: y = (y + x/y) * 0.5.

- **`sf64_tanh`** (`metal_softfloat.h:1226`): Computed as (e^(2x)-1)/(e^(2x)+1)
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
  (`metal_softfloat.h:1329`). Two-pass: first pass reduces chunks of 256
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

**Initialization** (`gpu_memory_cuda.cpp:69-89`): Device enumeration via
`cudaGetDeviceCount`, device 0 selection, CUDA stream creation, cuBLAS
handle creation and stream binding. Thread-safe via `std::mutex` +
double-check pattern in `eshkol_gpu_init` (`gpu_memory_cuda.cpp:267-296`).

**Memory management** (`gpu_memory_cuda.cpp:102-157`): Three allocation modes:
- `ESHKOL_MEM_UNIFIED`: `cudaMallocManaged()` -- automatic page migration
  between host and device
- `ESHKOL_MEM_DEVICE`: `cudaMalloc()` -- device-only, maximum throughput
- `ESHKOL_MEM_HOST_PINNED`: `cudaMallocHost()` -- pinned host memory for
  async DMA

Host wrapping (`gpu_memory_cuda.cpp:221-240`) uses `cudaHostRegister()` to
pin existing memory. If registration fails (e.g., already-registered or
non-page-aligned), it falls back to unified allocation + memcpy.

Sync (`gpu_memory_cuda.cpp:159-181`) handles bidirectional transfers:
unified memory uses `cudaStreamSynchronize`, pinned memory uses
`cudaMemcpyAsync` with explicit direction.

**Matmul via cuBLAS** (`gpu_memory_cuda.cpp:183-219`): Both f64 (`cublasDgemm`)
and f32 (`cublasSgemm`) are implemented. The row-major to column-major
conversion uses the standard transpose trick:

```cpp
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K, &alpha, B_ptr, N, A_ptr, K, &beta, C_ptr, N);
```

This computes C^T = B^T * A^T in column-major, which is equivalent to
C = A * B in row-major.

**Custom kernels** (`gpu_cuda_kernels.cu`, 409 lines, launched via extern "C"
declarations in `gpu_memory_cuda.cpp:31-47`):
- Elementwise: 15 operations dispatched by integer op code, 256 threads/block
- Reduce: Two-pass block reduction with `__shfl_down_sync` warp shuffle
- Reduce axis: N-dimensional axis reduction
- Transpose: Tiled with `TILE_DIM=32`, `BLOCK_ROWS=8`, shared memory +1 padding
- Softmax: Three-pass (max, exp+sum, normalize)
- Normalize: Layer normalization (mean, variance, scale+shift)

**Full public API**: All `eshkol_gpu_*` functions
(`gpu_memory_cuda.cpp:267-757`) are implemented: `init`, `shutdown`, `alloc`,
`alloc_aligned`, `free`, `wrap_host`, `sync`, `sync_async`, `wait`,
`matmul_f64`, `matmul_f32`, `elementwise_f64`, `reduce_f64`,
`reduce_axis_f64`, `transpose_f64`, `softmax_f64`, `normalize_f64`,
`set_threshold`, `get_threshold`, `should_use`, and `matmul_dispatch`.

Every GPU function has a CPU fallback path: if the CUDA backend is not
active or the GPU operation fails, the function falls through to a scalar
CPU implementation in the same file (`gpu_memory_cuda.cpp:454-507`).

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

1. `cublasCreate` + `cublasSetStream` at init (`gpu_memory_cuda.cpp:80-86`)
2. `cublasDgemm` / `cublasSgemm` for matmul (`gpu_memory_cuda.cpp:190-218`)
3. `cublasDestroy` at shutdown (`gpu_memory_cuda.cpp:93`)

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

- **Runtime**: `eshkol_gpu_init` (`gpu_memory_cuda.cpp:267`) tries CUDA
  first (on builds with `ESHKOL_GPU_CUDA_AVAILABLE`). The Metal backend
  (`gpu_memory.mm`) follows the same pattern. The first backend that
  successfully initializes sets `g_active_backend`; subsequent API calls
  dispatch to it. Only one backend is active at a time.

- **Backend query**: `eshkol_gpu_get_backend()` returns the active enum
  (`ESHKOL_GPU_METAL`, `ESHKOL_GPU_CUDA`, or `ESHKOL_GPU_NONE`).
  `eshkol_gpu_supports_f64()` returns true for CUDA (native) and false
  for "no GPU" (`gpu_memory_cuda.cpp:329-332`). The Metal backend reports
  f64 support via SF64 emulation.
