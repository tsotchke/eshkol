# XLA Backend — Eshkol Compiler Internals

**Status**: Production (v1.1-accelerate)
**Source**: `lib/backend/xla/xla_runtime.cpp` (1249 lines), `inc/eshkol/backend/xla/xla_codegen.h`, `inc/eshkol/backend/xla/xla_runtime.h`
**Build flag**: `-DESHKOL_XLA_ENABLED`

---

## 1. Overview

The Eshkol XLA backend provides a JAX/StableHLO-style API surface for tensor operations while executing through direct LLVM codegen calling 11 C runtime functions. The design is pragmatic: the API contract and dispatch hierarchy mirror XLA's design, but the current execution path skips MLIR/StableHLO compilation entirely and delegates to calibrated BLAS and GPU libraries.

This approach provides three benefits. First, the codegen layer (`XLACodegen`) can be upgraded to true XLA JIT in the future without changing any call sites. Second, the existing `eshkol_xla_*` C functions already handle N-dimensional tensors, broadcasting, autodiff gradients, and GPU dispatch — the full semantic scope of XLA operations. Third, the cost model separates dispatch decisions from computation, enabling hardware-specific tuning without recompilation.

**Dispatch hierarchy** (outermost to innermost):

```
Eshkol tensor op (Scheme source)
  └─ LLVM IR call site (XLACodegen::emit*)
       └─ eshkol_xla_* C runtime function
            ├─ GPU path: eshkol_gpu_should_use(n) → Metal/CUDA kernel
            └─ CPU path: cBLAS (matmul) / scalar loop (elementwise, reduce)
```

For matmul specifically, the CPU path calls `eshkol_matmul_f64`, which itself dispatches through the BLAS backend:

```
eshkol_xla_matmul
  ├─ GPU (large N*M): eshkol_gpu_matmul_f64 → Metal MPS / cuBLAS
  └─ CPU: eshkol_matmul_f64 → Apple Accelerate AMX / BLAS / SIMD fallback
```

The XLA threshold governs whether `XLACodegen` emits an XLA runtime call at all; below the threshold, the tensor codegen uses its inline SIMD path. This threshold defaults to 100,000 elements (~316×316 matrix) and is configurable via `ESHKOL_XLA_THRESHOLD`.

---

## 2. Architecture

### 2.1 Layer Stack

| Layer | Files | Role |
|---|---|---|
| Eshkol source | `*.esk` | User-facing tensor expressions |
| AST/Parser | `lib/frontend/parser.cpp` | S-expression to AST |
| LLVM codegen | `lib/backend/llvm_codegen.cpp` | Dispatches to TensorCodegen |
| TensorCodegen | `lib/backend/tensor_codegen.cpp` | SIMD/XLA dispatch gate |
| XLACodegen | `lib/backend/xla/` (headers) | Emits `eshkol_xla_*` call IR |
| XLA runtime | `lib/backend/xla/xla_runtime.cpp` | C functions called from IR |
| GPU memory | `lib/backend/gpu/gpu_memory.h` | Metal/CUDA buffer abstraction |
| BLAS backend | `lib/backend/blas_backend.cpp` | AMX/NEON/AVX matmul |

### 2.2 Threshold Configuration

Three compile-time-adjustable thresholds govern dispatch:

```
Scalar fallback:  N <= 16 elements      (handled in blas_backend.cpp)
SIMD:             16 < N < 4,000        (TensorCodegen SIMD path)
cBLAS:            N >= 4,000 elements   (blas_backend: BLAS dispatch)
XLA/GPU:          N >= 100,000 elements (eshkol_gpu_should_use)
```

The XLA threshold can be overridden at runtime:

```bash
ESHKOL_XLA_THRESHOLD=50000 ./my_binary
```

Or programmatically via `xla_set_threshold(n)` (`inc/eshkol/backend/xla/xla_codegen.h:37`).

### 2.3 GPU Initialization

GPU initialization uses `std::call_once` to guarantee thread-safe one-time setup
(`lib/backend/xla/xla_runtime.cpp:33-39`):

```cpp
static std::once_flag g_xla_gpu_init_flag;

static void ensure_gpu_initialized() {
    std::call_once(g_xla_gpu_init_flag, []() {
        eshkol_gpu_init();
    });
}
```

Every XLA runtime function that may dispatch to GPU calls `ensure_gpu_initialized()` before
calling `eshkol_gpu_should_use(n)`. This pattern prevents races in multi-threaded compilation
or JIT contexts.

---

## 3. Tensor Representation

### 3.1 Core Struct

Tensors in the XLA runtime are represented by `eshkol_tensor_t`
(`lib/backend/xla/xla_runtime.cpp:49-54`):

```c
typedef struct eshkol_tensor {
    uint64_t* dimensions;     // idx 0: array of dimension sizes, length = num_dimensions
    uint64_t  num_dimensions; // idx 1: rank (0 = scalar, 1 = 1-D, 2 = matrix, ...)
    int64_t*  elements;       // idx 2: doubles stored as int64 bit patterns
    uint64_t  total_elements; // idx 3: product of all dimensions
} eshkol_tensor_t;
```

The struct is laid out contiguously in arena memory immediately after an 8-byte object header
(which stores `HEAP_SUBTYPE_TENSOR`). The object header sits at `ptr - 8` relative to the
tensor struct pointer; this is how `vector-for-each` and similar operations distinguish tensors
from heterogeneous vectors at runtime.

### 3.2 int64 Bit Pattern Encoding

The `elements` field is typed as `int64_t*` but stores IEEE 754 `double` values via bit-pattern
reinterpretation. This is not lossy: `sizeof(double) == sizeof(int64_t) == 8` and the reinterpret
cast is safe under the C++ strict aliasing rules for `std::memcpy`-equivalent accesses. All XLA
runtime functions cast the pointer before use:

```c
double* out = reinterpret_cast<double*>(result->elements);
```

The int64 representation serves two purposes: (1) the tagged value system can store a tensor
pointer in the `data` field of a 16-byte tagged value without type confusion, and (2) tensor
elements can be compared and hashed as integers when needed.

### 3.3 Arena Allocation

All tensors are allocated from the Eshkol arena via `arena_allocate_tensor_full`:

```c
extern "C" eshkol_tensor_t* arena_allocate_tensor_full(
    void* arena, uint64_t num_dims, uint64_t total_elements);
// lib/backend/xla/xla_runtime.cpp:56-57
```

This function allocates a single contiguous block containing:

```
[8-byte header: HEAP_SUBTYPE_TENSOR tag]
[eshkol_tensor_t struct: 32 bytes]
[dimensions array: num_dims * 8 bytes]
[elements array: total_elements * 8 bytes]
```

Every XLA runtime function checks the return value for `nullptr` and returns `nullptr` on
allocation failure, propagating OOM to the caller as a raised Eshkol error.

### 3.4 Maximum Rank

The XLA runtime supports up to **rank 16** for `reduce`, `argreduce`, `transpose`, `broadcast`,
and `slice`. This limit is enforced by stack-allocated working arrays of size 16:

```c
if (rank > 16) return nullptr; // max 16D tensors supported
// lib/backend/xla/xla_runtime.cpp:310
```

---

## 4. Operation Catalog

### 4.1 Matrix Multiplication

```c
void* eshkol_xla_matmul(
    void* arena,
    const double* a_data, const double* b_data,
    const int64_t* a_shape, const int64_t* b_shape,
    int64_t a_rank, int64_t b_rank)
// lib/backend/xla/xla_runtime.cpp:67
```

**Constraints**: Only 2-D tensors. Inner dimensions must match (`a_shape[1] == b_shape[0]`).
Result has shape `[a_shape[0], b_shape[1]]`.

**Dispatch** (in order):
1. `eshkol_gpu_should_use(M * N)` — GPU via `eshkol_gpu_matmul_f64` (Metal MPS or cuBLAS)
2. CPU fallback via `eshkol_matmul_f64` (Apple Accelerate AMX or manual BLAS/SIMD)

**Complexity**: O(M × K × N), ~2MKN FLOPs.

### 4.2 Element-wise Operations

```c
void* eshkol_xla_elementwise(
    void* arena,
    const double* a_data, const double* b_data,
    int64_t total_elements,
    const uint64_t* shape, int64_t rank,
    int64_t op_code)
// lib/backend/xla/xla_runtime.cpp:134
```

**Op codes** (match `ElementwiseOp` enum in `inc/eshkol/backend/xla/xla_codegen.h:41-53`):

| Code | Name | Arity | Formula |
|---|---|---|---|
| 0 | ADD | binary | `a[i] + b[i]` |
| 1 | SUB | binary | `a[i] - b[i]` |
| 2 | MUL | binary | `a[i] * b[i]` |
| 3 | DIV | binary | `a[i] / b[i]` |
| 4 | EXP | unary | `exp(a[i])` |
| 5 | LOG | unary | `log(a[i])` |
| 6 | SIN | unary | `sin(a[i])` |
| 7 | COS | unary | `cos(a[i])` |
| 8 | TANH | unary | `tanh(a[i])` |
| 9 | RELU | unary | `max(a[i], 0.0)` |
| 10 | SIGMOID | unary | `1 / (1 + exp(-a[i]))` |

Binary ops (codes 0-3) require non-null `b_data`. Unary ops (codes 4-10) ignore `b_data`.

**GPU op-code mapping**: The GPU `EshkolElementwiseOp` enum has two additional codes inserted
at positions 4 (NEG) and 5 (ABS) relative to the XLA enum. A static translation table
remaps XLA codes 4-10 to GPU codes 6-12:

```c
static const int xla_to_gpu_elemwise[] = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12};
// lib/backend/xla/xla_runtime.cpp:164
```

**Complexity**: O(N) per element.

### 4.3 Reduction

```c
void* eshkol_xla_reduce(
    void* arena,
    const double* data, int64_t total_elements,
    const uint64_t* shape, int64_t rank,
    int64_t axis, int64_t op_code)
// lib/backend/xla/xla_runtime.cpp:234
```

**Op codes** (match `ReduceOp` enum in `inc/eshkol/backend/xla/xla_codegen.h:56-62`):

| Code | Name | Identity | GPU code |
|---|---|---|---|
| 0 | SUM | 0.0 | 0 |
| 1 | MEAN | 0.0 | 4 |
| 2 | MAX | -∞ | 3 |
| 3 | MIN | +∞ | 2 |
| 4 | PROD | 1.0 | 1 |

`axis == -1` reduces over all elements, producing a rank-1 tensor with a single element.
Axis-specific reduction removes that axis from the output shape.

**GPU op-code mapping**: The GPU `EshkolReduceOp` enum has a different ordering
(SUM=0, PROD=1, MIN=2, MAX=3, MEAN=4). Translation:

```c
static const int xla_to_gpu_reduce[] = {0, 4, 3, 2, 1};
// lib/backend/xla/xla_runtime.cpp:261
```

**CPU axis-reduce algorithm**: Uses outer/inner stride decomposition. For an input
of shape `[d0, d1, ..., d_{axis}, ..., d_{rank-1}]`:

```
outer_stride = d_axis * inner_stride
inner_stride = product(d_{axis+1} ... d_{rank-1})

for outer in range(out_total / inner_stride):
    for inner in range(inner_stride):
        acc = identity
        for k in range(axis_len):
            src = outer * outer_stride + k * inner_stride + inner
            acc = combine(acc, data[src])
        out[outer * inner_stride + inner] = acc
```

**Complexity**: O(N) for full reduce; O(N) for axis reduce with constant factor from
stride arithmetic.

### 4.4 In-place Scale

```c
void* eshkol_xla_scale_inplace(double* data, int64_t total_elements, double scale)
// lib/backend/xla/xla_runtime.cpp:391
```

Multiplies every element of `data` by `scale` in-place without allocation. Used by the
MEAN gradient backward pass to divide the broadcasted upstream gradient by the axis length.
Returns `data` cast to `void*` for ABI uniformity.

**Complexity**: O(N).

### 4.5 Softmax

```c
void* eshkol_xla_softmax(
    void* arena,
    const double* data, int64_t total_elements,
    const uint64_t* shape, int64_t rank,
    int64_t axis)
// lib/backend/xla/xla_runtime.cpp:404
```

Numerically stable softmax using the log-sum-exp trick. For each slice along `axis`:

1. Compute `max_val = max(slice)`
2. Compute `out[i] = exp(data[i] - max_val)` and accumulate `sum_exp`
3. Normalize: `out[i] /= sum_exp`

A floor of `1e-10` prevents division by zero when all inputs are `-Inf`.

`axis == -1` performs global softmax over all elements. Axis-specific softmax dispatches
to GPU when `inner_stride == 1` (contiguous last-axis case), making it GPU-eligible for
standard batch × classes layouts.

**Complexity**: O(N) with 3 passes over each slice (max, exp+sum, normalize). GPU path
reduces to O(N / parallelism) via Metal compute shader or cuDNN kernel.

### 4.6 Layer Normalization

```c
void* eshkol_xla_normalize(
    void* arena,
    const double* data, int64_t total_elements,
    const uint64_t* shape, int64_t rank,
    int64_t axis, double gamma, double beta, double epsilon)
// lib/backend/xla/xla_runtime.cpp:513
```

Computes layer normalization along `axis`:

```
y = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta
```

`gamma` and `beta` are scalar values applied uniformly. For per-feature affine parameters,
the caller must perform a follow-up element-wise MUL and ADD with per-feature tensors.

GPU dispatch requires `inner_stride == 1` (i.e., `axis` is the last dimension or the only
dimension). The GPU path calls `eshkol_gpu_normalize_f64` which submits the computation
as a single Metal/CUDA kernel.

**Complexity**: O(N) with 3 passes (mean, variance, normalize).

### 4.7 Argreduce (argmax / argmin)

```c
void* eshkol_xla_argreduce(
    void* arena,
    const double* data, int64_t total_elements,
    const uint64_t* shape, int64_t rank,
    int64_t axis, int64_t is_max)   // 1=argmax, 0=argmin
// lib/backend/xla/xla_runtime.cpp:593
```

Returns a tensor of indices (stored as `double` values) locating the maximum or minimum
along `axis`. The output has the same shape as the input with `axis` removed.

`axis == -1` returns a scalar tensor containing the flat index of the global extremum.

**Note on index type**: Indices are returned as `double` rather than `int64_t` because all
XLA runtime results are tensors of doubles. The caller is responsible for converting to
integer if needed. Conversion is exact for indices below 2^53.

**Complexity**: O(N). No GPU dispatch (argreduce requires scatter-like index tracking that
is not expressed in the current GPU kernel set).

### 4.8 Reduce Gradient (Autodiff)

```c
void* eshkol_xla_reduce_gradient(
    void* arena,
    const double* grad_data,
    const double* input_data,
    const uint64_t* input_shape,
    int64_t input_rank,
    int64_t total_input,
    int64_t axis,
    int64_t op_code)   // 2=MAX, 3=MIN, 4=PROD
// lib/backend/xla/xla_runtime.cpp:677
```

Computes `dL/dinput` given `dL/doutput = grad_data` and the forward `input_data`. The
result has the same shape as `input_data`.

**SUM gradient**: Constant 1 — handled in codegen directly by broadcasting `grad_data` back
to input shape via `eshkol_xla_broadcast`. Not routed through this function.

**MEAN gradient**: Constant `1/N` — handled in codegen by broadcast + `eshkol_xla_scale_inplace`.

**MAX/MIN gradient** (op_code 2 or 3): The extremum is located; gradient is split equally
among all positions attaining the extremum:

```
grad_in[i] = (input[i] == extremum) ? upstream_grad / count : 0
```

**PROD gradient** (op_code 4): Uses the divide-by-element identity, with careful handling
of zero elements to avoid NaN:

```
if zero_count > 1:  grad[i] = 0               # product is 0, grad is 0
if zero_count == 1: grad[zero_idx] = total_product * upstream; others = 0
else:               grad[i] = (total_product / input[i]) * upstream
```

**Complexity**: O(N) with one or two passes depending on op.

### 4.9 Transpose

```c
void* eshkol_xla_transpose(
    void* arena,
    const double* data,
    const uint64_t* shape,
    int64_t rank,
    const int64_t* perm)
// lib/backend/xla/xla_runtime.cpp:807
```

General N-dimensional axis permutation. `perm[i]` specifies which source axis maps to output
axis `i`. For a 2-D matrix, `perm = [1, 0]` yields the standard transpose.

GPU dispatch for `rank == 2` calls `eshkol_gpu_transpose_f64`, which submits a tiled
transpose kernel that avoids cache thrashing on large matrices.

**CPU algorithm**: Flat index decomposition via destination strides, then source index
reconstruction using `perm`:

```c
for flat in range(total):
    remaining = flat
    src_flat = 0
    for d in range(rank):
        idx = remaining / dst_strides[d]
        remaining %= dst_strides[d]
        src_flat += idx * src_strides[perm[d]]
    out[flat] = data[src_flat]
// lib/backend/xla/xla_runtime.cpp:865-876
```

**Complexity**: O(N) with rank-proportional index arithmetic per element.

### 4.10 Broadcast

```c
void* eshkol_xla_broadcast(
    void* arena,
    const double* data,
    const uint64_t* src_shape, int64_t src_rank,
    const uint64_t* tgt_shape, int64_t tgt_rank)
// lib/backend/xla/xla_runtime.cpp:883
```

NumPy-style right-aligned broadcasting. Shapes are aligned from the right; source dimensions
of size 1 (or absent, treated as size 1) are expanded to match the target.

The alignment offset is `offset = tgt_rank - src_rank`. For output dimension `d`, the
corresponding source dimension is `d - offset`. If `d - offset < 0` (extra leading dimensions
in target) or `src_shape[d - offset] == 1` (broadcast dimension), the source index is zero.

```c
int64_t src_d = d - offset;
if (src_d >= 0 && src_d < src_rank && src_shape[src_d] > 1) {
    src_flat += idx * src_strides[src_d];
}
// lib/backend/xla/xla_runtime.cpp:928-931
```

**Complexity**: O(N_target) — proportional to output size, independent of source size.

### 4.11 Slice

```c
void* eshkol_xla_slice(
    void* arena,
    const double* data,
    const uint64_t* shape, int64_t rank,
    const int64_t* starts,
    const int64_t* limits,
    const int64_t* strides)
// lib/backend/xla/xla_runtime.cpp:941
```

Extracts a sub-tensor with per-dimension `[start, limit)` ranges and optional stride.
Output dimension `i` has size `ceil((limits[i] - starts[i]) / strides[i])`.

`strides` may be `nullptr`, in which case all strides default to 1. Non-unit strides
implement the equivalent of Python's `tensor[::2]` downsampling.

**Complexity**: O(N_output).

---

## 5. GPU Dispatch

### 5.1 Cost Model

The GPU dispatch decision is made by `eshkol_gpu_should_use(n)` defined in
`inc/eshkol/backend/gpu/gpu_memory.h:237`. The underlying cost model
(`lib/backend/blas_backend.cpp:44-65`) calibrates two backends:

| Backend | Peak throughput | Fixed overhead | Efficiency scale |
|---|---|---|---|
| Apple Accelerate AMX (cBLAS) | ~1100 GFLOPS | 5 µs | 10,000 elements |
| Metal sf64 (GPU softfloat) | ~200 GFLOPS | 200 µs | 100,000,000 elements |

The GPU overhead of 200 µs reflects Metal command buffer submission latency plus the cost
of wrapping arena-allocated host buffers via `eshkol_gpu_wrap_host` (which creates an
`MTLBuffer` over existing memory without copying on Apple Silicon unified memory).

At equal peak utilization, GPU becomes faster than cBLAS only when:

```
gpu_overhead + N/gpu_gflops < blas_overhead + N/blas_gflops
```

Solving for N:

```
N > (gpu_overhead - blas_overhead) / (1/blas_gflops - 1/gpu_gflops)
  ≈ (200e3 - 5e3) ns / (1/1100 - 1/200) GFLOPS^-1
  ≈ 195e3 / 0.00409 ns/GFLOP
  ≈ 47.7 billion elements
```

This analysis explains the `g_gpu_matmul_threshold = 1,000,000,000` default
(`lib/backend/blas_backend.cpp:71`): GPU matmul is effectively reserved for matrices
beyond ~31,623 × 31,623. For elementwise operations, which have lower per-element cost
on CPU, the threshold is `ESHKOL_XLA_THRESHOLD = 100,000`.

### 5.2 GPU Buffer Wrapping

All XLA runtime functions that dispatch to GPU use `eshkol_gpu_wrap_host` to create GPU
buffer handles over the arena-allocated host memory:

```c
EshkolGPUBuffer buf_a;
if (eshkol_gpu_wrap_host((void*)a_data, M * K * sizeof(double), &buf_a) == 0) {
    // Metal: creates MTLBuffer from existing VA — zero copy on Apple Silicon
    // CUDA: calls cudaHostRegister to pin the memory
    ...
    eshkol_gpu_free(&buf_a);  // release MTLBuffer reference / unpin
}
// lib/backend/xla/xla_runtime.cpp:113-122
```

On Apple Silicon with unified memory, `eshkol_gpu_wrap_host` is zero-copy: it creates an
`MTLBuffer` backed by the existing virtual address. On CUDA with discrete GPU memory, it
pins the host page and may initiate an async DMA transfer.

### 5.3 GPU Fallback Semantics

All GPU dispatch blocks are wrapped in `if (...) { if (...success...) { return result; } }`
chains. If any step fails (GPU init, buffer wrap, kernel submission), control falls through
to the CPU loop without error propagation. This is intentional: GPU operations are
opportunistic acceleration, not a required code path.

---

## 6. XLARuntime C++ Class

### 6.1 Implementation Structure

The `XLARuntime` class uses the PIMPL idiom (`lib/backend/xla/xla_runtime.cpp:1003-1011`):

```cpp
class XLARuntime::Impl {
public:
    bool initialized_ = false;
    Target target_ = Target::CPU;       // CPU, Metal, CUDA, Vulkan
    size_t allocated_bytes_ = 0;
    size_t peak_bytes_ = 0;
    std::unordered_map<void*, std::future<ExecutionResult>> async_handles_;
    size_t next_handle_id_ = 1;
};
```

### 6.2 Target Enum

```cpp
enum class Target {        // inc/eshkol/backend/xla/xla_codegen.h:65-70
    CPU,      // LLVM direct dispatch
    CUDA,     // NVIDIA GPU via CUDA
    Metal,    // Apple GPU via Metal
    Vulkan    // Cross-platform (future)
};
```

### 6.3 Buffer Descriptor

```cpp
struct BufferDescriptor {         // inc/eshkol/backend/xla/xla_runtime.h:27-32
    void* data;
    std::vector<int64_t> shape;
    size_t element_size;          // 8 for double
    bool on_device;
};
```

### 6.4 Execution Result

```cpp
struct ExecutionResult {          // inc/eshkol/backend/xla/xla_runtime.h:37-41
    bool success;
    std::string error_message;
    int64_t execution_time_ns;
};
```

### 6.5 Initialization

`initialize(Target)` sets `impl_->target_` and calls `eshkol_gpu_init()` for non-CPU
targets. If the requested GPU backend is not available, it still marks `initialized_ = true`
because all XLA runtime functions have CPU fallbacks
(`lib/backend/xla/xla_runtime.cpp:1020-1047`).

The `getDefaultRuntime()` singleton auto-detects the best available backend via
`eshkol_gpu_get_backend()` and initializes once using `std::call_once`
(`lib/backend/xla/xla_runtime.cpp:1229-1246`).

### 6.6 Execute and ExecuteAsync

`execute()` casts the `executable` pointer to a function pointer of type
`void(*)(const void* const*, void* const*)` and calls it directly. This interface is
designed for future StableHLO JIT executables, which conform to the same calling convention
(`lib/backend/xla/xla_runtime.cpp:1062-1098`).

`executeAsync()` wraps the `execute()` call in `std::async(std::launch::async, ...)` and
stores the resulting `std::future<ExecutionResult>` keyed by an integer handle. Callers
retrieve results via `wait(handle)` which calls `future.get()` and removes the entry.

### 6.7 Buffer Management

For CPU target, `allocateDevice` uses `std::calloc` and tracks `allocated_bytes_` and
`peak_bytes_`. `toDevice` is a zero-copy wrap (host pointer re-used). `toHost` uses
`memcpy` if the source and destination pointers differ.

Arena-allocated tensors must **not** be freed via `freeBuffer`; only buffers allocated
through `allocateDevice` are safe to free. This distinction is by convention — the
`on_device` flag is `false` for both, so callers must track provenance.

### 6.8 Synchronize

`synchronize()` drains all pending `async_handles_` by calling `.wait()` on each future.
On CPU this is always a no-op because `execute()` is synchronous.

---

## 7. Autodiff Integration

The XLA backend participates in Eshkol's compiler-integrated automatic differentiation
through two mechanisms.

### 7.1 Reduce Gradient

`eshkol_xla_reduce_gradient` provides the backward pass for reduction operations where
the gradient cannot be expressed as a simple broadcast. For SUM and MEAN, the codegen
emits an `eshkol_xla_broadcast` call (for SUM, upstream grad × 1) or a broadcast followed
by `eshkol_xla_scale_inplace` (for MEAN, divide by `axis_len`).

For MAX, MIN, and PROD, the gradient depends on the forward input values, so
`eshkol_xla_reduce_gradient` receives both `grad_data` (upstream gradient in reduced shape)
and `input_data` (forward input in full shape).

### 7.2 Matmul Gradient

The `XLACodegen::emitGradient` method (declared in `inc/eshkol/backend/xla/xla_codegen.h:178`)
computes the two matmul gradients via transpose + matmul:

```
dL/dA = dL/dC @ B^T   (calls emitTranspose(B) then emitMatmul(grad, B_T))
dL/dB = A^T @ dL/dC   (calls emitTranspose(A) then emitMatmul(A_T, grad))
```

Both gradient matmuls go through the same `eshkol_xla_matmul` runtime function and thus
benefit from the same GPU/BLAS dispatch.

### 7.3 Elementwise Gradient

`XLACodegen::emitElementwiseGradient` applies chain rule per operation. Examples:
- EXP: `grad * result` (since d/dx exp(x) = exp(x))
- TANH: `grad * (1 - result^2)`
- RELU: `grad * (input > 0)` — requires a threshold mask
- SIGMOID: `grad * result * (1 - result)`

These gradients are all themselves elementwise operations, so they route back through
`eshkol_xla_elementwise`.

---

## 8. Broadcasting Rules

Eshkol's broadcast follows NumPy semantics with right-alignment:

1. Shapes are padded on the left with 1s until they have equal rank.
2. For each dimension `d`, the output size is `max(src_shape[d], tgt_shape[d])`.
3. A dimension can be broadcast only if it equals 1 or equals the target size.

**Examples**:

| Source shape | Target shape | Result shape |
|---|---|---|
| `[3]` | `[4, 3]` | `[4, 3]` — prepend 1: `[1, 3]` → expand |
| `[1, 3]` | `[4, 3]` | `[4, 3]` — first dim expanded |
| `[4, 1]` | `[4, 3]` | `[4, 3]` — second dim expanded |
| `[]` (scalar) | `[4, 3]` | `[4, 3]` — treated as `[1, 1]` |

**Implementation note**: `eshkol_xla_broadcast` does not validate broadcast compatibility.
Passing incompatible shapes produces undefined memory reads. Shape compatibility must be
verified at the codegen level before emission.

The `XLATypes::broadcastShape` static method (`inc/eshkol/backend/xla/xla_types.h:161`)
computes the result shape at compile time and checks compatibility.

---

## 9. Code Examples

### 9.1 Basic Tensor Operations

```scheme
;;; Create a 3x4 tensor filled with 1.0
(define A (make-tensor (list 3 4) 1.0))

;;; Element-wise operations
(define B (tensor-add A A))        ; calls eshkol_xla_elementwise, op=ADD
(define C (tensor-exp A))          ; calls eshkol_xla_elementwise, op=EXP
(define D (tensor-tanh C))         ; calls eshkol_xla_elementwise, op=TANH

;;; Reduction
(define row-sums (tensor-reduce A 1 'sum))  ; reduce axis 1, shape [3]
(define total    (tensor-reduce A -1 'sum)) ; full reduce, shape [1]

;;; Matmul (3x4 @ 4x2 = 3x2)
(define W (make-tensor (list 4 2) 0.5))
(define out (matmul A W))

;;; Softmax over last axis
(define probs (softmax out -1))
```

### 9.2 Gradient Computation

```scheme
;;; Forward pass
(define (linear-layer x w)
  (tensor-relu (matmul x w)))

;;; Backward pass (automatic via compiler AD)
;;; dL/dW = x^T @ upstream_grad
;;; dL/dx = upstream_grad @ w^T
;;; dL/d(relu input) = upstream_grad * (relu_input > 0)
```

### 9.3 Layer Normalization

```scheme
;;; Normalize along the feature axis (axis=1 for [batch, features])
;;; gamma=1.0, beta=0.0, epsilon=1e-5
(define normed (tensor-normalize X 1 1.0 0.0 1e-5))
```

### 9.4 Checking GPU Dispatch Threshold

```bash
# Override via environment
ESHKOL_XLA_THRESHOLD=50000 eshkol-run my_model.esk -o my_model
ESHKOL_GPU_MATMUL_THRESHOLD=100000000 ./my_model  # use GPU for N >= 10000x10000
```

---

## 10. Performance Considerations

### 10.1 When cBLAS Dominates

On Apple Silicon (M-series), `eshkol_matmul_f64` via Apple Accelerate achieves ~1.1-1.2
TFLOPS through AMX (Apple Matrix Extension) hardware. This is substantially faster than
Metal's softfloat-64 path (~200 GFLOPS), which must emulate double precision arithmetic
in shader code. As a result:

- Matmul at 2048×2048: ~cBLAS handles in under 10 ms at full AMX utilization.
- Matmul at 15000×15000: ~1.2 TFLOPS via cBLAS (see `benchmarks/matmul_bench.esk`).
- GPU matmul is only faster than cBLAS at sizes where GPU memory bandwidth dominates
  arithmetic compute — this threshold is above 31,623×31,623 on Apple Silicon.

### 10.2 When GPU Wins

Elementwise and reduction operations on large tensors (≥100K elements) benefit from GPU
parallelism because their arithmetic intensity is low (few FLOPs per byte), making
memory bandwidth the bottleneck. GPU memory bandwidth on Apple Silicon (unified) is
~500 GB/s, while CPU achieves ~150 GB/s.

Softmax and layer normalization are the primary GPU-accelerated operations for inference
workloads due to their large slice sizes and high parallelism.

### 10.3 Arena Allocation Impact

All XLA operations allocate their result tensors from the arena. The arena is a monotonic
bump allocator with no per-tensor free. Long-running programs that accumulate many
intermediate tensors will see memory grow until the arena is reset at a scope boundary.
The `benchmarks/matmul_bench.esk` benchmark notes: "Each test creates ~4 NxN matrices.
Extreme sizes use separate processes" due to arena accumulation.

### 10.4 Elementwise Loop Optimization

The CPU fallback loops in `eshkol_xla_elementwise` (`lib/backend/xla/xla_runtime.cpp:190-226`)
are simple scalar loops. LLVM's auto-vectorizer will vectorize these loops when the target
has AVX/NEON support. The `TensorCodegen::attachLoopMetadata` method
(`lib/backend/tensor_codegen.cpp:73`) emits LLVM loop metadata to hint at vectorization
width when the direct SIMD path is used instead.

### 10.5 Slice and Transpose Cache Behavior

Both `eshkol_xla_transpose` and `eshkol_xla_slice` access `data` non-sequentially for
non-trivial permutations or strides. For large tensors, this causes cache-miss-dominated
performance. The GPU 2-D transpose kernel uses shared memory tiling to mitigate this,
making it worthwhile to enable GPU dispatch for large transpose operations even at lower
element counts than matmul.

---

## 11. Type System Integration

The `XLATypes` class (`inc/eshkol/backend/xla/xla_types.h`) bridges Eshkol's HoTT-inspired
gradual type system to XLA/MLIR types. The primary element type in Eshkol is `F64`
(IEEE 754 double), corresponding to `ElementType::F64`. The type mapping is used during
codegen to emit correctly-typed LLVM IR for tensor operations and to validate broadcast
compatibility at compile time.

Eshkol tensors are always homogeneous `f64` at the XLA backend level. The tagged value
system above the tensor layer allows heterogeneous collections, but tensors proper (heap
subtype `HEAP_SUBTYPE_TENSOR`) store raw doubles as int64 bit patterns.

---

## 12. See Also

- `lib/backend/xla/xla_runtime.cpp` — all 11 C runtime functions (1249 lines)
- `inc/eshkol/backend/xla/xla_codegen.h` — XLACodegen LLVM emit API
- `inc/eshkol/backend/xla/xla_runtime.h` — XLARuntime C++ class API
- `inc/eshkol/backend/xla/xla_types.h` — XLATypes HoTT→MLIR type mapping
- `inc/eshkol/backend/gpu/gpu_memory.h` — EshkolGPUBuffer, GPU backend API
- `lib/backend/blas_backend.cpp` — cost model, AMX/BLAS/SIMD dispatch
- `lib/backend/tensor_codegen.cpp` — TensorCodegen, XLA threshold gate
- `benchmarks/matmul_bench.esk` — matmul performance across all dispatch tiers
- `docs/breakdown/README.md` — project overview and documentation index
