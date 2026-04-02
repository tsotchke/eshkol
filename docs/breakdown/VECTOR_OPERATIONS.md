# Vectors and Tensors in Eshkol

**Status**: v1.1-accelerate - Production-ready
**Last Updated**: March 2026

> This document describes the **actual implemented** vector and tensor systems based on source code verification.

---

## Table of Contents

- [Vectors and Tensors in Eshkol](#vectors-and-tensors-in-eshkol)
  - [Table of Contents](#table-of-contents)
  - [Overview: Two Types of Arrays](#overview-two-types-of-arrays)
  - [Scheme Vectors (Heterogeneous)](#scheme-vectors-heterogeneous)
    - [Creation](#creation)
    - [Access](#access)
    - [Storage Layout](#storage-layout)
    - [Operations](#operations)
  - [Tensors (Homogeneous Numeric)](#tensors-homogeneous-numeric)
    - [Creation](#creation-1)
    - [Access](#access-1)
    - [Storage Layout](#storage-layout-1)
    - [Linear Algebra Operations](#linear-algebra-operations)
    - [Automatic Differentiation with Tensors](#automatic-differentiation-with-tensors)
    - [Working Example: Neural Network Layer](#working-example-neural-network-layer)
  - [Performance Considerations](#performance-considerations)
    - [Memory Layout](#memory-layout)
    - [When to Use Each](#when-to-use-each)
  - [See Also](#see-also)

---

## Overview: Two Types of Arrays

Eshkol supports **two distinct array types**, each optimized for different use cases:

| Feature | **Scheme Vectors** | **Tensors** |
|---------|-------------------|-------------|
| **Element types** | Heterogeneous (any type) | Homogeneous (numeric only) |
| **Storage** | Array of 16-byte tagged values | Flat array of double bit patterns |
| **Creation** | `(vector ...)`, `make-vector` | `zeros`, `ones`, `eye`, `arange` |
| **Access** | `vector-ref`, `vector-set!` | `vref`, `tensor-get` |
| **Use case** | Mixed data, collections | Numeric computation, ML |
| **Autodiff** | No | **Yes** (AD-aware) |
| **Linear algebra** | No | **Yes** (`tensor-dot`, etc.) |

**Key Distinction**:
- **Scheme vectors** = Scheme's built-in heterogeneous arrays (like Python lists)
- **Tensors** = NumPy-style numeric arrays for scientific computing

---

## Scheme Vectors (Heterogeneous)

**Implementation**: [`lib/backend/collection_codegen.cpp`](../../lib/backend/collection_codegen.cpp:1195)

Scheme vectors can hold **any type** of element:

### Creation

```scheme
;; Vector literal
(define v (vector 1 "hello" 3.14 #t))

;; make-vector: create with fill value
(define v2 (make-vector 10 0))     ; 10 elements, all 0

;; vector-copy: duplicate
(define v3 (vector-copy v))
```

### Access

```scheme
;; vector-ref: get element at index
(vector-ref v 0)        ; → 1
(vector-ref v 1)        ; → "hello"
(vector-ref v 2)        ; → 3.14

;; vector-set!: mutate element
(vector-set! v 0 42)
(vector-ref v 0)        ; → 42

;; vector-length: get size
(vector-length v)       ; → 4
```

### Storage Layout

```
Scheme Vector Memory Layout:
┌─────────────┬──────────────┬──────────────┬──────────────┐
│ length (8)  │ elem0 (16)   │ elem1 (16)   │ elem2 (16)   │
└─────────────┴──────────────┴──────────────┴──────────────┘
              ↑                ↑                ↑
              Each element is a full eshkol_tagged_value_t
              Can hold ANY type (int, string, closure, etc.)
```

**Size**: 8 + (16 × num_elements) bytes

### Operations

**Scheme vectors are first-class Scheme values**:

```scheme
;; Can be elements in lists
(define mixed (list v 42 "test"))

;; Can be passed to functions
(define (process-vector v)
  (vector-ref v 0))

;; car/cdr work on vectors (returns first element)
(car v)  ; → same as (vector-ref v 0)
```

**NOT supported on Scheme vectors**:
- ❌ Linear algebra operations
- ❌ Automatic differentiation
- ❌ Element-wise arithmetic

For numeric computation, use **tensors** instead.

---

## Tensors (Homogeneous Numeric)

**Implementation**: [`lib/backend/tensor_codegen.cpp`](../../lib/backend/tensor_codegen.cpp:1) (3,041 lines)

Tensors are **N-dimensional numeric arrays** optimized for scientific computing:

### Creation

```scheme
;; zeros: create tensor filled with 0.0
(zeros 10)              ; 1D: shape [10]
(zeros (list 3 3))      ; 2D: shape [3, 3]
(zeros (list 2 3 4))    ; 3D: shape [2, 3, 4]

;; ones: create tensor filled with 1.0
(ones 5)
(ones (list 4 4))

;; eye: identity matrix
(eye 3)                 ; 3x3 identity

;; arange: range with step
(arange 0 10 0.5)       ; [0.0, 0.5, 1.0, ..., 9.5]

;; linspace: evenly spaced values
(linspace 0 1 100)      ; 100 points from 0 to 1

;; reshape: change dimensions (zero-copy)
(reshape (arange 9) 3 3)  ; 1D [9] → 2D [3, 3]
```

### Access

```scheme
;; vref: 1D access (AD-aware!)
(define v (vector 1.0 2.0 3.0))
(vref v 0)              ; → 1.0
(vref v 1)              ; → 2.0

;; tensor-get: N-D access
(define M (reshape (arange 9) 3 3))
(tensor-get M 0 0)      ; → 0.0 (element [0,0])
(tensor-get M 1 2)      ; → 5.0 (element [1,2])

;; Partial indexing returns slice (view, not copy)
(tensor-get M 1)        ; → 1D tensor [3.0, 4.0, 5.0] (row 1)
```

### Storage Layout

```
Tensor Memory Layout:
┌───────────────────────────────────────────────────────────┐
│ eshkol_tensor_t structure (32 bytes, cache-aligned)      │
├───────────────────────────────────────────────────────────┤
│ dimensions*    (8 bytes) → [dim0, dim1, ..., dimN]       │
│ num_dimensions (8 bytes)   Rank (1-4 typical)            │
│ elements*      (8 bytes) → flat array of int64 bit patterns │
│ total_elements (8 bytes)   Total element count           │
└───────────────────────────────────────────────────────────┘

Elements stored as int64 bit patterns of doubles (reinterpret_cast)
Row-major (C-style) ordering for cache efficiency
```

**Note:** Elements are stored as `int64_t` values that are bit patterns of `double` values. This enables storage in cons cells which use int64/double union.

### Linear Algebra Operations

```scheme
;; Element-wise operations
(tensor-add t1 t2)          ; t1 + t2 (element-wise)
(tensor-sub t1 t2)          ; t1 - t2 (element-wise)
(tensor-mul t1 t2)          ; t1 * t2 (element-wise, NOT matrix multiplication)
(tensor-div t1 t2)          ; t1 / t2 (element-wise)

;; Matrix operations
(tensor-dot A B)            ; Matrix multiplication / dot product
(tensor-transpose M)        ; Transpose matrix
(tensor-reshape t shape)    ; Change dimensions (zero-copy)

;; Reduction operations
(tensor-sum t)              ; Sum all elements
(tensor-mean t)             ; Average of all elements
(tensor-max t)              ; Maximum element
(tensor-min t)              ; Minimum element

;; Shape operations
(tensor-shape t)            ; Returns dimensions as vector
(tensor-rank t)             ; Returns number of dimensions
(tensor-size t)             ; Returns total number of elements
```

### Automatic Differentiation with Tensors

The `vref` operator is **AD-aware**: when used during gradient computation, it creates computational graph nodes:

```scheme
;; Define function using tensor access
(define (norm-squared v)
  (let ((x (vref v 0))
        (y (vref v 1)))
    (+ (* x x) (* y y))))

;; Compute gradient
(gradient norm-squared (vector 3.0 4.0))
;; Returns: #(6.0 8.0)
;; Because ∂/∂x(x²+y²) = 2x = 6.0
;; And     ∂/∂y(x²+y²) = 2y = 8.0

;; vref creates AD nodes when in gradient context
;; But is a simple memory access in normal execution
```

### Working Example: Neural Network Layer

```scheme
;; Forward pass through dense layer
(define (dense-layer inputs weights biases)
  (let ((z (tensor-dot inputs weights)))  ; Linear transformation
    (tensor-add z biases)))                 ; Add bias

;; Activation function (applied element-wise)
(define (relu x)
  (if (> x 0.0) x 0.0))

;; Compute loss gradient
(define (layer-gradient inputs targets weights biases)
  (gradient (lambda (w)
              (let* ((z (dense-layer inputs w biases))
                     (activated (tensor-map relu z))
                     (loss (mse-loss activated targets)))
                loss))
            weights))
```

---

## Performance Considerations

### Memory Layout

**Scheme vectors:**
- Cache-unfriendly for numeric computation (16-byte tagged values)
- Type-flexible (can mix types)
- Suitable for: data structures, mixed collections

**Tensors:**
- Cache-friendly (contiguous doubles)
- SIMD-vectorizable
- Suitable for: linear algebra, numerical algorithms, ML

### When to Use Each

Use **Scheme vectors** when:
- Elements have different types
- Implementing data structures (stacks, queues)
- Building collections of mixed values

Use **tensors** when:
- All elements are numeric
- Performing linear algebra
- Computing gradients
- Optimizing for cache/SIMD

---

## Tensor Acceleration

### 1. Cost Model Dispatch

Tensor matmul uses a three-tier dispatch hierarchy: **SIMD -> cBLAS -> GPU**. The dispatch logic lives in `eshkol_matmul_f64()` (`lib/backend/blas_backend.cpp:747`) and is backed by an adaptive cost model defined in the `CostModelParams` struct at line 44.

**Cost model parameters** (from `lib/backend/blas_backend.cpp:44-65`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `blas_peak_gflops` | 1100 | Measured Apple Accelerate AMX throughput on M-series chips |
| `blas_overhead_ns` | 5000 | Fixed cBLAS dispatch overhead (5 microseconds) |
| `blas_efficiency_scale` | 10000 | Elements needed for cBLAS to reach full efficiency |
| `gpu_peak_gflops` | 200 | Measured sf64 (softfloat double) throughput on Metal compute shaders |
| `gpu_overhead_ns` | 200000 | Metal command buffer creation + data transfer overhead (200 microseconds) |
| `gpu_efficiency_scale` | 100000000 | GPU needs 100M+ elements to saturate |
| `simd_peak_gflops` | 25 | Peak SIMD throughput (NEON/AVX/SSE2) |
| `simd_overhead_ns` | 100 | Minimal SIMD dispatch overhead |
| `simd_efficiency_scale` | 1000 | SIMD saturates quickly at ~1000 elements |

**Cost estimation formula** (from `estimate_matmul_cost()` at line 99):

For a matrix multiply of dimensions M x K x N:

```
flops = 2.0 * M * K * N
elements = M * N

efficiency = min(1.0, elements / efficiency_scale)
compute_ns = flops / (peak_gflops * efficiency * 1e9) * 1e9
total_ns   = overhead_ns + compute_ns
```

The `select_best_backend()` function (line 134) evaluates this formula for all available backends and selects the one with the lowest estimated time.

**Fast-path dispatch** (line 752-777) bypasses the cost model entirely for common cases:

- **Tiny matrices** (output elements <= 16, up to 4x4): scalar multiplication. SIMD and BLAS overhead exceeds compute time at this scale.
- **Small to large matrices** (17 to 1B output elements): cBLAS via Apple Accelerate AMX. Even at 225M elements (15000x15000), cBLAS achieves 1100+ GFLOPS.
- **Super-massive matrices** (>= 1B output elements, ~31600x31600): the cost model is invoked and GPU dispatch is considered. The GPU matmul threshold defaults to `g_gpu_matmul_threshold = 1000000000` (line 71).

The critical insight is that GPU is almost never faster for double-precision: sf64 (softfloat f64 emulation on Metal compute shaders) peaks at ~200 GFLOPS, while Apple Accelerate AMX achieves ~1100 GFLOPS natively. GPU dispatch only wins when the matrix is so large that the 200 microsecond overhead is amortized over enough elements to exploit GPU parallelism.

**Environment variables for tuning:**

- `ESHKOL_BLAS_PEAK_GFLOPS` -- override measured BLAS peak throughput (line 163)
- `ESHKOL_GPU_PEAK_GFLOPS` -- override measured GPU peak throughput (line 166)
- `ESHKOL_GPU_MATMUL_THRESHOLD` -- override element count for GPU consideration (line 79)
- `ESHKOL_GPU_PRECISION` -- select GPU precision tier: `exact` (sf64, default), `high` (df64), `fast` (f32) (line 83)
- `ESHKOL_BLAS_THRESHOLD` -- minimum elements for cBLAS path (default 64, line 183)

### 2. GPU Tensor Transfer

Eshkol tensors store elements as `int64_t` bit-patterns representing `double` values. When transferring to GPU, the `double*` data pointer is obtained by `reinterpret_cast<double*>(result->elements)` (see `xla_runtime.cpp:104`), which is safe because `sizeof(double) == sizeof(int64_t) == 8`.

**Metal buffer wrapping** (`metal_wrap_host()` in `gpu_memory.mm:2626`):

On Apple Silicon, the GPU and CPU share a unified memory architecture. The primary wrapping strategy uses `newBufferWithBytesNoCopy:` with `MTLResourceStorageModeShared`, which creates a Metal buffer backed by the existing host pointer with zero-copy semantics:

```objc
id<MTLBuffer> buffer = [g_metal_device newBufferWithBytesNoCopy:host_ptr
                                                         length:size_bytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
```

If `newBufferWithBytesNoCopy` fails (typically because the pointer is not page-aligned), the fallback path allocates a new `MTLResourceStorageModeShared` buffer and copies the data with `memcpy` (`gpu_memory.mm:2633-2636`).

The unified memory flag is detected at initialization: `g_metal_unified_memory = [g_metal_device supportsFamily:MTLGPUFamilyApple1]` (line 749). On Apple Silicon, this is always true, meaning Metal buffers with `StorageModeShared` share the same physical memory as CPU allocations -- no DMA transfer occurs.

**Result marshaling:**

After GPU kernel execution, results are available in the output buffer immediately (unified memory). If the output buffer was allocated via fallback (host_ptr diverged from original pointer), an explicit `memcpy` copies results back (`blas_backend.cpp:2474-2475`):

```cpp
if (buf_c.host_ptr != (void*)C) {
    memcpy((void*)C, buf_c.host_ptr, M * N * sizeof(double));
}
```

**Buffer lifecycle:**

GPU buffers are freed immediately after each operation via `eshkol_gpu_free()` (line 3299), which releases the `MTLBuffer` reference through ARC (`metal_free()` at line 1339 uses `__bridge_transfer` to release ownership). A **buffer pool** (`gpu_memory.mm:617-653`) recycles `MTLBuffer` objects using power-of-2 size buckets to reduce allocation overhead in tight loops (e.g., ML training). Each bucket retains up to `POOL_MAX_PER_BUCKET = 8` buffers.

### 3. XLA Integration

The XLA layer provides a high-level runtime for tensor operations that internally dispatches to GPU, BLAS, or SIMD depending on tensor size. XLA dispatch is governed by a threshold of **100,000 elements** (~316x316 matrix), configurable via `ESHKOL_XLA_THRESHOLD` (defined in `lib/backend/xla/xla_codegen.cpp:55`).

**Dispatch hierarchy at codegen time** (from `tensor_codegen.cpp`):

The LLVM codegen emits a size check: if `total_elements >= xla_get_threshold()`, it calls the XLA C runtime function; otherwise, it uses inline SIMD code. Each XLA runtime function then internally checks `eshkol_gpu_should_use(num_elements)` (which tests `num_elements >= g_gpu_threshold`, default 100K from `gpu_memory.mm:46`) before attempting GPU dispatch.

**Operations with XLA runtime paths** (from `xla_runtime.cpp`):

| Runtime Function | Op Codes | GPU Dispatch |
|-----------------|----------|--------------|
| `eshkol_xla_matmul` | 2D matrix multiply | Yes, via `eshkol_gpu_matmul_f64` |
| `eshkol_xla_elementwise` | ADD(0), SUB(1), MUL(2), DIV(3), EXP(4), LOG(5), SIN(6), COS(7), TANH(8), RELU(9), SIGMOID(10) | Yes, with op code translation to GPU enum |
| `eshkol_xla_reduce` | SUM(0), MEAN(1), MAX(2), MIN(3), PROD(4) | Yes, both global and axis-specific |
| `eshkol_xla_softmax` | Numerically stable softmax | Yes, for contiguous last-axis |
| `eshkol_xla_normalize` | Layer/batch normalization | Yes, for contiguous last-axis |
| `eshkol_xla_transpose` | N-dimensional permutation | Yes, for 2D case |
| `eshkol_xla_broadcast` | Shape broadcasting | CPU only |
| `eshkol_xla_slice` | Strided slicing | CPU only |
| `eshkol_xla_argreduce` | Argmax/argmin | CPU only |
| `eshkol_xla_scale_inplace` | Scalar multiply (for MEAN gradient) | CPU only |

**Op code translation:** XLA and GPU enums differ because the GPU enum inserts NEG(4) and ABS(5). The translation table at `xla_runtime.cpp:164` maps XLA unary ops 4-10 to GPU ops 6-12.

**Broadcasting semantics** (`eshkol_xla_broadcast` at line 883): Source shape is right-aligned with target shape. For each target dimension, if the corresponding source dimension is 1 (or absent because source rank is lower), the value is replicated. This follows NumPy/XLA broadcasting conventions. Implemented via stride-based index mapping on CPU.

**Fusion:** Currently, XLA operations are dispatched as individual C runtime calls (LLVM-only mode, `xla_codegen.cpp:570`). The MLIR/StableHLO path (guarded by `ESHKOL_XLA_FULL_MLIR`) would enable operation fusion for compound patterns like matmul + bias + activation, but this path is not yet active. Each runtime function independently checks for GPU dispatch, so a fused `matmul -> relu` currently incurs two GPU dispatch round-trips.

### 4. AD with Tensors

Eshkol implements reverse-mode automatic differentiation for all major tensor operations. Forward passes record AD nodes with saved tensor data onto a tape. During backpropagation, `eshkol_tensor_backward_dispatch()` (`tensor_backward.cpp:719`) reads each node's type and dispatches to the appropriate backward function.

**Operations with backward passes** (from `tensor_backward.cpp` and `tensor_backward.h`):

| AD Node Type | ID | Backward Function | Gradient Rule |
|-------------|----|--------------------|---------------|
| `AD_NODE_CONV2D` | 19 | `eshkol_backward_conv2d` | `dInput = full_conv(grad, flipped_kernel)`, `dKernel = xcorr(input, grad)` |
| `AD_NODE_MAXPOOL2D` | 20 | `eshkol_backward_maxpool2d` | Scatter gradient to max element positions |
| `AD_NODE_AVGPOOL2D` | 21 | `eshkol_backward_avgpool2d` | Distribute gradient uniformly across pooling window |
| `AD_NODE_BATCHNORM` | 22 | `eshkol_backward_batchnorm` | Standard batchnorm backward formula |
| `AD_NODE_LAYERNORM` | 23 | `eshkol_backward_layernorm` | Per-sample normalization gradient |
| `AD_NODE_MATMUL` | 24 | `eshkol_backward_matmul` | `dA = grad @ B^T`, `dB = A^T @ grad` |
| `AD_NODE_TRANSPOSE` | 25 | `eshkol_backward_transpose` | Transpose of the upstream gradient |
| `AD_NODE_RESHAPE` | 26 | `eshkol_backward_reshape` | `memcpy` (reshape is shape-only, gradient is identical data) |
| `AD_NODE_SUM` | 27 | `eshkol_backward_sum` | Broadcast scalar gradient to all elements |
| `AD_NODE_MEAN` | 28 | `eshkol_backward_mean` | Broadcast scalar gradient / N |
| `AD_NODE_ATTENTION` | 29 | `eshkol_backward_attention` | `dV = attn^T @ grad`, `dQ = softmax_bwd(grad @ V^T) @ K`, `dK = softmax_bwd(...)^T @ Q` |
| `AD_NODE_MULTIHEAD_ATTENTION` | 30 | `eshkol_backward_multihead_attention` | Per-head attention backward + weight projection gradients |
| `AD_NODE_POSITIONAL_ENCODING` | 31 | `eshkol_backward_positional_encoding` | Pass-through (additive constant) |
| `AD_NODE_EMBEDDING` | 32 | `eshkol_backward_embedding` | Scatter-add gradients to weight rows |

**Matmul gradient** (lines 324-356): For `C = A @ B` where A is (M,K) and B is (K,N):

- `dA = grad_C @ B^T` -- shape (M,N) @ (N,K) = (M,K)
- `dB = A^T @ grad_C` -- shape (K,M) @ (M,N) = (K,N)

The BLAS backend also provides `matmul_backward()` (`blas_backend.cpp:262`) which uses `cblas_dgemm` with `CblasTrans` for efficient transposed multiplication with `beta=1.0` for gradient accumulation.

**Reshape/transpose gradients:** Reshape backward is a `memcpy` because reshape only changes the dimension metadata, not the underlying data. Transpose backward is simply the transpose of the upstream gradient: `grad_in[j*rows + i] = grad_out[i*cols + j]`.

**Gradient accumulation:** `eshkol_accumulate_tensor_grad()` (line 1059) implements element-wise `tensor_gradient[i] += grad[i]`. If the node's `tensor_gradient` is NULL, it allocates and zero-fills before accumulating. This supports multiple upstream consumers contributing gradients to the same tensor.

**Gradient seeding:** `eshkol_seed_tensor_gradient()` (line 678) fills the output node's gradient with all-ones (`dL/dL = 1.0`) to initiate backpropagation.

### 5. Reshape Semantics

Reshape in Eshkol is a **zero-copy** operation. The implementation (`tensor_codegen.cpp:7323-7713`) allocates only a new tensor struct header and dimension array via the arena allocator; the `elements` pointer is reused from the source tensor without copying element data.

The key line is at `tensor_codegen.cpp:7706`:

```cpp
// Field 2: elements pointer (reused from source - no copy for reshape)
ctx_.builder().CreateStore(src_elements_ptr, elements_field_ptr);
```

This means reshape creates a new tensor struct `{dims_ptr, ndim, elements, total}` where `elements` points directly into the original tensor's data buffer. The new dimensions array is allocated separately via `arena_alloc` and populated with the requested shape.

**Stride computation:** Eshkol tensors use row-major (C-contiguous) layout. Strides are computed implicitly at access time rather than stored explicitly. For a tensor with shape `[d0, d1, ..., d(n-1)]`, the stride for dimension `i` is `total_elements / product(d0..di)`, computed via division during `tensor-get` (line 364):

```cpp
llvm::Value* stride_i = ctx_.builder().CreateUDiv(total_elements, prod_dims_next);
```

**Contiguity requirement:** Because reshape reuses the element pointer without copying, it relies on the source tensor being contiguous in memory. All Eshkol tensors are contiguous at creation -- there is no explicit stride array in the tensor struct, so non-contiguous views (as might result from slicing with strides) are not possible. A `tensor-get` with fewer indices than dimensions returns a contiguous sub-tensor (a view) whose `elements` pointer is offset into the original (line 458):

```cpp
llvm::Value* slice_start = ctx_.builder().CreateGEP(
    ctx_.int64Type(), elements_ptr, linear_offset);
```

**Input validation:** Reshape supports three argument forms:
1. Individual dimension arguments: `(reshape tensor 3 3)`
2. List argument: `(reshape tensor (list 3 3 2))`
3. Tensor-of-dimensions: `(reshape tensor #(3 3 2))`

All paths merge into a single code path that stores the new dimensions and reuses the source elements. OOM from the arena allocation triggers an exception via `eshkol_raise`.

---

## See Also

- [Automatic Differentiation](AUTODIFF.md) - How `vref` creates AD nodes
- [Type System](TYPE_SYSTEM.md) - HEAP_SUBTYPE_VECTOR vs. HEAP_SUBTYPE_TENSOR
- [Memory Management](MEMORY_MANAGEMENT.md) - Object headers on vectors/tensors
- [API Reference](../API_REFERENCE.md) - Complete tensor operation reference
