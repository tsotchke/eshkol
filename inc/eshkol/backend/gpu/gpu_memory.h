/*
 * GPU Memory Abstraction Layer for Eshkol
 *
 * Provides unified interface for GPU memory management across:
 * - Metal (macOS/iOS) with unified memory architecture
 * - CUDA (Linux/Windows) with discrete GPU memory
 *
 * Design principles:
 * - Zero-copy for unified memory (Metal on Apple Silicon)
 * - Pinned memory for efficient CPU-GPU transfer (CUDA)
 * - Automatic device selection and fallback
 * - Integration with Eshkol's OALR memory model
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_GPU_MEMORY_H
#define ESHKOL_GPU_MEMORY_H

/* C-compatible standard headers: use stddef.h / stdint.h so C consumers
 * (e.g. Moonlab, lilirrep) can include this header without compiling as
 * C++. <cstddef> / <cstdint> are C++-only. */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===== GPU Backend Types =====

typedef enum {
    ESHKOL_GPU_NONE = 0,    // CPU only
    ESHKOL_GPU_METAL = 1,   // Apple Metal (unified memory)
    ESHKOL_GPU_CUDA = 2,    // NVIDIA CUDA
    ESHKOL_GPU_VULKAN = 3   // Vulkan compute (future)
} EshkolGPUBackend;

typedef enum {
    ESHKOL_MEM_HOST = 0,          // Regular host memory
    ESHKOL_MEM_HOST_PINNED = 1,   // Pinned host memory (for fast GPU transfer)
    ESHKOL_MEM_DEVICE = 2,        // Device-only memory
    ESHKOL_MEM_UNIFIED = 3        // Unified/shared memory (Metal, CUDA managed)
} EshkolMemoryType;

typedef enum {
    ESHKOL_SYNC_NONE = 0,         // No synchronization needed
    ESHKOL_SYNC_HOST_TO_DEVICE,   // Copy host to device
    ESHKOL_SYNC_DEVICE_TO_HOST,   // Copy device to host
    ESHKOL_SYNC_BIDIRECTIONAL     // Full sync both ways
} EshkolSyncDirection;

// ===== GPU Buffer Handle =====

/**
 * Opaque handle to GPU buffer.
 * Contains both host and device pointers plus metadata.
 */
typedef struct EshkolGPUBuffer {
    void* host_ptr;          // Host-accessible pointer
    void* device_ptr;        // Device pointer (may be same as host_ptr for unified)
    size_t size_bytes;       // Buffer size
    EshkolMemoryType mem_type;
    EshkolGPUBackend backend;
    uint32_t flags;          // Backend-specific flags
    void* backend_data;      // Backend-specific data (MTLBuffer*, etc.)
} EshkolGPUBuffer;

// ===== Device Management =====

/**
 * Initialize GPU subsystem and detect available backends.
 * Call once at program startup. Idempotent — subsequent calls are no-ops.
 *
 * @return 1 if a GPU backend (Metal or CUDA) was initialised and is now
 *         active, 0 if no GPU is available (library falls back to CPU).
 *
 * NOTE: This is a BOOLEAN return, not a device count. The convention is
 * deliberately inverted from POSIX (0 = success, nonzero = error). A C
 * consumer should write `if (eshkol_gpu_init()) { ... }` or
 * `if (!eshkol_gpu_init()) fprintf(stderr, "no GPU, using CPU\n");` —
 * NOT `if (eshkol_gpu_init() != 0) fail();` which would wrongly error on
 * every host that actually has a GPU.
 *
 * For the "is a GPU present at all?" check use `eshkol_gpu_get_backend()`
 * and compare against `ESHKOL_GPU_NONE`. The return of this function is
 * the init success/failure signal, not device enumeration.
 */
int eshkol_gpu_init(void);

/**
 * Shutdown GPU subsystem and free resources.
 */
void eshkol_gpu_shutdown(void);

/**
 * Get the best available GPU backend.
 * @return Backend type (ESHKOL_GPU_NONE if no GPU)
 */
EshkolGPUBackend eshkol_gpu_get_backend(void);

/**
 * Get human-readable backend name.
 * @param backend Backend type
 * @return Static string describing backend
 */
const char* eshkol_gpu_backend_name(EshkolGPUBackend backend);

/**
 * Check if a specific backend is available.
 * @param backend Backend to check
 * @return 1 if available, 0 otherwise
 */
int eshkol_gpu_backend_available(EshkolGPUBackend backend);

/**
 * Check if the active GPU backend has NATIVE hardware f64 support.
 *
 * This returns 0 on Apple Silicon (Metal compute has no hardware fp64)
 * even though Eshkol CAN still run fp64-grade workloads on that host via
 * SF64 / Ozaki-II CRT emulation. If you want "can this Eshkol build run
 * my fp64 work?" use `eshkol_gpu_has_fp64()` instead — that returns 1
 * whenever any fp64 path (native OR emulated) is available.
 *
 * Keep this function for callers that specifically need to know whether
 * native hardware fp64 is available (e.g. to decide whether to allocate
 * a cuDNN-style workspace that only exists on native-fp64 GPUs).
 *
 * @return 1 if the active backend has native hardware f64 support
 *         (CUDA on all modern NVIDIA GPUs), 0 otherwise (Metal, no GPU).
 */
int eshkol_gpu_supports_f64(void);

/**
 * Check if ANY f64 (double-precision) path is available on the active
 * backend, including software-emulated paths (SF64 on Metal).
 *
 * This is the right query for "can I hand this library fp64 data and
 * expect correct-to-ULP results?". SF64 on Apple Silicon gives bit-exact
 * IEEE-754 fp64 via Ozaki-II CRT expansion — see
 * `ESHKOL_GPU_PRECISION=default` in the precision-tier docs. On CUDA
 * this returns 1 (same as `eshkol_gpu_supports_f64`). On Metal this
 * returns 1 whenever GPU init succeeded (SF64 is always available).
 * Returns 0 only when no GPU backend is active at all.
 *
 * @return 1 if any fp64 path is usable (native or emulated), 0 otherwise.
 */
int eshkol_gpu_has_fp64(void);

// ===== Memory Allocation =====

/**
 * Allocate GPU-accessible buffer.
 *
 * For Metal: Uses MTLBuffer with shared storage mode (unified memory).
 * For CUDA: Uses cudaMallocHost (pinned) or cudaMalloc (device).
 *
 * @param size_bytes Size to allocate
 * @param mem_type Type of memory to allocate
 * @param out_buffer Output buffer handle
 * @return 0 on success, error code otherwise
 */
int eshkol_gpu_alloc(size_t size_bytes, EshkolMemoryType mem_type,
                     EshkolGPUBuffer* out_buffer);

/**
 * Allocate buffer with specific alignment.
 * @param size_bytes Size to allocate
 * @param alignment Alignment in bytes (must be power of 2)
 * @param mem_type Type of memory
 * @param out_buffer Output buffer handle
 * @return 0 on success, error code otherwise
 */
int eshkol_gpu_alloc_aligned(size_t size_bytes, size_t alignment,
                              EshkolMemoryType mem_type,
                              EshkolGPUBuffer* out_buffer);

/**
 * Free GPU buffer.
 * @param buffer Buffer to free
 */
void eshkol_gpu_free(EshkolGPUBuffer* buffer);

/**
 * Wrap existing host pointer for GPU use.
 * For Metal: Creates MTLBuffer from existing memory.
 * For CUDA: Registers memory with cudaHostRegister.
 *
 * @param host_ptr Existing host pointer
 * @param size_bytes Size of buffer
 * @param out_buffer Output buffer handle
 * @return 0 on success, error code otherwise
 */
int eshkol_gpu_wrap_host(void* host_ptr, size_t size_bytes,
                          EshkolGPUBuffer* out_buffer);

// ===== Data Transfer =====

/**
 * Synchronize buffer between host and device.
 * No-op for unified memory (Metal on Apple Silicon).
 *
 * @param buffer Buffer to synchronize
 * @param direction Sync direction
 * @return 0 on success, error code otherwise
 */
int eshkol_gpu_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction);

/**
 * Asynchronous sync (non-blocking).
 * @param buffer Buffer to synchronize
 * @param direction Sync direction
 * @param stream_handle Backend-specific stream/queue (NULL for default)
 * @return 0 on success, error code otherwise
 */
int eshkol_gpu_sync_async(EshkolGPUBuffer* buffer, EshkolSyncDirection direction,
                           void* stream_handle);

/**
 * Wait for all pending operations on buffer.
 * @param buffer Buffer to wait on
 */
void eshkol_gpu_wait(EshkolGPUBuffer* buffer);

// ===== Matrix Operations (GPU-accelerated) =====

/**
 * GPU matrix multiplication: C = A * B
 *
 * Uses Metal Performance Shaders (MPS) on Metal.
 * Uses cuBLAS on CUDA.
 *
 * @param A Matrix A buffer (M x K)
 * @param B Matrix B buffer (K x N)
 * @param C Output matrix buffer (M x N)
 * @param M Rows of A and C
 * @param K Columns of A / Rows of B
 * @param N Columns of B and C
 * @return 0 on success, error code otherwise
 */
int eshkol_gpu_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B,
                           EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N);

/**
 * GPU matrix multiplication with single precision.
 * Often faster than f64 and sufficient for ML workloads.
 */
int eshkol_gpu_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B,
                           EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N);

// ===== Threshold Configuration =====

/**
 * Minimum element count to use GPU.
 * Below this threshold, CPU (BLAS/SIMD) is used instead.
 * Default: 100000 (same as XLA threshold)
 */
extern size_t g_gpu_threshold;

/**
 * Set GPU dispatch threshold.
 * @param threshold Minimum elements to use GPU
 */
void eshkol_gpu_set_threshold(size_t threshold);

/**
 * Get current GPU threshold.
 * @return Current threshold
 */
size_t eshkol_gpu_get_threshold(void);

/**
 * Check if GPU should be used for an operation.
 * @param num_elements Number of elements in operation
 * @return 1 if GPU should be used, 0 for CPU
 */
int eshkol_gpu_should_use(size_t num_elements);

// ===== Elementwise Operations =====

typedef enum {
    ESHKOL_ELEMWISE_ADD = 0,
    ESHKOL_ELEMWISE_SUB = 1,
    ESHKOL_ELEMWISE_MUL = 2,
    ESHKOL_ELEMWISE_DIV = 3,
    ESHKOL_ELEMWISE_NEG = 4,
    ESHKOL_ELEMWISE_ABS = 5,
    ESHKOL_ELEMWISE_EXP = 6,
    ESHKOL_ELEMWISE_LOG = 7,
    ESHKOL_ELEMWISE_SIN = 8,
    ESHKOL_ELEMWISE_COS = 9,
    ESHKOL_ELEMWISE_TANH = 10,
    ESHKOL_ELEMWISE_RELU = 11,
    ESHKOL_ELEMWISE_SIGMOID = 12,
    ESHKOL_ELEMWISE_SQRT = 13,
    ESHKOL_ELEMWISE_RECIPROCAL = 14
} EshkolElementwiseOp;

typedef enum {
    ESHKOL_REDUCE_SUM = 0,
    ESHKOL_REDUCE_PROD = 1,
    ESHKOL_REDUCE_MIN = 2,
    ESHKOL_REDUCE_MAX = 3,
    ESHKOL_REDUCE_MEAN = 4
} EshkolReduceOp;

/**
 * GPU elementwise operation on f64 arrays.
 * Binary ops (ADD-DIV): out[i] = a[i] op b[i]
 * Unary ops (NEG-RECIPROCAL): out[i] = op(a[i]), b ignored (can be NULL)
 *
 * @param a Input buffer A
 * @param b Input buffer B (NULL for unary ops)
 * @param out Output buffer
 * @param n Number of elements
 * @param op Operation to perform
 * @return 0 on success, error code otherwise
 */
int eshkol_gpu_elementwise_f64(EshkolGPUBuffer* a, EshkolGPUBuffer* b,
                                EshkolGPUBuffer* out, uint64_t n,
                                EshkolElementwiseOp op);

/**
 * GPU reduction on f64 array.
 *
 * @param in Input buffer
 * @param out Output buffer (single element)
 * @param n Number of input elements
 * @param op Reduction operation
 * @return 0 on success, error code otherwise
 */
int eshkol_gpu_reduce_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                           uint64_t n, EshkolReduceOp op);

/**
 * GPU axis-specific reduction on f64 N-D tensor.
 * Reduces along a single axis, producing output with that axis removed.
 * Output has total_elements / shape[axis] elements.
 *
 * @param in Input buffer (flattened N-D tensor)
 * @param out Output buffer (flattened (N-1)-D tensor)
 * @param rank Number of dimensions
 * @param shape Dimension sizes array [rank]
 * @param axis Which axis to reduce along
 * @param op Reduction operation
 * @return 0 on success, error code otherwise
 */
int eshkol_gpu_reduce_axis_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                                uint64_t rank, const uint64_t* shape,
                                uint64_t axis, EshkolReduceOp op);

/**
 * GPU matrix transpose: out = transpose(in)
 *
 * @param in Input buffer (rows x cols)
 * @param out Output buffer (cols x rows)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return 0 on success, error code otherwise
 */
int eshkol_gpu_transpose_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                              uint64_t rows, uint64_t cols);

// ===== Softmax / Normalize =====

/**
 * GPU numerically-stable softmax over contiguous slices.
 * Each slice of slice_len elements is independently softmaxed.
 *
 * @param in Input buffer (num_slices * slice_len elements)
 * @param out Output buffer (same size)
 * @param num_slices Number of independent slices
 * @param slice_len Length of each slice
 * @return 0 on success, error code otherwise
 */
int eshkol_gpu_softmax_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                            uint64_t num_slices, uint64_t slice_len);

/**
 * GPU layer normalization over contiguous slices.
 * y = gamma * (x - mean) / sqrt(var + epsilon) + beta
 *
 * @param in Input buffer (num_slices * slice_len elements)
 * @param out Output buffer (same size)
 * @param num_slices Number of independent slices
 * @param slice_len Length of each slice
 * @param gamma Scale factor
 * @param beta Shift factor
 * @param epsilon Numerical stability constant
 * @return 0 on success, error code otherwise
 */
int eshkol_gpu_normalize_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                              uint64_t num_slices, uint64_t slice_len,
                              double gamma, double beta, double epsilon);

// ===== Backward Pass GPU Operations =====

/**
 * GPU conv2d backward — compute grad_input.
 */
int eshkol_gpu_conv2d_backward_input_f64(
    EshkolGPUBuffer* grad_out, EshkolGPUBuffer* kernel_weights,
    EshkolGPUBuffer* grad_input,
    uint64_t in_h, uint64_t in_w, uint64_t k_h, uint64_t k_w,
    uint64_t out_h, uint64_t out_w,
    uint64_t stride_h, uint64_t stride_w,
    uint64_t channels_in, uint64_t channels_out, uint64_t batch_size);

/**
 * GPU conv2d backward — compute grad_kernel.
 */
int eshkol_gpu_conv2d_backward_kernel_f64(
    EshkolGPUBuffer* grad_out, EshkolGPUBuffer* saved_input,
    EshkolGPUBuffer* grad_kernel,
    uint64_t in_h, uint64_t in_w, uint64_t k_h, uint64_t k_w,
    uint64_t out_h, uint64_t out_w,
    uint64_t stride_h, uint64_t stride_w,
    uint64_t channels_in, uint64_t channels_out, uint64_t batch_size);

/**
 * GPU batchnorm backward — compute grad_input, grad_gamma, grad_beta.
 */
int eshkol_gpu_batchnorm_backward_f64(
    EshkolGPUBuffer* grad_out,
    EshkolGPUBuffer* saved_input, EshkolGPUBuffer* saved_mean,
    EshkolGPUBuffer* saved_inv_std, EshkolGPUBuffer* saved_gamma,
    EshkolGPUBuffer* grad_input, EshkolGPUBuffer* grad_gamma,
    EshkolGPUBuffer* grad_beta,
    uint64_t batch_size, uint64_t feature_size);

/**
 * GPU layernorm backward — compute grad_input.
 */
int eshkol_gpu_layernorm_backward_f64(
    EshkolGPUBuffer* grad_out,
    EshkolGPUBuffer* saved_input, EshkolGPUBuffer* saved_mean,
    EshkolGPUBuffer* saved_inv_std, EshkolGPUBuffer* saved_gamma,
    EshkolGPUBuffer* grad_input,
    uint64_t num_samples, uint64_t feature_size);

// ===== Runtime Integration =====

/**
 * High-level matmul with automatic GPU/CPU dispatch.
 * Called from generated code for tensor operations.
 *
 * Dispatch order:
 * 1. GPU (if available and num_elements >= threshold)
 * 2. XLA (if available and num_elements >= xla_threshold)
 * 3. BLAS (if available and num_elements >= blas_threshold)
 * 4. SIMD/scalar fallback
 *
 * @param A Matrix A data (row-major, M x K)
 * @param B Matrix B data (row-major, K x N)
 * @param C Output matrix (row-major, M x N)
 * @param M Rows of A and C
 * @param K Columns of A / Rows of B
 * @param N Columns of B and C
 */
/* dtype is the result tensor's element dtype (eshkol_tensor_dtype_t code:
 * 0=f64, 1=f32, 2=f16, 3=bf16, 4=i8). f16/bf16 take the cuBLAS GemmEx
 * tensor-core path on CUDA (ESH-0021); other dtypes use the f64 dispatch. */
void eshkol_matmul_dispatch(const double* A, const double* B, double* C,
                             uint64_t M, uint64_t K, uint64_t N, int32_t dtype);

/* Batched matmul dispatch (ESH-0024): `batch` independent GEMMs C[b]=A[b]·B[b].
 * f16/bf16 take the cublasGemmStridedBatched tensor-core path on CUDA (one
 * launch for all experts); other dtypes use the CPU batched f64 path. */
void eshkol_batch_matmul_dispatch(const double* a, const double* b, double* c,
                                  int64_t batch, int64_t M, int64_t K, int64_t N,
                                  int32_t dtype);

#ifdef __cplusplus
}
#endif

// ===== C++ API =====

#ifdef __cplusplus

namespace eshkol {
namespace gpu {

/**
 * RAII wrapper for GPU buffer.
 */
class Buffer {
public:
    Buffer() : buffer_{} {}

    explicit Buffer(size_t size_bytes, EshkolMemoryType mem_type = ESHKOL_MEM_UNIFIED) {
        eshkol_gpu_alloc(size_bytes, mem_type, &buffer_);
    }

    ~Buffer() {
        if (buffer_.host_ptr || buffer_.device_ptr) {
            eshkol_gpu_free(&buffer_);
        }
    }

    // Move only
    Buffer(Buffer&& other) noexcept : buffer_(other.buffer_) {
        other.buffer_ = {};
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            if (buffer_.host_ptr || buffer_.device_ptr) {
                eshkol_gpu_free(&buffer_);
            }
            buffer_ = other.buffer_;
            other.buffer_ = {};
        }
        return *this;
    }

    // No copy
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // Accessors
    void* host_ptr() const { return buffer_.host_ptr; }
    void* device_ptr() const { return buffer_.device_ptr; }
    size_t size() const { return buffer_.size_bytes; }
    EshkolMemoryType mem_type() const { return buffer_.mem_type; }
    EshkolGPUBackend backend() const { return buffer_.backend; }

    EshkolGPUBuffer* raw() { return &buffer_; }
    const EshkolGPUBuffer* raw() const { return &buffer_; }

    // Sync helpers
    void syncToDevice() { eshkol_gpu_sync(&buffer_, ESHKOL_SYNC_HOST_TO_DEVICE); }
    void syncToHost() { eshkol_gpu_sync(&buffer_, ESHKOL_SYNC_DEVICE_TO_HOST); }
    void wait() { eshkol_gpu_wait(&buffer_); }

private:
    EshkolGPUBuffer buffer_;
};

/**
 * Get available backend.
 */
inline EshkolGPUBackend getBackend() {
    return eshkol_gpu_get_backend();
}

/**
 * Check if GPU is available.
 */
inline bool isAvailable() {
    return eshkol_gpu_get_backend() != ESHKOL_GPU_NONE;
}

/**
 * Initialize GPU subsystem.
 */
inline int init() {
    return eshkol_gpu_init();
}

/**
 * Shutdown GPU subsystem.
 */
inline void shutdown() {
    eshkol_gpu_shutdown();
}

} // namespace gpu
} // namespace eshkol

#endif // __cplusplus

#endif // ESHKOL_GPU_MEMORY_H
