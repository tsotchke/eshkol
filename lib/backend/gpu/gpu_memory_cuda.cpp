/*
 * CUDA GPU Memory Implementation for Eshkol
 *
 * Provides GPU acceleration via NVIDIA CUDA + cuBLAS on Linux/Windows.
 * This file is a CUDA-only extraction of the unified gpu_memory.mm,
 * compiled as standard C++ (not Objective-C++) for non-Apple platforms.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/gpu/gpu_memory.h"
#include <eshkol/logger.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <atomic>
#include <mutex>

// ============================================================================
// Platform Detection
// ============================================================================

#if defined(ESHKOL_GPU_CUDA_ENABLED)
#define ESHKOL_GPU_CUDA_AVAILABLE 1
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Forward declarations for real CUDA kernel launchers (in gpu_cuda_kernels.cu)
extern "C" int cuda_launch_elementwise_f64(const double* a, const double* b, double* out,
                                            int64_t n, int op, void* stream);
extern "C" int cuda_launch_reduce_f64(const double* in, double* out, int64_t n, int op,
                                       void* stream);
extern "C" int cuda_launch_reduce_axis_f64(const double* in, double* out,
                                            uint64_t rank, const uint64_t* dims,
                                            uint64_t axis, int op, uint64_t out_size,
                                            void* stream);
extern "C" int cuda_launch_transpose_f64(const double* in, double* out,
                                          uint64_t rows, uint64_t cols, void* stream);
extern "C" int cuda_launch_softmax_f64(const double* in, double* out,
                                        uint64_t num_slices, uint64_t slice_len,
                                        void* stream);
extern "C" int cuda_launch_normalize_f64(const double* in, double* out,
                                          uint64_t num_slices, uint64_t slice_len,
                                          double gamma, double beta, double epsilon,
                                          void* stream);
#endif

// ============================================================================
// Global State
// ============================================================================

size_t g_gpu_threshold = 100000;

static EshkolGPUBackend g_active_backend = ESHKOL_GPU_NONE;
static std::atomic<bool> g_gpu_initialized{false};
static std::mutex g_gpu_init_mutex;

// ============================================================================
// CUDA Backend
// ============================================================================

#if ESHKOL_GPU_CUDA_AVAILABLE

static cudaStream_t g_cuda_stream = nullptr;
static cublasHandle_t g_cublas_handle = nullptr;

static int cuda_init(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) return -1;

    err = cudaSetDevice(0);
    if (err != cudaSuccess) return -1;

    err = cudaStreamCreate(&g_cuda_stream);
    if (err != cudaSuccess) return -1;

    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(g_cuda_stream);
        return -1;
    }

    cublasSetStream(g_cublas_handle, g_cuda_stream);

    return 0;
}

static void cuda_shutdown(void) {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
    if (g_cuda_stream) {
        cudaStreamDestroy(g_cuda_stream);
        g_cuda_stream = nullptr;
    }
}

static int cuda_alloc(size_t size_bytes, EshkolMemoryType mem_type, EshkolGPUBuffer* out) {
    cudaError_t err;
    void* ptr = nullptr;

    switch (mem_type) {
        case ESHKOL_MEM_UNIFIED:
            err = cudaMallocManaged(&ptr, size_bytes);
            if (err != cudaSuccess) return -1;
            out->host_ptr = ptr;
            out->device_ptr = ptr;
            break;

        case ESHKOL_MEM_DEVICE:
            err = cudaMalloc(&ptr, size_bytes);
            if (err != cudaSuccess) return -1;
            out->host_ptr = nullptr;
            out->device_ptr = ptr;
            break;

        case ESHKOL_MEM_HOST_PINNED:
            err = cudaMallocHost(&ptr, size_bytes);
            if (err != cudaSuccess) return -1;
            out->host_ptr = ptr;
            out->device_ptr = nullptr;
            break;

        default:
            err = cudaMallocManaged(&ptr, size_bytes);
            if (err != cudaSuccess) return -1;
            out->host_ptr = ptr;
            out->device_ptr = ptr;
            break;
    }

    out->size_bytes = size_bytes;
    out->mem_type = mem_type;
    out->backend = ESHKOL_GPU_CUDA;
    out->backend_data = nullptr;
    out->flags = 0;

    return 0;
}

static void cuda_free(EshkolGPUBuffer* buffer) {
    switch (buffer->mem_type) {
        case ESHKOL_MEM_HOST_PINNED:
            if (buffer->host_ptr) cudaFreeHost(buffer->host_ptr);
            if (buffer->device_ptr && buffer->device_ptr != buffer->host_ptr) {
                cudaFree(buffer->device_ptr);
            }
            break;
        default:
            if (buffer->device_ptr) cudaFree(buffer->device_ptr);
            break;
    }
}

static int cuda_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    if (buffer->mem_type == ESHKOL_MEM_UNIFIED) {
        cudaStreamSynchronize(g_cuda_stream);
        return 0;
    }

    if (buffer->mem_type == ESHKOL_MEM_HOST_PINNED && buffer->device_ptr) {
        cudaError_t err;
        if (direction == ESHKOL_SYNC_HOST_TO_DEVICE || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
            err = cudaMemcpyAsync(buffer->device_ptr, buffer->host_ptr, buffer->size_bytes,
                                   cudaMemcpyHostToDevice, g_cuda_stream);
            if (err != cudaSuccess) return -1;
        }
        if (direction == ESHKOL_SYNC_DEVICE_TO_HOST || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
            err = cudaMemcpyAsync(buffer->host_ptr, buffer->device_ptr, buffer->size_bytes,
                                   cudaMemcpyDeviceToHost, g_cuda_stream);
            if (err != cudaSuccess) return -1;
        }
        cudaStreamSynchronize(g_cuda_stream);
    }

    return 0;
}

static int cuda_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                            uint64_t M, uint64_t K, uint64_t N) {
    const double alpha = 1.0;
    const double beta = 0.0;

    // cuBLAS uses column-major, so we compute C^T = B * A (in cuBLAS terms)
    // which gives us row-major C = A * B
    cublasStatus_t status = cublasDgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        (const double*)B->device_ptr, (int)N,
        (const double*)A->device_ptr, (int)K,
        &beta,
        (double*)C->device_ptr, (int)N);

    cudaStreamSynchronize(g_cuda_stream);
    return (status == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

static int cuda_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                            uint64_t M, uint64_t K, uint64_t N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        (const float*)B->device_ptr, (int)N,
        (const float*)A->device_ptr, (int)K,
        &beta,
        (float*)C->device_ptr, (int)N);

    cudaStreamSynchronize(g_cuda_stream);
    return (status == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

static int cuda_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out) {
    cudaError_t err = cudaHostRegister(host_ptr, size_bytes, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        // Fallback: allocate and copy
        int result = cuda_alloc(size_bytes, ESHKOL_MEM_UNIFIED, out);
        if (result != 0) return result;
        memcpy(out->host_ptr, host_ptr, size_bytes);
        return 0;
    }

    out->host_ptr = host_ptr;
    out->device_ptr = host_ptr;
    out->size_bytes = size_bytes;
    out->mem_type = ESHKOL_MEM_HOST_PINNED;
    out->backend = ESHKOL_GPU_CUDA;
    out->flags = 1;  // Wrapped
    out->backend_data = nullptr;

    return 0;
}

#endif // ESHKOL_GPU_CUDA_AVAILABLE

// ============================================================================
// GPU Dispatch Logging
// ============================================================================

static bool g_gpu_verbose = false;
static bool g_gpu_verbose_checked = false;

static bool gpu_verbose(void) {
    if (!g_gpu_verbose_checked) {
        g_gpu_verbose = (getenv("ESHKOL_GPU_VERBOSE") != nullptr);
        g_gpu_verbose_checked = true;
    }
    return g_gpu_verbose;
}

#define GPU_LOG(fmt, ...) do { if (gpu_verbose()) fprintf(stderr, "[GPU] " fmt "\n", ##__VA_ARGS__); } while(0)

// ============================================================================
// Public API Implementation
// ============================================================================

extern "C" {

int eshkol_gpu_init(void) {
    if (g_gpu_initialized.load(std::memory_order_acquire)) {
        return (g_active_backend != ESHKOL_GPU_NONE) ? 1 : 0;
    }

    std::lock_guard<std::mutex> lock(g_gpu_init_mutex);
    // Double-check after acquiring the lock
    if (g_gpu_initialized.load(std::memory_order_relaxed)) {
        return (g_active_backend != ESHKOL_GPU_NONE) ? 1 : 0;
    }

    // Allow override of GPU dispatch threshold via environment variable
    if (const char* env = std::getenv("ESHKOL_GPU_THRESHOLD")) {
        size_t val = static_cast<size_t>(std::atol(env));
        if (val > 0) g_gpu_threshold = val;
    }

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (cuda_init() == 0) {
        g_active_backend = ESHKOL_GPU_CUDA;
        g_gpu_initialized.store(true, std::memory_order_release);
        eshkol_info("GPU initialized: NVIDIA CUDA");
        return 1;
    }
#endif

    g_active_backend = ESHKOL_GPU_NONE;
    g_gpu_initialized.store(true, std::memory_order_release);
    return 0;
}

void eshkol_gpu_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_gpu_init_mutex);

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        cuda_shutdown();
    }
#endif

    g_active_backend = ESHKOL_GPU_NONE;
    g_gpu_initialized.store(false, std::memory_order_release);
}

EshkolGPUBackend eshkol_gpu_get_backend(void) {
    return g_active_backend;
}

const char* eshkol_gpu_backend_name(EshkolGPUBackend backend) {
    switch (backend) {
        case ESHKOL_GPU_NONE: return "CPU (no GPU)";
        case ESHKOL_GPU_METAL: return "Apple Metal (not available on Linux)";
        case ESHKOL_GPU_CUDA: return "NVIDIA CUDA";
        case ESHKOL_GPU_VULKAN: return "Vulkan";
    }
    return "Unknown";
}

int eshkol_gpu_backend_available(EshkolGPUBackend backend) {
    return (g_active_backend == backend) ? 1 : 0;
}

int eshkol_gpu_supports_f64(void) {
    // CUDA supports f64 on all modern GPUs
    return (g_active_backend == ESHKOL_GPU_CUDA) ? 1 : 0;
}

int eshkol_gpu_alloc(size_t size_bytes, EshkolMemoryType mem_type, EshkolGPUBuffer* out_buffer) {
    if (!out_buffer || size_bytes == 0) return -1;

    memset(out_buffer, 0, sizeof(EshkolGPUBuffer));

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        return cuda_alloc(size_bytes, mem_type, out_buffer);
    }
#endif

    // CPU fallback
    out_buffer->host_ptr = malloc(size_bytes);
    if (!out_buffer->host_ptr) return -1;
    out_buffer->device_ptr = out_buffer->host_ptr;
    out_buffer->size_bytes = size_bytes;
    out_buffer->mem_type = ESHKOL_MEM_HOST;
    out_buffer->backend = ESHKOL_GPU_NONE;
    return 0;
}

int eshkol_gpu_alloc_aligned(size_t size_bytes, size_t alignment,
                              EshkolMemoryType mem_type, EshkolGPUBuffer* out_buffer) {
    size_t aligned_size = (size_bytes + alignment - 1) & ~(alignment - 1);
    return eshkol_gpu_alloc(aligned_size, mem_type, out_buffer);
}

void eshkol_gpu_free(EshkolGPUBuffer* buffer) {
    if (!buffer) return;

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_CUDA) {
        cuda_free(buffer);
        memset(buffer, 0, sizeof(EshkolGPUBuffer));
        return;
    }
#endif

    if (buffer->host_ptr && !(buffer->flags & 1)) {
        free(buffer->host_ptr);
    }
    memset(buffer, 0, sizeof(EshkolGPUBuffer));
}

int eshkol_gpu_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out_buffer) {
    if (!host_ptr || !out_buffer || size_bytes == 0) return -1;

    memset(out_buffer, 0, sizeof(EshkolGPUBuffer));

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        return cuda_wrap_host(host_ptr, size_bytes, out_buffer);
    }
#endif

    // CPU: just use the pointer directly
    out_buffer->host_ptr = host_ptr;
    out_buffer->device_ptr = host_ptr;
    out_buffer->size_bytes = size_bytes;
    out_buffer->mem_type = ESHKOL_MEM_HOST;
    out_buffer->backend = ESHKOL_GPU_NONE;
    out_buffer->flags = 1;
    return 0;
}

int eshkol_gpu_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    if (!buffer) return -1;

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_CUDA) {
        return cuda_sync(buffer, direction);
    }
#endif

    return 0;
}

int eshkol_gpu_sync_async(EshkolGPUBuffer* buffer, EshkolSyncDirection direction, void* stream) {
    (void)stream;
    return eshkol_gpu_sync(buffer, direction);
}

void eshkol_gpu_wait(EshkolGPUBuffer* buffer) {
    (void)buffer;
#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && g_cuda_stream) {
        cudaStreamSynchronize(g_cuda_stream);
    }
#endif
}

int eshkol_gpu_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        GPU_LOG("matmul %llux%llu @ %llux%llu → CUDA cuBLAS", M, K, K, N);
        return cuda_matmul_f64(A, B, C, M, K, N);
    }
#endif

    GPU_LOG("matmul %llux%llu @ %llux%llu → CPU", M, K, K, N);
    // CPU fallback
    extern void eshkol_matmul_f64(const double*, const double*, double*, uint64_t, uint64_t, uint64_t);
    eshkol_matmul_f64((const double*)A->host_ptr, (const double*)B->host_ptr,
                      (double*)C->host_ptr, M, K, N);
    return 0;
}

int eshkol_gpu_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        return cuda_matmul_f32(A, B, C, M, K, N);
    }
#endif

    // CPU fallback: f32 scalar matmul
    {
        const float* a = (const float*)A->host_ptr;
        const float* b = (const float*)B->host_ptr;
        float* c = (float*)C->host_ptr;
        if (!a || !b || !c) return -1;

        for (uint64_t i = 0; i < M; i++) {
            for (uint64_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (uint64_t k = 0; k < K; k++) {
                    sum += a[i * K + k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
        return 0;
    }
}

void eshkol_gpu_set_threshold(size_t threshold) {
    g_gpu_threshold = threshold;
}

size_t eshkol_gpu_get_threshold(void) {
    return g_gpu_threshold;
}

int eshkol_gpu_should_use(size_t num_elements) {
    return (g_active_backend != ESHKOL_GPU_NONE && num_elements >= g_gpu_threshold) ? 1 : 0;
}

void eshkol_matmul_dispatch(const double* A, const double* B, double* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    size_t num_elements = M * N;

    if (eshkol_gpu_should_use(num_elements)) {
        EshkolGPUBuffer buf_a, buf_b, buf_c;
        if (eshkol_gpu_wrap_host((void*)A, M * K * sizeof(double), &buf_a) == 0 &&
            eshkol_gpu_wrap_host((void*)B, K * N * sizeof(double), &buf_b) == 0 &&
            eshkol_gpu_wrap_host((void*)C, M * N * sizeof(double), &buf_c) == 0) {

            if (eshkol_gpu_matmul_f64(&buf_a, &buf_b, &buf_c, M, K, N) == 0) {
                eshkol_gpu_free(&buf_a);
                eshkol_gpu_free(&buf_b);
                eshkol_gpu_free(&buf_c);
                return;
            }
        }
    }

    extern void eshkol_matmul_f64(const double*, const double*, double*, uint64_t, uint64_t, uint64_t);
    eshkol_matmul_f64(A, B, C, M, K, N);
}

int eshkol_gpu_elementwise_f64(EshkolGPUBuffer* a, EshkolGPUBuffer* b,
                                EshkolGPUBuffer* out, uint64_t n,
                                EshkolElementwiseOp op) {
    if (!a || !out || n == 0) return -1;

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && a->device_ptr && out->device_ptr) {
        const double* dp_b = (b && b->device_ptr) ? static_cast<const double*>(b->device_ptr) : nullptr;
        GPU_LOG("elementwise op=%d n=%llu → CUDA kernel", (int)op, (unsigned long long)n);
        int result = cuda_launch_elementwise_f64(
            static_cast<const double*>(a->device_ptr), dp_b,
            static_cast<double*>(out->device_ptr), static_cast<int64_t>(n),
            static_cast<int>(op), static_cast<void*>(g_cuda_stream));
        if (result == 0) return 0;
    }
#endif

    GPU_LOG("elementwise op=%d n=%llu → CPU", (int)op, (unsigned long long)n);
    // CPU fallback
    const double* ap = static_cast<const double*>(a->host_ptr);
    const double* bp = (b && b->host_ptr) ? static_cast<const double*>(b->host_ptr) : nullptr;
    double* cp = static_cast<double*>(out->host_ptr);
    if (!ap || !cp) return -1;
    for (uint64_t i = 0; i < n; i++) {
        switch (op) {
            case ESHKOL_ELEMWISE_ADD: cp[i] = ap[i] + (bp ? bp[i] : 0); break;
            case ESHKOL_ELEMWISE_SUB: cp[i] = ap[i] - (bp ? bp[i] : 0); break;
            case ESHKOL_ELEMWISE_MUL: cp[i] = ap[i] * (bp ? bp[i] : 1); break;
            case ESHKOL_ELEMWISE_DIV: cp[i] = ap[i] / (bp ? bp[i] : 1); break;
            case ESHKOL_ELEMWISE_NEG: cp[i] = -ap[i]; break;
            case ESHKOL_ELEMWISE_ABS: cp[i] = ap[i] < 0 ? -ap[i] : ap[i]; break;
            case ESHKOL_ELEMWISE_EXP: cp[i] = exp(ap[i]); break;
            case ESHKOL_ELEMWISE_LOG: cp[i] = log(ap[i]); break;
            case ESHKOL_ELEMWISE_SIN: cp[i] = sin(ap[i]); break;
            case ESHKOL_ELEMWISE_COS: cp[i] = cos(ap[i]); break;
            case ESHKOL_ELEMWISE_TANH: cp[i] = tanh(ap[i]); break;
            case ESHKOL_ELEMWISE_RELU: cp[i] = ap[i] > 0 ? ap[i] : 0; break;
            case ESHKOL_ELEMWISE_SIGMOID: cp[i] = 1.0 / (1.0 + exp(-ap[i])); break;
            case ESHKOL_ELEMWISE_SQRT: cp[i] = sqrt(ap[i]); break;
            case ESHKOL_ELEMWISE_RECIPROCAL: cp[i] = 1.0 / ap[i]; break;
        }
    }
    return 0;
}

int eshkol_gpu_reduce_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                           uint64_t n, EshkolReduceOp op) {
    if (!in || !out || n == 0) return -1;

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && in->device_ptr && out->device_ptr) {
        int result = cuda_launch_reduce_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            static_cast<int64_t>(n), static_cast<int>(op),
            static_cast<void*>(g_cuda_stream));
        if (result == 0) {
            GPU_LOG("reduce op=%d n=%llu → CUDA kernel", (int)op, (unsigned long long)n);
            return 0;
        }
    }
#endif

    GPU_LOG("reduce op=%d n=%llu → CPU", (int)op, (unsigned long long)n);
    // CPU fallback
    const double* inp = static_cast<const double*>(in->host_ptr);
    double* outp = static_cast<double*>(out->host_ptr);
    if (!inp || !outp) return -1;
    double result;
    switch (op) {
        case ESHKOL_REDUCE_SUM: case ESHKOL_REDUCE_MEAN: result = 0.0; break;
        case ESHKOL_REDUCE_PROD: result = 1.0; break;
        case ESHKOL_REDUCE_MIN: result = INFINITY; break;
        case ESHKOL_REDUCE_MAX: result = -INFINITY; break;
    }
    for (uint64_t i = 0; i < n; i++) {
        switch (op) {
            case ESHKOL_REDUCE_SUM: case ESHKOL_REDUCE_MEAN: result += inp[i]; break;
            case ESHKOL_REDUCE_PROD: result *= inp[i]; break;
            case ESHKOL_REDUCE_MIN: result = (inp[i] < result) ? inp[i] : result; break;
            case ESHKOL_REDUCE_MAX: result = (inp[i] > result) ? inp[i] : result; break;
        }
    }
    if (op == ESHKOL_REDUCE_MEAN) result /= (double)n;
    outp[0] = result;
    return 0;
}

int eshkol_gpu_reduce_axis_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                                uint64_t rank, const uint64_t* shape,
                                uint64_t axis, EshkolReduceOp op) {
    if (!in || !out || !shape || rank == 0 || axis >= rank) return -1;

    uint64_t axis_len = shape[axis];
    uint64_t total_in = 1;
    for (uint64_t i = 0; i < rank; i++) total_in *= shape[i];
    uint64_t out_total = total_in / axis_len;

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && in->device_ptr && out->device_ptr) {
        int result = cuda_launch_reduce_axis_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            rank, shape, axis, static_cast<int>(op), out_total,
            static_cast<void*>(g_cuda_stream));
        if (result == 0) {
            GPU_LOG("reduce_axis op=%d axis=%llu rank=%llu → CUDA kernel", (int)op, (unsigned long long)axis, (unsigned long long)rank);
            return 0;
        }
    }
#endif

    GPU_LOG("reduce_axis op=%d axis=%llu rank=%llu → CPU", (int)op, (unsigned long long)axis, (unsigned long long)rank);
    // CPU fallback
    const double* inp = static_cast<const double*>(in->host_ptr);
    double* outp = static_cast<double*>(out->host_ptr);
    if (!inp || !outp) return -1;

    uint64_t inner_stride = 1;
    for (uint64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];
    uint64_t outer_stride = axis_len * inner_stride;

    for (uint64_t out_idx = 0; out_idx < out_total; out_idx++) {
        uint64_t outer = out_idx / inner_stride;
        uint64_t inner = out_idx % inner_stride;
        double acc;
        switch (op) {
            case ESHKOL_REDUCE_SUM: case ESHKOL_REDUCE_MEAN: acc = 0.0; break;
            case ESHKOL_REDUCE_PROD: acc = 1.0; break;
            case ESHKOL_REDUCE_MIN: acc = INFINITY; break;
            case ESHKOL_REDUCE_MAX: acc = -INFINITY; break;
        }
        for (uint64_t k = 0; k < axis_len; k++) {
            uint64_t src_idx = outer * outer_stride + k * inner_stride + inner;
            double val = inp[src_idx];
            switch (op) {
                case ESHKOL_REDUCE_SUM: case ESHKOL_REDUCE_MEAN: acc += val; break;
                case ESHKOL_REDUCE_PROD: acc *= val; break;
                case ESHKOL_REDUCE_MIN: acc = (val < acc) ? val : acc; break;
                case ESHKOL_REDUCE_MAX: acc = (val > acc) ? val : acc; break;
            }
        }
        if (op == ESHKOL_REDUCE_MEAN) acc /= (double)axis_len;
        outp[out_idx] = acc;
    }
    return 0;
}

int eshkol_gpu_transpose_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                              uint64_t rows, uint64_t cols) {
    if (!in || !out || rows == 0 || cols == 0) return -1;

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && in->device_ptr && out->device_ptr) {
        int result = cuda_launch_transpose_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            rows, cols, static_cast<void*>(g_cuda_stream));
        if (result == 0) {
            GPU_LOG("transpose %llux%llu → CUDA kernel", (unsigned long long)rows, (unsigned long long)cols);
            return 0;
        }
    }
#endif

    GPU_LOG("transpose %llux%llu → CPU", (unsigned long long)rows, (unsigned long long)cols);
    // CPU fallback
    const double* inp = static_cast<const double*>(in->host_ptr);
    double* outp = static_cast<double*>(out->host_ptr);
    if (!inp || !outp) return -1;
    for (uint64_t i = 0; i < rows; i++) {
        for (uint64_t j = 0; j < cols; j++) {
            outp[j * rows + i] = inp[i * cols + j];
        }
    }
    return 0;
}

int eshkol_gpu_softmax_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                            uint64_t num_slices, uint64_t slice_len) {
    if (!in || !out || num_slices == 0 || slice_len == 0) return -1;

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && in->device_ptr && out->device_ptr) {
        int result = cuda_launch_softmax_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            num_slices, slice_len, static_cast<void*>(g_cuda_stream));
        if (result == 0) return 0;
    }
#endif

    // CPU fallback
    const double* inp = static_cast<const double*>(in->host_ptr);
    double* outp = static_cast<double*>(out->host_ptr);
    if (!inp || !outp) return -1;
    for (uint64_t s = 0; s < num_slices; s++) {
        uint64_t base = s * slice_len;
        double max_val = inp[base];
        for (uint64_t k = 1; k < slice_len; k++)
            if (inp[base + k] > max_val) max_val = inp[base + k];
        double sum_exp = 0.0;
        for (uint64_t k = 0; k < slice_len; k++) {
            outp[base + k] = std::exp(inp[base + k] - max_val);
            sum_exp += outp[base + k];
        }
        if (sum_exp == 0.0) sum_exp = 1.0;
        for (uint64_t k = 0; k < slice_len; k++)
            outp[base + k] /= sum_exp;
    }
    return 0;
}

int eshkol_gpu_normalize_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                              uint64_t num_slices, uint64_t slice_len,
                              double gamma, double beta, double epsilon) {
    if (!in || !out || num_slices == 0 || slice_len == 0) return -1;

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && in->device_ptr && out->device_ptr) {
        int result = cuda_launch_normalize_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            num_slices, slice_len, gamma, beta, epsilon,
            static_cast<void*>(g_cuda_stream));
        if (result == 0) return 0;
    }
#endif

    // CPU fallback
    const double* inp = static_cast<const double*>(in->host_ptr);
    double* outp = static_cast<double*>(out->host_ptr);
    if (!inp || !outp) return -1;
    for (uint64_t s = 0; s < num_slices; s++) {
        uint64_t base = s * slice_len;
        double sum = 0.0;
        for (uint64_t k = 0; k < slice_len; k++) sum += inp[base + k];
        double mean = sum / static_cast<double>(slice_len);
        double var_sum = 0.0;
        for (uint64_t k = 0; k < slice_len; k++) {
            double diff = inp[base + k] - mean;
            var_sum += diff * diff;
        }
        double inv_std = 1.0 / std::sqrt(var_sum / static_cast<double>(slice_len) + epsilon);
        for (uint64_t k = 0; k < slice_len; k++)
            outp[base + k] = gamma * (inp[base + k] - mean) * inv_std + beta;
    }
    return 0;
}

} // extern "C"
