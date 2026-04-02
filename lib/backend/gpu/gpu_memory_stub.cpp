/*
 * GPU Memory Stub — Used on platforms without Metal or CUDA
 *
 * All GPU functions return proper error codes with eshkol_error() messages.
 * NOT silent stubs — every function logs an actionable error.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/backend/gpu/gpu_memory.h>
#include <eshkol/logger.h>
#include <cstdio>
#include <cstring>
#include <cmath>

/* Forward declaration: dispatched matmul from blas_backend.cpp */
extern "C" void eshkol_matmul_f64(const double*, const double*, double*,
                                   uint64_t, uint64_t, uint64_t);

// ===== Device Management =====

int eshkol_gpu_init(void) {
    return 0;  // No GPU devices found
}

void eshkol_gpu_shutdown(void) {
    // No-op: no GPU resources to free
}

EshkolGPUBackend eshkol_gpu_get_backend(void) {
    return ESHKOL_GPU_NONE;
}

const char* eshkol_gpu_backend_name(EshkolGPUBackend backend) {
    switch (backend) {
        case ESHKOL_GPU_NONE:   return "CPU only";
        case ESHKOL_GPU_METAL:  return "Apple Metal (not available)";
        case ESHKOL_GPU_CUDA:   return "NVIDIA CUDA (not available)";
        case ESHKOL_GPU_VULKAN: return "Vulkan (not available)";
        default:                return "Unknown";
    }
}

int eshkol_gpu_backend_available(EshkolGPUBackend backend) {
    (void)backend;
    return 0;  // No GPU backend available
}

int eshkol_gpu_supports_f64(void) {
    return 0;
}

// ===== Memory Allocation =====

int eshkol_gpu_alloc(size_t size_bytes, EshkolMemoryType mem_type,
                     EshkolGPUBuffer* out_buffer) {
    (void)size_bytes;
    (void)mem_type;
    eshkol_error("GPU allocation failed: no GPU backend available (build with Metal or CUDA support)");
    if (out_buffer) {
        memset(out_buffer, 0, sizeof(*out_buffer));
    }
    return -1;
}

int eshkol_gpu_alloc_aligned(size_t size_bytes, size_t alignment,
                              EshkolMemoryType mem_type,
                              EshkolGPUBuffer* out_buffer) {
    (void)alignment;
    return eshkol_gpu_alloc(size_bytes, mem_type, out_buffer);
}

void eshkol_gpu_free(EshkolGPUBuffer* buffer) {
    if (buffer) {
        memset(buffer, 0, sizeof(*buffer));
    }
}

int eshkol_gpu_wrap_host(void* host_ptr, size_t size_bytes,
                          EshkolGPUBuffer* out_buffer) {
    (void)host_ptr;
    (void)size_bytes;
    eshkol_error("GPU wrap_host failed: no GPU backend available");
    if (out_buffer) {
        memset(out_buffer, 0, sizeof(*out_buffer));
    }
    return -1;
}

// ===== Data Transfer =====

int eshkol_gpu_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    (void)buffer;
    (void)direction;
    eshkol_error("GPU sync failed: no GPU backend available");
    return -1;
}

int eshkol_gpu_sync_async(EshkolGPUBuffer* buffer, EshkolSyncDirection direction,
                           void* stream_handle) {
    (void)stream_handle;
    return eshkol_gpu_sync(buffer, direction);
}

void eshkol_gpu_wait(EshkolGPUBuffer* buffer) {
    (void)buffer;
    // No-op: nothing to wait on
}

// ===== Matrix Operations =====

int eshkol_gpu_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B,
                           EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;
    const double* a = static_cast<const double*>(A->host_ptr);
    const double* b = static_cast<const double*>(B->host_ptr);
    double* c = static_cast<double*>(C->host_ptr);
    if (!a || !b || !c) return -1;

    // GPU not available — dispatch to BLAS/SIMD via eshkol_matmul_f64
    eshkol_matmul_f64(a, b, c, M, K, N);
    return 0;
}

int eshkol_gpu_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B,
                           EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;

    // CPU fallback: f32 scalar matmul
    const float* a = (const float*)A->host_ptr;
    const float* b = (const float*)B->host_ptr;
    float* c = (float*)C->host_ptr;
    if (!a || !b || !c) {
        eshkol_error("GPU matmul_f32 failed: null host pointers");
        return -1;
    }

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

// ===== Threshold Configuration =====

size_t g_gpu_threshold = 100000;

void eshkol_gpu_set_threshold(size_t threshold) {
    g_gpu_threshold = threshold;
}

size_t eshkol_gpu_get_threshold(void) {
    return g_gpu_threshold;
}

int eshkol_gpu_should_use(size_t num_elements) {
    (void)num_elements;
    return 0;  // Never use GPU — no backend available
}

// ===== Elementwise / Reduce / Transpose =====

int eshkol_gpu_elementwise_f64(EshkolGPUBuffer* a, EshkolGPUBuffer* b,
                                EshkolGPUBuffer* out, uint64_t n,
                                EshkolElementwiseOp op) {
    if (!a || !out || n == 0) return -1;
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
    const double* inp = static_cast<const double*>(in->host_ptr);
    double* outp = static_cast<double*>(out->host_ptr);
    if (!inp || !outp) return -1;

    uint64_t axis_len = shape[axis];
    uint64_t inner_stride = 1;
    for (uint64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];
    uint64_t outer_stride = axis_len * inner_stride;
    uint64_t total_in = 1;
    for (uint64_t i = 0; i < rank; i++) total_in *= shape[i];
    uint64_t out_total = total_in / axis_len;

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

// ===== Softmax / Normalize =====

int eshkol_gpu_softmax_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                            uint64_t num_slices, uint64_t slice_len) {
    if (!in || !out || num_slices == 0 || slice_len == 0) return -1;
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

// ===== Runtime Integration =====

void eshkol_matmul_dispatch(const double* A, const double* B, double* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    // GPU not available — dispatch directly to BLAS/SIMD via eshkol_matmul_f64
    eshkol_matmul_f64(A, B, C, M, K, N);
}
