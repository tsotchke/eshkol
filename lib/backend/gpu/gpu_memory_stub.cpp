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
    (void)A; (void)B; (void)C; (void)M; (void)K; (void)N;
    eshkol_error("GPU matmul_f64 failed: no GPU backend available (falling back to CPU)");
    return -1;
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

// ===== Runtime Integration =====

void eshkol_matmul_dispatch(const double* A, const double* B, double* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    // GPU not available — dispatch directly to BLAS/SIMD via eshkol_matmul_f64
    extern "C" void eshkol_matmul_f64(const double*, const double*, double*,
                                       uint64_t, uint64_t, uint64_t);
    eshkol_matmul_f64(A, B, C, M, K, N);
}
