/*
 * GPU Memory Stub — Used on platforms without Metal or CUDA
 *
 * All GPU functions return proper error codes with eshkol_error() messages.
 * NOT silent stubs — every function logs an actionable error.
 *
 * With ESHKOL_GPU_VULKAN_ENABLED (Linux builds without CUDA) this file is the
 * host side of the Vulkan compute backend: eshkol_gpu_init() opens a Vulkan
 * device through gpu_vulkan.c and the buffer/kernel entry points route
 * ESHKOL_GPU_VULKAN buffers there. Operations the device does not run
 * (anything outside matmul/elementwise/reduce, or a kernel that failed its
 * admission probe) use the CPU implementations below on host_ptr, which for
 * Vulkan buffers is always the caller-visible data.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/backend/gpu/gpu_memory.h>
#include <eshkol/logger.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#if defined(ESHKOL_GPU_VULKAN_ENABLED)
#include "gpu_vulkan.h"
#include <atomic>
#include <mutex>

static EshkolGPUBackend g_stub_backend = ESHKOL_GPU_NONE;
static std::atomic<bool> g_stub_initialized{false};
static std::mutex g_stub_init_mutex;
extern size_t g_gpu_threshold;

static bool is_vk(const EshkolGPUBuffer* b) {
    return b && b->backend == ESHKOL_GPU_VULKAN;
}
#endif

/* Forward declaration: dispatched matmul from blas_backend.cpp */
extern "C" void eshkol_matmul_f64(const double*, const double*, double*,
                                   uint64_t, uint64_t, uint64_t);

// ===== Device Management =====

/** @brief GPU init. Without Vulkan support compiled in this always reports
 *         zero devices. With it, opens a Vulkan compute device unless
 *         ESHKOL_GPU_BACKEND says cpu/cuda; auto selection takes GPU-type
 *         devices only, ESHKOL_GPU_BACKEND=vulkan also accepts CPU-type ones.
 *         Idempotent and thread-safe. Returns 1 when a backend is active. */
int eshkol_gpu_init(void) {
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    if (g_stub_initialized.load(std::memory_order_acquire)) {
        return g_stub_backend != ESHKOL_GPU_NONE ? 1 : 0;
    }
    std::lock_guard<std::mutex> lock(g_stub_init_mutex);
    if (g_stub_initialized.load(std::memory_order_relaxed)) {
        return g_stub_backend != ESHKOL_GPU_NONE ? 1 : 0;
    }
    if (const char* env = std::getenv("ESHKOL_GPU_THRESHOLD")) {
        size_t val = static_cast<size_t>(std::atol(env));
        if (val > 0) g_gpu_threshold = val;
    }
    const EshkolGPURequest req = eshkol_gpu_backend_request();
    if (req == ESHKOL_GPU_REQUEST_AUTO || req == ESHKOL_GPU_REQUEST_VULKAN) {
        const unsigned flags = (req == ESHKOL_GPU_REQUEST_VULKAN) ? ESHKOL_VK_ALLOW_CPU_DEVICE : 0u;
        if (eshkol_vk_init(flags)) {
            g_stub_backend = ESHKOL_GPU_VULKAN;
            eshkol_info("GPU initialized: Vulkan compute (%s)", eshkol_vk_device_name());
        } else if (req == ESHKOL_GPU_REQUEST_VULKAN) {
            eshkol_warn("ESHKOL_GPU_BACKEND=vulkan: no usable Vulkan device (%s); using CPU",
                        eshkol_vk_status());
        }
    }
    g_stub_initialized.store(true, std::memory_order_release);
    return g_stub_backend != ESHKOL_GPU_NONE ? 1 : 0;
#else
    return 0;  // No GPU devices found
#endif
}

/** @brief GPU shutdown: closes the Vulkan device when one is open; otherwise
 *         a no-op. A later eshkol_gpu_init() re-runs selection. */
void eshkol_gpu_shutdown(void) {
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    std::lock_guard<std::mutex> lock(g_stub_init_mutex);
    if (g_stub_backend == ESHKOL_GPU_VULKAN) eshkol_vk_shutdown();
    g_stub_backend = ESHKOL_GPU_NONE;
    g_stub_initialized.store(false, std::memory_order_release);
#endif
}

/** @brief Active backend: ESHKOL_GPU_VULKAN when a Vulkan device is open,
 *         otherwise ESHKOL_GPU_NONE. */
EshkolGPUBackend eshkol_gpu_get_backend(void) {
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    return g_stub_backend;
#else
    return ESHKOL_GPU_NONE;
#endif
}

/** @brief Human-readable name for a GPU backend enum value (stub build: all
 *         non-CPU backends are annotated "(not available)"). */
const char* eshkol_gpu_backend_name(EshkolGPUBackend backend) {
    switch (backend) {
        case ESHKOL_GPU_NONE:   return "CPU only";
        case ESHKOL_GPU_METAL:  return "Apple Metal (not available)";
        case ESHKOL_GPU_CUDA:   return "NVIDIA CUDA (not available)";
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
        case ESHKOL_GPU_VULKAN: return "Vulkan compute";
#else
        case ESHKOL_GPU_VULKAN: return "Vulkan (not available)";
#endif
        case ESHKOL_GPU_WEBGPU: return "WebGPU (not available)";
        default:                return "Unknown";
    }
}

/** @brief Availability check: true only for the active backend (Vulkan when
 *         a device is open); no other GPU backend exists in this build. */
int eshkol_gpu_backend_available(EshkolGPUBackend backend) {
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    return (backend != ESHKOL_GPU_NONE && backend == g_stub_backend) ? 1 : 0;
#else
    (void)backend;
    return 0;  // No GPU backend available
#endif
}

/** @brief fp64 support: a Vulkan device is only selected when it has
 *         shaderFloat64, so this is true exactly when Vulkan is active. */
int eshkol_gpu_supports_f64(void) {
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    return g_stub_backend == ESHKOL_GPU_VULKAN ? 1 : 0;
#else
    return 0;
#endif
}

/** @brief Stub fp64 support check (alternate name): always false. */
int eshkol_gpu_has_fp64(void) {
    return eshkol_gpu_supports_f64();
}

// ===== Memory Allocation =====

/** @brief Stub GPU allocation: always fails with a logged error explaining
 *         no GPU backend was compiled in; zeroes `*out_buffer` if given. */
int eshkol_gpu_alloc(size_t size_bytes, EshkolMemoryType mem_type,
                     EshkolGPUBuffer* out_buffer) {
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    if (g_stub_backend == ESHKOL_GPU_VULKAN) {
        (void)mem_type;  // Vulkan buffers are host-mapped, with a VRAM twin when the device has one
        return eshkol_vk_alloc(size_bytes, out_buffer) == ESHKOL_VK_OK ? 0 : -1;
    }
#endif
    (void)size_bytes;
    (void)mem_type;
    eshkol_error("GPU allocation failed: no GPU backend available (build with Metal or CUDA support)");
    if (out_buffer) {
        memset(out_buffer, 0, sizeof(*out_buffer));
    }
    return -1;
}

/** @brief Stub aligned GPU allocation: ignores `alignment` and delegates to
 *         eshkol_gpu_alloc() (which always fails). */
int eshkol_gpu_alloc_aligned(size_t size_bytes, size_t alignment,
                              EshkolMemoryType mem_type,
                              EshkolGPUBuffer* out_buffer) {
    (void)alignment;
    return eshkol_gpu_alloc(size_bytes, mem_type, out_buffer);
}

/** @brief Stub GPU free: zeroes the buffer descriptor (there is no GPU
 *         resource to actually release). */
void eshkol_gpu_free(EshkolGPUBuffer* buffer) {
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    if (is_vk(buffer)) {
        eshkol_vk_free(buffer);
        return;
    }
#endif
    if (buffer) {
        memset(buffer, 0, sizeof(*buffer));
    }
}

/** @brief Stub host-pointer wrap: always fails with a logged error (no GPU
 *         backend to wrap the pointer for). */
int eshkol_gpu_wrap_host(void* host_ptr, size_t size_bytes,
                          EshkolGPUBuffer* out_buffer) {
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    if (g_stub_backend == ESHKOL_GPU_VULKAN) {
        return eshkol_vk_wrap_host(host_ptr, size_bytes, out_buffer) == ESHKOL_VK_OK ? 0 : -1;
    }
#endif
    (void)host_ptr;
    (void)size_bytes;
    eshkol_error("GPU wrap_host failed: no GPU backend available");
    if (out_buffer) {
        memset(out_buffer, 0, sizeof(*out_buffer));
    }
    return -1;
}

// ===== Data Transfer =====

/** @brief Stub host/device sync: always fails with a logged error. */
int eshkol_gpu_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    if (is_vk(buffer)) {
        return eshkol_vk_sync(buffer, direction) == ESHKOL_VK_OK ? 0 : -1;
    }
#endif
    (void)buffer;
    (void)direction;
    eshkol_error("GPU sync failed: no GPU backend available");
    return -1;
}

/** @brief Stub async sync: ignores `stream_handle` and delegates to
 *         eshkol_gpu_sync() (which always fails). */
int eshkol_gpu_sync_async(EshkolGPUBuffer* buffer, EshkolSyncDirection direction,
                           void* stream_handle) {
    (void)stream_handle;
    return eshkol_gpu_sync(buffer, direction);
}

/** @brief Stub GPU wait: no-op, since there is never an in-flight GPU
 *         operation to wait on. */
void eshkol_gpu_wait(EshkolGPUBuffer* buffer) {
    (void)buffer;
    // No-op: nothing to wait on
}

// ===== Matrix Operations =====

/** @brief GPU-absent matmul: reads host pointers from the buffer
 *         descriptors and dispatches directly to the CPU BLAS/SIMD
 *         eshkol_matmul_f64() implementation. */
int eshkol_gpu_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B,
                           EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    if (is_vk(A) && is_vk(B) && is_vk(C) &&
        eshkol_vk_matmul_f64(A, B, C, M, K, N) == ESHKOL_VK_OK) {
        return 0;
    }
#endif
    const double* a = static_cast<const double*>(A->host_ptr);
    const double* b = static_cast<const double*>(B->host_ptr);
    double* c = static_cast<double*>(C->host_ptr);
    if (!a || !b || !c) return -1;

    // GPU not available — dispatch to BLAS/SIMD via eshkol_matmul_f64
    eshkol_matmul_f64(a, b, c, M, K, N);
    return 0;
}

/** @brief GPU-absent single-precision matmul: naive triple-loop scalar
 *         implementation directly on the buffers' host pointers. */
int eshkol_gpu_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B,
                           EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    if (is_vk(A) && is_vk(B) && is_vk(C) &&
        eshkol_vk_matmul_f32(A, B, C, M, K, N) == ESHKOL_VK_OK) {
        return 0;
    }
#endif

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

/** @brief Set the (unused, in this stub build) minimum-element GPU dispatch threshold. */
void eshkol_gpu_set_threshold(size_t threshold) {
    g_gpu_threshold = threshold;
}

/** @brief Get the (unused, in this stub build) GPU dispatch threshold. */
size_t eshkol_gpu_get_threshold(void) {
    return g_gpu_threshold;
}

/** @brief Dispatch decision: true when a backend (Vulkan) is active and
 *         `num_elements` meets the threshold; always false without one. */
int eshkol_gpu_should_use(size_t num_elements) {
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    return (g_stub_backend != ESHKOL_GPU_NONE && num_elements >= g_gpu_threshold) ? 1 : 0;
#else
    (void)num_elements;
    return 0;  // Never use GPU — no backend available
#endif
}

// ===== Elementwise / Reduce / Transpose =====

/** @brief GPU-absent elementwise op: CPU scalar loop over `n` elements
 *         applying the unary/binary op selected by `op` directly on the
 *         buffers' host pointers (missing `b` treated as an identity
 *         operand per op: 0 for add/sub, 1 for mul/div). */
int eshkol_gpu_elementwise_f64(EshkolGPUBuffer* a, EshkolGPUBuffer* b,
                                EshkolGPUBuffer* out, uint64_t n,
                                EshkolElementwiseOp op) {
    if (!a || !out || n == 0) return -1;
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    if (is_vk(a) && is_vk(out) && (!b || is_vk(b)) &&
        eshkol_vk_elementwise_f64(a, b, out, n, op) == ESHKOL_VK_OK) {
        return 0;
    }
#endif
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

/** @brief GPU-absent full reduction: CPU scalar loop applying `op`
 *         (sum/mean/prod/min/max) over all `n` input elements to a single
 *         output value. */
int eshkol_gpu_reduce_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                           uint64_t n, EshkolReduceOp op) {
    if (!in || !out || n == 0) return -1;
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    if (is_vk(in) && is_vk(out) && eshkol_vk_reduce_f64(in, out, n, op) == ESHKOL_VK_OK) {
        return 0;
    }
#endif
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

/** @brief GPU-absent axis reduction: CPU implementation that computes
 *         inner/outer strides around the reduced `axis` and applies `op`
 *         over each slice. */
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

/** @brief GPU-absent 2-D transpose: CPU nested loop swapping row/column strides. */
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

/** @brief GPU-absent numerically-stable softmax: for each of `num_slices`
 *         contiguous slices of length `slice_len`, subtracts the slice max
 *         before exponentiating and normalizes by the sum (guarding against
 *         a zero sum). */
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

/** @brief GPU-absent layer-normalize: for each of `num_slices` slices of
 *         length `slice_len`, computes mean/variance and applies
 *         `gamma * (x - mean) / sqrt(var + epsilon) + beta`. */
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

/** @brief Runtime matmul entry point used by generated code: GPU absent, so
 *         dispatches straight to the CPU BLAS/SIMD eshkol_matmul_f64().
 *         `dtype` is ignored — storage is already precision-reduced for the
 *         logical dtype, so plain f64 matmul stays correct. */
void eshkol_matmul_dispatch(const double* A, const double* B, double* C,
                             uint64_t M, uint64_t K, uint64_t N, int32_t /*dtype*/) {
#if defined(ESHKOL_GPU_VULKAN_ENABLED)
    // Same shape as the CUDA dispatcher: lazy init, then the GPU above the
    // element threshold, CPU on any refusal. dtype is ignored (no 16-bit path).
    eshkol_gpu_init();
    if (eshkol_gpu_should_use(M * N)) {
        EshkolGPUBuffer ba{}, bb{}, bc{};
        const bool wa = eshkol_gpu_wrap_host((void*)A, M * K * sizeof(double), &ba) == 0;
        const bool wb = eshkol_gpu_wrap_host((void*)B, K * N * sizeof(double), &bb) == 0;
        const bool wc = eshkol_gpu_wrap_host((void*)C, M * N * sizeof(double), &bc) == 0;
        const bool done = wa && wb && wc &&
                          eshkol_vk_matmul_f64(&ba, &bb, &bc, M, K, N) == ESHKOL_VK_OK;
        if (wa) eshkol_gpu_free(&ba);
        if (wb) eshkol_gpu_free(&bb);
        if (wc) eshkol_gpu_free(&bc);
        if (done) return;
    }
#endif
    // GPU not available — dispatch directly to BLAS/SIMD via eshkol_matmul_f64.
    // dtype is ignored (no tensor-core path without a GPU); storage is already
    // f64 precision-reduced for the logical dtype, so this stays correct.
    eshkol_matmul_f64(A, B, C, M, K, N);
}

extern "C" void eshkol_batch_matmul_f64(const double*, const double*, double*,
                                        int64_t, int64_t, int64_t, int64_t);

/** @brief Runtime batched-matmul entry point used by generated code: GPU
 *         absent, so dispatches straight to the CPU batched f64 path. */
void eshkol_batch_matmul_dispatch(const double* a, const double* b, double* c,
                                  int64_t batch, int64_t M, int64_t K, int64_t N,
                                  int32_t /*dtype*/) {
    // No GPU — CPU batched f64 path (dtype ignored; storage already
    // precision-reduced for the logical dtype).
    eshkol_batch_matmul_f64(a, b, c, batch, M, K, N);
}

// ===== Backward Pass GPU Stubs =====

/** @brief Stub conv2d input-gradient backward pass: unimplemented on this
 *         platform (no GPU), always fails. Callers must fall back to a CPU
 *         AD path elsewhere. */
int eshkol_gpu_conv2d_backward_input_f64(
    EshkolGPUBuffer*, EshkolGPUBuffer*, EshkolGPUBuffer*,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) {
    return -1; // No GPU
}

/** @brief Stub conv2d kernel-gradient backward pass: unimplemented on this
 *         platform (no GPU), always fails. */
int eshkol_gpu_conv2d_backward_kernel_f64(
    EshkolGPUBuffer*, EshkolGPUBuffer*, EshkolGPUBuffer*,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) {
    return -1;
}

/** @brief Stub batch-norm backward pass: unimplemented on this platform (no
 *         GPU), always fails. */
int eshkol_gpu_batchnorm_backward_f64(
    EshkolGPUBuffer*, EshkolGPUBuffer*, EshkolGPUBuffer*,
    EshkolGPUBuffer*, EshkolGPUBuffer*, EshkolGPUBuffer*,
    EshkolGPUBuffer*, EshkolGPUBuffer*, uint64_t, uint64_t) {
    return -1;
}

/** @brief Stub layer-norm backward pass: unimplemented on this platform (no
 *         GPU), always fails. */
int eshkol_gpu_layernorm_backward_f64(
    EshkolGPUBuffer*, EshkolGPUBuffer*, EshkolGPUBuffer*,
    EshkolGPUBuffer*, EshkolGPUBuffer*, EshkolGPUBuffer*,
    uint64_t, uint64_t) {
    return -1;
}
