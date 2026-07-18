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
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>
#include <utility>

// Forward declarations for real CUDA kernel launchers (in gpu_cuda_kernels.cu)
/** Launch the elementwise unary/binary op kernel (add/sub/mul/.../sigmoid,
 *  selected by `op`) over `n` device-resident doubles; returns 0 on success. */
extern "C" int cuda_launch_elementwise_f64(const double* a, const double* b, double* out,
                                            int64_t n, int op, void* stream);
/** Launch the full-array reduction kernel (sum/mean/prod/min/max, selected by
 *  `op`) over `n` device-resident doubles, writing the scalar result to `out`. */
extern "C" int cuda_launch_reduce_f64(const double* in, double* out, int64_t n, int op,
                                       void* stream);
/** Launch the axis-reduction kernel, reducing the `rank`-dimensional tensor
 *  `in` (shape `dims`) along `axis` with reduction op `op` into `out`. */
extern "C" int cuda_launch_reduce_axis_f64(const double* in, double* out,
                                            uint64_t rank, const uint64_t* dims,
                                            uint64_t axis, int op, uint64_t out_size,
                                            void* stream);
/** Launch the 2D matrix transpose kernel: `in` (rows x cols) -> `out` (cols x rows). */
extern "C" int cuda_launch_transpose_f64(const double* in, double* out,
                                          uint64_t rows, uint64_t cols, void* stream);
/** Launch the softmax kernel, normalizing each of `num_slices` contiguous
 *  runs of `slice_len` doubles independently. */
extern "C" int cuda_launch_softmax_f64(const double* in, double* out,
                                        uint64_t num_slices, uint64_t slice_len,
                                        void* stream);
/** Launch the normalization kernel (batch/layer-norm forward), applying
 *  scale `gamma`/shift `beta` with variance floor `epsilon` per slice. */
extern "C" int cuda_launch_normalize_f64(const double* in, double* out,
                                          uint64_t num_slices, uint64_t slice_len,
                                          double gamma, double beta, double epsilon,
                                          void* stream);

// INT8-Ozaki f64 GEMM device kernels (gpu_cuda_kernels.cu). Split the f64
// operands into per-row/col-scaled int8 slices, then reconstruct the f64 result
// from the diagonal-fused int32 tensor-core GEMM outputs. See
// cuda_matmul_ozaki_int8_f64 below for the cuBLAS orchestration.
/** Per-row max of A (row-major MxK), then split A into `s` int8 slices (natural
 *  layout == col-major KxM A^T). Writes rmax[M] and slices[s*M*K]. */
extern "C" int cuda_ozaki_split_a_f64(const double* A, uint64_t M, uint64_t K, int s,
                                      double* rmax, int8_t* slices, void* stream);
/** Per-col max of B (row-major KxN), then split B into `s` int8 slices stored
 *  transposed to col-major KxN. Writes cmax[N] and slices[s*K*N]. */
extern "C" int cuda_ozaki_split_b_f64(const double* B, uint64_t K, uint64_t N, int s,
                                      double* cmax, int8_t* slices, void* stream);
/** Reconstruct C (row-major MxN) from `nbuf` int32 diagonal buffers with
 *  host-supplied per-buffer weights, scaling each entry by rmax[i]*cmax[j]. */
extern "C" int cuda_ozaki_reconstruct_f64(double* C, const int32_t* Gall,
                                          const double* h_weights, int nbuf,
                                          const double* rmax, const double* cmax,
                                          uint64_t M, uint64_t N, void* stream);
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

static constexpr uint32_t CUDA_FLAG_WRAPPED_HOST = 1u << 0;
static constexpr uint32_t CUDA_FLAG_COPY_BACK = 1u << 1;
static constexpr uint32_t CUDA_FLAG_HOST_REGISTERED = 1u << 2;

/** @brief Initialize CUDA: verify at least one device is present, select
 *         device 0, and create the shared stream and cuBLAS handle used by
 *         all subsequent CUDA operations. Returns 0 on success, -1 on any
 *         failure (cleaning up any partially-created resources). */
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

/** @brief Destroy the shared cuBLAS handle and CUDA stream, if created. */
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

/** @brief Allocate a CUDA buffer of `size_bytes` per `mem_type`: unified
 *         (cudaMallocManaged, host_ptr==device_ptr), device-only
 *         (cudaMalloc, no host_ptr), or host-pinned (cudaMallocHost, no
 *         device_ptr); any other type defaults to unified memory. */
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

/** @brief Free a CUDA buffer, first flushing a pending copy-back to the
 *         original host pointer (CUDA_FLAG_COPY_BACK case from
 *         cuda_wrap_host's fallback path) if one is pending, then releasing
 *         the underlying CUDA allocation appropriate to its memory type
 *         (unregistering a wrapped-pinned host pointer rather than freeing it). */
static void cuda_free(EshkolGPUBuffer* buffer) {
    if ((buffer->flags & CUDA_FLAG_COPY_BACK) && buffer->backend_data &&
        buffer->host_ptr && buffer->host_ptr != buffer->backend_data) {
        cudaStreamSynchronize(g_cuda_stream);
        memcpy(buffer->backend_data, buffer->host_ptr, buffer->size_bytes);
    }

    switch (buffer->mem_type) {
        case ESHKOL_MEM_HOST_PINNED:
            if ((buffer->flags & CUDA_FLAG_HOST_REGISTERED) && buffer->host_ptr) {
                cudaHostUnregister(buffer->host_ptr);
            } else {
                if (buffer->host_ptr) cudaFreeHost(buffer->host_ptr);
                if (buffer->device_ptr && buffer->device_ptr != buffer->host_ptr) {
                    cudaFree(buffer->device_ptr);
                }
            }
            break;
        default:
            if (buffer->device_ptr) cudaFree(buffer->device_ptr);
            break;
    }
}

/** @brief Synchronize a CUDA buffer per `direction`: for unified memory,
 *         waits on the stream and, if wrapping an external host pointer
 *         with pending copy-back, mirrors data to/from that pointer; for a
 *         registered-pinned wrap, just waits on the stream; for
 *         host-pinned buffers with a separate device pointer, issues an
 *         async memcpy in the requested direction and waits. */
static int cuda_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    if (buffer->mem_type == ESHKOL_MEM_UNIFIED) {
        cudaStreamSynchronize(g_cuda_stream);
        if ((buffer->flags & CUDA_FLAG_COPY_BACK) && buffer->backend_data &&
            buffer->host_ptr && buffer->host_ptr != buffer->backend_data &&
            (direction == ESHKOL_SYNC_DEVICE_TO_HOST ||
             direction == ESHKOL_SYNC_BIDIRECTIONAL)) {
            memcpy(buffer->backend_data, buffer->host_ptr, buffer->size_bytes);
        }
        if ((buffer->flags & CUDA_FLAG_COPY_BACK) && buffer->backend_data &&
            buffer->host_ptr && buffer->host_ptr != buffer->backend_data &&
            (direction == ESHKOL_SYNC_HOST_TO_DEVICE ||
             direction == ESHKOL_SYNC_BIDIRECTIONAL)) {
            memcpy(buffer->host_ptr, buffer->backend_data, buffer->size_bytes);
        }
        return 0;
    }

    if ((buffer->flags & CUDA_FLAG_HOST_REGISTERED) && buffer->mem_type == ESHKOL_MEM_HOST_PINNED) {
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

// ============================================================================
// INT8 tensor-core Ozaki f64 GEMM (opt-in) — ESHKOL_CUDA_F64_KERNEL=ozaki-int8
// ============================================================================
// Recover FP64-accurate C = A*B from the INT8 (IMMA) tensor cores, which run
// ~500x faster than the crippled native FP64 pipeline on consumer/prosumer
// NVIDIA GPUs. Measured on an RTX 3090 (sm_86): 4.74 TFLOP/s-eq at full f64
// accuracy (normwise err 2.7e-15) = 8.8x native cublasDgemm; up to 16.6x at
// ~1e-11. On an RTX PRO 6000 Blackwell: ~30 TFLOP/s (20x native f64).
//
// Default OFF: the f64 GPU matmul path stays cublasDgemm unless
// ESHKOL_CUDA_F64_KERNEL=ozaki-int8 is set. The accuracy/throughput knob is
// ESHKOL_OZAKI_CUDA_T (default 6 = full f64; T=4 ~1e-11 and ~2x faster). This
// mirrors the Metal Ozaki-II env handling (ESHKOL_SF64_KERNEL / ESHKOL_OZAKI_*)
// in gpu_memory.mm. See docs/breakdown/GPU_ACCELERATION.md and gpu_cuda_kernels.cu.

static bool g_cuda_ozaki_enabled = false;
static bool g_cuda_ozaki_checked = false;
static int  g_cuda_ozaki_T = 6;

/** @brief Parse (once, cached) the INT8-Ozaki env selection: enabled iff
 *         ESHKOL_CUDA_F64_KERNEL=ozaki-int8; the accuracy knob T comes from
 *         ESHKOL_OZAKI_CUDA_T (default 6, clamped to [1,8] with a loud note on
 *         an out-of-range request, mirroring #307's loud-clamp philosophy). */
static void cuda_ozaki_check_env(void) {
    if (g_cuda_ozaki_checked) return;
    const char* k = std::getenv("ESHKOL_CUDA_F64_KERNEL");
    g_cuda_ozaki_enabled = (k && strcmp(k, "ozaki-int8") == 0);
    int T = 6;
    if (const char* t = std::getenv("ESHKOL_OZAKI_CUDA_T")) {
        int v = atoi(t);
        if (v >= 1 && v <= 8) T = v;
        else fprintf(stderr,
            "[GPU] ESHKOL_OZAKI_CUDA_T=%d out of range [1,8] (T=6 is full f64, "
            "T=4 ~1e-11); using default 6\n", v);
    }
    g_cuda_ozaki_T = T;
    g_cuda_ozaki_checked = true;
    if (g_cuda_ozaki_enabled)
        eshkol_info("CUDA: INT8-Ozaki f64 GEMM enabled (T=%d)", g_cuda_ozaki_T);
}

/** @brief INT8 tensor-core Ozaki f64 GEMM: C = A*B (row-major MxK, KxN, MxN) via
 *         per-row/col-scaled 7-bit integer slicing, INT8->INT32 cublasGemmEx on
 *         the fast TN/IMMA path, and diagonal-fused FP64 reconstruction. All
 *         operands are device pointers (A,B read; C written). T is the accuracy
 *         knob (kept slice-pair diagonal); s=T+1 slices/operand participate.
 *
 *         Returns 0 on success. Returns -1 (caller falls back to cublasDgemm) on
 *         any allocation/cuBLAS failure, or when K exceeds the int32-exact bound
 *         (~133,000) — the latter emits a one-time loud stderr note, since K-panel
 *         splitting is not implemented. The INT32 accumulation is exact; the only
 *         error source is dropping slice-pairs with p+q>T. */
static int cuda_matmul_ozaki_int8_f64(const double* dA, const double* dB, double* dC,
                                      uint64_t M, uint64_t K, uint64_t N, int T) {
    if (!g_cublas_handle || !dA || !dB || !dC) return -1;
    if (M == 0 || K == 0 || N == 0) return -1;
    // cuBLAS int GEMM dims/leading dims are int; reject sizes that would truncate.
    if (M > (uint64_t)0x7fffffff || K > (uint64_t)0x7fffffff || N > (uint64_t)0x7fffffff)
        return -1;

    // INT32-exactness guard: one INT8 GEMM accumulates |G| <= K*127^2 = K*16129,
    // exact in int32 only while K*16129 < 2^31 -> K < 133,151. Beyond that a
    // single pair already overflows and K-panel splitting is not implemented, so
    // fall back LOUDLY to cublasDgemm (mirrors #307's loud-clamp philosophy).
    static std::atomic<bool> warned_k{false};
    const uint64_t K_MAX_INT32_EXACT = 133000;  // conservative; K*16129 < 2^31
    if (K > K_MAX_INT32_EXACT) {
        if (!warned_k.exchange(true))
            fprintf(stderr,
                "[GPU] INT8-Ozaki: K=%llu exceeds the int32-exact bound %llu; "
                "falling back to cublasDgemm for this and future GEMMs at this K "
                "(unset ESHKOL_CUDA_F64_KERNEL or reduce K)\n",
                (unsigned long long)K, (unsigned long long)K_MAX_INT32_EXACT);
        return -1;
    }

    const int s = T + 1;  // slice indices 0..T participate (pairs with p+q<=T)

    // int32-exact cap: max same-weight pairs per diagonal buffer so that the
    // beta=1 accumulation count*K*16129 stays < 2^31.
    long long cap = (long long)(2147483647.0 / ((double)K * 16129.0));
    if (cap < 1) return -1;  // defensive; already covered by the K guard above

    // Plan the diagonal-fused buffers: pairs (p,q) with p+q=d share the weight
    // 128^-(d+2). A diagonal with more than `cap` pairs is split across multiple
    // same-weight int32 buffers (provably int32-exact at any N). Mirrors the
    // measured Blackwell fused reconstruction (ozaki_fused2.cu).
    struct Buf { double w; std::vector<std::pair<int,int>> pairs; };
    std::vector<Buf> bufs;
    try {
        for (int d = 0; d <= T; d++) {
            std::vector<std::pair<int,int>> ps;
            for (int p = 0; p < s; p++) { int q = d - p; if (q >= 0 && q < s) ps.emplace_back(p, q); }
            double w = pow(2.0, -7.0 * (d + 2));
            for (size_t off = 0; off < ps.size(); off += (size_t)cap) {
                Buf b; b.w = w;
                for (size_t t2 = off; t2 < ps.size() && t2 < off + (size_t)cap; t2++)
                    b.pairs.push_back(ps[t2]);
                bufs.push_back(std::move(b));
            }
        }
    } catch (...) { return -1; }
    const int nbuf = (int)bufs.size();

    const size_t totA = (size_t)M * K, totB = (size_t)K * N, totC = (size_t)M * N;

    int8_t*  dAs   = nullptr;
    int8_t*  dBs   = nullptr;
    int32_t* dGall = nullptr;
    double*  dr    = nullptr;
    double*  dc    = nullptr;
    int rc = -1;
    do {
        if (cudaMalloc(&dAs,   (size_t)s * totA * sizeof(int8_t))  != cudaSuccess) break;
        if (cudaMalloc(&dBs,   (size_t)s * totB * sizeof(int8_t))  != cudaSuccess) break;
        if (cudaMalloc(&dGall, (size_t)nbuf * totC * sizeof(int32_t)) != cudaSuccess) break;
        if (cudaMalloc(&dr,    (size_t)M * sizeof(double))         != cudaSuccess) break;
        if (cudaMalloc(&dc,    (size_t)N * sizeof(double))         != cudaSuccess) break;

        if (cuda_ozaki_split_a_f64(dA, M, K, s, dr, dAs, g_cuda_stream) != 0) break;
        if (cuda_ozaki_split_b_f64(dB, K, N, s, dc, dBs, g_cuda_stream) != 0) break;

        // Per-pair INT8->INT32 GEMMs. TN layout (OP_T on the B-slice, OP_N on the
        // A-slice) is MANDATORY for the fast IMMA path on sm_86. Diagonal fusion:
        // all pairs of one buffer accumulate in int32 via beta=1 (stays on the
        // tensor path, exact). alpha/beta are int32 for CUBLAS_COMPUTE_32I.
        const int ialpha = 1;
        bool gemm_ok = true;
        for (int b = 0; b < nbuf && gemm_ok; b++) {
            int32_t* Gd = dGall + (size_t)b * totC;
            bool first = true;
            for (const auto& pq : bufs[b].pairs) {
                const int ibeta = first ? 0 : 1;
                first = false;
                const int8_t* Ap = dAs + (size_t)pq.first  * totA;  // col-major KxM (A_p^T bytes)
                const int8_t* Bq = dBs + (size_t)pq.second * totB;  // col-major KxN (transposed)
                cublasStatus_t st = cublasGemmEx(
                    g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)N, (int)M, (int)K,
                    &ialpha,
                    Bq, CUDA_R_8I, (int)K,
                    Ap, CUDA_R_8I, (int)K,
                    &ibeta,
                    Gd, CUDA_R_32I, (int)N,
                    CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS) { gemm_ok = false; break; }
            }
        }
        if (!gemm_ok) break;

        std::vector<double> hw;
        try { hw.resize((size_t)nbuf); } catch (...) { break; }
        for (int b = 0; b < nbuf; b++) hw[b] = bufs[b].w;
        if (cuda_ozaki_reconstruct_f64(dC, dGall, hw.data(), nbuf, dr, dc, M, N,
                                       g_cuda_stream) != 0) break;

        rc = 0;
    } while (0);

    if (dAs)   cudaFree(dAs);
    if (dBs)   cudaFree(dBs);
    if (dGall) cudaFree(dGall);
    if (dr)    cudaFree(dr);
    if (dc)    cudaFree(dc);
    return rc;
}

/** @brief Double-precision matmul via cuBLAS DGEMM, exploiting the
 *         column-major/row-major duality (computing C^T = B*A in cuBLAS
 *         terms yields row-major C = A*B without an explicit transpose).
 *         When ESHKOL_CUDA_F64_KERNEL=ozaki-int8 is set, the INT8 tensor-core
 *         Ozaki path is tried first and cublasDgemm is the fallback. */
static int cuda_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                            uint64_t M, uint64_t K, uint64_t N) {
    cuda_ozaki_check_env();
    if (g_cuda_ozaki_enabled) {
        GPU_LOG("matmul %llux%llu @ %llux%llu -> CUDA INT8-Ozaki (T=%d)",
                (unsigned long long)M, (unsigned long long)K,
                (unsigned long long)K, (unsigned long long)N, g_cuda_ozaki_T);
        int rc = cuda_matmul_ozaki_int8_f64(
            (const double*)A->device_ptr, (const double*)B->device_ptr,
            (double*)C->device_ptr, M, K, N, g_cuda_ozaki_T);
        if (rc == 0) {
            cudaStreamSynchronize(g_cuda_stream);
            return 0;
        }
        // Any failure / K-guard -> fall through to cublasDgemm below.
        GPU_LOG("matmul %llux%llu @ %llux%llu -> INT8-Ozaki fell back to cublasDgemm",
                (unsigned long long)M, (unsigned long long)K,
                (unsigned long long)K, (unsigned long long)N);
    }

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

/** @brief Single-precision matmul via cuBLAS SGEMM, same column/row-major trick as cuda_matmul_f64. */
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

// f16 cuBLAS GemmEx (tensor cores) — the GPU-LLM fast path (ESH-0021).
// Tensor storage is f64 bit patterns even for f16-dtype tensors, so we convert
// f64 -> half on the host (CUDA __float2half is host-callable), run GemmEx with
// 16F operands and 32F accumulate (the standard LLM combo), then convert the
// f16 result back to f64. Mirrors cuda_matmul_f32's column-major swap: row-major
// C = A·B is computed as cuBLAS column-major C^T(N×M) = B^T·A^T.
// Returns 0 on success; any failure returns -1 so the caller falls back to f64.
static int cuda_matmul_f16_from_f64(const double* A, const double* B, double* C,
                                    uint64_t M, uint64_t K, uint64_t N) {
    if (!g_cublas_handle || !A || !B || !C) return -1;
    // P1: cuBLAS takes the dims and leading dimensions as int; reject sizes that
    // would silently truncate on the (int) casts below (wrong-shape GEMM).
    if (M > (uint64_t)0x7fffffff || K > (uint64_t)0x7fffffff || N > (uint64_t)0x7fffffff) return -1;
    const size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;

    // P1: these host-side half buffers can throw std::bad_alloc for large GEMMs.
    // This function is reached across an extern "C" boundary, where an uncaught
    // C++ exception is undefined behavior (std::terminate) instead of the
    // documented "-1 → CPU f64 fallback". Catch and return -1.
    std::vector<__half> hA, hB, hC;
    try {
        hA.resize(aN); hB.resize(bN); hC.resize(cN);
    } catch (...) { return -1; }
    for (size_t i = 0; i < aN; i++) hA[i] = __float2half((float)A[i]);
    for (size_t i = 0; i < bN; i++) hB[i] = __float2half((float)B[i]);

    __half *dA = nullptr, *dB = nullptr, *dC = nullptr;
    int rc = -1;
    do {
        if (cudaMalloc(&dA, aN * sizeof(__half)) != cudaSuccess) break;
        if (cudaMalloc(&dB, bN * sizeof(__half)) != cudaSuccess) break;
        if (cudaMalloc(&dC, cN * sizeof(__half)) != cudaSuccess) break;
        if (cudaMemcpy(dA, hA.data(), aN * sizeof(__half), cudaMemcpyHostToDevice) != cudaSuccess) break;
        if (cudaMemcpy(dB, hB.data(), bN * sizeof(__half), cudaMemcpyHostToDevice) != cudaSuccess) break;

        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(
            g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            (int)N, (int)M, (int)K,
            &alpha,
            dB, CUDA_R_16F, (int)N,
            dA, CUDA_R_16F, (int)K,
            &beta,
            dC, CUDA_R_16F, (int)N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) break;
        if (cudaStreamSynchronize(g_cuda_stream) != cudaSuccess) break;
        if (cudaMemcpy(hC.data(), dC, cN * sizeof(__half), cudaMemcpyDeviceToHost) != cudaSuccess) break;

        for (size_t i = 0; i < cN; i++) C[i] = (double)__half2float(hC[i]);
        rc = 0;
    } while (0);

    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    return rc;
}

// Batched f16 cuBLAS GemmEx (tensor cores) — the MoE multi-expert win
// (ESH-0024). Computes `batch` independent row-major GEMMs C[b]=A[b]·B[b] in a
// single cublasGemmStridedBatchedEx launch (16F operands, 32F accumulate),
// beating one launch per expert. Same f64<->half conversion + column-major swap
// as cuda_matmul_f16_from_f64. Returns 0 on success, -1 on failure (caller
// falls back to the CPU batched f64 path).
static int cuda_batch_matmul_f16_from_f64(const double* a, const double* b, double* c,
                                          int64_t batch, int64_t M, int64_t K, int64_t N) {
    if (!g_cublas_handle || !a || !b || !c || batch <= 0) return -1;
    // P1: cuBLAS strided-batched takes int dims; reject sizes that truncate.
    if (M > 0x7fffffff || K > 0x7fffffff || N > 0x7fffffff || batch > 0x7fffffff) return -1;
    const size_t aN = (size_t)batch * M * K;
    const size_t bN = (size_t)batch * K * N;
    const size_t cN = (size_t)batch * M * N;

    // P1: catch std::bad_alloc — uncaught across the extern "C" boundary is UB.
    std::vector<__half> hA, hB, hC;
    try {
        hA.resize(aN); hB.resize(bN); hC.resize(cN);
    } catch (...) { return -1; }
    for (size_t i = 0; i < aN; i++) hA[i] = __float2half((float)a[i]);
    for (size_t i = 0; i < bN; i++) hB[i] = __float2half((float)b[i]);

    __half *dA = nullptr, *dB = nullptr, *dC = nullptr;
    int rc = -1;
    do {
        if (cudaMalloc(&dA, aN * sizeof(__half)) != cudaSuccess) break;
        if (cudaMalloc(&dB, bN * sizeof(__half)) != cudaSuccess) break;
        if (cudaMalloc(&dC, cN * sizeof(__half)) != cudaSuccess) break;
        if (cudaMemcpy(dA, hA.data(), aN * sizeof(__half), cudaMemcpyHostToDevice) != cudaSuccess) break;
        if (cudaMemcpy(dB, hB.data(), bN * sizeof(__half), cudaMemcpyHostToDevice) != cudaSuccess) break;

        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t st = cublasGemmStridedBatchedEx(
            g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            (int)N, (int)M, (int)K,
            &alpha,
            dB, CUDA_R_16F, (int)N, (long long)(K * N),
            dA, CUDA_R_16F, (int)K, (long long)(M * K),
            &beta,
            dC, CUDA_R_16F, (int)N, (long long)(M * N),
            (int)batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) break;
        if (cudaStreamSynchronize(g_cuda_stream) != cudaSuccess) break;
        if (cudaMemcpy(hC.data(), dC, cN * sizeof(__half), cudaMemcpyDeviceToHost) != cudaSuccess) break;

        for (size_t i = 0; i < cN; i++) c[i] = (double)__half2float(hC[i]);
        rc = 0;
    } while (0);

    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    return rc;
}

/** @brief Wrap an existing host pointer for zero-copy GPU access: tries
 *         `cudaHostRegister` (page-locking it in place and mapping a device
 *         pointer to it); if that fails (e.g. the memory isn't
 *         page-alignable, such as stack or arbitrary heap allocations),
 *         falls back to allocating separate unified memory, copying the
 *         data in, and marking it for copy-back to the original pointer on
 *         sync/free. */
static int cuda_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out) {
    cudaError_t err = cudaHostRegister(host_ptr, size_bytes, cudaHostRegisterMapped);
    if (err == cudaSuccess) {
        void* device_ptr = nullptr;
        err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
        if (err == cudaSuccess && device_ptr) {
            out->host_ptr = host_ptr;
            out->device_ptr = device_ptr;
            out->size_bytes = size_bytes;
            out->mem_type = ESHKOL_MEM_HOST_PINNED;
            out->backend = ESHKOL_GPU_CUDA;
            out->flags = CUDA_FLAG_WRAPPED_HOST | CUDA_FLAG_HOST_REGISTERED;
            out->backend_data = nullptr;
            return 0;
        }
        cudaHostUnregister(host_ptr);
    }

    // Fallback: allocate managed memory and keep the caller's host pointer
    // for synchronization/copy-back. This path covers stack/unaligned memory
    // and hosts where mapped pinned registration is not available.
    int result = cuda_alloc(size_bytes, ESHKOL_MEM_UNIFIED, out);
    if (result != 0) return result;
    memcpy(out->host_ptr, host_ptr, size_bytes);
    out->flags = CUDA_FLAG_WRAPPED_HOST | CUDA_FLAG_COPY_BACK;
    out->backend_data = host_ptr;

    return 0;
}

#endif // ESHKOL_GPU_CUDA_AVAILABLE

// ============================================================================
// GPU Dispatch Logging
// ============================================================================

static bool g_gpu_verbose = false;
static bool g_gpu_verbose_checked = false;

/** @brief True if `ESHKOL_GPU_VERBOSE` is set in the environment (checked
 *         once and cached), enabling GPU_LOG diagnostic output. */
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

/** @brief Thread-safe, idempotent GPU init (double-checked locking):
 *         applies an optional `ESHKOL_GPU_THRESHOLD` env override, then
 *         attempts CUDA initialization. Returns 1 if a GPU backend is
 *         active, 0 if only CPU is available (still a "successful" init). */
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

/** @brief Thread-safe GPU shutdown: tears down the active backend (if CUDA)
 *         and resets init state so a later eshkol_gpu_init() can re-run. */
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

/** @brief The currently active GPU backend (ESHKOL_GPU_NONE if not initialized or no GPU found). */
EshkolGPUBackend eshkol_gpu_get_backend(void) {
    return g_active_backend;
}

/** @brief Human-readable name for a GPU backend enum value. */
const char* eshkol_gpu_backend_name(EshkolGPUBackend backend) {
    switch (backend) {
        case ESHKOL_GPU_NONE: return "CPU (no GPU)";
        case ESHKOL_GPU_METAL: return "Apple Metal (not available on Linux)";
        case ESHKOL_GPU_CUDA: return "NVIDIA CUDA";
        case ESHKOL_GPU_VULKAN: return "Vulkan";
    }
    return "Unknown";
}

/** @brief True if `backend` is the currently active backend (this build
 *         supports only a single active backend at a time, so this is
 *         equivalent to `backend == eshkol_gpu_get_backend()`). */
int eshkol_gpu_backend_available(EshkolGPUBackend backend) {
    return (g_active_backend == backend) ? 1 : 0;
}

/** @brief True if the active backend is CUDA (which supports f64 on all modern GPUs). */
int eshkol_gpu_supports_f64(void) {
    // CUDA supports f64 on all modern GPUs
    return (g_active_backend == ESHKOL_GPU_CUDA) ? 1 : 0;
}

/** @brief Alias for eshkol_gpu_supports_f64(). */
int eshkol_gpu_has_fp64(void) {
    return eshkol_gpu_supports_f64();
}

/** @brief Allocate a GPU (or, absent a GPU, plain host) buffer of
 *         `size_bytes`, dispatching to cuda_alloc() when CUDA is active or
 *         falling back to malloc() otherwise. */
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

/** @brief Allocate a GPU buffer with `size_bytes` rounded up to a multiple
 *         of `alignment` (must be a power of two), then delegates to
 *         eshkol_gpu_alloc(). */
int eshkol_gpu_alloc_aligned(size_t size_bytes, size_t alignment,
                              EshkolMemoryType mem_type, EshkolGPUBuffer* out_buffer) {
    size_t aligned_size = (size_bytes + alignment - 1) & ~(alignment - 1);
    return eshkol_gpu_alloc(aligned_size, mem_type, out_buffer);
}

/** @brief Free a GPU (or host) buffer: dispatches to cuda_free() for CUDA
 *         buffers, otherwise frees the host pointer directly unless it was
 *         a non-owning wrap (flag bit 1). Zeroes the descriptor either way. */
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

/** @brief Wrap an existing host pointer as a GPU-accessible buffer:
 *         dispatches to cuda_wrap_host() for CUDA, or on CPU just wraps the
 *         pointer directly with device_ptr aliasing host_ptr. */
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

/** @brief Synchronize a buffer's host/device data per `direction`;
 *         dispatches to cuda_sync() for CUDA buffers, no-op otherwise. */
int eshkol_gpu_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    if (!buffer) return -1;

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_CUDA) {
        return cuda_sync(buffer, direction);
    }
#endif

    return 0;
}

/** @brief Async sync variant: `stream` is currently unused (all ops share
 *         the global CUDA stream), so this just delegates to eshkol_gpu_sync(). */
int eshkol_gpu_sync_async(EshkolGPUBuffer* buffer, EshkolSyncDirection direction, void* stream) {
    (void)stream;
    return eshkol_gpu_sync(buffer, direction);
}

/** @brief Block until all outstanding work on the shared CUDA stream
 *         completes (no-op if CUDA isn't the active backend). `buffer` is
 *         unused — this waits on the global stream, not per-buffer. */
void eshkol_gpu_wait(EshkolGPUBuffer* buffer) {
    (void)buffer;
#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && g_cuda_stream) {
        cudaStreamSynchronize(g_cuda_stream);
    }
#endif
}

/** @brief Double-precision matmul: dispatches to cuda_matmul_f64() on CUDA
 *         (cuBLAS DGEMM), else falls back to the CPU BLAS/SIMD
 *         eshkol_matmul_f64(). Logs the chosen path when GPU_LOG is verbose. */
int eshkol_gpu_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        GPU_LOG("matmul %llux%llu @ %llux%llu -> CUDA cuBLAS",
                (unsigned long long)M, (unsigned long long)K,
                (unsigned long long)K, (unsigned long long)N);
        return cuda_matmul_f64(A, B, C, M, K, N);
    }
#endif

    GPU_LOG("matmul %llux%llu @ %llux%llu -> CPU",
            (unsigned long long)M, (unsigned long long)K,
            (unsigned long long)K, (unsigned long long)N);
    // CPU fallback
    extern void eshkol_matmul_f64(const double*, const double*, double*, uint64_t, uint64_t, uint64_t);
    eshkol_matmul_f64((const double*)A->host_ptr, (const double*)B->host_ptr,
                      (double*)C->host_ptr, M, K, N);
    return 0;
}

/** @brief Single-precision matmul: dispatches to cuda_matmul_f32() on CUDA
 *         (cuBLAS SGEMM), else falls back to a naive CPU scalar triple loop. */
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

/** @brief Set the minimum-element GPU dispatch threshold. */
void eshkol_gpu_set_threshold(size_t threshold) {
    g_gpu_threshold = threshold;
}

/** @brief Get the current GPU dispatch element-count threshold. */
size_t eshkol_gpu_get_threshold(void) {
    return g_gpu_threshold;
}

/** @brief True if a GPU backend is active and `num_elements` meets the dispatch threshold. */
int eshkol_gpu_should_use(size_t num_elements) {
    return (g_active_backend != ESHKOL_GPU_NONE && num_elements >= g_gpu_threshold) ? 1 : 0;
}

/** @brief Runtime matmul entry point used by generated code: lazily
 *         initializes the GPU on first call, routes f16/bf16-dtype operands
 *         through the tensor-core GemmEx fast path regardless of size
 *         (bypassing the element threshold), and otherwise wraps the host
 *         buffers and dispatches through eshkol_gpu_matmul_f64() when the
 *         threshold is met, falling back to CPU BLAS/SIMD f64 matmul on any
 *         failure or when GPU isn't warranted. */
void eshkol_matmul_dispatch(const double* A, const double* B, double* C,
                             uint64_t M, uint64_t K, uint64_t N, int32_t dtype) {
    size_t num_elements = M * N;

    // Lazy GPU init: the (matmul ...) codegen path reaches this dispatcher
    // directly, but unlike the BLAS dispatcher (eshkol_matmul_f64) it never
    // primed the backend. Without this, g_active_backend stays ESHKOL_GPU_NONE
    // forever and both the f16/bf16 tensor-core fast path and the f64 GPU path
    // are unreachable from the language. eshkol_gpu_init() is idempotent
    // (double-checked lock), so this is a one-time cost on the first matmul.
    if (g_active_backend == ESHKOL_GPU_NONE) {
        eshkol_gpu_init();
    }

    // GPU-LLM fast path (ESH-0021): f16/bf16-dtype operands go through cuBLAS
    // GemmEx (16F operands, 32F accumulate, tensor cores). Tensor-core GEMM
    // wins even at modest sizes, so this bypasses the f64 element threshold.
    // dtype codes: 2=f16, 3=bf16 (see eshkol_tensor_dtype_t).
    if ((dtype == 2 || dtype == 3) && g_active_backend == ESHKOL_GPU_CUDA) {
        if (cuda_matmul_f16_from_f64(A, B, C, M, K, N) == 0) {
            return;
        }
        // GemmEx failed — fall through to the f64 path below.
    }

    if (eshkol_gpu_should_use(num_elements)) {
        EshkolGPUBuffer buf_a{};
        EshkolGPUBuffer buf_b{};
        EshkolGPUBuffer buf_c{};
        const int wrapped_a = (eshkol_gpu_wrap_host((void*)A, M * K * sizeof(double), &buf_a) == 0);
        const int wrapped_b = (eshkol_gpu_wrap_host((void*)B, K * N * sizeof(double), &buf_b) == 0);
        const int wrapped_c = (eshkol_gpu_wrap_host((void*)C, M * N * sizeof(double), &buf_c) == 0);

        if (wrapped_a && wrapped_b && wrapped_c) {
            if (eshkol_gpu_matmul_f64(&buf_a, &buf_b, &buf_c, M, K, N) == 0) {
                eshkol_gpu_sync(&buf_c, ESHKOL_SYNC_DEVICE_TO_HOST);
                eshkol_gpu_free(&buf_a);
                eshkol_gpu_free(&buf_b);
                eshkol_gpu_free(&buf_c);
                return;
            }
        }

        if (wrapped_a) eshkol_gpu_free(&buf_a);
        if (wrapped_b) eshkol_gpu_free(&buf_b);
        if (wrapped_c) eshkol_gpu_free(&buf_c);
    }

    extern void eshkol_matmul_f64(const double*, const double*, double*, uint64_t, uint64_t, uint64_t);
    eshkol_matmul_f64(A, B, C, M, K, N);
}

extern "C" void eshkol_batch_matmul_f64(const double*, const double*, double*,
                                        int64_t, int64_t, int64_t, int64_t);

// Batched matmul dispatch (ESH-0024). f16/bf16 + CUDA → strided GemmEx (one
// launch for all `batch` GEMMs); else the CPU batched f64 path. dtype codes:
// 2=f16, 3=bf16.
void eshkol_batch_matmul_dispatch(const double* a, const double* b, double* c,
                                  int64_t batch, int64_t M, int64_t K, int64_t N,
                                  int32_t dtype) {
    if ((dtype == 2 || dtype == 3) && g_active_backend == ESHKOL_GPU_CUDA) {
        if (cuda_batch_matmul_f16_from_f64(a, b, c, batch, M, K, N) == 0) {
            return;
        }
        // GemmStridedBatched failed — fall through to the f64 path.
    }
    eshkol_batch_matmul_f64(a, b, c, batch, M, K, N);
}

/** @brief Elementwise unary/binary op: launches the CUDA elementwise kernel
 *         when both operand buffers have device pointers, falling back to a
 *         CPU scalar loop over `n` elements if the kernel launch fails or
 *         no GPU is active. */
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

/** @brief Full reduction to a scalar: launches the CUDA reduce kernel when
 *         device pointers are available, falling back to a CPU scalar loop
 *         (sum/mean/prod/min/max) if the kernel launch fails or no GPU is active. */
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

/** @brief Axis reduction: launches the CUDA axis-reduce kernel when device
 *         pointers are available, falling back to a CPU implementation
 *         (computing inner/outer strides around `axis`) if the kernel
 *         launch fails or no GPU is active. */
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

/** @brief 2-D transpose: launches the CUDA transpose kernel when device
 *         pointers are available, falling back to a CPU nested loop if the
 *         kernel launch fails or no GPU is active. */
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

/** @brief Numerically-stable softmax over `num_slices` slices of length
 *         `slice_len`: launches the CUDA softmax kernel when device
 *         pointers are available, falling back to a CPU implementation
 *         (max-subtraction then normalized exp) if the kernel launch fails
 *         or no GPU is active. */
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

/** @brief Layer-normalize over `num_slices` slices of length `slice_len`:
 *         launches the CUDA normalize kernel when device pointers are
 *         available, falling back to a CPU implementation (mean/variance
 *         then scale+shift) if the kernel launch fails or no GPU is active. */
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

// ===== Backward Pass GPU — CUDA kernel dispatch =====

// Forward declarations of CUDA kernel launchers (in gpu_cuda_kernels.cu)
/** Launch the CUDA conv2D input-gradient kernel; computes grad_input from
 *  grad_out and the kernel weights. */
extern int cuda_conv2d_backward_input_f64(
    const double*, const double*, double*,
    int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
/** Launch the CUDA conv2D kernel-gradient kernel; computes grad_kernel from
 *  grad_out and the saved input. */
extern int cuda_conv2d_backward_kernel_f64(
    const double*, const double*, double*,
    int, int, int, int, int, int, int, int, int, int, int, cudaStream_t);
/** Launch the CUDA batch-normalization backward kernel; computes
 *  grad_input/grad_gamma/grad_beta from the saved forward statistics. */
extern int cuda_batchnorm_backward_f64(
    const double*, const double*, const double*, const double*, const double*,
    double*, double*, double*, int, int, cudaStream_t);
/** Launch the CUDA layer-normalization backward kernel; computes grad_input
 *  from the saved forward statistics. */
extern int cuda_layernorm_backward_f64(
    const double*, const double*, const double*, const double*, const double*,
    double*, int, int, cudaStream_t);

/** @brief Conv2D input-gradient backward pass on CUDA (dispatches to
 *         cuda_conv2d_backward_input_f64() in gpu_cuda_kernels.cu). Requires
 *         CUDA to be the active backend and all buffers to have device
 *         pointers; returns -1 otherwise (no CPU fallback here). */
int eshkol_gpu_conv2d_backward_input_f64(
    EshkolGPUBuffer* grad_out, EshkolGPUBuffer* kernel_weights,
    EshkolGPUBuffer* grad_input,
    uint64_t in_h, uint64_t in_w, uint64_t k_h, uint64_t k_w,
    uint64_t out_h, uint64_t out_w,
    uint64_t stride_h, uint64_t stride_w,
    uint64_t channels_in, uint64_t channels_out, uint64_t batch_size) {
    if (g_active_backend != ESHKOL_GPU_CUDA) return -1;
    if (!grad_out || !kernel_weights || !grad_input ||  // P1: guard null buffers/device ptrs
        !grad_out->device_ptr || !kernel_weights->device_ptr || !grad_input->device_ptr) return -1;
    return cuda_conv2d_backward_input_f64(
        (const double*)grad_out->device_ptr,
        (const double*)kernel_weights->device_ptr,
        (double*)grad_input->device_ptr,
        (int)in_h, (int)in_w, (int)k_h, (int)k_w,
        (int)out_h, (int)out_w, (int)stride_h, (int)stride_w,
        (int)channels_in, (int)channels_out, (int)batch_size,
        g_cuda_stream);
}

/** @brief Conv2D kernel-gradient backward pass on CUDA (dispatches to
 *         cuda_conv2d_backward_kernel_f64()). Requires CUDA active and all
 *         buffers device-resident; returns -1 otherwise. */
int eshkol_gpu_conv2d_backward_kernel_f64(
    EshkolGPUBuffer* grad_out, EshkolGPUBuffer* saved_input,
    EshkolGPUBuffer* grad_kernel,
    uint64_t in_h, uint64_t in_w, uint64_t k_h, uint64_t k_w,
    uint64_t out_h, uint64_t out_w,
    uint64_t stride_h, uint64_t stride_w,
    uint64_t channels_in, uint64_t channels_out, uint64_t batch_size) {
    if (g_active_backend != ESHKOL_GPU_CUDA) return -1;
    if (!grad_out || !saved_input || !grad_kernel ||  // P1: guard null buffers/device ptrs
        !grad_out->device_ptr || !saved_input->device_ptr || !grad_kernel->device_ptr) return -1;
    return cuda_conv2d_backward_kernel_f64(
        (const double*)grad_out->device_ptr,
        (const double*)saved_input->device_ptr,
        (double*)grad_kernel->device_ptr,
        (int)in_h, (int)in_w, (int)k_h, (int)k_w,
        (int)out_h, (int)out_w, (int)stride_h, (int)stride_w,
        (int)channels_in, (int)channels_out, (int)batch_size,
        g_cuda_stream);
}

/** @brief Batch-normalization backward pass on CUDA (dispatches to
 *         cuda_batchnorm_backward_f64()), producing grad_input, grad_gamma,
 *         and grad_beta. Requires CUDA active and all buffers
 *         device-resident; returns -1 otherwise. */
int eshkol_gpu_batchnorm_backward_f64(
    EshkolGPUBuffer* grad_out,
    EshkolGPUBuffer* saved_input, EshkolGPUBuffer* saved_mean,
    EshkolGPUBuffer* saved_inv_std, EshkolGPUBuffer* saved_gamma,
    EshkolGPUBuffer* grad_input, EshkolGPUBuffer* grad_gamma,
    EshkolGPUBuffer* grad_beta,
    uint64_t batch_size, uint64_t feature_size) {
    if (g_active_backend != ESHKOL_GPU_CUDA) return -1;
    if (!grad_out || !saved_input || !saved_mean || !saved_inv_std || !saved_gamma ||
        !grad_input || !grad_gamma || !grad_beta ||  // P1: guard null buffers/device ptrs
        !grad_out->device_ptr || !saved_input->device_ptr || !saved_mean->device_ptr ||
        !saved_inv_std->device_ptr || !saved_gamma->device_ptr || !grad_input->device_ptr ||
        !grad_gamma->device_ptr || !grad_beta->device_ptr) return -1;
    return cuda_batchnorm_backward_f64(
        (const double*)grad_out->device_ptr,
        (const double*)saved_input->device_ptr,
        (const double*)saved_mean->device_ptr,
        (const double*)saved_inv_std->device_ptr,
        (const double*)saved_gamma->device_ptr,
        (double*)grad_input->device_ptr,
        (double*)grad_gamma->device_ptr,
        (double*)grad_beta->device_ptr,
        (int)batch_size, (int)feature_size,
        g_cuda_stream);
}

/** @brief Layer-normalization backward pass on CUDA (dispatches to
 *         cuda_layernorm_backward_f64()), producing grad_input. Requires
 *         CUDA active and all buffers device-resident; returns -1 otherwise. */
int eshkol_gpu_layernorm_backward_f64(
    EshkolGPUBuffer* grad_out,
    EshkolGPUBuffer* saved_input, EshkolGPUBuffer* saved_mean,
    EshkolGPUBuffer* saved_inv_std, EshkolGPUBuffer* saved_gamma,
    EshkolGPUBuffer* grad_input,
    uint64_t num_samples, uint64_t feature_size) {
    if (g_active_backend != ESHKOL_GPU_CUDA) return -1;
    if (!grad_out || !saved_input || !saved_mean || !saved_inv_std || !saved_gamma ||
        !grad_input ||  // P1: guard null buffers/device ptrs
        !grad_out->device_ptr || !saved_input->device_ptr || !saved_mean->device_ptr ||
        !saved_inv_std->device_ptr || !saved_gamma->device_ptr || !grad_input->device_ptr) return -1;
    return cuda_layernorm_backward_f64(
        (const double*)grad_out->device_ptr,
        (const double*)saved_input->device_ptr,
        (const double*)saved_mean->device_ptr,
        (const double*)saved_inv_std->device_ptr,
        (const double*)saved_gamma->device_ptr,
        (double*)grad_input->device_ptr,
        (int)num_samples, (int)feature_size,
        g_cuda_stream);
}

} // extern "C"
