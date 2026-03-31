/*
 * Cross-Platform GPU Memory Implementation for Eshkol
 *
 * Provides unified GPU acceleration across:
 * - Metal (macOS/iOS) with unified memory
 * - CUDA (Linux/Windows) with discrete GPU memory
 * - Fallback to CPU when no GPU available
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
#include <dispatch/dispatch.h>

// ============================================================================
// Platform Detection
// ============================================================================

// Metal (macOS/iOS with Apple GPU)
#if defined(__APPLE__) && defined(ESHKOL_GPU_METAL_ENABLED)
#define ESHKOL_GPU_METAL_AVAILABLE 1
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

// CUDA (NVIDIA GPU)
#if defined(ESHKOL_GPU_CUDA_ENABLED)
#define ESHKOL_GPU_CUDA_AVAILABLE 1
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

// ============================================================================
// Global State
// ============================================================================

// GPU threshold (elements) - default 100K
size_t g_gpu_threshold = 100000;

// Active backend
static EshkolGPUBackend g_active_backend = ESHKOL_GPU_NONE;
static std::atomic<bool> g_gpu_initialized{false};
static std::mutex g_gpu_init_mutex;

// ============================================================================
// Metal Backend State
// ============================================================================

#if ESHKOL_GPU_METAL_AVAILABLE

static id<MTLDevice> g_metal_device = nil;
static id<MTLCommandQueue> g_metal_queue = nil;
static id<MTLComputePipelineState> g_matmul_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_matmul_sf64_v2_pipeline = nil;
static id<MTLComputePipelineState> g_matmul_df64_pipeline = nil;
static id<MTLComputePipelineState> g_matmul_f32_pipeline = nil;
static id<MTLComputePipelineState> g_matmul_f32_simd_pipeline = nil;
static id<MTLComputePipelineState> g_elementwise_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_reduce_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_transpose_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_reduce_axis_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_softmax_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_normalize_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_word_swap_sf64_pipeline = nil;
static id<MTLComputePipelineState> g_convert_f64_to_f32_pipeline = nil;
static id<MTLComputePipelineState> g_convert_f32_to_f64_pipeline = nil;
static id<MTLComputePipelineState> g_convert_f64_to_df64_pipeline = nil;
static id<MTLComputePipelineState> g_convert_df64_to_f64_pipeline = nil;
static id<MTLComputePipelineState> g_matmul_df64_pure_pipeline = nil;
static id<MTLComputePipelineState> g_matmul_fp24_pipeline = nil;
static id<MTLComputePipelineState> g_matmul_fp53_pipeline = nil;
static id<MTLComputePipelineState> g_matmul_f32_simd_128_pipeline = nil;
// g_matmul_ozaki_pipeline removed — Ozaki-I replaced by Ozaki-II (g_matmul_ozaki_gemm_pipeline)
static id<MTLLibrary> g_metal_library = nil;
static bool g_metal_unified_memory = false;

// GPU precision tier: 0=exact (sf64), 1=high (df64), 2=fast (f32), 3=ml (fp24)
static int g_metal_precision_tier = 0;

// ============================================================================
// Unified Kernel Configuration System
// ============================================================================
// Every kernel parameter is computed from hardware capabilities via an
// occupancy-aware scoring function. No hardcoded tile sizes.

enum KernelType {
    KERNEL_SF64 = 0,
    KERNEL_DF64,
    KERNEL_F32_SIMD,
    KERNEL_F32_SIMD_128,
    KERNEL_FP24,
    KERNEL_FP53,
    KERNEL_OZAKI,
    KERNEL_COUNT
};

struct HardwareProfile {
    uint32_t max_tg_mem;         // maxThreadgroupMemoryLength
    uint32_t max_threads_per_tg; // conservative estimate (varies per pipeline)
    uint32_t thread_exec_width;  // 32 on all Apple Silicon
    int      gpu_family;         // 7=M1, 8=M2, 9=M3, 10=M4+
    uint64_t device_mem;
};

struct KernelConfig {
    uint32_t bm, bn, bk;           // Tile dimensions
    uint32_t threads;               // Threads per threadgroup
    uint32_t sa_stride, sb_stride;  // Bank-conflict-free strides
    uint32_t shared_bytes;          // Total threadgroup memory
    // Simdgroup kernels (f32_simd, f32_simd_128):
    uint32_t wm, wn, tm, tn;
    // Scalar-thread kernels (df64, fp24, fp53, sf64):
    uint32_t tg, tt;
    float    occupancy_score;
};

static HardwareProfile g_hw;
static KernelConfig g_cfg_sf64;
static KernelConfig g_cfg_df64;
static KernelConfig g_cfg_f32s;       // f32_simd (small/default)
static KernelConfig g_cfg_f32s_128;   // f32_simd_128 (large)
static KernelConfig g_cfg_fp24;
static KernelConfig g_cfg_fp53;
// g_cfg_ozaki removed — Ozaki-II reuses g_cfg_f32s (same simdgroup tile sizes)

// ============================================================================
// Ozaki-II CRT Constants — Chinese Remainder Theorem exact DGEMM
// ============================================================================
// 49 pairwise coprime moduli ≤ 256 (Ozaki et al., Eq. 10 of 2602.02549v1)
static const int OZAKI_MODULI[49] = {
    256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
    223, 217, 211, 199, 197, 193, 191, 181, 179, 173,
    167, 163, 157, 151, 149, 139, 137, 131, 127, 113,
    109, 107, 103, 101,  97,  89,  83,  79,  73,  71,
     67,  61,  59,  53,  47,  43,  41,  37,  29
};

struct OzakiCRTConstants {
    double P1, P2;             // P ≈ P1 + P2 (double-double)
    double Pinv;               // ≈ 1/P
    double sl1[49], sl2[49];   // sℓ1, sℓ2 for CRT accumulation
    int    q[49];              // modular multiplicative inverses
    int    num_moduli;
    double log2P;              // log2(P) for scaling
};

static OzakiCRTConstants g_ozaki_crt;
static int g_ozaki_num_moduli = 16;
static id<MTLComputePipelineState> g_matmul_ozaki_gemm_pipeline = nil;

// Cumulative log2 of moduli products: OZAKI_LOG2_CUMUL[n] = sum(log2(OZAKI_MODULI[0..n-1]))
// Used for adaptive N selection without overflow.
static double OZAKI_LOG2_CUMUL[50];  // [0]=0, [1]=log2(256), [2]=log2(256*255), ...

// CRT constants cache indexed by N (avoids recomputation when adaptive N is stable)
// Index 0-49, where index N holds constants for N moduli. initialized = false sentinel.
static OzakiCRTConstants g_ozaki_crt_cache[50];
static bool g_ozaki_crt_cached[50] = {};

// Extended GCD for modular inverse computation
static int64_t extended_gcd(int64_t a, int64_t b, int64_t& x, int64_t& y) {
    if (a == 0) { x = 0; y = 1; return b; }
    int64_t x1, y1;
    int64_t g = extended_gcd(b % a, a, x1, y1);
    x = y1 - (b / a) * x1;
    y = x1;
    return g;
}

// Compute modular multiplicative inverse: a^{-1} mod m
static int mod_inverse(int64_t a, int64_t m) {
    int64_t x, y;
    extended_gcd(a % m, m, x, y);
    return (int)((x % m + m) % m);
}

// Precompute CRT constants for N moduli using __int128 for exact P.
// ARM64 note: long double == double (64-bit), so we use __int128 subtraction
// for the double-double splits instead of long double arithmetic.
static void precompute_ozaki_constants(int N, OzakiCRTConstants* crt) {
    crt->num_moduli = N;

    // Compute P = product of first N moduli using __int128 for exactness
    __int128 P = 1;
    for (int l = 0; l < N; l++) {
        P *= OZAKI_MODULI[l];
    }

    // P as double-double: P1 + P2 where P1 = round(P) and P2 = P - P1 (exact via __int128)
    crt->P1 = (double)P;  // rounds to nearest double (53 significant bits)
    __int128 P1_int = (__int128)crt->P1;
    __int128 P2_int = P - P1_int;           // exact residual (~72 bits)
    crt->P2 = (double)P2_int;               // captures high 53 of those ~72 bits
    crt->Pinv = 1.0 / (double)P;
    crt->log2P = log2((double)P);

    for (int l = 0; l < N; l++) {
        int p = OZAKI_MODULI[l];
        __int128 Pp = P / p;  // P/p_l
        int64_t Pp_mod_p = (int64_t)(Pp % p);
        crt->q[l] = mod_inverse(Pp_mod_p, p);

        // CRT coefficient: s_l = (P/p_l) * q_l — huge integer (~125 bits)
        __int128 s = Pp * (__int128)crt->q[l];

        // Double-double split via Dekker + __int128 exact residual.
        // sl1 retains the upper (53 - bits_to_remove) bits of s so that
        // sl1 * W_l is exact in f64 (W_l has ≤ 12 bits for K ≤ 4096).
        int bits_to_remove = 8 + (int)ceil(log2((double)N));
        double s_d = (double)s;
        double factor = (double)(1LL << (bits_to_remove + 1)) + 1.0;
        double t = factor * s_d;
        crt->sl1[l] = t - (t - s_d);  // high part, bottom bits_to_remove bits zeroed

        // Exact residual via __int128: sl2 = s - sl1
        __int128 sl1_int = (__int128)crt->sl1[l];
        __int128 sl2_int = s - sl1_int;
        crt->sl2[l] = (double)sl2_int;  // captures high 53 of ~(72+bits_to_remove) bits
    }
}

// Get CRT constants for a given N, caching to avoid recomputation
static const OzakiCRTConstants& get_ozaki_crt(int N) {
    if (N < 2) N = 2;
    if (N > 49) N = 49;
    if (!g_ozaki_crt_cached[N]) {
        precompute_ozaki_constants(N, &g_ozaki_crt_cache[N]);
        g_ozaki_crt_cached[N] = true;
    }
    return g_ozaki_crt_cache[N];
}

// Initialize the cumulative log2 table (called once at init)
static void init_ozaki_log2_table() {
    OZAKI_LOG2_CUMUL[0] = 0.0;
    for (int i = 0; i < 49; i++) {
        OZAKI_LOG2_CUMUL[i + 1] = OZAKI_LOG2_CUMUL[i] + log2((double)OZAKI_MODULI[i]);
    }
}

// Adaptive N selection: compute minimum number of moduli for correct CRT reconstruction.
//
// CRT uniqueness condition: |C'_ij| < P/2 for all i,j, where C' = A' * B'.
// After scaling with E(N), mu_i = E - floor(log2(r_i)), where r_i = max_h|A_ih|:
//   |A'_ij| ≤ r_i * 2^(E - floor(log2(r_i))) = 2^(E + frac(log2(r_i)))
// So max|A'_ij| = 2^(E + frac_A) where frac_A = max_i frac(log2(r_i)).
//
// For exact powers of 2 (ones, identity): frac=0, |A'|=2^E (tightest).
// For arbitrary data: frac→1, |A'|<2^(E+1) (worst case adds 2 bits to P needed).
//
// CRT condition: log2(K) + 2*E + frac_A + frac_B < log2(P) - 1
static int ozaki_compute_adaptive_N(const double* A, const double* B,
                                     uint64_t M, uint64_t K, uint64_t N_cols,
                                     int max_moduli) {
    // Compute infinity norms of input matrices (unscaled)
    double A_inf = 0.0, B_inf = 0.0;

    // Parallel row max-abs for A
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    std::vector<double> row_maxes(M, 0.0);
    double* rm_p = row_maxes.data();
    dispatch_apply(M, q, ^(size_t i) {
        double mx = 0.0;
        for (size_t j = 0; j < K; j++) {
            double v = fabs(A[i * K + j]);
            if (v > mx) mx = v;
        }
        rm_p[i] = mx;
    });
    // Parallel col max-abs for B
    std::vector<double> col_maxes(N_cols, 0.0);
    double* cm_p = col_maxes.data();
    dispatch_apply(N_cols, q, ^(size_t j) {
        double mx = 0.0;
        for (size_t h = 0; h < K; h++) {
            double v = fabs(B[h * N_cols + j]);
            if (v > mx) mx = v;
        }
        cm_p[j] = mx;
    });

    // Compute max fractional parts of log2(norms)
    double frac_A = 0.0, frac_B = 0.0;
    bool all_zero_A = true, all_zero_B = true;
    for (size_t i = 0; i < M; i++) {
        if (row_maxes[i] > 0.0) {
            all_zero_A = false;
            double lg = log2(row_maxes[i]);
            double frac = lg - floor(lg);
            if (frac > frac_A) frac_A = frac;
        }
    }
    for (size_t j = 0; j < N_cols; j++) {
        if (col_maxes[j] > 0.0) {
            all_zero_B = false;
            double lg = log2(col_maxes[j]);
            double frac = lg - floor(lg);
            if (frac > frac_B) frac_B = frac;
        }
    }

    if (all_zero_A || all_zero_B) return 2;

    double log2K = log2((double)K);

    // For each candidate N, compute E_max accounting for actual data frac.
    // E_max = floor((log2P - 1 - log2K - frac_A - frac_B) / 2)
    // CRT is satisfied iff E_max >= 0 (there exists a valid scaling exponent).
    //
    // Note: E controls precision — E=0 means integer truncation, E=52 means full f64.
    // For ones-matrices (frac=0), even E=1 gives exact results.
    // For general data, lower E means more truncation error in the input scaling,
    // but the CRT reconstruction is still exact for the truncated values.
    for (int N = 2; N <= max_moduli; N++) {
        double log2P = OZAKI_LOG2_CUMUL[N];
        // E_max: largest scaling exponent where CRT uniqueness holds
        double E_max = floor((log2P - 1.0 - log2K - frac_A - frac_B) / 2.0);
        if (E_max >= 0.0) {
            return N;
        }
    }

    return max_moduli;
}

// Backward-compat aliases (used by existing dispatch functions during transition)
static uint32_t g_sf64_tg = 8;
static uint32_t g_sf64_tt = 4;
static uint32_t g_sf64_blk = 32;
static uint32_t g_sf64_tile_k = 8;
static uint32_t g_fp_bm = 64, g_fp_bn = 64, g_fp_bk = 32, g_fp_tt = 4, g_fp_threads = 256;
static uint32_t g_fp53_bm = 64, g_fp53_bn = 64, g_fp53_bk = 8, g_fp53_tt = 4;
static uint32_t g_fp53_threads = 256, g_fp53_sb_stride = 65;

// --- Shared memory cost computation per kernel type ---
static uint32_t compute_shared_bytes(KernelType type, uint32_t bm, uint32_t bn,
                                      uint32_t bk, uint32_t sa_stride, uint32_t sb_stride) {
    switch (type) {
        case KERNEL_F32_SIMD: {
            uint32_t sa = bm * sa_stride;
            uint32_t sb = bk * sb_stride;
            uint32_t edge = 4 * 64;  // 4 SG × 64 floats
            return (sa + sb + edge) * 4;
        }
        case KERNEL_F32_SIMD_128: {
            uint32_t sa = bm * sa_stride;
            uint32_t sb = bk * sb_stride;
            // Edge scratch reuses sA region, no extra alloc
            return (sa + sb) * 4;
        }
        case KERNEL_DF64: {
            // float2 = 8 bytes per element
            uint32_t sa = bm * sa_stride;
            uint32_t sb = bk * sb_stride;
            return (sa + sb) * 8;
        }
        case KERNEL_FP24: {
            // uint mantissa = 4 bytes, int64 accum is in registers
            uint32_t sa = bm * (bk + 2);
            uint32_t sb = bk * (bn + 2);
            return (sa + sb) * 4;
        }
        case KERNEL_FP53: {
            // uint2 = 8 bytes. sA = BM*(BK+1), sB = BK*(BN+1), + sign arrays
            uint32_t sa = bm * (bk + 1);
            uint32_t sb = bk * (bn + 1);
            uint32_t sign_a = ((bm * bk + 31) / 32);
            uint32_t sign_b = ((bk * bn + 31) / 32);
            return (sa + sb) * 8 + (sign_a + sign_b) * 4;
        }
        case KERNEL_SF64: {
            uint32_t blk = bm;
            uint32_t sa = blk * sa_stride;
            uint32_t sb = bk * blk;
            return (sa + sb) * 8;
        }
        case KERNEL_OZAKI: {
            // 6 A slices + 6 B slices + scratch (max 4 SGs × 64 floats)
            uint32_t nslice = 6;
            uint32_t sa_slice = bm * sa_stride;
            uint32_t sb_slice = bk * sb_stride;
            uint32_t scratch = 4 * 64;  // conservative: max 4 simdgroups
            return (nslice * (sa_slice + sb_slice) + scratch) * 4;
        }
        default: return UINT32_MAX;
    }
}

// --- Occupancy-aware scoring function ---
// Balances occupancy (latency hiding), arithmetic intensity, tile coverage,
// and BK amortization (fewer barriers/loop iterations with larger K-blocks).
static float score_config(KernelType type, uint32_t bm, uint32_t bn, uint32_t bk,
                           uint32_t threads, uint32_t shared_bytes, const HardwareProfile& hw) {
    if (shared_bytes > hw.max_tg_mem || shared_bytes == 0) return -1.0f;
    if (threads > hw.max_threads_per_tg || threads == 0) return -1.0f;

    uint32_t tg_by_mem = hw.max_tg_mem / shared_bytes;
    uint32_t tg_by_threads = hw.max_threads_per_tg / threads;
    uint32_t occupancy = std::min(tg_by_mem, tg_by_threads);
    if (occupancy == 0) return -1.0f;
    if (occupancy > 4) occupancy = 4;

    // Element size for memory bandwidth calculation
    uint32_t elem_bytes = 4;
    if (type == KERNEL_DF64 || type == KERNEL_FP53 || type == KERNEL_SF64) elem_bytes = 8;

    // Arithmetic intensity: FLOPs per byte loaded per K-block
    float bytes_loaded = (float)(bm * bk + bk * bn) * (float)elem_bytes;
    float flops = 2.0f * (float)bm * (float)bn * (float)bk;
    float ai = (bytes_loaded > 0) ? flops / bytes_loaded : 0;

    // BK amortization: larger K-blocks mean fewer barriers and loop iterations.
    // Linear in BK because barrier cost is per-iteration and each iteration
    // processes BK elements. This correctly favors BK=16 over BK=8 when
    // the occupancy difference is modest.
    float bk_amort = (float)bk;

    return (float)occupancy * ai * sqrtf((float)(bm * bn)) * bk_amort;
}

// --- Per-kernel parameter search functions ---

static void search_f32_simd_config(KernelConfig& cfg, const HardwareProfile& hw, bool large_tile) {
    const uint32_t wm_range[] = {1, 2, 4};
    const uint32_t wn_range[] = {1, 2, 4};
    const uint32_t tm_range[] = {2, 4};
    const uint32_t tn_range[] = {2, 4};
    const uint32_t bk_range[] = {8, 16, 32};

    KernelConfig best = {};
    float best_score = -1;

    for (auto wm : wm_range) for (auto wn : wn_range) {
        uint32_t threads = wm * wn * 32;
        if (threads > hw.max_threads_per_tg) continue;
        for (auto tm : tm_range) for (auto tn : tn_range) {
            uint32_t bm = wm * tm * 8;
            uint32_t bn = wn * tn * 8;
            // Filter by target size
            if (large_tile && bm * bn <= 64 * 64) continue;
            if (!large_tile && bm * bn > 64 * 64) continue;

            for (auto bk : bk_range) {
                uint32_t sa_pad = (bk % 8 == 0) ? 4 : 1;
                uint32_t sb_pad = (bn % 8 == 0) ? 4 : 1;
                uint32_t sa_stride = bk + sa_pad;
                uint32_t sb_stride = bn + sb_pad;

                KernelType kt = large_tile ? KERNEL_F32_SIMD_128 : KERNEL_F32_SIMD;
                uint32_t shared = compute_shared_bytes(kt, bm, bn, bk, sa_stride, sb_stride);
                float score = score_config(kt, bm, bn, bk, threads, shared, hw);
                if (score > best_score) {
                    best_score = score;
                    best = {bm, bn, bk, threads, sa_stride, sb_stride, shared,
                            wm, wn, tm, tn, 0, 0, score};
                }
            }
        }
    }
    cfg = best;
}

static void search_df64_config(KernelConfig& cfg, const HardwareProfile& hw) {
    const uint32_t tg_range[] = {8, 16};
    const uint32_t tt_range[] = {2, 4};   // TT≤4: df64 acc[TT][TT] = 8*TT² bytes/thread
    const uint32_t bk_range[] = {8, 16, 32};

    KernelConfig best = {};
    float best_score = -1;

    for (auto tg : tg_range) {
        uint32_t threads = tg * tg;
        if (threads > hw.max_threads_per_tg) continue;
        for (auto tt : tt_range) {
            uint32_t bm = tg * tt;
            uint32_t bn = bm;
            for (auto bk : bk_range) {
                uint32_t sa_stride = bk + 2;
                uint32_t sb_stride = bn;
                uint32_t shared = compute_shared_bytes(KERNEL_DF64, bm, bn, bk, sa_stride, sb_stride);
                float score = score_config(KERNEL_DF64, bm, bn, bk, threads, shared, hw);
                if (score > best_score) {
                    best_score = score;
                    best = {bm, bn, bk, threads, sa_stride, sb_stride, shared,
                            0, 0, 0, 0, tg, tt, score};
                }
            }
        }
    }
    cfg = best;
}

static void search_fp_config(KernelType type, KernelConfig& cfg, const HardwareProfile& hw) {
    const uint32_t tt_range[] = {2, 4};
    const uint32_t bk_range[] = {8, 16, 32};

    KernelConfig best = {};
    float best_score = -1;

    for (auto tt : tt_range) {
        // FP kernels use TG×TG threads, TT per thread → BM = sqrt(threads)*TT
        for (uint32_t tg_dim = 8; tg_dim <= 16; tg_dim += 8) {
            uint32_t threads = tg_dim * tg_dim;
            if (threads > hw.max_threads_per_tg) continue;
            uint32_t bm = tg_dim * tt;
            uint32_t bn = bm;
            for (auto bk : bk_range) {
                // fp53 BK must be ≤ 16 (128-bit accumulation overflow)
                if (type == KERNEL_FP53 && bk > 16) continue;
                uint32_t sa_stride = (type == KERNEL_FP53) ? bk + 1 : bk + 2;
                uint32_t sb_stride = bn + ((type == KERNEL_FP53) ? 1 : 2);
                uint32_t shared = compute_shared_bytes(type, bm, bn, bk, sa_stride, sb_stride);
                float score = score_config(type, bm, bn, bk, threads, shared, hw);
                if (score > best_score) {
                    best_score = score;
                    best = {bm, bn, bk, threads, sa_stride, sb_stride, shared,
                            0, 0, 0, 0, tg_dim, tt, score};
                }
            }
        }
    }
    cfg = best;
}

static void search_sf64_config(KernelConfig& cfg, const HardwareProfile& hw) {
    const uint32_t tg_range[] = {8, 16};
    const uint32_t tt_range[] = {2, 4};
    const uint32_t bk_range[] = {4, 8, 16};

    KernelConfig best = {};
    float best_score = -1;

    for (auto tg : tg_range) {
        uint32_t threads = tg * tg;
        if (threads > hw.max_threads_per_tg) continue;
        for (auto tt : tt_range) {
            uint32_t bm = tg * tt;
            uint32_t bn = bm;
            for (auto bk : bk_range) {
                uint32_t sa_stride = bk + 2;
                uint32_t sb_stride = bn;  // sf64 sB has no padding
                uint32_t shared = compute_shared_bytes(KERNEL_SF64, bm, bn, bk, sa_stride, sb_stride);
                float score = score_config(KERNEL_SF64, bm, bn, bk, threads, shared, hw);
                if (score > best_score) {
                    best_score = score;
                    best = {bm, bn, bk, threads, sa_stride, sb_stride, shared,
                            0, 0, 0, 0, tg, tt, score};
                }
            }
        }
    }
    cfg = best;
}

// search_ozaki_config removed — Ozaki-II reuses g_cfg_f32s (same tile sizes)

// --- Compute all kernel configs from hardware profile ---
static void compute_all_configs(const HardwareProfile& hw) {
    search_sf64_config(g_cfg_sf64, hw);
    search_df64_config(g_cfg_df64, hw);
    search_f32_simd_config(g_cfg_f32s, hw, false);      // small tiles (≤64×64)
    search_f32_simd_config(g_cfg_f32s_128, hw, true);    // large tiles (>64×64)
    search_fp_config(KERNEL_FP24, g_cfg_fp24, hw);
    search_fp_config(KERNEL_FP53, g_cfg_fp53, hw);
    // Ozaki-II reuses g_cfg_f32s — no separate config search needed

    // Sync backward-compat globals
    g_sf64_tg = g_cfg_sf64.tg; g_sf64_tt = g_cfg_sf64.tt;
    g_sf64_tile_k = g_cfg_sf64.bk; g_sf64_blk = g_cfg_sf64.bm;
    g_fp_bm = g_cfg_fp24.bm; g_fp_bn = g_cfg_fp24.bn;
    g_fp_bk = g_cfg_fp24.bk; g_fp_tt = g_cfg_fp24.tt; g_fp_threads = g_cfg_fp24.threads;
    g_fp53_bm = g_cfg_fp53.bm; g_fp53_bn = g_cfg_fp53.bn;
    g_fp53_bk = g_cfg_fp53.bk; g_fp53_tt = g_cfg_fp53.tt;
    g_fp53_threads = g_cfg_fp53.threads; g_fp53_sb_stride = g_cfg_fp53.bn + 1;
}

// --- Adaptive chunk size for row-blocking ---
static uint32_t compute_chunk_m(KernelType type, uint32_t K, uint32_t bm) {
    // Weight factor: heavier kernels need smaller chunks to stay within GPU timeout
    float weight;
    switch (type) {
        case KERNEL_SF64:  weight = 8.0f; break;
        case KERNEL_OZAKI: weight = 3.0f; break;  // lighter than fp53 (simdgroup MMA)
        case KERNEL_FP53:  weight = 4.0f; break;
        case KERNEL_FP24:  weight = 2.0f; break;
        case KERNEL_DF64:  weight = 1.5f; break;
        default:           weight = 1.0f; break;
    }
    // Target: keep GPU time per chunk under ~200ms
    // Base: 4096 rows at K=4096 for weight=1 → scale inversely with K and weight
    uint32_t target = (uint32_t)(4096.0f * 4096.0f / ((float)K * weight));
    if (target < bm) target = bm;
    // Align to tile boundary
    target = ((target + bm - 1) / bm) * bm;
    // Cap
    if (target > 4096) target = 4096;
    return target;
}

// ============================================================================
// Metal Buffer Pool — reuses MTLBuffer objects to reduce allocation overhead
// ============================================================================
// Size-binned pool: rounds requested size up to next power-of-2, reuses buffers
// of that bucket size. Significant win for batched operations (ML training loops).

#include <unordered_map>
#include <vector>

static std::unordered_map<size_t, std::vector<id<MTLBuffer>>> g_buffer_pool;

static size_t pool_bucket(size_t bytes) {
    if (bytes == 0) return 1;
    size_t v = bytes - 1;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16; v |= v >> 32;
    return v + 1;
}

static id<MTLBuffer> pool_alloc(size_t bytes) {
    size_t bucket = pool_bucket(bytes);
    auto it = g_buffer_pool.find(bucket);
    if (it != g_buffer_pool.end() && !it->second.empty()) {
        id<MTLBuffer> buf = it->second.back();
        it->second.pop_back();
        return buf;
    }
    return [g_metal_device newBufferWithLength:bucket
                                       options:MTLResourceStorageModeShared];
}

static constexpr size_t POOL_MAX_PER_BUCKET = 8;

static void pool_release(id<MTLBuffer> buf) {
    if (!buf) return;
    size_t bucket = pool_bucket(buf.length);
    auto& vec = g_buffer_pool[bucket];
    if (vec.size() >= POOL_MAX_PER_BUCKET) {
        return;  // Drop buffer — ARC will free the MTLBuffer
    }
    vec.push_back(buf);
}

static void pool_drain() {
    g_buffer_pool.clear();
}

// --- Build full shader config string from all kernel configs ---
// Centralized: used by initial compilation AND recompilation paths.
static NSString* build_shader_config_string() {
    return [NSString stringWithFormat:
        // sf64
        @"#define ESHKOL_SF64_TG %u\n"
        @"#define ESHKOL_SF64_TT %u\n"
        @"#define ESHKOL_SF64_TILE_K %u\n"
        // fp24
        @"#define FP_BM %u\n"
        @"#define FP_BN %u\n"
        @"#define FP_BK %u\n"
        @"#define FP_TT %u\n"
        @"#define FP_THREADS %u\n"
        // fp53
        @"#define FP53_BM %u\n"
        @"#define FP53_BN %u\n"
        @"#define FP53_BK %u\n"
        @"#define FP53_TT %u\n"
        @"#define FP53_THREADS %u\n"
        @"#define FP53_SB_STRIDE %u\n"
        // df64
        @"#define DF64_BM %u\n"
        @"#define DF64_BN %u\n"
        @"#define DF64_BK %u\n"
        @"#define DF64_TG %u\n"
        @"#define DF64_TT %u\n"
        @"#define DF64_THREADS %u\n"
        // f32_simd
        @"#define F32S_BM %u\n"
        @"#define F32S_BN %u\n"
        @"#define F32S_BK %u\n"
        @"#define F32S_TM %u\n"
        @"#define F32S_TN %u\n"
        @"#define F32S_WM %u\n"
        @"#define F32S_WN %u\n"
        @"#define F32S_THREADS %u\n"
        // f32_simd_128
        @"#define F32S128_BM %u\n"
        @"#define F32S128_BN %u\n"
        @"#define F32S128_BK %u\n"
        @"#define F32S128_TM %u\n"
        @"#define F32S128_TN %u\n"
        @"#define F32S128_WM %u\n"
        @"#define F32S128_WN %u\n"
        @"#define F32S128_THREADS %u\n",
        // Ozaki-I defines removed — Ozaki-II reuses F32S constants
        // sf64 values
        g_cfg_sf64.tg, g_cfg_sf64.tt, g_cfg_sf64.bk,
        // fp24 values
        g_cfg_fp24.bm, g_cfg_fp24.bn, g_cfg_fp24.bk, g_cfg_fp24.tt, g_cfg_fp24.threads,
        // fp53 values
        g_cfg_fp53.bm, g_cfg_fp53.bn, g_cfg_fp53.bk, g_cfg_fp53.tt, g_cfg_fp53.threads,
        g_cfg_fp53.bn + 1,
        // df64 values
        g_cfg_df64.bm, g_cfg_df64.bn, g_cfg_df64.bk, g_cfg_df64.tg, g_cfg_df64.tt, g_cfg_df64.threads,
        // f32_simd values
        g_cfg_f32s.bm, g_cfg_f32s.bn, g_cfg_f32s.bk, g_cfg_f32s.tm, g_cfg_f32s.tn,
        g_cfg_f32s.wm, g_cfg_f32s.wn, g_cfg_f32s.threads,
        // f32_simd_128 values
        g_cfg_f32s_128.bm, g_cfg_f32s_128.bn, g_cfg_f32s_128.bk, g_cfg_f32s_128.tm, g_cfg_f32s_128.tn,
        g_cfg_f32s_128.wm, g_cfg_f32s_128.wn, g_cfg_f32s_128.threads
    ];
}

// ============================================================================
// SoftFloat IEEE 754 f64 Emulation for Metal
// ============================================================================
// Apple Silicon GPUs lack native f64 hardware. We emulate f64 using uint2
// (two 32-bit integers) with full IEEE 754 compliance including:
// - 52-bit mantissa precision (bit-exact with CPU)
// - Proper rounding (round-to-nearest-even)
// - Special value handling (zero, infinity, NaN)
// Based on Berkeley SoftFloat library algorithms.
//
// SINGLE SOURCE OF TRUTH: lib/backend/gpu/metal_softfloat.h
// The shader source below is auto-generated from metal_softfloat.h at build time
// via CMake custom command. Do NOT edit inline — modify metal_softfloat.h instead.
#include "metal_sf64_embedded.inc"


static int metal_init(void) {
    @autoreleasepool {
        g_metal_device = MTLCreateSystemDefaultDevice();
        if (!g_metal_device) return -1;

        g_metal_queue = [g_metal_device newCommandQueue];
        if (!g_metal_queue) {
            g_metal_device = nil;
            return -1;
        }

        // Check for unified memory (Apple Silicon)
        if (@available(macOS 10.15, *)) {
            g_metal_unified_memory = [g_metal_device supportsFamily:MTLGPUFamilyApple1];
        }

        // Detect GPU architecture family
        int gpu_family = 0;
        const char* gpu_gen = "pre-M1";
        if (@available(macOS 12.0, *)) {
            if ([g_metal_device supportsFamily:MTLGPUFamilyApple7]) {
                gpu_family = 7; gpu_gen = "M1 (Apple7)";
            }
        }
        if (@available(macOS 13.0, *)) {
            if ([g_metal_device supportsFamily:MTLGPUFamilyApple8]) {
                gpu_family = 8; gpu_gen = "M2 (Apple8)";
            }
        }
        if (@available(macOS 14.0, *)) {
            if ([g_metal_device supportsFamily:MTLGPUFamilyApple9]) {
                gpu_family = 9; gpu_gen = "M3 (Apple9)";
            }
        }

        // Log GPU info
        NSUInteger max_buf = [g_metal_device maxBufferLength];
        fprintf(stderr, "[GPU] Metal: %s, %s, maxBuffer=%luMB, unified=%s\n",
                [[g_metal_device name] UTF8String], gpu_gen,
                (unsigned long)(max_buf / (1024*1024)),
                g_metal_unified_memory ? "yes" : "no");

        // Populate hardware profile from device capabilities
        g_hw.max_tg_mem = (uint32_t)[g_metal_device maxThreadgroupMemoryLength];
        g_hw.max_threads_per_tg = 1024;  // Conservative; checked per-pipeline later
        g_hw.thread_exec_width = 32;     // All Apple Silicon
        g_hw.gpu_family = gpu_family;
        g_hw.device_mem = [g_metal_device recommendedMaxWorkingSetSize];

        fprintf(stderr, "[GPU] hardware: maxTGMem=%u, maxThreads=%u, family=%d, vram=%lluMB\n",
                g_hw.max_tg_mem, g_hw.max_threads_per_tg, g_hw.gpu_family,
                (unsigned long long)(g_hw.device_mem / (1024*1024)));

        // === Unified adaptive parameter computation ===
        // All kernel parameters computed from hardware via occupancy-aware scoring.
        compute_all_configs(g_hw);

        // === Environment variable overrides (for benchmarking/tuning) ===
        // sf64
        if (const char* v = std::getenv("ESHKOL_SF64_TG")) {
            unsigned val = (unsigned)atoi(v);
            if (val == 8 || val == 16 || val == 32) {
                g_cfg_sf64.tg = val; g_cfg_sf64.bm = val * g_cfg_sf64.tt;
                g_cfg_sf64.bn = g_cfg_sf64.bm;
            }
        }
        if (const char* v = std::getenv("ESHKOL_SF64_TILE_K")) {
            unsigned val = (unsigned)atoi(v);
            if (val == 4 || val == 8 || val == 16 || val == 32) g_cfg_sf64.bk = val;
        }
        // fp24
        if (const char* v = std::getenv("ESHKOL_FP_BK")) {
            int val = atoi(v);
            if (val == 8 || val == 16 || val == 32) g_cfg_fp24.bk = (uint32_t)val;
        }
        // fp53
        if (const char* v = std::getenv("ESHKOL_FP53_BK")) {
            int val = atoi(v);
            if (val == 8 || val == 16) g_cfg_fp53.bk = (uint32_t)val;
        }
        // df64 (NEW)
        if (const char* v = std::getenv("ESHKOL_DF64_TG")) {
            unsigned val = (unsigned)atoi(v);
            if (val == 8 || val == 16) {
                g_cfg_df64.tg = val; g_cfg_df64.threads = val * val;
                g_cfg_df64.bm = val * g_cfg_df64.tt; g_cfg_df64.bn = g_cfg_df64.bm;
            }
        }
        if (const char* v = std::getenv("ESHKOL_DF64_BK")) {
            int val = atoi(v);
            if (val == 8 || val == 16 || val == 32) g_cfg_df64.bk = (uint32_t)val;
        }
        // f32_simd (NEW)
        if (const char* v = std::getenv("ESHKOL_F32S_WM")) {
            unsigned val = (unsigned)atoi(v);
            if (val == 1 || val == 2 || val == 4) {
                g_cfg_f32s.wm = val;
                g_cfg_f32s.bm = val * g_cfg_f32s.tm * 8;
                g_cfg_f32s.threads = val * g_cfg_f32s.wn * 32;
            }
        }
        if (const char* v = std::getenv("ESHKOL_F32S_WN")) {
            unsigned val = (unsigned)atoi(v);
            if (val == 1 || val == 2 || val == 4) {
                g_cfg_f32s.wn = val;
                g_cfg_f32s.bn = val * g_cfg_f32s.tn * 8;
                g_cfg_f32s.threads = g_cfg_f32s.wm * val * 32;
            }
        }
        if (const char* v = std::getenv("ESHKOL_F32S_BK")) {
            int val = atoi(v);
            if (val == 8 || val == 16 || val == 32) g_cfg_f32s.bk = (uint32_t)val;
        }
        // f32_simd_128 (NEW)
        if (const char* v = std::getenv("ESHKOL_F32S128_WM")) {
            unsigned val = (unsigned)atoi(v);
            if (val == 1 || val == 2 || val == 4) {
                g_cfg_f32s_128.wm = val;
                g_cfg_f32s_128.bm = val * g_cfg_f32s_128.tm * 8;
                g_cfg_f32s_128.threads = val * g_cfg_f32s_128.wn * 32;
            }
        }
        if (const char* v = std::getenv("ESHKOL_F32S128_WN")) {
            unsigned val = (unsigned)atoi(v);
            if (val == 1 || val == 2 || val == 4) {
                g_cfg_f32s_128.wn = val;
                g_cfg_f32s_128.bn = val * g_cfg_f32s_128.tn * 8;
                g_cfg_f32s_128.threads = g_cfg_f32s_128.wm * val * 32;
            }
        }
        if (const char* v = std::getenv("ESHKOL_F32S128_BK")) {
            int val = atoi(v);
            if (val == 8 || val == 16 || val == 32) g_cfg_f32s_128.bk = (uint32_t)val;
        }

        // Re-sync backward-compat globals after env overrides
        g_sf64_tg = g_cfg_sf64.tg; g_sf64_tt = g_cfg_sf64.tt;
        g_sf64_tile_k = g_cfg_sf64.bk; g_sf64_blk = g_cfg_sf64.bm;
        g_fp_bm = g_cfg_fp24.bm; g_fp_bn = g_cfg_fp24.bn;
        g_fp_bk = g_cfg_fp24.bk; g_fp_tt = g_cfg_fp24.tt; g_fp_threads = g_cfg_fp24.threads;
        g_fp53_bm = g_cfg_fp53.bm; g_fp53_bn = g_cfg_fp53.bn;
        g_fp53_bk = g_cfg_fp53.bk; g_fp53_tt = g_cfg_fp53.tt;
        g_fp53_threads = g_cfg_fp53.threads; g_fp53_sb_stride = g_cfg_fp53.bn + 1;

        // Log all kernel configs with occupancy scores
        fprintf(stderr, "[GPU] sf64   config: TG=%u TT=%u BK=%u BLK=%u shared=%u score=%.1f\n",
                g_cfg_sf64.tg, g_cfg_sf64.tt, g_cfg_sf64.bk, g_cfg_sf64.bm, g_cfg_sf64.shared_bytes, g_cfg_sf64.occupancy_score);
        fprintf(stderr, "[GPU] df64   config: BM=%u BN=%u BK=%u TG=%u TT=%u threads=%u shared=%u score=%.1f\n",
                g_cfg_df64.bm, g_cfg_df64.bn, g_cfg_df64.bk, g_cfg_df64.tg, g_cfg_df64.tt, g_cfg_df64.threads, g_cfg_df64.shared_bytes, g_cfg_df64.occupancy_score);
        fprintf(stderr, "[GPU] f32s   config: BM=%u BN=%u BK=%u WM=%u WN=%u TM=%u TN=%u threads=%u shared=%u score=%.1f\n",
                g_cfg_f32s.bm, g_cfg_f32s.bn, g_cfg_f32s.bk, g_cfg_f32s.wm, g_cfg_f32s.wn, g_cfg_f32s.tm, g_cfg_f32s.tn, g_cfg_f32s.threads, g_cfg_f32s.shared_bytes, g_cfg_f32s.occupancy_score);
        fprintf(stderr, "[GPU] f32s128 config: BM=%u BN=%u BK=%u WM=%u WN=%u TM=%u TN=%u threads=%u shared=%u score=%.1f\n",
                g_cfg_f32s_128.bm, g_cfg_f32s_128.bn, g_cfg_f32s_128.bk, g_cfg_f32s_128.wm, g_cfg_f32s_128.wn, g_cfg_f32s_128.tm, g_cfg_f32s_128.tn, g_cfg_f32s_128.threads, g_cfg_f32s_128.shared_bytes, g_cfg_f32s_128.occupancy_score);
        fprintf(stderr, "[GPU] fp24   config: BM=%u BN=%u BK=%u TT=%u threads=%u shared=%u score=%.1f\n",
                g_cfg_fp24.bm, g_cfg_fp24.bn, g_cfg_fp24.bk, g_cfg_fp24.tt, g_cfg_fp24.threads, g_cfg_fp24.shared_bytes, g_cfg_fp24.occupancy_score);
        fprintf(stderr, "[GPU] fp53   config: BM=%u BN=%u BK=%u TT=%u threads=%u shared=%u score=%.1f\n",
                g_cfg_fp53.bm, g_cfg_fp53.bn, g_cfg_fp53.bk, g_cfg_fp53.tt, g_cfg_fp53.threads, g_cfg_fp53.shared_bytes, g_cfg_fp53.occupancy_score);

        // Prepend GPU-specific #define directives to shader source (ALL kernels)
        NSString* config = build_shader_config_string();
        NSString* fullSource = [config stringByAppendingString:g_matmul_sf64_source];

        // Compile sf64 (SoftFloat) shader with architecture-specific configuration
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        // Disable fast-math to ensure exact IEEE 754 rounding behavior
        options.fastMathEnabled = NO;

        g_metal_library = [g_metal_device newLibraryWithSource:fullSource
                                                       options:options
                                                         error:&error];
        if (!g_metal_library) {
            eshkol_error("Metal: failed to compile sf64 shader: %s",
                    [[error localizedDescription] UTF8String]);
            g_metal_queue = nil;
            g_metal_device = nil;
            return -1;  // Shader compilation failed — GPU unusable
        }

        id<MTLFunction> func = [g_metal_library newFunctionWithName:@"matmul_sf64"];
        if (!func) {
            eshkol_error("Metal: failed to find matmul_sf64 kernel in compiled library");
            g_metal_library = nil;
            g_metal_queue = nil;
            g_metal_device = nil;
            return -1;
        }

        g_matmul_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:func
                                                                               error:&error];
        if (!g_matmul_sf64_pipeline) {
            eshkol_error("Metal: failed to create sf64 compute pipeline: %s",
                    [[error localizedDescription] UTF8String]);
            g_metal_library = nil;
            g_metal_queue = nil;
            g_metal_device = nil;
            return -1;
        }

        // Log V1 pipeline diagnostics
        fprintf(stderr, "[GPU] sf64 v1: maxTG=%lu, execWidth=%lu, staticTGMem=%lu\n",
            (unsigned long)[g_matmul_sf64_pipeline maxTotalThreadsPerThreadgroup],
            (unsigned long)[g_matmul_sf64_pipeline threadExecutionWidth],
            (unsigned long)[g_matmul_sf64_pipeline staticThreadgroupMemoryLength]);

        // Create v2 (optimized) matmul pipeline — deferred rounding, bitwise-first
        id<MTLFunction> v2_func = [g_metal_library newFunctionWithName:@"matmul_sf64_v2"];
        if (v2_func) {
            g_matmul_sf64_v2_pipeline = [g_metal_device newComputePipelineStateWithFunction:v2_func error:&error];
            if (!g_matmul_sf64_v2_pipeline) {
                eshkol_error("Metal: failed to create sf64_v2 pipeline: %s",
                             [[error localizedDescription] UTF8String]);
            } else {
                fprintf(stderr, "[GPU] sf64 v2: maxTG=%lu, execWidth=%lu, staticTGMem=%lu\n",
                    (unsigned long)[g_matmul_sf64_v2_pipeline maxTotalThreadsPerThreadgroup],
                    (unsigned long)[g_matmul_sf64_v2_pipeline threadExecutionWidth],
                    (unsigned long)[g_matmul_sf64_v2_pipeline staticThreadgroupMemoryLength]);
            }
        }

        // Create df64 (double-float f32 FMA) matmul pipelines
        id<MTLFunction> df64_func = [g_metal_library newFunctionWithName:@"matmul_df64"];
        if (df64_func) {
            g_matmul_df64_pipeline = [g_metal_device newComputePipelineStateWithFunction:df64_func error:&error];
        }

        // Create pure df64 pipeline (pre-converted float2 input, config-driven threads)
        id<MTLFunction> df64_pure_func = [g_metal_library newFunctionWithName:@"matmul_df64_pure"];
        if (df64_pure_func) {
            g_matmul_df64_pure_pipeline = [g_metal_device newComputePipelineStateWithFunction:df64_pure_func error:&error];
            if (g_matmul_df64_pure_pipeline) {
                NSUInteger maxTG = [g_matmul_df64_pure_pipeline maxTotalThreadsPerThreadgroup];
                fprintf(stderr, "[GPU] df64_pure: maxTG=%lu, execWidth=%lu, staticTGMem=%lu\n",
                    (unsigned long)maxTG,
                    (unsigned long)[g_matmul_df64_pure_pipeline threadExecutionWidth],
                    (unsigned long)[g_matmul_df64_pure_pipeline staticThreadgroupMemoryLength]);
                if (maxTG < g_cfg_df64.threads) {
                    fprintf(stderr, "[GPU] df64_pure: maxTG=%lu < %u, re-searching with constraint\n",
                            (unsigned long)maxTG, g_cfg_df64.threads);
                    HardwareProfile hw_reduced = g_hw;
                    hw_reduced.max_threads_per_tg = (uint32_t)maxTG;
                    search_df64_config(g_cfg_df64, hw_reduced);
                    // Recompile with updated config
                    NSString* src = [build_shader_config_string() stringByAppendingString:g_matmul_sf64_source];
                    id<MTLLibrary> df64_lib = [g_metal_device newLibraryWithSource:src options:options error:&error];
                    if (df64_lib) {
                        g_metal_library = df64_lib;
                        id<MTLFunction> df64_refunc = [df64_lib newFunctionWithName:@"matmul_df64_pure"];
                        if (df64_refunc) {
                            g_matmul_df64_pure_pipeline = [g_metal_device newComputePipelineStateWithFunction:df64_refunc error:&error];
                            fprintf(stderr, "[GPU] df64_pure recompiled: maxTG=%lu, TG=%u threads=%u\n",
                                (unsigned long)[g_matmul_df64_pure_pipeline maxTotalThreadsPerThreadgroup],
                                g_cfg_df64.tg, g_cfg_df64.threads);
                        }
                    }
                }
            }
        }

        // Create df64 conversion pipelines (f64 ↔ df64)
        id<MTLFunction> cvt_f64_to_df64 = [g_metal_library newFunctionWithName:@"convert_f64_to_df64"];
        if (cvt_f64_to_df64)
            g_convert_f64_to_df64_pipeline = [g_metal_device newComputePipelineStateWithFunction:cvt_f64_to_df64 error:&error];
        id<MTLFunction> cvt_df64_to_f64 = [g_metal_library newFunctionWithName:@"convert_df64_to_f64"];
        if (cvt_df64_to_f64)
            g_convert_df64_to_f64_pipeline = [g_metal_device newComputePipelineStateWithFunction:cvt_df64_to_f64 error:&error];

        // Create f32 (native float) matmul pipeline
        id<MTLFunction> f32_func = [g_metal_library newFunctionWithName:@"matmul_f32"];
        if (f32_func) {
            g_matmul_f32_pipeline = [g_metal_device newComputePipelineStateWithFunction:f32_func error:&error];
        }

        // Create pure f32 simdgroup_matrix matmul pipeline (reads float*, not uint2*)
        id<MTLFunction> f32_simd_func = [g_metal_library newFunctionWithName:@"matmul_f32_simd_pure"];
        if (f32_simd_func) {
            g_matmul_f32_simd_pipeline = [g_metal_device newComputePipelineStateWithFunction:f32_simd_func error:&error];
            if (g_matmul_f32_simd_pipeline) {
                fprintf(stderr, "[GPU] f32_simd: maxTG=%lu, execWidth=%lu, staticTGMem=%lu\n",
                    (unsigned long)[g_matmul_f32_simd_pipeline maxTotalThreadsPerThreadgroup],
                    (unsigned long)[g_matmul_f32_simd_pipeline threadExecutionWidth],
                    (unsigned long)[g_matmul_f32_simd_pipeline staticThreadgroupMemoryLength]);
            } else {
                eshkol_error("Metal: f32 simdgroup_matrix pipeline creation FAILED: %s",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            }
        } else {
            eshkol_error("Metal: matmul_f32_simd_pure function not found in library");
        }

        // Create 128x128 f32 simdgroup matmul pipeline (config-driven thread count)
        id<MTLFunction> f32_simd_128_func = [g_metal_library newFunctionWithName:@"matmul_f32_simd_128"];
        if (f32_simd_128_func) {
            g_matmul_f32_simd_128_pipeline = [g_metal_device newComputePipelineStateWithFunction:f32_simd_128_func error:&error];
            if (g_matmul_f32_simd_128_pipeline) {
                NSUInteger maxTG = [g_matmul_f32_simd_128_pipeline maxTotalThreadsPerThreadgroup];
                fprintf(stderr, "[GPU] f32_simd_128: maxTG=%lu, execWidth=%lu, staticTGMem=%lu\n",
                    (unsigned long)maxTG,
                    (unsigned long)[g_matmul_f32_simd_128_pipeline threadExecutionWidth],
                    (unsigned long)[g_matmul_f32_simd_128_pipeline staticThreadgroupMemoryLength]);
                if (maxTG < g_cfg_f32s_128.threads) {
                    fprintf(stderr, "[GPU] f32_simd_128: maxTG=%lu < %u, disabling 128x128 kernel\n",
                            (unsigned long)maxTG, g_cfg_f32s_128.threads);
                    g_matmul_f32_simd_128_pipeline = nil;
                }
            } else {
                fprintf(stderr, "[GPU] f32_simd_128: pipeline creation failed: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            }
        }

        // Create GPU f64↔f32 conversion pipelines (for MPS path)
        id<MTLFunction> cvt_f64_to_f32 = [g_metal_library newFunctionWithName:@"convert_f64_to_f32"];
        if (cvt_f64_to_f32)
            g_convert_f64_to_f32_pipeline = [g_metal_device newComputePipelineStateWithFunction:cvt_f64_to_f32 error:&error];
        id<MTLFunction> cvt_f32_to_f64 = [g_metal_library newFunctionWithName:@"convert_f32_to_f64"];
        if (cvt_f32_to_f64)
            g_convert_f32_to_f64_pipeline = [g_metal_device newComputePipelineStateWithFunction:cvt_f32_to_f64 error:&error];

        // Create fixed-point ML (fp24) pipeline — Quake-3-inspired bitwise kernel (V2)
        // No find_max_exp pipeline needed — max exponent computed on CPU during memcpy
        id<MTLFunction> fp24_func = [g_metal_library newFunctionWithName:@"matmul_fp24"];
        if (fp24_func) {
            g_matmul_fp24_pipeline = [g_metal_device newComputePipelineStateWithFunction:fp24_func error:&error];
            if (g_matmul_fp24_pipeline) {
                NSUInteger maxTG = [g_matmul_fp24_pipeline maxTotalThreadsPerThreadgroup];
                fprintf(stderr, "[GPU] fp24: maxTG=%lu, execWidth=%lu, staticTGMem=%lu\n",
                    (unsigned long)maxTG,
                    (unsigned long)[g_matmul_fp24_pipeline threadExecutionWidth],
                    (unsigned long)[g_matmul_fp24_pipeline staticThreadgroupMemoryLength]);

                // Hardware-adaptive recompilation: if register pressure limits threads
                if (maxTG < g_fp_threads) {
                    fprintf(stderr, "[GPU] fp24: maxTG=%lu < desired %u, reducing parameters\n",
                            (unsigned long)maxTG, g_fp_threads);
                    // Re-search with reduced max_threads constraint
                    HardwareProfile hw_reduced = g_hw;
                    hw_reduced.max_threads_per_tg = (uint32_t)maxTG;
                    search_fp_config(KERNEL_FP24, g_cfg_fp24, hw_reduced);
                    // Sync backward-compat globals
                    g_fp_bm = g_cfg_fp24.bm; g_fp_bn = g_cfg_fp24.bn;
                    g_fp_bk = g_cfg_fp24.bk; g_fp_tt = g_cfg_fp24.tt; g_fp_threads = g_cfg_fp24.threads;
                    fprintf(stderr, "[GPU] fp24: recompiling with BM=%u, BN=%u, BK=%u, threads=%u\n",
                            g_fp_bm, g_fp_bn, g_fp_bk, g_fp_threads);

                    // Recompile with ALL defines (centralized helper)
                    NSString* fp_source = [build_shader_config_string() stringByAppendingString:g_matmul_sf64_source];
                    id<MTLLibrary> fp_lib = [g_metal_device newLibraryWithSource:fp_source
                                                                         options:options error:&error];
                    if (fp_lib) {
                        g_metal_library = fp_lib;
                        id<MTLFunction> fp24_refunc = [fp_lib newFunctionWithName:@"matmul_fp24"];
                        if (fp24_refunc) {
                            g_matmul_fp24_pipeline = [g_metal_device newComputePipelineStateWithFunction:fp24_refunc error:&error];
                            fprintf(stderr, "[GPU] fp24 recompiled: maxTG=%lu\n",
                                (unsigned long)[g_matmul_fp24_pipeline maxTotalThreadsPerThreadgroup]);
                        }
                    }
                }
            } else {
                eshkol_error("Metal: fp24 pipeline creation FAILED: %s",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            }
        } else {
            eshkol_error("Metal: matmul_fp24 function not found in library");
        }

        // Create fixed-point exact (fp53) pipeline — full 53-bit mantissa, 128-bit accumulation
        id<MTLFunction> fp53_func = [g_metal_library newFunctionWithName:@"matmul_fp53"];
        if (fp53_func) {
            g_matmul_fp53_pipeline = [g_metal_device newComputePipelineStateWithFunction:fp53_func error:&error];
            if (g_matmul_fp53_pipeline) {
                NSUInteger maxTG = [g_matmul_fp53_pipeline maxTotalThreadsPerThreadgroup];
                fprintf(stderr, "[GPU] fp53: maxTG=%lu, execWidth=%lu, staticTGMem=%lu\n",
                    (unsigned long)maxTG,
                    (unsigned long)[g_matmul_fp53_pipeline threadExecutionWidth],
                    (unsigned long)[g_matmul_fp53_pipeline staticThreadgroupMemoryLength]);

                if (maxTG < g_fp53_threads) {
                    fprintf(stderr, "[GPU] fp53: maxTG=%lu < desired %u, reducing parameters\n",
                            (unsigned long)maxTG, g_fp53_threads);
                    // Re-search with reduced max_threads constraint
                    HardwareProfile hw_reduced = g_hw;
                    hw_reduced.max_threads_per_tg = (uint32_t)maxTG;
                    search_fp_config(KERNEL_FP53, g_cfg_fp53, hw_reduced);
                    // Sync backward-compat globals
                    g_fp53_bm = g_cfg_fp53.bm; g_fp53_bn = g_cfg_fp53.bn;
                    g_fp53_bk = g_cfg_fp53.bk; g_fp53_tt = g_cfg_fp53.tt;
                    g_fp53_threads = g_cfg_fp53.threads; g_fp53_sb_stride = g_cfg_fp53.bn + 1;
                    fprintf(stderr, "[GPU] fp53: recompiling with BM=%u, BN=%u, BK=%u, threads=%u\n",
                            g_fp53_bm, g_fp53_bn, g_fp53_bk, g_fp53_threads);

                    // Recompile with ALL defines (centralized helper)
                    NSString* fp53_source = [build_shader_config_string() stringByAppendingString:g_matmul_sf64_source];
                    id<MTLLibrary> fp53_lib = [g_metal_device newLibraryWithSource:fp53_source
                                                                         options:options error:&error];
                    if (fp53_lib) {
                        g_metal_library = fp53_lib;
                        id<MTLFunction> fp53_refunc = [fp53_lib newFunctionWithName:@"matmul_fp53"];
                        if (fp53_refunc) {
                            g_matmul_fp53_pipeline = [g_metal_device newComputePipelineStateWithFunction:fp53_refunc error:&error];
                            fprintf(stderr, "[GPU] fp53 recompiled: maxTG=%lu\n",
                                (unsigned long)[g_matmul_fp53_pipeline maxTotalThreadsPerThreadgroup]);
                        }
                    }
                }
            } else {
                eshkol_error("Metal: fp53 pipeline creation FAILED: %s",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            }
        } else {
            eshkol_error("Metal: matmul_fp53 function not found in library");
        }

        // Create Ozaki-II CRT modular GEMM pipeline (reuses F32S constants)
        id<MTLFunction> ozaki_gemm_func = [g_metal_library newFunctionWithName:@"matmul_ozaki_gemm"];
        if (ozaki_gemm_func) {
            g_matmul_ozaki_gemm_pipeline = [g_metal_device newComputePipelineStateWithFunction:ozaki_gemm_func error:&error];
            if (g_matmul_ozaki_gemm_pipeline) {
                fprintf(stderr, "[GPU] Ozaki-II GEMM pipeline: maxTG=%lu, execWidth=%lu\n",
                    (unsigned long)[g_matmul_ozaki_gemm_pipeline maxTotalThreadsPerThreadgroup],
                    (unsigned long)[g_matmul_ozaki_gemm_pipeline threadExecutionWidth]);
            } else {
                fprintf(stderr, "[GPU] Ozaki-II GEMM pipeline creation FAILED: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            }
        } else {
            fprintf(stderr, "[GPU] matmul_ozaki_gemm function not found (will use fp53 for exact tier)\n");
        }

        // Initialize Ozaki-II tables and CRT constants
        init_ozaki_log2_table();
        if (const char* nmod_env = std::getenv("ESHKOL_OZAKI_NUM_MODULI")) {
            int n = atoi(nmod_env);
            if (n >= 2 && n <= 49) g_ozaki_num_moduli = n;
            else fprintf(stderr, "[GPU] ESHKOL_OZAKI_NUM_MODULI=%d out of range [2,49], using default %d\n", n, g_ozaki_num_moduli);
        }
        precompute_ozaki_constants(g_ozaki_num_moduli, &g_ozaki_crt);
        fprintf(stderr, "[GPU] Ozaki-II: N=%d moduli, log2(P)=%.1f\n", g_ozaki_num_moduli, g_ozaki_crt.log2P);

        // Read precision tier from env var
        if (const char* env = std::getenv("ESHKOL_GPU_PRECISION")) {
            if (strcmp(env, "high") == 0 && g_matmul_df64_pipeline) g_metal_precision_tier = 1;
            else if (strcmp(env, "fast") == 0) g_metal_precision_tier = 2;
            else if (strcmp(env, "ml") == 0 && g_matmul_fp24_pipeline) g_metal_precision_tier = 3;
            else g_metal_precision_tier = 0;
        }

        // Select sf64 variant and exact tier kernel
        // Default: tier 0 uses fp53 (fixed-point exact). ESHKOL_SF64_KERNEL=legacy uses sf64 V2.
        bool use_legacy_sf64 = false;
        if (const char* env = std::getenv("ESHKOL_SF64_KERNEL")) {
            if (strcmp(env, "legacy") == 0) {
                use_legacy_sf64 = true;
                eshkol_info("Metal: using legacy sf64 kernel for exact tier");
            } else if (strcmp(env, "v2") == 0 && g_matmul_sf64_v2_pipeline) {
                use_legacy_sf64 = true;
                g_matmul_sf64_pipeline = g_matmul_sf64_v2_pipeline;
                eshkol_info("Metal: using sf64 v2 kernel (deferred rounding)");
            }
        }
        if (g_metal_precision_tier == 0 && !use_legacy_sf64 && g_matmul_fp53_pipeline) {
            eshkol_info("Metal: precision tier 0 (fp53, fixed-point exact, Quake-3 bitwise)");
        } else if (g_metal_precision_tier == 0) {
            eshkol_info("Metal: precision tier 0 (sf64 legacy, exact IEEE f64)");
        } else if (g_metal_precision_tier == 1) {
            eshkol_info("Metal: precision tier 1 (df64, dual-float ~48-bit)");
        } else if (g_metal_precision_tier == 2) {
            eshkol_info("Metal: precision tier 2 (f32, fast)");
        } else if (g_metal_precision_tier == 3) {
            eshkol_info("Metal: precision tier 3 (fp24, ML fixed-point, Quake-3 bitwise)");
        }

        // Create pipeline states for elementwise, reduce, transpose kernels
        id<MTLFunction> elem_func = [g_metal_library newFunctionWithName:@"elementwise_sf64"];
        if (elem_func) {
            g_elementwise_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:elem_func error:&error];
        }
        id<MTLFunction> reduce_func = [g_metal_library newFunctionWithName:@"reduce_sf64"];
        if (reduce_func) {
            g_reduce_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:reduce_func error:&error];
        }
        id<MTLFunction> transpose_func = [g_metal_library newFunctionWithName:@"transpose_sf64"];
        if (transpose_func) {
            g_transpose_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:transpose_func error:&error];
        }
        id<MTLFunction> reduce_axis_func = [g_metal_library newFunctionWithName:@"reduce_sf64_axis"];
        if (reduce_axis_func) {
            g_reduce_axis_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:reduce_axis_func error:&error];
        }
        id<MTLFunction> softmax_func = [g_metal_library newFunctionWithName:@"softmax_sf64"];
        if (softmax_func) {
            g_softmax_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:softmax_func error:&error];
        }
        id<MTLFunction> normalize_func = [g_metal_library newFunctionWithName:@"normalize_sf64"];
        if (normalize_func) {
            g_normalize_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:normalize_func error:&error];
        }
        id<MTLFunction> swap_func = [g_metal_library newFunctionWithName:@"word_swap_sf64"];
        if (swap_func) {
            g_word_swap_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:swap_func error:&error];
            if (!g_word_swap_sf64_pipeline) {
                eshkol_error("Metal: failed to create word_swap pipeline: %s",
                             [[error localizedDescription] UTF8String]);
            }
        } else {
            eshkol_error("Metal: word_swap_sf64 kernel not found in compiled library");
        }

        eshkol_info("Metal: sf64 shader compiled (%ux%u threadgroup, %ux%u block, TILE_K=%u)",
                    g_sf64_tg, g_sf64_tg, g_sf64_blk, g_sf64_blk, g_sf64_tile_k);

        // GPU warmup: tiny 1×1 dispatch to prime the pipeline and avoid first-dispatch crash
        {
            id<MTLBuffer> warmup_buf = [g_metal_device newBufferWithLength:64
                                                                   options:MTLResourceStorageModeShared];
            if (warmup_buf) {
                memset([warmup_buf contents], 0, 64);
                id<MTLCommandBuffer> warmup_cmd = [g_metal_queue commandBuffer];
                id<MTLComputeCommandEncoder> warmup_enc = [warmup_cmd computeCommandEncoder];
                [warmup_enc setComputePipelineState:g_matmul_sf64_pipeline];
                [warmup_enc setBuffer:warmup_buf offset:0 atIndex:0];
                [warmup_enc setBuffer:warmup_buf offset:0 atIndex:1];
                [warmup_enc setBuffer:warmup_buf offset:0 atIndex:2];
                uint32_t one = 1;
                [warmup_enc setBytes:&one length:sizeof(uint32_t) atIndex:3];
                [warmup_enc setBytes:&one length:sizeof(uint32_t) atIndex:4];
                [warmup_enc setBytes:&one length:sizeof(uint32_t) atIndex:5];
                [warmup_enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(g_sf64_tg, g_sf64_tg, 1)];
                [warmup_enc endEncoding];
                [warmup_cmd commit];
                [warmup_cmd waitUntilCompleted];
                warmup_buf = nil;  // release
            }
        }

        return 0;
    }
}

static void metal_shutdown(void) {
    @autoreleasepool {
        pool_drain();
        g_matmul_sf64_pipeline = nil;
        g_matmul_sf64_v2_pipeline = nil;
        g_matmul_df64_pipeline = nil;
        g_matmul_df64_pure_pipeline = nil;
        g_convert_f64_to_df64_pipeline = nil;
        g_convert_df64_to_f64_pipeline = nil;
        g_matmul_fp24_pipeline = nil;
        g_matmul_fp53_pipeline = nil;
        g_matmul_ozaki_gemm_pipeline = nil;
        g_matmul_f32_pipeline = nil;
        g_elementwise_sf64_pipeline = nil;
        g_reduce_sf64_pipeline = nil;
        g_reduce_axis_sf64_pipeline = nil;
        g_transpose_sf64_pipeline = nil;
        g_softmax_sf64_pipeline = nil;
        g_normalize_sf64_pipeline = nil;
        g_word_swap_sf64_pipeline = nil;
        g_metal_library = nil;
        g_metal_queue = nil;
        g_metal_device = nil;
    }
}

static int metal_alloc(size_t size_bytes, EshkolMemoryType mem_type, EshkolGPUBuffer* out) {
    @autoreleasepool {
        MTLResourceOptions options;
        switch (mem_type) {
            case ESHKOL_MEM_DEVICE:
                options = MTLResourceStorageModePrivate;
                break;
            case ESHKOL_MEM_HOST_PINNED:
                options = g_metal_unified_memory ? MTLResourceStorageModeShared
                                                  : MTLResourceStorageModeManaged;
                break;
            default:
                options = MTLResourceStorageModeShared;
                break;
        }

        id<MTLBuffer> buffer = [g_metal_device newBufferWithLength:size_bytes options:options];
        if (!buffer) return -1;

        out->size_bytes = size_bytes;
        out->mem_type = mem_type;
        out->backend = ESHKOL_GPU_METAL;
        out->backend_data = (__bridge_retained void*)buffer;

        if (options != MTLResourceStorageModePrivate) {
            out->host_ptr = [buffer contents];
            out->device_ptr = out->host_ptr;
        } else {
            out->host_ptr = nullptr;
            out->device_ptr = (__bridge void*)buffer;
        }

        return 0;
    }
}

static void metal_free(EshkolGPUBuffer* buffer) {
    @autoreleasepool {
        if (buffer->backend_data) {
            id<MTLBuffer> mtl_buf = (__bridge_transfer id<MTLBuffer>)buffer->backend_data;
            mtl_buf = nil;
        }
    }
}

static int metal_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    @autoreleasepool {
        // Unified memory: no sync needed
        if (g_metal_unified_memory && buffer->mem_type != ESHKOL_MEM_DEVICE) {
            return 0;
        }

        id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buffer->backend_data;

        if (direction == ESHKOL_SYNC_HOST_TO_DEVICE || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
            if (@available(macOS 10.11, *)) {
                [mtl_buf didModifyRange:NSMakeRange(0, buffer->size_bytes)];
            }
        }

        if (direction == ESHKOL_SYNC_DEVICE_TO_HOST || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
            id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            if (@available(macOS 10.11, *)) {
                [blit synchronizeResource:mtl_buf];
            }
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        return 0;
    }
}

// Dispatch word-swap kernel on encoder (does NOT end encoding or commit)
static void dispatch_word_swap(id<MTLComputeCommandEncoder> encoder,
                                id<MTLBuffer> buf, uint32_t count) {
    [encoder setComputePipelineState:g_word_swap_sf64_pipeline];
    [encoder setBuffer:buf offset:0 atIndex:0];
    [encoder setBytes:&count length:sizeof(uint32_t) atIndex:1];
    NSUInteger groups = (count + 255) / 256;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// MPS path: f64 input → GPU f32 conversion → MPSMatrixMultiplication → GPU f64 conversion
static int metal_matmul_f64_via_mps(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                                     uint64_t M, uint64_t K, uint64_t N) {
    @autoreleasepool {
        size_t elementsA = M * K;
        size_t elementsB = K * N;
        size_t elementsC = M * N;

        // Wrap f64 input as MTLBuffers (zero-copy on unified memory)
        id<MTLBuffer> f64_a = A->backend_data ? (__bridge id<MTLBuffer>)A->backend_data : nil;
        id<MTLBuffer> f64_b = B->backend_data ? (__bridge id<MTLBuffer>)B->backend_data : nil;
        id<MTLBuffer> f64_c = C->backend_data ? (__bridge id<MTLBuffer>)C->backend_data : nil;

        bool did_alloc_a = false, did_alloc_b = false, did_alloc_c = false;
        if (!f64_a) {
            f64_a = pool_alloc(elementsA * 8);
            if (!f64_a) return -1;
            memcpy([f64_a contents], A->host_ptr, elementsA * 8);
            did_alloc_a = true;
        }
        if (!f64_b) {
            f64_b = pool_alloc(elementsB * 8);
            if (!f64_b) { if (did_alloc_a) pool_release(f64_a); return -1; }
            memcpy([f64_b contents], B->host_ptr, elementsB * 8);
            did_alloc_b = true;
        }
        if (!f64_c) {
            f64_c = pool_alloc(elementsC * 8);
            if (!f64_c) {
                if (did_alloc_a) pool_release(f64_a);
                if (did_alloc_b) pool_release(f64_b);
                return -1;
            }
            did_alloc_c = true;
        }

        // Allocate f32 buffers for MPS
        id<MTLBuffer> f32_a = pool_alloc(elementsA * sizeof(float));
        id<MTLBuffer> f32_b = pool_alloc(elementsB * sizeof(float));
        id<MTLBuffer> f32_c = pool_alloc(elementsC * sizeof(float));
        if (!f32_a || !f32_b || !f32_c) {
            if (f32_a) pool_release(f32_a);
            if (f32_b) pool_release(f32_b);
            if (f32_c) pool_release(f32_c);
            if (did_alloc_a) pool_release(f64_a);
            if (did_alloc_b) pool_release(f64_b);
            if (did_alloc_c) pool_release(f64_c);
            return -1;
        }

        // GPU f64 → f32 conversion (or CPU fallback if conversion pipeline unavailable)
        if (g_convert_f64_to_f32_pipeline && g_convert_f32_to_f64_pipeline) {
            // GPU conversion: single command buffer for both A and B
            id<MTLCommandBuffer> cvt_cmd = [g_metal_queue commandBuffer];
            id<MTLComputeCommandEncoder> cvt_enc = [cvt_cmd computeCommandEncoder];

            [cvt_enc setComputePipelineState:g_convert_f64_to_f32_pipeline];

            // Convert A
            [cvt_enc setBuffer:f64_a offset:0 atIndex:0];
            [cvt_enc setBuffer:f32_a offset:0 atIndex:1];
            uint32_t countA = static_cast<uint32_t>(elementsA);
            [cvt_enc setBytes:&countA length:sizeof(uint32_t) atIndex:2];
            NSUInteger groupsA = (elementsA + 255) / 256;
            [cvt_enc dispatchThreadgroups:MTLSizeMake(groupsA, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // Convert B
            [cvt_enc setBuffer:f64_b offset:0 atIndex:0];
            [cvt_enc setBuffer:f32_b offset:0 atIndex:1];
            uint32_t countB = static_cast<uint32_t>(elementsB);
            [cvt_enc setBytes:&countB length:sizeof(uint32_t) atIndex:2];
            NSUInteger groupsB = (elementsB + 255) / 256;
            [cvt_enc dispatchThreadgroups:MTLSizeMake(groupsB, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [cvt_enc endEncoding];
            [cvt_cmd commit];
            [cvt_cmd waitUntilCompleted];
        } else {
            // CPU fallback conversion
            const double* srcA = (const double*)(did_alloc_a ? [f64_a contents] : A->host_ptr);
            const double* srcB = (const double*)(did_alloc_b ? [f64_b contents] : B->host_ptr);
            float* dstA = (float*)[f32_a contents];
            float* dstB = (float*)[f32_b contents];
            for (size_t i = 0; i < elementsA; i++) dstA[i] = (float)srcA[i];
            for (size_t i = 0; i < elementsB; i++) dstB[i] = (float)srcB[i];
        }

        // MPS matmul (Apple's hand-optimized implementation)
        MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
                                                                           rowBytes:K * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N
                                                                           rowBytes:N * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
                                                                           rowBytes:N * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];

        MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:f32_a descriptor:desc_a];
        MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:f32_b descriptor:desc_b];
        MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:f32_c descriptor:desc_c];

        MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_metal_device
            transposeLeft:NO transposeRight:NO
            resultRows:M resultColumns:N interiorColumns:K
            alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        [matmul encodeToCommandBuffer:cmd leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
        [cmd commit];
        [cmd waitUntilCompleted];

        // GPU f32 → f64 conversion back to output
        if (g_convert_f32_to_f64_pipeline) {
            id<MTLCommandBuffer> cvt_cmd = [g_metal_queue commandBuffer];
            id<MTLComputeCommandEncoder> cvt_enc = [cvt_cmd computeCommandEncoder];

            [cvt_enc setComputePipelineState:g_convert_f32_to_f64_pipeline];
            [cvt_enc setBuffer:f32_c offset:0 atIndex:0];
            [cvt_enc setBuffer:f64_c offset:0 atIndex:1];
            uint32_t countC = static_cast<uint32_t>(elementsC);
            [cvt_enc setBytes:&countC length:sizeof(uint32_t) atIndex:2];
            NSUInteger groupsC = (elementsC + 255) / 256;
            [cvt_enc dispatchThreadgroups:MTLSizeMake(groupsC, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [cvt_enc endEncoding];
            [cvt_cmd commit];
            [cvt_cmd waitUntilCompleted];

            // Copy f64 result back if output was fallback-allocated
            if (did_alloc_c) {
                memcpy(C->host_ptr, [f64_c contents], elementsC * 8);
            }
        } else {
            // CPU fallback f32→f64
            const float* srcC = (const float*)[f32_c contents];
            double* dstC = (double*)C->host_ptr;
            for (size_t i = 0; i < elementsC; i++) dstC[i] = (double)srcC[i];
        }

        pool_release(f32_a);
        pool_release(f32_b);
        pool_release(f32_c);
        if (did_alloc_a) pool_release(f64_a);
        if (did_alloc_b) pool_release(f64_b);
        if (did_alloc_c) pool_release(f64_c);
        return 0;
    }
}

// CPU-side f64 ↔ sf64 word-order conversion.
// ONLY for scalar kernel parameters (normalize gamma/beta/epsilon).
// Buffer data uses raw memcpy — kernels handle byte-swap inline via native_to_sf64().
static void convert_f64_to_sf64(const double* src, uint32_t* dst, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint64_t bits;
        memcpy(&bits, &src[i], sizeof(double));
        dst[i * 2] = static_cast<uint32_t>(bits >> 32);      // High word
        dst[i * 2 + 1] = static_cast<uint32_t>(bits);        // Low word
    }
}

// Dispatch a single matmul: C = A[M×K] × B[K×N].
// Uses enhanced error reporting and GPU timing for diagnostics.
static int metal_matmul_dispatch_one(id<MTLBuffer> buf_a, id<MTLBuffer> buf_b,
                                      id<MTLBuffer> buf_c,
                                      uint32_t M, uint32_t K, uint32_t N) {
    // Enhanced error reporting for encoder execution status
    MTLCommandBufferDescriptor* desc = [[MTLCommandBufferDescriptor alloc] init];
    desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    id<MTLCommandBuffer> cmd = [g_metal_queue commandBufferWithDescriptor:desc];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:g_matmul_sf64_pipeline];
    [enc setBuffer:buf_a offset:0 atIndex:0];
    [enc setBuffer:buf_b offset:0 atIndex:1];
    [enc setBuffer:buf_c offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];

    NSUInteger groupsX = (N + g_sf64_blk - 1) / g_sf64_blk;
    NSUInteger groupsY = (M + g_sf64_blk - 1) / g_sf64_blk;

    fprintf(stderr, "[GPU] dispatch: M=%u K=%u N=%u groups=(%lu,%lu) TG=%u BLK=%u "
            "bufA=%luMB bufB=%luMB bufC=%luMB\n",
            M, K, N, (unsigned long)groupsX, (unsigned long)groupsY,
            g_sf64_tg, g_sf64_blk,
            (unsigned long)([buf_a length] / (1024*1024)),
            (unsigned long)([buf_b length] / (1024*1024)),
            (unsigned long)([buf_c length] / (1024*1024)));

    [enc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1)
            threadsPerThreadgroup:MTLSizeMake(g_sf64_tg, g_sf64_tg, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] == MTLCommandBufferStatusCompleted) {
        double gpu_ms = ([cmd GPUEndTime] - [cmd GPUStartTime]) * 1000.0;
        double gflops = (2.0 * M * K * N) / (gpu_ms / 1000.0) / 1e9;
        fprintf(stderr, "[GPU] completed: %.1fms %.0f GFLOPS (M=%u K=%u N=%u)\n",
                gpu_ms, gflops, M, K, N);
        return 0;
    }

    if ([cmd status] == MTLCommandBufferStatusError) {
        NSError* err = [cmd error];
        fprintf(stderr, "[GPU] matmul error (code=%ld): %s "
                "(M=%u, K=%u, N=%u, TG=%u, BLK=%u, TILE_K=%u)\n",
                (long)[err code],
                err ? [[err localizedDescription] UTF8String] : "unknown",
                M, K, N, g_sf64_tg, g_sf64_blk, g_sf64_tile_k);
        return -3;
    }

    fprintf(stderr, "[GPU] matmul unexpected status=%lu\n", (unsigned long)[cmd status]);
    return -4;
}

// Dispatch for f32 simdgroup_matrix kernel — completely separate from sf64/df64 path.
// Uses 128 threads per threadgroup (4 simdgroups × 32), 64×64 output tiles.
static int metal_matmul_dispatch_f32_simd(id<MTLBuffer> buf_a, id<MTLBuffer> buf_b,
                                           id<MTLBuffer> buf_c,
                                           uint32_t M, uint32_t K, uint32_t N) {
    MTLCommandBufferDescriptor* desc = [[MTLCommandBufferDescriptor alloc] init];
    desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    id<MTLCommandBuffer> cmd = [g_metal_queue commandBufferWithDescriptor:desc];

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:g_matmul_f32_simd_pipeline];
    [enc setBuffer:buf_a offset:0 atIndex:0];
    [enc setBuffer:buf_b offset:0 atIndex:1];
    [enc setBuffer:buf_c offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];

    // f32_simd: BM=64, BN=64, 128 threads (4 simdgroups × 32)
    NSUInteger groupsX = (N + 63) / 64;
    NSUInteger groupsY = (M + 63) / 64;

    fprintf(stderr, "[GPU] f32_simd dispatch: M=%u K=%u N=%u groups=(%lu,%lu) threads=128\n",
            M, K, N, (unsigned long)groupsX, (unsigned long)groupsY);

    [enc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] == MTLCommandBufferStatusCompleted) {
        double gpu_ms = ([cmd GPUEndTime] - [cmd GPUStartTime]) * 1000.0;
        double gflops = (2.0 * M * K * N) / (gpu_ms / 1000.0) / 1e9;
        fprintf(stderr, "[GPU] f32_simd completed: %.1fms %.0f GFLOPS (M=%u K=%u N=%u)\n",
                gpu_ms, gflops, M, K, N);
        return 0;
    }

    if ([cmd status] == MTLCommandBufferStatusError) {
        NSError* err = [cmd error];
        fprintf(stderr, "[GPU] f32_simd error (code=%ld): %s (M=%u, K=%u, N=%u)\n",
                (long)[err code],
                err ? [[err localizedDescription] UTF8String] : "unknown",
                M, K, N);
        return -3;
    }

    return -4;
}

// ============================================================================
// Per-Tier Dispatch Functions — each tier owns its full pipeline
// ============================================================================

// Tier 0: sf64 exact dispatch with row-blocking for large matrices
// sf64 at 4096×4096 takes ~1.9s on GPU. At 8192 it'd be ~15s, risking GPU timeout.
// Row-blocking splits M into chunks dispatched separately, each well within timeout.
// Uses Metal buffer offsets: A[m_off*K..] and C[m_off*N..] with M=chunk_size.
static int metal_matmul_sf64_dispatch(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                                       uint64_t M, uint64_t K, uint64_t N) {
    @autoreleasepool {
        if (!g_matmul_sf64_pipeline) return -1;

        size_t elementsA = M * K;
        size_t elementsB = K * N;
        size_t elementsC = M * N;

        id<MTLBuffer> buf_a = pool_alloc(elementsA * 8);
        id<MTLBuffer> buf_b = pool_alloc(elementsB * 8);
        id<MTLBuffer> buf_c = pool_alloc(elementsC * 8);
        if (!buf_a || !buf_b || !buf_c) {
            if (buf_a) pool_release(buf_a);
            if (buf_b) pool_release(buf_b);
            if (buf_c) pool_release(buf_c);
            return -1;
        }

        memcpy([buf_a contents], A->host_ptr, elementsA * 8);
        memcpy([buf_b contents], B->host_ptr, elementsB * 8);

        // Determine chunk size: if M > 4096, split into row-blocks
        // Each chunk dispatches C_chunk = A_chunk × B where B is full
        uint32_t chunk_M = (uint32_t)M;
        if (M > 4096) {
            // Align chunk to block size for clean tile boundaries
            chunk_M = ((4096 + g_sf64_blk - 1) / g_sf64_blk) * g_sf64_blk;
        }

        int result = 0;
        double total_gpu_ms = 0.0;

        for (uint32_t m_off = 0; m_off < (uint32_t)M; m_off += chunk_M) {
            uint32_t m_len = (m_off + chunk_M <= (uint32_t)M) ? chunk_M : ((uint32_t)M - m_off);

            MTLCommandBufferDescriptor* desc = [[MTLCommandBufferDescriptor alloc] init];
            desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
            id<MTLCommandBuffer> cmd = [g_metal_queue commandBufferWithDescriptor:desc];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            [enc setComputePipelineState:g_matmul_sf64_pipeline];
            // A sub-matrix at row offset m_off: kernel sees rows 0..m_len-1
            [enc setBuffer:buf_a offset:(size_t)m_off * K * 8 atIndex:0];
            [enc setBuffer:buf_b offset:0 atIndex:1];
            [enc setBuffer:buf_c offset:(size_t)m_off * N * 8 atIndex:2];
            uint32_t uK = (uint32_t)K, uN = (uint32_t)N;
            [enc setBytes:&m_len length:4 atIndex:3];
            [enc setBytes:&uK length:4 atIndex:4];
            [enc setBytes:&uN length:4 atIndex:5];

            NSUInteger groupsX = (N + g_sf64_blk - 1) / g_sf64_blk;
            NSUInteger groupsY = (m_len + g_sf64_blk - 1) / g_sf64_blk;

            if (m_off == 0) {
                fprintf(stderr, "[GPU] sf64 dispatch: M=%lu K=%lu N=%lu chunk=%u groups=(%lu,%lu)\n",
                        (unsigned long)M, (unsigned long)K, (unsigned long)N, m_len,
                        (unsigned long)groupsX, (unsigned long)groupsY);
            }

            [enc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1)
                    threadsPerThreadgroup:MTLSizeMake(g_sf64_tg, g_sf64_tg, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            if ([cmd status] == MTLCommandBufferStatusCompleted) {
                double gpu_ms = ([cmd GPUEndTime] - [cmd GPUStartTime]) * 1000.0;
                total_gpu_ms += gpu_ms;
            } else if ([cmd status] == MTLCommandBufferStatusError) {
                NSError* err = [cmd error];
                fprintf(stderr, "[GPU] sf64 error (code=%ld): %s (chunk m_off=%u m_len=%u)\n",
                        (long)[err code],
                        err ? [[err localizedDescription] UTF8String] : "unknown",
                        m_off, m_len);
                result = -3;
                break;
            } else {
                result = -4;
                break;
            }
        }

        if (result == 0) {
            double gflops = (2.0 * M * K * N) / (total_gpu_ms / 1000.0) / 1e9;
            fprintf(stderr, "[GPU] sf64 completed: %.1fms %.0f GFLOPS (M=%lu K=%lu N=%lu)\n",
                    total_gpu_ms, gflops, (unsigned long)M, (unsigned long)K, (unsigned long)N);
            memcpy(C->host_ptr, [buf_c contents], elementsC * 8);
        }

        pool_release(buf_a);
        pool_release(buf_b);
        pool_release(buf_c);
        return result;
    }
}

// Tier 1: df64 dispatch — pre-converted float2 pipeline with pure df64 kernel
// Architecture: f64 buf → GPU convert_f64_to_df64 → float2 buf → pure df64 kernel → float2 buf → GPU convert_df64_to_f64 → f64 buf
static int metal_matmul_df64_dispatch(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                                       uint64_t M, uint64_t K, uint64_t N) {
    @autoreleasepool {
        if (!g_matmul_df64_pure_pipeline || !g_convert_f64_to_df64_pipeline || !g_convert_df64_to_f64_pipeline) {
            // Fall back to legacy df64 if pure pipeline unavailable
            if (!g_matmul_df64_pipeline) return -1;

            // Legacy path: single kernel with sf64 tile constants
            size_t elementsA = M * K, elementsB = K * N, elementsC = M * N;
            id<MTLBuffer> buf_a = pool_alloc(elementsA * 8);
            id<MTLBuffer> buf_b = pool_alloc(elementsB * 8);
            id<MTLBuffer> buf_c = pool_alloc(elementsC * 8);
            if (!buf_a || !buf_b || !buf_c) {
                if (buf_a) pool_release(buf_a); if (buf_b) pool_release(buf_b); if (buf_c) pool_release(buf_c);
                return -1;
            }
            memcpy([buf_a contents], A->host_ptr, elementsA * 8);
            memcpy([buf_b contents], B->host_ptr, elementsB * 8);

            MTLCommandBufferDescriptor* desc = [[MTLCommandBufferDescriptor alloc] init];
            desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
            id<MTLCommandBuffer> cmd = [g_metal_queue commandBufferWithDescriptor:desc];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_matmul_df64_pipeline];
            [enc setBuffer:buf_a offset:0 atIndex:0];
            [enc setBuffer:buf_b offset:0 atIndex:1];
            [enc setBuffer:buf_c offset:0 atIndex:2];
            uint32_t uM = (uint32_t)M, uK = (uint32_t)K, uN = (uint32_t)N;
            [enc setBytes:&uM length:4 atIndex:3];
            [enc setBytes:&uK length:4 atIndex:4];
            [enc setBytes:&uN length:4 atIndex:5];
            NSUInteger groupsX = (N + g_sf64_blk - 1) / g_sf64_blk;
            NSUInteger groupsY = (M + g_sf64_blk - 1) / g_sf64_blk;
            [enc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1)
                    threadsPerThreadgroup:MTLSizeMake(g_sf64_tg, g_sf64_tg, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            int result = -4;
            if ([cmd status] == MTLCommandBufferStatusCompleted) {
                double gpu_ms = ([cmd GPUEndTime] - [cmd GPUStartTime]) * 1000.0;
                double gflops = (2.0 * M * K * N) / (gpu_ms / 1000.0) / 1e9;
                fprintf(stderr, "[GPU] df64 legacy completed: %.1fms %.0f GFLOPS (M=%u K=%u N=%u)\n",
                        gpu_ms, gflops, uM, uK, uN);
                memcpy(C->host_ptr, [buf_c contents], elementsC * 8);
                result = 0;
            } else if ([cmd status] == MTLCommandBufferStatusError) {
                NSError* err = [cmd error];
                fprintf(stderr, "[GPU] df64 legacy error (code=%ld): %s\n",
                        (long)[err code], err ? [[err localizedDescription] UTF8String] : "unknown");
                result = -3;
            }
            pool_release(buf_a); pool_release(buf_b); pool_release(buf_c);
            return result;
        }

        // Pure df64 pipeline with separate conversion
        size_t elementsA = M * K;
        size_t elementsB = K * N;
        size_t elementsC = M * N;

        // f64 buffers (host format)
        id<MTLBuffer> f64_a = pool_alloc(elementsA * 8);
        id<MTLBuffer> f64_b = pool_alloc(elementsB * 8);
        id<MTLBuffer> f64_c = pool_alloc(elementsC * 8);
        // df64 buffers (working format: float2 = 8 bytes each, same size)
        id<MTLBuffer> df64_a = pool_alloc(elementsA * 8);
        id<MTLBuffer> df64_b = pool_alloc(elementsB * 8);
        id<MTLBuffer> df64_c = pool_alloc(elementsC * 8);

        if (!f64_a || !f64_b || !f64_c || !df64_a || !df64_b || !df64_c) {
            if (f64_a) pool_release(f64_a); if (f64_b) pool_release(f64_b); if (f64_c) pool_release(f64_c);
            if (df64_a) pool_release(df64_a); if (df64_b) pool_release(df64_b); if (df64_c) pool_release(df64_c);
            return -1;
        }

        // Host → GPU (f64 raw)
        memcpy([f64_a contents], A->host_ptr, elementsA * 8);
        memcpy([f64_b contents], B->host_ptr, elementsB * 8);

        // Single command buffer: convert → matmul → convert
        MTLCommandBufferDescriptor* desc = [[MTLCommandBufferDescriptor alloc] init];
        desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
        id<MTLCommandBuffer> cmd = [g_metal_queue commandBufferWithDescriptor:desc];

        // Encoder 1: f64 → df64 conversion for A and B
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_convert_f64_to_df64_pipeline];

            [enc setBuffer:f64_a offset:0 atIndex:0];
            [enc setBuffer:df64_a offset:0 atIndex:1];
            uint32_t countA = (uint32_t)elementsA;
            [enc setBytes:&countA length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((elementsA + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setBuffer:f64_b offset:0 atIndex:0];
            [enc setBuffer:df64_b offset:0 atIndex:1];
            uint32_t countB = (uint32_t)elementsB;
            [enc setBytes:&countB length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((elementsB + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc endEncoding];
        }

        // Encoder 2: Pure df64 matmul (config-driven tile size and threads)
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_matmul_df64_pure_pipeline];
            [enc setBuffer:df64_a offset:0 atIndex:0];
            [enc setBuffer:df64_b offset:0 atIndex:1];
            [enc setBuffer:df64_c offset:0 atIndex:2];
            uint32_t uM = (uint32_t)M, uK = (uint32_t)K, uN = (uint32_t)N;
            [enc setBytes:&uM length:4 atIndex:3];
            [enc setBytes:&uK length:4 atIndex:4];
            [enc setBytes:&uN length:4 atIndex:5];

            NSUInteger groupsX = (N + g_cfg_df64.bn - 1) / g_cfg_df64.bn;
            NSUInteger groupsY = (M + g_cfg_df64.bm - 1) / g_cfg_df64.bm;
            [enc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1)
                    threadsPerThreadgroup:MTLSizeMake(g_cfg_df64.tg, g_cfg_df64.tg, 1)];
            [enc endEncoding];
        }

        // Encoder 3: df64 → f64 conversion for C
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_convert_df64_to_f64_pipeline];
            [enc setBuffer:df64_c offset:0 atIndex:0];
            [enc setBuffer:f64_c offset:0 atIndex:1];
            uint32_t countC = (uint32_t)elementsC;
            [enc setBytes:&countC length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((elementsC + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        int result = -4;
        if ([cmd status] == MTLCommandBufferStatusCompleted) {
            double gpu_ms = ([cmd GPUEndTime] - [cmd GPUStartTime]) * 1000.0;
            double gflops = (2.0 * M * K * N) / (gpu_ms / 1000.0) / 1e9;
            fprintf(stderr, "[GPU] df64 completed: %.1fms %.0f GFLOPS (M=%u K=%u N=%u)\n",
                    gpu_ms, gflops, (uint32_t)M, (uint32_t)K, (uint32_t)N);
            memcpy(C->host_ptr, [f64_c contents], elementsC * 8);
            result = 0;
        } else if ([cmd status] == MTLCommandBufferStatusError) {
            NSError* err = [cmd error];
            fprintf(stderr, "[GPU] df64 error (code=%ld): %s\n",
                    (long)[err code], err ? [[err localizedDescription] UTF8String] : "unknown");
            result = -3;
        }

        pool_release(f64_a); pool_release(f64_b); pool_release(f64_c);
        pool_release(df64_a); pool_release(df64_b); pool_release(df64_c);
        return result;
    }
}

// Tier 2 (SIMD): f32 dispatch with SEPARATE conversion pipeline
// Architecture: f64 buf → GPU convert_f64_to_f32 → f32 buf → pure f32 SIMD → f32 buf → GPU convert_f32_to_f64 → f64 buf
static int metal_matmul_f32_simd_full_dispatch(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                                                uint64_t M, uint64_t K, uint64_t N) {
    @autoreleasepool {
        if (!g_matmul_f32_simd_pipeline || !g_convert_f64_to_f32_pipeline || !g_convert_f32_to_f64_pipeline)
            return -1;

        size_t elementsA = M * K;
        size_t elementsB = K * N;
        size_t elementsC = M * N;

        // Allocate f64 buffers (input/output format)
        id<MTLBuffer> f64_a = pool_alloc(elementsA * 8);
        id<MTLBuffer> f64_b = pool_alloc(elementsB * 8);
        id<MTLBuffer> f64_c = pool_alloc(elementsC * 8);
        // Allocate f32 buffers (kernel working format — half the size)
        id<MTLBuffer> f32_a = pool_alloc(elementsA * 4);
        id<MTLBuffer> f32_b = pool_alloc(elementsB * 4);
        id<MTLBuffer> f32_c = pool_alloc(elementsC * 4);

        if (!f64_a || !f64_b || !f64_c || !f32_a || !f32_b || !f32_c) {
            if (f64_a) pool_release(f64_a); if (f64_b) pool_release(f64_b); if (f64_c) pool_release(f64_c);
            if (f32_a) pool_release(f32_a); if (f32_b) pool_release(f32_b); if (f32_c) pool_release(f32_c);
            return -1;
        }

        // Step 1: Host → GPU (f64 format)
        memcpy([f64_a contents], A->host_ptr, elementsA * 8);
        memcpy([f64_b contents], B->host_ptr, elementsB * 8);

        // ALL GPU work in a SINGLE command buffer: convert → matmul → convert back
        // This eliminates 2 extra commit+wait cycles (~10-50ms each)
        MTLCommandBufferDescriptor* desc = [[MTLCommandBufferDescriptor alloc] init];
        desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
        id<MTLCommandBuffer> cmd = [g_metal_queue commandBufferWithDescriptor:desc];

        // Encoder 1: f64→f32 conversion for A and B
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_convert_f64_to_f32_pipeline];

            [enc setBuffer:f64_a offset:0 atIndex:0];
            [enc setBuffer:f32_a offset:0 atIndex:1];
            uint32_t countA = (uint32_t)elementsA;
            [enc setBytes:&countA length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((elementsA + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setBuffer:f64_b offset:0 atIndex:0];
            [enc setBuffer:f32_b offset:0 atIndex:1];
            uint32_t countB = (uint32_t)elementsB;
            [enc setBytes:&countB length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((elementsB + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc endEncoding];
        }

        // Encoder 2: Pure f32 SIMD matmul (config-driven tile size and threads)
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            uint32_t uM = (uint32_t)M, uK = (uint32_t)K, uN = (uint32_t)N;
            [enc setBuffer:f32_a offset:0 atIndex:0];
            [enc setBuffer:f32_b offset:0 atIndex:1];
            [enc setBuffer:f32_c offset:0 atIndex:2];
            [enc setBytes:&uM length:4 atIndex:3];
            [enc setBytes:&uK length:4 atIndex:4];
            [enc setBytes:&uN length:4 atIndex:5];

            // 128×128 kernel disabled: scalar loads limit throughput vs 64×64 with
            // vectorized float4 loads. Will re-enable once 128×128 is optimized.
            // Env var ESHKOL_F32S_USE_128=1 forces it for benchmarking.
            bool use_128 = false;
            if (const char* v = std::getenv("ESHKOL_F32S_USE_128")) {
                use_128 = g_matmul_f32_simd_128_pipeline && atoi(v) == 1;
            }
            if (use_128) {
                // Large tile kernel (config-driven)
                [enc setComputePipelineState:g_matmul_f32_simd_128_pipeline];
                NSUInteger groupsX = (N + g_cfg_f32s_128.bn - 1) / g_cfg_f32s_128.bn;
                NSUInteger groupsY = (M + g_cfg_f32s_128.bm - 1) / g_cfg_f32s_128.bm;
                [enc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1)
                        threadsPerThreadgroup:MTLSizeMake(g_cfg_f32s_128.threads, 1, 1)];
            } else {
                // Standard tile kernel (config-driven)
                [enc setComputePipelineState:g_matmul_f32_simd_pipeline];
                NSUInteger groupsX = (N + g_cfg_f32s.bn - 1) / g_cfg_f32s.bn;
                NSUInteger groupsY = (M + g_cfg_f32s.bm - 1) / g_cfg_f32s.bm;
                [enc dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1)
                        threadsPerThreadgroup:MTLSizeMake(g_cfg_f32s.threads, 1, 1)];
            }
            [enc endEncoding];
        }

        // Encoder 3: f32→f64 conversion for C
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_convert_f32_to_f64_pipeline];
            [enc setBuffer:f32_c offset:0 atIndex:0];
            [enc setBuffer:f64_c offset:0 atIndex:1];
            uint32_t countC = (uint32_t)elementsC;
            [enc setBytes:&countC length:4 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake((elementsC + 255) / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Single commit+wait for the entire pipeline
        [cmd commit];
        [cmd waitUntilCompleted];

        int result = -4;
        if ([cmd status] == MTLCommandBufferStatusCompleted) {
            double gpu_ms = ([cmd GPUEndTime] - [cmd GPUStartTime]) * 1000.0;
            double gflops = (2.0 * M * K * N) / (gpu_ms / 1000.0) / 1e9;
            fprintf(stderr, "[GPU] f32_simd completed: %.1fms %.0f GFLOPS (M=%lu K=%lu N=%lu)\n",
                    gpu_ms, gflops, (unsigned long)M, (unsigned long)K, (unsigned long)N);
            // Copy result back
            memcpy(C->host_ptr, [f64_c contents], elementsC * 8);
            result = 0;
        } else if ([cmd status] == MTLCommandBufferStatusError) {
            NSError* err = [cmd error];
            fprintf(stderr, "[GPU] f32_simd error (code=%ld): %s\n",
                    (long)[err code], err ? [[err localizedDescription] UTF8String] : "unknown");
            result = -3;
        }

        pool_release(f64_a); pool_release(f64_b); pool_release(f64_c);
        pool_release(f32_a); pool_release(f32_b); pool_release(f32_c);
        return result;
    }
}

// ============================================================================
// Fixed-Point ML (fp24) Dispatch V2 — CPU max_exp + Single Signed Accumulator
// ============================================================================
// Pipeline (V2 optimized):
//   1. CPU scans max exponent during data copy (data hot in cache, ~3ms for 8K)
//   2. matmul_fp24 V2: signed accumulator, sB stride padding, no GPU pre-pass
// Single command buffer, row-blocking for M > 4096.

static int metal_matmul_fp24_dispatch(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                                       uint64_t M, uint64_t K, uint64_t N) {
    if (!g_matmul_fp24_pipeline) return -1;

    @autoreleasepool {
        size_t elementsA = M * K;
        size_t elementsB = K * N;
        size_t elementsC = M * N;

        // Allocate GPU buffers
        id<MTLBuffer> buf_a = pool_alloc(elementsA * 8);
        id<MTLBuffer> buf_b = pool_alloc(elementsB * 8);
        id<MTLBuffer> buf_c = pool_alloc(elementsC * 8);

        // Small buffer for max exponents: [0]=max_exp_A, [1]=max_exp_B
        id<MTLBuffer> max_exp_buf = pool_alloc(8);  // 2 × uint32

        if (!buf_a || !buf_b || !buf_c || !max_exp_buf) {
            if (buf_a) pool_release(buf_a);
            if (buf_b) pool_release(buf_b);
            if (buf_c) pool_release(buf_c);
            if (max_exp_buf) pool_release(max_exp_buf);
            return -2;
        }

        // Copy input data to GPU buffers
        memcpy([buf_a contents], A->host_ptr, elementsA * 8);
        memcpy([buf_b contents], B->host_ptr, elementsB * 8);

        // === CPU-side max exponent scan ===
        // Data is hot in cache from memcpy. On UMA (Apple Silicon), this reads
        // from the same physical memory the GPU will use. ~3ms for 67M elements.
        // Eliminates 2 GPU command encoders and their atomic contention overhead.
        {
            uint32_t max_exp_a = 0, max_exp_b = 0;
            const uint64_t* a_data = (const uint64_t*)A->host_ptr;
            const uint64_t* b_data = (const uint64_t*)B->host_ptr;
            for (size_t i = 0; i < elementsA; i++) {
                uint32_t exp = (uint32_t)((a_data[i] >> 52) & 0x7FF);
                if (exp > max_exp_a) max_exp_a = exp;
            }
            for (size_t i = 0; i < elementsB; i++) {
                uint32_t exp = (uint32_t)((b_data[i] >> 52) & 0x7FF);
                if (exp > max_exp_b) max_exp_b = exp;
            }
            uint32_t* exp_ptr = (uint32_t*)[max_exp_buf contents];
            exp_ptr[0] = max_exp_a;
            exp_ptr[1] = max_exp_b;
        }

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        uint32_t M32 = (uint32_t)M, K32 = (uint32_t)K, N32 = (uint32_t)N;

        // === matmul_fp24 V2 (with adaptive row-blocking) ===
        uint32_t chunk_M = compute_chunk_m(KERNEL_FP24, K32, g_cfg_fp24.bm);
        if (M32 <= chunk_M) chunk_M = M32;

        for (uint32_t m_off = 0; m_off < M32; m_off += chunk_M) {
            uint32_t m_len = (m_off + chunk_M <= M32) ? chunk_M : (M32 - m_off);

            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_matmul_fp24_pipeline];
            [enc setBuffer:buf_a offset:(size_t)m_off * K * 8 atIndex:0];
            [enc setBuffer:buf_b offset:0 atIndex:1];
            [enc setBuffer:buf_c offset:(size_t)m_off * N * 8 atIndex:2];
            [enc setBytes:&m_len length:sizeof(m_len) atIndex:3];
            [enc setBytes:&K32 length:sizeof(K32) atIndex:4];
            [enc setBytes:&N32 length:sizeof(N32) atIndex:5];
            [enc setBuffer:max_exp_buf offset:0 atIndex:6];

            MTLSize tg_size = MTLSizeMake(g_cfg_fp24.threads, 1, 1);
            NSUInteger grid_x = (N32 + g_cfg_fp24.bn - 1) / g_cfg_fp24.bn;
            NSUInteger grid_y = (m_len + g_cfg_fp24.bm - 1) / g_cfg_fp24.bm;
            [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
                threadsPerThreadgroup:tg_size];
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        int result = -1;
        if ([cmd status] == MTLCommandBufferStatusCompleted) {
            memcpy(C->host_ptr, [buf_c contents], elementsC * 8);
            result = 0;
        } else if ([cmd status] == MTLCommandBufferStatusError) {
            NSError* err = [cmd error];
            fprintf(stderr, "[GPU] fp24 matmul error (code=%ld): %s\n",
                    (long)[err code], err ? [[err localizedDescription] UTF8String] : "unknown");
            result = -3;
        }

        pool_release(buf_a); pool_release(buf_b); pool_release(buf_c);
        pool_release(max_exp_buf);
        return result;
    }
}

// ============================================================================
// Fixed-Point Exact (fp53) Dispatch — Full 53-bit mantissa, 128-bit accumulation
// ============================================================================
// Same dispatch pattern as fp24: CPU max_exp scan, row-blocking for large M.

static int metal_matmul_fp53_dispatch(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                                       uint64_t M, uint64_t K, uint64_t N) {
    if (!g_matmul_fp53_pipeline) return -1;

    @autoreleasepool {
        size_t elementsA = M * K;
        size_t elementsB = K * N;
        size_t elementsC = M * N;

        id<MTLBuffer> buf_a = pool_alloc(elementsA * 8);
        id<MTLBuffer> buf_b = pool_alloc(elementsB * 8);
        id<MTLBuffer> buf_c = pool_alloc(elementsC * 8);
        id<MTLBuffer> max_exp_buf = pool_alloc(8);  // 2 × uint32

        if (!buf_a || !buf_b || !buf_c || !max_exp_buf) {
            if (buf_a) pool_release(buf_a);
            if (buf_b) pool_release(buf_b);
            if (buf_c) pool_release(buf_c);
            if (max_exp_buf) pool_release(max_exp_buf);
            return -2;
        }

        memcpy([buf_a contents], A->host_ptr, elementsA * 8);
        memcpy([buf_b contents], B->host_ptr, elementsB * 8);

        // CPU-side max exponent scan (same as fp24)
        {
            uint32_t max_exp_a = 0, max_exp_b = 0;
            const uint64_t* a_data = (const uint64_t*)A->host_ptr;
            const uint64_t* b_data = (const uint64_t*)B->host_ptr;
            for (size_t i = 0; i < elementsA; i++) {
                uint32_t exp = (uint32_t)((a_data[i] >> 52) & 0x7FF);
                if (exp > max_exp_a) max_exp_a = exp;
            }
            for (size_t i = 0; i < elementsB; i++) {
                uint32_t exp = (uint32_t)((b_data[i] >> 52) & 0x7FF);
                if (exp > max_exp_b) max_exp_b = exp;
            }
            uint32_t* exp_ptr = (uint32_t*)[max_exp_buf contents];
            exp_ptr[0] = max_exp_a;
            exp_ptr[1] = max_exp_b;
        }

        uint32_t M32 = (uint32_t)M, K32 = (uint32_t)K, N32 = (uint32_t)N;

        // Adaptive row-blocking for large M — separate command buffers per chunk
        // to avoid Metal GPU watchdog timeout. fp53 with 128-bit accumulation is
        // ~3x heavier per FMA than fp24, so we use compute_chunk_m for sizing.
        uint32_t chunk_M = compute_chunk_m(KERNEL_FP53, K32, g_cfg_fp53.bm);
        if (M32 <= chunk_M) chunk_M = M32;

        int result = 0;
        for (uint32_t m_off = 0; m_off < M32; m_off += chunk_M) {
            uint32_t m_len = (m_off + chunk_M <= M32) ? chunk_M : (M32 - m_off);

            id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_matmul_fp53_pipeline];
            [enc setBuffer:buf_a offset:(size_t)m_off * K * 8 atIndex:0];
            [enc setBuffer:buf_b offset:0 atIndex:1];
            [enc setBuffer:buf_c offset:(size_t)m_off * N * 8 atIndex:2];
            [enc setBytes:&m_len length:sizeof(m_len) atIndex:3];
            [enc setBytes:&K32 length:sizeof(K32) atIndex:4];
            [enc setBytes:&N32 length:sizeof(N32) atIndex:5];
            [enc setBuffer:max_exp_buf offset:0 atIndex:6];

            MTLSize tg_size = MTLSizeMake(g_cfg_fp53.threads, 1, 1);
            NSUInteger grid_x = (N32 + g_cfg_fp53.bn - 1) / g_cfg_fp53.bn;
            NSUInteger grid_y = (m_len + g_cfg_fp53.bm - 1) / g_cfg_fp53.bm;
            [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
                threadsPerThreadgroup:tg_size];
            [enc endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];

            if ([cmd status] != MTLCommandBufferStatusCompleted) {
                NSError* err = [cmd error];
                fprintf(stderr, "[GPU] fp53 matmul error (code=%ld): %s\n",
                        (long)[err code], err ? [[err localizedDescription] UTF8String] : "unknown");
                result = -3;
                break;
            }
        }

        if (result == 0) {
            memcpy(C->host_ptr, [buf_c contents], elementsC * 8);
        }

        pool_release(buf_a); pool_release(buf_b); pool_release(buf_c);
        pool_release(max_exp_buf);
        return result;
    }
}

// ============================================================================
// Ozaki-II Exact DGEMM — Chinese Remainder Theorem + f32 simdgroup MMA
// ============================================================================
// Algorithm: Scale inputs to integers, decompose mod N coprime moduli,
// run N independent f32 GEMMs via simdgroup MMA, reconstruct via CRT.
// Features: Adaptive N selection (minimum moduli for input data range),
//           pipelined dispatch (overlapped CPU CRT with GPU GEMMs).
// References: Ozaki et al. 2504.08009v3, Uchino et al. 2602.02549v1

static int ozaki_ii_dispatch(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                              uint64_t M, uint64_t K, uint64_t N_cols) {
    if (!g_matmul_ozaki_gemm_pipeline) return -1;

    @autoreleasepool {
        const double* a_data = (const double*)A->host_ptr;
        const double* b_data = (const double*)B->host_ptr;
        dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

        // --- Adaptive N: choose minimum moduli for this input's data range ---
        int max_N = g_ozaki_num_moduli;
        const char* adapt_env = std::getenv("ESHKOL_OZAKI_ADAPTIVE");
        bool adaptive = !adapt_env || strcmp(adapt_env, "0") != 0;
        int num_moduli = adaptive
            ? ozaki_compute_adaptive_N(a_data, b_data, M, K, N_cols, max_N)
            : max_N;

        const OzakiCRTConstants& crt = get_ozaki_crt(num_moduli);

        fprintf(stderr, "[GPU] Ozaki-II: N=%d%s, log2(P)=%.1f\n",
                num_moduli, adaptive ? " (adaptive)" : "", crt.log2P);

        // --- CPU Step 1: Compute scaling vectors (parallel by row/col) ---
        int E = (int)floor((crt.log2P - 1.0 - log2((double)K)) / 2.0);
        if (E > 52) E = 52;
        if (E < 0) E = 0;

        std::vector<int> mu(M), nu(N_cols);
        int* mu_p = mu.data();
        int* nu_p = nu.data();
        dispatch_apply(M, q, ^(size_t i) {
            double max_abs = 0.0;
            for (size_t j = 0; j < K; j++) {
                double v = fabs(a_data[i * K + j]);
                if (v > max_abs) max_abs = v;
            }
            mu_p[i] = (max_abs == 0.0) ? 0 : E - (int)floor(log2(max_abs));
        });
        dispatch_apply(N_cols, q, ^(size_t j) {
            double max_abs = 0.0;
            for (size_t h = 0; h < K; h++) {
                double v = fabs(b_data[h * N_cols + j]);
                if (v > max_abs) max_abs = v;
            }
            nu_p[j] = (max_abs == 0.0) ? 0 : E - (int)floor(log2(max_abs));
        });

        // --- CPU Step 2: Scale to doubles (no int64 — all f64 exact arithmetic) ---
        std::vector<double> A_scaled(M * K), B_scaled(K * N_cols);
        double* as_p = A_scaled.data();
        double* bs_p = B_scaled.data();
        dispatch_apply(M, q, ^(size_t i) {
            double scale = ldexp(1.0, mu_p[i]);
            for (size_t j = 0; j < K; j++)
                as_p[i * K + j] = trunc(a_data[i * K + j] * scale);
        });
        dispatch_apply(N_cols, q, ^(size_t j) {
            double scale = ldexp(1.0, nu_p[j]);
            for (size_t h = 0; h < K; h++)
                bs_p[h * N_cols + j] = trunc(b_data[h * N_cols + j] * scale);
        });

        // --- Allocate per-modulus GPU buffers for batched dispatch ---
        size_t al_bytes = M * K * sizeof(float);
        size_t bl_bytes = K * N_cols * sizeof(float);
        size_t wl_bytes = M * N_cols * sizeof(float);
        size_t out_size = M * N_cols;
        size_t total_batch = (size_t)num_moduli * (al_bytes + bl_bytes + wl_bytes);

        // Use batched path if total allocation < 4GB, else fall back to serial
        bool use_batch = (total_batch < (size_t)4 * 1024 * 1024 * 1024);

        if (use_batch) {
            id<MTLBuffer> all_al = pool_alloc(al_bytes * num_moduli);
            id<MTLBuffer> all_bl = pool_alloc(bl_bytes * num_moduli);
            id<MTLBuffer> all_wl = pool_alloc(wl_bytes * num_moduli);

            if (!all_al || !all_bl || !all_wl) {
                if (all_al) pool_release(all_al);
                if (all_bl) pool_release(all_bl);
                if (all_wl) pool_release(all_wl);
                use_batch = false;
            } else {
                // --- CPU: Parallel mod-reduce all moduli at once ---
                uint8_t* al_base = (uint8_t*)[all_al contents];
                uint8_t* bl_base = (uint8_t*)[all_bl contents];
                dispatch_apply(num_moduli, q, ^(size_t l) {
                    int p = OZAKI_MODULI[l];
                    double p_d = (double)p;
                    double p_inv = 1.0 / p_d;
                    float* al = (float*)(al_base + l * al_bytes);
                    float* bl = (float*)(bl_base + l * bl_bytes);
                    for (size_t i = 0; i < M * K; i++)
                        al[i] = (float)(as_p[i] - p_d * rint(as_p[i] * p_inv));
                    for (size_t i = 0; i < K * N_cols; i++)
                        bl[i] = (float)(bs_p[i] - p_d * rint(bs_p[i] * p_inv));
                });

                // --- GPU: Batch all N GEMMs in a single command buffer ---
                uint32_t M32 = (uint32_t)M, K32 = (uint32_t)K, N32 = (uint32_t)N_cols;
                MTLSize tg_size = MTLSizeMake(g_cfg_f32s.threads, 1, 1);
                NSUInteger grid_x = (N32 + g_cfg_f32s.bn - 1) / g_cfg_f32s.bn;
                NSUInteger grid_y = (M32 + g_cfg_f32s.bm - 1) / g_cfg_f32s.bm;

                id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
                for (int l = 0; l < num_moduli; l++) {
                    uint32_t p32 = (uint32_t)OZAKI_MODULI[l];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    [enc setComputePipelineState:g_matmul_ozaki_gemm_pipeline];
                    [enc setBuffer:all_al offset:l * al_bytes atIndex:0];
                    [enc setBuffer:all_bl offset:l * bl_bytes atIndex:1];
                    [enc setBuffer:all_wl offset:l * wl_bytes atIndex:2];
                    [enc setBytes:&M32 length:4 atIndex:3];
                    [enc setBytes:&K32 length:4 atIndex:4];
                    [enc setBytes:&N32 length:4 atIndex:5];
                    [enc setBytes:&p32 length:4 atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
                        threadsPerThreadgroup:tg_size];
                    [enc endEncoding];
                }
                [cmd commit];
                [cmd waitUntilCompleted];

                if ([cmd status] != MTLCommandBufferStatusCompleted) {
                    NSError* err = [cmd error];
                    fprintf(stderr, "[GPU] ozaki-ii batch error (code=%ld): %s\n",
                            (long)[err code], err ? [[err localizedDescription] UTF8String] : "unknown");
                    pool_release(all_al); pool_release(all_bl); pool_release(all_wl);
                    return -3;
                }

                // --- CPU: Parallel CRT accumulation ---
                std::vector<double> S1(out_size, 0.0), S2(out_size, 0.0);
                double* S1_p = S1.data();
                double* S2_p = S2.data();
                const uint8_t* wl_base = (const uint8_t*)[all_wl contents];
                const double* sl1_arr = crt.sl1;
                const double* sl2_arr = crt.sl2;

                size_t nchunks = 64;
                size_t chunk = (out_size + nchunks - 1) / nchunks;
                dispatch_apply(nchunks, q, ^(size_t c) {
                    size_t start = c * chunk;
                    size_t end = start + chunk;
                    if (end > out_size) end = out_size;
                    for (int l = 0; l < num_moduli; l++) {
                        const float* wl = (const float*)(wl_base + l * wl_bytes);
                        double sl1 = sl1_arr[l], sl2 = sl2_arr[l];
                        for (size_t i = start; i < end; i++) {
                            double w = (double)wl[i];
                            S1_p[i] += sl1 * w;
                            S2_p[i] += sl2 * w;
                        }
                    }
                });

                // --- CPU: Parallel CRT reconstruction + inverse scaling ---
                double* c_data = (double*)C->host_ptr;
                double P1 = crt.P1, P2 = crt.P2, Pinv = crt.Pinv;
                dispatch_apply(M, q, ^(size_t i) {
                    double mu_inv = ldexp(1.0, -mu_p[i]);
                    for (size_t j = 0; j < N_cols; j++) {
                        size_t idx = i * N_cols + j;
                        double s = S1_p[idx] + S2_p[idx];
                        double Q = round(s * Pinv);
                        double c_prime = fma(-P1, Q, S1_p[idx]) + fma(-P2, Q, S2_p[idx]);
                        c_data[idx] = c_prime * mu_inv * ldexp(1.0, -nu_p[j]);
                    }
                });

                pool_release(all_al); pool_release(all_bl); pool_release(all_wl);
                return 0;
            }
        }

        // --- Serial fallback (memory-constrained or batch alloc failed) ---
        id<MTLBuffer> al_buf = pool_alloc(al_bytes);
        id<MTLBuffer> bl_buf = pool_alloc(bl_bytes);
        id<MTLBuffer> wl_buf = pool_alloc(wl_bytes);

        if (!al_buf || !bl_buf || !wl_buf) {
            if (al_buf) pool_release(al_buf);
            if (bl_buf) pool_release(bl_buf);
            if (wl_buf) pool_release(wl_buf);
            return -2;
        }

        std::vector<double> S1(out_size, 0.0), S2(out_size, 0.0);
        int result = 0;
        uint32_t M32 = (uint32_t)M, K32 = (uint32_t)K, N32 = (uint32_t)N_cols;

        for (int l = 0; l < num_moduli && result == 0; l++) {
            int p = OZAKI_MODULI[l];
            double p_d = (double)p;
            double p_inv = 1.0 / p_d;

            float* al = (float*)[al_buf contents];
            float* bl = (float*)[bl_buf contents];
            size_t a_size = M * K, b_size = K * N_cols;
            size_t nchk = 16;
            size_t a_chunk = (a_size + nchk - 1) / nchk;
            size_t b_chunk = (b_size + nchk - 1) / nchk;
            dispatch_apply(nchk, q, ^(size_t c) {
                size_t start = c * a_chunk;
                size_t end = start + a_chunk;
                if (end > a_size) end = a_size;
                for (size_t i = start; i < end; i++)
                    al[i] = (float)(as_p[i] - p_d * rint(as_p[i] * p_inv));
            });
            dispatch_apply(nchk, q, ^(size_t c) {
                size_t start = c * b_chunk;
                size_t end = start + b_chunk;
                if (end > b_size) end = b_size;
                for (size_t i = start; i < end; i++)
                    bl[i] = (float)(bs_p[i] - p_d * rint(bs_p[i] * p_inv));
            });

            uint32_t p32 = (uint32_t)p;
            id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_matmul_ozaki_gemm_pipeline];
            [enc setBuffer:al_buf offset:0 atIndex:0];
            [enc setBuffer:bl_buf offset:0 atIndex:1];
            [enc setBuffer:wl_buf offset:0 atIndex:2];
            [enc setBytes:&M32 length:4 atIndex:3];
            [enc setBytes:&K32 length:4 atIndex:4];
            [enc setBytes:&N32 length:4 atIndex:5];
            [enc setBytes:&p32 length:4 atIndex:6];

            MTLSize tg_size = MTLSizeMake(g_cfg_f32s.threads, 1, 1);
            NSUInteger grid_x = (N32 + g_cfg_f32s.bn - 1) / g_cfg_f32s.bn;
            NSUInteger grid_y = (M32 + g_cfg_f32s.bm - 1) / g_cfg_f32s.bm;
            [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
                threadsPerThreadgroup:tg_size];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            if ([cmd status] != MTLCommandBufferStatusCompleted) {
                result = -3; break;
            }

            const float* wl = (const float*)[wl_buf contents];
            double sl1 = crt.sl1[l], sl2 = crt.sl2[l];
            for (size_t i = 0; i < out_size; i++) {
                S1[i] += sl1 * (double)wl[i];
                S2[i] += sl2 * (double)wl[i];
            }
        }

        if (result == 0) {
            double* c_data = (double*)C->host_ptr;
            double P1 = crt.P1, P2 = crt.P2, Pinv = crt.Pinv;
            double* s1_p = S1.data();
            double* s2_p = S2.data();
            dispatch_apply(M, q, ^(size_t i) {
                double mu_inv = ldexp(1.0, -mu_p[i]);
                for (size_t j = 0; j < N_cols; j++) {
                    size_t idx = i * N_cols + j;
                    double s = s1_p[idx] + s2_p[idx];
                    double Q = round(s * Pinv);
                    double c_prime = fma(-P1, Q, s1_p[idx]) + fma(-P2, Q, s2_p[idx]);
                    c_data[idx] = c_prime * mu_inv * ldexp(1.0, -nu_p[j]);
                }
            });
        }

        pool_release(al_buf); pool_release(bl_buf); pool_release(wl_buf);
        return result;
    }
}

// Router: dispatches to the correct tier-specific function
static int metal_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    switch (g_metal_precision_tier) {
        case 1:
            return metal_matmul_df64_dispatch(A, B, C, M, K, N);
        case 2: {
            const char* f32_env = std::getenv("ESHKOL_F32_KERNEL");
            bool use_simd = f32_env && strcmp(f32_env, "simd") == 0 && g_matmul_f32_simd_pipeline;
            return use_simd ? metal_matmul_f32_simd_full_dispatch(A, B, C, M, K, N)
                            : metal_matmul_f64_via_mps(A, B, C, M, K, N);
        }
        case 3:
            return metal_matmul_fp24_dispatch(A, B, C, M, K, N);
        default: {
            // Tier 0: fp53 default, Ozaki-II opt-in for exact DGEMM
            const char* sf64_env = std::getenv("ESHKOL_SF64_KERNEL");
            bool use_legacy = sf64_env && (strcmp(sf64_env, "legacy") == 0 || strcmp(sf64_env, "v2") == 0);
            bool use_fp53 = sf64_env && strcmp(sf64_env, "fp53") == 0;
            bool force_ozaki = sf64_env && strcmp(sf64_env, "ozaki") == 0;
            // Ozaki-II: CRT-based exact f64 via N f32 GEMMs (opt-in)
            if (force_ozaki && g_matmul_ozaki_gemm_pipeline
                && M >= 512 && K >= 512 && N >= 512)
                return ozaki_ii_dispatch(A, B, C, M, K, N);
            if (!use_legacy && g_matmul_fp53_pipeline)
                return metal_matmul_fp53_dispatch(A, B, C, M, K, N);
            return metal_matmul_sf64_dispatch(A, B, C, M, K, N);
        }
    }
}

static int metal_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    @autoreleasepool {
        id<MTLBuffer> buf_a = (__bridge id<MTLBuffer>)A->backend_data;
        id<MTLBuffer> buf_b = (__bridge id<MTLBuffer>)B->backend_data;
        id<MTLBuffer> buf_c = (__bridge id<MTLBuffer>)C->backend_data;

        // Use MPS for f32 (highly optimized)
        MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
                                                                           rowBytes:K * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N
                                                                           rowBytes:N * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
                                                                           rowBytes:N * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];

        MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a descriptor:desc_a];
        MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b descriptor:desc_b];
        MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c descriptor:desc_c];

        MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_metal_device
            transposeLeft:NO transposeRight:NO
            resultRows:M resultColumns:N interiorColumns:K
            alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        [matmul encodeToCommandBuffer:cmd leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

static int metal_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [g_metal_device newBufferWithBytesNoCopy:host_ptr
                                                                 length:size_bytes
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        if (!buffer) {
            // Fallback: allocate and copy
            int result = metal_alloc(size_bytes, ESHKOL_MEM_UNIFIED, out);
            if (result != 0) return result;
            memcpy(out->host_ptr, host_ptr, size_bytes);
            return 0;
        }

        out->host_ptr = host_ptr;
        out->device_ptr = host_ptr;
        out->size_bytes = size_bytes;
        out->mem_type = ESHKOL_MEM_UNIFIED;
        out->backend = ESHKOL_GPU_METAL;
        out->flags = 1;  // Wrapped, don't free underlying memory
        out->backend_data = (__bridge_retained void*)buffer;

        return 0;
    }
}

static int metal_elementwise_f64(EshkolGPUBuffer* a, EshkolGPUBuffer* b,
                                  EshkolGPUBuffer* out, uint64_t n,
                                  int op) {
    @autoreleasepool {
        if (!g_elementwise_sf64_pipeline) return -1;

        bool has_b = (b && b->host_ptr && op <= 3);
        id<MTLBuffer> buf_a = pool_alloc(n * 8);
        id<MTLBuffer> buf_b = pool_alloc(n * 8);
        id<MTLBuffer> buf_c = pool_alloc(n * 8);
        if (!buf_a || !buf_b || !buf_c) {
            if (buf_a) pool_release(buf_a);
            if (buf_b) pool_release(buf_b);
            return -1;
        }

        // Direct memcpy — raw f64 bytes (kernel handles byte-swap inline)
        memcpy([buf_a contents], a->host_ptr, n * 8);
        if (has_b) {
            memcpy([buf_b contents], b->host_ptr, n * 8);
        }

        // Single kernel dispatch — no word-swap passes needed
        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_elementwise_sf64_pipeline];
        [encoder setBuffer:buf_a offset:0 atIndex:0];
        [encoder setBuffer:buf_b offset:0 atIndex:1];
        [encoder setBuffer:buf_c offset:0 atIndex:2];
        uint32_t n32 = static_cast<uint32_t>(n);
        uint32_t op32 = static_cast<uint32_t>(op);
        [encoder setBytes:&n32 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&op32 length:sizeof(uint32_t) atIndex:4];

        NSUInteger groups = (n + 255) / 256;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out->host_ptr, [buf_c contents], n * 8);
        pool_release(buf_a);
        pool_release(buf_b);
        pool_release(buf_c);
        return 0;
    }
}

static int metal_reduce_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                             uint64_t n, int op) {
    @autoreleasepool {
        if (!g_reduce_sf64_pipeline) return -1;

        // Two-pass reduction: first pass reduces to partial results per threadgroup
        uint64_t groups = (n + 255) / 256;

        id<MTLBuffer> buf_in = pool_alloc(n * 8);
        id<MTLBuffer> buf_partial = pool_alloc(groups * 8);
        if (!buf_in || !buf_partial) return -1;

        // Direct memcpy — raw f64 bytes (kernel handles byte-swap inline)
        memcpy([buf_in contents], in->host_ptr, n * 8);

        // Single kernel dispatch — no word-swap passes needed
        uint32_t n32 = static_cast<uint32_t>(n);
        uint32_t op32 = static_cast<uint32_t>(op);
        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_reduce_sf64_pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_partial offset:0 atIndex:1];
        [encoder setBytes:&n32 length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&op32 length:sizeof(uint32_t) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Pass 2: reduce partial results on CPU (small number of groups)
        // Kernel writes native f64 — CPU reads directly as double
        double* partials = new double[groups];
        memcpy(partials, [buf_partial contents], groups * 8);

        double result;
        switch (op) {
            case 0: case 4: result = 0.0; break;
            case 1: result = 1.0; break;
            case 2: result = INFINITY; break;
            case 3: result = -INFINITY; break;
            default: result = 0.0; break;
        }
        for (uint64_t i = 0; i < groups; i++) {
            switch (op) {
                case 0: case 4: result += partials[i]; break;
                case 1: result *= partials[i]; break;
                case 2: result = (partials[i] < result) ? partials[i] : result; break;
                case 3: result = (partials[i] > result) ? partials[i] : result; break;
                default: break;
            }
        }
        if (op == 4) result /= (double)n; // Mean = sum / n
        delete[] partials;

        static_cast<double*>(out->host_ptr)[0] = result;
        pool_release(buf_in);
        pool_release(buf_partial);
        return 0;
    }
}

static int metal_reduce_axis_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                                  uint64_t rank, const uint64_t* shape,
                                  uint64_t axis, int op) {
    @autoreleasepool {
        if (!g_reduce_axis_sf64_pipeline) return -1;

        // Compute total input elements and output elements
        uint64_t total_in = 1;
        for (uint64_t i = 0; i < rank; i++) total_in *= shape[i];
        uint64_t out_total = total_in / shape[axis];
        if (out_total == 0) return -1;

        // Allocate Metal buffers from pool
        id<MTLBuffer> buf_in = pool_alloc(total_in * 8);
        id<MTLBuffer> buf_out = pool_alloc(out_total * 8);
        id<MTLBuffer> buf_dims = pool_alloc(rank * sizeof(uint32_t));
        if (!buf_in || !buf_out || !buf_dims) return -1;

        // Direct memcpy — raw f64 bytes (kernel handles byte-swap inline)
        memcpy([buf_in contents], in->host_ptr, total_in * 8);

        // Copy dims as uint32 (plain integers)
        uint32_t* dims32 = static_cast<uint32_t*>([buf_dims contents]);
        for (uint64_t i = 0; i < rank; i++) dims32[i] = static_cast<uint32_t>(shape[i]);

        uint32_t rank32 = static_cast<uint32_t>(rank);
        uint32_t axis32 = static_cast<uint32_t>(axis);
        uint32_t op32 = static_cast<uint32_t>(op);
        uint32_t out_size32 = static_cast<uint32_t>(out_total);

        // Single kernel dispatch — no word-swap passes needed
        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_reduce_axis_sf64_pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&rank32 length:sizeof(uint32_t) atIndex:2];
        [encoder setBuffer:buf_dims offset:0 atIndex:3];
        [encoder setBytes:&axis32 length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&op32 length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&out_size32 length:sizeof(uint32_t) atIndex:6];

        NSUInteger groups = (out_total + 255) / 256;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out->host_ptr, [buf_out contents], out_total * 8);
        pool_release(buf_in);
        pool_release(buf_out);
        pool_release(buf_dims);
        return 0;
    }
}

static int metal_transpose_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                                uint64_t rows, uint64_t cols) {
    @autoreleasepool {
        if (!g_transpose_sf64_pipeline) return -1;
        uint64_t n = rows * cols;

        id<MTLBuffer> buf_in = pool_alloc(n * 8);
        id<MTLBuffer> buf_out = pool_alloc(n * 8);
        if (!buf_in || !buf_out) return -1;

        // Direct memcpy — raw f64 bytes (transpose kernel just reindexes, no byte-swap needed)
        memcpy([buf_in contents], in->host_ptr, n * 8);

        uint32_t rows32 = static_cast<uint32_t>(rows);
        uint32_t cols32 = static_cast<uint32_t>(cols);

        // Single kernel dispatch — no word-swap passes needed
        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_transpose_sf64_pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&rows32 length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&cols32 length:sizeof(uint32_t) atIndex:3];

        NSUInteger groups_x = (cols + 15) / 16;
        NSUInteger groups_y = (rows + 15) / 16;
        [encoder dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out->host_ptr, [buf_out contents], n * 8);
        pool_release(buf_in);
        pool_release(buf_out);
        return 0;
    }
}

static int metal_softmax_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                              uint64_t num_slices, uint64_t slice_len) {
    @autoreleasepool {
        if (!g_softmax_sf64_pipeline) return -1;
        uint64_t total = num_slices * slice_len;

        id<MTLBuffer> buf_in = pool_alloc(total * 8);
        id<MTLBuffer> buf_out = pool_alloc(total * 8);
        if (!buf_in || !buf_out) return -1;

        // Direct memcpy — raw f64 bytes (kernel handles byte-swap inline)
        memcpy([buf_in contents], in->host_ptr, total * 8);

        uint32_t slice_len32 = static_cast<uint32_t>(slice_len);
        uint32_t num_slices32 = static_cast<uint32_t>(num_slices);

        // Single kernel dispatch — no word-swap passes needed
        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_softmax_sf64_pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&slice_len32 length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&num_slices32 length:sizeof(uint32_t) atIndex:3];

        NSUInteger groups = (num_slices + 255) / 256;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out->host_ptr, [buf_out contents], total * 8);
        pool_release(buf_in);
        pool_release(buf_out);
        return 0;
    }
}

static int metal_normalize_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out,
                                uint64_t num_slices, uint64_t slice_len,
                                double gamma, double beta, double epsilon) {
    @autoreleasepool {
        if (!g_normalize_sf64_pipeline) return -1;
        uint64_t total = num_slices * slice_len;

        id<MTLBuffer> buf_in = pool_alloc(total * 8);
        id<MTLBuffer> buf_out = pool_alloc(total * 8);
        if (!buf_in || !buf_out) return -1;

        // Direct memcpy — raw f64 bytes (kernel handles byte-swap inline)
        memcpy([buf_in contents], in->host_ptr, total * 8);

        uint32_t slice_len32 = static_cast<uint32_t>(slice_len);
        uint32_t num_slices32 = static_cast<uint32_t>(num_slices);

        // Scalar params: CPU word-swap (3 values, negligible overhead)
        // Kernel expects these as sf64 (word-swapped) since they're used directly in sf64 arithmetic
        uint32_t gamma_sf[2], beta_sf[2], epsilon_sf[2];
        convert_f64_to_sf64(&gamma, gamma_sf, 1);
        convert_f64_to_sf64(&beta, beta_sf, 1);
        convert_f64_to_sf64(&epsilon, epsilon_sf, 1);

        // Single kernel dispatch — no word-swap passes needed
        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_normalize_sf64_pipeline];
        [encoder setBuffer:buf_in offset:0 atIndex:0];
        [encoder setBuffer:buf_out offset:0 atIndex:1];
        [encoder setBytes:&slice_len32 length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&num_slices32 length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:gamma_sf length:8 atIndex:4];
        [encoder setBytes:beta_sf length:8 atIndex:5];
        [encoder setBytes:epsilon_sf length:8 atIndex:6];

        NSUInteger groups = (num_slices + 255) / 256;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out->host_ptr, [buf_out contents], total * 8);
        pool_release(buf_in);
        pool_release(buf_out);
        return 0;
    }
}

#endif // ESHKOL_GPU_METAL_AVAILABLE

// ============================================================================
// CUDA Backend State
// ============================================================================

#if ESHKOL_GPU_CUDA_AVAILABLE

// Forward declarations for CUDA kernel launchers (in gpu_cuda_kernels.cu)
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
            out->device_ptr = nullptr;  // Need separate device alloc for pinned
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

    // cuBLAS uses column-major, so we compute C = B^T * A^T = (A * B)^T
    // But since we want row-major C, we do: C^T = B * A (in cuBLAS terms)
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
    // Register host memory with CUDA
    cudaError_t err = cudaHostRegister(host_ptr, size_bytes, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        // Fallback: allocate and copy
        int result = cuda_alloc(size_bytes, ESHKOL_MEM_UNIFIED, out);
        if (result != 0) return result;
        memcpy(out->host_ptr, host_ptr, size_bytes);
        return 0;
    }

    out->host_ptr = host_ptr;
    out->device_ptr = host_ptr;  // Managed access
    out->size_bytes = size_bytes;
    out->mem_type = ESHKOL_MEM_HOST_PINNED;
    out->backend = ESHKOL_GPU_CUDA;
    out->flags = 1;  // Wrapped
    out->backend_data = nullptr;

    return 0;
}

#endif // ESHKOL_GPU_CUDA_AVAILABLE

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

    // Try Metal first (macOS)
#if ESHKOL_GPU_METAL_AVAILABLE
    if (metal_init() == 0) {
        g_active_backend = ESHKOL_GPU_METAL;
        g_gpu_initialized.store(true, std::memory_order_release);
        return 1;
    }
#endif

    // Try CUDA
#if ESHKOL_GPU_CUDA_AVAILABLE
    if (cuda_init() == 0) {
        g_active_backend = ESHKOL_GPU_CUDA;
        g_gpu_initialized.store(true, std::memory_order_release);
        return 1;
    }
#endif

    // No GPU available
    g_active_backend = ESHKOL_GPU_NONE;
    g_gpu_initialized.store(true, std::memory_order_release);
    return 0;
}

void eshkol_gpu_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_gpu_init_mutex);

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        metal_shutdown();
    }
#endif

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
        case ESHKOL_GPU_METAL: return "Apple Metal";
        case ESHKOL_GPU_CUDA: return "NVIDIA CUDA";
        case ESHKOL_GPU_VULKAN: return "Vulkan";
    }
    return "Unknown";
}

int eshkol_gpu_backend_available(EshkolGPUBackend backend) {
    return (g_active_backend == backend) ? 1 : 0;
}

int eshkol_gpu_supports_f64(void) {
    // Metal does NOT support f64 (double precision) in compute shaders
    // CUDA supports f64 on all modern GPUs
    switch (g_active_backend) {
        case ESHKOL_GPU_CUDA:
            return 1;  // CUDA supports f64
        case ESHKOL_GPU_METAL:
            return 0;  // Metal does NOT support f64
        default:
            return 0;
    }
}

int eshkol_gpu_alloc(size_t size_bytes, EshkolMemoryType mem_type, EshkolGPUBuffer* out_buffer) {
    if (!out_buffer || size_bytes == 0) return -1;

    memset(out_buffer, 0, sizeof(EshkolGPUBuffer));

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_alloc(size_bytes, mem_type, out_buffer);
    }
#endif

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

#if ESHKOL_GPU_METAL_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_METAL) {
        metal_free(buffer);
        memset(buffer, 0, sizeof(EshkolGPUBuffer));
        return;
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_CUDA) {
        cuda_free(buffer);
        memset(buffer, 0, sizeof(EshkolGPUBuffer));
        return;
    }
#endif

    // CPU fallback
    if (buffer->host_ptr && !(buffer->flags & 1)) {
        free(buffer->host_ptr);
    }
    memset(buffer, 0, sizeof(EshkolGPUBuffer));
}

int eshkol_gpu_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out_buffer) {
    if (!host_ptr || !out_buffer || size_bytes == 0) return -1;

    memset(out_buffer, 0, sizeof(EshkolGPUBuffer));

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_wrap_host(host_ptr, size_bytes, out_buffer);
    }
#endif

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
    out_buffer->flags = 1;  // Wrapped
    return 0;
}

int eshkol_gpu_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    if (!buffer) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_METAL) {
        return metal_sync(buffer, direction);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_CUDA) {
        return cuda_sync(buffer, direction);
    }
#endif

    return 0;  // CPU: no sync needed
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

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_matmul_f64(A, B, C, M, K, N);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        return cuda_matmul_f64(A, B, C, M, K, N);
    }
#endif

    // CPU fallback
    extern void eshkol_matmul_f64(const double*, const double*, double*, uint64_t, uint64_t, uint64_t);
    eshkol_matmul_f64((const double*)A->host_ptr, (const double*)B->host_ptr,
                      (double*)C->host_ptr, M, K, N);
    return 0;
}

int eshkol_gpu_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_matmul_f32(A, B, C, M, K, N);
    }
#endif

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
    // Lazy GPU init — ensures g_active_backend is set before threshold check
    if (!g_gpu_initialized.load(std::memory_order_acquire)) {
        eshkol_gpu_init();
    }

    size_t num_elements = M * N;

    // GPU path if available and large enough
    if (eshkol_gpu_should_use(num_elements)) {
        EshkolGPUBuffer buf_a, buf_b, buf_c;
        if (eshkol_gpu_wrap_host((void*)A, M * K * sizeof(double), &buf_a) == 0 &&
            eshkol_gpu_wrap_host((void*)B, K * N * sizeof(double), &buf_b) == 0 &&
            eshkol_gpu_wrap_host((void*)C, M * N * sizeof(double), &buf_c) == 0) {

            if (eshkol_gpu_matmul_f64(&buf_a, &buf_b, &buf_c, M, K, N) == 0) {
                // Copy back if fallback allocation was used (host_ptr diverged from original)
                // This happens when newBufferWithBytesNoCopy fails (non-page-aligned pointer)
                if (buf_c.host_ptr != (void*)C) {
                    memcpy((void*)C, buf_c.host_ptr, M * N * sizeof(double));
                }
                eshkol_gpu_free(&buf_a);
                eshkol_gpu_free(&buf_b);
                eshkol_gpu_free(&buf_c);
                return;
            }
        }
        // GPU failed, fall through to CPU
        fprintf(stderr, "[GPU] matmul dispatch failed, falling back to CPU BLAS\n");
    }

    // CPU fallback (BLAS/SIMD/scalar)
    extern void eshkol_matmul_f64(const double*, const double*, double*, uint64_t, uint64_t, uint64_t);
    eshkol_matmul_f64(A, B, C, M, K, N);
}

int eshkol_gpu_elementwise_f64(EshkolGPUBuffer* a, EshkolGPUBuffer* b,
                                EshkolGPUBuffer* out, uint64_t n,
                                EshkolElementwiseOp op) {
    if (!a || !out || n == 0) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_elementwise_f64(a, b, out, n, (int)op);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && a->device_ptr && out->device_ptr) {
        const double* dp_b = (b && b->device_ptr) ? static_cast<const double*>(b->device_ptr) : nullptr;
        int result = cuda_launch_elementwise_f64(
            static_cast<const double*>(a->device_ptr), dp_b,
            static_cast<double*>(out->device_ptr), static_cast<int64_t>(n),
            static_cast<int>(op), static_cast<void*>(g_cuda_stream));
        if (result == 0) return 0;
    }
#endif

    // CPU fallback: scalar loop
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

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_reduce_f64(in, out, n, (int)op);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && in->device_ptr && out->device_ptr) {
        int result = cuda_launch_reduce_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            static_cast<int64_t>(n), static_cast<int>(op),
            static_cast<void*>(g_cuda_stream));
        if (result == 0) {
            if (op == ESHKOL_REDUCE_MEAN) {
                cudaStreamSynchronize(g_cuda_stream);
                double sum_val;
                cudaMemcpy(&sum_val, out->device_ptr, sizeof(double), cudaMemcpyDeviceToHost);
                sum_val /= (double)n;
                cudaMemcpy(out->device_ptr, &sum_val, sizeof(double), cudaMemcpyHostToDevice);
            }
            return 0;
        }
    }
#endif

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

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_reduce_axis_f64(in, out, rank, shape, axis, (int)op);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        uint64_t total_in = 1;
        for (uint64_t i = 0; i < rank; i++) total_in *= shape[i];
        uint64_t out_total = total_in / shape[axis];
        int result = cuda_launch_reduce_axis_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            rank, shape, axis, static_cast<int>(op), out_total,
            static_cast<void*>(g_cuda_stream));
        if (result == 0) {
            if (op == ESHKOL_REDUCE_MEAN) {
                // MEAN needs post-divide
                double* outp = static_cast<double*>(out->host_ptr);
                cudaStreamSynchronize(g_cuda_stream);
                for (uint64_t i = 0; i < out_total; i++) outp[i] /= (double)shape[axis];
            }
            return 0;
        }
    }
#endif

    // CPU fallback: N-D axis reduction with stride computation
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

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_transpose_f64(in, out, rows, cols);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && in->device_ptr && out->device_ptr) {
        int result = cuda_launch_transpose_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            rows, cols, static_cast<void*>(g_cuda_stream));
        if (result == 0) return 0;
    }
#endif

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

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_softmax_f64(in, out, num_slices, slice_len);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && in->device_ptr && out->device_ptr) {
        int result = cuda_launch_softmax_f64(
            static_cast<const double*>(in->device_ptr),
            static_cast<double*>(out->device_ptr),
            num_slices, slice_len, static_cast<void*>(g_cuda_stream));
        if (result == 0) return 0;
    }
#endif

    // CPU fallback: numerically stable softmax
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

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_normalize_f64(in, out, num_slices, slice_len,
                                   gamma, beta, epsilon);
    }
#endif

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

    // CPU fallback: layer normalization
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
