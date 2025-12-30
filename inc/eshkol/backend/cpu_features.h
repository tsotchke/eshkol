/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * CPU Features Detection for SIMD Optimization
 *
 * This module provides compile-time and runtime detection of CPU SIMD
 * capabilities (SSE2, AVX, AVX2, AVX-512, NEON) to enable optimal
 * vector width selection for tensor operations.
 */
#ifndef ESHKOL_BACKEND_CPU_FEATURES_H
#define ESHKOL_BACKEND_CPU_FEATURES_H

#include <cstdint>
#include <string>

namespace eshkol {

/**
 * SIMD instruction set levels, ordered by capability.
 */
enum class SIMDLevel {
    SCALAR = 0,     // No SIMD, scalar operations only
    SSE2 = 1,       // 128-bit: 2 doubles (x86)
    NEON = 2,       // 128-bit: 2 doubles (ARM64)
    AVX = 3,        // 256-bit: 4 doubles (x86)
    AVX2 = 4,       // 256-bit: 4 doubles + more ops (x86)
    AVX512 = 5      // 512-bit: 8 doubles (x86)
};

/**
 * CPU feature flags for detailed capability checking.
 */
struct CPUFeatures {
    // x86 features
    bool has_sse2 = false;
    bool has_sse3 = false;
    bool has_ssse3 = false;
    bool has_sse41 = false;
    bool has_sse42 = false;
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_fma = false;
    bool has_avx512f = false;   // AVX-512 Foundation
    bool has_avx512dq = false;  // AVX-512 Doubleword/Quadword
    bool has_avx512vl = false;  // AVX-512 Vector Length

    // ARM features
    bool has_neon = false;
    bool has_neon_fp16 = false;     // FP16 arithmetic (FEAT_FP16)
    bool has_neon_dotprod = false;  // SDOT/UDOT dot product (FEAT_DotProd)
    bool has_neon_fhm = false;      // FP16 fused multiply-accumulate (FEAT_FHM)
    bool has_neon_bf16 = false;     // BFloat16 support (FEAT_BF16)
    bool has_neon_i8mm = false;     // Int8 matrix multiply (FEAT_I8MM)

    // Architecture detection
    bool is_x86 = false;
    bool is_arm64 = false;

    /**
     * Get the highest available SIMD level.
     */
    SIMDLevel getBestSIMDLevel() const;

    /**
     * Get the optimal vector width in number of doubles.
     * Returns 1 (scalar), 2 (SSE2/NEON), 4 (AVX), or 8 (AVX-512).
     */
    unsigned getOptimalVectorWidth() const;

    /**
     * Get LLVM target features string for this CPU.
     * Example: "+avx2,+fma" or "+neon"
     */
    std::string getLLVMTargetFeatures() const;

    /**
     * Get a human-readable description of detected features.
     */
    std::string getDescription() const;
};

/**
 * CPUFeatureDetector provides runtime detection of CPU capabilities.
 *
 * Usage:
 *   CPUFeatureDetector detector;
 *   auto features = detector.detect();
 *   unsigned width = features.getOptimalVectorWidth();
 */
class CPUFeatureDetector {
public:
    /**
     * Detect CPU features for the current host.
     * This performs runtime detection using CPUID (x86) or
     * reading system registers (ARM).
     */
    static CPUFeatures detect();

    /**
     * Get features for a specific target triple.
     * Used when cross-compiling for a different architecture.
     */
    static CPUFeatures forTarget(const std::string& target_triple);

    /**
     * Force a specific SIMD level (for testing/debugging).
     */
    static CPUFeatures forSIMDLevel(SIMDLevel level);

private:
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    static void detectX86Features(CPUFeatures& features);
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    static void detectARMFeatures(CPUFeatures& features);
#endif
};

/**
 * Global singleton for cached CPU features.
 * Call once at startup, then use throughout compilation.
 */
class CPUCapabilities {
public:
    static CPUCapabilities& instance();

    const CPUFeatures& getFeatures() const { return features_; }
    SIMDLevel getSIMDLevel() const { return features_.getBestSIMDLevel(); }
    unsigned getVectorWidth() const { return features_.getOptimalVectorWidth(); }

    // Convenience accessors - x86
    bool hasAVX() const { return features_.has_avx; }
    bool hasAVX2() const { return features_.has_avx2; }
    bool hasAVX512() const { return features_.has_avx512f; }
    bool hasFMA() const { return features_.has_fma; }

    // Convenience accessors - ARM
    bool hasNEON() const { return features_.has_neon; }
    bool hasNEON_FP16() const { return features_.has_neon_fp16; }
    bool hasNEON_DotProd() const { return features_.has_neon_dotprod; }
    bool hasNEON_FHM() const { return features_.has_neon_fhm; }
    bool hasNEON_BF16() const { return features_.has_neon_bf16; }
    bool hasNEON_I8MM() const { return features_.has_neon_i8mm; }

    // Architecture detection
    bool isARM64() const { return features_.is_arm64; }
    bool isX86() const { return features_.is_x86; }

private:
    CPUCapabilities();
    CPUFeatures features_;
};

} // namespace eshkol

#endif // ESHKOL_BACKEND_CPU_FEATURES_H
