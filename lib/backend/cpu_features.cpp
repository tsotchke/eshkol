/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * CPU Features Detection Implementation
 */

#include "eshkol/backend/cpu_features.h"
#include <sstream>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#endif

namespace eshkol {

// ============================================================================
// CPUFeatures Implementation
// ============================================================================

SIMDLevel CPUFeatures::getBestSIMDLevel() const {
    if (has_avx512f && has_avx512dq && has_avx512vl) {
        return SIMDLevel::AVX512;
    }
    if (has_avx2) {
        return SIMDLevel::AVX2;
    }
    if (has_avx) {
        return SIMDLevel::AVX;
    }
    if (has_neon) {
        return SIMDLevel::NEON;
    }
    if (has_sse2) {
        return SIMDLevel::SSE2;
    }
    return SIMDLevel::SCALAR;
}

unsigned CPUFeatures::getOptimalVectorWidth() const {
    switch (getBestSIMDLevel()) {
        case SIMDLevel::AVX512:
            return 8;  // 512-bit = 8 doubles
        case SIMDLevel::AVX2:
        case SIMDLevel::AVX:
            return 4;  // 256-bit = 4 doubles
        case SIMDLevel::NEON:
        case SIMDLevel::SSE2:
            return 2;  // 128-bit = 2 doubles
        case SIMDLevel::SCALAR:
        default:
            return 1;  // No vectorization
    }
}

std::string CPUFeatures::getLLVMTargetFeatures() const {
    std::ostringstream features;

    if (is_x86) {
        // Build x86 feature string
        if (has_sse2) features << "+sse2,";
        if (has_sse3) features << "+sse3,";
        if (has_ssse3) features << "+ssse3,";
        if (has_sse41) features << "+sse4.1,";
        if (has_sse42) features << "+sse4.2,";
        if (has_avx) features << "+avx,";
        if (has_avx2) features << "+avx2,";
        if (has_fma) features << "+fma,";
        if (has_avx512f) features << "+avx512f,";
        if (has_avx512dq) features << "+avx512dq,";
        if (has_avx512vl) features << "+avx512vl,";
    } else if (is_arm64) {
        // Build ARM64 feature string
        if (has_neon) features << "+neon,";
        if (has_neon_fp16) features << "+fullfp16,";
        if (has_neon_dotprod) features << "+dotprod,";
        if (has_neon_fhm) features << "+fp16fml,";
        if (has_neon_bf16) features << "+bf16,";
        if (has_neon_i8mm) features << "+i8mm,";
    }

    std::string result = features.str();
    // Remove trailing comma if present
    if (!result.empty() && result.back() == ',') {
        result.pop_back();
    }
    return result;
}

std::string CPUFeatures::getDescription() const {
    std::ostringstream desc;

    if (is_x86) {
        desc << "x86";
        if (has_avx512f) desc << " AVX-512";
        else if (has_avx2) desc << " AVX2";
        else if (has_avx) desc << " AVX";
        else if (has_sse2) desc << " SSE2";
        if (has_fma) desc << "+FMA";
    } else if (is_arm64) {
        desc << "ARM64 NEON";
        if (has_neon_fp16) desc << "+FP16";
        if (has_neon_dotprod) desc << "+DotProd";
        if (has_neon_fhm) desc << "+FHM";
        if (has_neon_bf16) desc << "+BF16";
        if (has_neon_i8mm) desc << "+I8MM";
    } else {
        desc << "Unknown (scalar)";
    }

    desc << " (vector width: " << getOptimalVectorWidth() << " doubles)";
    return desc.str();
}

// ============================================================================
// CPUFeatureDetector Implementation
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

void CPUFeatureDetector::detectX86Features(CPUFeatures& features) {
    features.is_x86 = true;

    unsigned int eax, ebx, ecx, edx;

    // Get highest supported CPUID function
#ifdef _MSC_VER
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    unsigned int max_id = cpuInfo[0];
#else
    unsigned int max_id;
    __get_cpuid(0, &max_id, &ebx, &ecx, &edx);
#endif

    if (max_id >= 1) {
        // CPUID function 1: processor info and feature bits
#ifdef _MSC_VER
        __cpuid(cpuInfo, 1);
        ecx = cpuInfo[2];
        edx = cpuInfo[3];
#else
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
#endif

        // EDX flags
        features.has_sse2 = (edx >> 26) & 1;

        // ECX flags
        features.has_sse3 = (ecx >> 0) & 1;
        features.has_ssse3 = (ecx >> 9) & 1;
        features.has_sse41 = (ecx >> 19) & 1;
        features.has_sse42 = (ecx >> 20) & 1;
        features.has_avx = (ecx >> 28) & 1;
        features.has_fma = (ecx >> 12) & 1;

        // Check OS support for AVX (XSAVE enabled)
        if (features.has_avx) {
            bool os_xsave = (ecx >> 27) & 1;
            if (os_xsave) {
                // Check XCR0 for AVX state support
                unsigned int xcr0_lo, xcr0_hi;
#ifdef _MSC_VER
                unsigned long long xcr0 = _xgetbv(0);
                xcr0_lo = (unsigned int)xcr0;
#else
                __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
#endif
                // Check if XMM and YMM state are enabled
                bool xmm_enabled = (xcr0_lo >> 1) & 1;
                bool ymm_enabled = (xcr0_lo >> 2) & 1;
                if (!xmm_enabled || !ymm_enabled) {
                    features.has_avx = false;
                    features.has_fma = false;
                }
            } else {
                features.has_avx = false;
                features.has_fma = false;
            }
        }
    }

    if (max_id >= 7 && features.has_avx) {
        // CPUID function 7: extended feature flags
#ifdef _MSC_VER
        __cpuidex(cpuInfo, 7, 0);
        ebx = cpuInfo[1];
        ecx = cpuInfo[2];
#else
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
#endif

        // EBX flags
        features.has_avx2 = (ebx >> 5) & 1;
        features.has_avx512f = (ebx >> 16) & 1;
        features.has_avx512dq = (ebx >> 17) & 1;
        features.has_avx512vl = (ebx >> 31) & 1;

        // Check OS support for AVX-512 (opmask and ZMM state)
        if (features.has_avx512f) {
            unsigned int xcr0_lo, xcr0_hi;
#ifdef _MSC_VER
            unsigned long long xcr0 = _xgetbv(0);
            xcr0_lo = (unsigned int)xcr0;
#else
            __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
#endif
            // Check if opmask (bit 5), ZMM_Hi256 (bit 6), Hi16_ZMM (bit 7) are enabled
            bool avx512_enabled = ((xcr0_lo >> 5) & 0x7) == 0x7;
            if (!avx512_enabled) {
                features.has_avx512f = false;
                features.has_avx512dq = false;
                features.has_avx512vl = false;
            }
        }
    }
}

#endif // x86

#if defined(__aarch64__) || defined(_M_ARM64)

void CPUFeatureDetector::detectARMFeatures(CPUFeatures& features) {
    features.is_arm64 = true;

    // NEON is mandatory on ARM64, always available
    features.has_neon = true;

#if defined(__APPLE__)
    // macOS: Use sysctlbyname to query CPU features
    auto hasFeature = [](const char* name) -> bool {
        int result = 0;
        size_t size = sizeof(result);
        if (sysctlbyname(name, &result, &size, nullptr, 0) == 0) {
            return result != 0;
        }
        return false;
    };

    // Query ARM feature flags via sysctl
    // hw.optional.arm.FEAT_* keys are available on Apple Silicon
    features.has_neon_fp16 = hasFeature("hw.optional.arm.FEAT_FP16") ||
                             hasFeature("hw.optional.neon_fp16");
    features.has_neon_dotprod = hasFeature("hw.optional.arm.FEAT_DotProd");
    features.has_neon_fhm = hasFeature("hw.optional.arm.FEAT_FHM");
    features.has_neon_bf16 = hasFeature("hw.optional.arm.FEAT_BF16");
    features.has_neon_i8mm = hasFeature("hw.optional.arm.FEAT_I8MM");

#elif defined(__linux__)
    // Linux: Use getauxval to read hardware capabilities
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    // HWCAP flags for ARM64 features (from asm/hwcap.h)
    #ifdef HWCAP_FPHP
    features.has_neon_fp16 = (hwcap & HWCAP_FPHP) && (hwcap & HWCAP_ASIMDHP);
    #endif

    #ifdef HWCAP_ASIMDDP
    features.has_neon_dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
    #endif

    #ifdef HWCAP_ASIMDFHM
    features.has_neon_fhm = (hwcap & HWCAP_ASIMDFHM) != 0;
    #endif

    // HWCAP2 flags (BF16, I8MM are in hwcap2)
    #ifdef HWCAP2_BF16
    features.has_neon_bf16 = (hwcap2 & HWCAP2_BF16) != 0;
    #endif

    #ifdef HWCAP2_I8MM
    features.has_neon_i8mm = (hwcap2 & HWCAP2_I8MM) != 0;
    #endif

#else
    // Unknown platform: conservative default
    features.has_neon_fp16 = false;
#endif
}

#endif // ARM64

CPUFeatures CPUFeatureDetector::detect() {
    CPUFeatures features;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    detectX86Features(features);
#elif defined(__aarch64__) || defined(_M_ARM64)
    detectARMFeatures(features);
#else
    // Unknown architecture, use scalar
    // No SIMD features available
#endif

    return features;
}

CPUFeatures CPUFeatureDetector::forTarget(const std::string& target_triple) {
    CPUFeatures features;

    // Parse target triple to determine architecture
    if (target_triple.find("x86_64") != std::string::npos ||
        target_triple.find("i386") != std::string::npos ||
        target_triple.find("i686") != std::string::npos) {
        features.is_x86 = true;
        // Default to AVX2 for modern x86-64
        features.has_sse2 = true;
        features.has_sse3 = true;
        features.has_ssse3 = true;
        features.has_sse41 = true;
        features.has_sse42 = true;
        features.has_avx = true;
        features.has_avx2 = true;
        features.has_fma = true;
    } else if (target_triple.find("aarch64") != std::string::npos ||
               target_triple.find("arm64") != std::string::npos) {
        features.is_arm64 = true;
        features.has_neon = true;
    }

    return features;
}

CPUFeatures CPUFeatureDetector::forSIMDLevel(SIMDLevel level) {
    CPUFeatures features;

    switch (level) {
        case SIMDLevel::AVX512:
            features.is_x86 = true;
            features.has_sse2 = true;
            features.has_sse3 = true;
            features.has_ssse3 = true;
            features.has_sse41 = true;
            features.has_sse42 = true;
            features.has_avx = true;
            features.has_avx2 = true;
            features.has_fma = true;
            features.has_avx512f = true;
            features.has_avx512dq = true;
            features.has_avx512vl = true;
            break;

        case SIMDLevel::AVX2:
            features.is_x86 = true;
            features.has_sse2 = true;
            features.has_sse3 = true;
            features.has_ssse3 = true;
            features.has_sse41 = true;
            features.has_sse42 = true;
            features.has_avx = true;
            features.has_avx2 = true;
            features.has_fma = true;
            break;

        case SIMDLevel::AVX:
            features.is_x86 = true;
            features.has_sse2 = true;
            features.has_sse3 = true;
            features.has_ssse3 = true;
            features.has_sse41 = true;
            features.has_sse42 = true;
            features.has_avx = true;
            break;

        case SIMDLevel::NEON:
            features.is_arm64 = true;
            features.has_neon = true;
            // Assume modern ARM64 with full feature set (Apple Silicon, etc.)
            features.has_neon_fp16 = true;
            features.has_neon_dotprod = true;
            features.has_neon_fhm = true;
            break;

        case SIMDLevel::SSE2:
            features.is_x86 = true;
            features.has_sse2 = true;
            break;

        case SIMDLevel::SCALAR:
        default:
            // No features
            break;
    }

    return features;
}

// ============================================================================
// CPUCapabilities Singleton
// ============================================================================

CPUCapabilities& CPUCapabilities::instance() {
    static CPUCapabilities instance;
    return instance;
}

CPUCapabilities::CPUCapabilities() : features_(CPUFeatureDetector::detect()) {
    // Features are detected once at construction
}

} // namespace eshkol
