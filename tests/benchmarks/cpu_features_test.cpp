/*
 * CPU Feature Detection Test
 * Verifies runtime detection of SIMD capabilities
 */
#include <iostream>
#include <cstring>
#include "eshkol/backend/cpu_features.h"

void printFeatures(const eshkol::CPUFeatures& features, const char* label) {
    std::cout << "\n" << label << "\n";
    std::cout << std::string(strlen(label), '-') << "\n";
    std::cout << "Description: " << features.getDescription() << "\n";
    std::cout << "LLVM Features: " << features.getLLVMTargetFeatures() << "\n";
    std::cout << "Vector Width: " << features.getOptimalVectorWidth() << " doubles\n";

    if (features.is_arm64) {
        std::cout << "ARM Features:\n";
        std::cout << "  NEON:    " << (features.has_neon ? "yes" : "no") << "\n";
        std::cout << "  FP16:    " << (features.has_neon_fp16 ? "yes" : "no") << "\n";
        std::cout << "  DotProd: " << (features.has_neon_dotprod ? "yes" : "no") << "\n";
        std::cout << "  FHM:     " << (features.has_neon_fhm ? "yes" : "no") << "\n";
        std::cout << "  BF16:    " << (features.has_neon_bf16 ? "yes" : "no") << "\n";
        std::cout << "  I8MM:    " << (features.has_neon_i8mm ? "yes" : "no") << "\n";
    }

    if (features.is_x86) {
        std::cout << "x86 Features:\n";
        std::cout << "  SSE2:     " << (features.has_sse2 ? "yes" : "no") << "\n";
        std::cout << "  SSE3:     " << (features.has_sse3 ? "yes" : "no") << "\n";
        std::cout << "  SSSE3:    " << (features.has_ssse3 ? "yes" : "no") << "\n";
        std::cout << "  SSE4.1:   " << (features.has_sse41 ? "yes" : "no") << "\n";
        std::cout << "  SSE4.2:   " << (features.has_sse42 ? "yes" : "no") << "\n";
        std::cout << "  AVX:      " << (features.has_avx ? "yes" : "no") << "\n";
        std::cout << "  AVX2:     " << (features.has_avx2 ? "yes" : "no") << "\n";
        std::cout << "  FMA:      " << (features.has_fma ? "yes" : "no") << "\n";
        std::cout << "  AVX-512F: " << (features.has_avx512f ? "yes" : "no") << "\n";
        std::cout << "  AVX-512DQ:" << (features.has_avx512dq ? "yes" : "no") << "\n";
        std::cout << "  AVX-512VL:" << (features.has_avx512vl ? "yes" : "no") << "\n";
    }
}

bool verifyFeatures(const eshkol::CPUFeatures& f, eshkol::SIMDLevel level) {
    unsigned width = f.getOptimalVectorWidth();
    switch (level) {
        case eshkol::SIMDLevel::SCALAR:
            return width == 1;
        case eshkol::SIMDLevel::SSE2:
            return width == 2 && f.is_x86 && f.has_sse2;
        case eshkol::SIMDLevel::NEON:
            return width == 2 && f.is_arm64 && f.has_neon;
        case eshkol::SIMDLevel::AVX:
            return width == 4 && f.is_x86 && f.has_avx;
        case eshkol::SIMDLevel::AVX2:
            return width == 4 && f.is_x86 && f.has_avx2 && f.has_fma;
        case eshkol::SIMDLevel::AVX512:
            return width == 8 && f.is_x86 && f.has_avx512f;
    }
    return false;
}

int main() {
    std::cout << "CPU Feature Detection Test\n";
    std::cout << "==========================\n";

    // Test 1: Runtime detection
    auto& caps = eshkol::CPUCapabilities::instance();
    auto& features = caps.getFeatures();
    printFeatures(features, "Runtime Detected Features");

    // Verify at least one architecture is detected
    if (!features.is_x86 && !features.is_arm64) {
        std::cerr << "\nERROR: No architecture detected!\n";
        return 1;
    }

    // Verify vector width is valid
    unsigned width = features.getOptimalVectorWidth();
    if (width != 1 && width != 2 && width != 4 && width != 8) {
        std::cerr << "\nERROR: Invalid vector width: " << width << "\n";
        return 1;
    }

    // Test 2: Verify forSIMDLevel() creates correct feature sets
    std::cout << "\n\nTesting forSIMDLevel() synthetic feature sets:\n";
    std::cout << "===============================================\n";

    struct TestCase {
        eshkol::SIMDLevel level;
        const char* name;
    };

    TestCase tests[] = {
        {eshkol::SIMDLevel::SCALAR, "SCALAR"},
        {eshkol::SIMDLevel::SSE2, "SSE2"},
        {eshkol::SIMDLevel::NEON, "NEON"},
        {eshkol::SIMDLevel::AVX, "AVX"},
        {eshkol::SIMDLevel::AVX2, "AVX2"},
        {eshkol::SIMDLevel::AVX512, "AVX-512"},
    };

    bool all_passed = true;
    for (const auto& test : tests) {
        auto f = eshkol::CPUFeatureDetector::forSIMDLevel(test.level);
        bool ok = verifyFeatures(f, test.level);
        std::cout << "  " << test.name << ": " << f.getDescription();
        std::cout << " [" << (ok ? "PASS" : "FAIL") << "]\n";
        if (!ok) all_passed = false;
    }

    // Test 3: Verify forTarget() target triple parsing
    std::cout << "\nTesting forTarget() target triple parsing:\n";
    std::cout << "==========================================\n";

    auto x86_features = eshkol::CPUFeatureDetector::forTarget("x86_64-unknown-linux-gnu");
    std::cout << "  x86_64-unknown-linux-gnu: " << x86_features.getDescription();
    std::cout << " [" << (x86_features.is_x86 ? "PASS" : "FAIL") << "]\n";
    if (!x86_features.is_x86) all_passed = false;

    auto arm_features = eshkol::CPUFeatureDetector::forTarget("aarch64-apple-darwin");
    std::cout << "  aarch64-apple-darwin: " << arm_features.getDescription();
    std::cout << " [" << (arm_features.is_arm64 ? "PASS" : "FAIL") << "]\n";
    if (!arm_features.is_arm64) all_passed = false;

    if (all_passed) {
        std::cout << "\nAll checks passed.\n";
        return 0;
    } else {
        std::cerr << "\nSome checks FAILED!\n";
        return 1;
    }
}
