/*
 * XLA Codegen Integration Tests
 *
 * Tests the XLA/StableHLO backend functionality.
 * Only builds when ESHKOL_USE_XLA=ON.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>

// XLA headers
#include "eshkol/backend/xla/xla_codegen.h"
#include "eshkol/backend/xla/xla_runtime.h"

// Test utilities
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(a, b, eps, msg) \
    TEST_ASSERT(std::abs((a) - (b)) < (eps), msg)

// Test counters
static int tests_passed = 0;
static int tests_failed = 0;

// ===== Runtime Function Tests =====

// External runtime function
extern "C" void* eshkol_xla_matmul(
    void* arena,
    const double* a_data,
    const double* b_data,
    const int64_t* a_shape,
    const int64_t* b_shape,
    int64_t a_rank,
    int64_t b_rank);

// Arena allocator — use the real Eshkol arena
typedef struct arena arena_t;
extern "C" arena_t* arena_create(size_t default_block_size);
extern "C" void arena_destroy(arena_t* arena);

// Canonical tensor struct (must match arena_memory.h eshkol_tensor_t)
struct EshkolTensor {
    uint64_t* dimensions;     // idx 0: dimension sizes
    uint64_t  num_dimensions; // idx 1: rank
    int64_t*  elements;       // idx 2: doubles as int64 bit patterns
    uint64_t  total_elements; // idx 3: product of all dimensions
};

// Global test arena
static arena_t* g_test_arena = nullptr;

// ===== Test: XLA Runtime Initialization =====
bool test_xla_runtime_init() {
    std::cout << "Test: XLA Runtime Initialization... ";

    auto& runtime = eshkol::xla::getDefaultRuntime();

    // Initialize for CPU target
    bool init_result = runtime.initialize(eshkol::xla::Target::CPU);
    TEST_ASSERT(init_result, "Runtime should initialize for CPU target");
    TEST_ASSERT(runtime.isInitialized(), "Runtime should be initialized");
    TEST_ASSERT(runtime.getTarget() == eshkol::xla::Target::CPU, "Target should be CPU");

    std::cout << "PASS" << std::endl;
    return true;
}

// ===== Test: XLA Runtime Description =====
bool test_xla_runtime_description() {
    std::cout << "Test: XLA Runtime Description... ";

    auto& runtime = eshkol::xla::getDefaultRuntime();
    std::string desc = runtime.getDescription();

    TEST_ASSERT(!desc.empty(), "Description should not be empty");
    TEST_ASSERT(desc.find("XLA") != std::string::npos, "Description should contain 'XLA'");

    std::cout << "PASS (" << desc << ")" << std::endl;
    return true;
}

// Helper: read double from int64 bit pattern
static double read_element(const EshkolTensor* t, size_t idx) {
    double val;
    memcpy(&val, &t->elements[idx], sizeof(double));
    return val;
}

// ===== Test: XLA Matmul 2x2 =====
bool test_xla_matmul_2x2() {
    std::cout << "Test: XLA Matmul 2x2... ";

    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // C = A * B = [[19, 22], [43, 50]]

    double a_data[] = {1.0, 2.0, 3.0, 4.0};
    double b_data[] = {5.0, 6.0, 7.0, 8.0};
    int64_t a_shape[] = {2, 2};
    int64_t b_shape[] = {2, 2};

    void* result = eshkol_xla_matmul(
        g_test_arena,
        a_data, b_data,
        a_shape, b_shape,
        2, 2);

    TEST_ASSERT(result != nullptr, "Matmul should return non-null result");

    auto* tensor = static_cast<EshkolTensor*>(result);

    TEST_ASSERT(tensor->num_dimensions == 2, "Result should have 2 dimensions");
    TEST_ASSERT(tensor->dimensions[0] == 2, "Result dim 0 should be 2");
    TEST_ASSERT(tensor->dimensions[1] == 2, "Result dim 1 should be 2");

    // Verify values
    double expected[] = {19.0, 22.0, 43.0, 50.0};
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_NEAR(read_element(tensor, i), expected[i], 1e-10,
            "Result value mismatch at index " + std::to_string(i));
    }

    std::cout << "PASS" << std::endl;
    return true;
}

// ===== Test: XLA Matmul 3x2 * 2x4 =====
bool test_xla_matmul_3x2_2x4() {
    std::cout << "Test: XLA Matmul 3x2 * 2x4... ";

    // A = [[1, 2], [3, 4], [5, 6]] (3x2)
    // B = [[1, 2, 3, 4], [5, 6, 7, 8]] (2x4)
    // C = A * B (3x4)

    double a_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double b_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    int64_t a_shape[] = {3, 2};
    int64_t b_shape[] = {2, 4};

    void* result = eshkol_xla_matmul(
        g_test_arena,
        a_data, b_data,
        a_shape, b_shape,
        2, 2);

    TEST_ASSERT(result != nullptr, "Matmul should return non-null result");

    auto* tensor = static_cast<EshkolTensor*>(result);

    TEST_ASSERT(tensor->num_dimensions == 2, "Result should have 2 dimensions");
    TEST_ASSERT(tensor->dimensions[0] == 3, "Result dim 0 should be 3");
    TEST_ASSERT(tensor->dimensions[1] == 4, "Result dim 1 should be 4");

    // Expected: C[i][j] = sum_k A[i][k] * B[k][j]
    double expected[] = {
        11.0, 14.0, 17.0, 20.0,
        23.0, 30.0, 37.0, 44.0,
        35.0, 46.0, 57.0, 68.0
    };

    for (int i = 0; i < 12; i++) {
        TEST_ASSERT_NEAR(read_element(tensor, i), expected[i], 1e-10,
            "Result value mismatch at index " + std::to_string(i));
    }

    std::cout << "PASS" << std::endl;
    return true;
}

// ===== Test: XLA Matmul Dimension Mismatch =====
bool test_xla_matmul_dim_mismatch() {
    std::cout << "Test: XLA Matmul Dimension Mismatch... ";

    double a_data[] = {1.0, 2.0, 3.0, 4.0};
    double b_data[] = {1.0, 2.0, 3.0};
    int64_t a_shape[] = {2, 2};  // 2x2
    int64_t b_shape[] = {3, 1};  // 3x1 - incompatible!

    void* result = eshkol_xla_matmul(
        g_test_arena,
        a_data, b_data,
        a_shape, b_shape,
        2, 2);

    TEST_ASSERT(result == nullptr, "Matmul should return null for incompatible dimensions");

    std::cout << "PASS" << std::endl;
    return true;
}

// ===== Test: XLA Matmul Invalid Rank =====
bool test_xla_matmul_invalid_rank() {
    std::cout << "Test: XLA Matmul Invalid Rank... ";

    double a_data[] = {1.0, 2.0, 3.0};
    double b_data[] = {1.0, 2.0, 3.0};
    int64_t a_shape[] = {3};     // 1D - invalid
    int64_t b_shape[] = {3, 1};  // 2D

    void* result = eshkol_xla_matmul(
        g_test_arena,
        a_data, b_data,
        a_shape, b_shape,
        1, 2);  // a_rank = 1

    TEST_ASSERT(result == nullptr, "Matmul should return null for non-2D tensors");

    std::cout << "PASS" << std::endl;
    return true;
}

// ===== Test: XLA Matmul Large =====
bool test_xla_matmul_large() {
    std::cout << "Test: XLA Matmul 100x100... ";

    const int N = 100;
    std::vector<double> a_data(N * N);
    std::vector<double> b_data(N * N);

    // Initialize with identity-like pattern for easy verification
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a_data[i * N + j] = (i == j) ? 1.0 : 0.0;
            b_data[i * N + j] = static_cast<double>(i * N + j);
        }
    }

    int64_t shape[] = {N, N};

    void* result = eshkol_xla_matmul(
        g_test_arena,
        a_data.data(), b_data.data(),
        shape, shape,
        2, 2);

    TEST_ASSERT(result != nullptr, "Matmul should return non-null result");

    auto* tensor = static_cast<EshkolTensor*>(result);

    TEST_ASSERT(tensor->dimensions[0] == static_cast<uint64_t>(N), "Result dim 0 should be N");
    TEST_ASSERT(tensor->dimensions[1] == static_cast<uint64_t>(N), "Result dim 1 should be N");

    // Identity * B = B
    for (int i = 0; i < N * N; i++) {
        TEST_ASSERT_NEAR(read_element(tensor, i), b_data[i], 1e-10,
            "Identity multiplication should preserve B");
    }

    std::cout << "PASS" << std::endl;
    return true;
}

// ===== Test: XLA Threshold Function =====
bool test_xla_threshold() {
    std::cout << "Test: XLA Threshold... ";

    size_t original = eshkol::xla::xla_get_threshold();

    // Test setting threshold
    eshkol::xla::xla_set_threshold(50000);
    TEST_ASSERT(eshkol::xla::xla_get_threshold() == 50000, "Threshold should be 50000");

    eshkol::xla::xla_set_threshold(100000);
    TEST_ASSERT(eshkol::xla::xla_get_threshold() == 100000, "Threshold should be 100000");

    // Restore original
    eshkol::xla::xla_set_threshold(original);

    std::cout << "PASS" << std::endl;
    return true;
}

// ===== Main Test Runner =====
int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  XLA/StableHLO Integration Tests" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;

    // Create real arena (4MB — enough for all tests)
    g_test_arena = arena_create(4 * 1024 * 1024);
    if (!g_test_arena) {
        std::cerr << "FATAL: Could not create test arena" << std::endl;
        return 1;
    }

    // Run tests
    auto run_test = [](bool (*test_func)()) {
        if (test_func()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
    };

    // Runtime tests
    run_test(test_xla_runtime_init);
    run_test(test_xla_runtime_description);

    // Matmul tests
    run_test(test_xla_matmul_2x2);
    run_test(test_xla_matmul_3x2_2x4);
    run_test(test_xla_matmul_dim_mismatch);
    run_test(test_xla_matmul_invalid_rank);
    run_test(test_xla_matmul_large);

    // Threshold tests
    run_test(test_xla_threshold);

    // Cleanup
    arena_destroy(g_test_arena);

    // Summary
    std::cout << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "  Test Results" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    std::cout << std::endl;

    return (tests_failed == 0) ? 0 : 1;
}
