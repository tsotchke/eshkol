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

// Simple arena for testing
class TestArena {
public:
    TestArena(size_t size) : buffer_(new char[size]), offset_(0), size_(size) {}
    ~TestArena() { delete[] buffer_; }

    void* alloc(size_t size, size_t alignment) {
        // Align offset
        size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
        if (aligned + size > size_) return nullptr;
        void* ptr = buffer_ + aligned;
        offset_ = aligned + size;
        return ptr;
    }

    void reset() { offset_ = 0; }

private:
    char* buffer_;
    size_t offset_;
    size_t size_;
};

// Global test arena
static TestArena* g_test_arena = nullptr;

// Arena allocator for tests
extern "C" void* eshkol_arena_alloc(void* arena, size_t size, size_t alignment) {
    if (arena == g_test_arena) {
        return g_test_arena->alloc(size, alignment);
    }
    return nullptr;
}

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

    g_test_arena->reset();

    void* result = eshkol_xla_matmul(
        g_test_arena,
        a_data, b_data,
        a_shape, b_shape,
        2, 2);

    TEST_ASSERT(result != nullptr, "Matmul should return non-null result");

    // Extract result tensor
    struct Tensor {
        int64_t num_dims;
        int64_t* dims;
        double* data;
    };
    auto* tensor = static_cast<Tensor*>(result);

    TEST_ASSERT(tensor->num_dims == 2, "Result should have 2 dimensions");
    TEST_ASSERT(tensor->dims[0] == 2, "Result dim 0 should be 2");
    TEST_ASSERT(tensor->dims[1] == 2, "Result dim 1 should be 2");

    // Verify values
    double expected[] = {19.0, 22.0, 43.0, 50.0};
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_NEAR(tensor->data[i], expected[i], 1e-10,
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

    g_test_arena->reset();

    void* result = eshkol_xla_matmul(
        g_test_arena,
        a_data, b_data,
        a_shape, b_shape,
        2, 2);

    TEST_ASSERT(result != nullptr, "Matmul should return non-null result");

    struct Tensor {
        int64_t num_dims;
        int64_t* dims;
        double* data;
    };
    auto* tensor = static_cast<Tensor*>(result);

    TEST_ASSERT(tensor->num_dims == 2, "Result should have 2 dimensions");
    TEST_ASSERT(tensor->dims[0] == 3, "Result dim 0 should be 3");
    TEST_ASSERT(tensor->dims[1] == 4, "Result dim 1 should be 4");

    // Expected: C[i][j] = sum_k A[i][k] * B[k][j]
    // C[0][0] = 1*1 + 2*5 = 11
    // C[0][1] = 1*2 + 2*6 = 14
    // etc.
    double expected[] = {
        11.0, 14.0, 17.0, 20.0,
        23.0, 30.0, 37.0, 44.0,
        35.0, 46.0, 57.0, 68.0
    };

    for (int i = 0; i < 12; i++) {
        TEST_ASSERT_NEAR(tensor->data[i], expected[i], 1e-10,
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

    g_test_arena->reset();

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

    g_test_arena->reset();

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

    g_test_arena->reset();

    void* result = eshkol_xla_matmul(
        g_test_arena,
        a_data.data(), b_data.data(),
        shape, shape,
        2, 2);

    TEST_ASSERT(result != nullptr, "Matmul should return non-null result");

    struct Tensor {
        int64_t num_dims;
        int64_t* dims;
        double* data;
    };
    auto* tensor = static_cast<Tensor*>(result);

    TEST_ASSERT(tensor->dims[0] == N, "Result dim 0 should be N");
    TEST_ASSERT(tensor->dims[1] == N, "Result dim 1 should be N");

    // Identity * B = B
    for (int i = 0; i < N * N; i++) {
        TEST_ASSERT_NEAR(tensor->data[i], b_data[i], 1e-10,
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

#ifdef ESHKOL_XLA_FULL_MLIR
// ===== Test: XLA Codegen Availability =====
bool test_xla_codegen_available() {
    std::cout << "Test: XLA Codegen Availability... ";

    // This test only runs when full MLIR is available
    // We can't easily test XLACodegen without a CodegenContext,
    // so we just verify the headers compile and threshold works

    TEST_ASSERT(eshkol::xla::xla_get_threshold() > 0, "Threshold should be positive");

    std::cout << "PASS (MLIR available)" << std::endl;
    return true;
}
#endif

// ===== Main Test Runner =====
int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  XLA/StableHLO Integration Tests" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;

    // Initialize test arena (1MB)
    g_test_arena = new TestArena(1024 * 1024);

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

#ifdef ESHKOL_XLA_FULL_MLIR
    run_test(test_xla_codegen_available);
#endif

    // Cleanup
    delete g_test_arena;

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
