/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "cuda_matmul_budget.h"
#include "eshkol/backend/gpu/matmul_range_guard.h"
#include <cmath>
#include <cstdio>
#include <limits>

int main() {
    const struct { const char* input; double expected; } cases[] = {
        {nullptr, -1.0}, {"", -1.0}, {"fast", 1e-2}, {"exact", 1e-15},
        {"full", 1e-15}, {"1e-4", 1e-4}, {"  0.002\t", 0.002},
        {"0", 1e-15}, {"-1", 1e-15}, {"   ", 1e-15}, {"NaN", 1e-15},
        {"inf", 1e-15}, {"1e999", 1e-15}, {"0.01oops", 1e-15},
        {"1e-4 1e-2", 1e-15}, {"unknown", 1e-15},
    };
    for (const auto& test : cases) {
        const double value = eshkol_cuda_parse_matmul_budget(test.input);
        if (value != test.expected) {
            std::fprintf(stderr, "FAIL: budget '%s': expected %.17g, got %.17g\n",
                         test.input ? test.input : "(unset)", test.expected, value);
            return 1;
        }
    }

    const struct {
        const char* accuracy;
        const char* precision;
        double expected;
    } resolve_cases[] = {
        {nullptr, nullptr, -1.0},
        {"", "", -1.0},
        {nullptr, "exact", 1e-15},
        {nullptr, "high", 1e-11},
        {nullptr, "fast", 1e-5},
        {nullptr, "unknown", 1e-15},
        {"1e-7", "fast", 1e-7},
        {"exact", "fast", 1e-15},
        {"fast", "exact", 1e-2},
    };
    for (const auto& test : resolve_cases) {
        const double value = eshkol_cuda_resolve_matmul_budget(test.accuracy, test.precision);
        if (value != test.expected) {
            std::fprintf(stderr, "FAIL: resolve acc='%s' prec='%s': expected %.17g, got %.17g\n",
                         test.accuracy ? test.accuracy : "(unset)",
                         test.precision ? test.precision : "(unset)",
                         test.expected, value);
            return 1;
        }
    }
    // Operand dynamic-range guard controls (prevent silent f32/df64 underflow/overflow).
    {
        const double tiny = std::ldexp(1.0, -180);
        const double huge = std::ldexp(1.0, 180);
        const double a_tiny[4] = {tiny, 0.0, 0.0, tiny};
        const double b_unit[4] = {1.0, 0.0, 0.0, 1.0};
        const double a_huge[4] = {huge, 0.0, 0.0, huge};
        const double a_norm[4] = {1.5, -0.0, 0.0, -2.25};
        const double b_norm[4] = {0.5, 3.0, -1.0, 0.0};
        const double a_nan[4] = {1.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0};
        const double a_inf[4] = {1.0, std::numeric_limits<double>::infinity(), 0.0, 1.0};
        const double a_df64_edge[4] = {std::ldexp(1.0, -50), 0.0, 0.0, 1.0};

        if (eshkol_f64_matmul_operands_safe_for_f32(a_tiny, b_unit, 2, 2, 2, 0) != 0) {
            std::fprintf(stderr, "FAIL: 2^-180 must be declined by f32 range guard\n");
            return 1;
        }
        if (eshkol_f64_matmul_operands_safe_for_f32(a_tiny, b_unit, 2, 2, 2, 1) != 0) {
            std::fprintf(stderr, "FAIL: 2^-180 must be declined by df64 range guard\n");
            return 1;
        }
        if (eshkol_f64_matmul_operands_safe_for_f32(a_huge, b_unit, 2, 2, 2, 0) != 0) {
            std::fprintf(stderr, "FAIL: 2^180 must be declined by f32 range guard\n");
            return 1;
        }
        if (eshkol_f64_matmul_operands_safe_for_f32(a_tiny, a_huge, 2, 2, 2, 0) != 0) {
            std::fprintf(stderr, "FAIL: mixed 2^-180 * 2^180 must be declined by f32 range guard\n");
            return 1;
        }
        if (eshkol_f64_matmul_operands_safe_for_f32(a_nan, b_norm, 2, 2, 2, 0) != 0 ||
            eshkol_f64_matmul_operands_safe_for_f32(a_inf, b_norm, 2, 2, 2, 0) != 0) {
            std::fprintf(stderr, "FAIL: non-finite operands must be declined\n");
            return 1;
        }
        if (eshkol_f64_matmul_operands_safe_for_f32(a_norm, b_norm, 2, 2, 2, 0) != 1 ||
            eshkol_f64_matmul_operands_safe_for_f32(a_norm, b_norm, 2, 2, 2, 1) != 1) {
            std::fprintf(stderr, "FAIL: normal in-range operands with +/-0.0 must be admitted\n");
            return 1;
        }
        if (eshkol_f64_matmul_operands_safe_for_f32(a_df64_edge, b_unit, 2, 2, 2, 0) != 1 ||
            eshkol_f64_matmul_operands_safe_for_f32(a_df64_edge, b_unit, 2, 2, 2, 1) != 0) {
            std::fprintf(stderr, "FAIL: 2^-50 boundary mismatch between f32 and df64 range guards\n");
            return 1;
        }
    }

    std::puts("PASS: CUDA matmul budget parsing and operand range guard");
    return 0;
}
