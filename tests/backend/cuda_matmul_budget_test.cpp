/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "cuda_matmul_budget.h"
#include <cstdio>

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
    std::puts("PASS: CUDA matmul budget parsing");
    return 0;
}
