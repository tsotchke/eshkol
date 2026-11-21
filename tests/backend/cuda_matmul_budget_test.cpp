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
    std::puts("PASS: CUDA matmul budget parsing");
    return 0;
}
