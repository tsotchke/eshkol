/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_CUDA_MATMUL_BUDGET_H
#define ESHKOL_CUDA_MATMUL_BUDGET_H

#include <cmath>
#include <cstdlib>
#include <cstring>

/* A missing override disables the selector. Only a finite, positive budget
 * or the explicit "fast" setting may opt into a reduced-accuracy tier. */
static inline double eshkol_cuda_parse_matmul_budget(const char* text) {
    if (!text || !*text) return -1.0;
    if (std::strcmp(text, "fast") == 0) return 1e-2;

    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end != text && std::isfinite(value) && value > 0.0) {
        end += std::strspn(end, " \t\r\n\f\v");
        if (*end == '\0') return value;
    }
    return 1e-15;  // "exact", "full", and invalid input select native f64.
}
#endif
