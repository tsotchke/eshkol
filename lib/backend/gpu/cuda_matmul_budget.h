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

/* Resolve the effective CUDA matmul accuracy budget from ESHKOL_MATMUL_ACCURACY
 * (taking precedence when set) and ESHKOL_GPU_PRECISION ("exact" / "high" /
 * "fast", matching Metal and WebGPU precision tiers). Invalid values fail closed
 * to 1e-15 (native cuBLAS DGEMM). */
static inline double eshkol_cuda_resolve_matmul_budget(const char* accuracy_env,
                                                       const char* precision_env) {
    if (accuracy_env && *accuracy_env) {
        return eshkol_cuda_parse_matmul_budget(accuracy_env);
    }
    if (!precision_env || !*precision_env) {
        return -1.0;
    }
    if (std::strcmp(precision_env, "fast") == 0) return 1e-5;
    if (std::strcmp(precision_env, "high") == 0) return 1e-11;
    return 1e-15;  // "exact" and unknown tiers fail closed to native f64.
}
#endif
