/*
 * The one comparator every XLA device/host parity harness grades with.
 *
 * WHY THIS FILE EXISTS.
 *
 * tests/xla/op_parity_test.cpp established the convention: a tolerance CLASS
 * per row, the rule "absolute or relative, whichever is looser", and a
 * negative control that runs before any device is required so that a
 * comparator which had regressed into `return true` is caught rather than
 * silently passing every row beneath it.
 *
 * The gradient harness needs exactly that convention. Copying it would create
 * a second one: two files that agree today and drift the first time a bound is
 * revised in one of them, after which the two gates report on different
 * contracts while both claim to report on docs/design/ESHKOL_S_FRAGMENT.md. So
 * the comparator, the tolerance classes and the control live here once, and
 * both harnesses include them.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_TESTS_XLA_PARITY_COMPARE_H
#define ESHKOL_TESTS_XLA_PARITY_COMPARE_H

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace eshkol_parity {

/**
 * @brief Which bound in docs/design/ESHKOL_S_FRAGMENT.md governs a row.
 *
 * Membership is decided by what the operation IS, not by what it measured
 * today. sin and cos currently come back at f32 rounding level on this
 * hardware, but they are still approximations of transcendental functions and
 * a future TPU generation, or a different input range, may evaluate them less
 * precisely without anything being wrong. Classifying by measurement would
 * mean re-deciding the contract every time a number moved.
 *
 * A GRADIENT ROW TAKES THE CLASS OF THE OPERATION IT DIFFERENTIATES. The VJP
 * of tanh is (1 - tanh(x)^2) times the cotangent: it evaluates the same
 * approximated elementary function the forward pass did, so it inherits the
 * same bound. The VJP of a matmul is two more matmuls, which are exact.
 */
enum class ToleranceClass {
    // Exact operations: their only error is the f32 rounding of inputs and
    // outputs. Add, subtract, multiply, divide, dot/matmul, every reduction,
    // and every pure data movement (transpose, reshape, broadcast).
    Arithmetic,
    // Approximated elementary functions evaluated by the device's
    // reduced-precision elementwise unit: exp, log, sin, cos, tanh, and
    // anything later added alongside them (sigmoid, sqrt, pow, erf).
    Transcendental
};

/**
 * @brief The two live bounds, set once from the device's element type.
 *
 * docs/design/ESHKOL_S_FRAGMENT.md, "parity rule": the arithmetic class keeps
 * the per-dtype bound (1e-5 f32, 1e-9 f64) and the transcendental class is
 * 100x it. Absolute or relative, whichever is looser, in both cases.
 */
inline double g_tol_arithmetic = 1e-5;
inline double g_tol_transcendental = 1e-3;

/** @brief Set both bounds from the device dtype name ("f32" / "f64" / "bf16"). */
inline void setTolerancesForDtype(const std::string& dtype) {
    if (dtype == "f64") {
        g_tol_arithmetic = 1e-9;
        g_tol_transcendental = g_tol_arithmetic * 100.0;
    } else if (dtype == "bf16") {
        // docs/design/ESHKOL_S_FRAGMENT.md: bf16 is the one dtype where the
        // transcendental class is NOT 100x the arithmetic bound — both stay
        // at the same unscaled 4e-2, because bf16's own ~3-decimal-digit
        // quantization already dominates whatever the elementwise unit adds.
        g_tol_arithmetic = 4e-2;
        g_tol_transcendental = 4e-2;
    } else {
        g_tol_arithmetic = 1e-5;
        g_tol_transcendental = g_tol_arithmetic * 100.0;
    }
}

inline const char* toleranceClassName(ToleranceClass c) {
    return c == ToleranceClass::Transcendental ? "transcendental" : "arithmetic";
}

inline double toleranceFor(ToleranceClass c) {
    return c == ToleranceClass::Transcendental ? g_tol_transcendental : g_tol_arithmetic;
}

struct Comparison {
    bool agreed = false;
    double max_abs = 0.0;
    double max_rel = 0.0;
    int worst_index = -1;
};

/**
 * @brief Compare device against host under the fragment contract's rule.
 *
 * An element agrees when |d - h| <= tol OR |d - h| <= tol * |h| — "absolute or
 * relative, whichever is looser", exactly as docs/design/ESHKOL_S_FRAGMENT.md
 * states it. Both errors are reported regardless, because a row that passes on
 * the absolute bound while its relative error is enormous is worth seeing.
 */
inline Comparison compareArrays(const std::vector<double>& device,
                                const std::vector<double>& host,
                                double tol) {
    Comparison c;
    if (device.size() != host.size() || device.empty()) return c;
    c.agreed = true;
    for (size_t i = 0; i < device.size(); ++i) {
        const double d = device[i];
        const double h = host[i];
        if (std::isnan(d) != std::isnan(h)) {
            c.agreed = false;
            if (c.worst_index < 0) c.worst_index = static_cast<int>(i);
            continue;
        }
        if (std::isnan(d)) continue;
        const double abs_err = std::fabs(d - h);
        const double rel_err = std::fabs(h) > 0.0 ? abs_err / std::fabs(h) : abs_err;
        if (abs_err > c.max_abs) c.max_abs = abs_err;
        if (rel_err > c.max_rel) c.max_rel = rel_err;
        const bool ok = (abs_err <= tol) || (abs_err <= tol * std::fabs(h));
        if (!ok) {
            c.agreed = false;
            if (c.worst_index < 0) c.worst_index = static_cast<int>(i);
        }
    }
    return c;
}

/**
 * @brief Negative control: the comparator must reject a wrong result.
 *
 * Runs before any device is required, so even a host with no PJRT plugin
 * proves that the thing grading every row is capable of returning
 * "disagreed". A comparator that always passed would make a whole harness
 * decorative.
 */
inline bool test_comparator_rejects_a_perturbed_result() {
    std::cout << "Control: comparator rejects a perturbed result... ";
    std::vector<double> host = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> exact = host;
    std::vector<double> wrong = host;
    wrong[2] += 1.0;   // far outside any tolerance in the contract

    Comparison good = compareArrays(exact, host, 1e-5);
    Comparison bad = compareArrays(wrong, host, 1e-5);

    if (!good.agreed) {
        std::cout << "FAIL (comparator rejected an exact match)" << std::endl;
        return false;
    }
    if (bad.agreed) {
        std::cout << "FAIL (comparator accepted a result off by 1.0)" << std::endl;
        return false;
    }
    if (bad.worst_index != 2) {
        std::cout << "FAIL (comparator reported the wrong index: " << bad.worst_index << ")"
                  << std::endl;
        return false;
    }
    std::cout << "PASS (exact accepted, off-by-1.0 rejected at index 2)" << std::endl;
    return true;
}

/** @brief Element count of a row-major shape; the empty shape is rank-0 (1). */
inline int64_t numElements(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

inline std::string shapeText(const std::vector<int64_t>& shape) {
    if (shape.empty()) return "[]";
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out += "x";
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

/** @brief Deterministic, well-conditioned test data. */
inline std::vector<double> makeData(int64_t n, double base, double step) {
    std::vector<double> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        v[static_cast<size_t>(i)] = base + step * static_cast<double>(i);
    }
    return v;
}

}  // namespace eshkol_parity

#endif  // ESHKOL_TESTS_XLA_PARITY_COMPARE_H
