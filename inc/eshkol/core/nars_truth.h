/*
 * Native NARS NAL-1..9 AIKR Truth-Value Calculus
 *
 * Implements Pei Wang's Non-Axiomatic Logic (NAL) truth-value functions over
 * (frequency, confidence) pairs subject to strict AIKR domain invariants:
 *   - 0.0 <= frequency <= 1.0 (finite)
 *   - 0.0 <= confidence < 1.0 (finite; finite evidence cannot reach infinity)
 *   - evidential horizon k > 0.0 (default k = 1.0)
 *
 * Every function validates its inputs and rejects NaN, Inf, and out-of-range
 * parameters by returning false.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_NARS_TRUTH_H
#define ESHKOL_CORE_NARS_TRUTH_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESHKOL_NARS_DEFAULT_HORIZON 1.0

typedef struct eshkol_nars_truth {
    double frequency;   /* f in [0.0, 1.0] */
    double confidence;  /* c in [0.0, 1.0) */
} eshkol_nars_truth_t;

typedef struct eshkol_nars_evidence {
    double positive;    /* w+ >= 0.0 */
    double total;       /* w  >= w+  */
} eshkol_nars_evidence_t;

/* Strict domain check: finite, 0 <= f <= 1, 0 <= c < 1. */
bool eshkol_nars_truth_is_valid(eshkol_nars_truth_t tv);

/* Construct and validate a truth value. Returns false on domain violation. */
bool eshkol_nars_make_truth(double frequency, double confidence, eshkol_nars_truth_t* out);

/* Convert between (f, c) and positive/total evidence (w+, w) with horizon k > 0. */
bool eshkol_nars_to_evidence(eshkol_nars_truth_t tv, double k, eshkol_nars_evidence_t* out);
bool eshkol_nars_from_evidence(double w_pos, double w_total, double k, eshkol_nars_truth_t* out);

/* Expectation e = c * (f - 0.5) + 0.5. Returns false if tv is invalid. */
bool eshkol_nars_expectation(eshkol_nars_truth_t tv, double* out_expectation);

/* Unary truth functions (NAL-1..5) */
bool eshkol_nars_negation(eshkol_nars_truth_t tv, eshkol_nars_truth_t* out);
bool eshkol_nars_conversion(eshkol_nars_truth_t tv, double k, eshkol_nars_truth_t* out);
bool eshkol_nars_contraposition(eshkol_nars_truth_t tv, double k, eshkol_nars_truth_t* out);

/* Binary strong & weak syllogistic rules (NAL-1..9) */
bool eshkol_nars_deduction(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                           eshkol_nars_truth_t* out);

bool eshkol_nars_abduction(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                           double k, eshkol_nars_truth_t* out);

bool eshkol_nars_induction(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                           double k, eshkol_nars_truth_t* out);

bool eshkol_nars_exemplification(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                                 double k, eshkol_nars_truth_t* out);

bool eshkol_nars_revision(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                          double k, eshkol_nars_truth_t* out);

bool eshkol_nars_analogy(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                         eshkol_nars_truth_t* out);

bool eshkol_nars_comparison(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                            double k, eshkol_nars_truth_t* out);

bool eshkol_nars_intersection(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                              eshkol_nars_truth_t* out);

bool eshkol_nars_union(eshkol_nars_truth_t v1, eshkol_nars_truth_t v2,
                       eshkol_nars_truth_t* out);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_NARS_TRUTH_H */
