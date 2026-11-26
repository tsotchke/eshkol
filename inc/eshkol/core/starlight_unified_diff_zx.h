/*
 * Starlight (v2.0) Unified Differentiate Operator, Exact ZX-Calculus Quantum
 * Cognitive Interference Engine, and Lean 4.12.0 Kernel Re-Verification Pipeline
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_STARLIGHT_UNIFIED_DIFF_ZX_H
#define ESHKOL_CORE_STARLIGHT_UNIFIED_DIFF_ZX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_ce12_status {
    ESHKOL_CE12_OK                         = 0,
    ESHKOL_CE12_ERR_INVALID_ARG            = 1,
    ESHKOL_UDIFF_ERR_MODE_DISAGREEMENT     = 2,
    ESHKOL_ZX_ERR_INTERFERENCE_SINGULARITY = 3,
    ESHKOL_STARLIGHT_ERR_KERNEL_DRIFT      = 4
} eshkol_ce12_status_t;

/* ============================================================================
 * Part 1: Unified `differentiate` Operator (Numeric AD / Implicit / DBSP Delta)
 * ============================================================================ */

typedef enum eshkol_udiff_mode {
    ESHKOL_UDIFF_NUMERIC     = 0, /* Exact dual-number numeric AD */
    ESHKOL_UDIFF_IMPLICIT_FP = 1, /* Implicit equilibrium differentiation (IFT) */
    ESHKOL_UDIFF_DBSP_DELTA  = 2  /* Incremental DBSP belief delta derivative */
} eshkol_udiff_mode_t;

typedef enum eshkol_udiff_fn_kind {
    ESHKOL_UDIFF_FN_LINEAR_DBSP        = 0, /* f(x) = a * x + b */
    ESHKOL_UDIFF_FN_SMOOTH_EQUILIBRIUM = 1, /* z* = c * tanh(z*) + a * x + b, |c| < 1 */
    ESHKOL_UDIFF_FN_ABS_KINK           = 2  /* f(x) = a * |x - c| + b (kink at x = c) */
} eshkol_udiff_fn_kind_t;

typedef struct eshkol_udiff_spec {
    eshkol_udiff_fn_kind_t kind;
    double a;
    double b;
    double c;
} eshkol_udiff_spec_t;

typedef struct eshkol_udiff_result {
    double value;
    double deriv_numeric;
    double deriv_implicit_fp;
    double deriv_dbsp_delta;
    double max_mode_disagreement;
    uint64_t fd_evals; /* Always 0 (pure analytical / IFT / DBSP tangent) */
} eshkol_udiff_result_t;

/*
 * Evaluate all three differentiation semantics (Numeric AD, Implicit Equilibrium,
 * DBSP Delta) and cross-check consistency within `consistency_tol`.
 * Fails closed with ESHKOL_UDIFF_ERR_MODE_DISAGREEMENT if modes diverge (e.g.
 * at a non-smooth kink) rather than silently picking one subgradient.
 */
eshkol_ce12_status_t eshkol_udiff_eval_unified(
    const eshkol_udiff_spec_t* spec,
    double x,
    double consistency_tol,
    eshkol_udiff_result_t* out_res);

/* ============================================================================
 * Part 2: Exact ZX-Calculus Quantum Cognitive Interference Engine
 * ============================================================================ */

typedef struct eshkol_zx_circuit {
    double w1;   /* Path-1 amplitude magnitude */
    double w2;   /* Path-2 amplitude magnitude */
    double phi;  /* Relative X/Z-spider phase offset (radians) */
} eshkol_zx_circuit_t;

typedef struct eshkol_zx_eval_result {
    double psi_re;
    double psi_im;
    double prob;                 /* Born probability |psi(theta)|^2 */
    double d_prob_analytical;    /* Exact analytical dP/dtheta */
    double d_prob_param_shift;   /* Exact ZX parameter-shift rule P(t+pi/4)-P(t-pi/4) */
    double d_phase_analytical;   /* Exact analytical d(arg psi)/dtheta */
    double amplitude_norm;       /* |psi(theta)| */
    uint64_t fd_evals;           /* Always 0 */
} eshkol_zx_eval_result_t;

/*
 * Evaluate ZX-calculus two-path spider interference and exact parameter-shift
 * derivative at angle `theta`. Detects destructive interference singularities
 * (|psi(theta)| < singularity_eps) where the phase branch-cut contradicts the
 * classical zero-gradient point and fails closed with
 * ESHKOL_ZX_ERR_INTERFERENCE_SINGULARITY.
 */
eshkol_ce12_status_t eshkol_zx_eval_interference(
    const eshkol_zx_circuit_t* circuit,
    double theta,
    double singularity_eps,
    eshkol_zx_eval_result_t* out_res);

/* ============================================================================
 * Part 3: Starlight Lean 4.12.0 Kernel Re-Verification Pipeline
 * ============================================================================ */

typedef struct eshkol_starlight_summary {
    uint32_t lob_rank_before;
    uint32_t lob_rank_after;
    uint32_t linear_resource_uses;
    uint32_t erased_runtime_uses;
    bool cubical_path_acyclic;
    double udiff_numeric;
    double udiff_implicit_fp;
    double udiff_dbsp_delta;
    double zx_prob;
    double zx_d_prob_analytical;
    double zx_d_prob_param_shift;
    double zx_amplitude_norm;
} eshkol_starlight_summary_t;

/*
 * Export standalone Lean 4.12.0 proof artifact certifying CE-11 Lob/QTT safety
 * + CE-12 unified differentiation consistency + ZX parameter-shift exactness,
 * and write the bit-exact SHA-256 digest of the exported bytes into
 * `out_artifact_sha256`.
 */
bool eshkol_starlight_export_lean4(
    const eshkol_starlight_summary_t* summary,
    const char* module_namespace,
    char* out_buf,
    size_t out_cap,
    size_t* out_len,
    uint8_t out_artifact_sha256[32]);

/*
 * Fail-closed kernel re-verification of an exported Lean 4.12.0 proof artifact:
 *   1. Recomputes SHA-256 of the file bytes and rejects ANY 1-bit drift with
 *      ESHKOL_STARLIGHT_ERR_KERNEL_DRIFT.
 *   2. Invokes the Lean 4.12.0 kernel binary on the file and rejects any
 *      non-zero kernel exit with ESHKOL_STARLIGHT_ERR_KERNEL_DRIFT.
 */
eshkol_ce12_status_t eshkol_starlight_verify_lean4_artifact(
    const char* lean_file_path,
    const uint8_t expected_artifact_sha256[32],
    const char* lean4_bin_path,
    int* out_lean_exit_code);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_STARLIGHT_UNIFIED_DIFF_ZX_H */
