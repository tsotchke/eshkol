/**
 * @file mixed_curvature_step.h
 * @brief One training step of the mixed-curvature geometric model, on the host.
 *
 * WHY THIS FILE EXISTS.
 *
 * Stage 4 of the XLA-to-TPU program (`xla_training_step_parity`) compares a
 * device training step against the host's own. Building it began by looking
 * for the host step to mirror, and there was not one:
 *
 *   - lib/ml/ is three .esk files; its `adam` is Euclidean and knows no
 *     manifold.
 *   - lib/bridge/qllm_bridge.cpp has HYPERBOLIC forward AD nodes only, plus
 *     geodesic attention and cross entropy. No spherical or Euclidean manifold
 *     node, and no optimizer of any kind.
 *   - Riemannian Adam exists once, as `static` functions inside
 *     lib/backend/vm_geometric.c, reachable only through VM opcodes 839-842
 *     and 860-861, retracting in float32, and applying the HYPERBOLIC exp map
 *     to every parameter whatever manifold it belongs to (and, without
 *     ESHKOL_GEOMETRIC_ENABLED, applying no map at all and discarding the
 *     curvature).
 *
 * So a C++ harness could not "call the host's own training step" — there was
 * no entry point to call, and the rule that a harness must never re-implement
 * the thing it grades meant the step had to become real host code. It is this
 * module. docs/design/XLA_TRAINING_STEP.md is the written definition; every
 * formula here is transcribed from it, and it in turn from the S4 device
 * primitives and the tests/qllm_oracle exporters behind them.
 *
 * WHAT MAKES THE PARITY TEST A TEST AND NOT A TAUTOLOGY.
 *
 * This module's backward pass is hand-derived: each stage's VJP is written out
 * from the chain rule, with the derivation in a comment above it. The device
 * program's backward is produced by StableHLOEmitter::emitVJP walking the
 * forward graph, with no rule written for this step anywhere. The two are
 * independent derivations of the same mathematics, which is what a
 * disagreement between them is evidence about.
 *
 * Everything is f64 and everything is deterministic. There is no allocation
 * inside the step beyond the scratch buffer the caller sizes with
 * eshkol_mixed_curvature_scratch_elements(), and no global state at all: two
 * threads may step two models at once. That last point is not decorative —
 * the VM's Riemannian Adam keeps its moments in a 16-slot pool keyed by shape
 * on the VM itself (see docs/reference/stdlib/geometry.md), so two loops over
 * same-shaped parameters silently share one moment estimate there. A K-step
 * parity comparison cannot be run against state that behaves like that, and
 * this module carries its moments in the caller's own struct instead.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_ML_MIXED_CURVATURE_STEP_H
#define ESHKOL_ML_MIXED_CURVATURE_STEP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The step's shape signature. Also the device executable's cache key.
 *
 * @c n is batch*seq flattened: the step has no sequence-dependent operator in
 * it (no attention over positions), so the row index is the only one that
 * matters and carrying `batch` and `seq` separately would be two names for one
 * extent. The harness reports the batch and sequence it flattened.
 */
typedef struct {
    int64_t n;  /**< rows, = batch * seq */
    int64_t d;  /**< model dimension */
    int64_t c;  /**< prototypes (classes) */
} EshkolMixedCurvatureShape;

/**
 * @brief Curvature, optimizer and scoring hyperparameters.
 *
 * Curvature is here rather than baked into the module for the reason
 * geometric_lowering.h states: a mixed-curvature model LEARNS its curvature,
 * and the device executable is cached by shape, so a curvature compiled into
 * it would answer a second call at a different curvature with the first one's.
 */
typedef struct {
    double curvature;  /**< c > 0; the Poincare ball has curvature -c */
    double alpha;      /**< tangent scale applied before exp_0 */
    double lr;
    double beta1;
    double beta2;
    double adam_eps;   /**< OUTSIDE the sqrt, as vm_riemannian_adam_delta has it */
    double guard_eps;  /**< the oracle's clamp_min guard */
    double w_hyp;
    double w_sph;
    double w_euc;
} EshkolMixedCurvatureHyper;

/** @brief Defaults: c=1, alpha=0.5, lr=0.05, 0.9/0.999/1e-8, eps_g=1e-5, weights 1. */
void eshkol_mixed_curvature_default_hyper(EshkolMixedCurvatureHyper* out);

/**
 * @brief The four parameter tensors. Row-major, caller-owned.
 *
 * `p_hyp` rows live in the Poincare ball of curvature `-c`; `p_sph` rows live
 * on the unit sphere. Both invariants are re-established by every step's
 * retraction, and the harness checks them on the DEVICE's outputs.
 */
typedef struct {
    double* w;      /**< [d,d], Euclidean */
    double* p_hyp;  /**< [c,d], Poincare ball */
    double* p_sph;  /**< [c,d], unit sphere */
    double* p_euc;  /**< [c,d], Euclidean */
} EshkolMixedCurvatureParams;

/**
 * @brief Adam's first and second moments, one pair per parameter, plus the
 *        shared step count.
 *
 * `step` is the count BEFORE the step runs; eshkol_mixed_curvature_train_step
 * increments it, and the bias corrections use the incremented value, exactly
 * as vm_riemannian_adam_delta does.
 */
typedef struct {
    double* m_w;
    double* v_w;
    double* m_hyp;
    double* v_hyp;
    double* m_sph;
    double* v_sph;
    double* m_euc;
    double* v_euc;
    int64_t step;
} EshkolMixedCurvatureMoments;

/** @brief Element counts of each tensor, for a caller sizing its buffers. */
int64_t eshkol_mixed_curvature_w_elements(EshkolMixedCurvatureShape s);
int64_t eshkol_mixed_curvature_p_elements(EshkolMixedCurvatureShape s);
int64_t eshkol_mixed_curvature_x_elements(EshkolMixedCurvatureShape s);
int64_t eshkol_mixed_curvature_t_elements(EshkolMixedCurvatureShape s);

/** @brief Scratch f64 elements one step needs. Zero is not a valid answer. */
int64_t eshkol_mixed_curvature_scratch_elements(EshkolMixedCurvatureShape s);

/**
 * @brief Deterministically initialise parameters, moments, batch and targets.
 *
 * ONE generator, called by both legs of the parity harness, so that "the
 * device and the host started from identical parameters" is a property of a
 * single function rather than of two loops that were copied and then drifted.
 * `seed` selects a reproducible instance; nothing here consults a clock, an
 * address, or a global RNG.
 *
 * The manifold parameters come out already ON their manifolds (hyperbolic rows
 * strictly inside the ball, spherical rows unit-norm), because a step is not
 * defined at a point that is not on the manifold and starting off it would
 * make the first step's disagreement mean nothing.
 *
 * `targets` rows are non-negative and sum to one. Any of the pointers may be
 * null to skip that piece.
 */
void eshkol_mixed_curvature_init(EshkolMixedCurvatureShape s,
                                 const EshkolMixedCurvatureHyper* hyper,
                                 uint64_t seed,
                                 EshkolMixedCurvatureParams* params,
                                 EshkolMixedCurvatureMoments* moments,
                                 double* batch,
                                 double* targets);

/**
 * @brief Forward only: the loss at the current parameters.
 *
 * Exposed because the harness's monotone-decrease row wants the loss the
 * PARAMETERS produce, and reading the loss a step returned would report the
 * loss BEFORE that step's update — off by one, in the direction that makes a
 * broken optimizer look like a working one.
 *
 * @return false only on a null pointer or a non-positive extent.
 */
bool eshkol_mixed_curvature_loss(EshkolMixedCurvatureShape s,
                                 const EshkolMixedCurvatureHyper* hyper,
                                 const EshkolMixedCurvatureParams* params,
                                 const double* batch,
                                 const double* targets,
                                 double* scratch,
                                 double* out_loss);

/**
 * @brief One full training step: forward, backward, Riemannian Adam update.
 *
 * @param params    Updated in place. On failure they are untouched.
 * @param moments   Updated in place, including `step`.
 * @param out_loss  The loss BEFORE this step's update — the loss of the
 *                  parameters that produced the gradient. Optional.
 * @param out_grads Euclidean gradients of the four parameters, before the
 *                  Riemannian rescaling. Optional; pass null to skip. The
 *                  harness uses it to attribute a disagreement to the backward
 *                  pass rather than to the optimizer.
 *
 * @return false on a null pointer, a non-positive extent, or a non-positive
 *         curvature. There is no partial success: a step that could not be
 *         completed leaves every parameter and every moment as it found them,
 *         because a half-applied optimizer update is a wrong model that still
 *         runs.
 */
bool eshkol_mixed_curvature_train_step(EshkolMixedCurvatureShape s,
                                       const EshkolMixedCurvatureHyper* hyper,
                                       EshkolMixedCurvatureParams* params,
                                       EshkolMixedCurvatureMoments* moments,
                                       const double* batch,
                                       const double* targets,
                                       double* scratch,
                                       double* out_loss,
                                       EshkolMixedCurvatureParams* out_grads);

/**
 * @brief Check the manifold constraints on a parameter set.
 *
 * @param margin   Required slack: `c |x|^2 <= 1 - margin` for hyperbolic rows.
 * @param sph_tol  Allowed deviation of a spherical row's norm from 1.
 * @param worst_hyp  Largest `c |x|^2` seen. Optional.
 * @param worst_sph  Largest `| |x| - 1 |` seen. Optional.
 *
 * Deliberately takes the raw parameter pointers rather than the params struct,
 * so the harness can run it over the tensors the DEVICE returned without
 * copying them into a host model first.
 */
bool eshkol_mixed_curvature_constraints_hold(EshkolMixedCurvatureShape s,
                                             double curvature,
                                             const double* p_hyp,
                                             const double* p_sph,
                                             double margin,
                                             double sph_tol,
                                             double* worst_hyp,
                                             double* worst_sph);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* ESHKOL_ML_MIXED_CURVATURE_STEP_H */
