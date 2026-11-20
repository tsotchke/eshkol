/**
 * @file curvature_runtime.h
 * @brief Native (JIT/AOT) entry points of the ADR-0018 signed-curvature chart.
 *
 * These are the `extern` targets of `core.curvature` (lib/core/curvature.esk).
 * Every function is a thin marshalling layer over the ONE f64 implementation in
 * inc/eshkol/backend/riemannian_core.h, the same core the bytecode VM's
 * geometry ops and the AD bridge call, so a native program and a VM program
 * compute the same bits for the same inputs.
 *
 * Calling convention: points and vectors are f64 tensors (`eshkol_tensor_t*`,
 * dtype F64); results are written into a caller-allocated output tensor of the
 * documented length. Each function returns 0 on success and 1 on a refusal,
 * whose reason eshkol_curvature_last_error() returns (thread-local, valid until
 * the next call on the same thread).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_CORE_CURVATURE_RUNTIME_H
#define ESHKOL_CORE_CURVATURE_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct eshkol_tensor;

/** Reason for the last refusal on this thread. */
const char* eshkol_curvature_last_error(void);
/** ESHKOL_GEOMETRY_CONVENTION_STEREOGRAPHIC_V2_ALPHA1. */
int64_t eshkol_curvature_convention_id(void);
/** The gauge alpha (1.0). */
double eshkol_curvature_alpha(void);
/** 0 when (convention, alpha, K) names the current convention, else 1. */
int32_t eshkol_curvature_check_header(int64_t convention, double alpha, double K);

/* out[0] */
int32_t eshkol_curvature_distance(const struct eshkol_tensor* x,
                                  const struct eshkol_tensor* y, double K,
                                  struct eshkol_tensor* out);
int32_t eshkol_curvature_sq_distance(const struct eshkol_tensor* x,
                                     const struct eshkol_tensor* y, double K,
                                     struct eshkol_tensor* out);
int32_t eshkol_curvature_conformal_factor(const struct eshkol_tensor* x, double K,
                                          struct eshkol_tensor* out);
/* out[0..2] = value, d/dK, d^2/dK^2 at fixed chart points */
int32_t eshkol_curvature_distance_jet(const struct eshkol_tensor* x,
                                      const struct eshkol_tensor* y, double K,
                                      struct eshkol_tensor* out);
int32_t eshkol_curvature_sq_distance_jet(const struct eshkol_tensor* x,
                                         const struct eshkol_tensor* y, double K,
                                         struct eshkol_tensor* out);
/* out[n] = d(d^2)/dx, the coordinate gradient -2 lambda_x^2 log_x(y) */
int32_t eshkol_curvature_sq_distance_grad(const struct eshkol_tensor* x,
                                          const struct eshkol_tensor* y, double K,
                                          struct eshkol_tensor* out);
/* out[n] */
int32_t eshkol_curvature_exp_map(const struct eshkol_tensor* x,
                                 const struct eshkol_tensor* v, double K,
                                 struct eshkol_tensor* out);
int32_t eshkol_curvature_log_map(const struct eshkol_tensor* x,
                                 const struct eshkol_tensor* y, double K,
                                 struct eshkol_tensor* out);
int32_t eshkol_curvature_transport(const struct eshkol_tensor* x,
                                   const struct eshkol_tensor* y,
                                   const struct eshkol_tensor* v, double K,
                                   struct eshkol_tensor* out);
int32_t eshkol_curvature_mobius_add(const struct eshkol_tensor* x,
                                    const struct eshkol_tensor* y, double K,
                                    struct eshkol_tensor* out);
int32_t eshkol_curvature_mobius_scalar(double r, const struct eshkol_tensor* x,
                                       double K, struct eshkol_tensor* out);
int32_t eshkol_curvature_project(const struct eshkol_tensor* x, double K,
                                 struct eshkol_tensor* out);
int32_t eshkol_curvature_riemannian_grad(const struct eshkol_tensor* g,
                                         const struct eshkol_tensor* x, double K,
                                         struct eshkol_tensor* out);
/* Q [nq, d], keys [nk, d], V [nk, dv]; out [nq, dv]; weights [nq, nk] */
int32_t eshkol_curvature_attention(const struct eshkol_tensor* q,
                                   const struct eshkol_tensor* keys,
                                   const struct eshkol_tensor* v, double K,
                                   double temperature, struct eshkol_tensor* out,
                                   struct eshkol_tensor* weights);
/* embedded sphere p in R^{n+1} <-> chart x in R^n */
int32_t eshkol_curvature_sphere_to_chart(const struct eshkol_tensor* p, double K,
                                         struct eshkol_tensor* out);
int32_t eshkol_curvature_chart_to_sphere(const struct eshkol_tensor* x, double K,
                                         struct eshkol_tensor* out);
/* legacy (LEGACY_BALL_V1) -> current */
int32_t eshkol_curvature_migrate_point(const struct eshkol_tensor* x, double K,
                                       struct eshkol_tensor* out);
int32_t eshkol_curvature_migrate_tangent(const struct eshkol_tensor* v, double K,
                                         struct eshkol_tensor* out);
int32_t eshkol_curvature_migrate_temperature(double legacy_scale,
                                             double reference_distance,
                                             struct eshkol_tensor* out);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_CURVATURE_RUNTIME_H */
