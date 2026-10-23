/**
 * @file curvature_runtime.c
 * @brief Native marshalling for the ADR-0018 signed-curvature chart.
 *
 * Every entry point validates its tensors, calls the shared f64 implementation
 * in inc/eshkol/backend/riemannian_core.h, and copies the result into the
 * caller's output tensor. There is no geometry in this file: the closed forms,
 * series switches, domain errors and curvature jets are the ones the VM ops
 * and the AD bridge evaluate. See inc/eshkol/core/curvature_runtime.h.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

/* arena_memory.h declares thread-local storage with the C++/C23 spelling
 * `thread_local`; provide it for the C build of this translation unit (the
 * same shim lib/core/runtime_taylor.c uses). */
#if !defined(__cplusplus) && !defined(thread_local)
#  if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#    define thread_local _Thread_local
#  else
#    define thread_local
#  endif
#endif

#include "arena_memory.h"
#include "../../inc/eshkol/backend/riemannian_core.h"
#include "../../inc/eshkol/core/curvature_runtime.h"

#include <stdlib.h>
#include <string.h>

static ESHKOL_RUNTIME_THREAD_LOCAL const char* g_curvature_error = "";

const char* eshkol_curvature_last_error(void) { return g_curvature_error; }

static int32_t curv_fail(const char* why) {
    g_curvature_error = why ? why : "refused";
    return 1;
}

static int32_t curv_ok(void) {
    g_curvature_error = "";
    return 0;
}

/* A readable f64 tensor of exactly @p want elements (any when want < 0). */
static const char* curv_vec(const eshkol_tensor_t* t, int64_t want,
                            const double** data, int* n) {
    if (!t) return "expected an f64 tensor";
    if (t->dtype != ESHKOL_TENSOR_DTYPE_F64)
        return "expected an f64 tensor (the shared core is f64; curvature "
               "derivatives come from the jet functions)";
    if (t->total_elements == 0 || !t->elements)
        return "expected a non-empty tensor";
    if (t->total_elements > (uint64_t)INT32_MAX)
        return "the tensor is too large";
    if (want >= 0 && (int64_t)t->total_elements != want)
        return "tensor sizes do not match";
    *data = (const double*)t->elements;
    *n = (int)t->total_elements;
    return NULL;
}

static const char* curv_out(eshkol_tensor_t* t, int64_t want, double** data) {
    if (!t || t->dtype != ESHKOL_TENSOR_DTYPE_F64 || !t->elements)
        return "the output must be an f64 tensor";
    if ((int64_t)t->total_elements != want)
        return "the output tensor has the wrong size";
    *data = (double*)t->elements;
    return NULL;
}

int64_t eshkol_curvature_convention_id(void) {
    return (int64_t)ESHKOL_GEOMETRY_CONVENTION_STEREOGRAPHIC_V2_ALPHA1;
}

double eshkol_curvature_alpha(void) { return ESHKOL_RM_ALPHA; }

int32_t eshkol_curvature_check_header(int64_t convention, double alpha, double K) {
    eshkol_geometry_header_t h;
    if (convention < 0 || convention > (int64_t)UINT32_MAX)
        return curv_fail("the geometry record names an unknown convention id");
    h.convention = (uint32_t)convention;
    h.reserved = 0u;
    h.alpha = alpha;
    h.curvature = K;
    const char* why = eshkol_rm_check_geometry_header(&h);
    return why ? curv_fail(why) : curv_ok();
}

/* Two points of one dimension. */
static const char* curv_pair(const eshkol_tensor_t* x, const eshkol_tensor_t* y,
                             const double** xd, const double** yd, int* n) {
    const char* why = curv_vec(x, -1, xd, n);
    if (why) return why;
    int m = 0;
    return curv_vec(y, *n, yd, &m);
}

int32_t eshkol_curvature_distance(const eshkol_tensor_t* x,
                                  const eshkol_tensor_t* y, double K,
                                  eshkol_tensor_t* out) {
    const double *xd, *yd; double* o; int n;
    const char* why = curv_pair(x, y, &xd, &yd, &n);
    if (!why) why = curv_out(out, 1, &o);
    if (!why) why = eshkol_rm_distance(xd, yd, K, n, &o[0]);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_sq_distance(const eshkol_tensor_t* x,
                                     const eshkol_tensor_t* y, double K,
                                     eshkol_tensor_t* out) {
    const double *xd, *yd; double* o; int n;
    const char* why = curv_pair(x, y, &xd, &yd, &n);
    if (!why) why = curv_out(out, 1, &o);
    if (!why) why = eshkol_rm_sq_distance(xd, yd, K, n, &o[0]);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_conformal_factor(const eshkol_tensor_t* x, double K,
                                          eshkol_tensor_t* out) {
    const double* xd; double* o; int n;
    const char* why = curv_vec(x, -1, &xd, &n);
    if (!why) why = curv_out(out, 1, &o);
    if (!why) why = eshkol_rm_check_point(xd, K, n);
    if (!why) {
        o[0] = eshkol_rm_lambda(xd, K, n);
        if (!isfinite(o[0])) why = "the conformal factor is not finite";
    }
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_distance_jet(const eshkol_tensor_t* x,
                                      const eshkol_tensor_t* y, double K,
                                      eshkol_tensor_t* out) {
    const double *xd, *yd; double* o; int n;
    const char* why = curv_pair(x, y, &xd, &yd, &n);
    if (!why) why = curv_out(out, 3, &o);
    if (!why) why = eshkol_rm_distance_dK(xd, yd, K, n, &o[0], &o[1], &o[2]);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_sq_distance_jet(const eshkol_tensor_t* x,
                                         const eshkol_tensor_t* y, double K,
                                         eshkol_tensor_t* out) {
    const double *xd, *yd; double* o; int n;
    const char* why = curv_pair(x, y, &xd, &yd, &n);
    if (!why) why = curv_out(out, 3, &o);
    if (!why) why = eshkol_rm_sq_distance_dK(xd, yd, K, n, &o[0], &o[1], &o[2]);
    return why ? curv_fail(why) : curv_ok();
}

/* mult*n (+2) doubles of zeroed heap scratch; the caller frees it. */
static double* curv_scratch(int n, int mult) {
    return (double*)calloc((size_t)n * (size_t)mult + 2u, sizeof(double));
}

int32_t eshkol_curvature_sq_distance_grad(const eshkol_tensor_t* x,
                                          const eshkol_tensor_t* y, double K,
                                          eshkol_tensor_t* out) {
    const double *xd, *yd; double* o; int n;
    const char* why = curv_pair(x, y, &xd, &yd, &n);
    if (!why) why = curv_out(out, n, &o);
    if (why) return curv_fail(why);
    double* scratch = curv_scratch(n, 2);
    if (!scratch) return curv_fail("out of memory");
    why = eshkol_rm_sq_distance_grad(xd, yd, K, n, o, NULL, scratch);
    free(scratch);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_exp_map(const eshkol_tensor_t* x,
                                 const eshkol_tensor_t* v, double K,
                                 eshkol_tensor_t* out) {
    const double *xd, *vd; double* o; int n;
    const char* why = curv_pair(x, v, &xd, &vd, &n);
    if (!why) why = curv_out(out, n, &o);
    if (why) return curv_fail(why);
    double* scratch = curv_scratch(n, 1);
    if (!scratch) return curv_fail("out of memory");
    why = eshkol_rm_exp_map(xd, vd, K, n, o, scratch);
    free(scratch);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_log_map(const eshkol_tensor_t* x,
                                 const eshkol_tensor_t* y, double K,
                                 eshkol_tensor_t* out) {
    const double *xd, *yd; double* o; int n;
    const char* why = curv_pair(x, y, &xd, &yd, &n);
    if (!why) why = curv_out(out, n, &o);
    if (why) return curv_fail(why);
    double* scratch = curv_scratch(n, 1);
    if (!scratch) return curv_fail("out of memory");
    why = eshkol_rm_log_map(xd, yd, K, n, o, scratch);
    free(scratch);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_transport(const eshkol_tensor_t* x,
                                   const eshkol_tensor_t* y,
                                   const eshkol_tensor_t* v, double K,
                                   eshkol_tensor_t* out) {
    const double *xd, *yd, *vd; double* o; int n, m = 0;
    const char* why = curv_pair(x, y, &xd, &yd, &n);
    if (!why) why = curv_vec(v, n, &vd, &m);
    if (!why) why = curv_out(out, n, &o);
    if (why) return curv_fail(why);
    double* scratch = curv_scratch(n, 1);
    if (!scratch) return curv_fail("out of memory");
    why = eshkol_rm_transport(xd, yd, vd, K, n, o, scratch);
    free(scratch);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_mobius_add(const eshkol_tensor_t* x,
                                    const eshkol_tensor_t* y, double K,
                                    eshkol_tensor_t* out) {
    const double *xd, *yd; double* o; int n;
    const char* why = curv_pair(x, y, &xd, &yd, &n);
    if (!why) why = curv_out(out, n, &o);
    if (!why) why = eshkol_rm_chart_mobius_add(xd, yd, K, n, o);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_mobius_scalar(double r, const eshkol_tensor_t* x,
                                       double K, eshkol_tensor_t* out) {
    const double* xd; double* o; int n;
    const char* why = curv_vec(x, -1, &xd, &n);
    if (!why) why = curv_out(out, n, &o);
    if (!why) why = eshkol_rm_mobius_scalar(r, xd, K, n, o);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_project(const eshkol_tensor_t* x, double K,
                                 eshkol_tensor_t* out) {
    const double* xd; double* o; int n;
    const char* why = curv_vec(x, -1, &xd, &n);
    if (!why) why = curv_out(out, n, &o);
    if (!why) why = eshkol_rm_project(xd, K, n, o);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_riemannian_grad(const eshkol_tensor_t* g,
                                         const eshkol_tensor_t* x, double K,
                                         eshkol_tensor_t* out) {
    const double *gd, *xd; double* o; int n;
    const char* why = curv_pair(g, x, &gd, &xd, &n);
    if (!why) why = curv_out(out, n, &o);
    if (!why) why = eshkol_rm_egrad_to_rgrad(gd, xd, K, n, o);
    return why ? curv_fail(why) : curv_ok();
}

/* Rows and columns of a rank-1 (one row) or rank-2 tensor. */
static const char* curv_matrix(const eshkol_tensor_t* t, int64_t* rows,
                               int64_t* cols) {
    if (!t || !t->dimensions) return "expected a rank-1 or rank-2 f64 tensor";
    if (t->num_dimensions == 1) {
        *rows = 1;
        *cols = (int64_t)t->dimensions[0];
    } else if (t->num_dimensions == 2) {
        *rows = (int64_t)t->dimensions[0];
        *cols = (int64_t)t->dimensions[1];
    } else {
        return "expected a rank-1 or rank-2 f64 tensor";
    }
    if (*rows <= 0 || *cols <= 0) return "expected a non-empty tensor";
    return NULL;
}

int32_t eshkol_curvature_attention(const eshkol_tensor_t* q,
                                   const eshkol_tensor_t* keys,
                                   const eshkol_tensor_t* v, double K,
                                   double temperature, eshkol_tensor_t* out,
                                   eshkol_tensor_t* weights) {
    int64_t nq, d, nk, dk, nv, dv;
    const double *qd, *kd, *vd; double *od, *wd; int tq, tk, tv;
    const char* why = curv_matrix(q, &nq, &d);
    if (!why) why = curv_matrix(keys, &nk, &dk);
    if (!why) why = curv_matrix(v, &nv, &dv);
    if (!why && dk != d) why = "queries and keys must have the same dimension";
    if (!why && nv != nk) why = "there must be one value row per key";
    if (!why) why = curv_vec(q, nq * d, &qd, &tq);
    if (!why) why = curv_vec(keys, nk * d, &kd, &tk);
    if (!why) why = curv_vec(v, nk * dv, &vd, &tv);
    if (!why) why = curv_out(out, nq * dv, &od);
    if (!why) why = curv_out(weights, nq * nk, &wd);
    if (why) return curv_fail(why);
    if (nk > INT32_MAX || d > INT32_MAX) return curv_fail("the attention is too large");
    double* sqd = curv_scratch((int)nk, 1);
    if (!sqd) return curv_fail("out of memory");
    for (int64_t i = 0; i < nq; i++) {
        double* row = wd + i * nk;
        why = eshkol_rm_attention_row(qd + i * d, kd, (long)d, (int)nk, (int)d,
                                      K, temperature, sqd, row);
        if (why) break;
        for (int64_t c = 0; c < dv; c++) {
            double acc = 0.0;
            for (int64_t j = 0; j < nk; j++) acc += row[j] * vd[j * dv + c];
            od[i * dv + c] = acc;
        }
    }
    free(sqd);
    if (!why) why = eshkol_rm_check_output(od, (int)(nq * dv),
                                           "the attention output is not finite");
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_sphere_to_chart(const eshkol_tensor_t* p, double K,
                                         eshkol_tensor_t* out) {
    const double* pd; double* o; int n;
    const char* why = curv_vec(p, -1, &pd, &n);
    if (!why && n < 2) why = "an embedded sphere point needs at least two coordinates";
    if (!why) why = curv_out(out, n - 1, &o);
    if (!why) why = eshkol_rm_sphere_to_chart(pd, K, n - 1, o);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_chart_to_sphere(const eshkol_tensor_t* x, double K,
                                         eshkol_tensor_t* out) {
    const double* xd; double* o; int n;
    const char* why = curv_vec(x, -1, &xd, &n);
    if (!why) why = curv_out(out, (int64_t)n + 1, &o);
    if (!why) why = eshkol_rm_chart_to_sphere(xd, K, n, o);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_migrate_point(const eshkol_tensor_t* x, double K,
                                       eshkol_tensor_t* out) {
    const double* xd; double* o; int n;
    const char* why = curv_vec(x, -1, &xd, &n);
    if (!why) why = curv_out(out, n, &o);
    if (!why) why = eshkol_rm_migrate_legacy_point(xd, K, n, o);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_migrate_tangent(const eshkol_tensor_t* v, double K,
                                         eshkol_tensor_t* out) {
    const double* vd; double* o; int n;
    const char* why = curv_vec(v, -1, &vd, &n);
    if (!why) why = curv_out(out, n, &o);
    if (!why) why = eshkol_rm_migrate_legacy_tangent(vd, K, n, o);
    return why ? curv_fail(why) : curv_ok();
}

int32_t eshkol_curvature_migrate_temperature(double legacy_scale,
                                             double reference_distance,
                                             eshkol_tensor_t* out) {
    double* o;
    const char* why = curv_out(out, 1, &o);
    if (!why) why = eshkol_rm_migrate_legacy_temperature(legacy_scale,
                                                         reference_distance, &o[0]);
    return why ? curv_fail(why) : curv_ok();
}
