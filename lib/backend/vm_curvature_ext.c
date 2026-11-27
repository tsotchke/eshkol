/**
 * @file vm_curvature_ext.c
 * @brief Bytecode VM native dispatch for core.curvature (ADR-0018) and
 *        Wave 1 runtime reflection builtins (fids 2350-2399).
 *
 * Calls the shared header-only f64 reference in
 * inc/eshkol/backend/riemannian_core.h directly on VmTensor storage with
 * zero malloc and zero mutable global state (last-error state lives on VM).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef VM_CURVATURE_EXT_C_INCLUDED
#define VM_CURVATURE_EXT_C_INCLUDED

#include <math.h>
#include <stdint.h>
#include <string.h>

static int64_t vm_curv_fail(VM* vm, const char* why) {
    if (vm) vm->curvature_last_error = why ? why : "refused";
    return 1;
}

static int64_t vm_curv_ok(VM* vm) {
    if (vm) vm->curvature_last_error = "";
    return 0;
}

static VmTensor* vm_curv_as_tensor(VM* vm, Value v) {
    if (!vm || v.type != VAL_TENSOR || !is_valid_heap_ptr(vm, v.as.ptr)) return NULL;
    HeapObject* obj = vm->heap.objects[v.as.ptr];
    if (!obj || obj->type != HEAP_TENSOR) return NULL;
    return (VmTensor*)obj->opaque.ptr;
}

static const char* vm_curv_read_vec(VM* vm, Value v, int64_t want,
                                    const double** data, int* n) {
    VmTensor* t = vm_curv_as_tensor(vm, v);
    if (t) {
        if (t->total <= 0 || !t->data) return "expected a non-empty tensor";
        if (t->total > (int64_t)INT32_MAX) return "the tensor is too large";
        if (want >= 0 && t->total != want) return "tensor sizes do not match";
        *data = t->data;
        *n = (int)t->total;
        return NULL;
    }
    if (v.type == VAL_VECTOR && is_valid_heap_ptr(vm, v.as.ptr) &&
        vm->heap.objects[v.as.ptr] &&
        vm->heap.objects[v.as.ptr]->type == HEAP_VECTOR) {
        VmVector* vec = (VmVector*)vm->heap.objects[v.as.ptr]->opaque.ptr;
        if (!vec || vec->len <= 0) return "expected a non-empty tensor";
        if (want >= 0 && (int64_t)vec->len != want) return "tensor sizes do not match";
        double* buf = (double*)vm_alloc(&vm->heap.regions, (size_t)vec->len * sizeof(double));
        if (!buf) return "out of memory";
        for (int i = 0; i < vec->len; ++i) {
            Value item = vec->items[i];
            if (item.type == VAL_FLOAT) buf[i] = item.as.f;
            else if (item.type == VAL_INT) buf[i] = (double)item.as.i;
            else return "expected an f64 tensor";
        }
        *data = buf;
        *n = vec->len;
        return NULL;
    }
    return "expected an f64 tensor";
}

static const char* vm_curv_read_pair(VM* vm, Value x, Value y,
                                     const double** xd, const double** yd, int* n) {
    const char* why = vm_curv_read_vec(vm, x, -1, xd, n);
    if (why) return why;
    int m = 0;
    return vm_curv_read_vec(vm, y, *n, yd, &m);
}

static const char* vm_curv_out_vec(VM* vm, Value v, int64_t want, double** data) {
    VmTensor* t = vm_curv_as_tensor(vm, v);
    if (!t || !t->data) return "the output must be an f64 tensor";
    if (t->total != want) return "the output tensor size does not match";
    *data = t->data;
    return NULL;
}

static const char* vm_curv_matrix_dims(VM* vm, Value v, int64_t* rows, int64_t* cols) {
    VmTensor* t = vm_curv_as_tensor(vm, v);
    if (!t || !t->shape) return "expected a rank-1 or rank-2 f64 tensor";
    if (t->n_dims == 1) {
        *rows = 1;
        *cols = t->shape[0];
    } else if (t->n_dims == 2) {
        *rows = t->shape[0];
        *cols = t->shape[1];
    } else {
        return "expected a rank-1 or rank-2 f64 tensor";
    }
    if (*rows <= 0 || *cols <= 0) return "expected a non-empty tensor";
    return NULL;
}

static double* vm_curv_scratch(VM* vm, int n, int mult) {
    size_t count = (size_t)n * (size_t)mult + 2u;
    double* buf = (double*)vm_alloc(&vm->heap.regions, count * sizeof(double));
    if (buf) memset(buf, 0, count * sizeof(double));
    return buf;
}

static Value vm_curv_make_string(VM* vm, const char* text, int is_symbol) {
    const char* s = text ? text : "";
    size_t len = strlen(s);
    VmString* str = vm_string_new(&vm->heap.regions, s, (int)len);
    int32_t ptr = str ? heap_alloc(&vm->heap) : -1;
    if (!str || ptr < 0) {
        vm->error = 1;
        return NIL_VAL;
    }
    vm->heap.objects[ptr]->type = HEAP_STRING;
    vm->heap.objects[ptr]->opaque.ptr = str;
    return (Value){.type = is_symbol ? VAL_SYMBOL : VAL_STRING, .as.ptr = ptr};
}

static void vm_dispatch_curvature_ext(VM* vm, int fid) {
    switch (fid) {
    case 2350: { /* chart-last-error-raw () -> ptr (VAL_STRING on VM) */
        const char* msg = vm->curvature_last_error ? vm->curvature_last_error : "";
        vm_push(vm, vm_curv_make_string(vm, msg, 0));
        break;
    }
    case 2351: { /* chart-convention-id-raw () -> i64 */
        vm_push(vm, INT_VAL((int64_t)ESHKOL_GEOMETRY_CONVENTION_STEREOGRAPHIC_V2_ALPHA1));
        break;
    }
    case 2352: { /* chart-alpha-raw () -> double */
        vm_push(vm, FLOAT_VAL(ESHKOL_RM_ALPHA));
        break;
    }
    case 2353: { /* chart-check-header-raw (convention, alpha, K) -> i32 */
        Value kv = vm_pop(vm), av = vm_pop(vm), cv = vm_pop(vm);
        int64_t conv = (cv.type == VAL_INT) ? cv.as.i : (int64_t)as_number(cv);
        double alpha = as_number(av);
        double K = as_number(kv);
        if (conv < 0 || conv > (int64_t)UINT32_MAX) {
            vm_push(vm, INT_VAL(vm_curv_fail(vm, "the geometry record names an unknown convention id")));
            break;
        }
        eshkol_geometry_header_t h;
        h.convention = (uint32_t)conv;
        h.reserved = 0u;
        h.alpha = alpha;
        h.curvature = K;
        const char* why = eshkol_rm_check_geometry_header(&h);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2354: { /* chart-distance-raw (x, y, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), y = vm_pop(vm), x = vm_pop(vm);
        const double *xd, *yd; double* o; int n;
        const char* why = vm_curv_read_pair(vm, x, y, &xd, &yd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, 1, &o);
        if (!why) why = eshkol_rm_distance(xd, yd, as_number(kv), n, &o[0]);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2355: { /* chart-sq-distance-raw (x, y, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), y = vm_pop(vm), x = vm_pop(vm);
        const double *xd, *yd; double* o; int n;
        const char* why = vm_curv_read_pair(vm, x, y, &xd, &yd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, 1, &o);
        if (!why) why = eshkol_rm_sq_distance(xd, yd, as_number(kv), n, &o[0]);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2356: { /* chart-conformal-factor-raw (x, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), x = vm_pop(vm);
        const double* xd; double* o; int n;
        double K = as_number(kv);
        const char* why = vm_curv_read_vec(vm, x, -1, &xd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, 1, &o);
        if (!why) why = eshkol_rm_check_point(xd, K, n);
        if (!why) {
            o[0] = eshkol_rm_lambda(xd, K, n);
            if (!isfinite(o[0])) why = "the conformal factor is not finite";
        }
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2357: { /* chart-distance-jet-raw (x, y, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), y = vm_pop(vm), x = vm_pop(vm);
        const double *xd, *yd; double* o; int n;
        const char* why = vm_curv_read_pair(vm, x, y, &xd, &yd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, 3, &o);
        if (!why) why = eshkol_rm_distance_dK(xd, yd, as_number(kv), n, &o[0], &o[1], &o[2]);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2358: { /* chart-sq-distance-jet-raw (x, y, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), y = vm_pop(vm), x = vm_pop(vm);
        const double *xd, *yd; double* o; int n;
        const char* why = vm_curv_read_pair(vm, x, y, &xd, &yd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, 3, &o);
        if (!why) why = eshkol_rm_sq_distance_dK(xd, yd, as_number(kv), n, &o[0], &o[1], &o[2]);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2359: { /* chart-sq-distance-grad-raw (x, y, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), y = vm_pop(vm), x = vm_pop(vm);
        const double *xd, *yd; double* o; int n;
        const char* why = vm_curv_read_pair(vm, x, y, &xd, &yd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, n, &o);
        if (!why) {
            double* scratch = vm_curv_scratch(vm, n, 2);
            if (!scratch) why = "out of memory";
            else why = eshkol_rm_sq_distance_grad(xd, yd, as_number(kv), n, o, NULL, scratch);
        }
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2360: { /* chart-exp-map-raw (x, v, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), v = vm_pop(vm), x = vm_pop(vm);
        const double *xd, *vd; double* o; int n;
        const char* why = vm_curv_read_pair(vm, x, v, &xd, &vd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, n, &o);
        if (!why) {
            double* scratch = vm_curv_scratch(vm, n, 1);
            if (!scratch) why = "out of memory";
            else why = eshkol_rm_exp_map(xd, vd, as_number(kv), n, o, scratch);
        }
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2361: { /* chart-log-map-raw (x, y, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), y = vm_pop(vm), x = vm_pop(vm);
        const double *xd, *yd; double* o; int n;
        const char* why = vm_curv_read_pair(vm, x, y, &xd, &yd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, n, &o);
        if (!why) {
            double* scratch = vm_curv_scratch(vm, n, 1);
            if (!scratch) why = "out of memory";
            else why = eshkol_rm_log_map(xd, yd, as_number(kv), n, o, scratch);
        }
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2362: { /* chart-transport-raw (x, y, v, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), v = vm_pop(vm), y = vm_pop(vm), x = vm_pop(vm);
        const double *xd, *yd, *vd; double* o; int n, m = 0;
        const char* why = vm_curv_read_pair(vm, x, y, &xd, &yd, &n);
        if (!why) why = vm_curv_read_vec(vm, v, n, &vd, &m);
        if (!why) why = vm_curv_out_vec(vm, out, n, &o);
        if (!why) {
            double* scratch = vm_curv_scratch(vm, n, 1);
            if (!scratch) why = "out of memory";
            else why = eshkol_rm_transport(xd, yd, vd, as_number(kv), n, o, scratch);
        }
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2363: { /* chart-mobius-add-raw (x, y, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), y = vm_pop(vm), x = vm_pop(vm);
        const double *xd, *yd; double* o; int n;
        const char* why = vm_curv_read_pair(vm, x, y, &xd, &yd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, n, &o);
        if (!why) why = eshkol_rm_chart_mobius_add(xd, yd, as_number(kv), n, o);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2364: { /* chart-mobius-scalar-raw (r, x, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), x = vm_pop(vm), rv = vm_pop(vm);
        const double* xd; double* o; int n;
        const char* why = vm_curv_read_vec(vm, x, -1, &xd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, n, &o);
        if (!why) why = eshkol_rm_mobius_scalar(as_number(rv), xd, as_number(kv), n, o);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2365: { /* chart-project-raw (x, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), x = vm_pop(vm);
        const double* xd; double* o; int n;
        const char* why = vm_curv_read_vec(vm, x, -1, &xd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, n, &o);
        if (!why) why = eshkol_rm_project(xd, as_number(kv), n, o);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2366: { /* chart-riemannian-grad-raw (g, x, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), x = vm_pop(vm), g = vm_pop(vm);
        const double *gd, *xd; double* o; int n;
        const char* why = vm_curv_read_pair(vm, g, x, &gd, &xd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, n, &o);
        if (!why) why = eshkol_rm_egrad_to_rgrad(gd, xd, as_number(kv), n, o);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2367: { /* chart-attention-raw (q, keys, v, K, temp, out, weights) -> i32 */
        Value weights = vm_pop(vm), out = vm_pop(vm), tempv = vm_pop(vm),
              kv = vm_pop(vm), v = vm_pop(vm), keys = vm_pop(vm), q = vm_pop(vm);
        int64_t nq = 0, d = 0, nk = 0, dk = 0, nv = 0, dv = 0;
        const double *qd = NULL, *kd = NULL, *vd = NULL;
        double *od = NULL, *wd = NULL;
        int tq = 0, tk = 0, tv = 0;
        const char* why = vm_curv_matrix_dims(vm, q, &nq, &d);
        if (!why) why = vm_curv_matrix_dims(vm, keys, &nk, &dk);
        if (!why) why = vm_curv_matrix_dims(vm, v, &nv, &dv);
        if (!why && dk != d) why = "queries and keys must have the same dimension";
        if (!why && nv != nk) why = "there must be one value row per key";
        if (!why) why = vm_curv_read_vec(vm, q, nq * d, &qd, &tq);
        if (!why) why = vm_curv_read_vec(vm, keys, nk * d, &kd, &tk);
        if (!why) why = vm_curv_read_vec(vm, v, nk * dv, &vd, &tv);
        if (!why) why = vm_curv_out_vec(vm, out, nq * dv, &od);
        if (!why) why = vm_curv_out_vec(vm, weights, nq * nk, &wd);
        if (!why && (nk > INT32_MAX || d > INT32_MAX)) why = "the attention is too large";
        if (!why) {
            double* sqd = vm_curv_scratch(vm, (int)nk, 1);
            if (!sqd) {
                why = "out of memory";
            } else {
                double K = as_number(kv);
                double temp = as_number(tempv);
                for (int64_t i = 0; i < nq; ++i) {
                    double* row = wd + i * nk;
                    why = eshkol_rm_attention_row(qd + i * d, kd, (long)d,
                                                  (int)nk, (int)d, K, temp, sqd, row);
                    if (why) break;
                    for (int64_t c = 0; c < dv; ++c) {
                        double acc = 0.0;
                        for (int64_t j = 0; j < nk; ++j) acc += row[j] * vd[j * dv + c];
                        od[i * dv + c] = acc;
                    }
                }
                if (!why) {
                    why = eshkol_rm_check_output(od, (int)(nq * dv),
                                                 "the attention output is not finite");
                }
            }
        }
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2368: { /* chart-sphere-to-chart-raw (p, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), p = vm_pop(vm);
        const double* pd; double* o; int n;
        const char* why = vm_curv_read_vec(vm, p, -1, &pd, &n);
        if (!why && n < 2) why = "an embedded sphere point needs at least two coordinates";
        if (!why) why = vm_curv_out_vec(vm, out, n - 1, &o);
        if (!why) why = eshkol_rm_sphere_to_chart(pd, as_number(kv), n - 1, o);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2369: { /* chart-chart-to-sphere-raw (x, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), x = vm_pop(vm);
        const double* xd; double* o; int n;
        const char* why = vm_curv_read_vec(vm, x, -1, &xd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, (int64_t)n + 1, &o);
        if (!why) why = eshkol_rm_chart_to_sphere(xd, as_number(kv), n, o);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2370: { /* chart-migrate-point-raw (x, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), x = vm_pop(vm);
        const double* xd; double* o; int n;
        const char* why = vm_curv_read_vec(vm, x, -1, &xd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, n, &o);
        if (!why) why = eshkol_rm_migrate_legacy_point(xd, as_number(kv), n, o);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2371: { /* chart-migrate-tangent-raw (v, K, out) -> i32 */
        Value out = vm_pop(vm), kv = vm_pop(vm), v = vm_pop(vm);
        const double* vd; double* o; int n;
        const char* why = vm_curv_read_vec(vm, v, -1, &vd, &n);
        if (!why) why = vm_curv_out_vec(vm, out, n, &o);
        if (!why) why = eshkol_rm_migrate_legacy_tangent(vd, as_number(kv), n, o);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2372: { /* chart-migrate-temperature-raw (scale, ref_dist, out) -> i32 */
        Value out = vm_pop(vm), refv = vm_pop(vm), scalev = vm_pop(vm);
        double* o;
        const char* why = vm_curv_out_vec(vm, out, 1, &o);
        if (!why) why = eshkol_rm_migrate_legacy_temperature(as_number(scalev), as_number(refv), &o[0]);
        vm_push(vm, INT_VAL(why ? vm_curv_fail(vm, why) : vm_curv_ok(vm)));
        break;
    }
    case 2373: { /* VM-internal ptr->string passthrough for typed extern error strings */
        Value val = vm_pop(vm);
        if (val.type == VAL_STRING) {
            vm_push(vm, val);
        } else {
            vm_push(vm, vm_curv_make_string(vm, "", 0));
        }
        break;
    }
    case 2380: { /* procedure-arity (proc) -> i64 */
        Value proc = vm_pop(vm);
        int64_t arity = -1;
        if (proc.type == VAL_CLOSURE && is_valid_heap_ptr(vm, proc.as.ptr)) {
            HeapObject* obj = vm->heap.objects[proc.as.ptr];
            if (obj && obj->type == HEAP_CLOSURE) {
                arity = (int64_t)obj->closure.arity;
            }
        }
        vm_push(vm, INT_VAL(arity));
        break;
    }
    case 2381: { /* features () -> list of feature symbols */
        static const char* const k_features[] = {
            "r7rs", "eshkol", "rationals", "exact-complex",
            "ieee-float", "full-unicode", "posix",
#ifdef __APPLE__
            "macosx", "darwin",
#elif defined(__linux__)
            "linux",
#endif
#if defined(__x86_64__) || defined(_M_X64)
            "x86-64",
#elif defined(__aarch64__) || defined(_M_ARM64)
            "arm64",
#endif
            "big-endian",
            "eshkol-1.1"
        };
        int count = (int)(sizeof(k_features) / sizeof(k_features[0]));
        Value list = NIL_VAL;
        for (int i = count - 1; i >= 0; --i) {
            Value sym = vm_curv_make_string(vm, k_features[i], 1);
            int32_t cell = heap_alloc(&vm->heap);
            if (cell < 0) {
                vm->error = 1;
                vm_push(vm, NIL_VAL);
                return;
            }
            vm->heap.objects[cell]->type = HEAP_CONS;
            vm->heap.objects[cell]->cons.car = sym;
            vm->heap.objects[cell]->cons.cdr = list;
            list = PAIR_VAL(cell);
        }
        vm_push(vm, list);
        break;
    }
    default:
        fprintf(stderr, "WARNING: unhandled curvature/reflection native call ID %d\n", fid);
        vm_push(vm, NIL_VAL);
        break;
    }
}

#endif /* VM_CURVATURE_EXT_C_INCLUDED */
