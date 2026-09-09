/**
 * @file vm_geometric.c
 * @brief VM geometric manifold dispatch — native IDs 804-861.
 *
 * ONE dispatch body: the closed forms of constant-curvature geometry in f64
 * (inc/eshkol/backend/riemannian_core.h) plus the differential-form jet calculus
 * (inc/eshkol/backend/differential_form_core.h), with manifolds allocated in the
 * VM arena. Handles are logically invalidated by manifold-destroy!; arena memory
 * remains owned by the VM region stack.
 *
 * This file used to carry a SECOND body for the same 62 names, selected by a
 * legacy qLLM build branch. It
 * has been deleted rather than repaired: measured against a current
 * libsemiclassical_qllm it produced 19 compile errors (moved arities on
 * qllm_hyperbolic_exp_map / _log_map / _distance / _parallel_transport /
 * _mobius_add / _mobius_scalar, qllm_hyperbolic_project taking a
 * qllm_tensor_t* where the call passed a float*, and SO(3)/SE(3) arms naming
 * types the library no longer declares), it was fp32 throughout -- which
 * frechet_mean_core.h documents as unable to satisfy the exact-derivative gate
 * -- and its manifold-dim returned 0 unconditionally. An unreachable body that
 * does not compile is not a second implementation; it is a second set of claims
 * no gate can check. The differentiable route to this geometry is
 * lib/bridge/qllm_bridge.cpp, which is a live, tested integration and is
 * untouched by that deletion.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#define HEAP_MANIFOLD        30
#define HEAP_MANIFOLD_POINT  31
#define HEAP_MANIFOLD_TANGENT 32

/** @brief Unwrap a tensor Value to its VmTensor*, or NULL if @p v isn't a
 *         tensor heap object. */
static VmTensor* vm_get_tensor(VM* vm, Value v) {
    if (!is_heap_type(vm, v, HEAP_TENSOR)) return NULL;
    return (VmTensor*)vm->heap.objects[v.as.ptr]->opaque.ptr;
}

/** @brief Push a scalar float result onto the VM stack. */
static void vm_push_float(VM* vm, double val) {
    vm_push(vm, FLOAT_VAL(val));
}

/** @brief Wrap an opaque manifold pointer as a heap-boxed VAL_MANIFOLD
 *         Value and push it (NIL if @p manifold is null or allocation
 *         fails). */
static void vm_push_manifold(VM* vm, void* manifold) {
    if (!manifold) { vm_push(vm, NIL_VAL); return; }
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) { vm_push(vm, NIL_VAL); return; }
    vm->heap.objects[ptr]->type = HEAP_MANIFOLD;
    vm->heap.objects[ptr]->opaque.ptr = manifold;
    vm_push(vm, (Value){.type = VAL_MANIFOLD, .as.ptr = ptr});
}

static VmTensor* vm_tensor_linear_combo_for_geometry(VM* vm, const VmTensor* a, double as,
                                                     const VmTensor* b, double bs);

/* Catchable error, defined in vm_native.c (included after this file by
 * eshkol_vm.c). Declared here for the same reason vm_native.c declares it ahead
 * of its own definition: a geometric domain violation must be a Scheme
 * condition the caller can `guard` on, not a process exit. */
static void vm_raise_error_msg(VM* vm, const char* msg);

/* The weighted Fréchet (Karcher) mean forward pass lives in
 * inc/eshkol/backend/frechet_mean_core.h so that this VM opcode and the AD
 * bridge producer (ad_frechet_mean, lib/bridge/qllm_bridge.cpp) compute the
 * mean with ONE implementation. The backward rule is gated on the
 * stationarity residual, so a second copy of the iteration that drifted from
 * this one would hand the derivative means it refuses — or, worse, means it
 * accepts that differ from the ones this opcode returns. See that header for
 * the derivation, the convergence criterion and why the gate is measured in
 * Riemannian units. */
#include "eshkol/backend/frechet_mean_core.h"

/* Closed-form constant-curvature geometry (Poincare ball / Euclidean / sphere)
 * in f64. Same reason as the header above for being a header: this file is a
 * unity-build include and `eshkol-vm-standalone-test` compiles it as one TU, so
 * there is no object file for it to call into. See that header for the K-sign
 * convention and for what these ops used to return. */
#include "eshkol/backend/riemannian_core.h"

/* Differential forms as jets at a point: the exterior derivative (835) and the
 * Hodge star (836). Same header rationale again. See that file for the
 * representation -- degree, dimension, jet order and the coefficient jets in
 * one tensor -- and for why a form's VALUES at a point cannot determine d. */
#include "eshkol/backend/differential_form_core.h"

/* Scratch for the closed forms below. The widest requirement is
 * eshkol_rm_transport's 5n doubles; the +2 keeps the tiny-n case from
 * degenerating to a zero-byte request. */
#define VM_GEOMETRIC_SCRATCH_MULT 5

/** @brief Allocate @p mult * @p n doubles of region-scoped scratch for the
 *         closed-form geometry routines. */
static double* vm_geometric_scratch(VM* vm, int n, int mult) {
    if (n <= 0) return NULL;
    return (double*)vm_alloc(&vm->heap.regions,
                             (size_t)((int64_t)n * mult + 2) * sizeof(double));
}

/** @brief Raise a catchable condition naming the builtin and the reason a
 *         constant-curvature op refused. Like every other raise in the VM this
 *         restores the handler's stack pointer, so the caller must `break`
 *         WITHOUT pushing a result. */
static void vm_geometric_raise(VM* vm, const char* op, const char* why, double K) {
    char msg[320];
    snprintf(msg, sizeof msg,
             "%s: %s (sectional curvature K = %.17g; K < 0 is the Poincare ball "
             "of radius 1/sqrt(-K), K = 0 is Euclidean, K > 0 is the sphere of "
             "radius 1/sqrt(K))", op, why, K);
    vm_raise_error_msg(vm, msg);
}

/** @brief Raise a catchable condition naming a differential-form op and the
 *         reason it refused. Unlike vm_geometric_raise there is no curvature to
 *         report: the exterior derivative is a property of the jet alone, and
 *         the Hodge star's curvature dependence enters through the metric it is
 *         handed. */
static void vm_form_raise(VM* vm, const char* op, const char* why) {
    char msg[512];
    snprintf(msg, sizeof msg,
             "%s: %s (a differential form is a tensor "
             "[k, n, r, coefficient jets...]: degree k, dimension n, jet order "
             "r, then C(n,k) blocks of 1 + n + ... + n^r doubles in the "
             "lexicographic increasing-multi-index basis -- see "
             "docs/reference/stdlib/geometry.md)", op, why);
    vm_raise_error_msg(vm, msg);
}

/**
 * @brief Shared implementation of native id 817 for both the portable and the
 *        linked-library builds: pops (points, weights, curvature), pushes the
 *        f64 Fréchet mean, or raises a catchable error.
 *
 * The residual gate is the point of the error path. A mean that has not reached
 * stationarity is exactly the input for which the implicit derivative returns a
 * plausible wrong number, so a silent near-answer here would be laundered into a
 * wrong gradient downstream.
 */
static void vm_dispatch_frechet_mean(VM* vm) {
    double K = as_number(vm_pop(vm));
    VmTensor* wv = vm_get_tensor(vm, vm_pop(vm));
    VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));

    if (!pv || !pv->data) {
        vm_raise_error_msg(vm, "frechet-mean: first argument must be a tensor of points");
        return;
    }
    int dim      = (pv->n_dims >= 2) ? (int)pv->shape[1] : (int)pv->total;
    int n_points = (pv->n_dims >= 2) ? (int)pv->shape[0] : 1;

    int64_t shape[1] = { dim };
    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
    double* scratch = (double*)vm_alloc(&vm->heap.regions, (size_t)(4 * dim) * sizeof(double));
    if (!out || !scratch) {
        vm_raise_error_msg(vm, "frechet-mean: out of memory");
        return;
    }

    double resid = 0.0;
    const char* why = eshkol_frechet_mean_compute(
        pv->data, wv ? wv->data : NULL, wv ? wv->total : 0,
        n_points, dim, K, out->data, scratch, &resid);

    if (why) {
        char msg[320];
        snprintf(msg, sizeof msg,
                 "frechet-mean: %s (points=%d, dim=%d, curvature=%.17g, "
                 "relative stationarity residual %.3e vs tolerance %.1e). The "
                 "Frechet mean is the solution of sum_i w_i log_mu(x_i) = 0; a "
                 "non-stationary iterate is not an approximate answer, because "
                 "its implicit derivative is a plausible wrong gradient.",
                 why, n_points, dim, K, resid, ESHKOL_FRECHET_RESID_TOL);
        vm_raise_error_msg(vm, msg);
        return;
    }
    VM_PUSH_TENSOR(vm, out);
}

typedef struct {
    int type;          /* 0 euclidean, 1 hyperbolic, 2 spherical, 3 product */
    int dim;
    double curvature;
} VmGeometricManifold;

/** @brief Unwrap a manifold Value to its constant-curvature
 *         VmGeometricManifold*, or NULL if @p mv isn't a manifold heap
 *         object. */
static VmGeometricManifold* vm_geometric_manifold(VM* vm, Value mv) {
    if (mv.type != VAL_MANIFOLD || !is_heap_type(vm, mv, HEAP_MANIFOLD)) return NULL;
    return (VmGeometricManifold*)vm->heap.objects[mv.as.ptr]->opaque.ptr;
}

/** @brief Allocate and push a constant-curvature manifold
 *         (type/dim/curvature); pushes NIL on invalid @p dim or allocation
 *         failure. */
static void vm_push_geometric_manifold(VM* vm, int type, int dim, double curvature) {
    if (dim <= 0) { vm_push(vm, NIL_VAL); return; }
    VmGeometricManifold* m = (VmGeometricManifold*)vm_alloc(&vm->heap.regions, sizeof(VmGeometricManifold));
    if (!m) { vm_push(vm, NIL_VAL); return; }
    m->type = type;
    m->dim = dim;
    m->curvature = curvature;
    vm_push_manifold(vm, m);
}

/**
 * @brief Riemannian-Adam optimizer state: the first moment as a TANGENT VECTOR
 *        at the current point, the second moment as ONE SCALAR, and the step
 *        counter driving bias correction.
 *
 * The second moment used to be a per-coordinate array, and the delta a
 * per-coordinate quotient. That is not an intrinsic operation on a manifold and
 * it does not preserve a tangent space: on the unit sphere at
 * x = (1,1,1)/sqrt(3) with the perfectly valid tangent gradient g = (1,1,-2),
 * dividing coordinate by coordinate gives a delta proportional to (1,1,-1),
 * whose inner product with x is -eta/sqrt(3) != 0 -- so the exponential map
 * refused a valid optimizer input. The intrinsic construction (Becigneul-Ganea,
 * and geoopt's RiemannianAdam, which the docs previously mis-cited as doing the
 * coordinate-wise thing) keeps ONE adaptivity scalar per manifold factor,
 *
 *     q_t = hypot(sqrt(beta2) q_{t-1}, sqrt(1 - beta2) ||g_t||_{x_t}),
 *
 * measured in the manifold's own metric, so the delta stays PARALLEL to the
 * first moment and therefore tangent.
 */
typedef struct {
    int      n_dims;
    int64_t  shape[VM_TENSOR_MAX_DIMS];
    int64_t  total;
    int64_t  step;
    const VmTensor* owner; /* current parameter identity; changes after success */
    double*  m;    /* first moment: a tangent vector at the current point */
    double   rms;  /* RMS second moment: one scalar for this manifold factor */
} VmRiemannianAdamState;

/** @brief Allocate @p size bytes either from the VM-lifetime global arena
 *         (@p vm_lifetime true) or the current region-scoped arena
 *         (@p vm_lifetime false). Explicit optimizer states use the latter. */
static void* vm_geometric_alloc(VM* vm, size_t size, int vm_lifetime) {
    if (!vm) return NULL;
    if (vm_lifetime) return vm_arena_alloc(&vm->heap.regions.global_arena, size);
    return vm_alloc(&vm->heap.regions, size);
}

/** @brief Allocate a zero-initialized Riemannian-Adam optimizer state
 *         (first/second moment buffers @c m/@c v, shaped like @p ref),
 *         with the given lifetime (see vm_geometric_alloc()). */
static VmRiemannianAdamState* vm_riemannian_adam_state_new_with_lifetime(
    VM* vm, const VmTensor* ref, int vm_lifetime) {
    if (!vm || !ref || ref->total <= 0 || ref->n_dims <= 0 || ref->n_dims > VM_TENSOR_MAX_DIMS)
        return NULL;

    VmRiemannianAdamState* st = (VmRiemannianAdamState*)vm_geometric_alloc(
        vm, sizeof(VmRiemannianAdamState), vm_lifetime);
    if (!st) return NULL;
    memset(st, 0, sizeof(VmRiemannianAdamState));
    st->n_dims = ref->n_dims;
    st->total = ref->total;
    st->owner = ref;
    memcpy(st->shape, ref->shape, (size_t)ref->n_dims * sizeof(int64_t));
    st->m = (double*)vm_geometric_alloc(vm, (size_t)ref->total * sizeof(double), vm_lifetime);
    if (!st->m) return NULL;
    memset(st->m, 0, (size_t)ref->total * sizeof(double));
    st->rms = 0.0;
    return st;
}

/** @brief Allocate a region-scoped (not VM-lifetime) Riemannian-Adam
 *         optimizer state shaped like @p ref. */
static VmRiemannianAdamState* vm_riemannian_adam_state_new(VM* vm, const VmTensor* ref) {
    return vm_riemannian_adam_state_new_with_lifetime(vm, ref, 0);
}

/** @brief Whether optimizer state @p st belongs to tensor @p ref.
 *
 * Shape equality is necessary for the moment buffer, but it is not identity:
 * two independent parameters routinely have the same shape. The state belongs
 * to the current parameter object and is retargeted only after a successful
 * step returns that step's new tensor.
 */
static int vm_riemannian_adam_state_matches(const VmRiemannianAdamState* st,
                                            const VmTensor* ref) {
    if (!st || !ref || st->owner != ref || st->total != ref->total ||
        st->n_dims != ref->n_dims) return 0;
    for (int i = 0; i < ref->n_dims; i++)
        if (st->shape[i] != ref->shape[i]) return 0;
    return 1;
}

/** @brief Unwrap a Value to its VmRiemannianAdamState*, or NULL if @p v
 *         isn't an optimizer-state heap object. */
static VmRiemannianAdamState* vm_riemannian_adam_state_from_value(VM* vm, Value v) {
    if (v.type != VAL_RIEMANNIAN_ADAM_STATE ||
        !is_heap_type(vm, v, HEAP_RIEMANNIAN_ADAM_STATE))
        return NULL;
    return (VmRiemannianAdamState*)vm->heap.objects[v.as.ptr]->opaque.ptr;
}

/** @brief Wrap an optimizer state pointer as a heap-boxed
 *         VAL_RIEMANNIAN_ADAM_STATE Value and push it (NIL on null/
 *         allocation failure). */
static void vm_push_riemannian_adam_state(VM* vm, VmRiemannianAdamState* st) {
    if (!st) { vm_push(vm, NIL_VAL); return; }
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) { vm_push(vm, NIL_VAL); return; }
    vm->heap.objects[ptr]->type = HEAP_RIEMANNIAN_ADAM_STATE;
    vm->heap.objects[ptr]->opaque.ptr = st;
    vm_push(vm, (Value){.type = VAL_RIEMANNIAN_ADAM_STATE, .as.ptr = ptr});
}

/**
 * @brief Compute one INTRINSIC Riemannian-Adam delta from @p grad and the
 *        optimizer state @p st, WITHOUT MUTATING @p st.
 *
 * The proposed moments and step counter go to @p m_next, @p rms_next and
 * @p step_next; the caller commits them only once the exponential map and the
 * moment transport have both succeeded. They used to be written straight into
 * the state before either could refuse, so a call that ended in a raise still
 * advanced the step counter and both moments -- and a retry with corrected
 * arguments then produced a different answer from the same inputs.
 *
 * The second moment is the SCALAR
 * s_t = beta2 s_{t-1} + (1 - beta2) ||g||^2_x, measured in the manifold's
 * metric, so the delta is parallel to the first moment and therefore tangent.
 * See VmRiemannianAdamState for why the coordinate-wise form it replaces is not
 * an operation on a manifold at all.
 *
 * @param m_next    @p grad->total doubles, the proposed first moment.
 * @param rms_next  the proposed RMS second moment.
 * @param step_next the proposed step counter.
 * @return the delta tensor, or NULL on a shape/allocation failure.
 */
static VmTensor* vm_riemannian_adam_delta(VM* vm, const VmTensor* point,
                                          const VmTensor* grad,
                                          const VmRiemannianAdamState* st,
                                          double lr, double beta1, double beta2,
                                          double K, double* m_next,
                                          double* rms_next, int64_t* step_next,
                                          const char** why_out) {
    if (why_out) *why_out = NULL;
    if (!vm || !point || !grad || !st || !grad->data || grad->total != st->total)
        return NULL;
    if (beta1 < 0.0 || beta1 >= 1.0) beta1 = 0.9;
    if (beta2 < 0.0 || beta2 >= 1.0) beta2 = 0.999;
    if (lr < 0.0) lr = -lr;

    VmTensor* delta = vm_tensor_zeros(&vm->heap.regions, grad->shape, grad->n_dims);
    if (!delta) return NULL;

    int64_t step = st->step + 1;
    double b1_corr = 1.0 - pow(beta1, (double)step);
    double b2_corr = 1.0 - pow(beta2, (double)step);
    if (b1_corr <= 0.0) b1_corr = 1.0;
    if (b2_corr <= 0.0) b2_corr = 1.0;

    double grad_rms = eshkol_rm_metric_norm(grad->data, point->data, K,
                                            (int)grad->total);
    if (!isfinite(grad_rms)) {
        if (why_out) *why_out = "the Adam gradient norm is not finite";
        return NULL;
    }
    double rms = hypot(sqrt(beta2) * st->rms,
                       sqrt(1.0 - beta2) * grad_rms);
    if (!isfinite(rms)) {
        if (why_out) *why_out = "the Adam RMS second moment is not finite";
        return NULL;
    }
    double rms_hat = rms / sqrt(b2_corr);
    double scale = -lr / (rms_hat + 1e-8);

    for (int64_t i = 0; i < grad->total; i++) {
        m_next[i] = beta1 * st->m[i] + (1.0 - beta1) * grad->data[i];
        delta->data[i] = scale * (m_next[i] / b1_corr);
        if (!isfinite(m_next[i]) || !isfinite(delta->data[i])) {
            if (why_out) *why_out = "the Adam update is not finite";
            return NULL;
        }
    }
    *rms_next = rms;
    *step_next = step;
    return delta;
}

/**
 * @brief Apply one Riemannian-Adam step on the manifold of curvature @p K:
 *        form the Adam delta in the tangent space at @p point, RETRACT along
 *        the geodesic with the exponential map, and parallel-transport the
 *        first-moment buffer to the new point.
 *
 * This used to be `point + delta`, an ambient vector addition. On the ball that
 * is not a point of the manifold for a large enough step -- the iterate simply
 * leaves the space -- and it discarded the curvature argument the op accepts,
 * so an optimizer named Riemannian ran plain Adam.
 *
 * The first moment is a tangent vector at the OLD point and is meaningless at
 * the new one until transported; this is the transport geoopt's RiemannianAdam
 * performs. The second moment is one scalar for this manifold factor, not a
 * tangent vector, and is left as is.
 *
 * @return NULL on allocation/shape failure, or when @p why_out is set (in which
 *         case the caller raises). @p why_out is set only on a geometry
 *         refusal.
 */
static VmTensor* vm_riemannian_adam_geodesic_step(VM* vm, const VmTensor* point,
                                                  const VmTensor* grad,
                                                  VmRiemannianAdamState* st,
                                                  double lr, double beta1,
                                                  double beta2, double K,
                                                  const char** why_out) {
    if (why_out) *why_out = NULL;
    if (!point || !grad || !point->data || point->total != grad->total ||
        point->total <= 0 || !vm_riemannian_adam_state_matches(st, point))
        return NULL;

    int n = (int)point->total;
    const char* why = eshkol_rm_check_point(point->data, K, n);
    if (!why) why = eshkol_rm_check_tangent(point->data, grad->data, K, n);
    if (why) { if (why_out) *why_out = why; return NULL; }

    /* Proposed state, held in scratch until BOTH the retraction and the moment
     * transport have succeeded. Nothing below writes through `st` before then:
     * a call that ends in a refusal must leave the optimizer exactly as it was,
     * or the retry that follows it answers a different question. */
    double* m_next = vm_geometric_scratch(vm, n, 1);
    double* moved  = vm_geometric_scratch(vm, n, 1);
    double* scratch = vm_geometric_scratch(vm, n, VM_GEOMETRIC_SCRATCH_MULT);
    if (!m_next || !moved || !scratch) return NULL;

    double rms_next = 0.0;
    int64_t step_next = 0;
    const char* delta_why = NULL;
    VmTensor* delta = vm_riemannian_adam_delta(vm, point, grad, st, lr, beta1,
                                               beta2, K, m_next, &rms_next,
                                               &step_next, &delta_why);
    if (!delta) {
        if (delta_why && why_out) *why_out = delta_why;
        return NULL;
    }

    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, point->shape, point->n_dims);
    if (!out) return NULL;

    why = eshkol_rm_exp_map(point->data, delta->data, K, n, out->data, scratch);
    if (why) { if (why_out) *why_out = why; return NULL; }

    /* The first moment is a tangent vector at the OLD point and is meaningless
     * at the new one until transported. */
    why = eshkol_rm_transport(point->data, out->data, m_next, K, n, moved, scratch);
    if (why) { if (why_out) *why_out = why; return NULL; }

    memcpy(st->m, moved, (size_t)n * sizeof(double));
    st->rms = rms_next;
    st->step = step_next;
    st->owner = out;
    return out;
}

/** @brief Whether @p mv is a manifold Value wrapping a non-null handle. */
static int vm_manifold_has_value(VM* vm, Value mv) {
    return mv.type == VAL_MANIFOLD && is_heap_type(vm, mv, HEAP_MANIFOLD) &&
           vm->heap.objects[mv.as.ptr]->opaque.ptr != NULL;
}

/** @brief Get manifold @p mv's stored constant curvature, setting *@p ok to
 *         whether the lookup succeeded. */
static double vm_geometric_manifold_curvature(VM* vm, Value mv, int* ok) {
    if (!vm_manifold_has_value(vm, mv)) {
        if (ok) *ok = 0;
        return 0.0;
    }
    VmGeometricManifold* m = vm_geometric_manifold(vm, mv);
    if (!m) {
        if (ok) *ok = 0;
        return 0.0;
    }
    if (ok) *ok = 1;
    return m->curvature;
}

/** @brief Get manifold @p mv's dimension, or 0 when @p mv is not a live
 *         manifold handle. This used to have a second branch that returned 0
 *         unconditionally (SW-72); that branch was in the deleted qLLM body and
 *         the one below is the only path any build ever compiled. */
static int vm_geometric_manifold_dim(VM* vm, Value mv) {
    VmGeometricManifold* m = vm_geometric_manifold(vm, mv);
    return m ? m->dim : 0;
}

/** @brief Deep-copy a tensor for use in geometric-op results (allocates a
 *         fresh zero tensor of @p src's shape and memcpy's the data). */
static VmTensor* vm_tensor_copy_for_geometry(VM* vm, const VmTensor* src) {
    if (!src || !src->data || src->n_dims <= 0) return NULL;
    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, src->shape, src->n_dims);
    if (!out) return NULL;
    memcpy(out->data, src->data, (size_t)src->total * sizeof(double));
    return out;
}

/** @brief Compute the element-wise linear combination @p as*a + @p bs*b
 *         into a freshly allocated tensor (shapes/totals must match). */
static VmTensor* vm_tensor_linear_combo_for_geometry(VM* vm, const VmTensor* a, double as,
                                                     const VmTensor* b, double bs) {
    if (!a || !b || !a->data || !b->data || a->total != b->total) return NULL;
    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, a->shape, a->n_dims);
    if (!out) return NULL;
    for (int64_t i = 0; i < out->total; i++) out->data[i] = as * a->data[i] + bs * b->data[i];
    return out;
}

/** @brief Scale a tensor's elements by @p scale into a fresh tensor. */
static VmTensor* vm_tensor_scale_for_geometry(VM* vm, const VmTensor* src, double scale) {
    if (!src || !src->data) return NULL;
    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, src->shape, src->n_dims);
    if (!out) return NULL;
    for (int64_t i = 0; i < out->total; i++) out->data[i] = src->data[i] * scale;
    return out;
}

/** @brief Euclidean dot product of two same-shaped tensors (0.0 if
 *         mismatched/null). */
static double vm_tensor_dot_for_geometry(const VmTensor* a, const VmTensor* b) {
    if (!a || !b || !a->data || !b->data || a->total != b->total) return 0.0;
    double sum = 0.0;
    for (int64_t i = 0; i < a->total; i++) sum += a->data[i] * b->data[i];
    return sum;
}

/** @brief Euclidean (L2) distance between two same-shaped tensors. */
static double vm_tensor_distance_for_geometry(const VmTensor* a, const VmTensor* b) {
    if (!a || !b || !a->data || !b->data || a->total != b->total) return 0.0;
    double sum = 0.0;
    for (int64_t i = 0; i < a->total; i++) {
        double d = a->data[i] - b->data[i];
        sum += d * d;
    }
    return sqrt(sum);
}

/** @brief L2-normalize a tensor's elements in place (no-op if its norm is
 *         zero). */
static void vm_tensor_normalize_for_geometry(VmTensor* t) {
    if (!t || !t->data) return;
    double norm2 = 0.0;
    for (int64_t i = 0; i < t->total; i++) norm2 += t->data[i] * t->data[i];
    if (norm2 <= 0.0) return;
    double inv = 1.0 / sqrt(norm2);
    for (int64_t i = 0; i < t->total; i++) t->data[i] *= inv;
}

/** @brief Wrap a tensor pointer as a heap-boxed VAL_TENSOR Value and push
 *         it (NIL, flagging vm->error, on null/allocation failure). */
static void vm_push_tensor_handle_for_geometry(VM* vm, VmTensor* t) {
    if (!t) { vm_push(vm, NIL_VAL); return; }
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) { vm->error = 1; vm_push(vm, NIL_VAL); return; }
    vm->heap.objects[ptr]->type = HEAP_TENSOR;
    vm->heap.objects[ptr]->opaque.ptr = t;
    vm_push(vm, (Value){.type = VAL_TENSOR, .as.ptr = ptr});
}

/** @brief Alias for vm_push_tensor_handle_for_geometry() used by opcode
 *         handlers whose result may legitimately be NULL/nil. */
static void vm_push_tensor_or_nil(VM* vm, VmTensor* t) {
    vm_push_tensor_handle_for_geometry(vm, t);
}

/** @brief VM opcode handler: `(manifold-metric-tensor m)` — pops a
 *         manifold, pushes its dim x dim identity metric tensor (a
 *         constant-curvature manifold's metric is Euclidean at this
 *         level of approximation). */
static void vm_geometric_metric_tensor(VM* vm) {
    Value mv = vm_pop(vm);
    int dim = vm_geometric_manifold_dim(vm, mv);
    if (dim <= 0 || dim > 256) { vm_push(vm, NIL_VAL); return; }
    int64_t shape[2] = {dim, dim};
    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 2);
    if (!out) { vm_push(vm, NIL_VAL); return; }
    for (int i = 0; i < dim; i++) out->data[i * dim + i] = 1.0;
    vm_push_tensor_handle_for_geometry(vm, out);
}

/** @brief VM opcode handler: `(manifold-christoffel-tensor m point)` —
 *         pops a manifold and a point tensor, pushes the dim x dim x dim
 *         Christoffel symbols for a constant-curvature-K space evaluated
 *         at @p point (a closed-form approximation, not a full
 *         Riemann-tensor computation). */
static void vm_geometric_christoffel_tensor(VM* vm) {
    Value pointv = vm_pop(vm);
    Value mv = vm_pop(vm);
    VmTensor* point = vm_get_tensor(vm, pointv);
    int ok = 0;
    double K = vm_geometric_manifold_curvature(vm, mv, &ok);
    if (!ok || !point || !point->data) { vm_push(vm, NIL_VAL); return; }
    int dim = vm_geometric_manifold_dim(vm, mv);
    if (dim <= 0 || dim > point->total) dim = (int)point->total;
    if (dim <= 0 || dim > 64) { vm_push(vm, NIL_VAL); return; }

    int64_t shape[3] = {dim, dim, dim};
    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 3);
    if (!out) { vm_push(vm, NIL_VAL); return; }

    for (int k = 0; k < dim; k++) {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                double xk = point->data[k];
                double xi = point->data[i];
                double xj = point->data[j];
                double v = K * ((i == j ? xk : 0.0) -
                                (j == k ? xi : 0.0) -
                                (i == k ? xj : 0.0));
                out->data[((int64_t)k * dim + i) * dim + j] = v;
            }
        }
    }
    vm_push_tensor_handle_for_geometry(vm, out);
}

/** @brief VM opcode handler: `(pullback-tensor form jacobian)` — pops a
 *         1-form (or a flattenable rows*cols source) and a Jacobian
 *         matrix, pushes their pullback form^T @ jacobian as a new 1D
 *         tensor. */
static void vm_geometric_pullback_tensor(VM* vm) {
    Value jacv = vm_pop(vm);
    Value formv = vm_pop(vm);
    VmTensor* jac = vm_get_tensor(vm, jacv);
    VmTensor* form = vm_get_tensor(vm, formv);
    if (!jac || !form || !jac->data || !form->data) { vm_push(vm, NIL_VAL); return; }

    int64_t rows = 0, cols = 0;
    if (jac->n_dims >= 2) {
        rows = jac->shape[0];
        cols = jac->shape[1];
    } else if (form->total > 0 && jac->total % form->total == 0) {
        rows = form->total;
        cols = jac->total / form->total;
    }
    if (rows <= 0 || cols <= 0 || rows * cols > jac->total || rows > form->total) {
        vm_push(vm, NIL_VAL);
        return;
    }

    int64_t shape[1] = {cols};
    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
    if (!out) { vm_push(vm, NIL_VAL); return; }
    for (int64_t j = 0; j < cols; j++) {
        double sum = 0.0;
        for (int64_t i = 0; i < rows; i++) sum += form->data[i] * jac->data[i * cols + j];
        out->data[j] = sum;
    }
    vm_push_tensor_handle_for_geometry(vm, out);
}

/** @brief Number of arguments the geometric native call @p fid (804-861)
 *         expects to pop from the VM stack, used by the caller to
 *         validate/marshal arguments before dispatch. */
static int vm_geometric_arity(int fid) {
    switch (fid) {
    case 804: case 806: case 808: case 823: case 824: case 825:
    case 826: case 827: case 829: case 831: case 832: case 835:
    case 851: case 857: case 858: case 859: case 860:
        return 1;
    case 805: case 807: case 813: case 819: case 821: case 822:
    case 828: case 830: case 834: case 836: case 837: case 838:
    case 846: case 850: case 852: case 855: case 856:
        return 2;
    case 809: case 810: case 811: case 814: case 815: case 816:
    case 817: case 820: case 833: case 841: case 842: case 844:
    case 845: case 853: case 854:
        return 3;
    case 812: case 839: case 843: case 847:
        return 4;
    case 840:
        return 6;
    case 861:
        return 7;
    default:
        return 0;
    }
}

/* Levenberg floor on the objective's curvature in the adaptive step. A Newton
 * step divides by L''(K); where the objective is flat or concave in K that
 * quotient is meaningless or points uphill, and flooring the denominator is the
 * standard damping that turns the step back into a descent step without
 * inventing a direction. */
#define VM_CURVATURE_NEWTON_FLOOR 1e-8

/* Backtracking budget. Halving 32 times shrinks the step by 2^-32, so a run
 * that exhausts it is not near-converged -- there is no admissible step at all,
 * and the op says so rather than moving K somewhere it cannot justify. */
#define VM_CURVATURE_BACKTRACKS 32

/**
 * @brief The curvature objective and its first two derivatives in K:
 *
 *     L(K) = sum_p d_K(x_p, y_p)
 *
 * over the point pairs packed consecutively in @p pairs (x_0, y_0, x_1, y_1,
 * ... , each of @p n coordinates). Every term's derivatives are the exact
 * closed forms in riemannian_core.h, so L' and L'' are exact too.
 *
 * @return NULL on success, else the reason the objective is not evaluable at
 *         @p K (a point outside the ball, a point off the sphere, or K = 0
 *         where the metric family is discontinuous).
 */
static const char* vm_curvature_objective(const VmTensor* pairs, int n, double K,
                                          double* L, double* L1, double* L2) {
    int64_t stride = (int64_t)2 * n;
    int64_t np = pairs->total / stride;
    double sL = 0.0, s1 = 0.0, s2 = 0.0;
    for (int64_t p = 0; p < np; p++) {
        const double* x = pairs->data + p * stride;
        const double* y = x + n;
        double d = 0.0, d1 = 0.0, d2 = 0.0;
        const char* why = eshkol_rm_distance_dK(x, y, K, n, &d, &d1, &d2);
        if (why) return why;
        sL += d; s1 += d1; s2 += d2;
    }
    if (L)  *L  = sL;
    if (L1) *L1 = s1;
    if (L2) *L2 = s2;
    return NULL;
}

/** @brief Evaluate a curvature trial in the family named by the objective.
 *
 * For K <= 0 the objective holds the supplied coordinates fixed. On the
 * spherical branch, changing K changes the sphere radius, so the documented
 * trial family holds each pair's angular positions fixed by rescaling every
 * point by sqrt(K/current_trial_K) before evaluating the trial distance. */
static const char* vm_curvature_trial_objective(const VmTensor* pairs, int n,
                                                double current_K, double trial_K,
                                                double* L) {
    if (!(current_K > 0.0 && trial_K > 0.0))
        return vm_curvature_objective(pairs, n, trial_K, L, NULL, NULL);

    double scale = sqrt(current_K / trial_K);
    if (!(scale > 0.0) || !isfinite(scale))
        return "the spherical curvature trial has a non-finite radius scale";
    int64_t stride = (int64_t)2 * n;
    int64_t np = pairs->total / stride;
    double total = 0.0;
    double x_trial[n > 0 ? n : 1];
    double y_trial[n > 0 ? n : 1];
    for (int64_t p = 0; p < np; p++) {
        const double* x = pairs->data + p * stride;
        const double* y = x + n;
        for (int i = 0; i < n; i++) {
            x_trial[i] = scale * x[i];
            y_trial[i] = scale * y[i];
        }
        double d = 0.0;
        const char* why = eshkol_rm_distance(x_trial, y_trial, trial_K, n, &d);
        if (why) return why;
        total += d;
    }
    if (L) *L = total;
    return NULL;
}

/**
 * @brief Pop the (manifold, pairs) arguments the three curvature-objective ops
 *        share and validate that the pairs tensor packs whole point pairs of
 *        the manifold's dimension.
 *
 * All three arguments are popped whichever way this goes, so the caller only
 * has to push a result. A tensor whose length is not a positive multiple of
 * 2*dim is a SHAPE failure and the caller pushes () for it, per this surface's
 * error convention; a well-shaped batch that the geometry refuses is a DOMAIN
 * failure and raises.
 *
 * @return 1 on success.
 */
static int vm_curvature_args(VM* vm, Value* mv_out, VmGeometricManifold** m_out,
                             VmTensor** pairs_out, int* n_out) {
    VmTensor* pairs = vm_get_tensor(vm, vm_pop(vm));
    Value mv = vm_pop(vm);
    VmGeometricManifold* m = vm_geometric_manifold(vm, mv);
    if (!m || !pairs || !pairs->data) return 0;
    int n = m->dim;
    if (n <= 0 || pairs->total <= 0 || pairs->total % ((int64_t)2 * n) != 0)
        return 0;
    *mv_out = mv; *m_out = m; *pairs_out = pairs; *n_out = n;
    return 1;
}

/**
 * @brief Top-level native-call dispatcher for the geometric primitives
 *        (IDs 804-861). Pops each op's arguments per vm_geometric_arity(fid)
 *        and pushes its result, covering manifold construction
 *        (Euclidean/hyperbolic/spherical/product), exp/log maps, geodesics,
 *        parallel transport, distance/norm/dot, curvature and
 *        Christoffel-symbol queries, the differential-form calculus, the
 *        curvature-objective derivatives and the Riemannian optimizers.
 *
 * The closed forms live in the two core headers rather than here so that the
 * VM opcode and the AD bridge cannot drift apart about what an operation
 * means.
 */
static void vm_dispatch_geometric(VM* vm, int fid) {
    switch (fid) {
    case 804: { /* make-euclidean-manifold(dim) */
        int dim = (int)as_number(vm_pop(vm));
        vm_push_geometric_manifold(vm, 0, dim, 0.0);
        break;
    }
    case 805: { /* make-hyperbolic-manifold(dim, curvature) */
        double c = as_number(vm_pop(vm));
        int dim = (int)as_number(vm_pop(vm));
        vm_push_geometric_manifold(vm, 1, dim, c);
        break;
    }
    case 806: { /* make-spherical-manifold(dim) */
        int dim = (int)as_number(vm_pop(vm));
        vm_push_geometric_manifold(vm, 2, dim, 1.0);
        break;
    }
    case 807: { /* make-product-manifold(m1, m2) */
        Value m2v = vm_pop(vm), m1v = vm_pop(vm);
        VmGeometricManifold* m1 = vm_geometric_manifold(vm, m1v);
        VmGeometricManifold* m2 = vm_geometric_manifold(vm, m2v);
        if (m1 && m2) vm_push_geometric_manifold(vm, 3, m1->dim + m2->dim,
                                                0.5 * (m1->curvature + m2->curvature));
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 808: case 831: case 851: { /* manifold-curvature/get-curvature/riemann-curvature */
        Value mv = vm_pop(vm);
        int ok = 0;
        double c = vm_geometric_manifold_curvature(vm, mv, &ok);
        if (ok) vm_push_float(vm, c);
        else vm_push(vm, NIL_VAL);
        break;
    }

    case 809: case 842: { /* exp-map/retraction(base, tangent, curvature) */
        /* Was `base + tangent` for every curvature. That is exp_x(v) only at
         * K = 0; on the ball the geodesic is a circular arc orthogonal to the
         * boundary and the endpoint is a Mobius sum, not a vector sum. */
        double K = as_number(vm_pop(vm));
        VmTensor* tangent = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* base = vm_get_tensor(vm, vm_pop(vm));
        if (!base || !tangent || !base->data || !tangent->data ||
            base->total != tangent->total || base->total <= 0) {
            vm_push(vm, NIL_VAL); break;
        }
        int n = (int)base->total;
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, base->shape, base->n_dims);
        double* scratch = vm_geometric_scratch(vm, n, VM_GEOMETRIC_SCRATCH_MULT);
        if (!out || !scratch) { vm_push(vm, NIL_VAL); break; }
        const char* why = eshkol_rm_exp_map(base->data, tangent->data, K, n,
                                            out->data, scratch);
        if (why) { vm_geometric_raise(vm, fid == 842 ? "retraction" : "exp-map", why, K); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 810: case 822: { /* log-map(base, point, curvature) / spherical-log(base, point) */
        /* Was `point - base` for every curvature, i.e. the K = 0 answer under
         * both names. `spherical-log` carries no curvature argument: it is the
         * unit sphere, K = +1. */
        double K = 1.0;
        if (fid == 810) K = as_number(vm_pop(vm));
        VmTensor* point = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* base = vm_get_tensor(vm, vm_pop(vm));
        if (!base || !point || !base->data || !point->data ||
            base->total != point->total || base->total <= 0) {
            vm_push(vm, NIL_VAL); break;
        }
        int n = (int)base->total;
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, base->shape, base->n_dims);
        double* scratch = vm_geometric_scratch(vm, n, VM_GEOMETRIC_SCRATCH_MULT);
        if (!out || !scratch) { vm_push(vm, NIL_VAL); break; }
        const char* why = eshkol_rm_log_map(base->data, point->data, K, n,
                                            out->data, scratch);
        if (why) { vm_geometric_raise(vm, fid == 822 ? "spherical-log" : "log-map", why, K); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 811: case 816: { /* geodesic-distance/poincare-distance(x, y, curvature) */
        /* Was the L2 distance for every curvature. On the ball the L2 chord is
         * bounded by the ball diameter while the geodesic distance diverges at
         * the boundary, so the two disagree without bound: at K = -1 the points
         * (0.9, 0) and (-0.9, 0) are 1.8 apart in L2 and 5.9 apart in the
         * metric the name promises. */
        double K = as_number(vm_pop(vm));
        VmTensor* y = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        if (!x || !y || !x->data || !y->data || x->total != y->total || x->total <= 0) {
            vm_push(vm, NIL_VAL); break;
        }
        double d = 0.0;
        const char* why = eshkol_rm_distance(x->data, y->data, K, (int)x->total, &d);
        if (why) {
            vm_geometric_raise(vm, fid == 816 ? "poincare-distance" : "geodesic-distance", why, K);
            break;
        }
        vm_push_float(vm, d);
        break;
    }
    case 812: case 843: { /* parallel/vector transport(x, y, v, curvature) */
        /* Was the identity on v, with x and y popped and discarded. Holonomy is
         * the whole content of the op: on a curved manifold transporting a
         * vector around a closed loop rotates it, and the identity map reports
         * that curvature is zero. */
        double K = as_number(vm_pop(vm));
        VmTensor* v = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* y = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        if (!x || !y || !v || !x->data || !y->data || !v->data ||
            x->total != y->total || x->total != v->total || x->total <= 0) {
            vm_push(vm, NIL_VAL); break;
        }
        int n = (int)x->total;
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, v->shape, v->n_dims);
        double* scratch = vm_geometric_scratch(vm, n, VM_GEOMETRIC_SCRATCH_MULT);
        if (!out || !scratch) { vm_push(vm, NIL_VAL); break; }
        const char* why = eshkol_rm_transport(x->data, y->data, v->data, K, n,
                                               out->data, scratch);
        if (why) {
            vm_geometric_raise(vm, fid == 843 ? "vector-transport" : "parallel-transport", why, K);
            break;
        }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 813: { /* manifold-project(x, curvature) */
        /* Was a copy, so a point outside the ball stayed outside and every op
         * downstream inherited an argument off the manifold. */
        double K = as_number(vm_pop(vm));
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        if (!x || !x->data || x->total <= 0) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, x->shape, x->n_dims);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        const char* why = eshkol_rm_project(x->data, K, (int)x->total, out->data);
        if (why) { vm_geometric_raise(vm, "manifold-project", why, K); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 814: { /* mobius-add(x, y, curvature) */
        /* Was x + y for every curvature. Mobius addition is NOT commutative and
         * NOT associative -- gyr[x,y] is exactly the failure of commutativity --
         * so a commutative stand-in erases the structure the op names. */
        double K = as_number(vm_pop(vm));
        VmTensor* y = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        if (!x || !y || !x->data || !y->data || x->total != y->total || x->total <= 0) {
            vm_push(vm, NIL_VAL); break;
        }
        if (K > 0.0) {
            vm_geometric_raise(vm, "mobius-add",
                               "Mobius addition is the gyrogroup operation of the "
                               "Poincare ball and is defined for K <= 0 only", K);
            break;
        }
        const char* why = eshkol_rm_check_point(x->data, K, (int)x->total);
        if (!why) why = eshkol_rm_check_point(y->data, K, (int)y->total);
        if (why) { vm_geometric_raise(vm, "mobius-add", why, K); break; }
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, x->shape, x->n_dims);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        eshkol_rm_mobius_add(x->data, y->data, eshkol_rm_ball_param(-K),
                             (int)x->total, out->data);
        why = eshkol_rm_require_interior(out->data, K, (int)x->total);
        if (why) { vm_geometric_raise(vm, "mobius-add", why, K); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 815: { /* mobius-scalar-mul(r, x, curvature) */
        /* Was r*x for every curvature, which leaves the ball for |r| large
         * enough -- the real operation cannot, because it is
         * (1/sqrt(c)) tanh(r artanh(sqrt(c)|x|)) x/|x| and tanh is bounded. */
        double K = as_number(vm_pop(vm));
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        double r = as_number(vm_pop(vm));
        if (!x || !x->data || x->total <= 0) { vm_push(vm, NIL_VAL); break; }
        if (K > 0.0) {
            vm_geometric_raise(vm, "mobius-scalar-mul",
                               "Mobius scalar multiplication is the gyrovector "
                               "operation of the Poincare ball and is defined for "
                               "K <= 0 only", K);
            break;
        }
        const char* why = eshkol_rm_check_point(x->data, K, (int)x->total);
        if (why) { vm_geometric_raise(vm, "mobius-scalar-mul", why, K); break; }
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, x->shape, x->n_dims);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        why = eshkol_rm_mobius_scalar(r, x->data, K, (int)x->total, out->data);
        if (why) { vm_geometric_raise(vm, "mobius-scalar-mul", why, K); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 817: /* frechet-mean(points, weights, curvature) */
        /* Real Riemannian center of mass in f64, gated on the stationarity
         * residual. This used to be the Euclidean weighted average with the
         * curvature argument discarded — see vm_dispatch_frechet_mean. */
        vm_dispatch_frechet_mean(vm);
        break;

    case 819: { /* great-circle-distance(x, y) */
        VmTensor* y = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        if (!x || !y || x->total != y->total) { vm_push(vm, NIL_VAL); break; }
        double nx = sqrt(vm_tensor_dot_for_geometry(x, x));
        double ny = sqrt(vm_tensor_dot_for_geometry(y, y));
        if (nx <= 0.0 || ny <= 0.0) { vm_push_float(vm, 0.0); break; }
        double cs = vm_tensor_dot_for_geometry(x, y) / (nx * ny);
        if (cs > 1.0) cs = 1.0;
        if (cs < -1.0) cs = -1.0;
        vm_push_float(vm, acos(cs));
        break;
    }
    case 820: { /* slerp(x, y, t) */
        double t = as_number(vm_pop(vm));
        VmTensor* y = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* out = vm_tensor_linear_combo_for_geometry(vm, x, 1.0 - t, y, t);
        vm_tensor_normalize_for_geometry(out);
        vm_push_tensor_or_nil(vm, out);
        break;
    }
    case 821: { /* spherical-exp(base, tangent) */
        /* Was normalize(base + tangent), a RETRACTION and not the exponential
         * map: it lands on the right geodesic but at the wrong arc length
         * (atan|v| instead of |v|), so it is first-order correct and wrong at
         * every order after. */
        VmTensor* tangent = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* base = vm_get_tensor(vm, vm_pop(vm));
        if (!base || !tangent || !base->data || !tangent->data ||
            base->total != tangent->total || base->total <= 0) {
            vm_push(vm, NIL_VAL); break;
        }
        int n = (int)base->total;
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, base->shape, base->n_dims);
        double* scratch = vm_geometric_scratch(vm, n, VM_GEOMETRIC_SCRATCH_MULT);
        if (!out || !scratch) { vm_push(vm, NIL_VAL); break; }
        const char* why = eshkol_rm_exp_map(base->data, tangent->data, 1.0, n,
                                             out->data, scratch);
        if (why) { vm_geometric_raise(vm, "spherical-exp", why, 1.0); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 823: { /* spherical-project(x) */
        /* Was an L2-normalising copy that returned the ZERO VECTOR unchanged
         * for a zero-norm input -- and the origin is not a point of the unit
         * sphere, so that was a value outside the op's own codomain, returned
         * without a diagnostic. eshkol_rm_project refuses it by name, the same
         * refusal the other spherical ops already make, so the precondition is
         * now stated in the value rather than left to the caller to discover. */
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        if (!x || !x->data || x->total <= 0) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, x->shape, x->n_dims);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        const char* why = eshkol_rm_project(x->data, 1.0, (int)x->total, out->data);
        if (why) { vm_geometric_raise(vm, "spherical-project", why, 1.0); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }

    case 824: { /* so3-exp(omega) */
        VmTensor* omega = vm_get_tensor(vm, vm_pop(vm));
        if (!omega || omega->total < 3) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[1] = {4};
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        double ox = omega->data[0], oy = omega->data[1], oz = omega->data[2];
        double theta = sqrt(ox * ox + oy * oy + oz * oz);
        if (theta <= 1e-12) {
            out->data[0] = 1.0;
        } else {
            double half = 0.5 * theta;
            double s = sin(half) / theta;
            out->data[0] = cos(half);
            out->data[1] = ox * s;
            out->data[2] = oy * s;
            out->data[3] = oz * s;
        }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 825: { /* so3-log(quat) */
        VmTensor* q = vm_get_tensor(vm, vm_pop(vm));
        if (!q || q->total < 4) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[1] = {3};
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        double n = sqrt(q->data[0] * q->data[0] + q->data[1] * q->data[1] +
                        q->data[2] * q->data[2] + q->data[3] * q->data[3]);
        if (n <= 0.0) { VM_PUSH_TENSOR(vm, out); break; }
        double w = q->data[0] / n;
        if (w > 1.0) w = 1.0;
        if (w < -1.0) w = -1.0;
        double x = q->data[1] / n, y = q->data[2] / n, z = q->data[3] / n;
        double vnorm = sqrt(x * x + y * y + z * z);
        if (vnorm > 1e-12) {
            double theta = 2.0 * atan2(vnorm, w);
            out->data[0] = x * theta / vnorm;
            out->data[1] = y * theta / vnorm;
            out->data[2] = z * theta / vnorm;
        }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 826: { /* se3-exp(twist) */
        VmTensor* twist = vm_get_tensor(vm, vm_pop(vm));
        if (!twist || twist->total < 6) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[1] = {7};
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        double ox = twist->data[0], oy = twist->data[1], oz = twist->data[2];
        double theta = sqrt(ox * ox + oy * oy + oz * oz);
        if (theta <= 1e-12) out->data[0] = 1.0;
        else {
            double half = 0.5 * theta;
            double s = sin(half) / theta;
            out->data[0] = cos(half);
            out->data[1] = ox * s;
            out->data[2] = oy * s;
            out->data[3] = oz * s;
        }
        out->data[4] = twist->data[3];
        out->data[5] = twist->data[4];
        out->data[6] = twist->data[5];
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 827: { /* se3-log(pose) */
        VmTensor* pose = vm_get_tensor(vm, vm_pop(vm));
        if (!pose || pose->total < 7) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[1] = {6};
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        double n = sqrt(pose->data[0] * pose->data[0] + pose->data[1] * pose->data[1] +
                        pose->data[2] * pose->data[2] + pose->data[3] * pose->data[3]);
        if (n > 0.0) {
            double w = pose->data[0] / n;
            if (w > 1.0) w = 1.0;
            if (w < -1.0) w = -1.0;
            double x = pose->data[1] / n, y = pose->data[2] / n, z = pose->data[3] / n;
            double vnorm = sqrt(x * x + y * y + z * z);
            if (vnorm > 1e-12) {
                double theta = 2.0 * atan2(vnorm, w);
                out->data[0] = x * theta / vnorm;
                out->data[1] = y * theta / vnorm;
                out->data[2] = z * theta / vnorm;
            }
        }
        out->data[3] = pose->data[4];
        out->data[4] = pose->data[5];
        out->data[5] = pose->data[6];
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 828: { /* quaternion-mul(q1, q2) */
        VmTensor* q2 = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* q1 = vm_get_tensor(vm, vm_pop(vm));
        if (!q1 || !q2 || q1->total < 4 || q2->total < 4) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[1] = {4};
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        double a = q1->data[0], b = q1->data[1], c = q1->data[2], d = q1->data[3];
        double e = q2->data[0], f = q2->data[1], g = q2->data[2], h = q2->data[3];
        out->data[0] = a * e - b * f - c * g - d * h;
        out->data[1] = a * f + b * e + c * h - d * g;
        out->data[2] = a * g - b * h + c * e + d * f;
        out->data[3] = a * h + b * g - c * f + d * e;
        VM_PUSH_TENSOR(vm, out);
        break;
    }

    case 829: { /* metric-tensor(manifold) */
        vm_geometric_metric_tensor(vm);
        break;
    }
    case 830: { /* christoffel(manifold, point) */
        vm_geometric_christoffel_tensor(vm);
        break;
    }
    case 832: { /* ricci-scalar(manifold) */
        Value mv = vm_pop(vm);
        int ok = 0;
        double K = vm_geometric_manifold_curvature(vm, mv, &ok);
        int dim = vm_geometric_manifold_dim(vm, mv);
        if (ok && dim > 0) vm_push_float(vm, (double)dim * (double)(dim - 1) * K);
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 833: { /* sectional-curvature(manifold, u, v) */
        (void)vm_pop(vm);
        (void)vm_pop(vm);
        Value mv = vm_pop(vm);
        int ok = 0;
        double K = vm_geometric_manifold_curvature(vm, mv, &ok);
        if (ok) vm_push_float(vm, K);
        else vm_push(vm, NIL_VAL);
        break;
    }

    case 834: { /* wedge-product(form_a, form_b) */
        VmTensor* b = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* a = vm_get_tensor(vm, vm_pop(vm));
        if (!a || !b || !a->data || !b->data) { vm_push(vm, NIL_VAL); break; }
        int64_t n = a->total < b->total ? a->total : b->total;
        int64_t count = (n > 1) ? (n * (n - 1)) / 2 : 1;
        int64_t shape[1] = {count};
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        int64_t p = 0;
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = i + 1; j < n; j++)
                out->data[p++] = a->data[i] * b->data[j] - a->data[j] * b->data[i];
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 835: { /* exterior-derivative(form) */
        /* Returned a zero tensor of the input's shape for every input, which
         * reads as "d(this form) = 0", i.e. that every form handed to it is
         * closed -- a statement about the form made without looking at one.
         * SW-73 turned that into a refusal naming the missing input: a
         * coefficient array AT A POINT carries no derivative information.
         *
         * The op now takes the input that DOES determine the answer. `d` is a
         * first-order operator, so the 1-jet of the coefficients at the point
         * is exactly enough, and the form representation carries it (together
         * with the degree and dimension a flat array never recorded). The
         * result is exact -- no difference quotient, no step -- and consumes
         * one jet order, so d(d(w)) is computable from an r >= 2 form and is
         * exactly zero rather than zero to a tolerance. */
        VmTensor* form = vm_get_tensor(vm, vm_pop(vm));
        if (!form || !form->data || form->total <= 0) { vm_push(vm, NIL_VAL); break; }
        int k = 0, n = 0, r = 0;
        const char* why = eshkol_form_header(form->data, (long)form->total, &k, &n, &r);
        if (why) { vm_push(vm, NIL_VAL); break; }
        if (r < 1 || k > n) {
            if (k > n) {
                long out_total = ESHKOL_FORM_HEADER;
                int64_t oshape[1] = { (int64_t)out_total };
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, oshape, 1);
                if (!out) { vm_push(vm, NIL_VAL); break; }
                why = eshkol_form_d(form->data, (long)form->total,
                                    out->data, out_total);
                if (why) { vm_form_raise(vm, "exterior-derivative", why); break; }
                VM_PUSH_TENSOR(vm, out);
                break;
            }
            vm_form_raise(vm, "exterior-derivative",
                          r < 1 ? "d is a derivative, so the coefficients' "
                                  "first partials must be supplied: this form "
                                  "has jet order r = 0"
                                : "the zero top-degree form has nothing to "
                                  "differentiate");
            break;
        }
        long out_total = eshkol_form_total(k + 1, n, r - 1);
        int64_t oshape[1] = { (int64_t)out_total };
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, oshape, 1);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        why = eshkol_form_d(form->data, (long)form->total, out->data, out_total);
        if (why) { vm_form_raise(vm, "exterior-derivative", why); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 836: { /* hodge-star(form, metric) */
        /* Returned its input unchanged -- the Hodge star only for a self-dual
         * middle-degree form in a Euclidean metric, i.e. the identity map
         * presented under the name of a duality. SW-73 turned that into a
         * refusal naming the missing input: the star of a k-form depends on k
         * and on n, and C(n,k) = C(n,n-k) leaves k ambiguous even when n and
         * the array length are both known.
         *
         * The form representation records k, n and r, so the duality being
         * asked for is now determined by the value rather than guessed. The
         * result is a 0-JET (n-k)-form and says so in its header: the star's
         * coefficients are functions of the metric, and a metric sampled at one
         * point carries nothing about how g varies, so propagating the input's
         * derivative blocks would assert that g is constant. */
        VmTensor* metric = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* form   = vm_get_tensor(vm, vm_pop(vm));
        if (!form || !form->data || form->total <= 0 ||
            !metric || !metric->data || metric->total <= 0) {
            vm_push(vm, NIL_VAL); break;
        }
        int k = 0, n = 0, r = 0;
        const char* why = eshkol_form_header(form->data, (long)form->total, &k, &n, &r);
        if (why) { vm_push(vm, NIL_VAL); break; }
        if (metric->total != (int64_t)n * (int64_t)n) { vm_push(vm, NIL_VAL); break; }
        if (k > n) {
            vm_form_raise(vm, "hodge-star",
                          "the zero top-degree form has no Hodge dual in this "
                          "representation");
            break;
        }
        long out_total = eshkol_form_total(n - k, n, 0);
        int64_t oshape[1] = { (int64_t)out_total };
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, oshape, 1);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        why = eshkol_form_star(form->data, (long)form->total, metric->data, n,
                               out->data, out_total);
        if (why) { vm_form_raise(vm, "hodge-star", why); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 837: { /* interior-product(vector, form) */
        VmTensor* form = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* vector = vm_get_tensor(vm, vm_pop(vm));
        if (!form || !vector || form->total != vector->total) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[1] = {1};
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        out->data[0] = vm_tensor_dot_for_geometry(vector, form);
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 838: { /* pullback(form, jacobian) */
        vm_geometric_pullback_tensor(vm);
        break;
    }

    case 839: { /* riemannian-sgd-step(point, gradient, lr, curvature) */
        /* Was point - lr*grad, which is the K = 0 update and leaves the ball
         * outright for a large enough step. The step is now a geodesic one:
         * exp_point(-lr * gradient). `gradient` is the RIEMANNIAN gradient (a
         * tangent vector); `riemannian-grad` is the op that converts a Euclidean
         * one, which is why they are separate names. */
        double K = as_number(vm_pop(vm));
        double lr = as_number(vm_pop(vm));
        VmTensor* grad = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* point = vm_get_tensor(vm, vm_pop(vm));
        if (!point || !grad || !point->data || !grad->data ||
            point->total != grad->total || point->total <= 0) {
            vm_push(vm, NIL_VAL); break;
        }
        int n = (int)point->total;
        VmTensor* step = vm_tensor_scale_for_geometry(vm, grad, -lr);
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, point->shape, point->n_dims);
        double* scratch = vm_geometric_scratch(vm, n, VM_GEOMETRIC_SCRATCH_MULT);
        if (!step || !out || !scratch) { vm_push(vm, NIL_VAL); break; }
        const char* why = eshkol_rm_exp_map(point->data, step->data, K, n,
                                             out->data, scratch);
        if (why) { vm_geometric_raise(vm, "riemannian-sgd-step", why, K); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 840: { /* riemannian-adam-step(point, gradient, lr, beta1, beta2, curvature) */
        /* REFUSES. This op carried an IMPLICIT optimizer state, drawn from a
         * sixteen-slot pool keyed by the point tensor's SHAPE. Adam state
         * belongs to a parameter, not to a shape: two independent parameters of
         * the same shape shared one set of moments and one step counter, so
         * each one's update depended on the other's history with nothing in the
         * returned tensor to show it. At K = 0, lr = 0.1, beta1 = 0.9,
         * beta2 = 0.999, a first parameter with g = +1 steps by -0.1 and a
         * SECOND, INDEPENDENT parameter with g = -1 should step by +0.1; the
         * shared pool gave it +0.00526316.
         *
         * There is no repair inside this arity. Keying by the point tensor's
         * identity instead of its shape does not work either, because this op
         * RETURNS A NEW TENSOR each call, so a per-iteration parameter would
         * never match its own state and Adam would silently degrade to a
         * bias-corrected SGD. The state has to be named by the caller, which is
         * exactly what riemannian-adam-step! (861) takes. */
        double K = as_number(vm_pop(vm));
        for (int i = 0; i < 5; i++) (void)vm_pop(vm);
        vm_geometric_raise(vm, "riemannian-adam-step",
            "an implicit state pool cannot tell two same-shaped parameters "
            "apart, so they would share moments and step count; use "
            "riemannian-adam-step! with a state from "
            "make-riemannian-adam-state", K);
        break;
    }
    case 860: { /* make-riemannian-adam-state(point) */
        VmTensor* point = vm_get_tensor(vm, vm_pop(vm));
        vm_push_riemannian_adam_state(vm, vm_riemannian_adam_state_new(vm, point));
        break;
    }
    case 861: { /* riemannian-adam-step!(state, point, gradient, lr, beta1, beta2, curvature) */
        double K = as_number(vm_pop(vm));
        double beta2 = as_number(vm_pop(vm));
        double beta1 = as_number(vm_pop(vm));
        double lr = as_number(vm_pop(vm));
        VmTensor* grad = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* point = vm_get_tensor(vm, vm_pop(vm));
        VmRiemannianAdamState* st = vm_riemannian_adam_state_from_value(vm, vm_pop(vm));
        const char* why = NULL;
        VmTensor* out = vm_riemannian_adam_geodesic_step(
            vm, point, grad, st, lr, beta1, beta2, K, &why);
        if (why) { vm_geometric_raise(vm, "riemannian-adam-step!", why, K); break; }
        vm_push_tensor_or_nil(vm, out);
        break;
    }
    case 841: { /* riemannian-grad(euclidean_grad, point, curvature) */
        /* Was a copy of the Euclidean gradient, with the point popped and
         * discarded -- i.e. it asserted the metric is the identity everywhere.
         * On the ball the metric is conformal with factor lambda_x, so the
         * Riemannian gradient is ((1 - c|x|^2)^2 / 4) times the Euclidean one,
         * a factor that goes to zero at the boundary. */
        double K = as_number(vm_pop(vm));
        VmTensor* point = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* grad = vm_get_tensor(vm, vm_pop(vm));
        if (!grad || !point || !grad->data || !point->data ||
            grad->total != point->total || grad->total <= 0) {
            vm_push(vm, NIL_VAL); break;
        }
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, grad->shape, grad->n_dims);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        const char* why = eshkol_rm_egrad_to_rgrad(grad->data, point->data, K,
                                                    (int)grad->total, out->data);
        if (why) { vm_geometric_raise(vm, "riemannian-grad", why, K); break; }
        VM_PUSH_TENSOR(vm, out);
        break;
    }

    case 844: { /* geodesic-attention-scores(Q, K, curvature) */
        double Kc = as_number(vm_pop(vm));
        VmTensor* k = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* q = vm_get_tensor(vm, vm_pop(vm));
        if (!q || !k || !q->data || !k->data) { vm_push(vm, NIL_VAL); break; }
        int nq = (q->n_dims >= 2) ? (int)q->shape[0] : 1;
        int nk = (k->n_dims >= 2) ? (int)k->shape[0] : 1;
        int qdim = (q->n_dims >= 2) ? (int)q->shape[1] : (int)q->total;
        int kdim = (k->n_dims >= 2) ? (int)k->shape[1] : (int)k->total;
        if (qdim != kdim) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[2] = {nq, nk};
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 2);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        /* Scored by the NEGATIVE GEODESIC distance, the same convention as
         * `ad_geodesic_attention` in lib/bridge/qllm_bridge.cpp. This used to be
         * the negative L2 distance for every curvature, so "geodesic attention"
         * on the ball ranked keys by the ambient chord. */
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < nk; j++) {
                double dist = 0.0;
                const char* why = eshkol_rm_distance(q->data + (int64_t)i * qdim,
                                                     k->data + (int64_t)j * kdim,
                                                     Kc, qdim, &dist);
                if (why) { vm_geometric_raise(vm, "geodesic-attention-scores", why, Kc); return; }
                out->data[i * nk + j] = -dist;
            }
        }
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 845: { /* geodesic-attention-values(scores, V, curvature) */
        (void)as_number(vm_pop(vm));
        VmTensor* values = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* scores = vm_get_tensor(vm, vm_pop(vm));
        if (!scores || !values || values->n_dims < 2) { vm_push(vm, NIL_VAL); break; }
        int n = (int)values->shape[0], dim = (int)values->shape[1];
        int64_t shape[1] = {dim};
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        double wsum = 0.0;
        for (int i = 0; i < n && i < scores->total; i++) {
            double w = scores->data[i];
            wsum += w;
            for (int d = 0; d < dim; d++) out->data[d] += w * values->data[i * dim + d];
        }
        if (wsum != 0.0) for (int d = 0; d < dim; d++) out->data[d] /= wsum;
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 846: { /* curvature-softmax(scores, curvature) */
        double c = as_number(vm_pop(vm));
        VmTensor* scores = vm_get_tensor(vm, vm_pop(vm));
        if (!scores || !scores->data || scores->total <= 0) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, scores->shape, scores->n_dims);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        double scale = fabs(c) > 0.0 ? 1.0 / sqrt(fabs(c)) : 1.0;
        double maxv = scores->data[0];
        for (int64_t i = 1; i < scores->total; i++) if (scores->data[i] > maxv) maxv = scores->data[i];
        double sum = 0.0;
        for (int64_t i = 0; i < scores->total; i++) {
            out->data[i] = exp((scores->data[i] - maxv) * scale);
            sum += out->data[i];
        }
        if (sum != 0.0) for (int64_t i = 0; i < scores->total; i++) out->data[i] /= sum;
        VM_PUSH_TENSOR(vm, out);
        break;
    }
    case 847: { /* geodesic-attention-forward(Q, K, V, curvature) */
        double Kc = as_number(vm_pop(vm));
        VmTensor* values = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* k = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* q = vm_get_tensor(vm, vm_pop(vm));
        if (!q || !k || !values || q->n_dims < 2 || k->n_dims < 2 || values->n_dims < 2) {
            vm_push(vm, NIL_VAL); break;
        }
        int nq = (int)q->shape[0], nk = (int)k->shape[0], dim = (int)q->shape[1];
        int vdim = (int)values->shape[1];
        if ((int)k->shape[1] != dim || (int)values->shape[0] < nk) { vm_push(vm, NIL_VAL); break; }
        int64_t shape[2] = {nq, vdim};
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 2);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        /* Softmax over the NEGATIVE GEODESIC distance, scaled by
         * 1/(sqrt(c) sqrt(dim)), then a Euclidean weighted sum of the value
         * rows -- the aggregation `ad_geodesic_attention` performs. Two things
         * changed here: the distance was the ambient L2 for every curvature, and
         * the weights were exp() with no max-shift, which overflows to inf for
         * strongly negative scores and then divides inf by inf. */
        double sc = (Kc < 0.0) ? eshkol_rm_sqrt_nonnegative(-Kc) : 1.0;
        double scale = 1.0 / (sc * sqrt((double)dim));
        double* row = vm_geometric_scratch(vm, nk, 1);
        if (!row) { vm_push(vm, NIL_VAL); break; }
        for (int i = 0; i < nq; i++) {
            double min_dist = HUGE_VAL;
            for (int j = 0; j < nk; j++) {
                double dist = 0.0;
                const char* domain_why = Kc > 0.0
                    ? eshkol_rm_sphere_distance_domain(q->data + (int64_t)i * dim,
                                                        k->data + (int64_t)j * dim,
                                                        Kc, dim)
                    : NULL;
                if (domain_why) {
                    vm_geometric_raise(vm, "geodesic-attention-forward",
                                       domain_why, Kc);
                    return;
                }
                const char* why = eshkol_rm_distance(q->data + (int64_t)i * dim,
                                                     k->data + (int64_t)j * dim,
                                                     Kc, dim, &dist);
                if (why) { vm_geometric_raise(vm, "geodesic-attention-forward", why, Kc); return; }
                row[j] = dist;
                if (dist < min_dist) min_dist = dist;
            }
            double wsum = 0.0;
            for (int j = 0; j < nk; j++) {
                double shifted = -(row[j] - min_dist) * scale;
                if (isnan(shifted) || shifted > 0.0) {
                    vm_geometric_raise(vm, "geodesic-attention-forward",
                                        "the shifted score is not finite", Kc);
                    return;
                }
                row[j] = exp(shifted);
                wsum += row[j];
            }
            if (!(wsum > 0.0) || !isfinite(wsum)) {
                vm_geometric_raise(vm, "geodesic-attention-forward",
                                   "softmax normalisation is not finite", Kc);
                return;
            }
            for (int j = 0; j < nk; j++) {
                double w = row[j] / wsum;
                for (int d = 0; d < vdim; d++)
                    out->data[i * vdim + d] += w * values->data[j * vdim + d];
            }
        }
        VM_PUSH_TENSOR(vm, out);
        break;
    }

    case 850: { /* set-curvature!(manifold, new_curvature) */
        double c = as_number(vm_pop(vm));
        Value mv = vm_pop(vm);
        VmGeometricManifold* m = vm_geometric_manifold(vm, mv);
        if (m) { m->curvature = c; vm_push(vm, mv); }
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 852: { /* curvature-gradient(manifold, pairs) */
        /* Returned the plain SUM of the tensor's elements. A sum is not a
         * derivative of anything with respect to K, and nothing in the returned
         * float showed that: it had the type and the magnitude a gradient would
         * have. This now differentiates a NAMED objective -- the total geodesic
         * distance over the supplied point pairs -- in closed form. */
        Value mv; VmGeometricManifold* m; VmTensor* pairs; int n;
        if (!vm_curvature_args(vm, &mv, &m, &pairs, &n)) { vm_push(vm, NIL_VAL); break; }
        double g = 0.0;
        const char* why = vm_curvature_objective(pairs, n, m->curvature, NULL, &g, NULL);
        if (why) { vm_geometric_raise(vm, "curvature-gradient", why, m->curvature); break; }
        vm_push_float(vm, g);
        break;
    }
    case 853: { /* transition-geometry!(manifold, target, rate) */
        double rate = as_number(vm_pop(vm));
        double target = as_number(vm_pop(vm));
        Value mv = vm_pop(vm);
        VmGeometricManifold* m = vm_geometric_manifold(vm, mv);
        if (m) {
            m->curvature = m->curvature + rate * (target - m->curvature);
            vm_push_float(vm, m->curvature);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 854: { /* manifold-interpolate(m1, m2, t) */
        double t = as_number(vm_pop(vm));
        Value m2v = vm_pop(vm), m1v = vm_pop(vm);
        VmGeometricManifold* m1 = vm_geometric_manifold(vm, m1v);
        VmGeometricManifold* m2 = vm_geometric_manifold(vm, m2v);
        if (m1 && m2) vm_push_float(vm, m1->curvature * (1.0 - t) + m2->curvature * t);
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 855: { /* curvature-hessian(manifold, pairs) */
        /* Returned the constant 0.0 for every manifold and every argument,
         * which is the assertion that the objective is AFFINE in K -- made
         * without examining an objective, and false for the geodesic distance
         * at every curvature. It is now the exact second derivative of the same
         * objective curvature-gradient differentiates. */
        Value mv; VmGeometricManifold* m; VmTensor* pairs; int n;
        if (!vm_curvature_args(vm, &mv, &m, &pairs, &n)) { vm_push(vm, NIL_VAL); break; }
        double h = 0.0;
        const char* why = vm_curvature_objective(pairs, n, m->curvature, NULL, NULL, &h);
        if (why) { vm_geometric_raise(vm, "curvature-hessian", why, m->curvature); break; }
        vm_push_float(vm, h);
        break;
    }
    case 856: { /* adaptive-curvature-step(manifold, pairs) */
        /* Was K <- K - 0.01 * sum(grad): a FIXED rate applied to a sum that was
         * not a gradient, so nothing about the step adapted to anything, and
         * the name said it did. It is now a damped Newton step on the same
         * objective the other two ops differentiate:
         *
         *   K <- K - t * L'(K) / max(L''(K), floor),
         *
         * with t halved until the new curvature keeps the same sign (crossing
         * K = 0 would change which geometry the manifold is), leaves the
         * objective evaluable for every supplied pair, and does not increase
         * it. Every one of those three conditions can fail, and when they all
         * do the op raises instead of moving K to a value it cannot justify. */
        Value mv; VmGeometricManifold* m; VmTensor* pairs; int n;
        if (!vm_curvature_args(vm, &mv, &m, &pairs, &n)) { vm_push(vm, NIL_VAL); break; }
        double K = m->curvature, L0 = 0.0, g = 0.0, h = 0.0;
        const char* why = vm_curvature_objective(pairs, n, K, &L0, &g, &h);
        if (why) { vm_geometric_raise(vm, "adaptive-curvature-step", why, K); break; }
        double heff = (h > VM_CURVATURE_NEWTON_FLOOR) ? h : VM_CURVATURE_NEWTON_FLOOR;
        double delta = g / heff;
        int accepted = 0;
        double t = 1.0;
        for (int i = 0; i < VM_CURVATURE_BACKTRACKS; i++, t *= 0.5) {
            double kn = K - t * delta;
            if (!((K < 0.0 && kn < 0.0) || (K > 0.0 && kn > 0.0))) continue;
            double ln = 0.0;
            if (vm_curvature_trial_objective(pairs, n, K, kn, &ln)) continue;
            if (!(ln <= L0)) continue;
            m->curvature = kn;
            accepted = 1;
            break;
        }
        if (!accepted) {
            /* Kept short deliberately: vm_geometric_raise appends the
             * curvature convention to a 320-byte buffer, and a reason long
             * enough to push the convention out of the message would cost the
             * reader the one fact that says which geometry K names. */
            vm_geometric_raise(vm, "adaptive-curvature-step",
                "no backtracked damped Newton step is admissible (each either "
                "flipped the sign of K, moved a point off the manifold, or "
                "raised the objective)", K);
            break;
        }
        vm_push(vm, mv);
        break;
    }
    case 857: { /* manifold-type(manifold) */
        Value mv = vm_pop(vm);
        VmGeometricManifold* m = vm_geometric_manifold(vm, mv);
        if (m) vm_push(vm, INT_VAL(m->type));
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 858: { /* manifold-dim/manifold-dimension(manifold) */
        Value mv = vm_pop(vm);
        VmGeometricManifold* m = vm_geometric_manifold(vm, mv);
        if (m) vm_push(vm, INT_VAL(m->dim));
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 859: { /* manifold-destroy!(manifold) */
        Value mv = vm_pop(vm);
        if (mv.type == VAL_MANIFOLD && is_heap_type(vm, mv, HEAP_MANIFOLD))
            vm->heap.objects[mv.as.ptr]->opaque.ptr = NULL;
        vm_push(vm, NIL_VAL);
        break;
    }

    default: {
        int nargs = vm_geometric_arity(fid);
        for (int i = 0; i < nargs; i++) vm_pop(vm);
        vm_push(vm, NIL_VAL);
        break;
    }
    }
}
