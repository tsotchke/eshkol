/**
 * @file vm_geometric.c
 * @brief VM geometric manifold dispatch — native IDs 804-859.
 *
 * When ESHKOL_GEOMETRIC_ENABLED is defined, calls semiclassical_qllm.
 * Otherwise uses a portable constant-curvature fallback allocated in the VM
 * arena. Handles are logically invalidated by manifold-destroy!; arena memory
 * remains owned by the VM region stack.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#define HEAP_MANIFOLD        30
#define HEAP_MANIFOLD_POINT  31
#define HEAP_MANIFOLD_TANGENT 32

#if defined(ESHKOL_GEOMETRIC_ENABLED)
#include <semiclassical_qllm/manifold.h>
#include <semiclassical_qllm/geodesic.h>
#include <semiclassical_qllm/hyperbolic.h>
#include <semiclassical_qllm/spherical.h>
#endif

/* Helper: convert VmTensor (f64) to float array for qllm API (VM-arena allocated) */
static float* vm_tensor_to_float(VM* vm, const VmTensor* t) {
    if (!t || t->total <= 0) return NULL;
    float* f = (float*)vm_alloc(&vm->heap.regions, (size_t)(t->total * sizeof(float)));
    if (!f) return NULL;
    for (int64_t i = 0; i < t->total; i++) f[i] = (float)t->data[i];
    return f;
}

/* Helper: get VmTensor from heap value */
static VmTensor* vm_get_tensor(VM* vm, Value v) {
    if (!is_heap_type(vm, v, HEAP_TENSOR)) return NULL;
    return (VmTensor*)vm->heap.objects[v.as.ptr]->opaque.ptr;
}

/* Helper: push a scalar float result */
static void vm_push_float(VM* vm, double val) {
    vm_push(vm, FLOAT_VAL(val));
}

/* Helper: push a manifold handle */
static void vm_push_manifold(VM* vm, void* manifold) {
    if (!manifold) { vm_push(vm, NIL_VAL); return; }
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) { vm_push(vm, NIL_VAL); return; }
    vm->heap.objects[ptr]->type = HEAP_MANIFOLD;
    vm->heap.objects[ptr]->opaque.ptr = manifold;
    vm_push(vm, (Value){.type = VAL_MANIFOLD, .as.ptr = ptr});
}

#if !defined(ESHKOL_GEOMETRIC_ENABLED)
typedef struct {
    int type;          /* 0 euclidean, 1 hyperbolic, 2 spherical, 3 product */
    int dim;
    double curvature;
} VmFallbackManifold;

static VmFallbackManifold* vm_fallback_manifold(VM* vm, Value mv) {
    if (mv.type != VAL_MANIFOLD || !is_heap_type(vm, mv, HEAP_MANIFOLD)) return NULL;
    return (VmFallbackManifold*)vm->heap.objects[mv.as.ptr]->opaque.ptr;
}

static void vm_push_fallback_manifold(VM* vm, int type, int dim, double curvature) {
    if (dim <= 0) { vm_push(vm, NIL_VAL); return; }
    VmFallbackManifold* m = (VmFallbackManifold*)vm_alloc(&vm->heap.regions, sizeof(VmFallbackManifold));
    if (!m) { vm_push(vm, NIL_VAL); return; }
    m->type = type;
    m->dim = dim;
    m->curvature = curvature;
    vm_push_manifold(vm, m);
}
#endif

static int vm_manifold_has_value(VM* vm, Value mv) {
    return mv.type == VAL_MANIFOLD && is_heap_type(vm, mv, HEAP_MANIFOLD) &&
           vm->heap.objects[mv.as.ptr]->opaque.ptr != NULL;
}

static double vm_geometric_manifold_curvature(VM* vm, Value mv, int* ok) {
    if (!vm_manifold_has_value(vm, mv)) {
        if (ok) *ok = 0;
        return 0.0;
    }
#if defined(ESHKOL_GEOMETRIC_ENABLED)
    if (ok) *ok = 1;
    return qllm_manifold_get_curvature((qllm_manifold_t*)vm->heap.objects[mv.as.ptr]->opaque.ptr);
#else
    VmFallbackManifold* m = vm_fallback_manifold(vm, mv);
    if (!m) {
        if (ok) *ok = 0;
        return 0.0;
    }
    if (ok) *ok = 1;
    return m->curvature;
#endif
}

static int vm_geometric_manifold_dim(VM* vm, Value mv) {
#if defined(ESHKOL_GEOMETRIC_ENABLED)
    (void)vm; (void)mv;
    return 0;
#else
    VmFallbackManifold* m = vm_fallback_manifold(vm, mv);
    return m ? m->dim : 0;
#endif
}

static VmTensor* vm_tensor_copy_for_geometry(VM* vm, const VmTensor* src) {
    if (!src || !src->data || src->n_dims <= 0) return NULL;
    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, src->shape, src->n_dims);
    if (!out) return NULL;
    memcpy(out->data, src->data, (size_t)src->total * sizeof(double));
    return out;
}

static VmTensor* vm_tensor_linear_combo_for_geometry(VM* vm, const VmTensor* a, double as,
                                                     const VmTensor* b, double bs) {
    if (!a || !b || !a->data || !b->data || a->total != b->total) return NULL;
    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, a->shape, a->n_dims);
    if (!out) return NULL;
    for (int64_t i = 0; i < out->total; i++) out->data[i] = as * a->data[i] + bs * b->data[i];
    return out;
}

static VmTensor* vm_tensor_scale_for_geometry(VM* vm, const VmTensor* src, double scale) {
    if (!src || !src->data) return NULL;
    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, src->shape, src->n_dims);
    if (!out) return NULL;
    for (int64_t i = 0; i < out->total; i++) out->data[i] = src->data[i] * scale;
    return out;
}

static double vm_tensor_dot_for_geometry(const VmTensor* a, const VmTensor* b) {
    if (!a || !b || !a->data || !b->data || a->total != b->total) return 0.0;
    double sum = 0.0;
    for (int64_t i = 0; i < a->total; i++) sum += a->data[i] * b->data[i];
    return sum;
}

static double vm_tensor_distance_for_geometry(const VmTensor* a, const VmTensor* b) {
    if (!a || !b || !a->data || !b->data || a->total != b->total) return 0.0;
    double sum = 0.0;
    for (int64_t i = 0; i < a->total; i++) {
        double d = a->data[i] - b->data[i];
        sum += d * d;
    }
    return sqrt(sum);
}

static void vm_tensor_normalize_for_geometry(VmTensor* t) {
    if (!t || !t->data) return;
    double norm2 = 0.0;
    for (int64_t i = 0; i < t->total; i++) norm2 += t->data[i] * t->data[i];
    if (norm2 <= 0.0) return;
    double inv = 1.0 / sqrt(norm2);
    for (int64_t i = 0; i < t->total; i++) t->data[i] *= inv;
}

static void vm_push_tensor_handle_for_geometry(VM* vm, VmTensor* t) {
    if (!t) { vm_push(vm, NIL_VAL); return; }
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) { vm->error = 1; vm_push(vm, NIL_VAL); return; }
    vm->heap.objects[ptr]->type = HEAP_TENSOR;
    vm->heap.objects[ptr]->opaque.ptr = t;
    vm_push(vm, (Value){.type = VAL_TENSOR, .as.ptr = ptr});
}

static void vm_push_tensor_or_nil(VM* vm, VmTensor* t) {
    vm_push_tensor_handle_for_geometry(vm, t);
}

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

static int vm_geometric_arity(int fid) {
    switch (fid) {
    case 804: case 806: case 808: case 823: case 824: case 825:
    case 826: case 827: case 829: case 831: case 832: case 835:
    case 851: case 857: case 858: case 859:
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
    default:
        return 0;
    }
}

#if !defined(ESHKOL_GEOMETRIC_ENABLED)
static void vm_dispatch_geometric_fallback(VM* vm, int fid) {
    switch (fid) {
    case 804: { /* make-euclidean-manifold(dim) */
        int dim = (int)as_number(vm_pop(vm));
        vm_push_fallback_manifold(vm, 0, dim, 0.0);
        break;
    }
    case 805: { /* make-hyperbolic-manifold(dim, curvature) */
        double c = as_number(vm_pop(vm));
        int dim = (int)as_number(vm_pop(vm));
        vm_push_fallback_manifold(vm, 1, dim, c);
        break;
    }
    case 806: { /* make-spherical-manifold(dim) */
        int dim = (int)as_number(vm_pop(vm));
        vm_push_fallback_manifold(vm, 2, dim, 1.0);
        break;
    }
    case 807: { /* make-product-manifold(m1, m2) */
        Value m2v = vm_pop(vm), m1v = vm_pop(vm);
        VmFallbackManifold* m1 = vm_fallback_manifold(vm, m1v);
        VmFallbackManifold* m2 = vm_fallback_manifold(vm, m2v);
        if (m1 && m2) vm_push_fallback_manifold(vm, 3, m1->dim + m2->dim,
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
        (void)as_number(vm_pop(vm));
        VmTensor* tangent = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* base = vm_get_tensor(vm, vm_pop(vm));
        vm_push_tensor_or_nil(vm, vm_tensor_linear_combo_for_geometry(vm, base, 1.0, tangent, 1.0));
        break;
    }
    case 810: case 822: { /* log-map(base, point, curvature) / spherical-log(base, point) */
        if (fid == 810) (void)as_number(vm_pop(vm));
        VmTensor* point = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* base = vm_get_tensor(vm, vm_pop(vm));
        vm_push_tensor_or_nil(vm, vm_tensor_linear_combo_for_geometry(vm, point, 1.0, base, -1.0));
        break;
    }
    case 811: case 816: { /* geodesic-distance/poincare-distance(x, y, curvature) */
        (void)as_number(vm_pop(vm));
        VmTensor* y = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        if (x && y && x->total == y->total) vm_push_float(vm, vm_tensor_distance_for_geometry(x, y));
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 812: case 843: { /* parallel/vector transport(x, y, v, curvature) */
        (void)as_number(vm_pop(vm));
        VmTensor* v = vm_get_tensor(vm, vm_pop(vm));
        (void)vm_pop(vm); /* y */
        (void)vm_pop(vm); /* x */
        vm_push_tensor_or_nil(vm, vm_tensor_copy_for_geometry(vm, v));
        break;
    }
    case 813: { /* manifold-project(x, curvature) */
        (void)as_number(vm_pop(vm));
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        vm_push_tensor_or_nil(vm, vm_tensor_copy_for_geometry(vm, x));
        break;
    }
    case 814: { /* mobius-add(x, y, curvature) */
        (void)as_number(vm_pop(vm));
        VmTensor* y = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        vm_push_tensor_or_nil(vm, vm_tensor_linear_combo_for_geometry(vm, x, 1.0, y, 1.0));
        break;
    }
    case 815: { /* mobius-scalar-mul(r, x, curvature) */
        (void)as_number(vm_pop(vm));
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        double r = as_number(vm_pop(vm));
        vm_push_tensor_or_nil(vm, vm_tensor_scale_for_geometry(vm, x, r));
        break;
    }
    case 817: { /* frechet-mean(points, weights, curvature) */
        (void)as_number(vm_pop(vm));
        VmTensor* weights = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* points = vm_get_tensor(vm, vm_pop(vm));
        if (!points || !points->data) { vm_push(vm, NIL_VAL); break; }
        int n = (points->n_dims >= 2) ? (int)points->shape[0] : 1;
        int dim = (points->n_dims >= 2) ? (int)points->shape[1] : (int)points->total;
        int64_t shape[1] = {dim};
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
        if (!out) { vm_push(vm, NIL_VAL); break; }
        double wsum = 0.0;
        for (int i = 0; i < n; i++) {
            double w = (weights && i < weights->total) ? weights->data[i] : 1.0;
            wsum += w;
            for (int d = 0; d < dim; d++) out->data[d] += w * points->data[i * dim + d];
        }
        if (wsum != 0.0) for (int d = 0; d < dim; d++) out->data[d] /= wsum;
        VM_PUSH_TENSOR(vm, out);
        break;
    }

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
        VmTensor* tangent = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* base = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* out = vm_tensor_linear_combo_for_geometry(vm, base, 1.0, tangent, 1.0);
        vm_tensor_normalize_for_geometry(out);
        vm_push_tensor_or_nil(vm, out);
        break;
    }
    case 823: { /* spherical-project(x) */
        VmTensor* x = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* out = vm_tensor_copy_for_geometry(vm, x);
        vm_tensor_normalize_for_geometry(out);
        vm_push_tensor_or_nil(vm, out);
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
        VmTensor* form = vm_get_tensor(vm, vm_pop(vm));
        if (!form) { vm_push(vm, NIL_VAL); break; }
        VmTensor* out = vm_tensor_zeros(&vm->heap.regions, form->shape, form->n_dims);
        vm_push_tensor_or_nil(vm, out);
        break;
    }
    case 836: { /* hodge-star(form, metric) */
        (void)vm_pop(vm);
        VmTensor* form = vm_get_tensor(vm, vm_pop(vm));
        vm_push_tensor_or_nil(vm, vm_tensor_copy_for_geometry(vm, form));
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
        (void)as_number(vm_pop(vm));
        double lr = as_number(vm_pop(vm));
        VmTensor* grad = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* point = vm_get_tensor(vm, vm_pop(vm));
        vm_push_tensor_or_nil(vm, vm_tensor_linear_combo_for_geometry(vm, point, 1.0, grad, -lr));
        break;
    }
    case 840: { /* riemannian-adam-step(point, gradient, lr, beta1, beta2, curvature) */
        (void)as_number(vm_pop(vm));
        (void)as_number(vm_pop(vm));
        (void)as_number(vm_pop(vm));
        double lr = as_number(vm_pop(vm));
        VmTensor* grad = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* point = vm_get_tensor(vm, vm_pop(vm));
        vm_push_tensor_or_nil(vm, vm_tensor_linear_combo_for_geometry(vm, point, 1.0, grad, -lr));
        break;
    }
    case 841: { /* riemannian-grad(euclidean_grad, point, curvature) */
        (void)as_number(vm_pop(vm));
        (void)vm_pop(vm);
        VmTensor* grad = vm_get_tensor(vm, vm_pop(vm));
        vm_push_tensor_or_nil(vm, vm_tensor_copy_for_geometry(vm, grad));
        break;
    }

    case 844: { /* geodesic-attention-scores(Q, K, curvature) */
        (void)as_number(vm_pop(vm));
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
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < nk; j++) {
                double sum = 0.0;
                for (int d = 0; d < qdim; d++) {
                    double diff = q->data[i * qdim + d] - k->data[j * kdim + d];
                    sum += diff * diff;
                }
                out->data[i * nk + j] = -sqrt(sum);
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
        (void)as_number(vm_pop(vm));
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
        for (int i = 0; i < nq; i++) {
            double wsum = 0.0;
            for (int j = 0; j < nk; j++) {
                double dist2 = 0.0;
                for (int d = 0; d < dim; d++) {
                    double diff = q->data[i * dim + d] - k->data[j * dim + d];
                    dist2 += diff * diff;
                }
                double w = exp(-sqrt(dist2));
                wsum += w;
                for (int d = 0; d < vdim; d++) out->data[i * vdim + d] += w * values->data[j * vdim + d];
            }
            if (wsum != 0.0)
                for (int d = 0; d < vdim; d++) out->data[i * vdim + d] /= wsum;
        }
        VM_PUSH_TENSOR(vm, out);
        break;
    }

    case 850: { /* set-curvature!(manifold, new_curvature) */
        double c = as_number(vm_pop(vm));
        Value mv = vm_pop(vm);
        VmFallbackManifold* m = vm_fallback_manifold(vm, mv);
        if (m) { m->curvature = c; vm_push(vm, mv); }
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 852: { /* curvature-gradient(manifold, loss_grad) */
        VmTensor* grad = vm_get_tensor(vm, vm_pop(vm));
        Value mv = vm_pop(vm);
        if (!vm_fallback_manifold(vm, mv) || !grad) { vm_push(vm, NIL_VAL); break; }
        double sum = 0.0;
        for (int64_t i = 0; i < grad->total; i++) sum += grad->data[i];
        vm_push_float(vm, sum);
        break;
    }
    case 853: { /* transition-geometry!(manifold, target, rate) */
        double rate = as_number(vm_pop(vm));
        double target = as_number(vm_pop(vm));
        Value mv = vm_pop(vm);
        VmFallbackManifold* m = vm_fallback_manifold(vm, mv);
        if (m) {
            m->curvature = m->curvature + rate * (target - m->curvature);
            vm_push_float(vm, m->curvature);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 854: { /* manifold-interpolate(m1, m2, t) */
        double t = as_number(vm_pop(vm));
        Value m2v = vm_pop(vm), m1v = vm_pop(vm);
        VmFallbackManifold* m1 = vm_fallback_manifold(vm, m1v);
        VmFallbackManifold* m2 = vm_fallback_manifold(vm, m2v);
        if (m1 && m2) vm_push_float(vm, m1->curvature * (1.0 - t) + m2->curvature * t);
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 855: { /* curvature-hessian(manifold, grad) */
        (void)vm_pop(vm);
        Value mv = vm_pop(vm);
        if (vm_fallback_manifold(vm, mv)) vm_push_float(vm, 0.0);
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 856: { /* adaptive-curvature-step(manifold, grad) */
        VmTensor* grad = vm_get_tensor(vm, vm_pop(vm));
        Value mv = vm_pop(vm);
        VmFallbackManifold* m = vm_fallback_manifold(vm, mv);
        if (m && grad) {
            double sum = 0.0;
            for (int64_t i = 0; i < grad->total; i++) sum += grad->data[i];
            m->curvature -= 0.01 * sum;
            vm_push(vm, mv);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 857: { /* manifold-type(manifold) */
        Value mv = vm_pop(vm);
        VmFallbackManifold* m = vm_fallback_manifold(vm, mv);
        if (m) vm_push(vm, INT_VAL(m->type));
        else vm_push(vm, NIL_VAL);
        break;
    }
    case 858: { /* manifold-dim/manifold-dimension(manifold) */
        Value mv = vm_pop(vm);
        VmFallbackManifold* m = vm_fallback_manifold(vm, mv);
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
#endif

static void vm_dispatch_geometric(VM* vm, int fid) {
#if defined(ESHKOL_GEOMETRIC_ENABLED)
    switch (fid) {

    /* ═══ Manifold creation (804-808) ═══ */
    case 804: { /* make-euclidean-manifold(dim) */
        int dim = (int)as_number(vm_pop(vm));
        qllm_manifold_options_t opts = {0}; opts.curvature = 0;
        vm_push_manifold(vm, qllm_manifold_euclidean_create(dim, &opts));
        break;
    }
    case 805: { /* make-hyperbolic-manifold(dim, curvature) */
        float c = (float)as_number(vm_pop(vm));
        int dim = (int)as_number(vm_pop(vm));
        qllm_manifold_options_t opts = {0}; opts.curvature = c;
        vm_push_manifold(vm, qllm_manifold_hyperbolic_create(dim, &opts));
        break;
    }
    case 806: { /* make-spherical-manifold(dim) */
        int dim = (int)as_number(vm_pop(vm));
        qllm_manifold_options_t opts = {0}; opts.curvature = 1.0f;
        vm_push_manifold(vm, qllm_manifold_spherical_create(dim, &opts));
        break;
    }
    case 807: { /* make-product-manifold(m1, m2) */
        Value m2v = vm_pop(vm), m1v = vm_pop(vm);
        if (is_heap_type(vm, m1v, HEAP_MANIFOLD) && is_heap_type(vm, m2v, HEAP_MANIFOLD)) {
            qllm_manifold_t* ms[2] = {
                (qllm_manifold_t*)vm->heap.objects[m1v.as.ptr]->opaque.ptr,
                (qllm_manifold_t*)vm->heap.objects[m2v.as.ptr]->opaque.ptr
            };
            vm_push_manifold(vm, qllm_manifold_product_create(ms, 2));
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 808: { /* manifold-curvature(m) */
        Value mv = vm_pop(vm);
        if (is_heap_type(vm, mv, HEAP_MANIFOLD))
            vm_push_float(vm, qllm_manifold_get_curvature(
                (qllm_manifold_t*)vm->heap.objects[mv.as.ptr]->opaque.ptr));
        else vm_push(vm, NIL_VAL);
        break;
    }

    /* ═══ Core manifold ops (805-809) ═══ */
    case 809: { /* exp-map(base_tensor, tangent_tensor, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* tv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* bv = vm_get_tensor(vm, vm_pop(vm));
        if (bv && tv) {
            float* bf = vm_tensor_to_float(vm, bv);
            float* tf = vm_tensor_to_float(vm, tv);
            int n = (int)bv->total;
            qllm_tensor_t* result = qllm_hyperbolic_exp_map(bf, tf, n, c);
            if (result) {
                /* Convert result back to VmTensor */
                float* rd = (float*)qllm_tensor_get_data(result);
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, bv->shape, bv->n_dims);
                if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 810: { /* log-map(base_tensor, point_tensor, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* bv = vm_get_tensor(vm, vm_pop(vm));
        if (bv && pv) {
            float* bf = vm_tensor_to_float(vm, bv);
            float* pf = vm_tensor_to_float(vm, pv);
            int n = (int)bv->total;
            qllm_tensor_t* result = qllm_hyperbolic_log_map(bf, pf, n, c);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, bv->shape, bv->n_dims);
                if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 811: { /* geodesic-distance(x_tensor, y_tensor, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv && xv->total == yv->total) {
            float* xf = vm_tensor_to_float(vm, xv);
            float* yf = vm_tensor_to_float(vm, yv);
            float dist = qllm_hyperbolic_distance(xf, yf, (int)xv->total, c);
            vm_push_float(vm, dist);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 812: { /* parallel-transport(x, y, v, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* vv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv && vv) {
            float* xf = vm_tensor_to_float(vm, xv);
            float* yf = vm_tensor_to_float(vm, yv);
            float* vf = vm_tensor_to_float(vm, vv);
            qllm_tensor_t* result = qllm_hyperbolic_parallel_transport(xf, yf, vf, (int)xv->total, c);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, xv->shape, xv->n_dims);
                if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 813: { /* project(x_tensor, curvature) — project onto manifold */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv) {
            float* xf = vm_tensor_to_float(vm, xv);
            qllm_tensor_t* result = qllm_hyperbolic_project(xf, (int)xv->total, c);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, xv->shape, xv->n_dims);
                if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ═══ Hyperbolic operations (810-814) ═══ */
    case 814: { /* mobius-add(x, y, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv) {
            float* xf = vm_tensor_to_float(vm, xv);
            float* yf = vm_tensor_to_float(vm, yv);
            qllm_tensor_t* result = qllm_hyperbolic_mobius_add(xf, yf, (int)xv->total, c);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, xv->shape, xv->n_dims);
                if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 815: { /* mobius-scalar-mul(r, x, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        float r = (float)as_number(vm_pop(vm));
        if (xv) {
            float* xf = vm_tensor_to_float(vm, xv);
            qllm_tensor_t* result = qllm_hyperbolic_mobius_scalar(r, xf, (int)xv->total, c);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, xv->shape, xv->n_dims);
                if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 816: { /* poincare-distance(x, y, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv && xv->total == yv->total) {
            float* xf = vm_tensor_to_float(vm, xv);
            float* yf = vm_tensor_to_float(vm, yv);
            float dist = qllm_hyperbolic_distance(xf, yf, (int)xv->total, c);
            vm_push_float(vm, dist);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 817: { /* frechet-mean(points_tensor, weights_tensor, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* wv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));
        if (pv && wv) {
            float* pf = vm_tensor_to_float(vm, pv);
            float* wf = vm_tensor_to_float(vm, wv);
            int dim = (pv->n_dims >= 2) ? (int)pv->shape[1] : (int)pv->total;
            int n_points = (pv->n_dims >= 2) ? (int)pv->shape[0] : 1;
            qllm_tensor_t* result = qllm_hyperbolic_frechet_mean(pf, wf, n_points, dim, c, 100, 1e-6f);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                int64_t shape[1] = {dim};
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
                if (out && rd) for (int i = 0; i < dim; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ═══ Spherical operations (815-819) ═══ */
    case 819: { /* great-circle-distance(x, y) */
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv && xv->total == yv->total) {
            float* xf = vm_tensor_to_float(vm, xv);
            float* yf = vm_tensor_to_float(vm, yv);
            float dist = qllm_spherical_distance(xf, yf, (int)xv->total);
            vm_push_float(vm, dist);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 820: { /* slerp(x, y, t) */
        float t = (float)as_number(vm_pop(vm));
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv) {
            float* xf = vm_tensor_to_float(vm, xv);
            float* yf = vm_tensor_to_float(vm, yv);
            qllm_tensor_t* result = qllm_spherical_slerp(xf, yf, (int)xv->total, t);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, xv->shape, xv->n_dims);
                if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 821: { /* spherical-exp(base, tangent) */
        VmTensor* tv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* bv = vm_get_tensor(vm, vm_pop(vm));
        if (bv && tv) {
            float* bf = vm_tensor_to_float(vm, bv);
            float* tf = vm_tensor_to_float(vm, tv);
            qllm_tensor_t* result = qllm_spherical_exp_map(bf, tf, (int)bv->total);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, bv->shape, bv->n_dims);
                if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 822: { /* spherical-log(base, point) */
        VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* bv = vm_get_tensor(vm, vm_pop(vm));
        if (bv && pv) {
            float* bf = vm_tensor_to_float(vm, bv);
            float* pf = vm_tensor_to_float(vm, pv);
            qllm_tensor_t* result = qllm_spherical_log_map(bf, pf, (int)bv->total);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, bv->shape, bv->n_dims);
                if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 823: { /* spherical-project(x) — project onto unit sphere */
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv) {
            float* xf = vm_tensor_to_float(vm, xv);
            qllm_tensor_t* result = qllm_spherical_project(xf, (int)xv->total);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, xv->shape, xv->n_dims);
                if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ═══ Lie group operations (820-824) ═══ */
    case 824: { /* so3-exp(omega_tensor) — axis-angle → rotation quaternion */
        VmTensor* omega = vm_get_tensor(vm, vm_pop(vm));
        if (omega && omega->total >= 3) {
            float* of = vm_tensor_to_float(vm, omega);
            qllm_so3_algebra_t alg = {{of[0], of[1], of[2]}};
            qllm_so3_t rot = qllm_so3_exp(&alg);
            /* Return quaternion as 4-element tensor */
            int64_t shape[1] = {4};
            VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
            if (out) { out->data[0]=rot.w; out->data[1]=rot.x; out->data[2]=rot.y; out->data[3]=rot.z; }
            if (out) { VM_PUSH_TENSOR(vm, out); break; }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 825: { /* so3-log(quat_tensor) — rotation quaternion → axis-angle */
        VmTensor* qv = vm_get_tensor(vm, vm_pop(vm));
        if (qv && qv->total >= 4) {
            float* qf = vm_tensor_to_float(vm, qv);
            qllm_so3_t rot = {qf[0], qf[1], qf[2], qf[3]};
            qllm_so3_algebra_t alg = qllm_so3_log(&rot);
            int64_t shape[1] = {3};
            VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
            if (out) { out->data[0]=alg.omega[0]; out->data[1]=alg.omega[1]; out->data[2]=alg.omega[2]; }
            if (out) { VM_PUSH_TENSOR(vm, out); break; }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 826: { /* se3-exp(twist_tensor) — twist → rigid transform */
        VmTensor* tv = vm_get_tensor(vm, vm_pop(vm));
        if (tv && tv->total >= 6) {
            float* tf = vm_tensor_to_float(vm, tv);
            qllm_se3_algebra_t twist;
            twist.omega.omega[0]=tf[0]; twist.omega.omega[1]=tf[1]; twist.omega.omega[2]=tf[2];
            twist.v[0]=tf[3]; twist.v[1]=tf[4]; twist.v[2]=tf[5];
            qllm_se3_t pose = qllm_se3_exp(&twist);
            /* Return as 7-element tensor: quat(4) + translation(3) */
            int64_t shape[1] = {7};
            VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
            if (out) {
                out->data[0]=pose.rotation.w; out->data[1]=pose.rotation.x;
                out->data[2]=pose.rotation.y; out->data[3]=pose.rotation.z;
                out->data[4]=pose.translation[0]; out->data[5]=pose.translation[1]; out->data[6]=pose.translation[2];
            }
            if (out) { VM_PUSH_TENSOR(vm, out); break; }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 827: { /* se3-log(pose_tensor) — rigid transform → twist */
        VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));
        if (pv && pv->total >= 7) {
            float* pf = vm_tensor_to_float(vm, pv);
            qllm_se3_t pose;
            pose.rotation = (qllm_so3_t){pf[0],pf[1],pf[2],pf[3]};
            pose.translation[0]=pf[4]; pose.translation[1]=pf[5]; pose.translation[2]=pf[6];
            qllm_se3_algebra_t twist = qllm_se3_log(&pose);
            int64_t shape[1] = {6};
            VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
            if (out) {
                out->data[0]=twist.omega.omega[0]; out->data[1]=twist.omega.omega[1]; out->data[2]=twist.omega.omega[2];
                out->data[3]=twist.v[0]; out->data[4]=twist.v[1]; out->data[5]=twist.v[2];
            }
            if (out) { VM_PUSH_TENSOR(vm, out); break; }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 828: { /* quaternion-mul(q1, q2) — Hamilton product */
        VmTensor* q2v = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* q1v = vm_get_tensor(vm, vm_pop(vm));
        if (q1v && q2v && q1v->total >= 4 && q2v->total >= 4) {
            float* q1f = vm_tensor_to_float(vm, q1v);
            float* q2f = vm_tensor_to_float(vm, q2v);
            qllm_so3_t r1 = {q1f[0],q1f[1],q1f[2],q1f[3]};
            qllm_so3_t r2 = {q2f[0],q2f[1],q2f[2],q2f[3]};
            qllm_so3_t result = qllm_so3_compose(&r1, &r2);
            int64_t shape[1] = {4};
            VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
            if (out) { out->data[0]=result.w; out->data[1]=result.x; out->data[2]=result.y; out->data[3]=result.z; }
            if (out) { VM_PUSH_TENSOR(vm, out); break; }
        }
        vm_push(vm, NIL_VAL); break;
    }

    /* ═══ Differential geometry (825-829) ═══ */
    case 829: { /* metric-tensor(manifold) — get metric at origin */
        Value mv = vm_pop(vm);
        if (is_heap_type(vm, mv, HEAP_MANIFOLD)) {
            qllm_manifold_t* m = (qllm_manifold_t*)vm->heap.objects[mv.as.ptr]->opaque.ptr;
            float curv = qllm_manifold_get_curvature(m);
            vm_push_float(vm, curv); /* Return curvature as scalar (full metric tensor requires dimension) */
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 830: { /* christoffel(manifold, point) — connection coefficients */
        vm_geometric_christoffel_tensor(vm);
        break;
    }
    case 831: { /* riemann-curvature(manifold) */
        Value mv = vm_pop(vm);
        if (is_heap_type(vm, mv, HEAP_MANIFOLD)) {
            float curv = qllm_manifold_get_curvature(
                (qllm_manifold_t*)vm->heap.objects[mv.as.ptr]->opaque.ptr);
            vm_push_float(vm, curv);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 832: { /* ricci-scalar(manifold) */
        Value mv = vm_pop(vm);
        if (is_heap_type(vm, mv, HEAP_MANIFOLD)) {
            /* Ricci scalar = n*(n-1)*K for constant curvature K */
            qllm_manifold_t* m = (qllm_manifold_t*)vm->heap.objects[mv.as.ptr]->opaque.ptr;
            float K = qllm_manifold_get_curvature(m);
            /* Approximate: R = dim*(dim-1)*K */
            vm_push_float(vm, K * 36.0f * 35.0f); /* D*(D-1)*K */
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 833: { /* sectional-curvature(manifold, u, v) */
        vm_pop(vm); vm_pop(vm); /* u, v vectors */
        Value mv = vm_pop(vm);
        if (is_heap_type(vm, mv, HEAP_MANIFOLD)) {
            float K = qllm_manifold_get_curvature(
                (qllm_manifold_t*)vm->heap.objects[mv.as.ptr]->opaque.ptr);
            vm_push_float(vm, K); /* Constant curvature → sectional = K */
        } else vm_push(vm, NIL_VAL);
        break;
    }

    /* ═══ Differential forms (830-834) ═══ */
    case 834: { /* wedge-product(form_a, form_b) */
        VmTensor* bv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* av = vm_get_tensor(vm, vm_pop(vm));
        if (av && bv) {
            /* Create forms from tensors */
            int dim = (int)(av->total > bv->total ? av->total : bv->total);
            qllm_differential_form_t* alpha = qllm_form_create(dim, 1);
            qllm_differential_form_t* beta = qllm_form_create(dim, 1);
            if (alpha && beta) {
                /* Set coefficients from tensor data */
                for (int64_t i = 0; i < av->total; i++)
                    qllm_form_set_component(alpha, (int)i, (float)av->data[i]);
                for (int64_t i = 0; i < bv->total; i++)
                    qllm_form_set_component(beta, (int)i, (float)bv->data[i]);
                qllm_differential_form_t* result = qllm_form_wedge(alpha, beta);
                if (result) {
                    int rsize = qllm_form_num_components(result);
                    int64_t shape[1] = {rsize};
                    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
                    if (out) {
                        for (int i = 0; i < rsize; i++)
                            out->data[i] = qllm_form_get_component(result, i);
                        VM_PUSH_TENSOR(vm, out);
                        qllm_form_destroy(result);
                        qllm_form_destroy(alpha);
                        qllm_form_destroy(beta);
                        break;
                    }
                    qllm_form_destroy(result);
                }
                qllm_form_destroy(alpha);
                qllm_form_destroy(beta);
            }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 835: { /* exterior-derivative(form) */
        VmTensor* fv = vm_get_tensor(vm, vm_pop(vm));
        if (fv) {
            int dim = (int)fv->total;
            qllm_differential_form_t* form = qllm_form_create(dim, 1);
            if (form) {
                for (int64_t i = 0; i < fv->total; i++)
                    qllm_form_set_component(form, (int)i, (float)fv->data[i]);
                qllm_differential_form_t* result = qllm_form_exterior_derivative(form);
                if (result) {
                    int rsize = qllm_form_num_components(result);
                    int64_t shape[1] = {rsize > 0 ? rsize : 1};
                    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
                    if (out) {
                        for (int i = 0; i < rsize; i++)
                            out->data[i] = qllm_form_get_component(result, i);
                        VM_PUSH_TENSOR(vm, out);
                        qllm_form_destroy(result);
                        qllm_form_destroy(form);
                        break;
                    }
                    qllm_form_destroy(result);
                }
                qllm_form_destroy(form);
            }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 836: { /* hodge-star(form, metric) */
        vm_pop(vm); /* metric */
        VmTensor* fv = vm_get_tensor(vm, vm_pop(vm));
        if (fv) {
            int dim = (int)fv->total;
            qllm_differential_form_t* form = qllm_form_create(dim, 1);
            if (form) {
                for (int64_t i = 0; i < fv->total; i++)
                    qllm_form_set_component(form, (int)i, (float)fv->data[i]);
                qllm_differential_form_t* result = qllm_form_hodge_star(form, dim);
                if (result) {
                    int rsize = qllm_form_num_components(result);
                    int64_t shape[1] = {rsize > 0 ? rsize : 1};
                    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
                    if (out) {
                        for (int i = 0; i < rsize; i++)
                            out->data[i] = qllm_form_get_component(result, i);
                        VM_PUSH_TENSOR(vm, out);
                        qllm_form_destroy(result);
                        qllm_form_destroy(form);
                        break;
                    }
                    qllm_form_destroy(result);
                }
                qllm_form_destroy(form);
            }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 837: { /* interior-product(vector, form) */
        VmTensor* fv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* vv = vm_get_tensor(vm, vm_pop(vm));
        if (vv && fv) {
            int dim = (int)vv->total;
            qllm_differential_form_t* form = qllm_form_create(dim, 2);
            if (form) {
                for (int64_t i = 0; i < fv->total; i++)
                    qllm_form_set_component(form, (int)i, (float)fv->data[i]);
                float vec[64]; for (int i = 0; i < dim && i < 64; i++) vec[i] = (float)vv->data[i];
                qllm_differential_form_t* result = qllm_form_interior_product(form, vec);
                if (result) {
                    int rsize = qllm_form_num_components(result);
                    int64_t shape[1] = {rsize > 0 ? rsize : 1};
                    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
                    if (out) for (int i = 0; i < rsize; i++) out->data[i] = qllm_form_get_component(result, i);
                    qllm_form_destroy(result); qllm_form_destroy(form);
                    if (out) { VM_PUSH_TENSOR(vm, out); break; }
                }
                qllm_form_destroy(form);
            }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 838: { /* pullback(form, jacobian) — 2 args */
        vm_geometric_pullback_tensor(vm);
        break;
    }

    /* ═══ Riemannian optimization (835-839) ═══ */
    case 839: { /* riemannian-sgd-step(point, gradient, lr, curvature) */
        float c = (float)as_number(vm_pop(vm));
        float lr = (float)as_number(vm_pop(vm));
        VmTensor* gv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));
        if (pv && gv) {
            float* pf = vm_tensor_to_float(vm, pv);
            float* gf = vm_tensor_to_float(vm, gv);
            int n = (int)pv->total;
            /* Riemannian SGD: retract(-lr * grad) from point */
            float* neg_scaled = (float*)vm_alloc(&vm->heap.regions, n * sizeof(float));
            if (neg_scaled) {
                for (int i = 0; i < n; i++) neg_scaled[i] = -lr * gf[i];
                qllm_tensor_t* result = qllm_hyperbolic_exp_map(pf, neg_scaled, n, c);
                if (result) {
                    float* rd = (float*)qllm_tensor_get_data(result);
                    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, pv->shape, pv->n_dims);
                    if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                    qllm_tensor_destroy(result);
                    if (out) { VM_PUSH_TENSOR(vm, out); break; }
                }
            }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 840: { /* riemannian-adam-step(point, gradient, lr, beta1, beta2, curvature) */
        float c = (float)as_number(vm_pop(vm));
        float b2 = (float)as_number(vm_pop(vm));
        float b1 = (float)as_number(vm_pop(vm));
        float lr = (float)as_number(vm_pop(vm));
        VmTensor* gv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));
        if (pv && gv) {
            /* Adam step: retract(-lr * grad) with momentum (simplified) */
            float* pf = vm_tensor_to_float(vm, pv);
            float* gf = vm_tensor_to_float(vm, gv);
            int n = (int)pv->total;
            float* step = (float*)vm_alloc(&vm->heap.regions, n * sizeof(float));
            if (step) {
                (void)b1; (void)b2; /* Full Adam needs state — use SGD as fallback */
                for (int i = 0; i < n; i++) step[i] = -lr * gf[i];
                qllm_tensor_t* result = qllm_hyperbolic_exp_map(pf, step, n, c);
                if (result) {
                    float* rd = (float*)qllm_tensor_get_data(result);
                    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, pv->shape, pv->n_dims);
                    if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                    qllm_tensor_destroy(result);
                    if (out) { VM_PUSH_TENSOR(vm, out); break; }
                }
            }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 841: { /* riemannian-grad(euclidean_grad, point, curvature) — project to tangent space */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* gv = vm_get_tensor(vm, vm_pop(vm));
        if (gv && pv) {
            /* Riemannian gradient = conformal_factor^2 * euclidean_gradient */
            float* pf = vm_tensor_to_float(vm, pv);
            float cf = qllm_hyperbolic_conformal_factor(pf, (int)pv->total, c);
            VmTensor* out = vm_tensor_zeros(&vm->heap.regions, gv->shape, gv->n_dims);
            if (out) {
                float scale = cf * cf;
                for (int64_t i = 0; i < out->total; i++) out->data[i] = gv->data[i] * scale;
                VM_PUSH_TENSOR(vm, out); break;
            }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 842: { /* retraction(base, tangent, curvature) — exp map */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* tv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* bv = vm_get_tensor(vm, vm_pop(vm));
        if (bv && tv) {
            float* bf = vm_tensor_to_float(vm, bv);
            float* tf = vm_tensor_to_float(vm, tv);
            qllm_tensor_t* result = qllm_hyperbolic_exp_map(bf, tf, (int)bv->total, c);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, bv->shape, bv->n_dims);
                if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 843: { /* vector-transport(x, y, v, curvature) — parallel transport */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* vv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv && vv) {
            float* xf = vm_tensor_to_float(vm, xv);
            float* yf = vm_tensor_to_float(vm, yv);
            float* vf = vm_tensor_to_float(vm, vv);
            qllm_tensor_t* result = qllm_hyperbolic_parallel_transport(xf, yf, vf, (int)xv->total, c);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, xv->shape, xv->n_dims);
                if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL); break;
    }

    /* ═══ Geodesic attention (844-849) ═══ */
    case 844: { /* geodesic-attention-scores(Q, K, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* kv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* qv = vm_get_tensor(vm, vm_pop(vm));
        if (qv && kv) {
            float* qf = vm_tensor_to_float(vm, qv);
            float* kf = vm_tensor_to_float(vm, kv);
            /* Compute pairwise hyperbolic distances as attention scores */
            int n_q = (qv->n_dims >= 2) ? (int)qv->shape[0] : 1;
            int n_k = (kv->n_dims >= 2) ? (int)kv->shape[0] : 1;
            int dim = (qv->n_dims >= 2) ? (int)qv->shape[1] : (int)qv->total;
            int64_t shape[2] = {n_q, n_k};
            VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 2);
            if (out) {
                for (int i = 0; i < n_q; i++)
                    for (int j = 0; j < n_k; j++) {
                        float d = qllm_hyperbolic_distance(qf + i*dim, kf + j*dim, dim, c);
                        out->data[i * n_k + j] = -d; /* negative distance as attention score */
                    }
                VM_PUSH_TENSOR(vm, out); break;
            }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 845: { /* geodesic-attention-values(scores, V, curvature) — weighted Fréchet mean */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* vv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* sv = vm_get_tensor(vm, vm_pop(vm));
        if (sv && vv && vv->n_dims >= 2) {
            int n = (int)vv->shape[0], dim = (int)vv->shape[1];
            float* vf = vm_tensor_to_float(vm, vv);
            float* sf = vm_tensor_to_float(vm, sv);
            /* Weighted Fréchet mean of value vectors */
            qllm_tensor_t* result = qllm_hyperbolic_frechet_mean(vf, sf, n, dim, c, 50, 1e-5f);
            if (result) {
                float* rd = (float*)qllm_tensor_get_data(result);
                int64_t shape[1] = {dim};
                VmTensor* out = vm_tensor_zeros(&vm->heap.regions, shape, 1);
                if (out && rd) for (int i = 0; i < dim; i++) out->data[i] = rd[i];
                qllm_tensor_destroy(result);
                if (out) { VM_PUSH_TENSOR(vm, out); break; }
            }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 846: { /* curvature-softmax(scores, curvature) — curvature-scaled softmax */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* sv = vm_get_tensor(vm, vm_pop(vm));
        if (sv) {
            VmTensor* out = vm_tensor_zeros(&vm->heap.regions, sv->shape, sv->n_dims);
            if (out) {
                double scale = fabsf(c) > 0 ? 1.0 / sqrtf(fabsf(c)) : 1.0;
                double max_val = sv->data[0];
                for (int64_t i = 1; i < sv->total; i++) if (sv->data[i] > max_val) max_val = sv->data[i];
                double sum = 0;
                for (int64_t i = 0; i < sv->total; i++) { out->data[i] = exp((sv->data[i] - max_val) * scale); sum += out->data[i]; }
                for (int64_t i = 0; i < sv->total; i++) out->data[i] /= sum;
                VM_PUSH_TENSOR(vm, out); break;
            }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 847: { /* geodesic-attention-forward(Q, K, V, curvature) — full attention */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* vv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* kv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* qv = vm_get_tensor(vm, vm_pop(vm));
        if (qv && kv && vv && qv->n_dims >= 2 && kv->n_dims >= 2) {
            int n_q = (int)qv->shape[0], n_k = (int)kv->shape[0];
            int dim = (int)qv->shape[1];
            float* qf = vm_tensor_to_float(vm, qv);
            float* kf = vm_tensor_to_float(vm, kv);
            float* vf = vm_tensor_to_float(vm, vv);
            /* Compute distance-based attention scores */
            int64_t out_shape[2] = {n_q, dim};
            VmTensor* out = vm_tensor_zeros(&vm->heap.regions, out_shape, 2);
            if (out) {
                for (int i = 0; i < n_q; i++) {
                    float scores[256]; float mx = -1e30f;
                    for (int j = 0; j < n_k && j < 256; j++) {
                        scores[j] = -qllm_hyperbolic_distance(qf+i*dim, kf+j*dim, dim, c);
                        if (scores[j] > mx) mx = scores[j];
                    }
                    float sum = 0;
                    for (int j = 0; j < n_k; j++) { scores[j] = expf(scores[j]-mx); sum += scores[j]; }
                    for (int j = 0; j < n_k; j++) scores[j] /= sum;
                    /* Weighted sum of values */
                    for (int d = 0; d < dim; d++) {
                        double v = 0;
                        for (int j = 0; j < n_k; j++) v += scores[j] * vf[j*dim+d];
                        out->data[i*dim+d] = v;
                    }
                }
                VM_PUSH_TENSOR(vm, out); break;
            }
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 848: case 849: {
        /* attention-backward, attention-mask, multi-head variants */
        int nargs = (fid <= 845) ? 4 : 3;
        for (int i = 0; i < nargs; i++) vm_pop(vm);
        vm_push(vm, NIL_VAL); break;
    }

    /* ═══ Adaptive curvature (850-859) ═══ */
    case 850: { /* set-curvature(manifold, new_curvature) */
        float new_c = (float)as_number(vm_pop(vm));
        Value mv = vm_pop(vm);
        if (is_heap_type(vm, mv, HEAP_MANIFOLD)) {
            qllm_manifold_set_curvature(
                (qllm_manifold_t*)vm->heap.objects[mv.as.ptr]->opaque.ptr, new_c);
            vm_push(vm, mv); /* return manifold */
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 851: { /* get-curvature(manifold) */
        Value mv = vm_pop(vm);
        if (is_heap_type(vm, mv, HEAP_MANIFOLD)) {
            float c = qllm_manifold_get_curvature(
                (qllm_manifold_t*)vm->heap.objects[mv.as.ptr]->opaque.ptr);
            vm_push_float(vm, c);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 852: { /* curvature-gradient(manifold, loss_grad) — gradient of loss w.r.t. curvature */
        VmTensor* grad = vm_get_tensor(vm, vm_pop(vm));
        Value mv = vm_pop(vm);
        if (is_heap_type(vm, mv, HEAP_MANIFOLD) && grad) {
            /* Curvature gradient: sum of loss gradient components (simplified) */
            double sum = 0;
            for (int64_t i = 0; i < grad->total; i++) sum += grad->data[i];
            vm_push_float(vm, sum);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 853: { /* transition-geometry(manifold, target_curvature, rate) */
        float rate = (float)as_number(vm_pop(vm));
        float target = (float)as_number(vm_pop(vm));
        Value mv = vm_pop(vm);
        if (is_heap_type(vm, mv, HEAP_MANIFOLD)) {
            qllm_manifold_t* m = (qllm_manifold_t*)vm->heap.objects[mv.as.ptr]->opaque.ptr;
            float cur = qllm_manifold_get_curvature(m);
            float new_c = cur + rate * (target - cur); /* linear interpolation */
            qllm_manifold_set_curvature(m, new_c);
            vm_push_float(vm, new_c);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 854: { /* manifold-interpolate(m1, m2, t) — interpolate between manifolds */
        float t = (float)as_number(vm_pop(vm));
        Value m2v = vm_pop(vm), m1v = vm_pop(vm);
        if (is_heap_type(vm, m1v, HEAP_MANIFOLD) && is_heap_type(vm, m2v, HEAP_MANIFOLD)) {
            float c1 = qllm_manifold_get_curvature((qllm_manifold_t*)vm->heap.objects[m1v.as.ptr]->opaque.ptr);
            float c2 = qllm_manifold_get_curvature((qllm_manifold_t*)vm->heap.objects[m2v.as.ptr]->opaque.ptr);
            float interp = c1 * (1.0f - t) + c2 * t;
            vm_push_float(vm, interp);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 855: case 856: case 857: case 858: case 859: {
        /* curvature-hessian, adaptive-step, manifold-type, manifold-dim, manifold-destroy */
        if (fid == 859) { /* manifold-destroy */
            Value mv = vm_pop(vm);
            if (is_heap_type(vm, mv, HEAP_MANIFOLD)) {
                qllm_manifold_destroy((qllm_manifold_t*)vm->heap.objects[mv.as.ptr]->opaque.ptr);
                vm->heap.objects[mv.as.ptr]->opaque.ptr = NULL;
            }
            vm_push(vm, NIL_VAL);
        } else if (fid == 857) { /* manifold-type */
            Value mv = vm_pop(vm);
            if (is_heap_type(vm, mv, HEAP_MANIFOLD))
                vm_push(vm, INT_VAL((int)qllm_manifold_get_type(
                    (qllm_manifold_t*)vm->heap.objects[mv.as.ptr]->opaque.ptr)));
            else vm_push(vm, NIL_VAL);
        } else if (fid == 858) {
            vm_pop(vm);
            vm_push(vm, NIL_VAL);
        } else {
            int nargs = vm_geometric_arity(fid); for (int i = 0; i < nargs; i++) vm_pop(vm);
            vm_push(vm, NIL_VAL);
        }
        break;
    }

    default:
        fprintf(stderr, "GEOMETRIC: unknown operation %d\n", fid);
        vm_push(vm, NIL_VAL); break;
    }
#else
    vm_dispatch_geometric_fallback(vm, fid);
#endif
}
