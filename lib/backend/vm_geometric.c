/**
 * @file vm_geometric.c
 * @brief VM geometric manifold dispatch — native IDs 800-859.
 *
 * When ESHKOL_GEOMETRIC_ENABLED is defined, calls semiclassical_qllm.
 * Otherwise returns NIL for all operations.
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

/* Helper: convert VmTensor (f64) to float array for qllm API */
static float* vm_tensor_to_float(const VmTensor* t) {
    if (!t || t->total <= 0) return NULL;
    float* f = (float*)malloc(t->total * sizeof(float));
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
    vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
}

static void vm_dispatch_geometric(VM* vm, int fid) {
#if defined(ESHKOL_GEOMETRIC_ENABLED)
    switch (fid) {

    /* ═══ Manifold creation (800-804) ═══ */
    case 800: { /* make-euclidean-manifold(dim) */
        int dim = (int)as_number(vm_pop(vm));
        qllm_manifold_options_t opts = {0}; opts.curvature = 0;
        vm_push_manifold(vm, qllm_manifold_euclidean_create(dim, &opts));
        break;
    }
    case 801: { /* make-hyperbolic-manifold(dim, curvature) */
        float c = (float)as_number(vm_pop(vm));
        int dim = (int)as_number(vm_pop(vm));
        qllm_manifold_options_t opts = {0}; opts.curvature = c;
        vm_push_manifold(vm, qllm_manifold_hyperbolic_create(dim, &opts));
        break;
    }
    case 802: { /* make-spherical-manifold(dim) */
        int dim = (int)as_number(vm_pop(vm));
        qllm_manifold_options_t opts = {0}; opts.curvature = 1.0f;
        vm_push_manifold(vm, qllm_manifold_spherical_create(dim, &opts));
        break;
    }
    case 803: { /* make-product-manifold(m1, m2) */
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
    case 804: { /* manifold-curvature(m) */
        Value mv = vm_pop(vm);
        if (is_heap_type(vm, mv, HEAP_MANIFOLD))
            vm_push_float(vm, qllm_manifold_get_curvature(
                (qllm_manifold_t*)vm->heap.objects[mv.as.ptr]->opaque.ptr));
        else vm_push(vm, NIL_VAL);
        break;
    }

    /* ═══ Core manifold ops (805-809) ═══ */
    case 805: { /* exp-map(base_tensor, tangent_tensor, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* tv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* bv = vm_get_tensor(vm, vm_pop(vm));
        if (bv && tv) {
            float* bf = vm_tensor_to_float(bv);
            float* tf = vm_tensor_to_float(tv);
            int n = (int)bv->total;
            qllm_tensor_t* result = qllm_hyperbolic_exp_map(bf, tf, n, c);
            free(bf); free(tf);
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
    case 806: { /* log-map(base_tensor, point_tensor, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* bv = vm_get_tensor(vm, vm_pop(vm));
        if (bv && pv) {
            float* bf = vm_tensor_to_float(bv);
            float* pf = vm_tensor_to_float(pv);
            int n = (int)bv->total;
            qllm_tensor_t* result = qllm_hyperbolic_log_map(bf, pf, n, c);
            free(bf); free(pf);
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
    case 807: { /* geodesic-distance(x_tensor, y_tensor, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv && xv->total == yv->total) {
            float* xf = vm_tensor_to_float(xv);
            float* yf = vm_tensor_to_float(yv);
            float dist = qllm_hyperbolic_distance(xf, yf, (int)xv->total, c);
            free(xf); free(yf);
            vm_push_float(vm, dist);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 808: { /* parallel-transport(x, y, v, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* vv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv && vv) {
            float* xf = vm_tensor_to_float(xv);
            float* yf = vm_tensor_to_float(yv);
            float* vf = vm_tensor_to_float(vv);
            qllm_tensor_t* result = qllm_hyperbolic_parallel_transport(xf, yf, vf, (int)xv->total, c);
            free(xf); free(yf); free(vf);
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
    case 809: { /* project(x_tensor, curvature) — project onto manifold */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv) {
            float* xf = vm_tensor_to_float(xv);
            qllm_tensor_t* result = qllm_hyperbolic_project(xf, (int)xv->total, c);
            free(xf);
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
    case 810: { /* mobius-add(x, y, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv) {
            float* xf = vm_tensor_to_float(xv);
            float* yf = vm_tensor_to_float(yv);
            qllm_tensor_t* result = qllm_hyperbolic_mobius_add(xf, yf, (int)xv->total, c);
            free(xf); free(yf);
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
    case 811: { /* mobius-scalar-mul(r, x, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        float r = (float)as_number(vm_pop(vm));
        if (xv) {
            float* xf = vm_tensor_to_float(xv);
            qllm_tensor_t* result = qllm_hyperbolic_mobius_scalar(r, xf, (int)xv->total, c);
            free(xf);
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
    case 812: { /* poincare-distance(x, y, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv && xv->total == yv->total) {
            float* xf = vm_tensor_to_float(xv);
            float* yf = vm_tensor_to_float(yv);
            float dist = qllm_hyperbolic_distance(xf, yf, (int)xv->total, c);
            free(xf); free(yf);
            vm_push_float(vm, dist);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 813: { /* frechet-mean(points_tensor, weights_tensor, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* wv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));
        if (pv && wv) {
            float* pf = vm_tensor_to_float(pv);
            float* wf = vm_tensor_to_float(wv);
            int dim = (pv->n_dims >= 2) ? (int)pv->shape[1] : (int)pv->total;
            int n_points = (pv->n_dims >= 2) ? (int)pv->shape[0] : 1;
            qllm_tensor_t* result = qllm_hyperbolic_frechet_mean(pf, wf, n_points, dim, c, 100, 1e-6f);
            free(pf); free(wf);
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
    case 815: { /* great-circle-distance(x, y) */
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv && xv->total == yv->total) {
            float* xf = vm_tensor_to_float(xv);
            float* yf = vm_tensor_to_float(yv);
            float dist = qllm_spherical_distance(xf, yf, (int)xv->total);
            free(xf); free(yf);
            vm_push_float(vm, dist);
        } else vm_push(vm, NIL_VAL);
        break;
    }
    case 816: { /* slerp(x, y, t) */
        float t = (float)as_number(vm_pop(vm));
        VmTensor* yv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv && yv) {
            float* xf = vm_tensor_to_float(xv);
            float* yf = vm_tensor_to_float(yv);
            qllm_tensor_t* result = qllm_spherical_slerp(xf, yf, (int)xv->total, t);
            free(xf); free(yf);
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
    case 817: { /* spherical-exp(base, tangent) */
        VmTensor* tv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* bv = vm_get_tensor(vm, vm_pop(vm));
        if (bv && tv) {
            float* bf = vm_tensor_to_float(bv);
            float* tf = vm_tensor_to_float(tv);
            qllm_tensor_t* result = qllm_spherical_exp_map(bf, tf, (int)bv->total);
            free(bf); free(tf);
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
    case 818: { /* spherical-log(base, point) */
        VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* bv = vm_get_tensor(vm, vm_pop(vm));
        if (bv && pv) {
            float* bf = vm_tensor_to_float(bv);
            float* pf = vm_tensor_to_float(pv);
            qllm_tensor_t* result = qllm_spherical_log_map(bf, pf, (int)bv->total);
            free(bf); free(pf);
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
    case 819: { /* spherical-project(x) — project onto unit sphere */
        VmTensor* xv = vm_get_tensor(vm, vm_pop(vm));
        if (xv) {
            float* xf = vm_tensor_to_float(xv);
            qllm_tensor_t* result = qllm_spherical_project(xf, (int)xv->total);
            free(xf);
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

    default:
        /* Remaining ops (820-859): Lie groups, differential forms, optimization, attention */
        /* These require more complex type conversions — wired on demand */
        fprintf(stderr, "GEOMETRIC: operation %d — pop args, return NIL\n", fid);
        /* Estimate arg count from ID range */
        if (fid >= 820 && fid <= 824) { vm_pop(vm); vm_pop(vm); } /* Lie: 2 args */
        else if (fid >= 825 && fid <= 829) { vm_pop(vm); } /* Differential: 1 arg */
        else if (fid >= 830 && fid <= 834) { vm_pop(vm); vm_pop(vm); } /* Forms: 2 args */
        else if (fid >= 835 && fid <= 839) { vm_pop(vm); vm_pop(vm); vm_pop(vm); } /* Optim: 3 args */
        else { vm_pop(vm); } /* Default: 1 arg */
        vm_push(vm, NIL_VAL);
        break;
    }
#else
    /* No geometric library available */
    (void)fid;
    vm_push(vm, NIL_VAL);
#endif
}
