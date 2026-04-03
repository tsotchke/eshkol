/**
 * @file vm_geometric.c
 * @brief VM native function wrappers for geometric manifold operations.
 *
 * Native call IDs 800-859. When ESHKOL_GEOMETRIC_ENABLED is defined
 * and libsemiclassical_qllm is linked, dispatches to the geometric
 * library. Otherwise returns error values.
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

static void vm_dispatch_geometric(VM* vm, int fid) {
#if defined(ESHKOL_GEOMETRIC_ENABLED)
    switch (fid) {
    /* ── Manifold creation (800-804) ── */
    case 800: { /* make-euclidean-manifold(dim) */
        Value dim_v = vm_pop(vm);
        int dim = (int)as_number(dim_v);
        qllm_manifold_t* m = qllm_manifold_create_euclidean(dim);
        if (m) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_MANIFOLD;
                vm->heap.objects[ptr]->opaque.ptr = m;
                vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 801: { /* make-hyperbolic-manifold(dim, curvature) */
        Value curv_v = vm_pop(vm), dim_v = vm_pop(vm);
        int dim = (int)as_number(dim_v);
        float curv = (float)as_number(curv_v);
        qllm_manifold_t* m = qllm_manifold_create_hyperbolic(dim, curv);
        if (m) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_MANIFOLD;
                vm->heap.objects[ptr]->opaque.ptr = m;
                vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }
    case 802: { /* make-spherical-manifold(dim) */
        Value dim_v = vm_pop(vm);
        int dim = (int)as_number(dim_v);
        qllm_manifold_t* m = qllm_manifold_create_spherical(dim);
        if (m) {
            int32_t ptr = heap_alloc(&vm->heap);
            if (ptr >= 0) {
                vm->heap.objects[ptr]->type = HEAP_MANIFOLD;
                vm->heap.objects[ptr]->opaque.ptr = m;
                vm_push(vm, (Value){.type = VAL_INT, .as.ptr = ptr});
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    /* ── Hyperbolic operations (810-814) ── */
    case 812: { /* poincare-distance(x, y, curvature) */
        Value c_v = vm_pop(vm), y_v = vm_pop(vm), x_v = vm_pop(vm);
        /* x, y are tensors on the Poincaré ball */
        if (x_v.type == VAL_INT && y_v.type == VAL_INT) {
            VmTensor* xt = (VmTensor*)vm->heap.objects[x_v.as.ptr]->opaque.ptr;
            VmTensor* yt = (VmTensor*)vm->heap.objects[y_v.as.ptr]->opaque.ptr;
            if (xt && yt && xt->total == yt->total) {
                /* Convert VmTensor (f64) to float array for qllm API */
                int n = (int)xt->total;
                float* xf = (float*)malloc(n * sizeof(float));
                float* yf = (float*)malloc(n * sizeof(float));
                for (int i = 0; i < n; i++) { xf[i] = (float)xt->data[i]; yf[i] = (float)yt->data[i]; }
                float dist = qllm_hyperbolic_distance(xf, yf, n, (float)as_number(c_v));
                free(xf); free(yf);
                vm_push(vm, FLOAT_VAL((double)dist));
                break;
            }
        }
        vm_push(vm, NIL_VAL);
        break;
    }

    default:
        /* Unimplemented geometric operation */
        fprintf(stderr, "GEOMETRIC: operation %d not yet wired\n", fid);
        /* Pop expected args (heuristic: 1-3 args) and push NIL */
        vm_push(vm, NIL_VAL);
        break;
    }
#else
    /* No geometric library — return error */
    (void)fid;
    /* Pop args (we don't know how many, but operations always push a result) */
    vm_push(vm, NIL_VAL);
#endif
}
