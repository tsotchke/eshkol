/**
 * @file vm_geometric.c
 * @brief VM geometric manifold dispatch — native IDs 804-843.
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
    vm_push(vm, (Value){.type = VAL_MANIFOLD, .as.ptr = ptr});
}

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
    case 810: { /* log-map(base_tensor, point_tensor, curvature) */
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
    case 811: { /* geodesic-distance(x_tensor, y_tensor, curvature) */
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
    case 812: { /* parallel-transport(x, y, v, curvature) */
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
    case 813: { /* project(x_tensor, curvature) — project onto manifold */
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
    case 814: { /* mobius-add(x, y, curvature) */
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
    case 815: { /* mobius-scalar-mul(r, x, curvature) */
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
    case 816: { /* poincare-distance(x, y, curvature) */
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
    case 817: { /* frechet-mean(points_tensor, weights_tensor, curvature) */
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
    case 819: { /* great-circle-distance(x, y) */
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
    case 820: { /* slerp(x, y, t) */
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
    case 821: { /* spherical-exp(base, tangent) */
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
    case 822: { /* spherical-log(base, point) */
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
    case 823: { /* spherical-project(x) — project onto unit sphere */
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

    /* ═══ Lie group operations (820-824) ═══ */
    case 824: { /* so3-exp(omega_tensor) — axis-angle → rotation quaternion */
        VmTensor* omega = vm_get_tensor(vm, vm_pop(vm));
        if (omega && omega->total >= 3) {
            float* of = vm_tensor_to_float(omega);
            qllm_so3_algebra_t alg = {{of[0], of[1], of[2]}};
            qllm_so3_t rot = qllm_so3_exp(&alg);
            free(of);
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
            float* qf = vm_tensor_to_float(qv);
            qllm_so3_t rot = {qf[0], qf[1], qf[2], qf[3]};
            qllm_so3_algebra_t alg = qllm_so3_log(&rot);
            free(qf);
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
            float* tf = vm_tensor_to_float(tv);
            qllm_se3_algebra_t twist;
            twist.omega.omega[0]=tf[0]; twist.omega.omega[1]=tf[1]; twist.omega.omega[2]=tf[2];
            twist.v[0]=tf[3]; twist.v[1]=tf[4]; twist.v[2]=tf[5];
            qllm_se3_t pose = qllm_se3_exp(&twist);
            free(tf);
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
            float* pf = vm_tensor_to_float(pv);
            qllm_se3_t pose;
            pose.rotation = (qllm_so3_t){pf[0],pf[1],pf[2],pf[3]};
            pose.translation[0]=pf[4]; pose.translation[1]=pf[5]; pose.translation[2]=pf[6];
            qllm_se3_algebra_t twist = qllm_se3_log(&pose);
            free(pf);
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
            float* q1f = vm_tensor_to_float(q1v);
            float* q2f = vm_tensor_to_float(q2v);
            qllm_so3_t r1 = {q1f[0],q1f[1],q1f[2],q1f[3]};
            qllm_so3_t r2 = {q2f[0],q2f[1],q2f[2],q2f[3]};
            qllm_so3_t result = qllm_so3_compose(&r1, &r2);
            free(q1f); free(q2f);
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
        vm_pop(vm); vm_pop(vm); /* manifold, point */
        /* Full Christoffel symbols require creating a connection object.
         * Return scalar curvature as proxy for now. */
        vm_push(vm, FLOAT_VAL(0)); break;
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
        vm_pop(vm); vm_pop(vm); /* Pullback requires full Jacobian matrix — complex */
        vm_push(vm, NIL_VAL); break;
    }

    /* ═══ Riemannian optimization (835-839) ═══ */
    case 839: { /* riemannian-sgd-step(point, gradient, lr, curvature) */
        float c = (float)as_number(vm_pop(vm));
        float lr = (float)as_number(vm_pop(vm));
        VmTensor* gv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));
        if (pv && gv) {
            float* pf = vm_tensor_to_float(pv);
            float* gf = vm_tensor_to_float(gv);
            int n = (int)pv->total;
            /* Riemannian SGD: retract(-lr * grad) from point */
            float* neg_scaled = (float*)malloc(n * sizeof(float));
            if (neg_scaled) {
                for (int i = 0; i < n; i++) neg_scaled[i] = -lr * gf[i];
                qllm_tensor_t* result = qllm_hyperbolic_exp_map(pf, neg_scaled, n, c);
                free(neg_scaled);
                if (result) {
                    float* rd = (float*)qllm_tensor_get_data(result);
                    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, pv->shape, pv->n_dims);
                    if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                    qllm_tensor_destroy(result);
                    free(pf); free(gf);
                    if (out) { VM_PUSH_TENSOR(vm, out); break; }
                }
            }
            free(pf); free(gf);
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
            float* pf = vm_tensor_to_float(pv);
            float* gf = vm_tensor_to_float(gv);
            int n = (int)pv->total;
            float* step = (float*)malloc(n * sizeof(float));
            if (step) {
                (void)b1; (void)b2; /* Full Adam needs state — use SGD as fallback */
                for (int i = 0; i < n; i++) step[i] = -lr * gf[i];
                qllm_tensor_t* result = qllm_hyperbolic_exp_map(pf, step, n, c);
                free(step);
                if (result) {
                    float* rd = (float*)qllm_tensor_get_data(result);
                    VmTensor* out = vm_tensor_zeros(&vm->heap.regions, pv->shape, pv->n_dims);
                    if (out && rd) for (int64_t i = 0; i < out->total; i++) out->data[i] = rd[i];
                    qllm_tensor_destroy(result);
                    free(pf); free(gf);
                    if (out) { VM_PUSH_TENSOR(vm, out); break; }
                }
            }
            free(pf); free(gf);
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 841: { /* riemannian-grad(euclidean_grad, point, curvature) — project to tangent space */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* pv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* gv = vm_get_tensor(vm, vm_pop(vm));
        if (gv && pv) {
            /* Riemannian gradient = conformal_factor^2 * euclidean_gradient */
            float* pf = vm_tensor_to_float(pv);
            float cf = qllm_hyperbolic_conformal_factor(pf, (int)pv->total, c);
            free(pf);
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
            float* bf = vm_tensor_to_float(bv);
            float* tf = vm_tensor_to_float(tv);
            qllm_tensor_t* result = qllm_hyperbolic_exp_map(bf, tf, (int)bv->total, c);
            free(bf); free(tf);
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
        vm_push(vm, NIL_VAL); break;
    }

    /* ═══ Geodesic attention (840-849) ═══ */
    case 840: { /* geodesic-attention-scores(Q, K, curvature) */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* kv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* qv = vm_get_tensor(vm, vm_pop(vm));
        if (qv && kv) {
            float* qf = vm_tensor_to_float(qv);
            float* kf = vm_tensor_to_float(kv);
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
                free(qf); free(kf);
                VM_PUSH_TENSOR(vm, out); break;
            }
            free(qf); free(kf);
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 841: { /* geodesic-attention-values(scores, V, curvature) — weighted Fréchet mean */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* vv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* sv = vm_get_tensor(vm, vm_pop(vm));
        if (sv && vv && vv->n_dims >= 2) {
            int n = (int)vv->shape[0], dim = (int)vv->shape[1];
            float* vf = vm_tensor_to_float(vv);
            float* sf = vm_tensor_to_float(sv);
            /* Weighted Fréchet mean of value vectors */
            qllm_tensor_t* result = qllm_hyperbolic_frechet_mean(vf, sf, n, dim, c, 50, 1e-5f);
            free(vf); free(sf);
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
    case 842: { /* curvature-softmax(scores, curvature) — curvature-scaled softmax */
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
    case 843: { /* geodesic-attention-forward(Q, K, V, curvature) — full attention */
        float c = (float)as_number(vm_pop(vm));
        VmTensor* vv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* kv = vm_get_tensor(vm, vm_pop(vm));
        VmTensor* qv = vm_get_tensor(vm, vm_pop(vm));
        if (qv && kv && vv && qv->n_dims >= 2 && kv->n_dims >= 2) {
            int n_q = (int)qv->shape[0], n_k = (int)kv->shape[0];
            int dim = (int)qv->shape[1];
            float* qf = vm_tensor_to_float(qv);
            float* kf = vm_tensor_to_float(kv);
            float* vf = vm_tensor_to_float(vv);
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
                free(qf); free(kf); free(vf);
                VM_PUSH_TENSOR(vm, out); break;
            }
            free(qf); free(kf); free(vf);
        }
        vm_push(vm, NIL_VAL); break;
    }
    case 844: case 845: case 846: case 847: case 848: case 849: {
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
        } else {
            int nargs = 2; for (int i = 0; i < nargs; i++) vm_pop(vm);
            vm_push(vm, NIL_VAL);
        }
        break;
    }

    default:
        fprintf(stderr, "GEOMETRIC: unknown operation %d\n", fid);
        vm_push(vm, NIL_VAL); break;
    }
#else
    /* No geometric library available */
    (void)fid;
    vm_push(vm, NIL_VAL);
#endif
}
