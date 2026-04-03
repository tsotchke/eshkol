/**
 * @file vm_geometric.c
 * @brief VM native function wrappers for geometric manifold operations.
 *
 * Provides native call IDs 800-859 for the Eshkol bytecode VM,
 * wrapping the semiclassical_qllm geometric library:
 *   800-804: Manifold creation (euclidean, hyperbolic, spherical, product, adaptive)
 *   805-809: Core ops (exp_map, log_map, geodesic_distance, parallel_transport, project)
 *   810-814: Hyperbolic (mobius_add, mobius_scalar_mul, poincare_distance, frechet_mean, curvature)
 *   815-819: Spherical (great_circle_distance, slerp, spherical_exp, spherical_log, rotation)
 *   820-824: Lie groups (so3_exp, so3_log, se3_exp, se3_log, quaternion_mul)
 *   825-829: Differential (metric_tensor, christoffel, riemann_curvature, ricci, sectional)
 *   830-834: Forms (wedge_product, exterior_derivative, hodge_star, interior_product, pullback)
 *   835-839: Optimization (riemannian_sgd_step, riemannian_adam_step, riemannian_grad, retraction, vector_transport)
 *   840-849: Geodesic attention
 *   850-859: Adaptive curvature
 *
 * On WASM or builds without semiclassical_qllm, all operations return NIL
 * with a warning message.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

/* Heap types for geometric objects */
#define HEAP_MANIFOLD        30
#define HEAP_MANIFOLD_POINT  31
#define HEAP_MANIFOLD_TANGENT 32

static void vm_dispatch_geometric(VM* vm, int fid) {
    /* Stub: geometric operations require libsemiclassical_qllm */
    /* When linked against the library, these dispatch to the FFI bridge */
#if defined(ESHKOL_GEOMETRIC_ENABLED)
    /* TODO: Wire each fid 800-859 to eshkol_ffi.h functions */
    switch (fid) {
    case 800: /* make-euclidean-manifold */
    case 801: /* make-hyperbolic-manifold */
    case 802: /* make-spherical-manifold */
    case 803: /* make-product-manifold */
    case 804: /* make-adaptive-manifold */
    case 805: /* exp-map */
    case 806: /* log-map */
    case 807: /* geodesic-distance */
    case 808: /* parallel-transport */
    case 809: /* project */
    case 810: /* mobius-add */
    case 811: /* mobius-scalar-mul */
    case 812: /* poincare-distance */
    case 813: /* frechet-mean */
    case 814: /* curvature */
    case 815: /* great-circle-distance */
    case 816: /* slerp */
    case 817: /* spherical-exp */
    case 818: /* spherical-log */
    case 819: /* rotation */
    case 820: /* so3-exp */
    case 821: /* so3-log */
    case 822: /* se3-exp */
    case 823: /* se3-log */
    case 824: /* quaternion-mul */
    case 825: /* metric-tensor */
    case 826: /* christoffel */
    case 827: /* riemann-curvature */
    case 828: /* ricci */
    case 829: /* sectional */
    case 830: /* wedge-product */
    case 831: /* exterior-derivative */
    case 832: /* hodge-star */
    case 833: /* interior-product */
    case 834: /* pullback */
    case 835: /* riemannian-sgd-step */
    case 836: /* riemannian-adam-step */
    case 837: /* riemannian-grad */
    case 838: /* retraction */
    case 839: /* vector-transport */
    default:
        printf("GEOMETRIC: operation %d not yet implemented\n", fid);
        vm_push(vm, NIL_VAL);
        break;
    }
#else
    (void)fid;
    printf("GEOMETRIC: requires libsemiclassical_qllm (native build only)\n");
    vm_push(vm, NIL_VAL);
#endif
}
