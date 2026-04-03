/**
 * @file vm_geometric.c
 * @brief VM native function wrappers for geometric manifold operations.
 *
 * Provides native call IDs 600-659 for the Eshkol bytecode VM,
 * wrapping the semiclassical_qllm geometric library:
 *   600-604: Manifold creation (euclidean, hyperbolic, spherical, product, adaptive)
 *   605-609: Core ops (exp_map, log_map, geodesic_distance, parallel_transport, project)
 *   610-614: Hyperbolic (mobius_add, mobius_scalar_mul, poincare_distance, frechet_mean, curvature)
 *   615-619: Spherical (great_circle_distance, slerp, spherical_exp, spherical_log, rotation)
 *   620-624: Lie groups (so3_exp, so3_log, se3_exp, se3_log, quaternion_mul)
 *   625-629: Differential (metric_tensor, christoffel, riemann_curvature, ricci, sectional)
 *   630-634: Forms (wedge_product, exterior_derivative, hodge_star, interior_product, pullback)
 *   635-639: Optimization (riemannian_sgd_step, riemannian_adam_step, riemannian_grad, retraction, vector_transport)
 *   640-649: Geodesic attention
 *   650-659: Adaptive curvature
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
    /* TODO: Wire each fid 600-659 to eshkol_ffi.h functions */
    switch (fid) {
    case 600: /* make-euclidean-manifold */
    case 601: /* make-hyperbolic-manifold */
    case 602: /* make-spherical-manifold */
    case 603: /* make-product-manifold */
    case 604: /* make-adaptive-manifold */
    case 605: /* exp-map */
    case 606: /* log-map */
    case 607: /* geodesic-distance */
    case 608: /* parallel-transport */
    case 609: /* project */
    case 610: /* mobius-add */
    case 611: /* mobius-scalar-mul */
    case 612: /* poincare-distance */
    case 613: /* frechet-mean */
    case 614: /* curvature */
    case 615: /* great-circle-distance */
    case 616: /* slerp */
    case 617: /* spherical-exp */
    case 618: /* spherical-log */
    case 619: /* rotation */
    case 620: /* so3-exp */
    case 621: /* so3-log */
    case 622: /* se3-exp */
    case 623: /* se3-log */
    case 624: /* quaternion-mul */
    case 625: /* metric-tensor */
    case 626: /* christoffel */
    case 627: /* riemann-curvature */
    case 628: /* ricci */
    case 629: /* sectional */
    case 630: /* wedge-product */
    case 631: /* exterior-derivative */
    case 632: /* hodge-star */
    case 633: /* interior-product */
    case 634: /* pullback */
    case 635: /* riemannian-sgd-step */
    case 636: /* riemannian-adam-step */
    case 637: /* riemannian-grad */
    case 638: /* retraction */
    case 639: /* vector-transport */
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
