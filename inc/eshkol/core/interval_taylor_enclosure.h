/*
 * Graded Interval-Taylor Certified Neural World Model Enclosures (Order-2)
 *
 * Implements outward-rounded IEEE-754 order-2 multivariate Taylor models:
 *   T(z) = a0 + g^T z + 0.5 * z^T H z + R,  z in [-1, 1]^d
 * over smooth C^infty activations (identity, tanh, sigmoid, softplus, sin, x^2),
 * affine layers, and residual skip connections, cancelling linear dependency
 * blowup across network depth.
 *
 * Explicit activation regularity tagging fails closed with
 * ESHKOL_TM_ERR_NON_LIPSCHITZ on discontinuous/non-Lipschitz activations
 * (e.g. Heaviside step), returning uncertified [-inf, +inf].
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_INTERVAL_TAYLOR_ENCLOSURE_H
#define ESHKOL_CORE_INTERVAL_TAYLOR_ENCLOSURE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_tm_activation {
    ESHKOL_TM_ACT_IDENTITY       = 0,
    ESHKOL_TM_ACT_TANH           = 1,
    ESHKOL_TM_ACT_SIGMOID        = 2,
    ESHKOL_TM_ACT_SOFTPLUS       = 3,
    ESHKOL_TM_ACT_SIN            = 4,
    ESHKOL_TM_ACT_SQUARE         = 5,
    ESHKOL_TM_ACT_HEAVISIDE_STEP = 6  /* Discontinuous / non-Lipschitz -> fail closed! */
} eshkol_tm_activation_t;

typedef enum eshkol_tm_status {
    ESHKOL_TM_OK                = 0,
    ESHKOL_TM_ERR_INVALID_ARG   = 1,
    ESHKOL_TM_ERR_NON_LIPSCHITZ = 2
} eshkol_tm_status_t;

typedef struct eshkol_tm_interval {
    double lo;
    double hi;
} eshkol_tm_interval_t;

typedef struct eshkol_tm_world_model eshkol_tm_world_model_t;

/* Create an empty neural world model accepting input dimension `input_dim`. */
eshkol_tm_world_model_t* eshkol_tm_model_create(uint32_t input_dim);
void eshkol_tm_model_destroy(eshkol_tm_world_model_t* model);

/*
 * Append a layer: y = act(W * x + b) + (add_residual ? x : 0).
 * W is row-major [out_dim x in_dim], b has length [out_dim].
 * When `add_residual` is true, requires `out_dim == in_dim`.
 */
bool eshkol_tm_model_add_layer(eshkol_tm_world_model_t* model,
                               uint32_t out_dim,
                               const double* weights,
                               const double* bias,
                               eshkol_tm_activation_t activation,
                               bool add_residual);

uint32_t eshkol_tm_model_input_dim(const eshkol_tm_world_model_t* model);
uint32_t eshkol_tm_model_output_dim(const eshkol_tm_world_model_t* model);

/* Exact point forward evaluation for Monte Carlo soundness checks. */
bool eshkol_tm_model_eval_point(const eshkol_tm_world_model_t* model,
                                const double* input,
                                double* out_output);

/*
 * Certified order-2 Interval-Taylor enclosure over input box
 * `input_box[0..input_dim-1]`. Writes certified outward-rounded bounds into
 * `out_bounds[0..output_dim-1]`.
 * Fails closed with ESHKOL_TM_ERR_NON_LIPSCHITZ (setting bounds to [-inf, +inf])
 * if any layer uses a non-Lipschitz/discontinuous activation.
 */
eshkol_tm_status_t eshkol_tm_model_certify(const eshkol_tm_world_model_t* model,
                                           const eshkol_tm_interval_t* input_box,
                                           eshkol_tm_interval_t* out_bounds);

/*
 * Baseline naive interval arithmetic (order-0 IBP) over the same input box,
 * for direct tightness comparison against order-2 Interval-Taylor models.
 */
eshkol_tm_status_t eshkol_tm_model_naive_interval(const eshkol_tm_world_model_t* model,
                                                  const eshkol_tm_interval_t* input_box,
                                                  eshkol_tm_interval_t* out_bounds);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_INTERVAL_TAYLOR_ENCLOSURE_H */
