/*
 * Real-Time Differentiable Signal Temporal Logic (STL) Control Loop
 *
 * Implements smooth quantitative STL robustness (Always / Eventually over a
 * safety tube |x - x_ref| <= eps) with exact analytical control gradients
 * (zero finite differences), fail-closed sensor finiteness/delay watchdog
 * clamping actuators safely on NaN/Inf/timeout, and an independent physical
 * state safety verifier that halts silent drift when a surrogate objective
 * decreases while true physical state breaches the STL safety tube.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_STL_CONTROL_LOOP_H
#define ESHKOL_CORE_STL_CONTROL_LOOP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum eshkol_stl_status {
    ESHKOL_STL_OK               = 0,
    ESHKOL_STL_ERR_INVALID_ARG  = 1,
    ESHKOL_STL_ERR_SENSOR_FAULT = 2, /* NaN, Inf, or stale/infinite sensor delay */
    ESHKOL_STL_ERR_SILENT_DRIFT = 3  /* Surrogate decreases while physical state breaches STL tube */
} eshkol_stl_status_t;

typedef struct eshkol_stl_controller eshkol_stl_controller_t;

/*
 * Create a differentiable STL controller for linear discrete plant:
 *   x_{t+1} = A * x_t + B * u_t
 * targeting safety tube ||x_t - x_ref||_2 <= tube_eps with actuator clamp
 * |u_i| <= u_max, smooth STL sharpness `beta > 0`, and sensor timeout
 * `max_sensor_delay > 0`.
 */
eshkol_stl_controller_t* eshkol_stl_controller_create(
    uint32_t state_dim,
    uint32_t control_dim,
    const double* A_row_major,
    const double* B_row_major,
    const double* x_ref,
    double tube_eps,
    double u_max,
    double beta,
    double max_sensor_delay);

void eshkol_stl_controller_destroy(eshkol_stl_controller_t* ctrl);

/*
 * Compute smooth STL Always-in-tube robustness rho_G and exact analytical
 * gradient d(rho_G)/d(u_seq) over horizon `horizon` steps from `x0`.
 * `u_seq` has length `horizon * control_dim`.
 * Always sets `*out_fd_evals = 0`.
 */
bool eshkol_stl_eval_robustness_grad(
    const eshkol_stl_controller_t* ctrl,
    const double* x0,
    const double* u_seq,
    uint32_t horizon,
    double* out_rho_always,
    double* out_rho_eventually,
    double* out_u_grads,
    uint64_t* out_fd_evals);

/* Exact physical safety-tube margin: tube_eps - ||x - x_ref||_2. */
double eshkol_stl_physical_margin(const eshkol_stl_controller_t* ctrl,
                                  const double* physical_state);

/*
 * Real-time closed-loop control step with sensor watchdog and silent-drift
 * physical verifier:
 *   - On NaN / Inf / stale or infinite dt: clamps `out_u` to 0.0 and returns
 *     ESHKOL_STL_ERR_SENSOR_FAULT.
 *   - If `has_surrogate` is true and surrogate loss decreased while true
 *     physical state diverged outside the STL tube: clamps `out_u` to 0.0 and
 *     returns ESHKOL_STL_ERR_SILENT_DRIFT.
 *   - Otherwise computes finite clamped actuator output via exact analytical
 *     STL robustness gradient ascent (`*out_fd_evals = 0`).
 */
eshkol_stl_status_t eshkol_stl_step(
    eshkol_stl_controller_t* ctrl,
    const double* sensor_state,
    double dt,
    bool has_surrogate,
    double surrogate_loss,
    uint32_t horizon,
    uint32_t opt_iters,
    double step_size,
    double* out_u,
    uint64_t* out_fd_evals);

/* Advance the discrete plant x_{t+1} = A * x_t + B * u_t for closed-loop tests. */
bool eshkol_stl_plant_step(const eshkol_stl_controller_t* ctrl,
                           const double* x_cur,
                           const double* u_cur,
                           double* x_next);

uint64_t eshkol_stl_sensor_fault_count(const eshkol_stl_controller_t* ctrl);
uint64_t eshkol_stl_silent_drift_count(const eshkol_stl_controller_t* ctrl);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_STL_CONTROL_LOOP_H */
