/*
 * Real-Time Differentiable Signal Temporal Logic (STL) Control Loop
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/stl_control_loop.h"

#include <algorithm>
#include <cmath>
#include <vector>

struct eshkol_stl_controller {
    uint32_t d = 0;
    uint32_t m = 0;
    std::vector<double> A;     /* [d x d] */
    std::vector<double> B;     /* [d x m] */
    std::vector<double> x_ref; /* [d] */
    double tube_eps = 1.0;
    double u_max = 1.0;
    double beta = 8.0;
    double max_sensor_delay = 0.1;

    bool has_prev_surrogate = false;
    double prev_surrogate = 0.0;
    double prev_physical_margin = 0.0;

    uint64_t sensor_faults = 0;
    uint64_t silent_drifts = 0;
};

namespace {

constexpr double kNormSmoothDelta = 1e-8;

inline void zero_actuators(double* out_u, uint32_t m) {
    if (!out_u) return;
    for (uint32_t i = 0; i < m; ++i) out_u[i] = 0.0;
}

inline double clamp_val(double v, double lo, double hi) {
    if (!std::isfinite(v)) return 0.0;
    return std::max(lo, std::min(hi, v));
}

} /* namespace */

extern "C" {

eshkol_stl_controller_t* eshkol_stl_controller_create(
    uint32_t state_dim,
    uint32_t control_dim,
    const double* A_row_major,
    const double* B_row_major,
    const double* x_ref,
    double tube_eps,
    double u_max,
    double beta,
    double max_sensor_delay) {
    if (state_dim == 0 || control_dim == 0 || !A_row_major || !B_row_major || !x_ref) {
        return nullptr;
    }
    if (!std::isfinite(tube_eps) || tube_eps <= 0.0 ||
        !std::isfinite(u_max) || u_max <= 0.0 ||
        !std::isfinite(beta) || beta <= 0.0 ||
        !std::isfinite(max_sensor_delay) || max_sensor_delay <= 0.0) {
        return nullptr;
    }

    auto* ctrl = new eshkol_stl_controller();
    ctrl->d = state_dim;
    ctrl->m = control_dim;
    ctrl->A.assign(A_row_major, A_row_major + static_cast<size_t>(state_dim) * state_dim);
    ctrl->B.assign(B_row_major, B_row_major + static_cast<size_t>(state_dim) * control_dim);
    ctrl->x_ref.assign(x_ref, x_ref + state_dim);
    ctrl->tube_eps = tube_eps;
    ctrl->u_max = u_max;
    ctrl->beta = beta;
    ctrl->max_sensor_delay = max_sensor_delay;
    return ctrl;
}

void eshkol_stl_controller_destroy(eshkol_stl_controller_t* ctrl) {
    delete ctrl;
}

double eshkol_stl_physical_margin(const eshkol_stl_controller_t* ctrl,
                                  const double* physical_state) {
    if (!ctrl || !physical_state) return -INFINITY;
    double sq = 0.0;
    for (uint32_t i = 0; i < ctrl->d; ++i) {
        if (!std::isfinite(physical_state[i])) return -INFINITY;
        double diff = physical_state[i] - ctrl->x_ref[i];
        sq += diff * diff;
    }
    return ctrl->tube_eps - std::sqrt(sq);
}

bool eshkol_stl_eval_robustness_grad(
    const eshkol_stl_controller_t* ctrl,
    const double* x0,
    const double* u_seq,
    uint32_t horizon,
    double* out_rho_always,
    double* out_rho_eventually,
    double* out_u_grads,
    uint64_t* out_fd_evals) {
    if (!ctrl || !x0 || !u_seq || horizon == 0) return false;
    if (out_fd_evals) *out_fd_evals = 0;

    const uint32_t d = ctrl->d;
    const uint32_t m = ctrl->m;
    const uint32_t T = horizon;

    /* Forward rollout: traj[0..T] each of length d */
    std::vector<double> traj(static_cast<size_t>(T + 1) * d, 0.0);
    for (uint32_t i = 0; i < d; ++i) {
        if (!std::isfinite(x0[i])) return false;
        traj[i] = x0[i];
    }
    for (uint32_t t = 0; t < T; ++t) {
        const double* xt = traj.data() + static_cast<size_t>(t) * d;
        const double* ut = u_seq + static_cast<size_t>(t) * m;
        double* xnext = traj.data() + static_cast<size_t>(t + 1) * d;
        for (uint32_t j = 0; j < m; ++j) {
            if (!std::isfinite(ut[j])) return false;
        }
        for (uint32_t i = 0; i < d; ++i) {
            double s = 0.0;
            for (uint32_t j = 0; j < d; ++j) s += ctrl->A[i * d + j] * xt[j];
            for (uint32_t j = 0; j < m; ++j) s += ctrl->B[i * m + j] * ut[j];
            xnext[i] = s;
        }
    }

    /* Per-step smooth tube robustness rho[1..T] (t=1..T affected by u_0..u_{T-1}). */
    std::vector<double> rho(T, 0.0);
    std::vector<double> norm_smooth(T, 0.0);
    for (uint32_t t = 1; t <= T; ++t) {
        const double* xt = traj.data() + static_cast<size_t>(t) * d;
        double sq = kNormSmoothDelta;
        for (uint32_t i = 0; i < d; ++i) {
            double diff = xt[i] - ctrl->x_ref[i];
            sq += diff * diff;
        }
        double ns = std::sqrt(sq);
        norm_smooth[t - 1] = ns;
        rho[t - 1] = ctrl->tube_eps - ns;
    }

    /* Smooth Always (soft-min): rho_G = -(1/beta) * log(sum_k exp(-beta * rho_k)) */
    const double beta = ctrl->beta;
    double max_neg = -beta * rho[0];
    double max_pos = beta * rho[0];
    for (uint32_t k = 1; k < T; ++k) {
        if (-beta * rho[k] > max_neg) max_neg = -beta * rho[k];
        if (beta * rho[k] > max_pos) max_pos = beta * rho[k];
    }

    double sum_neg = 0.0;
    double sum_pos = 0.0;
    std::vector<double> w_always(T, 0.0);
    for (uint32_t k = 0; k < T; ++k) {
        double en = std::exp(-beta * rho[k] - max_neg);
        w_always[k] = en;
        sum_neg += en;
        sum_pos += std::exp(beta * rho[k] - max_pos);
    }

    const double rho_always = -(max_neg + std::log(sum_neg)) / beta;
    const double rho_eventually = (max_pos + std::log(sum_pos)) / beta;

    if (out_rho_always) *out_rho_always = rho_always;
    if (out_rho_eventually) *out_rho_eventually = rho_eventually;

    if (out_u_grads) {
        for (uint32_t k = 0; k < T; ++k) w_always[k] /= sum_neg;

        /* Exact reverse-mode adjoint through plant rollout (zero FD). */
        std::vector<double> adj_x(d, 0.0);
        std::vector<double> adj_x_prev(d, 0.0);

        for (int t = static_cast<int>(T) - 1; t >= 0; --t) {
            const uint32_t step_idx = static_cast<uint32_t>(t);
            const double* xnext = traj.data() + static_cast<size_t>(step_idx + 1) * d;
            /* Local gradient from rho[step_idx] into x_{t+1} */
            const double inv_ns = w_always[step_idx] / norm_smooth[step_idx];
            for (uint32_t i = 0; i < d; ++i) {
                adj_x[i] += -inv_ns * (xnext[i] - ctrl->x_ref[i]);
            }

            /* d(rho_G)/d(u_t) = B^T * adj_x */
            double* gu = out_u_grads + static_cast<size_t>(step_idx) * m;
            for (uint32_t j = 0; j < m; ++j) {
                double s = 0.0;
                for (uint32_t i = 0; i < d; ++i) s += ctrl->B[i * m + j] * adj_x[i];
                gu[j] = s;
            }

            /* Propagate adjoint backward: adj_x_prev = A^T * adj_x */
            for (uint32_t j = 0; j < d; ++j) {
                double s = 0.0;
                for (uint32_t i = 0; i < d; ++i) s += ctrl->A[i * d + j] * adj_x[i];
                adj_x_prev[j] = s;
            }
            adj_x.swap(adj_x_prev);
        }
    }

    return true;
}

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
    uint64_t* out_fd_evals) {
    if (!ctrl || !sensor_state || !out_u || horizon == 0) {
        if (ctrl && out_u) zero_actuators(out_u, ctrl->m);
        return ESHKOL_STL_ERR_INVALID_ARG;
    }
    if (out_fd_evals) *out_fd_evals = 0;

    /* 1. Sensor finiteness & freshness watchdog: fail-closed clamp to 0.0. */
    if (!std::isfinite(dt) || dt <= 0.0 || dt > ctrl->max_sensor_delay) {
        zero_actuators(out_u, ctrl->m);
        ++ctrl->sensor_faults;
        return ESHKOL_STL_ERR_SENSOR_FAULT;
    }
    for (uint32_t i = 0; i < ctrl->d; ++i) {
        if (!std::isfinite(sensor_state[i])) {
            zero_actuators(out_u, ctrl->m);
            ++ctrl->sensor_faults;
            return ESHKOL_STL_ERR_SENSOR_FAULT;
        }
    }
    if (has_surrogate && !std::isfinite(surrogate_loss)) {
        zero_actuators(out_u, ctrl->m);
        ++ctrl->sensor_faults;
        return ESHKOL_STL_ERR_SENSOR_FAULT;
    }

    /* 2. Independent physical-state STL silent-drift verifier. */
    const double true_margin = eshkol_stl_physical_margin(ctrl, sensor_state);
    if (has_surrogate) {
        if (ctrl->has_prev_surrogate) {
            bool surrogate_improved = (surrogate_loss < ctrl->prev_surrogate - 1e-12);
            bool physical_worsened = (true_margin < ctrl->prev_physical_margin - 1e-12);
            bool outside_tube = (true_margin < 0.0);
            if (surrogate_improved && physical_worsened && outside_tube) {
                zero_actuators(out_u, ctrl->m);
                ++ctrl->silent_drifts;
                ctrl->prev_surrogate = surrogate_loss;
                ctrl->prev_physical_margin = true_margin;
                return ESHKOL_STL_ERR_SILENT_DRIFT;
            }
        }
        ctrl->has_prev_surrogate = true;
        ctrl->prev_surrogate = surrogate_loss;
        ctrl->prev_physical_margin = true_margin;
    }

    /* 3. Differentiable STL receding-horizon gradient ascent (zero FD). */
    const size_t n_u = static_cast<size_t>(horizon) * ctrl->m;
    std::vector<double> u_seq(n_u, 0.0);
    std::vector<double> u_grads(n_u, 0.0);
    uint64_t fde = 0;

    for (uint32_t it = 0; it < opt_iters; ++it) {
        double rho_g = 0.0;
        if (!eshkol_stl_eval_robustness_grad(
                ctrl, sensor_state, u_seq.data(), horizon,
                &rho_g, nullptr, u_grads.data(), &fde)) {
            zero_actuators(out_u, ctrl->m);
            ++ctrl->sensor_faults;
            return ESHKOL_STL_ERR_SENSOR_FAULT;
        }
        for (size_t k = 0; k < n_u; ++k) {
            double next_v = u_seq[k] + step_size * u_grads[k];
            u_seq[k] = clamp_val(next_v, -ctrl->u_max, ctrl->u_max);
        }
    }

    for (uint32_t j = 0; j < ctrl->m; ++j) {
        out_u[j] = clamp_val(u_seq[j], -ctrl->u_max, ctrl->u_max);
    }
    return ESHKOL_STL_OK;
}

bool eshkol_stl_plant_step(const eshkol_stl_controller_t* ctrl,
                           const double* x_cur,
                           const double* u_cur,
                           double* x_next) {
    if (!ctrl || !x_cur || !u_cur || !x_next) return false;
    const uint32_t d = ctrl->d;
    const uint32_t m = ctrl->m;
    std::vector<double> tmp(d, 0.0);
    for (uint32_t i = 0; i < d; ++i) {
        double s = 0.0;
        for (uint32_t j = 0; j < d; ++j) s += ctrl->A[i * d + j] * x_cur[j];
        for (uint32_t j = 0; j < m; ++j) s += ctrl->B[i * m + j] * u_cur[j];
        tmp[i] = s;
    }
    for (uint32_t i = 0; i < d; ++i) x_next[i] = tmp[i];
    return true;
}

uint64_t eshkol_stl_sensor_fault_count(const eshkol_stl_controller_t* ctrl) {
    return ctrl ? ctrl->sensor_faults : 0;
}

uint64_t eshkol_stl_silent_drift_count(const eshkol_stl_controller_t* ctrl) {
    return ctrl ? ctrl->silent_drifts : 0;
}

} /* extern "C" */
