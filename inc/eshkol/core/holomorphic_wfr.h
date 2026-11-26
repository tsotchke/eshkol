/*
 * Holomorphic Equilibrium Propagation (hEP) & Wasserstein-Fisher-Rao (WFR) Flow
 *
 * Implements:
 *   1. Holomorphic Equilibrium Propagation (hEP) on complex-valued energy
 *      networks: computes K-point Cauchy-Fourier contour gradients around a
 *      circle of radius r in the complex nudging plane (zero parameter FD evals;
 *      K=4 cancels O(r), O(r^2), O(r^3) bias leaving O(r^4) error vs exact BP)
 *   2. Wasserstein-Fisher-Rao (WFR) Particle Flow: alternating Wasserstein KDE
 *      transport + Fisher-Rao birth-death weight update over separated bimodal
 *      mixture potentials (recovering exact mode mass balance across high energy
 *      barriers where pure Wasserstein transport remains trapped)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_HOLOMORPHIC_WFR_H
#define ESHKOL_CORE_HOLOMORPHIC_WFR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1. Holomorphic Equilibrium Propagation (hEP)
 * ============================================================================ */

typedef struct eshkol_hep_net eshkol_hep_net_t;

eshkol_hep_net_t* eshkol_hep_create(const uint32_t* layer_dims,
                                    uint32_t num_layers,
                                    uint64_t seed);

void eshkol_hep_destroy(eshkol_hep_net_t* net);

uint32_t eshkol_hep_total_params(const eshkol_hep_net_t* net);

/*
 * K-point Holomorphic Equilibrium Propagation around complex circle |beta| = radius:
 *   grad = Re[ (1/K) sum_{k=0..K-1} (e^{-i 2 pi k / K} / radius) * dPhi/dW(z*(radius * e^{i 2 pi k / K})) ]
 * - K = 1: real-only positive nudge (+radius) -> O(radius) first-order bias
 * - K = 2: symmetric real nudge (+-radius)    -> O(radius^2) bias
 * - K = 4: holomorphic 4-point Fourier circle -> O(radius^4) bias (cancels O(r), O(r^2), O(r^3))
 * Sets *out_fd_evals = 0.
 */
bool eshkol_hep_fourier_grad(const eshkol_hep_net_t* net,
                             const double* input,
                             const double* target,
                             double radius,
                             uint32_t num_fourier_points,
                             uint32_t relax_steps,
                             double relax_lr,
                             double* out_grads,
                             uint32_t grads_len,
                             uint64_t* out_fd_evals);

/* Exact analytical backpropagation gradients of 0.5 * ||y_hat - target||^2. */
bool eshkol_hep_exact_backprop_grad(const eshkol_hep_net_t* net,
                                    const double* input,
                                    const double* target,
                                    double* out_grads,
                                    uint32_t grads_len,
                                    double* out_loss);

/*
 * Non-Hermitian Complex Holomorphic Equilibrium Propagation (Wirtinger Check):
 * Recurrent equilibrium z* = W_rec * tanh(z*) + u (with arbitrary non-Hermitian,
 * non-symmetric complex W_rec in C^{d x d}, ||W_rec|| < 0.4), readout y_hat = W_out * tanh(z*),
 * holomorphic loss L_C = 0.5 * (y - y_hat)^T (y - y_hat), real objective L = Re(L_C).
 * Evaluates L = Re(L_C) and computes exact Wirtinger partials dL/dRe(W_rec) and
 * dL/dIm(W_rec) via K-point complex Cauchy-Fourier contour integration (fd_evals == 0).
 */
bool eshkol_hep_nonhermitian_eval_loss(uint32_t d,
                                       uint32_t m,
                                       const double* w_rec_re,
                                       const double* w_rec_im,
                                       const double* w_out_re,
                                       const double* w_out_im,
                                       const double* u_re,
                                       const double* u_im,
                                       const double* y_re,
                                       const double* y_im,
                                       uint32_t relax_steps,
                                       double* out_loss_re);

bool eshkol_hep_nonhermitian_wirtinger_grad(uint32_t d,
                                            uint32_t m,
                                            const double* w_rec_re,
                                            const double* w_rec_im,
                                            const double* w_out_re,
                                            const double* w_out_im,
                                            const double* u_re,
                                            const double* u_im,
                                            const double* y_re,
                                            const double* y_im,
                                            double radius,
                                            uint32_t num_fourier_points,
                                            uint32_t relax_steps,
                                            double relax_lr,
                                            double* out_grad_re,
                                            double* out_grad_im,
                                            uint64_t* out_fd_evals);

/* ============================================================================
 * 2. Wasserstein-Fisher-Rao (WFR) Particle Flow on Separated Bimodal Mixtures
 * ============================================================================ */

typedef struct eshkol_wfr_bimodal_target {
    double mean1;
    double mean2;
    double sigma1;
    double sigma2;
    double weight1; /* Target mode-1 mass in (0, 1); mode-2 mass = 1 - weight1 */
} eshkol_wfr_bimodal_target_t;

typedef struct eshkol_wfr_config {
    uint32_t num_particles;
    double init_mode1_fraction; /* Deliberately imbalanced initial mode-1 particle weight */
    uint32_t num_steps;
    double eta_wasserstein;     /* Transport step size */
    double eta_fisher_rao;      /* Birth-death weight step size (0.0 = pure Wasserstein!) */
    double kde_bandwidth;       /* Gaussian KDE bandwidth h > 0 */
    uint64_t seed;
} eshkol_wfr_config_t;

typedef struct eshkol_wfr_result {
    double mode1_mass;          /* Final particle mass on mode-1 side of midpoint */
    double mode2_mass;          /* Final particle mass on mode-2 side of midpoint */
    double mode1_mass_error;    /* |mode1_mass - target.weight1| */
    double mode1_mean;          /* Weighted mean of particles in mode 1 */
    double mode2_mean;          /* Weighted mean of particles in mode 2 */
    bool valid;
} eshkol_wfr_result_t;

/* Run WFR (or pure Wasserstein when eta_fisher_rao == 0) on a separated bimodal target. */
bool eshkol_wfr_run_bimodal(const eshkol_wfr_bimodal_target_t* target,
                            const eshkol_wfr_config_t* cfg,
                            eshkol_wfr_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_HOLOMORPHIC_WFR_H */
