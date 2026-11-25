/*
 * Holomorphic Equilibrium Propagation (hEP) & Wasserstein-Fisher-Rao (WFR) Flow
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/holomorphic_wfr.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kTwoPi = 2.0 * kPi;

using cd = std::complex<double>;

inline uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline double uniform01(uint64_t& state) {
    uint64_t r = splitmix64_next(state) >> 11;
    return (static_cast<double>(r) + 0.5) * (1.0 / 9007199254740992.0);
}

inline double normal01(uint64_t& state) {
    double u1 = std::max(1e-15, uniform01(state));
    double u2 = uniform01(state);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
}

struct HepLayer {
    uint32_t in_dim = 0;
    uint32_t out_dim = 0;
    std::vector<double> W;
    std::vector<double> b;
    uint32_t param_offset = 0;
};

inline cd c_tanh(cd z) {
    return std::tanh(z);
}

inline cd c_tanh_deriv(cd z) {
    cd t = std::tanh(z);
    return cd(1.0, 0.0) - t * t;
}

} /* anonymous namespace */

/* ============================================================================
 * 1. Holomorphic Equilibrium Propagation (hEP)
 * ============================================================================ */

struct eshkol_hep_net {
    std::vector<uint32_t> dims;
    std::vector<HepLayer> layers;
    uint32_t total_params = 0;

    eshkol_hep_net(const uint32_t* d, uint32_t nl, uint64_t seed)
        : dims(d, d + nl) {
        uint64_t rng = seed ^ 0x3c6ef372fe94f82bULL;
        layers.resize(nl - 1u);
        for (uint32_t l = 0; l + 1 < nl; l++) {
            HepLayer& ly = layers[l];
            ly.in_dim = dims[l];
            ly.out_dim = dims[l + 1u];
            ly.param_offset = total_params;
            ly.W.resize(static_cast<size_t>(ly.out_dim) * ly.in_dim);
            ly.b.resize(ly.out_dim);
            double scale = 0.55 / std::sqrt(static_cast<double>(ly.in_dim));
            for (double& w : ly.W) w = (uniform01(rng) * 2.0 - 1.0) * scale;
            for (double& bi : ly.b) bi = (uniform01(rng) * 2.0 - 1.0) * 0.25;
            total_params += static_cast<uint32_t>(ly.W.size() + ly.b.size());
        }
    }

    void forward_init(const double* input, std::vector<std::vector<cd>>& z) const {
        const size_t L = layers.size();
        z.resize(L + 1u);
        z[0].resize(dims[0]);
        for (uint32_t j = 0; j < dims[0]; j++) z[0][j] = cd(input[j], 0.0);
        for (size_t l = 0; l < L; l++) {
            const HepLayer& ly = layers[l];
            z[l + 1u].assign(ly.out_dim, cd(0.0, 0.0));
            for (uint32_t i = 0; i < ly.out_dim; i++) {
                cd acc(ly.b[i], 0.0);
                for (uint32_t j = 0; j < ly.in_dim; j++) {
                    cd prev = (l == 0) ? z[0][j] : c_tanh(z[l][j]);
                    acc += ly.W[i * ly.in_dim + j] * prev;
                }
                z[l + 1u][i] = acc;
            }
        }
    }

    /* Solve holomorphic equilibrium dPhi/dz_l = 0 for complex nudging beta in C. */
    void solve_holomorphic_equilibrium(const std::vector<std::vector<cd>>& z_init,
                                       const double* target,
                                       cd beta,
                                       uint32_t relax_steps,
                                       double relax_lr,
                                       std::vector<std::vector<cd>>& z,
                                       std::vector<std::vector<cd>>& eps) const {
        const size_t L = layers.size();
        z = z_init;
        std::vector<std::vector<cd>> mu(L + 1u);
        eps.resize(L + 1u);
        for (size_t l = 1; l <= L; l++) {
            mu[l].assign(dims[l], cd(0.0, 0.0));
            eps[l].assign(dims[l], cd(0.0, 0.0));
        }

        auto eval_eps = [&]() {
            for (size_t l = 0; l < L; l++) {
                const HepLayer& ly = layers[l];
                for (uint32_t i = 0; i < ly.out_dim; i++) {
                    cd acc(ly.b[i], 0.0);
                    for (uint32_t j = 0; j < ly.in_dim; j++) {
                        cd prev = (l == 0) ? z[0][j] : c_tanh(z[l][j]);
                        acc += ly.W[i * ly.in_dim + j] * prev;
                    }
                    mu[l + 1u][i] = acc;
                }
            }
            for (size_t l = 1; l < L; l++) {
                for (uint32_t i = 0; i < dims[l]; i++) {
                    eps[l][i] = z[l][i] - mu[l][i];
                }
            }
            for (uint32_t i = 0; i < dims[L]; i++) {
                eps[L][i] = beta * (cd(target[i], 0.0) - mu[L][i]);
            }
        };

        std::vector<std::vector<cd>> dz(L);
        for (size_t l = 1; l < L; l++) dz[l].assign(dims[l], cd(0.0, 0.0));

        for (uint32_t it = 0; it < relax_steps; it++) {
            eval_eps();
            double max_res = 0.0;
            for (size_t l = 1; l < L; l++) {
                const HepLayer& next_ly = layers[l];
                for (uint32_t j = 0; j < dims[l]; j++) {
                    cd top_down(0.0, 0.0);
                    for (uint32_t i = 0; i < next_ly.out_dim; i++) {
                        top_down += next_ly.W[i * next_ly.in_dim + j] * eps[l + 1u][i];
                    }
                    cd g = -eps[l][j] + c_tanh_deriv(z[l][j]) * top_down;
                    dz[l][j] = g;
                    max_res = std::max(max_res, std::abs(g));
                }
            }
            for (size_t l = 1; l < L; l++) {
                for (uint32_t j = 0; j < dims[l]; j++) {
                    z[l][j] += relax_lr * dz[l][j];
                }
            }
            if (max_res < 1e-14) break;
        }
        eval_eps();
    }

    bool fourier_grad(const double* input,
                      const double* target,
                      double radius,
                      uint32_t K,
                      uint32_t relax_steps,
                      double relax_lr,
                      double* out_grads,
                      uint32_t grads_len,
                      uint64_t* out_fd_evals) const {
        if (!input || !target || !out_grads || grads_len < total_params) return false;
        if (!std::isfinite(radius) || radius <= 0.0 || K == 0 || relax_steps == 0 || relax_lr <= 0.0) {
            return false;
        }
        if (out_fd_evals) *out_fd_evals = 0;

        std::vector<std::vector<cd>> z_free;
        forward_init(input, z_free);

        std::vector<cd> accum(total_params, cd(0.0, 0.0));
        std::vector<std::vector<cd>> z_k, eps_k;
        const size_t L = layers.size();

        for (uint32_t k = 0; k < K; k++) {
            double theta = (kTwoPi * static_cast<double>(k)) / static_cast<double>(K);
            cd beta(radius * std::cos(theta), radius * std::sin(theta));
            cd weight = cd(std::cos(-theta), std::sin(-theta)) / (radius * static_cast<double>(K));

            solve_holomorphic_equilibrium(z_free, target, beta, relax_steps, relax_lr, z_k, eps_k);

            /* Local holomorphic energy gradient g_W(beta) = -eps_{l+1} * sigma(z_l)^T */
            for (size_t l = 0; l < L; l++) {
                const HepLayer& ly = layers[l];
                uint32_t off_w = ly.param_offset;
                uint32_t off_b = off_w + static_cast<uint32_t>(ly.W.size());
                for (uint32_t i = 0; i < ly.out_dim; i++) {
                    accum[off_b + i] += weight * (-eps_k[l + 1u][i]);
                    for (uint32_t j = 0; j < ly.in_dim; j++) {
                        cd prev = (l == 0) ? z_k[0][j] : c_tanh(z_k[l][j]);
                        accum[off_w + i * ly.in_dim + j] += weight * (-eps_k[l + 1u][i] * prev);
                    }
                }
            }
        }

        for (uint32_t p = 0; p < total_params; p++) {
            out_grads[p] = accum[p].real();
        }
        return true;
    }

    bool exact_backprop_grad(const double* input,
                             const double* target,
                             double* out_grads,
                             uint32_t grads_len,
                             double* out_loss) const {
        if (!input || !target || !out_grads || grads_len < total_params) return false;
        std::vector<std::vector<cd>> zc;
        forward_init(input, zc);
        const size_t L = layers.size();

        double loss = 0.0;
        std::vector<std::vector<double>> delta(L + 1u);
        delta[L].resize(dims[L]);
        for (uint32_t i = 0; i < dims[L]; i++) {
            double diff = zc[L][i].real() - target[i];
            loss += 0.5 * diff * diff;
            delta[L][i] = diff;
        }
        if (out_loss) *out_loss = loss;

        for (size_t l = L; l-- > 1u; ) {
            const HepLayer& next_ly = layers[l];
            delta[l].assign(dims[l], 0.0);
            for (uint32_t j = 0; j < dims[l]; j++) {
                double acc = 0.0;
                for (uint32_t i = 0; i < next_ly.out_dim; i++) {
                    acc += next_ly.W[i * next_ly.in_dim + j] * delta[l + 1u][i];
                }
                double t = std::tanh(zc[l][j].real());
                delta[l][j] = acc * (1.0 - t * t);
            }
        }

        for (size_t l = 0; l < L; l++) {
            const HepLayer& ly = layers[l];
            double* gw = out_grads + ly.param_offset;
            double* gb = gw + ly.W.size();
            for (uint32_t i = 0; i < ly.out_dim; i++) {
                gb[i] = delta[l + 1u][i];
                for (uint32_t j = 0; j < ly.in_dim; j++) {
                    double prev = (l == 0) ? zc[0][j].real() : std::tanh(zc[l][j].real());
                    gw[i * ly.in_dim + j] = delta[l + 1u][i] * prev;
                }
            }
        }
        return true;
    }
};

extern "C" eshkol_hep_net_t* eshkol_hep_create(const uint32_t* layer_dims,
                                               uint32_t num_layers,
                                               uint64_t seed) {
    if (!layer_dims || num_layers < 2) return nullptr;
    for (uint32_t i = 0; i < num_layers; i++) {
        if (layer_dims[i] == 0) return nullptr;
    }
    return new (std::nothrow) eshkol_hep_net(layer_dims, num_layers, seed);
}

extern "C" void eshkol_hep_destroy(eshkol_hep_net_t* net) {
    delete net;
}

extern "C" uint32_t eshkol_hep_total_params(const eshkol_hep_net_t* net) {
    return net ? net->total_params : 0u;
}

extern "C" bool eshkol_hep_fourier_grad(const eshkol_hep_net_t* net,
                                        const double* input,
                                        const double* target,
                                        double radius,
                                        uint32_t num_fourier_points,
                                        uint32_t relax_steps,
                                        double relax_lr,
                                        double* out_grads,
                                        uint32_t grads_len,
                                        uint64_t* out_fd_evals) {
    if (!net) return false;
    return net->fourier_grad(input, target, radius, num_fourier_points,
                             relax_steps, relax_lr, out_grads, grads_len, out_fd_evals);
}

extern "C" bool eshkol_hep_exact_backprop_grad(const eshkol_hep_net_t* net,
                                               const double* input,
                                               const double* target,
                                               double* out_grads,
                                               uint32_t grads_len,
                                               double* out_loss) {
    if (!net) return false;
    return net->exact_backprop_grad(input, target, out_grads, grads_len, out_loss);
}

namespace {

bool solve_nonhermitian_free(uint32_t d,
                             const std::vector<cd>& W_rec,
                             const std::vector<cd>& u,
                             uint32_t max_steps,
                             std::vector<cd>& z_out) {
    z_out = u;
    std::vector<cd> next_z(d, cd(0.0, 0.0));
    for (uint32_t it = 0; it < max_steps; it++) {
        double max_diff = 0.0;
        for (uint32_t i = 0; i < d; i++) {
            cd acc = u[i];
            for (uint32_t j = 0; j < d; j++) {
                acc += W_rec[i * d + j] * c_tanh(z_out[j]);
            }
            next_z[i] = acc;
            max_diff = std::max(max_diff, std::abs(acc - z_out[i]));
        }
        z_out.swap(next_z);
        if (max_diff < 1e-14) return true;
    }
    return true;
}

void solve_nonhermitian_nudged(uint32_t d,
                               uint32_t m,
                               const std::vector<cd>& W_rec,
                               const std::vector<cd>& W_out,
                               const std::vector<cd>& u,
                               const std::vector<cd>& y,
                               const std::vector<cd>& z_init,
                               cd beta,
                               uint32_t relax_steps,
                               double relax_lr,
                               std::vector<cd>& z,
                               std::vector<cd>& eps1) {
    z = z_init;
    eps1.assign(d, cd(0.0, 0.0));
    std::vector<cd> s(d), sp(d), eps2(m), grad_z(d);

    auto compute_eps = [&]() {
        for (uint32_t j = 0; j < d; j++) {
            s[j] = c_tanh(z[j]);
            sp[j] = c_tanh_deriv(z[j]);
        }
        for (uint32_t i = 0; i < d; i++) {
            cd mu = u[i];
            for (uint32_t j = 0; j < d; j++) mu += W_rec[i * d + j] * s[j];
            eps1[i] = z[i] - mu;
        }
        for (uint32_t k = 0; k < m; k++) {
            cd yhat(0.0, 0.0);
            for (uint32_t j = 0; j < d; j++) yhat += W_out[k * d + j] * s[j];
            eps2[k] = beta * (y[k] - yhat);
        }
    };

    for (uint32_t it = 0; it < relax_steps; it++) {
        compute_eps();
        double max_res = 0.0;
        for (uint32_t j = 0; j < d; j++) {
            cd rec_fb(0.0, 0.0);
            for (uint32_t i = 0; i < d; i++) rec_fb += W_rec[i * d + j] * eps1[i];
            cd out_fb(0.0, 0.0);
            for (uint32_t k = 0; k < m; k++) out_fb += W_out[k * d + j] * eps2[k];
            /* Holomorphic stationarity dPhi/dz_j = eps1_j - sigma'(z_j) * (W_rec^T eps1 + W_out^T eps2) */
            cd res = eps1[j] - sp[j] * (rec_fb + out_fb);
            grad_z[j] = res;
            max_res = std::max(max_res, std::abs(res));
        }
        for (uint32_t j = 0; j < d; j++) {
            z[j] -= relax_lr * grad_z[j];
        }
        if (max_res < 1e-14) break;
    }
    compute_eps();
}

} /* anonymous namespace */

extern "C" bool eshkol_hep_nonhermitian_eval_loss(uint32_t d,
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
                                                  double* out_loss_re) {
    if (d == 0 || m == 0 || !w_rec_re || !w_rec_im || !w_out_re || !w_out_im ||
        !u_re || !u_im || !y_re || !y_im || !out_loss_re) {
        return false;
    }
    std::vector<cd> W_rec(static_cast<size_t>(d) * d);
    std::vector<cd> W_out(static_cast<size_t>(m) * d);
    std::vector<cd> u(d), y(m);
    for (size_t i = 0; i < W_rec.size(); i++) W_rec[i] = cd(w_rec_re[i], w_rec_im[i]);
    for (size_t i = 0; i < W_out.size(); i++) W_out[i] = cd(w_out_re[i], w_out_im[i]);
    for (uint32_t i = 0; i < d; i++) u[i] = cd(u_re[i], u_im[i]);
    for (uint32_t k = 0; k < m; k++) y[k] = cd(y_re[k], y_im[k]);

    std::vector<cd> z_star;
    solve_nonhermitian_free(d, W_rec, u, relax_steps > 0 ? relax_steps : 300u, z_star);

    cd loss_c(0.0, 0.0);
    for (uint32_t k = 0; k < m; k++) {
        cd yhat(0.0, 0.0);
        for (uint32_t j = 0; j < d; j++) yhat += W_out[k * d + j] * c_tanh(z_star[j]);
        cd diff = y[k] - yhat;
        loss_c += cd(0.5, 0.0) * diff * diff;
    }
    *out_loss_re = loss_c.real();
    return std::isfinite(*out_loss_re);
}

extern "C" bool eshkol_hep_nonhermitian_wirtinger_grad(uint32_t d,
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
                                                       uint64_t* out_fd_evals) {
    if (d == 0 || m == 0 || !w_rec_re || !w_rec_im || !w_out_re || !w_out_im ||
        !u_re || !u_im || !y_re || !y_im || !out_grad_re || !out_grad_im) {
        return false;
    }
    if (!std::isfinite(radius) || radius <= 0.0 || num_fourier_points == 0 ||
        relax_steps == 0 || relax_lr <= 0.0) {
        return false;
    }
    if (out_fd_evals) *out_fd_evals = 0;

    const size_t dd = static_cast<size_t>(d) * d;
    std::vector<cd> W_rec(dd);
    std::vector<cd> W_out(static_cast<size_t>(m) * d);
    std::vector<cd> u(d), y(m);
    for (size_t i = 0; i < dd; i++) W_rec[i] = cd(w_rec_re[i], w_rec_im[i]);
    for (size_t i = 0; i < W_out.size(); i++) W_out[i] = cd(w_out_re[i], w_out_im[i]);
    for (uint32_t i = 0; i < d; i++) u[i] = cd(u_re[i], u_im[i]);
    for (uint32_t k = 0; k < m; k++) y[k] = cd(y_re[k], y_im[k]);

    std::vector<cd> z_free;
    solve_nonhermitian_free(d, W_rec, u, relax_steps, z_free);

    std::vector<cd> accum(dd, cd(0.0, 0.0));
    std::vector<cd> z_k, eps1_k;
    for (uint32_t k = 0; k < num_fourier_points; k++) {
        double theta = (kTwoPi * static_cast<double>(k)) / static_cast<double>(num_fourier_points);
        cd beta(radius * std::cos(theta), radius * std::sin(theta));
        cd weight = cd(std::cos(-theta), std::sin(-theta)) /
                    (radius * static_cast<double>(num_fourier_points));

        solve_nonhermitian_nudged(d, m, W_rec, W_out, u, y, z_free,
                                  beta, relax_steps, relax_lr, z_k, eps1_k);
        for (uint32_t i = 0; i < d; i++) {
            for (uint32_t j = 0; j < d; j++) {
                /* Local holomorphic partial dPhi/dW_{ij} = -eps1_i * tanh(z_j) */
                accum[i * d + j] += weight * (-eps1_k[i] * c_tanh(z_k[j]));
            }
        }
    }

    /* Cauchy-Riemann / Wirtinger identity for L = Re(L_C):
     *   dL/dRe(W) =  Re(dL_C/dW)
     *   dL/dIm(W) = -Im(dL_C/dW) */
    for (size_t i = 0; i < dd; i++) {
        out_grad_re[i] = accum[i].real();
        out_grad_im[i] = -accum[i].imag();
    }
    return true;
}

/* ============================================================================
 * 2. Wasserstein-Fisher-Rao (WFR) Particle Flow on Separated Bimodal Mixtures
 * ============================================================================ */

extern "C" bool eshkol_wfr_run_bimodal(const eshkol_wfr_bimodal_target_t* target,
                                       const eshkol_wfr_config_t* cfg,
                                       eshkol_wfr_result_t* out_result) {
    if (!target || !cfg || !out_result) return false;
    std::memset(out_result, 0, sizeof(*out_result));

    if (cfg->num_particles < 4 || cfg->num_steps == 0 ||
        target->sigma1 <= 0.0 || target->sigma2 <= 0.0 ||
        target->weight1 <= 0.0 || target->weight1 >= 1.0 ||
        cfg->init_mode1_fraction <= 0.0 || cfg->init_mode1_fraction >= 1.0 ||
        cfg->kde_bandwidth <= 0.0 || cfg->eta_wasserstein < 0.0 || cfg->eta_fisher_rao < 0.0) {
        return false;
    }

    const uint32_t N = cfg->num_particles;
    const double w1_tgt = target->weight1;
    const double w2_tgt = 1.0 - w1_tgt;
    const double inv_s1_sq = 1.0 / (target->sigma1 * target->sigma1);
    const double inv_s2_sq = 1.0 / (target->sigma2 * target->sigma2);
    const double norm1 = w1_tgt / (std::sqrt(kTwoPi) * target->sigma1);
    const double norm2 = w2_tgt / (std::sqrt(kTwoPi) * target->sigma2);

    /* Potential V(x) = -log pi(x) and exact score grad V(x) */
    auto eval_potential = [&](double x, double& out_v, double& out_dv) {
        double e1 = -0.5 * (x - target->mean1) * (x - target->mean1) * inv_s1_sq;
        double e2 = -0.5 * (x - target->mean2) * (x - target->mean2) * inv_s2_sq;
        double l1 = std::log(norm1) + e1;
        double l2 = std::log(norm2) + e2;
        double m = std::max(l1, l2);
        double p1 = std::exp(l1 - m);
        double p2 = std::exp(l2 - m);
        double denom = p1 + p2;
        out_v = -(m + std::log(denom));
        double r1 = p1 / denom;
        double r2 = p2 / denom;
        out_dv = r1 * (x - target->mean1) * inv_s1_sq + r2 * (x - target->mean2) * inv_s2_sq;
    };

    /* Initialize particles with equal counts near both modes (or imbalanced weights) */
    uint64_t rng = cfg->seed ^ 0x510e527fade682d1ULL;
    const uint32_t n1 = N / 2u;
    const uint32_t n2 = N - n1;
    std::vector<double> x(N, 0.0);
    std::vector<double> log_a(N, 0.0);
    std::vector<double> a(N, 0.0);

    for (uint32_t i = 0; i < n1; i++) {
        /* Spread deterministically across mode 1 */
        x[i] = target->mean1 + target->sigma1 * (0.9 * normal01(rng));
        a[i] = cfg->init_mode1_fraction / static_cast<double>(n1);
        log_a[i] = std::log(a[i]);
    }
    for (uint32_t i = n1; i < N; i++) {
        x[i] = target->mean2 + target->sigma2 * (0.9 * normal01(rng));
        a[i] = (1.0 - cfg->init_mode1_fraction) / static_cast<double>(n2);
        log_a[i] = std::log(a[i]);
    }

    const double h = cfg->kde_bandwidth;
    const double inv_h2 = 1.0 / (h * h);
    const double k_norm = 1.0 / (std::sqrt(kTwoPi) * h);

    std::vector<double> rho(N, 0.0);
    std::vector<double> grad_log_rho(N, 0.0);
    std::vector<double> pot_v(N, 0.0);
    std::vector<double> pot_dv(N, 0.0);

    for (uint32_t step = 0; step < cfg->num_steps; step++) {
        /* Evaluate Gaussian KDE rho(x_i), grad log rho(x_i), and V(x_i), grad V(x_i) */
        for (uint32_t i = 0; i < N; i++) {
            eval_potential(x[i], pot_v[i], pot_dv[i]);
            double r = 0.0;
            double dr = 0.0;
            for (uint32_t j = 0; j < N; j++) {
                double diff = x[i] - x[j];
                double k_ij = k_norm * std::exp(-0.5 * diff * diff * inv_h2);
                r += a[j] * k_ij;
                dr += a[j] * k_ij * (-diff * inv_h2);
            }
            rho[i] = std::max(r, 1e-300);
            grad_log_rho[i] = dr / rho[i];
        }

        /* Step A: Wasserstein transport step x_i <- x_i - eta_W * (grad V + grad log rho) */
        if (cfg->eta_wasserstein > 0.0) {
            for (uint32_t i = 0; i < N; i++) {
                double vel = pot_dv[i] + grad_log_rho[i];
                /* Guard extreme outliers */
                if (vel > 50.0) vel = 50.0;
                if (vel < -50.0) vel = -50.0;
                x[i] -= cfg->eta_wasserstein * vel;
            }
        }

        /* Step B: Fisher-Rao birth-death weight update (when eta_fisher_rao > 0) */
        if (cfg->eta_fisher_rao > 0.0) {
            double g_bar = 0.0;
            std::vector<double> chem(N, 0.0);
            for (uint32_t i = 0; i < N; i++) {
                chem[i] = std::log(rho[i]) + pot_v[i];
                g_bar += a[i] * chem[i];
            }
            double max_la = -std::numeric_limits<double>::infinity();
            for (uint32_t i = 0; i < N; i++) {
                log_a[i] -= cfg->eta_fisher_rao * (chem[i] - g_bar);
                if (log_a[i] > max_la) max_la = log_a[i];
            }
            double sum_a = 0.0;
            for (uint32_t i = 0; i < N; i++) {
                a[i] = std::exp(log_a[i] - max_la);
                sum_a += a[i];
            }
            double log_sum = max_la + std::log(sum_a);
            for (uint32_t i = 0; i < N; i++) {
                a[i] /= sum_a;
                log_a[i] -= log_sum;
            }
        }
    }

    const double midpoint = 0.5 * (target->mean1 + target->mean2);
    double m1 = 0.0, m2 = 0.0;
    double mu1 = 0.0, mu2 = 0.0;
    for (uint32_t i = 0; i < N; i++) {
        bool in_mode1 = (target->mean1 < target->mean2) ? (x[i] < midpoint) : (x[i] > midpoint);
        if (in_mode1) {
            m1 += a[i];
            mu1 += a[i] * x[i];
        } else {
            m2 += a[i];
            mu2 += a[i] * x[i];
        }
    }

    out_result->mode1_mass = m1;
    out_result->mode2_mass = m2;
    out_result->mode1_mass_error = std::fabs(m1 - w1_tgt);
    out_result->mode1_mean = (m1 > 1e-12) ? (mu1 / m1) : target->mean1;
    out_result->mode2_mean = (m2 > 1e-12) ? (mu2 / m2) : target->mean2;
    out_result->valid = std::isfinite(m1) && std::isfinite(m2);
    return out_result->valid;
}
