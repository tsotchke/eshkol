/*
 * Trajectory-Balance GFlowNet & Predictive Coding Network (PCN) Energy Core
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/gflownet_ebm.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {

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

struct GfnEdge {
    uint32_t from = 0;
    uint32_t to = 0;
    double logit = 0.0;
};

} /* anonymous namespace */

/* ============================================================================
 * 1. Trajectory-Balance (TB) GFlowNet Over Finite DAGs
 * ============================================================================ */

struct eshkol_gfn_dag {
    uint32_t num_states = 0;
    std::vector<GfnEdge> edges;
    std::vector<std::vector<uint32_t>> out_edges;
    std::vector<std::vector<uint32_t>> in_edges;
    std::vector<uint8_t> is_terminal;
    std::vector<double> reward;
    double log_z = 0.0;

    explicit eshkol_gfn_dag(uint32_t ns)
        : num_states(ns),
          out_edges(ns),
          in_edges(ns),
          is_terminal(ns, 0u),
          reward(ns, 0.0) {}

    void softmax_out(uint32_t u, std::vector<double>& probs, std::vector<double>& log_probs) const {
        const auto& oe = out_edges[u];
        probs.resize(oe.size());
        log_probs.resize(oe.size());
        if (oe.empty()) return;
        double m = -std::numeric_limits<double>::infinity();
        for (uint32_t eid : oe) m = std::max(m, edges[eid].logit);
        double sum = 0.0;
        for (size_t i = 0; i < oe.size(); i++) {
            probs[i] = std::exp(edges[oe[i]].logit - m);
            sum += probs[i];
        }
        double log_sum = m + std::log(sum);
        for (size_t i = 0; i < oe.size(); i++) {
            probs[i] /= sum;
            log_probs[i] = edges[oe[i]].logit - log_sum;
        }
    }

    bool tb_loss_and_grad(const uint32_t* edge_seq,
                          uint32_t seq_len,
                          double* out_loss,
                          double* out_grad_log_z,
                          double* out_grad_logits,
                          uint32_t logits_len,
                          uint64_t* out_fd_evals) const {
        if (!edge_seq || seq_len == 0) return false;
        if (out_grad_logits && logits_len < edges.size()) return false;
        if (out_fd_evals) *out_fd_evals = 0;

        uint32_t cur = 0;
        double sum_log_pf = 0.0;
        double sum_log_pb = 0.0;

        std::vector<std::vector<double>> step_probs(seq_len);
        std::vector<size_t> chosen_local(seq_len, 0);

        for (uint32_t t = 0; t < seq_len; t++) {
            uint32_t eid = edge_seq[t];
            if (eid >= edges.size() || edges[eid].from != cur) return false;
            std::vector<double> probs, log_probs;
            softmax_out(cur, probs, log_probs);

            const auto& oe = out_edges[cur];
            size_t loc = oe.size();
            for (size_t i = 0; i < oe.size(); i++) {
                if (oe[i] == eid) { loc = i; break; }
            }
            if (loc == oe.size()) return false;

            sum_log_pf += log_probs[loc];
            uint32_t nxt = edges[eid].to;
            if (in_edges[nxt].empty()) return false;
            sum_log_pb += -std::log(static_cast<double>(in_edges[nxt].size()));

            step_probs[t] = std::move(probs);
            chosen_local[t] = loc;
            cur = nxt;
        }

        if (!is_terminal[cur] || reward[cur] <= 0.0) return false;

        double delta = log_z + sum_log_pf - std::log(reward[cur]) - sum_log_pb;
        if (out_loss) *out_loss = 0.5 * delta * delta;
        if (out_grad_log_z) *out_grad_log_z = delta;

        if (out_grad_logits) {
            std::memset(out_grad_logits, 0, edges.size() * sizeof(double));
            uint32_t u = 0;
            for (uint32_t t = 0; t < seq_len; t++) {
                const auto& oe = out_edges[u];
                for (size_t i = 0; i < oe.size(); i++) {
                    double ind = (i == chosen_local[t]) ? 1.0 : 0.0;
                    out_grad_logits[oe[i]] += delta * (ind - step_probs[t][i]);
                }
                u = edges[edge_seq[t]].to;
            }
        }
        return true;
    }

    bool sample_trajectory(uint64_t& rng, std::vector<uint32_t>& out_edges_seq) const {
        out_edges_seq.clear();
        uint32_t cur = 0;
        std::vector<double> probs, log_probs;
        uint32_t guard = 0;
        while (!is_terminal[cur]) {
            if (out_edges[cur].empty() || ++guard > num_states) return false;
            softmax_out(cur, probs, log_probs);
            double u = uniform01(rng);
            double cdf = 0.0;
            size_t pick = probs.size() - 1u;
            for (size_t i = 0; i < probs.size(); i++) {
                cdf += probs[i];
                if (u <= cdf) { pick = i; break; }
            }
            uint32_t eid = out_edges[cur][pick];
            out_edges_seq.push_back(eid);
            cur = edges[eid].to;
        }
        return !out_edges_seq.empty();
    }

    bool train(uint32_t num_steps, double lr_logits, double lr_log_z, uint64_t seed) {
        if (edges.empty() || num_steps == 0) return false;
        const size_t E = edges.size();
        std::vector<double> m_l(E, 0.0), v_l(E, 0.0);
        double m_z = 0.0, v_z = 0.0;
        const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
        uint64_t rng = seed ^ 0x6a09e667f3bcc908ULL;

        std::vector<uint32_t> traj;
        std::vector<double> g_l(E, 0.0);
        std::vector<double> acc_g_l(E, 0.0);
        const uint32_t batch = 8u;

        for (uint32_t step = 1; step <= num_steps; step++) {
            std::fill(acc_g_l.begin(), acc_g_l.end(), 0.0);
            double acc_g_z = 0.0;
            for (uint32_t b = 0; b < batch; b++) {
                if (!sample_trajectory(rng, traj)) return false;
                double gz = 0.0;
                if (!tb_loss_and_grad(traj.data(), static_cast<uint32_t>(traj.size()),
                                      nullptr, &gz, g_l.data(), static_cast<uint32_t>(E), nullptr)) {
                    return false;
                }
                acc_g_z += gz;
                for (size_t i = 0; i < E; i++) acc_g_l[i] += g_l[i];
            }
            const double inv_b = 1.0 / static_cast<double>(batch);
            acc_g_z *= inv_b;
            for (size_t i = 0; i < E; i++) acc_g_l[i] *= inv_b;

            double b1t = 1.0 - std::pow(b1, static_cast<double>(step));
            double b2t = 1.0 - std::pow(b2, static_cast<double>(step));

            m_z = b1 * m_z + (1.0 - b1) * acc_g_z;
            v_z = b2 * v_z + (1.0 - b2) * acc_g_z * acc_g_z;
            log_z -= lr_log_z * (m_z / b1t) / (std::sqrt(v_z / b2t) + eps);

            for (size_t i = 0; i < E; i++) {
                m_l[i] = b1 * m_l[i] + (1.0 - b1) * acc_g_l[i];
                v_l[i] = b2 * v_l[i] + (1.0 - b2) * acc_g_l[i] * acc_g_l[i];
                edges[i].logit -= lr_logits * (m_l[i] / b1t) / (std::sqrt(v_l[i] / b2t) + eps);
            }
        }
        return true;
    }

    bool eval_distribution(double* out_state_probs,
                           uint32_t probs_len,
                           double* out_tv,
                           double* out_exact_log_z) const {
        if (out_state_probs && probs_len < num_states) return false;
        std::vector<double> flow(num_states, 0.0);
        flow[0] = 1.0;

        std::vector<double> probs, log_probs;
        for (uint32_t u = 0; u < num_states; u++) {
            if (is_terminal[u] || out_edges[u].empty() || flow[u] == 0.0) continue;
            softmax_out(u, probs, log_probs);
            for (size_t i = 0; i < out_edges[u].size(); i++) {
                uint32_t v = edges[out_edges[u][i]].to;
                flow[v] += flow[u] * probs[i];
            }
        }

        double z_exact = 0.0;
        for (uint32_t s = 0; s < num_states; s++) {
            if (is_terminal[s]) z_exact += reward[s];
        }
        if (z_exact <= 0.0) return false;

        double tv = 0.0;
        for (uint32_t s = 0; s < num_states; s++) {
            if (is_terminal[s]) {
                double p_target = reward[s] / z_exact;
                tv += std::fabs(flow[s] - p_target);
            }
        }
        tv *= 0.5;

        if (out_state_probs) {
            std::memcpy(out_state_probs, flow.data(), num_states * sizeof(double));
        }
        if (out_tv) *out_tv = tv;
        if (out_exact_log_z) *out_exact_log_z = std::log(z_exact);
        return true;
    }

    bool eval_per_state_tv(double* out_state_tv,
                           uint32_t tv_len,
                           double* out_max_state_tv,
                           double* out_max_terminal_rel_err) const {
        if (out_state_tv && tv_len < num_states) return false;

        /* 1. Forward state flows */
        std::vector<double> f_flow(num_states, 0.0);
        f_flow[0] = 1.0;
        std::vector<double> probs, log_probs;
        for (uint32_t u = 0; u < num_states; u++) {
            if (is_terminal[u] || out_edges[u].empty() || f_flow[u] == 0.0) continue;
            softmax_out(u, probs, log_probs);
            for (size_t i = 0; i < out_edges[u].size(); i++) {
                uint32_t v = edges[out_edges[u][i]].to;
                f_flow[v] += f_flow[u] * probs[i];
            }
        }

        /* 2. Exact backward target flows under uniform P_B(u|v) = 1/|in(v)| */
        double z_exact = 0.0;
        for (uint32_t s = 0; s < num_states; s++) {
            if (is_terminal[s]) z_exact += reward[s];
        }
        if (z_exact <= 0.0) return false;

        std::vector<double> b_flow(num_states, 0.0);
        std::vector<double> b_edge_flow(edges.size(), 0.0);
        for (uint32_t s = num_states; s-- > 0; ) {
            if (is_terminal[s]) {
                b_flow[s] = reward[s] / z_exact;
            }
            if (s > 0 && !in_edges[s].empty() && b_flow[s] > 0.0) {
                double pb = 1.0 / static_cast<double>(in_edges[s].size());
                for (uint32_t eid : in_edges[s]) {
                    double ef = b_flow[s] * pb;
                    b_edge_flow[eid] = ef;
                    b_flow[edges[eid].from] += ef;
                }
            }
        }

        /* 3. Per-state local TV and terminal relative error */
        double max_stv = 0.0;
        double max_rel = 0.0;
        for (uint32_t s = 0; s < num_states; s++) {
            double stv = 0.0;
            if (is_terminal[s]) {
                double p_tgt = reward[s] / z_exact;
                stv = std::fabs(f_flow[s] - p_tgt);
                double rel = stv / p_tgt;
                if (rel > max_rel) max_rel = rel;
            } else if (!out_edges[s].empty() && b_flow[s] > 0.0) {
                softmax_out(s, probs, log_probs);
                double acc = 0.0;
                for (size_t i = 0; i < out_edges[s].size(); i++) {
                    double q_b = b_edge_flow[out_edges[s][i]] / b_flow[s];
                    acc += std::fabs(probs[i] - q_b);
                }
                stv = 0.5 * acc;
            }
            if (out_state_tv) out_state_tv[s] = stv;
            if (stv > max_stv) max_stv = stv;
        }

        if (out_max_state_tv) *out_max_state_tv = max_stv;
        if (out_max_terminal_rel_err) *out_max_terminal_rel_err = max_rel;
        return true;
    }
};

extern "C" eshkol_gfn_dag_t* eshkol_gfn_create(uint32_t num_states) {
    if (num_states < 2) return nullptr;
    return new (std::nothrow) eshkol_gfn_dag(num_states);
}

extern "C" void eshkol_gfn_destroy(eshkol_gfn_dag_t* gfn) {
    delete gfn;
}

extern "C" uint32_t eshkol_gfn_add_edge(eshkol_gfn_dag_t* gfn,
                                        uint32_t from_state,
                                        uint32_t to_state,
                                        double init_logit) {
    if (!gfn || !std::isfinite(init_logit)) return UINT32_MAX;
    if (from_state >= gfn->num_states || to_state >= gfn->num_states || from_state >= to_state) {
        return UINT32_MAX; /* Enforce topological DAG order u < v */
    }
    uint32_t eid = static_cast<uint32_t>(gfn->edges.size());
    gfn->edges.push_back({from_state, to_state, init_logit});
    gfn->out_edges[from_state].push_back(eid);
    gfn->in_edges[to_state].push_back(eid);
    return eid;
}

extern "C" bool eshkol_gfn_set_terminal_reward(eshkol_gfn_dag_t* gfn,
                                               uint32_t state,
                                               double reward) {
    if (!gfn || state == 0 || state >= gfn->num_states || !std::isfinite(reward) || reward <= 0.0) {
        return false;
    }
    gfn->is_terminal[state] = 1u;
    gfn->reward[state] = reward;
    return true;
}

extern "C" uint32_t eshkol_gfn_num_edges(const eshkol_gfn_dag_t* gfn) {
    return gfn ? static_cast<uint32_t>(gfn->edges.size()) : 0u;
}

extern "C" double eshkol_gfn_get_log_z(const eshkol_gfn_dag_t* gfn) {
    return gfn ? gfn->log_z : 0.0;
}

extern "C" void eshkol_gfn_set_log_z(eshkol_gfn_dag_t* gfn, double log_z) {
    if (gfn && std::isfinite(log_z)) gfn->log_z = log_z;
}

extern "C" bool eshkol_gfn_get_edge_logit(const eshkol_gfn_dag_t* gfn,
                                          uint32_t edge_id,
                                          double* out_logit) {
    if (!gfn || !out_logit || edge_id >= gfn->edges.size()) return false;
    *out_logit = gfn->edges[edge_id].logit;
    return true;
}

extern "C" bool eshkol_gfn_set_edge_logit(eshkol_gfn_dag_t* gfn,
                                          uint32_t edge_id,
                                          double logit) {
    if (!gfn || !std::isfinite(logit) || edge_id >= gfn->edges.size()) return false;
    gfn->edges[edge_id].logit = logit;
    return true;
}

extern "C" bool eshkol_gfn_tb_loss_and_grad(const eshkol_gfn_dag_t* gfn,
                                            const uint32_t* edge_seq,
                                            uint32_t seq_len,
                                            double* out_loss,
                                            double* out_grad_log_z,
                                            double* out_grad_logits,
                                            uint32_t logits_len,
                                            uint64_t* out_fd_evals) {
    if (!gfn) return false;
    return gfn->tb_loss_and_grad(edge_seq, seq_len, out_loss, out_grad_log_z,
                                 out_grad_logits, logits_len, out_fd_evals);
}

extern "C" bool eshkol_gfn_train(eshkol_gfn_dag_t* gfn,
                                 uint32_t num_steps,
                                 double lr_logits,
                                 double lr_log_z,
                                 uint64_t seed) {
    if (!gfn) return false;
    return gfn->train(num_steps, lr_logits, lr_log_z, seed);
}

extern "C" bool eshkol_gfn_eval_distribution(const eshkol_gfn_dag_t* gfn,
                                             double* out_state_probs,
                                             uint32_t probs_len,
                                             double* out_tv_distance,
                                             double* out_exact_log_z) {
    if (!gfn) return false;
    return gfn->eval_distribution(out_state_probs, probs_len, out_tv_distance, out_exact_log_z);
}

extern "C" bool eshkol_gfn_eval_per_state_tv(const eshkol_gfn_dag_t* gfn,
                                             double* out_state_tv,
                                             uint32_t tv_len,
                                             double* out_max_state_tv,
                                             double* out_max_terminal_rel_err) {
    if (!gfn) return false;
    return gfn->eval_per_state_tv(out_state_tv, tv_len, out_max_state_tv, out_max_terminal_rel_err);
}

/* ============================================================================
 * 2. Predictive Coding Network (PCN) / Energy-Based Layer
 * ============================================================================ */

struct PcnLayer {
    uint32_t in_dim = 0;
    uint32_t out_dim = 0;
    std::vector<double> W; /* out_dim x in_dim */
    std::vector<double> b; /* out_dim */
    uint32_t param_offset = 0;
};

struct eshkol_pcn {
    std::vector<uint32_t> dims;
    std::vector<PcnLayer> layers;
    uint32_t total_params = 0;

    eshkol_pcn(const uint32_t* d, uint32_t nl, uint64_t seed)
        : dims(d, d + nl) {
        uint64_t rng = seed ^ 0xbb67ae8584caa73bULL;
        layers.resize(nl - 1u);
        for (uint32_t l = 0; l + 1 < nl; l++) {
            PcnLayer& ly = layers[l];
            ly.in_dim = dims[l];
            ly.out_dim = dims[l + 1u];
            ly.param_offset = total_params;
            ly.W.resize(static_cast<size_t>(ly.out_dim) * ly.in_dim);
            ly.b.resize(ly.out_dim);
            double scale = 0.6 / std::sqrt(static_cast<double>(ly.in_dim));
            for (double& w : ly.W) w = (uniform01(rng) * 2.0 - 1.0) * scale;
            for (double& bi : ly.b) bi = (uniform01(rng) * 2.0 - 1.0) * 0.2;
            total_params += static_cast<uint32_t>(ly.W.size() + ly.b.size());
        }
    }

    static inline double act(double z) { return std::tanh(z); }
    static inline double act_deriv(double z) {
        double t = std::tanh(z);
        return 1.0 - t * t;
    }

    /* Forward pass computing pre-activations x[0..L] where x[0] = input,
     * mu[l] = W[l-1] * phi(x[l-1]) + b[l-1] (input layer uses identity phi(x[0]) = x[0]). */
    void forward_init(const double* input, std::vector<std::vector<double>>& x) const {
        const size_t L = layers.size();
        x.resize(L + 1u);
        x[0].assign(input, input + dims[0]);
        for (size_t l = 0; l < L; l++) {
            const PcnLayer& ly = layers[l];
            x[l + 1u].assign(ly.out_dim, 0.0);
            for (uint32_t i = 0; i < ly.out_dim; i++) {
                double acc = ly.b[i];
                for (uint32_t j = 0; j < ly.in_dim; j++) {
                    double prev = (l == 0) ? x[0][j] : act(x[l][j]);
                    acc += ly.W[i * ly.in_dim + j] * prev;
                }
                x[l + 1u][i] = acc;
            }
        }
    }

    /* Relax hidden activities x[1..L-1] to minimize variational PCN free energy
     * E(x; beta) = 0.5 * sum_{l=1..L-1} ||x_l - mu_l||^2 + 0.5 * beta * ||target - mu_L||^2
     * using purely local predictive-coding error dynamics. */
    void relax_equilibrium(const std::vector<std::vector<double>>& x_init,
                           const double* target,
                           double beta,
                           uint32_t infer_steps,
                           double infer_lr,
                           std::vector<std::vector<double>>& x,
                           std::vector<std::vector<double>>& eps) const {
        const size_t L = layers.size();
        x = x_init;
        std::vector<std::vector<double>> mu(L + 1u);
        eps.resize(L + 1u);
        for (size_t l = 1; l <= L; l++) {
            mu[l].assign(dims[l], 0.0);
            eps[l].assign(dims[l], 0.0);
        }

        auto recompute_errors = [&]() {
            for (size_t l = 0; l < L; l++) {
                const PcnLayer& ly = layers[l];
                for (uint32_t i = 0; i < ly.out_dim; i++) {
                    double acc = ly.b[i];
                    for (uint32_t j = 0; j < ly.in_dim; j++) {
                        double prev = (l == 0) ? x[0][j] : act(x[l][j]);
                        acc += ly.W[i * ly.in_dim + j] * prev;
                    }
                    mu[l + 1u][i] = acc;
                }
            }
            for (size_t l = 1; l < L; l++) {
                for (uint32_t i = 0; i < dims[l]; i++) {
                    eps[l][i] = x[l][i] - mu[l][i];
                }
            }
            /* Output prediction error weighted by nudging factor beta */
            for (uint32_t i = 0; i < dims[L]; i++) {
                eps[L][i] = beta * (target[i] - mu[L][i]);
            }
        };

        std::vector<std::vector<double>> dx(L);
        for (size_t l = 1; l < L; l++) dx[l].assign(dims[l], 0.0);

        for (uint32_t it = 0; it < infer_steps; it++) {
            recompute_errors();
            double max_g = 0.0;
            for (size_t l = 1; l < L; l++) {
                const PcnLayer& next_ly = layers[l];
                for (uint32_t j = 0; j < dims[l]; j++) {
                    double top_down = 0.0;
                    for (uint32_t i = 0; i < next_ly.out_dim; i++) {
                        top_down += next_ly.W[i * next_ly.in_dim + j] * eps[l + 1u][i];
                    }
                    /* -dE/dx_l = -eps_l + phi'(x_l) * W_{l+1}^T * eps_{l+1} */
                    double g = -eps[l][j] + act_deriv(x[l][j]) * top_down;
                    dx[l][j] = g;
                    max_g = std::max(max_g, std::fabs(g));
                }
            }
            for (size_t l = 1; l < L; l++) {
                for (uint32_t j = 0; j < dims[l]; j++) {
                    x[l][j] += infer_lr * dx[l][j];
                }
            }
            if (max_g < 1e-14) break;
        }
        recompute_errors();
    }

    bool local_hebbian_grad(const double* input,
                            const double* target,
                            double beta,
                            uint32_t infer_steps,
                            double infer_lr,
                            double* out_grads,
                            uint32_t grads_len,
                            uint64_t* out_fd_evals) const {
        if (!input || !target || !out_grads || grads_len < total_params) return false;
        if (!std::isfinite(beta) || beta <= 0.0 || infer_steps == 0 || infer_lr <= 0.0) return false;
        if (out_fd_evals) *out_fd_evals = 0;

        std::vector<std::vector<double>> x_free;
        forward_init(input, x_free);

        /* Centered symmetric nudging (+beta, -beta) eliminates O(beta) bias,
         * yielding O(beta^2) exact agreement with backpropagation via purely local Hebbian rules. */
        std::vector<std::vector<double>> x_pos, eps_pos, x_neg, eps_neg;
        relax_equilibrium(x_free, target, +beta, infer_steps, infer_lr, x_pos, eps_pos);
        relax_equilibrium(x_free, target, -beta, infer_steps, infer_lr, x_neg, eps_neg);

        const size_t L = layers.size();
        const double inv_2beta = 1.0 / (2.0 * beta);
        for (size_t l = 0; l < L; l++) {
            const PcnLayer& ly = layers[l];
            double* gw = out_grads + ly.param_offset;
            double* gb = gw + ly.W.size();
            for (uint32_t i = 0; i < ly.out_dim; i++) {
                /* Local Hebbian product: -1/(2*beta) * (eps_pos * phi(x_pos) - eps_neg * phi(x_neg)) */
                gb[i] = -inv_2beta * (eps_pos[l + 1u][i] - eps_neg[l + 1u][i]);
                for (uint32_t j = 0; j < ly.in_dim; j++) {
                    double prev_p = (l == 0) ? x_pos[0][j] : act(x_pos[l][j]);
                    double prev_n = (l == 0) ? x_neg[0][j] : act(x_neg[l][j]);
                    gw[i * ly.in_dim + j] =
                        -inv_2beta * (eps_pos[l + 1u][i] * prev_p - eps_neg[l + 1u][i] * prev_n);
                }
            }
        }
        return true;
    }

    bool exact_backprop_grad(const double* input,
                             const double* target,
                             double* out_grads,
                             uint32_t grads_len,
                             double* out_loss) const {
        if (!input || !target || !out_grads || grads_len < total_params) return false;
        std::vector<std::vector<double>> x;
        forward_init(input, x);
        const size_t L = layers.size();

        double loss = 0.0;
        std::vector<std::vector<double>> delta(L + 1u);
        delta[L].resize(dims[L]);
        for (uint32_t i = 0; i < dims[L]; i++) {
            double diff = x[L][i] - target[i];
            loss += 0.5 * diff * diff;
            delta[L][i] = diff;
        }
        if (out_loss) *out_loss = loss;

        for (size_t l = L; l-- > 1u; ) {
            const PcnLayer& next_ly = layers[l];
            delta[l].assign(dims[l], 0.0);
            for (uint32_t j = 0; j < dims[l]; j++) {
                double acc = 0.0;
                for (uint32_t i = 0; i < next_ly.out_dim; i++) {
                    acc += next_ly.W[i * next_ly.in_dim + j] * delta[l + 1u][i];
                }
                delta[l][j] = acc * act_deriv(x[l][j]);
            }
        }

        for (size_t l = 0; l < L; l++) {
            const PcnLayer& ly = layers[l];
            double* gw = out_grads + ly.param_offset;
            double* gb = gw + ly.W.size();
            for (uint32_t i = 0; i < ly.out_dim; i++) {
                gb[i] = delta[l + 1u][i];
                for (uint32_t j = 0; j < ly.in_dim; j++) {
                    double prev = (l == 0) ? x[0][j] : act(x[l][j]);
                    gw[i * ly.in_dim + j] = delta[l + 1u][i] * prev;
                }
            }
        }
        return true;
    }
};

extern "C" eshkol_pcn_t* eshkol_pcn_create(const uint32_t* layer_dims,
                                           uint32_t num_layers,
                                           uint64_t seed) {
    if (!layer_dims || num_layers < 2) return nullptr;
    for (uint32_t i = 0; i < num_layers; i++) {
        if (layer_dims[i] == 0) return nullptr;
    }
    return new (std::nothrow) eshkol_pcn(layer_dims, num_layers, seed);
}

extern "C" void eshkol_pcn_destroy(eshkol_pcn_t* pcn) {
    delete pcn;
}

extern "C" uint32_t eshkol_pcn_total_params(const eshkol_pcn_t* pcn) {
    return pcn ? pcn->total_params : 0u;
}

extern "C" bool eshkol_pcn_local_hebbian_grad(const eshkol_pcn_t* pcn,
                                              const double* input,
                                              const double* target,
                                              double beta,
                                              uint32_t infer_steps,
                                              double infer_lr,
                                              double* out_grads,
                                              uint32_t grads_len,
                                              uint64_t* out_fd_evals) {
    if (!pcn) return false;
    return pcn->local_hebbian_grad(input, target, beta, infer_steps, infer_lr,
                                   out_grads, grads_len, out_fd_evals);
}

extern "C" bool eshkol_pcn_exact_backprop_grad(const eshkol_pcn_t* pcn,
                                               const double* input,
                                               const double* target,
                                               double* out_grads,
                                               uint32_t grads_len,
                                               double* out_loss) {
    if (!pcn) return false;
    return pcn->exact_backprop_grad(input, target, out_grads, grads_len, out_loss);
}
