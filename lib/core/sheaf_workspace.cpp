/*
 * Sheaf-Cohomological & GNWT Global Workspace
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/sheaf_workspace.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

namespace {

using Mat = std::vector<double>; /* row-major d x d */
using Vec = std::vector<double>; /* length d */

Mat mat_identity(uint32_t d) {
    Mat m(static_cast<size_t>(d) * d, 0.0);
    for (uint32_t i = 0; i < d; i++) m[i * d + i] = 1.0;
    return m;
}

Mat mat_mul(const Mat& A, const Mat& B, uint32_t d) {
    Mat C(static_cast<size_t>(d) * d, 0.0);
    for (uint32_t i = 0; i < d; i++) {
        for (uint32_t k = 0; k < d; k++) {
            double aik = A[i * d + k];
            if (aik == 0.0) continue;
            for (uint32_t j = 0; j < d; j++) {
                C[i * d + j] += aik * B[k * d + j];
            }
        }
    }
    return C;
}

Vec mat_vec(const Mat& A, const Vec& x, uint32_t d) {
    Vec y(d, 0.0);
    for (uint32_t i = 0; i < d; i++) {
        double acc = 0.0;
        for (uint32_t j = 0; j < d; j++) {
            acc += A[i * d + j] * x[j];
        }
        y[i] = acc;
    }
    return y;
}

/* Gauss-Jordan matrix inverse for general invertible d x d restriction maps. */
bool mat_invert(const Mat& A, Mat& out_inv, uint32_t d) {
    std::vector<double> aug(static_cast<size_t>(d) * (2u * d), 0.0);
    for (uint32_t i = 0; i < d; i++) {
        for (uint32_t j = 0; j < d; j++) {
            aug[i * (2u * d) + j] = A[i * d + j];
        }
        aug[i * (2u * d) + d + i] = 1.0;
    }
    for (uint32_t col = 0; col < d; col++) {
        uint32_t piv = col;
        double max_abs = std::fabs(aug[col * (2u * d) + col]);
        for (uint32_t r = col + 1; r < d; r++) {
            double v = std::fabs(aug[r * (2u * d) + col]);
            if (v > max_abs) { max_abs = v; piv = r; }
        }
        if (max_abs < 1e-13) return false;
        if (piv != col) {
            for (uint32_t c = 0; c < 2u * d; c++) {
                std::swap(aug[col * (2u * d) + c], aug[piv * (2u * d) + c]);
            }
        }
        double inv_p = 1.0 / aug[col * (2u * d) + col];
        for (uint32_t c = 0; c < 2u * d; c++) aug[col * (2u * d) + c] *= inv_p;
        for (uint32_t r = 0; r < d; r++) {
            if (r == col) continue;
            double f = aug[r * (2u * d) + col];
            if (f == 0.0) continue;
            for (uint32_t c = 0; c < 2u * d; c++) {
                aug[r * (2u * d) + c] -= f * aug[col * (2u * d) + c];
            }
        }
    }
    out_inv.resize(static_cast<size_t>(d) * d);
    for (uint32_t i = 0; i < d; i++) {
        for (uint32_t j = 0; j < d; j++) {
            out_inv[i * d + j] = aug[i * (2u * d) + d + j];
        }
    }
    return true;
}

double vec_norm(const Vec& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

struct Specialist {
    std::string name;
    bool is_dialectic = false;
    bool has_proposal = false;
    double base_salience = 0.0;
    uint32_t refractory_remaining = 0;
    Vec proposal;
};

struct Transition {
    uint32_t from = 0;
    uint32_t to = 0;
    Mat rho;
    Mat rho_inv;
};

} /* anonymous namespace */

struct eshkol_sheaf_workspace {
    uint32_t dim = 0;
    eshkol_sheaf_ws_config_t cfg{};
    std::vector<Specialist> specialists;
    std::vector<Transition> transitions;

    /* Attention Schema (AST) predictive distribution over specialists */
    std::vector<double> ast_probs;

    /* Latest broadcast sections: [0] = glued H^0 or split branch 0; [1] = split branch 1 */
    std::vector<Vec> broadcasts;

    eshkol_sheaf_workspace(uint32_t d, const eshkol_sheaf_ws_config_t& c)
        : dim(d), cfg(c) {}

    void ensure_ast_normalized() {
        const size_t n = specialists.size();
        if (ast_probs.size() < n) {
            size_t old = ast_probs.size();
            ast_probs.resize(n, 0.0);
            if (old == 0) {
                for (double& p : ast_probs) p = 1.0 / static_cast<double>(n);
            } else {
                double new_share = 1.0 / static_cast<double>(n);
                double scale = 1.0 - new_share * static_cast<double>(n - old);
                for (size_t i = 0; i < old; i++) ast_probs[i] *= scale;
                for (size_t i = old; i < n; i++) ast_probs[i] = new_share;
            }
        }
        double s = 0.0;
        for (double p : ast_probs) s += p;
        if (s > 0.0) {
            for (double& p : ast_probs) p /= s;
        }
    }

    bool step(eshkol_sheaf_ws_step_result_t* out) {
        if (!out) return false;
        std::memset(out, 0, sizeof(*out));
        out->winner_id = UINT32_MAX;
        broadcasts.clear();

        const uint32_t K = static_cast<uint32_t>(specialists.size());
        if (K == 0) return true;
        ensure_ast_normalized();

        /* Collect active proposing charts */
        std::vector<uint32_t> active_charts;
        for (uint32_t i = 0; i < K; i++) {
            if (specialists[i].has_proposal) active_charts.push_back(i);
        }

        /* 1. Cech 1-cocycle holonomy & coboundary analysis over overlap graph */
        double coboundary_sq = 0.0;
        double signal_sq = 0.0;
        for (uint32_t i : active_charts) {
            for (double x : specialists[i].proposal) signal_sq += x * x;
        }

        /* Build undirected adjacency for spanning tree parallel transport */
        struct AdjEdge {
            uint32_t to;
            uint32_t trans_idx;
            bool forward;
        };
        std::vector<std::vector<AdjEdge>> adj(K);
        for (uint32_t ti = 0; ti < transitions.size(); ti++) {
            const Transition& tr = transitions[ti];
            adj[tr.from].push_back({tr.to, ti, true});
            adj[tr.to].push_back({tr.from, ti, false});

            if (specialists[tr.from].has_proposal && specialists[tr.to].has_proposal) {
                Vec transported = mat_vec(tr.rho, specialists[tr.from].proposal, dim);
                for (uint32_t j = 0; j < dim; j++) {
                    double diff = transported[j] - specialists[tr.to].proposal[j];
                    coboundary_sq += diff * diff;
                }
            }
        }
        double coboundary_res = (signal_sq > 1e-18) ? std::sqrt(coboundary_sq / signal_sq) : 0.0;

        /* Spanning forest BFS to compute cumulative transport T[u] from component root -> u */
        std::vector<uint8_t> visited(K, 0);
        std::vector<int32_t> comp_root(K, -1);
        std::vector<Mat> T_root_to(K, mat_identity(dim));
        std::vector<Mat> T_to_root(K, mat_identity(dim));
        std::vector<uint8_t> is_tree_edge(transitions.size(), 0);

        for (uint32_t start = 0; start < K; start++) {
            if (visited[start] || !specialists[start].has_proposal) continue;
            visited[start] = 1;
            comp_root[start] = static_cast<int32_t>(start);
            std::deque<uint32_t> q;
            q.push_back(start);
            while (!q.empty()) {
                uint32_t u = q.front();
                q.pop_front();
                for (const AdjEdge& ae : adj[u]) {
                    if (!specialists[ae.to].has_proposal) continue;
                    if (!visited[ae.to]) {
                        visited[ae.to] = 1;
                        comp_root[ae.to] = static_cast<int32_t>(start);
                        is_tree_edge[ae.trans_idx] = 1;
                        const Mat& step_m = ae.forward ? transitions[ae.trans_idx].rho
                                                       : transitions[ae.trans_idx].rho_inv;
                        const Mat& step_inv = ae.forward ? transitions[ae.trans_idx].rho_inv
                                                         : transitions[ae.trans_idx].rho;
                        T_root_to[ae.to] = mat_mul(step_m, T_root_to[u], dim);
                        T_to_root[ae.to] = mat_mul(T_to_root[u], step_inv, dim);
                        q.push_back(ae.to);
                    }
                }
            }
        }

        /* Compute exact cycle holonomy H_e = T_{root<-v} * rho_{u->v} * T_{root->u}
         * for every non-tree overlap edge e = (u -> v) closing a loop! */
        double max_holonomy_norm = 0.0;
        Vec branch0(dim, 0.0);
        Vec branch1(dim, 0.0);
        bool have_split_branches = false;
        Vec naive_loop_avg(dim, 0.0);
        bool computed_naive_loop = false;

        for (uint32_t ti = 0; ti < transitions.size(); ti++) {
            if (is_tree_edge[ti]) continue;
            const Transition& tr = transitions[ti];
            if (!specialists[tr.from].has_proposal || !specialists[tr.to].has_proposal) continue;
            if (comp_root[tr.from] != comp_root[tr.to]) continue;

            Mat H = mat_mul(T_to_root[tr.to], mat_mul(tr.rho, T_root_to[tr.from], dim), dim);
            /* Pull section at root through identity vs cycle holonomy H */
            uint32_t root = static_cast<uint32_t>(comp_root[tr.from]);
            const Vec& s_root = specialists[root].proposal;
            Vec Hs = mat_vec(H, s_root, dim);
            Vec diff(dim, 0.0);
            for (uint32_t j = 0; j < dim; j++) diff[j] = Hs[j] - s_root[j];
            double h_norm = vec_norm(diff);

            if (h_norm > max_holonomy_norm) {
                max_holonomy_norm = h_norm;
                branch0 = s_root;
                branch1 = Hs;
                have_split_branches = true;
                for (uint32_t j = 0; j < dim; j++) {
                    naive_loop_avg[j] = 0.5 * (s_root[j] + Hs[j]);
                }
                computed_naive_loop = true;
            }
        }

        /* Also check single-overlap coboundary inconsistency when no cycle holonomy exists */
        double max_edge_mismatch = 0.0;
        if (!have_split_branches) {
            for (const Transition& tr : transitions) {
                if (!specialists[tr.from].has_proposal || !specialists[tr.to].has_proposal) continue;
                Vec transported = mat_vec(tr.rho, specialists[tr.from].proposal, dim);
                const Vec& target = specialists[tr.to].proposal;
                Vec diff(dim, 0.0);
                for (uint32_t j = 0; j < dim; j++) diff[j] = transported[j] - target[j];
                double m = vec_norm(diff);
                if (m > max_edge_mismatch) {
                    max_edge_mismatch = m;
                    branch0 = transported;
                    branch1 = target;
                    have_split_branches = true;
                    for (uint32_t j = 0; j < dim; j++) {
                        naive_loop_avg[j] = 0.5 * (transported[j] + target[j]);
                    }
                    computed_naive_loop = true;
                }
            }
        }

        /* Compute parallel-transported average in root frame across all active charts */
        Vec glued(dim, 0.0);
        uint32_t glued_cnt = 0;
        for (uint32_t i : active_charts) {
            Vec pulled = mat_vec(T_to_root[i], specialists[i].proposal, dim);
            for (uint32_t j = 0; j < dim; j++) glued[j] += pulled[j];
            glued_cnt++;
        }
        if (glued_cnt > 0) {
            for (double& x : glued) x /= static_cast<double>(glued_cnt);
        }

        double naive_norm = computed_naive_loop ? vec_norm(naive_loop_avg) : vec_norm(glued);
        bool h1_obstructed = (max_holonomy_norm > cfg.h1_threshold) ||
                             (coboundary_res > cfg.h1_threshold);

        out->cech_holonomy_norm = max_holonomy_norm;
        out->cech_coboundary_residual = coboundary_res;
        out->naive_average_norm = naive_norm;

        if (h1_obstructed) {
            out->status = ESHKOL_WS_H1_OBSTRUCTION;
            out->glued_section_norm = 0.0;
            out->num_split_charts = have_split_branches ? 2u : 0u;
            if (have_split_branches) {
                out->split_branch_norms[0] = vec_norm(branch0);
                out->split_branch_norms[1] = vec_norm(branch1);
                broadcasts.push_back(std::move(branch0));
                broadcasts.push_back(std::move(branch1));
            }
        } else {
            out->status = ESHKOL_WS_GLUED_H0;
            out->glued_section_norm = vec_norm(glued);
            out->num_split_charts = 0u;
            broadcasts.push_back(std::move(glued));
        }

        /* 2. GNWT Ignition Competition with Refractory Gating + Dialectic Boost */
        uint32_t best_id = UINT32_MAX;
        double best_sal = -1.0;
        for (uint32_t i = 0; i < K; i++) {
            const Specialist& sp = specialists[i];
            if (sp.refractory_remaining > 0) continue;
            double eff_sal = sp.base_salience;
            if (h1_obstructed && sp.is_dialectic) {
                eff_sal += cfg.dialectic_boost;
            }
            if (eff_sal >= cfg.ignition_threshold && eff_sal > best_sal) {
                best_sal = eff_sal;
                best_id = i;
            }
        }

        /* Decrement refractory timers on all other specialists */
        for (uint32_t i = 0; i < K; i++) {
            if (specialists[i].refractory_remaining > 0) {
                specialists[i].refractory_remaining--;
            }
        }

        if (best_id == UINT32_MAX) {
            out->status = ESHKOL_WS_NO_IGNITION;
            out->winner_id = UINT32_MAX;
            return true;
        }

        /* 3. AST Metacognitive Surprise & Refractory Latch */
        double pred_p = std::max(1e-12, ast_probs[best_id]);
        out->winner_id = best_id;
        out->winner_effective_salience = best_sal;
        out->ast_predicted_prob = pred_p;
        out->ast_surprise = -std::log(pred_p);

        specialists[best_id].refractory_remaining = cfg.refractory_steps;

        /* Update AST predictor toward observed ignition winner */
        const double lr = std::max(0.0, std::min(1.0, cfg.ast_learning_rate));
        for (uint32_t i = 0; i < K; i++) {
            double target = (i == best_id) ? 1.0 : 0.0;
            ast_probs[i] = (1.0 - lr) * ast_probs[i] + lr * target;
        }
        ensure_ast_normalized();
        return true;
    }
};

/* ============================================================================
 * Public C API
 * ============================================================================ */

extern "C" eshkol_sheaf_ws_config_t eshkol_sheaf_ws_default_config(void) {
    eshkol_sheaf_ws_config_t cfg;
    cfg.h1_threshold = 1e-4;
    cfg.ignition_threshold = 0.1;
    cfg.refractory_steps = 1u;
    cfg.dialectic_boost = 1.5;
    cfg.ast_learning_rate = 0.35;
    return cfg;
}

extern "C" eshkol_sheaf_workspace_t* eshkol_sheaf_ws_create(uint32_t stalk_dim,
                                                            const eshkol_sheaf_ws_config_t* cfg) {
    if (stalk_dim == 0) return nullptr;
    eshkol_sheaf_ws_config_t eff = cfg ? *cfg : eshkol_sheaf_ws_default_config();
    return new (std::nothrow) eshkol_sheaf_workspace(stalk_dim, eff);
}

extern "C" void eshkol_sheaf_ws_destroy(eshkol_sheaf_workspace_t* ws) {
    delete ws;
}

extern "C" uint32_t eshkol_sheaf_ws_add_specialist(eshkol_sheaf_workspace_t* ws,
                                                   const char* name,
                                                   bool is_dialectic) {
    if (!ws) return UINT32_MAX;
    Specialist sp;
    sp.name = name ? name : "";
    sp.is_dialectic = is_dialectic;
    sp.proposal.assign(ws->dim, 0.0);
    uint32_t id = static_cast<uint32_t>(ws->specialists.size());
    ws->specialists.push_back(std::move(sp));
    return id;
}

extern "C" uint32_t eshkol_sheaf_ws_num_specialists(const eshkol_sheaf_workspace_t* ws) {
    return ws ? static_cast<uint32_t>(ws->specialists.size()) : 0u;
}

extern "C" bool eshkol_sheaf_ws_propose(eshkol_sheaf_workspace_t* ws,
                                        uint32_t specialist_id,
                                        const double* section,
                                        uint32_t dim,
                                        double salience) {
    if (!ws || !section || dim != ws->dim || specialist_id >= ws->specialists.size()) return false;
    if (!std::isfinite(salience) || salience < 0.0) return false;
    Specialist& sp = ws->specialists[specialist_id];
    sp.proposal.assign(section, section + dim);
    sp.base_salience = salience;
    sp.has_proposal = true;
    return true;
}

extern "C" bool eshkol_sheaf_ws_add_transition(eshkol_sheaf_workspace_t* ws,
                                               uint32_t from_id,
                                               uint32_t to_id,
                                               const double* rho_matrix,
                                               uint32_t dim) {
    if (!ws || !rho_matrix || dim != ws->dim) return false;
    if (from_id >= ws->specialists.size() || to_id >= ws->specialists.size() || from_id == to_id) {
        return false;
    }
    Mat rho(rho_matrix, rho_matrix + static_cast<size_t>(dim) * dim);
    Mat rho_inv;
    if (!mat_invert(rho, rho_inv, dim)) return false;
    ws->transitions.push_back({from_id, to_id, std::move(rho), std::move(rho_inv)});
    return true;
}

extern "C" bool eshkol_sheaf_ws_step(eshkol_sheaf_workspace_t* ws,
                                     eshkol_sheaf_ws_step_result_t* out_result) {
    if (!ws) return false;
    return ws->step(out_result);
}

extern "C" bool eshkol_sheaf_ws_get_broadcast(const eshkol_sheaf_workspace_t* ws,
                                              uint32_t branch_idx,
                                              double* out_section,
                                              uint32_t dim) {
    if (!ws || !out_section || dim != ws->dim || branch_idx >= ws->broadcasts.size()) return false;
    std::memcpy(out_section, ws->broadcasts[branch_idx].data(),
                static_cast<size_t>(dim) * sizeof(double));
    return true;
}

extern "C" uint32_t eshkol_sheaf_ws_refractory_remaining(const eshkol_sheaf_workspace_t* ws,
                                                         uint32_t specialist_id) {
    if (!ws || specialist_id >= ws->specialists.size()) return 0u;
    return ws->specialists[specialist_id].refractory_remaining;
}
