/*
 * CSR Differentiable Forney Factor Graphs & Certified Free-Energy Enclosures
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/forney_fg.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>

namespace {

constexpr double kNegInf = -1e300;

/* Outward-rounded IEEE-754 interval arithmetic using exact nextafter bounds. */
struct Interval {
    double lo;
    double hi;

    static Interval point(double x) { return {x, x}; }
    static Interval hull(double a, double b) { return {std::min(a, b), std::max(a, b)}; }

    static double dn(double x) {
        return std::isfinite(x) ? std::nextafter(x, -std::numeric_limits<double>::infinity()) : x;
    }
    static double up(double x) {
        return std::isfinite(x) ? std::nextafter(x, std::numeric_limits<double>::infinity()) : x;
    }
    static double dn2(double x) { return dn(dn(x)); }
    static double up2(double x) { return up(up(x)); }

    Interval operator+(const Interval& o) const {
        return {dn(lo + o.lo), up(hi + o.hi)};
    }
    Interval operator-(const Interval& o) const {
        return {dn(lo - o.hi), up(hi - o.lo)};
    }
    Interval operator-() const {
        return {-hi, -lo};
    }
    Interval operator*(const Interval& o) const {
        double p1 = lo * o.lo;
        double p2 = lo * o.hi;
        double p3 = hi * o.lo;
        double p4 = hi * o.hi;
        return {dn(std::min({p1, p2, p3, p4})), up(std::max({p1, p2, p3, p4}))};
    }
    Interval scale_nonneg(double s) const {
        return {dn(lo * s), up(hi * s)};
    }
};

/* Outward-rounded exp and log (2-ULP conservative margin for libm rounding). */
inline Interval iv_exp(const Interval& x) {
    double elo = std::exp(x.lo);
    double ehi = std::exp(x.hi);
    return {std::max(0.0, Interval::dn2(elo)), Interval::up2(ehi)};
}

inline Interval iv_log(const Interval& x) {
    if (x.lo <= 0.0) {
        return {-std::numeric_limits<double>::infinity(), Interval::up2(std::log(std::max(x.hi, 1e-300)))};
    }
    return {Interval::dn2(std::log(x.lo)), Interval::up2(std::log(x.hi))};
}

/* Outward-rounded log-sum-exp over a vector of intervals. */
inline Interval iv_log_sum_exp(const std::vector<Interval>& vals) {
    if (vals.empty()) return Interval::point(-std::numeric_limits<double>::infinity());
    double m = -std::numeric_limits<double>::infinity();
    for (const auto& v : vals) {
        if (v.hi > m) m = v.hi;
    }
    if (!std::isfinite(m)) return Interval::point(m);
    Interval shift = Interval::point(m);
    Interval sum = Interval::point(0.0);
    for (const auto& v : vals) {
        sum = sum + iv_exp(v - shift);
    }
    return shift + iv_log(sum);
}

/* Outward-rounded x * log(x) on [0, 1] (minimum at x = 1/e, value = -1/e). */
inline Interval iv_xlogx(const Interval& x) {
    double a = std::max(0.0, std::min(1.0, x.lo));
    double b = std::max(0.0, std::min(1.0, x.hi));
    auto eval_dn = [](double t) -> double {
        if (t <= 0.0) return 0.0;
        return Interval::dn2(t * Interval::dn2(std::log(t)));
    };
    auto eval_up = [](double t) -> double {
        if (t <= 0.0) return 0.0;
        return Interval::up2(t * Interval::up2(std::log(t)));
    };
    const double inv_e = std::exp(-1.0);
    double lo = std::min(eval_dn(a), eval_dn(b));
    double hi = std::max(eval_up(a), eval_up(b));
    if (a <= inv_e && inv_e <= b) {
        lo = std::min(lo, Interval::dn2(-inv_e));
    }
    return {lo, hi};
}

inline double log_sum_exp(const double* vals, uint32_t n) {
    double m = -std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < n; i++) {
        if (vals[i] > m) m = vals[i];
    }
    if (!std::isfinite(m)) return m;
    double s = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        s += std::exp(vals[i] - m);
    }
    return m + std::log(s);
}

inline void normalize_log_vec(std::vector<double>& v) {
    double z = log_sum_exp(v.data(), static_cast<uint32_t>(v.size()));
    if (std::isfinite(z)) {
        for (double& x : v) x -= z;
    }
}

struct FactorNode {
    std::vector<uint32_t> vars;
    std::vector<uint32_t> strides;
    std::vector<double> log_pot;
    double rho = 1.0;
    uint32_t param_offset = 0;
};

struct Edge {
    uint32_t factor_idx = 0;
    uint32_t local_slot = 0;
    uint32_t var_idx = 0;
};

struct DisjointSet {
    std::vector<uint32_t> parent;
    explicit DisjointSet(uint32_t n) : parent(n) {
        std::iota(parent.begin(), parent.end(), 0u);
    }
    uint32_t find(uint32_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    bool unite(uint32_t a, uint32_t b) {
        uint32_t ra = find(a);
        uint32_t rb = find(b);
        if (ra == rb) return false;
        parent[ra] = rb;
        return true;
    }
};

} /* anonymous namespace */

struct eshkol_forney_fg {
    uint32_t num_vars = 0;
    std::vector<uint32_t> var_dims;
    std::vector<int32_t> obs_state; /* -1 = unobserved, >= 0 = clamped state */
    std::vector<FactorNode> factors;
    uint32_t total_params = 0;

    /* CSR incidence */
    bool csr_dirty = true;
    std::vector<Edge> edges;
    std::vector<uint32_t> factor_edge_offsets;
    std::vector<uint32_t> var_edge_offsets;
    std::vector<uint32_t> var_edges; /* edge indices grouped by variable */

    /* Converged log-space messages & beliefs */
    bool has_messages = false;
    bool last_trw = false;
    std::vector<std::vector<double>> msg_fv;
    std::vector<std::vector<double>> msg_vf;
    std::vector<std::vector<double>> var_belief;     /* normalized probabilities */
    std::vector<std::vector<double>> factor_belief;  /* normalized joint probabilities */

    eshkol_forney_fg(uint32_t nv, const uint32_t* dims)
        : num_vars(nv),
          var_dims(dims, dims + nv),
          obs_state(nv, -1),
          var_belief(nv) {
        for (uint32_t i = 0; i < nv; i++) {
            var_belief[i].assign(var_dims[i], 1.0 / static_cast<double>(var_dims[i]));
        }
    }

    void rebuild_csr() {
        if (!csr_dirty) return;
        edges.clear();
        factor_edge_offsets.assign(factors.size() + 1u, 0u);
        std::vector<uint32_t> var_deg(num_vars, 0u);

        for (uint32_t fi = 0; fi < factors.size(); fi++) {
            factor_edge_offsets[fi] = static_cast<uint32_t>(edges.size());
            const FactorNode& f = factors[fi];
            for (uint32_t slot = 0; slot < f.vars.size(); slot++) {
                uint32_t v = f.vars[slot];
                edges.push_back({fi, slot, v});
                var_deg[v]++;
            }
        }
        factor_edge_offsets[factors.size()] = static_cast<uint32_t>(edges.size());

        var_edge_offsets.assign(num_vars + 1u, 0u);
        for (uint32_t v = 0; v < num_vars; v++) {
            var_edge_offsets[v + 1u] = var_edge_offsets[v] + var_deg[v];
        }
        var_edges.assign(edges.size(), 0u);
        std::vector<uint32_t> cursor = var_edge_offsets;
        for (uint32_t e = 0; e < edges.size(); e++) {
            uint32_t v = edges[e].var_idx;
            var_edges[cursor[v]++] = e;
        }
        csr_dirty = false;
    }

    void init_messages() {
        rebuild_csr();
        msg_fv.resize(edges.size());
        msg_vf.resize(edges.size());
        for (uint32_t e = 0; e < edges.size(); e++) {
            uint32_t d = var_dims[edges[e].var_idx];
            double u = -std::log(static_cast<double>(d));
            msg_fv[e].assign(d, u);
            msg_vf[e].assign(d, u);
        }
        has_messages = true;
    }

    void apply_obs_mask(uint32_t v, std::vector<double>& log_vec) const {
        if (obs_state[v] >= 0) {
            for (uint32_t s = 0; s < var_dims[v]; s++) {
                if (static_cast<int32_t>(s) != obs_state[v]) {
                    log_vec[s] = kNegInf;
                }
            }
        }
    }

    /* Compute raw normalized factor-to-variable message for edge `e`. */
    void compute_fv_message(uint32_t e, bool use_trw, std::vector<double>& out) const {
        const Edge& ed = edges[e];
        const FactorNode& f = factors[ed.factor_idx];
        const uint32_t arity = static_cast<uint32_t>(f.vars.size());
        const uint32_t target_slot = ed.local_slot;
        const uint32_t target_var = ed.var_idx;
        const uint32_t target_dim = var_dims[target_var];
        const uint32_t e_begin = factor_edge_offsets[ed.factor_idx];
        const double inv_rho = use_trw ? (1.0 / f.rho) : 1.0;

        out.assign(target_dim, -std::numeric_limits<double>::infinity());
        std::vector<double> max_per_state(target_dim, -std::numeric_limits<double>::infinity());
        const uint32_t cpt_n = static_cast<uint32_t>(f.log_pot.size());
        std::vector<double> scores(cpt_n, 0.0);

        for (uint32_t idx = 0; idx < cpt_n; idx++) {
            double sc = f.log_pot[idx] * inv_rho;
            uint32_t target_s = 0;
            for (uint32_t slot = 0; slot < arity; slot++) {
                uint32_t v = f.vars[slot];
                uint32_t st = (idx / f.strides[slot]) % var_dims[v];
                if (slot == target_slot) {
                    target_s = st;
                } else {
                    sc += msg_vf[e_begin + slot][st];
                }
            }
            scores[idx] = sc;
            if (sc > max_per_state[target_s]) max_per_state[target_s] = sc;
        }

        std::vector<double> sum_exp(target_dim, 0.0);
        for (uint32_t idx = 0; idx < cpt_n; idx++) {
            uint32_t target_s = (idx / f.strides[target_slot]) % target_dim;
            if (std::isfinite(max_per_state[target_s])) {
                sum_exp[target_s] += std::exp(scores[idx] - max_per_state[target_s]);
            }
        }
        for (uint32_t s = 0; s < target_dim; s++) {
            if (std::isfinite(max_per_state[s]) && sum_exp[s] > 0.0) {
                out[s] = max_per_state[s] + std::log(sum_exp[s]);
            } else {
                out[s] = kNegInf;
            }
        }
        apply_obs_mask(target_var, out);
        normalize_log_vec(out);
    }

    /* Compute raw normalized variable-to-factor message for edge `e`. */
    void compute_vf_message(uint32_t e, bool use_trw, std::vector<double>& out) const {
        const Edge& ed = edges[e];
        const uint32_t v = ed.var_idx;
        const uint32_t dim = var_dims[v];
        out.assign(dim, 0.0);

        const uint32_t vb = var_edge_offsets[v];
        const uint32_t ve = var_edge_offsets[v + 1u];
        for (uint32_t k = vb; k < ve; k++) {
            uint32_t other_e = var_edges[k];
            double weight = 1.0;
            if (other_e == e) {
                if (!use_trw) continue;
                weight = factors[ed.factor_idx].rho - 1.0;
                if (weight == 0.0) continue;
            } else if (use_trw) {
                weight = factors[edges[other_e].factor_idx].rho;
            }
            for (uint32_t s = 0; s < dim; s++) {
                out[s] += weight * msg_fv[other_e][s];
            }
        }
        apply_obs_mask(v, out);
        normalize_log_vec(out);
    }

    void update_beliefs(bool use_trw) {
        var_belief.resize(num_vars);
        for (uint32_t v = 0; v < num_vars; v++) {
            uint32_t d = var_dims[v];
            std::vector<double> logb(d, 0.0);
            for (uint32_t k = var_edge_offsets[v]; k < var_edge_offsets[v + 1u]; k++) {
                uint32_t e = var_edges[k];
                double w = use_trw ? factors[edges[e].factor_idx].rho : 1.0;
                for (uint32_t s = 0; s < d; s++) {
                    logb[s] += w * msg_fv[e][s];
                }
            }
            apply_obs_mask(v, logb);
            normalize_log_vec(logb);
            var_belief[v].resize(d);
            double sum = 0.0;
            for (uint32_t s = 0; s < d; s++) {
                var_belief[v][s] = std::exp(logb[s]);
                sum += var_belief[v][s];
            }
            if (sum > 0.0) {
                for (double& p : var_belief[v]) p /= sum;
            }
        }

        factor_belief.resize(factors.size());
        for (uint32_t fi = 0; fi < factors.size(); fi++) {
            const FactorNode& f = factors[fi];
            const uint32_t cpt_n = static_cast<uint32_t>(f.log_pot.size());
            const uint32_t arity = static_cast<uint32_t>(f.vars.size());
            const uint32_t e_begin = factor_edge_offsets[fi];
            const double inv_rho = use_trw ? (1.0 / f.rho) : 1.0;

            std::vector<double> logb(cpt_n, 0.0);
            for (uint32_t idx = 0; idx < cpt_n; idx++) {
                double sc = f.log_pot[idx] * inv_rho;
                for (uint32_t slot = 0; slot < arity; slot++) {
                    uint32_t v = f.vars[slot];
                    uint32_t st = (idx / f.strides[slot]) % var_dims[v];
                    sc += msg_vf[e_begin + slot][st];
                    if (obs_state[v] >= 0 && static_cast<int32_t>(st) != obs_state[v]) {
                        sc = kNegInf;
                    }
                }
                logb[idx] = sc;
            }
            normalize_log_vec(logb);
            factor_belief[fi].resize(cpt_n);
            double sum = 0.0;
            for (uint32_t idx = 0; idx < cpt_n; idx++) {
                factor_belief[fi][idx] = std::exp(logb[idx]);
                sum += factor_belief[fi][idx];
            }
            if (sum > 0.0) {
                for (double& p : factor_belief[fi]) p /= sum;
            }
        }
    }

    double compute_bethe_fe(bool use_trw) const {
        double fe = 0.0;
        for (uint32_t fi = 0; fi < factors.size(); fi++) {
            const FactorNode& f = factors[fi];
            const auto& ba = factor_belief[fi];
            double rho = use_trw ? f.rho : 1.0;
            for (uint32_t idx = 0; idx < ba.size(); idx++) {
                double p = ba[idx];
                if (p > 1e-300) {
                    fe -= p * f.log_pot[idx];
                    fe += rho * p * std::log(p);
                }
            }
        }
        for (uint32_t v = 0; v < num_vars; v++) {
            double rho_sum = 0.0;
            for (uint32_t k = var_edge_offsets[v]; k < var_edge_offsets[v + 1u]; k++) {
                rho_sum += use_trw ? factors[edges[var_edges[k]].factor_idx].rho : 1.0;
            }
            double coeff = 1.0 - rho_sum;
            const auto& bv = var_belief[v];
            for (double p : bv) {
                if (p > 1e-300) {
                    fe += coeff * p * std::log(p);
                }
            }
        }
        return fe;
    }

    bool infer(const eshkol_bp_config_t& cfg, eshkol_bp_stats_t* out_stats) {
        init_messages();
        const bool use_trw = (cfg.schedule == ESHKOL_BP_TRW);
        last_trw = use_trw;
        const double damp = (cfg.damping > 0.0 && cfg.damping <= 1.0) ? cfg.damping : 0.7;
        const uint32_t max_it = cfg.max_iterations > 0 ? cfg.max_iterations : 200u;
        const double tol = cfg.tolerance > 0.0 ? cfg.tolerance : 1e-10;
        const uint32_t num_edges = static_cast<uint32_t>(edges.size());

        eshkol_bp_stats_t st{};
        std::memset(&st, 0, sizeof(st));

        if (num_edges == 0) {
            update_beliefs(use_trw);
            st.converged = true;
            st.bethe_free_energy = compute_bethe_fe(false);
            st.trw_free_energy = compute_bethe_fe(use_trw);
            if (out_stats) *out_stats = st;
            return true;
        }

        if (cfg.schedule == ESHKOL_BP_RESIDUAL_ASYNC) {
            /* Residual-priority asynchronous BP */
            std::vector<std::vector<double>> cand_fv(num_edges);
            std::vector<double> residual(num_edges, 0.0);
            using QItem = std::pair<double, uint32_t>;
            std::priority_queue<QItem> pq;

            for (uint32_t e = 0; e < num_edges; e++) {
                compute_vf_message(e, false, msg_vf[e]);
            }
            for (uint32_t e = 0; e < num_edges; e++) {
                compute_fv_message(e, false, cand_fv[e]);
                double r = 0.0;
                for (size_t s = 0; s < cand_fv[e].size(); s++) {
                    r = std::max(r, std::fabs(cand_fv[e][s] - msg_fv[e][s]));
                }
                residual[e] = r;
                pq.push({r, e});
            }

            const uint64_t max_updates = static_cast<uint64_t>(max_it) * static_cast<uint64_t>(num_edges);
            uint64_t updates = 0;
            while (!pq.empty() && updates < max_updates) {
                auto [r_top, e] = pq.top();
                pq.pop();
                if (std::fabs(r_top - residual[e]) > 1e-15) continue;
                if (r_top < tol) break;

                /* Commit damped update on highest-residual edge `e` */
                for (size_t s = 0; s < msg_fv[e].size(); s++) {
                    msg_fv[e][s] = damp * cand_fv[e][s] + (1.0 - damp) * msg_fv[e][s];
                }
                normalize_log_vec(msg_fv[e]);
                updates++;

                /* Recompute own residual */
                compute_fv_message(e, false, cand_fv[e]);
                double own_r = 0.0;
                for (size_t s = 0; s < cand_fv[e].size(); s++) {
                    own_r = std::max(own_r, std::fabs(cand_fv[e][s] - msg_fv[e][s]));
                }
                residual[e] = own_r;
                if (own_r >= tol) pq.push({own_r, e});

                /* Propagate updated variable-to-factor messages to neighbor factors */
                uint32_t v = edges[e].var_idx;
                for (uint32_t k = var_edge_offsets[v]; k < var_edge_offsets[v + 1u]; k++) {
                    uint32_t nb_e = var_edges[k];
                    if (nb_e == e) continue;
                    compute_vf_message(nb_e, false, msg_vf[nb_e]);
                    uint32_t nb_f = edges[nb_e].factor_idx;
                    for (uint32_t fe = factor_edge_offsets[nb_f]; fe < factor_edge_offsets[nb_f + 1u]; fe++) {
                        if (fe == nb_e) continue;
                        compute_fv_message(fe, false, cand_fv[fe]);
                        double nr = 0.0;
                        for (size_t s = 0; s < cand_fv[fe].size(); s++) {
                            nr = std::max(nr, std::fabs(cand_fv[fe][s] - msg_fv[fe][s]));
                        }
                        residual[fe] = nr;
                        pq.push({nr, fe});
                    }
                }
            }

            double max_r = 0.0;
            for (double r : residual) max_r = std::max(max_r, r);
            st.edge_updates = updates;
            st.iterations = static_cast<uint32_t>((updates + num_edges - 1u) / num_edges);
            st.max_residual = max_r;
            st.converged = (max_r < tol);
        } else {
            /* Damped synchronous BP or TRW-BP */
            std::vector<std::vector<double>> next_fv(num_edges);
            std::vector<std::vector<double>> next_vf(num_edges);
            double max_delta = 0.0;
            uint32_t iter = 0;
            for (; iter < max_it; iter++) {
                max_delta = 0.0;
                for (uint32_t e = 0; e < num_edges; e++) {
                    compute_vf_message(e, use_trw, next_vf[e]);
                }
                for (uint32_t e = 0; e < num_edges; e++) {
                    for (size_t s = 0; s < msg_vf[e].size(); s++) {
                        double nv = damp * next_vf[e][s] + (1.0 - damp) * msg_vf[e][s];
                        max_delta = std::max(max_delta, std::fabs(nv - msg_vf[e][s]));
                        msg_vf[e][s] = nv;
                    }
                    normalize_log_vec(msg_vf[e]);
                }
                for (uint32_t e = 0; e < num_edges; e++) {
                    compute_fv_message(e, use_trw, next_fv[e]);
                }
                for (uint32_t e = 0; e < num_edges; e++) {
                    for (size_t s = 0; s < msg_fv[e].size(); s++) {
                        double nv = damp * next_fv[e][s] + (1.0 - damp) * msg_fv[e][s];
                        max_delta = std::max(max_delta, std::fabs(nv - msg_fv[e][s]));
                        msg_fv[e][s] = nv;
                    }
                    normalize_log_vec(msg_fv[e]);
                    st.edge_updates++;
                }
                if (max_delta < tol) {
                    iter++;
                    break;
                }
            }
            st.iterations = iter;
            st.max_residual = max_delta;
            st.converged = (max_delta < tol);
        }

        update_beliefs(use_trw);
        st.bethe_free_energy = compute_bethe_fe(false);
        st.trw_free_energy = compute_bethe_fe(use_trw);
        if (out_stats) *out_stats = st;
        return st.converged;
    }

    /* Exact outward-rounded tree partition function log Z(tree) for a forest/tree
     * defined by unary potentials `unary_iv[v][s]` and a list of pairwise/higher-order
     * tree factors `(fi, scale)` where the induced factor graph is acyclic! */
    bool exact_tree_log_z_interval(
        const std::vector<std::vector<Interval>>& unary_iv,
        const std::vector<std::pair<uint32_t, double>>& tree_factors,
        Interval& out_log_z) const {

        /* Build bipartite tree adjacency: nodes 0..V-1 are vars, V..V+F-1 are tree factors */
        const uint32_t V = num_vars;
        const uint32_t TF = static_cast<uint32_t>(tree_factors.size());
        const uint32_t N = V + TF;
        std::vector<std::vector<uint32_t>> adj(N);
        for (uint32_t t = 0; t < TF; t++) {
            const FactorNode& f = factors[tree_factors[t].first];
            for (uint32_t v : f.vars) {
                adj[v].push_back(V + t);
                adj[V + t].push_back(v);
            }
        }

        std::vector<uint8_t> visited(N, 0);
        Interval total_log_z = Interval::point(0.0);

        for (uint32_t root = 0; root < V; root++) {
            if (visited[root]) continue;
            /* BFS order for this tree component */
            std::vector<uint32_t> order;
            std::vector<int32_t> parent(N, -1);
            order.push_back(root);
            visited[root] = 1;
            for (size_t qi = 0; qi < order.size(); qi++) {
                uint32_t u = order[qi];
                for (uint32_t nb : adj[u]) {
                    if (!visited[nb]) {
                        visited[nb] = 1;
                        parent[nb] = static_cast<int32_t>(u);
                        order.push_back(nb);
                    } else if (parent[u] != static_cast<int32_t>(nb)) {
                        /* Cycle detected — not a valid spanning tree! */
                        return false;
                    }
                }
            }

            /* Upward messages from child -> parent */
            std::vector<std::vector<Interval>> msg_up(N);

            for (size_t oi = order.size(); oi-- > 0; ) {
                uint32_t u = order[oi];
                int32_t par = parent[u];

                if (u < V) {
                    /* Variable node: combine unary_iv[u] + all child factor messages */
                    uint32_t d = var_dims[u];
                    std::vector<Interval> acc = unary_iv[u];
                    for (uint32_t nb : adj[u]) {
                        if (static_cast<int32_t>(nb) == par) continue;
                        for (uint32_t s = 0; s < d; s++) {
                            acc[s] = acc[s] + msg_up[nb][s];
                        }
                    }
                    if (obs_state[u] >= 0) {
                        for (uint32_t s = 0; s < d; s++) {
                            if (static_cast<int32_t>(s) != obs_state[u]) {
                                acc[s] = Interval::point(-std::numeric_limits<double>::infinity());
                            }
                        }
                    }
                    if (par < 0) {
                        /* Component root: sum out remaining variable */
                        total_log_z = total_log_z + iv_log_sum_exp(acc);
                    } else {
                        msg_up[u] = std::move(acc);
                    }
                } else {
                    /* Factor node sending message up to parent variable `par` */
                    uint32_t t = u - V;
                    const FactorNode& f = factors[tree_factors[t].first];
                    const double scale = tree_factors[t].second;
                    const uint32_t arity = static_cast<uint32_t>(f.vars.size());
                    uint32_t par_slot = 0;
                    for (uint32_t sl = 0; sl < arity; sl++) {
                        if (static_cast<int32_t>(f.vars[sl]) == par) {
                            par_slot = sl;
                            break;
                        }
                    }
                    uint32_t pd = var_dims[par];
                    std::vector<std::vector<Interval>> terms_by_state(pd);
                    const uint32_t cpt_n = static_cast<uint32_t>(f.log_pot.size());
                    for (uint32_t idx = 0; idx < cpt_n; idx++) {
                        Interval sc = Interval::point(f.log_pot[idx]).scale_nonneg(scale);
                        uint32_t ps = 0;
                        for (uint32_t sl = 0; sl < arity; sl++) {
                            uint32_t v = f.vars[sl];
                            uint32_t st = (idx / f.strides[sl]) % var_dims[v];
                            if (sl == par_slot) {
                                ps = st;
                            } else {
                                sc = sc + msg_up[v][st];
                            }
                        }
                        terms_by_state[ps].push_back(sc);
                    }
                    msg_up[u].resize(pd);
                    for (uint32_t s = 0; s < pd; s++) {
                        msg_up[u][s] = iv_log_sum_exp(terms_by_state[s]);
                    }
                }
            }
        }
        out_log_z = total_log_z;
        return true;
    }

    /* Certify outward-rounded [F_L, F_U] bounding exact -log Z. */
    bool certify_free_energy(const eshkol_bp_config_t* trw_cfg_opt,
                             eshkol_fe_interval_t* out_iv) {
        if (!out_iv) return false;
        std::memset(out_iv, 0, sizeof(*out_iv));
        rebuild_csr();

        /* Partition factors into unary (arity == 1) and higher-order (arity >= 2) */
        std::vector<uint32_t> unary_factors;
        std::vector<uint32_t> multi_factors;
        for (uint32_t fi = 0; fi < factors.size(); fi++) {
            if (factors[fi].vars.size() <= 1) unary_factors.push_back(fi);
            else multi_factors.push_back(fi);
        }

        /* Build a spanning-tree cover of all multi-variable factors:
         * Every multi-variable factor must appear in at least one spanning tree. */
        std::vector<std::vector<uint32_t>> trees;
        std::vector<uint32_t> factor_count(factors.size(), 0u);

        auto try_add_factor_to_dsu = [&](DisjointSet& dsu, uint32_t fi) -> bool {
            const auto& vs = factors[fi].vars;
            if (vs.size() < 2) return true;
            /* Check that all vertices in this factor are currently in distinct components */
            std::vector<uint32_t> roots;
            roots.reserve(vs.size());
            for (uint32_t v : vs) {
                uint32_t r = dsu.find(v);
                for (uint32_t prev : roots) {
                    if (prev == r) return false;
                }
                roots.push_back(r);
            }
            for (size_t i = 1; i < vs.size(); i++) dsu.unite(vs[0], vs[i]);
            return true;
        };

        if (multi_factors.empty()) {
            trees.emplace_back(); /* Single empty tree (unary only) */
        } else {
            /* First pass: greedily build spanning trees covering uncovered multi-factors */
            std::vector<uint8_t> covered(multi_factors.size(), 0);
            size_t num_covered = 0;
            while (num_covered < multi_factors.size()) {
                DisjointSet dsu(num_vars);
                std::vector<uint32_t> tree;
                /* Prioritize uncovered factors first */
                for (size_t i = 0; i < multi_factors.size(); i++) {
                    if (!covered[i] && try_add_factor_to_dsu(dsu, multi_factors[i])) {
                        tree.push_back(multi_factors[i]);
                        covered[i] = 1;
                        num_covered++;
                    }
                }
                /* Fill remaining tree slots with already-covered factors */
                for (size_t i = 0; i < multi_factors.size(); i++) {
                    if (covered[i] &&
                        std::find(tree.begin(), tree.end(), multi_factors[i]) == tree.end()) {
                        if (try_add_factor_to_dsu(dsu, multi_factors[i])) {
                            tree.push_back(multi_factors[i]);
                        }
                    }
                }
                if (tree.empty()) return false;
                for (uint32_t fi : tree) factor_count[fi]++;
                trees.push_back(std::move(tree));
            }
        }

        const uint32_t M = static_cast<uint32_t>(trees.size());
        const double mu = 1.0 / static_cast<double>(M);

        /* Assign exact tree appearance probabilities rho_a = count(a) / M */
        std::vector<double> saved_rho(factors.size(), 1.0);
        for (uint32_t fi = 0; fi < factors.size(); fi++) {
            saved_rho[fi] = factors[fi].rho;
            if (factors[fi].vars.size() <= 1) {
                factors[fi].rho = 1.0;
            } else {
                factors[fi].rho = static_cast<double>(factor_count[fi]) / static_cast<double>(M);
            }
        }

        /* Run TRW-BP to obtain dual reparameterization messages */
        eshkol_bp_config_t trw_cfg = trw_cfg_opt ? *trw_cfg_opt : eshkol_bp_default_config();
        trw_cfg.schedule = ESHKOL_BP_TRW;
        if (trw_cfg.damping > 0.7) trw_cfg.damping = 0.5;
        eshkol_bp_stats_t trw_st{};
        infer(trw_cfg, &trw_st);

        /* Save TRW messages & beliefs for reparameterization and upper bound */
        auto trw_msg_fv = msg_fv;
        auto trw_var_belief = var_belief;

        /* Precompute unary base potentials per variable */
        std::vector<std::vector<Interval>> base_unary(num_vars);
        for (uint32_t v = 0; v < num_vars; v++) {
            base_unary[v].assign(var_dims[v], Interval::point(0.0));
        }
        for (uint32_t fi : unary_factors) {
            uint32_t v = factors[fi].vars[0];
            for (uint32_t s = 0; s < var_dims[v]; s++) {
                base_unary[v][s] = base_unary[v][s] + Interval::point(factors[fi].log_pot[s]);
            }
        }

        /* Compute zero-sum TRW reparameterization shifts delta_k[v][s] across trees.
         * By exact algebra, average across k=0..M-1 is 0; any tiny rounding residual
         * is bounded conservatively in interval arithmetic so Jensen's inequality
         * log Z(theta) <= sum_k mu_k log Z(theta^{(k)}) + residual_bound is 100% sound. */
        std::vector<std::vector<std::vector<double>>> tree_shift(
            M, std::vector<std::vector<double>>(num_vars));
        for (uint32_t k = 0; k < M; k++) {
            std::vector<uint8_t> in_tree(factors.size(), 0);
            for (uint32_t fi : trees[k]) in_tree[fi] = 1;
            for (uint32_t v = 0; v < num_vars; v++) {
                tree_shift[k][v].assign(var_dims[v], 0.0);
                for (uint32_t ei = var_edge_offsets[v]; ei < var_edge_offsets[v + 1u]; ei++) {
                    uint32_t e = var_edges[ei];
                    uint32_t fi = edges[e].factor_idx;
                    if (factors[fi].vars.size() <= 1) continue;
                    double coeff = (in_tree[fi] ? 0.0 : 1.0) - (1.0 - factors[fi].rho);
                    for (uint32_t s = 0; s < var_dims[v]; s++) {
                        tree_shift[k][v][s] += coeff * trw_msg_fv[e][s];
                    }
                }
            }
        }

        /* Compute exact conservative residual bound for sum_k mu_k * tree_shift[k][v][s] */
        double shift_err_bound = 0.0;
        for (uint32_t v = 0; v < num_vars; v++) {
            double max_v_err = 0.0;
            for (uint32_t s = 0; s < var_dims[v]; s++) {
                Interval avg = Interval::point(0.0);
                for (uint32_t k = 0; k < M; k++) {
                    avg = avg + Interval::point(tree_shift[k][v][s]).scale_nonneg(mu);
                }
                double err = std::max(std::fabs(avg.lo), std::fabs(avg.hi));
                if (err > max_v_err) max_v_err = err;
            }
            shift_err_bound = Interval::up(shift_err_bound + max_v_err);
        }

        /* Evaluate exact outward-rounded log Z(theta^{(k)}) on every spanning tree! */
        Interval trw_log_z_sum = Interval::point(0.0);
        for (uint32_t k = 0; k < M; k++) {
            std::vector<std::vector<Interval>> u_iv = base_unary;
            for (uint32_t v = 0; v < num_vars; v++) {
                for (uint32_t s = 0; s < var_dims[v]; s++) {
                    u_iv[v][s] = u_iv[v][s] + Interval::point(tree_shift[k][v][s]);
                }
            }
            std::vector<std::pair<uint32_t, double>> tf;
            tf.reserve(trees[k].size());
            for (uint32_t fi : trees[k]) {
                tf.emplace_back(fi, 1.0 / factors[fi].rho);
            }
            Interval tree_lz{};
            if (!exact_tree_log_z_interval(u_iv, tf, tree_lz)) {
                for (uint32_t fi = 0; fi < factors.size(); fi++) factors[fi].rho = saved_rho[fi];
                return false;
            }
            trw_log_z_sum = trw_log_z_sum + tree_lz.scale_nonneg(mu);
        }

        /* Account for any rounding error in exact factor scaling sum_{k: a in T_k} mu / rho_a == 1 */
        double scale_err_bound = 0.0;
        for (uint32_t fi : multi_factors) {
            Interval ratio = Interval::point(1.0 / factors[fi].rho).scale_nonneg(
                mu * static_cast<double>(factor_count[fi]));
            double r_err = std::max(std::fabs(ratio.lo - 1.0), std::fabs(ratio.hi - 1.0));
            double max_pot = 0.0;
            for (double p : factors[fi].log_pot) max_pot = std::max(max_pot, std::fabs(p));
            scale_err_bound = Interval::up(scale_err_bound + Interval::up(r_err * max_pot));
        }

        /* Rigorous lower bound F_L = -(log_z_upper + shift_err + scale_err) <= -log Z */
        double log_z_upper = Interval::up(
            Interval::up(trw_log_z_sum.hi + shift_err_bound) + scale_err_bound);
        double f_lower = -log_z_upper;

        /* Restore original rho values and run standard BP for Bethe point estimate & upper bound */
        for (uint32_t fi = 0; fi < factors.size(); fi++) factors[fi].rho = saved_rho[fi];
        eshkol_bp_config_t std_cfg = eshkol_bp_default_config();
        std_cfg.schedule = ESHKOL_BP_SYNC;
        std_cfg.damping = 0.7;
        eshkol_bp_stats_t std_st{};
        infer(std_cfg, &std_st);

        /* Rigorous upper bound F_U >= -log Z:
         * Case 1: Graph is a single tree (M == 1) -> trw_log_z_sum.lo is exact lower bound on log Z,
         * so -trw_log_z_sum.lo is an exact outward-rounded upper bound on -log Z!
         * Case 2: General (possibly loopy) graph -> Gibbs variational upper bound F_MF[q]
         * evaluated in outward-rounded interval arithmetic for any valid product distribution q! */
        auto compute_mf_upper = [&](const std::vector<std::vector<double>>& q) -> double {
            /* Normalize q strictly in interval arithmetic */
            std::vector<std::vector<Interval>> q_iv(num_vars);
            for (uint32_t v = 0; v < num_vars; v++) {
                uint32_t d = var_dims[v];
                q_iv[v].resize(d);
                double sum = 0.0;
                for (uint32_t s = 0; s < d; s++) {
                    double p = (obs_state[v] >= 0) ? (static_cast<int32_t>(s) == obs_state[v] ? 1.0 : 0.0)
                                                   : std::max(0.0, q[v][s]);
                    sum += p;
                }
                if (sum <= 0.0) sum = 1.0;
                for (uint32_t s = 0; s < d; s++) {
                    double p = (obs_state[v] >= 0) ? (static_cast<int32_t>(s) == obs_state[v] ? 1.0 : 0.0)
                                                   : std::max(0.0, q[v][s]);
                    double pn = p / sum;
                    q_iv[v][s] = {std::max(0.0, Interval::dn2(pn)), std::min(1.0, Interval::up2(pn))};
                }
            }
            Interval fe_iv = Interval::point(0.0);
            /* Negative expected energy: -sum_a E_q[theta_a] */
            for (const FactorNode& f : factors) {
                const uint32_t arity = static_cast<uint32_t>(f.vars.size());
                const uint32_t cpt_n = static_cast<uint32_t>(f.log_pot.size());
                for (uint32_t idx = 0; idx < cpt_n; idx++) {
                    Interval prob = Interval::point(1.0);
                    for (uint32_t sl = 0; sl < arity; sl++) {
                        uint32_t v = f.vars[sl];
                        uint32_t st = (idx / f.strides[sl]) % var_dims[v];
                        prob = prob * q_iv[v][st];
                    }
                    fe_iv = fe_iv - (prob * Interval::point(f.log_pot[idx]));
                }
            }
            /* Negative entropy: +sum_v sum_s q_v(s) log q_v(s) */
            for (uint32_t v = 0; v < num_vars; v++) {
                for (const Interval& p : q_iv[v]) {
                    fe_iv = fe_iv + iv_xlogx(p);
                }
            }
            return fe_iv.hi;
        };

        double f_upper = std::numeric_limits<double>::infinity();
        if (M == 1) {
            /* Single tree: exact outward-rounded enclosure of -log Z */
            f_upper = -Interval::dn(
                Interval::dn(trw_log_z_sum.lo - shift_err_bound) - scale_err_bound);
        } else {
            double u1 = compute_mf_upper(var_belief);
            double u2 = compute_mf_upper(trw_var_belief);
            f_upper = std::min(u1, u2);
            /* Also evaluate single-point delta upper bound on highest-belief state */
            std::vector<std::vector<double>> map_q(num_vars);
            for (uint32_t v = 0; v < num_vars; v++) {
                map_q[v].assign(var_dims[v], 0.0);
                uint32_t best_s = 0;
                if (obs_state[v] >= 0) {
                    best_s = static_cast<uint32_t>(obs_state[v]);
                } else {
                    for (uint32_t s = 1; s < var_dims[v]; s++) {
                        if (var_belief[v][s] > var_belief[v][best_s]) best_s = s;
                    }
                }
                map_q[v][best_s] = 1.0;
            }
            f_upper = std::min(f_upper, compute_mf_upper(map_q));
        }

        out_iv->lower = f_lower;
        out_iv->upper = f_upper;
        out_iv->bethe_estimate = std_st.bethe_free_energy;
        out_iv->num_trees = M;
        out_iv->valid = std::isfinite(f_lower) && std::isfinite(f_upper) && (f_lower <= f_upper);
        return out_iv->valid;
    }

    /* Exact implicit differentiation of BP marginals via adjoint linear solver. */
    bool marginal_vjp(const double* dL_db_vars,
                      uint32_t var_states_len,
                      double* out_dL_dtheta,
                      uint32_t theta_len,
                      uint32_t max_vjp_iters,
                      double tol,
                      uint64_t* out_fd_evals) const {
        if (!has_messages || last_trw || !dL_db_vars || !out_dL_dtheta) return false;
        if (theta_len < total_params) return false;

        uint32_t expected_vs = 0;
        for (uint32_t d : var_dims) expected_vs += d;
        if (var_states_len < expected_vs) return false;

        if (out_fd_evals) *out_fd_evals = 0;
        std::memset(out_dL_dtheta, 0, static_cast<size_t>(total_params) * sizeof(double));

        const uint32_t num_edges = static_cast<uint32_t>(edges.size());
        if (num_edges == 0) return true;

        /* Step 1: Softmax pullback from variable marginals b_i to incoming m_{a->i} */
        std::vector<std::vector<double>> src_fv(num_edges);
        std::vector<std::vector<double>> var_z_bar(num_vars);
        uint32_t offset = 0;
        for (uint32_t v = 0; v < num_vars; v++) {
            uint32_t d = var_dims[v];
            var_z_bar[v].assign(d, 0.0);
            double dot = 0.0;
            for (uint32_t s = 0; s < d; s++) {
                dot += var_belief[v][s] * dL_db_vars[offset + s];
            }
            for (uint32_t s = 0; s < d; s++) {
                var_z_bar[v][s] = var_belief[v][s] * (dL_db_vars[offset + s] - dot);
            }
            offset += d;
        }
        for (uint32_t e = 0; e < num_edges; e++) {
            src_fv[e] = var_z_bar[edges[e].var_idx];
        }

        /* Precompute per-edge conditional/joint distributions q_{a->i}(x_{a\i}|x_i) and p_{a->i}(x_a) */
        struct EdgeJacobianCache {
            std::vector<double> p_joint;
            std::vector<double> q_cond;
        };
        std::vector<EdgeJacobianCache> cache(num_edges);
        for (uint32_t e = 0; e < num_edges; e++) {
            const Edge& ed = edges[e];
            const FactorNode& f = factors[ed.factor_idx];
            const uint32_t arity = static_cast<uint32_t>(f.vars.size());
            const uint32_t cpt_n = static_cast<uint32_t>(f.log_pot.size());
            const uint32_t e_begin = factor_edge_offsets[ed.factor_idx];
            const uint32_t td = var_dims[ed.var_idx];

            std::vector<double> eta(cpt_n, 0.0);
            std::vector<double> max_by_s(td, -std::numeric_limits<double>::infinity());
            double max_all = -std::numeric_limits<double>::infinity();
            for (uint32_t idx = 0; idx < cpt_n; idx++) {
                double sc = f.log_pot[idx];
                uint32_t ts = 0;
                for (uint32_t sl = 0; sl < arity; sl++) {
                    uint32_t v = f.vars[sl];
                    uint32_t st = (idx / f.strides[sl]) % var_dims[v];
                    if (sl == ed.local_slot) {
                        ts = st;
                    } else {
                        sc += msg_vf[e_begin + sl][st];
                    }
                    if (obs_state[v] >= 0 && static_cast<int32_t>(st) != obs_state[v]) {
                        sc = kNegInf;
                    }
                }
                eta[idx] = sc;
                if (sc > max_all) max_all = sc;
                if (sc > max_by_s[ts]) max_by_s[ts] = sc;
            }

            double z_all = 0.0;
            std::vector<double> z_by_s(td, 0.0);
            for (uint32_t idx = 0; idx < cpt_n; idx++) {
                uint32_t ts = (idx / f.strides[ed.local_slot]) % td;
                if (std::isfinite(max_all)) z_all += std::exp(eta[idx] - max_all);
                if (std::isfinite(max_by_s[ts])) z_by_s[ts] += std::exp(eta[idx] - max_by_s[ts]);
            }
            cache[e].p_joint.assign(cpt_n, 0.0);
            cache[e].q_cond.assign(cpt_n, 0.0);
            for (uint32_t idx = 0; idx < cpt_n; idx++) {
                uint32_t ts = (idx / f.strides[ed.local_slot]) % td;
                if (std::isfinite(max_all) && z_all > 0.0) {
                    cache[e].p_joint[idx] = std::exp(eta[idx] - max_all) / z_all;
                }
                if (std::isfinite(max_by_s[ts]) && z_by_s[ts] > 0.0) {
                    cache[e].q_cond[idx] = std::exp(eta[idx] - max_by_s[ts]) / z_by_s[ts];
                }
            }
        }

        /* Step 2: Solve adjoint equilibrium m_bar = src_fv + J_BP^T m_bar */
        std::vector<std::vector<double>> m_bar = src_fv;
        std::vector<std::vector<double>> vf_bar(num_edges);
        std::vector<std::vector<double>> next_m_bar(num_edges);
        const uint32_t iters = max_vjp_iters > 0 ? max_vjp_iters : 400u;
        const double eps = tol > 0.0 ? tol : 1e-12;

        for (uint32_t it = 0; it < iters; it++) {
            for (uint32_t e = 0; e < num_edges; e++) {
                vf_bar[e].assign(var_dims[edges[e].var_idx], 0.0);
            }

            /* Pullback through factor-to-variable messages -> vf_bar */
            for (uint32_t e = 0; e < num_edges; e++) {
                const Edge& ed = edges[e];
                const FactorNode& f = factors[ed.factor_idx];
                const uint32_t arity = static_cast<uint32_t>(f.vars.size());
                const uint32_t cpt_n = static_cast<uint32_t>(f.log_pot.size());
                const uint32_t e_begin = factor_edge_offsets[ed.factor_idx];
                const uint32_t td = var_dims[ed.var_idx];
                double sum_m = 0.0;
                for (double x : m_bar[e]) sum_m += x;

                for (uint32_t idx = 0; idx < cpt_n; idx++) {
                    uint32_t ts = (idx / f.strides[ed.local_slot]) % td;
                    double eta_bar = cache[e].q_cond[idx] * m_bar[e][ts]
                                   - cache[e].p_joint[idx] * sum_m;
                    for (uint32_t sl = 0; sl < arity; sl++) {
                        if (sl == ed.local_slot) continue;
                        uint32_t v = f.vars[sl];
                        uint32_t st = (idx / f.strides[sl]) % var_dims[v];
                        vf_bar[e_begin + sl][st] += eta_bar;
                    }
                }
            }

            /* Pullback through variable-to-factor messages -> next_m_bar */
            next_m_bar = src_fv;
            for (uint32_t e = 0; e < num_edges; e++) {
                uint32_t v = edges[e].var_idx;
                uint32_t d = var_dims[v];
                double sum_vf = 0.0;
                for (double x : vf_bar[e]) sum_vf += x;
                for (uint32_t k = var_edge_offsets[v]; k < var_edge_offsets[v + 1u]; k++) {
                    uint32_t other_e = var_edges[k];
                    if (other_e == e) continue;
                    for (uint32_t s = 0; s < d; s++) {
                        double u_bar = vf_bar[e][s] - std::exp(msg_vf[e][s]) * sum_vf;
                        next_m_bar[other_e][s] += u_bar;
                    }
                }
            }

            double max_diff = 0.0;
            for (uint32_t e = 0; e < num_edges; e++) {
                for (size_t s = 0; s < m_bar[e].size(); s++) {
                    max_diff = std::max(max_diff, std::fabs(next_m_bar[e][s] - m_bar[e][s]));
                }
            }
            m_bar.swap(next_m_bar);
            if (max_diff < eps) break;
        }

        /* Step 3: Accumulate exact dL/d(theta_a) = sum_{i in a} eta_bar_{a->i} */
        for (uint32_t e = 0; e < num_edges; e++) {
            const Edge& ed = edges[e];
            const FactorNode& f = factors[ed.factor_idx];
            const uint32_t cpt_n = static_cast<uint32_t>(f.log_pot.size());
            const uint32_t td = var_dims[ed.var_idx];
            double sum_m = 0.0;
            for (double x : m_bar[e]) sum_m += x;
            for (uint32_t idx = 0; idx < cpt_n; idx++) {
                uint32_t ts = (idx / f.strides[ed.local_slot]) % td;
                double eta_bar = cache[e].q_cond[idx] * m_bar[e][ts]
                               - cache[e].p_joint[idx] * sum_m;
                out_dL_dtheta[f.param_offset + idx] += eta_bar;
            }
        }
        return true;
    }
};

/* ============================================================================
 * Public C API
 * ============================================================================ */

extern "C" eshkol_bp_config_t eshkol_bp_default_config(void) {
    eshkol_bp_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.schedule = ESHKOL_BP_RESIDUAL_ASYNC;
    cfg.max_iterations = 200u;
    cfg.tolerance = 1e-10;
    cfg.damping = 0.7;
    return cfg;
}

extern "C" eshkol_forney_fg_t* eshkol_forney_fg_create(uint32_t num_vars,
                                                       const uint32_t* var_dims) {
    if (num_vars == 0 || !var_dims) return nullptr;
    for (uint32_t i = 0; i < num_vars; i++) {
        if (var_dims[i] == 0) return nullptr;
    }
    return new (std::nothrow) eshkol_forney_fg(num_vars, var_dims);
}

extern "C" void eshkol_forney_fg_destroy(eshkol_forney_fg_t* fg) {
    delete fg;
}

extern "C" uint32_t eshkol_forney_fg_add_factor(eshkol_forney_fg_t* fg,
                                                const uint32_t* vars,
                                                uint32_t arity,
                                                const double* log_potentials,
                                                uint32_t cpt_size) {
    if (!fg || !vars || arity == 0 || !log_potentials) return UINT32_MAX;
    uint64_t expected_size = 1;
    for (uint32_t i = 0; i < arity; i++) {
        if (vars[i] >= fg->num_vars) return UINT32_MAX;
        for (uint32_t j = 0; j < i; j++) {
            if (vars[j] == vars[i]) return UINT32_MAX;
        }
        expected_size *= fg->var_dims[vars[i]];
        if (expected_size > 10000000ULL) return UINT32_MAX;
    }
    if (expected_size != cpt_size) return UINT32_MAX;
    for (uint32_t i = 0; i < cpt_size; i++) {
        if (std::isnan(log_potentials[i])) return UINT32_MAX;
    }

    FactorNode fn;
    fn.vars.assign(vars, vars + arity);
    fn.strides.resize(arity, 1u);
    for (uint32_t i = arity; i-- > 1; ) {
        fn.strides[i - 1u] = fn.strides[i] * fg->var_dims[vars[i]];
    }
    fn.log_pot.assign(log_potentials, log_potentials + cpt_size);
    fn.rho = 1.0;
    fn.param_offset = fg->total_params;
    fg->total_params += cpt_size;

    uint32_t idx = static_cast<uint32_t>(fg->factors.size());
    fg->factors.push_back(std::move(fn));
    fg->csr_dirty = true;
    fg->has_messages = false;
    return idx;
}

extern "C" bool eshkol_forney_fg_set_factor_potentials(eshkol_forney_fg_t* fg,
                                                       uint32_t factor_idx,
                                                       const double* log_potentials,
                                                       uint32_t cpt_size) {
    if (!fg || !log_potentials || factor_idx >= fg->factors.size()) return false;
    FactorNode& f = fg->factors[factor_idx];
    if (f.log_pot.size() != cpt_size) return false;
    for (uint32_t i = 0; i < cpt_size; i++) {
        if (std::isnan(log_potentials[i])) return false;
    }
    std::memcpy(f.log_pot.data(), log_potentials, static_cast<size_t>(cpt_size) * sizeof(double));
    fg->has_messages = false;
    return true;
}

extern "C" bool eshkol_forney_fg_set_factor_rho(eshkol_forney_fg_t* fg,
                                                uint32_t factor_idx,
                                                double rho) {
    if (!fg || factor_idx >= fg->factors.size()) return false;
    if (!std::isfinite(rho) || rho <= 0.0 || rho > 1.0) return false;
    fg->factors[factor_idx].rho = rho;
    return true;
}

extern "C" bool eshkol_forney_fg_observe(eshkol_forney_fg_t* fg,
                                         uint32_t var_idx,
                                         int32_t state) {
    if (!fg || var_idx >= fg->num_vars) return false;
    if (state >= static_cast<int32_t>(fg->var_dims[var_idx])) return false;
    fg->obs_state[var_idx] = state < 0 ? -1 : state;
    fg->has_messages = false;
    return true;
}

extern "C" uint32_t eshkol_forney_fg_num_vars(const eshkol_forney_fg_t* fg) {
    return fg ? fg->num_vars : 0u;
}

extern "C" uint32_t eshkol_forney_fg_num_factors(const eshkol_forney_fg_t* fg) {
    return fg ? static_cast<uint32_t>(fg->factors.size()) : 0u;
}

extern "C" uint32_t eshkol_forney_fg_total_params(const eshkol_forney_fg_t* fg) {
    return fg ? fg->total_params : 0u;
}

extern "C" uint32_t eshkol_forney_fg_factor_size(const eshkol_forney_fg_t* fg,
                                                 uint32_t factor_idx) {
    if (!fg || factor_idx >= fg->factors.size()) return 0u;
    return static_cast<uint32_t>(fg->factors[factor_idx].log_pot.size());
}

extern "C" bool eshkol_forney_fg_infer(eshkol_forney_fg_t* fg,
                                       const eshkol_bp_config_t* cfg,
                                       eshkol_bp_stats_t* out_stats) {
    if (!fg) return false;
    eshkol_bp_config_t eff = cfg ? *cfg : eshkol_bp_default_config();
    return fg->infer(eff, out_stats);
}

extern "C" bool eshkol_forney_fg_var_marginal(const eshkol_forney_fg_t* fg,
                                              uint32_t var_idx,
                                              double* out_probs,
                                              uint32_t out_len) {
    if (!fg || !out_probs || var_idx >= fg->num_vars) return false;
    if (out_len < fg->var_dims[var_idx]) return false;
    std::memcpy(out_probs, fg->var_belief[var_idx].data(),
                static_cast<size_t>(fg->var_dims[var_idx]) * sizeof(double));
    return true;
}

extern "C" bool eshkol_forney_fg_factor_marginal(const eshkol_forney_fg_t* fg,
                                                 uint32_t factor_idx,
                                                 double* out_probs,
                                                 uint32_t out_len) {
    if (!fg || !out_probs || !fg->has_messages || factor_idx >= fg->factor_belief.size()) {
        return false;
    }
    const auto& fb = fg->factor_belief[factor_idx];
    if (out_len < fb.size()) return false;
    std::memcpy(out_probs, fb.data(), fb.size() * sizeof(double));
    return true;
}

extern "C" double eshkol_forney_fg_bethe_free_energy(const eshkol_forney_fg_t* fg) {
    if (!fg || !fg->has_messages) return std::numeric_limits<double>::quiet_NaN();
    return fg->compute_bethe_fe(false);
}

extern "C" bool eshkol_forney_fg_exact_neg_log_z(const eshkol_forney_fg_t* fg,
                                                 double* out_neg_log_z) {
    if (!fg || !out_neg_log_z) return false;
    uint64_t total_states = 1;
    for (uint32_t v = 0; v < fg->num_vars; v++) {
        uint64_t mult = (fg->obs_state[v] >= 0) ? 1ULL : fg->var_dims[v];
        total_states *= mult;
        if (total_states > (1ULL << 20)) return false;
    }

    std::vector<uint32_t> assign(fg->num_vars, 0u);
    std::vector<double> scores;
    scores.reserve(static_cast<size_t>(total_states));

    for (uint64_t s_idx = 0; s_idx < total_states; s_idx++) {
        uint64_t rem = s_idx;
        for (uint32_t v = fg->num_vars; v-- > 0; ) {
            if (fg->obs_state[v] >= 0) {
                assign[v] = static_cast<uint32_t>(fg->obs_state[v]);
            } else {
                assign[v] = static_cast<uint32_t>(rem % fg->var_dims[v]);
                rem /= fg->var_dims[v];
            }
        }
        double log_p = 0.0;
        for (const FactorNode& f : fg->factors) {
            uint32_t fidx = 0;
            for (uint32_t sl = 0; sl < f.vars.size(); sl++) {
                fidx += assign[f.vars[sl]] * f.strides[sl];
            }
            log_p += f.log_pot[fidx];
        }
        scores.push_back(log_p);
    }

    double log_z = log_sum_exp(scores.data(), static_cast<uint32_t>(scores.size()));
    *out_neg_log_z = -log_z;
    return std::isfinite(*out_neg_log_z);
}

extern "C" bool eshkol_fg_free_energy_grad(const eshkol_forney_fg_t* fg,
                                           double* out_grads,
                                           uint32_t grads_len,
                                           uint64_t* out_fd_evals) {
    if (!fg || !fg->has_messages || !out_grads || grads_len < fg->total_params) {
        return false;
    }
    if (out_fd_evals) *out_fd_evals = 0;
    /* Analytic Bethe envelope theorem: d(F_Bethe)/d(theta_a(x_a)) = -b_a*(x_a) */
    for (uint32_t fi = 0; fi < fg->factors.size(); fi++) {
        const FactorNode& f = fg->factors[fi];
        const auto& ba = fg->factor_belief[fi];
        for (uint32_t idx = 0; idx < ba.size(); idx++) {
            out_grads[f.param_offset + idx] = -ba[idx];
        }
    }
    return true;
}

extern "C" bool eshkol_fg_marginal_vjp(const eshkol_forney_fg_t* fg,
                                       const double* dL_db_vars,
                                       uint32_t var_states_len,
                                       double* out_dL_dtheta,
                                       uint32_t theta_len,
                                       uint32_t max_vjp_iters,
                                       double tol,
                                       uint64_t* out_fd_evals) {
    if (!fg) return false;
    return fg->marginal_vjp(dL_db_vars, var_states_len, out_dL_dtheta, theta_len,
                            max_vjp_iters, tol, out_fd_evals);
}

extern "C" bool eshkol_fg_certify_free_energy(eshkol_forney_fg_t* fg,
                                              const eshkol_bp_config_t* trw_cfg,
                                              eshkol_fe_interval_t* out_interval) {
    if (!fg || !out_interval) return false;
    return fg->certify_free_energy(trw_cfg, out_interval);
}

extern "C" bool eshkol_fg_interval_contains(const eshkol_fe_interval_t* interval,
                                            double value) {
    if (!interval || !interval->valid || !std::isfinite(value)) return false;
    return value >= interval->lower && value <= interval->upper;
}
