/*
 * Colored Differentiable E-Graph (diff-egglog + SmoothE Cost Extraction)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/diff_egglog.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct ENode {
    uint32_t op_id = 0;
    std::vector<uint32_t> children;
    double cost = 1.0;
};

struct MergeRecord {
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t color = 0;
};

struct CanonKey {
    uint32_t op_id = 0;
    std::vector<uint32_t> children;

    bool operator==(const CanonKey& o) const {
        return op_id == o.op_id && children == o.children;
    }
};

struct CanonKeyHash {
    size_t operator()(const CanonKey& k) const {
        uint64_t h = 14695981039346656037ULL;
        h ^= k.op_id;
        h *= 1099511628211ULL;
        for (uint32_t c : k.children) {
            h ^= c;
            h *= 1099511628211ULL;
        }
        return static_cast<size_t>(h);
    }
};

struct UnionFind {
    std::vector<uint32_t> parent;

    void reset(uint32_t n) {
        parent.resize(n);
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
        if (ra < rb) parent[rb] = ra;
        else parent[ra] = rb;
        return true;
    }
};

struct ColorView {
    bool valid = false;
    UnionFind uf;
};

} /* anonymous namespace */

struct eshkol_egraph {
    /* Color 0 is the root base color (parent 0). */
    std::vector<uint32_t> color_parents = {0u};
    std::vector<ENode> nodes;
    std::vector<MergeRecord> merges;
    std::vector<ColorView> views;

    eshkol_egraph() {
        views.resize(1u);
    }

    void invalidate_views() {
        for (auto& v : views) v.valid = false;
    }

    void collect_color_ancestry(uint32_t color_id, std::unordered_set<uint32_t>& anc) const {
        anc.clear();
        uint32_t cur = color_id;
        while (true) {
            anc.insert(cur);
            if (cur == 0u) break;
            uint32_t p = color_parents[cur];
            if (p == cur) break;
            cur = p;
        }
    }

    void ensure_color_view(uint32_t color_id) {
        if (color_id >= views.size()) return;
        ColorView& cv = views[color_id];
        if (cv.valid) return;

        const uint32_t n = static_cast<uint32_t>(nodes.size());
        cv.uf.reset(n);

        std::unordered_set<uint32_t> active_colors;
        collect_color_ancestry(color_id, active_colors);

        for (const MergeRecord& m : merges) {
            if (active_colors.find(m.color) != active_colors.end()) {
                cv.uf.unite(m.a, m.b);
            }
        }

        /* Exact congruence closure saturation */
        bool changed = true;
        std::unordered_map<CanonKey, uint32_t, CanonKeyHash> hashcons;
        while (changed) {
            changed = false;
            hashcons.clear();
            hashcons.reserve(n * 2u + 8u);
            for (uint32_t id = 0; id < n; id++) {
                CanonKey key;
                key.op_id = nodes[id].op_id;
                key.children.resize(nodes[id].children.size());
                for (size_t k = 0; k < nodes[id].children.size(); k++) {
                    key.children[k] = cv.uf.find(nodes[id].children[k]);
                }
                uint32_t rep = cv.uf.find(id);
                auto it = hashcons.find(key);
                if (it == hashcons.end()) {
                    hashcons.emplace(std::move(key), rep);
                } else {
                    if (cv.uf.unite(it->second, rep)) {
                        changed = true;
                    }
                }
            }
        }
        cv.valid = true;
    }

    uint32_t add_node(uint32_t op_id, const uint32_t* children, uint32_t num_children, double cost) {
        if ((num_children > 0 && !children) || !std::isfinite(cost)) return UINT32_MAX;
        for (uint32_t i = 0; i < num_children; i++) {
            if (children[i] >= nodes.size()) return UINT32_MAX;
        }

        /* Hash-cons against existing nodes in base color 0 */
        ensure_color_view(0u);
        std::vector<uint32_t> canon_ch(num_children);
        for (uint32_t i = 0; i < num_children; i++) {
            canon_ch[i] = views[0].uf.find(children[i]);
        }
        for (uint32_t id = 0; id < nodes.size(); id++) {
            if (nodes[id].op_id != op_id || nodes[id].children.size() != num_children) continue;
            bool match = true;
            for (uint32_t k = 0; k < num_children; k++) {
                if (views[0].uf.find(nodes[id].children[k]) != canon_ch[k]) {
                    match = false;
                    break;
                }
            }
            if (match) return views[0].uf.find(id);
        }

        ENode node;
        node.op_id = op_id;
        if (num_children > 0) node.children.assign(children, children + num_children);
        node.cost = cost;
        uint32_t new_id = static_cast<uint32_t>(nodes.size());
        nodes.push_back(std::move(node));
        invalidate_views();
        return new_id;
    }

    bool smoothe_extract(uint32_t root_eclass,
                         uint32_t color_id,
                         double temperature,
                         uint32_t num_iters,
                         eshkol_smoothe_result_t* out_res,
                         double* out_node_grads,
                         uint32_t grads_len) {
        if (!out_res) return false;
        std::memset(out_res, 0, sizeof(*out_res));
        if (color_id >= views.size() || root_eclass >= nodes.size()) return false;
        if (!std::isfinite(temperature) || temperature <= 0.0) return false;
        if (out_node_grads && grads_len < nodes.size()) return false;

        ensure_color_view(color_id);
        UnionFind& uf = views[color_id].uf;
        const uint32_t N = static_cast<uint32_t>(nodes.size());
        const uint32_t root_rep = uf.find(root_eclass);

        /* Compact active e-classes and canonicalized node children in this color */
        std::vector<int32_t> rep_to_cid(N, -1);
        std::vector<std::vector<uint32_t>> class_nodes;
        std::vector<uint32_t> node_class(N, 0u);
        std::vector<std::vector<uint32_t>> node_child_cids(N);

        for (uint32_t i = 0; i < N; i++) {
            uint32_t r = uf.find(i);
            if (rep_to_cid[r] < 0) {
                rep_to_cid[r] = static_cast<int32_t>(class_nodes.size());
                class_nodes.emplace_back();
            }
            uint32_t cid = static_cast<uint32_t>(rep_to_cid[r]);
            node_class[i] = cid;
            class_nodes[cid].push_back(i);
        }
        for (uint32_t i = 0; i < N; i++) {
            node_child_cids[i].resize(nodes[i].children.size());
            for (size_t k = 0; k < nodes[i].children.size(); k++) {
                node_child_cids[i][k] = static_cast<uint32_t>(rep_to_cid[uf.find(nodes[i].children[k])]);
            }
        }

        const uint32_t C = static_cast<uint32_t>(class_nodes.size());
        const uint32_t root_cid = static_cast<uint32_t>(rep_to_cid[root_rep]);
        const uint32_t T = num_iters > 0 ? num_iters : 32u;
        constexpr double kInf = std::numeric_limits<double>::infinity();

        /* Forward tape: V[t][c] and per-node Gibbs weights W[t][n] for t = 1..T */
        std::vector<std::vector<double>> V(T + 1u, std::vector<double>(C, kInf));
        std::vector<std::vector<double>> W(T + 1u, std::vector<double>(N, 0.0));

        for (uint32_t t = 1; t <= T; t++) {
            for (uint32_t c = 0; c < C; c++) {
                const auto& members = class_nodes[c];
                std::vector<double> q_vals(members.size(), kInf);
                double min_q = kInf;
                for (size_t mi = 0; mi < members.size(); mi++) {
                    uint32_t nid = members[mi];
                    double q = nodes[nid].cost;
                    bool finite_children = true;
                    for (uint32_t ch_cid : node_child_cids[nid]) {
                        double vc = V[t - 1u][ch_cid];
                        if (!std::isfinite(vc)) {
                            finite_children = false;
                            break;
                        }
                        q += vc;
                    }
                    if (finite_children) {
                        q_vals[mi] = q;
                        if (q < min_q) min_q = q;
                    }
                }
                if (!std::isfinite(min_q)) {
                    V[t][c] = kInf;
                    continue;
                }
                double sum_exp = 0.0;
                for (size_t mi = 0; mi < members.size(); mi++) {
                    if (std::isfinite(q_vals[mi])) {
                        sum_exp += std::exp(-(q_vals[mi] - min_q) / temperature);
                    }
                }
                V[t][c] = min_q - temperature * std::log(sum_exp);
                for (size_t mi = 0; mi < members.size(); mi++) {
                    if (std::isfinite(q_vals[mi])) {
                        W[t][members[mi]] = std::exp(-(q_vals[mi] - min_q) / temperature) / sum_exp;
                    }
                }
            }
        }

        double root_cost = V[T][root_cid];
        if (!std::isfinite(root_cost)) return false;

        out_res->root_cost = root_cost;
        out_res->steps_used = T;
        out_res->fd_evals = 0;
        out_res->valid = true;

        /* Exact analytical reverse-mode differentiation through the T SmoothE steps */
        if (out_node_grads) {
            std::memset(out_node_grads, 0, static_cast<size_t>(N) * sizeof(double));
            std::vector<double> v_bar(C, 0.0);
            std::vector<double> prev_v_bar(C, 0.0);
            v_bar[root_cid] = 1.0;

            for (uint32_t t = T; t >= 1u; t--) {
                std::fill(prev_v_bar.begin(), prev_v_bar.end(), 0.0);
                for (uint32_t c = 0; c < C; c++) {
                    double vb = v_bar[c];
                    if (vb == 0.0) continue;
                    for (uint32_t nid : class_nodes[c]) {
                        double w = W[t][nid];
                        if (w == 0.0) continue;
                        double q_bar = vb * w;
                        out_node_grads[nid] += q_bar;
                        for (uint32_t ch_cid : node_child_cids[nid]) {
                            prev_v_bar[ch_cid] += q_bar;
                        }
                    }
                }
                v_bar.swap(prev_v_bar);
            }
        }

        return true;
    }
};

/* ============================================================================
 * Public C API
 * ============================================================================ */

extern "C" eshkol_egraph_t* eshkol_egraph_create(void) {
    return new (std::nothrow) eshkol_egraph();
}

extern "C" void eshkol_egraph_destroy(eshkol_egraph_t* eg) {
    delete eg;
}

extern "C" uint32_t eshkol_egraph_create_color(eshkol_egraph_t* eg, uint32_t parent_color) {
    if (!eg || parent_color >= eg->color_parents.size()) return UINT32_MAX;
    uint32_t cid = static_cast<uint32_t>(eg->color_parents.size());
    eg->color_parents.push_back(parent_color);
    eg->views.emplace_back();
    return cid;
}

extern "C" uint32_t eshkol_egraph_add_node(eshkol_egraph_t* eg,
                                           uint32_t op_id,
                                           const uint32_t* children,
                                           uint32_t num_children,
                                           double cost) {
    if (!eg) return UINT32_MAX;
    return eg->add_node(op_id, children, num_children, cost);
}

extern "C" bool eshkol_egraph_set_node_cost(eshkol_egraph_t* eg, uint32_t node_id, double cost) {
    if (!eg || node_id >= eg->nodes.size() || !std::isfinite(cost)) return false;
    eg->nodes[node_id].cost = cost;
    return true;
}

extern "C" bool eshkol_egraph_union(eshkol_egraph_t* eg,
                                    uint32_t class_a,
                                    uint32_t class_b,
                                    uint32_t color_id) {
    if (!eg || color_id >= eg->color_parents.size()) return false;
    if (class_a >= eg->nodes.size() || class_b >= eg->nodes.size()) return false;
    eg->merges.push_back({class_a, class_b, color_id});
    eg->invalidate_views();
    return true;
}

extern "C" bool eshkol_egraph_rebuild(eshkol_egraph_t* eg) {
    if (!eg) return false;
    for (uint32_t c = 0; c < eg->views.size(); c++) {
        eg->ensure_color_view(c);
    }
    return true;
}

extern "C" uint32_t eshkol_egraph_find(eshkol_egraph_t* eg, uint32_t eclass, uint32_t color_id) {
    if (!eg || color_id >= eg->views.size() || eclass >= eg->nodes.size()) return UINT32_MAX;
    eg->ensure_color_view(color_id);
    return eg->views[color_id].uf.find(eclass);
}

extern "C" bool eshkol_egraph_equiv(eshkol_egraph_t* eg,
                                    uint32_t a,
                                    uint32_t b,
                                    uint32_t color_id) {
    uint32_t ra = eshkol_egraph_find(eg, a, color_id);
    uint32_t rb = eshkol_egraph_find(eg, b, color_id);
    return ra != UINT32_MAX && ra == rb;
}

extern "C" bool eshkol_egraph_smoothe_extract(eshkol_egraph_t* eg,
                                              uint32_t root_eclass,
                                              uint32_t color_id,
                                              double temperature,
                                              uint32_t num_iters,
                                              eshkol_smoothe_result_t* out_result,
                                              double* out_node_grads,
                                              uint32_t grads_len) {
    if (!eg) return false;
    return eg->smoothe_extract(root_eclass, color_id, temperature, num_iters,
                               out_result, out_node_grads, grads_len);
}

extern "C" uint32_t eshkol_egraph_num_colors(const eshkol_egraph_t* eg) {
    return eg ? static_cast<uint32_t>(eg->color_parents.size()) : 0u;
}

extern "C" uint32_t eshkol_egraph_color_parent(const eshkol_egraph_t* eg, uint32_t color_id) {
    if (!eg || color_id >= eg->color_parents.size()) return UINT32_MAX;
    return eg->color_parents[color_id];
}

extern "C" uint32_t eshkol_egraph_num_nodes(const eshkol_egraph_t* eg) {
    return eg ? static_cast<uint32_t>(eg->nodes.size()) : 0u;
}

extern "C" bool eshkol_egraph_inspect_node(const eshkol_egraph_t* eg,
                                           uint32_t node_id,
                                           uint32_t* out_op_id,
                                           const uint32_t** out_children,
                                           uint32_t* out_num_children,
                                           double* out_cost) {
    if (!eg || node_id >= eg->nodes.size()) return false;
    const ENode& n = eg->nodes[node_id];
    if (out_op_id) *out_op_id = n.op_id;
    if (out_children) *out_children = n.children.empty() ? nullptr : n.children.data();
    if (out_num_children) *out_num_children = static_cast<uint32_t>(n.children.size());
    if (out_cost) *out_cost = n.cost;
    return true;
}

extern "C" uint32_t eshkol_egraph_num_merges(const eshkol_egraph_t* eg) {
    return eg ? static_cast<uint32_t>(eg->merges.size()) : 0u;
}

extern "C" bool eshkol_egraph_inspect_merge(const eshkol_egraph_t* eg,
                                            uint32_t merge_idx,
                                            uint32_t* out_class_a,
                                            uint32_t* out_class_b,
                                            uint32_t* out_color_id) {
    if (!eg || merge_idx >= eg->merges.size()) return false;
    const MergeRecord& m = eg->merges[merge_idx];
    if (out_class_a) *out_class_a = m.a;
    if (out_class_b) *out_class_b = m.b;
    if (out_color_id) *out_color_id = m.color;
    return true;
}
