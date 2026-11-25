/*
 * Complementary Learning Systems (CLS) Memory: Hippocampal Episodic Z-Set Buffer
 * + Neocortical Semantic Network with Fisher-Protected Consolidation
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/cls_zset_memory.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double next_sym() {
        double u = static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
        return 2.0 * u - 1.0;
    }
};

struct EpisodeEntry {
    int64_t weight = 0;
    std::vector<double> x;
    std::vector<double> y;
};

} /* namespace */

struct eshkol_cls_memory {
    uint32_t in_dim = 0;
    uint32_t hidden_dim = 0;
    uint32_t out_dim = 0;

    /* Neocortical semantic network weights (flattened):
     *   W1 [hidden x in], b1 [hidden], W2 [out x hidden], b2 [out]. */
    std::vector<double> weights;
    std::vector<double> anchor_weights;
    std::vector<double> fisher_diag;
    bool has_anchor = false;

    /* Strictly isolated hippocampal episodic Z-set buffer. */
    std::unordered_map<uint64_t, EpisodeEntry> episodic_zset;

    size_t off_W1() const { return 0; }
    size_t off_b1() const { return static_cast<size_t>(hidden_dim) * in_dim; }
    size_t off_W2() const { return off_b1() + hidden_dim; }
    size_t off_b2() const { return off_W2() + static_cast<size_t>(out_dim) * hidden_dim; }
    size_t total_params() const { return off_b2() + out_dim; }

    double sample_loss_and_grad(const double* x, const double* y,
                                std::vector<double>& grad_accum) const {
        std::vector<double> h(hidden_dim, 0.0);
        std::vector<double> out(out_dim, 0.0);
        const size_t oW1 = off_W1(), ob1 = off_b1(), oW2 = off_W2(), ob2 = off_b2();

        for (uint32_t r = 0; r < hidden_dim; ++r) {
            double s = weights[ob1 + r];
            for (uint32_t c = 0; c < in_dim; ++c) {
                s += weights[oW1 + r * in_dim + c] * x[c];
            }
            h[r] = std::tanh(s);
        }

        double loss = 0.0;
        std::vector<double> d_out(out_dim, 0.0);
        for (uint32_t r = 0; r < out_dim; ++r) {
            double s = weights[ob2 + r];
            for (uint32_t c = 0; c < hidden_dim; ++c) {
                s += weights[oW2 + r * hidden_dim + c] * h[c];
            }
            out[r] = s;
            double diff = s - y[r];
            loss += 0.5 * diff * diff;
            d_out[r] = diff;
        }

        std::vector<double> d_h(hidden_dim, 0.0);
        for (uint32_t r = 0; r < out_dim; ++r) {
            grad_accum[ob2 + r] += d_out[r];
            for (uint32_t c = 0; c < hidden_dim; ++c) {
                grad_accum[oW2 + r * hidden_dim + c] += d_out[r] * h[c];
                d_h[c] += d_out[r] * weights[oW2 + r * hidden_dim + c];
            }
        }

        for (uint32_t r = 0; r < hidden_dim; ++r) {
            double dz = d_h[r] * (1.0 - h[r] * h[r]);
            grad_accum[ob1 + r] += dz;
            for (uint32_t c = 0; c < in_dim; ++c) {
                grad_accum[oW1 + r * in_dim + c] += dz * x[c];
            }
        }
        return loss;
    }

    double dataset_loss(const double* xs, const double* ys, size_t n_samples) const {
        if (!xs || !ys || n_samples == 0) return 0.0;
        const size_t oW1 = off_W1(), ob1 = off_b1(), oW2 = off_W2(), ob2 = off_b2();
        std::vector<double> h(hidden_dim, 0.0);
        double total = 0.0;
        for (size_t s = 0; s < n_samples; ++s) {
            const double* x = xs + s * in_dim;
            const double* y = ys + s * out_dim;
            for (uint32_t r = 0; r < hidden_dim; ++r) {
                double acc = weights[ob1 + r];
                for (uint32_t c = 0; c < in_dim; ++c) {
                    acc += weights[oW1 + r * in_dim + c] * x[c];
                }
                h[r] = std::tanh(acc);
            }
            for (uint32_t r = 0; r < out_dim; ++r) {
                double acc = weights[ob2 + r];
                for (uint32_t c = 0; c < hidden_dim; ++c) {
                    acc += weights[oW2 + r * hidden_dim + c] * h[c];
                }
                double diff = acc - y[r];
                total += 0.5 * diff * diff;
            }
        }
        return total / static_cast<double>(n_samples);
    }
};

extern "C" {

eshkol_cls_memory_t* eshkol_cls_memory_create(uint32_t in_dim,
                                              uint32_t hidden_dim,
                                              uint32_t out_dim,
                                              uint64_t seed) {
    if (in_dim == 0 || hidden_dim == 0 || out_dim == 0) return nullptr;
    auto* cls = new eshkol_cls_memory();
    cls->in_dim = in_dim;
    cls->hidden_dim = hidden_dim;
    cls->out_dim = out_dim;

    const size_t P = cls->total_params();
    cls->weights.assign(P, 0.0);
    cls->anchor_weights.assign(P, 0.0);
    cls->fisher_diag.assign(P, 0.0);

    SplitMix64 rng(seed);
    const double scale1 = std::sqrt(2.0 / static_cast<double>(in_dim + hidden_dim));
    const double scale2 = std::sqrt(2.0 / static_cast<double>(hidden_dim + out_dim));
    for (size_t i = cls->off_W1(); i < cls->off_b1(); ++i) cls->weights[i] = rng.next_sym() * scale1;
    for (size_t i = cls->off_W2(); i < cls->off_b2(); ++i) cls->weights[i] = rng.next_sym() * scale2;
    return cls;
}

void eshkol_cls_memory_destroy(eshkol_cls_memory_t* cls) {
    delete cls;
}

bool eshkol_cls_semantic_train(eshkol_cls_memory_t* cls,
                               const double* xs,
                               const double* ys,
                               size_t n_samples,
                               uint32_t epochs,
                               double lr) {
    if (!cls || !xs || !ys || n_samples == 0 || !std::isfinite(lr) || lr <= 0.0) {
        return false;
    }
    const size_t P = cls->total_params();
    std::vector<double> grads(P, 0.0);
    std::vector<double> m(P, 0.0);
    std::vector<double> v(P, 0.0);
    const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

    for (uint32_t ep = 1; ep <= epochs; ++ep) {
        std::fill(grads.begin(), grads.end(), 0.0);
        for (size_t s = 0; s < n_samples; ++s) {
            cls->sample_loss_and_grad(xs + s * cls->in_dim, ys + s * cls->out_dim, grads);
        }
        const double inv_n = 1.0 / static_cast<double>(n_samples);
        const double b1_corr = 1.0 - std::pow(beta1, static_cast<double>(ep));
        const double b2_corr = 1.0 - std::pow(beta2, static_cast<double>(ep));
        for (size_t i = 0; i < P; ++i) {
            double g = grads[i] * inv_n;
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            double m_hat = m[i] / b1_corr;
            double v_hat = v[i] / b2_corr;
            cls->weights[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }

    /* Snapshot anchor weights & compute diagonal empirical Fisher information. */
    cls->anchor_weights = cls->weights;
    std::fill(cls->fisher_diag.begin(), cls->fisher_diag.end(), 0.0);
    std::vector<double> sgrad(P, 0.0);
    for (size_t s = 0; s < n_samples; ++s) {
        std::fill(sgrad.begin(), sgrad.end(), 0.0);
        cls->sample_loss_and_grad(xs + s * cls->in_dim, ys + s * cls->out_dim, sgrad);
        for (size_t i = 0; i < P; ++i) {
            cls->fisher_diag[i] += sgrad[i] * sgrad[i];
        }
    }
    const double inv_n = 1.0 / static_cast<double>(n_samples);
    for (size_t i = 0; i < P; ++i) {
        cls->fisher_diag[i] = cls->fisher_diag[i] * inv_n + 1e-2; /* Positive definite floor */
    }
    cls->has_anchor = true;
    return true;
}

bool eshkol_cls_semantic_eval(const eshkol_cls_memory_t* cls,
                              const double* xs,
                              const double* ys,
                              size_t n_samples,
                              double* out_loss,
                              double* out_grad_norm,
                              uint64_t* out_fd_evals) {
    if (!cls || !xs || !ys || n_samples == 0) return false;
    if (out_fd_evals) *out_fd_evals = 0;

    const size_t P = cls->total_params();
    std::vector<double> grads(P, 0.0);
    double total_loss = 0.0;
    for (size_t s = 0; s < n_samples; ++s) {
        total_loss += cls->sample_loss_and_grad(
            xs + s * cls->in_dim, ys + s * cls->out_dim, grads);
    }
    const double inv_n = 1.0 / static_cast<double>(n_samples);
    if (out_loss) *out_loss = total_loss * inv_n;
    if (out_grad_norm) {
        double sq = 0.0;
        for (size_t i = 0; i < P; ++i) {
            double g = grads[i] * inv_n;
            sq += g * g;
        }
        *out_grad_norm = std::sqrt(sq);
    }
    return true;
}

bool eshkol_cls_episodic_ingest(eshkol_cls_memory_t* cls,
                                uint64_t episode_id,
                                const double* x,
                                const double* y,
                                int64_t weight_delta) {
    if (!cls || !x || !y) return false;
    if (weight_delta == 0) return true;

    auto it = cls->episodic_zset.find(episode_id);
    if (it == cls->episodic_zset.end()) {
        EpisodeEntry entry;
        entry.weight = weight_delta;
        entry.x.assign(x, x + cls->in_dim);
        entry.y.assign(y, y + cls->out_dim);
        cls->episodic_zset.emplace(episode_id, std::move(entry));
    } else {
        it->second.weight += weight_delta;
        if (it->second.weight == 0) {
            cls->episodic_zset.erase(it);
        } else {
            it->second.x.assign(x, x + cls->in_dim);
            it->second.y.assign(y, y + cls->out_dim);
        }
    }
    /* Strictly isolated: neocortical cls->weights is never touched here. */
    return true;
}

size_t eshkol_cls_episodic_active_count(const eshkol_cls_memory_t* cls) {
    return cls ? cls->episodic_zset.size() : 0;
}

uint64_t eshkol_cls_semantic_weights_checksum(const eshkol_cls_memory_t* cls) {
    if (!cls) return 0;
    uint64_t h = 14695981039346656037ULL;
    for (double w : cls->weights) {
        uint64_t bits = 0;
        std::memcpy(&bits, &w, sizeof(double));
        for (int i = 0; i < 8; ++i) {
            h ^= static_cast<uint64_t>((bits >> (8 * i)) & 0xFFu);
            h *= 1099511628211ULL;
        }
    }
    return h;
}

bool eshkol_cls_consolidate_guarded(eshkol_cls_memory_t* cls,
                                    const double* anchor_xs,
                                    const double* anchor_ys,
                                    size_t n_anchor_samples,
                                    double lr,
                                    double fisher_lambda,
                                    double max_semantic_loss_delta,
                                    size_t* out_accepted_steps,
                                    size_t* out_rejected_steps) {
    if (!cls || !anchor_xs || !anchor_ys || n_anchor_samples == 0 ||
        !std::isfinite(lr) || lr <= 0.0 || !std::isfinite(fisher_lambda) || fisher_lambda < 0.0) {
        return false;
    }

    const size_t P = cls->total_params();
    const double base_loss = cls->dataset_loss(anchor_xs, anchor_ys, n_anchor_samples);

    /* Sort episode ids for deterministic consolidation replay. */
    std::vector<uint64_t> ids;
    ids.reserve(cls->episodic_zset.size());
    for (const auto& kv : cls->episodic_zset) {
        if (kv.second.weight > 0) ids.push_back(kv.first);
    }
    std::sort(ids.begin(), ids.end());

    size_t accepted = 0;
    size_t rejected = 0;
    std::vector<double> grad(P, 0.0);
    std::vector<double> saved_weights(P, 0.0);

    const size_t batch_size = 64;
    for (size_t start = 0; start < ids.size(); start += batch_size) {
        size_t end = std::min(ids.size(), start + batch_size);
        std::fill(grad.begin(), grad.end(), 0.0);

        double total_w = 0.0;
        for (size_t idx = start; idx < end; ++idx) {
            const EpisodeEntry& ep = cls->episodic_zset.at(ids[idx]);
            std::vector<double> ep_g(P, 0.0);
            cls->sample_loss_and_grad(ep.x.data(), ep.y.data(), ep_g);
            double mult = static_cast<double>(ep.weight);
            for (size_t p = 0; p < P; ++p) grad[p] += mult * ep_g[p];
            total_w += mult;
        }
        if (total_w <= 0.0) continue;

        saved_weights = cls->weights;
        const double inv_w = 1.0 / total_w;
        for (size_t p = 0; p < P; ++p) {
            double ewc_grad = cls->has_anchor
                ? fisher_lambda * cls->fisher_diag[p] * (cls->weights[p] - cls->anchor_weights[p])
                : 0.0;
            double total_g = grad[p] * inv_w + ewc_grad;
            /* Trust-region clip per parameter step */
            double step = std::max(-0.05, std::min(0.05, lr * total_g));
            cls->weights[p] -= step;
        }

        /* Semantic-retention verification gate: reject batches that breach retention bound. */
        double candidate_loss = cls->dataset_loss(anchor_xs, anchor_ys, n_anchor_samples);
        if (!std::isfinite(candidate_loss) || candidate_loss > base_loss + max_semantic_loss_delta) {
            cls->weights = saved_weights;
            ++rejected;
        } else {
            ++accepted;
        }
    }

    if (out_accepted_steps) *out_accepted_steps = accepted;
    if (out_rejected_steps) *out_rejected_steps = rejected;
    return true;
}

bool eshkol_cls_unguarded_direct_step(eshkol_cls_memory_t* cls,
                                      const double* x,
                                      const double* y,
                                      double lr) {
    if (!cls || !x || !y || !std::isfinite(lr) || lr <= 0.0) return false;
    const size_t P = cls->total_params();
    std::vector<double> grad(P, 0.0);
    cls->sample_loss_and_grad(x, y, grad);
    for (size_t i = 0; i < P; ++i) {
        cls->weights[i] -= lr * grad[i];
    }
    return true;
}

} /* extern "C" */
