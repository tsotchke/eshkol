/*
 * ADR-0005 Differentiable Program-Capsule Foundations
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/program_capsule.h"

#include <cmath>
#include <vector>

namespace {

inline double stable_sigmoid(double z) {
    if (z >= 0.0) {
        double ez = std::exp(-z);
        return 1.0 / (1.0 + ez);
    } else {
        double ez = std::exp(z);
        return ez / (1.0 + ez);
    }
}

struct BranchSpec {
    bool is_hard_step;
    /* For hard_step: gate = w_a^T x + b_a, then = w_b^T x + b_b, else = w_c^T x + b_c.
     * For smooth:    alpha * tanh(w_a^T x + b_a) + c, where alpha = b_b, c = b_c. */
    std::vector<double> w_a;
    double b_a;
    std::vector<double> w_b;
    double b_b;
    std::vector<double> w_c;
    double b_c;
};

} /* namespace */

struct eshkol_sym_rule {
    uint32_t input_dim = 0;
    std::vector<BranchSpec> branches;
};

struct eshkol_program_capsule {
    uint32_t input_dim = 0;
    eshkol_capsule_mode_t mode = ESHKOL_CAPSULE_MODE_STRICT;
    double tau = 1.0;
    /* Branch metadata + flattened mutable private parameter buffer (decoupled from rule). */
    std::vector<bool> branch_is_step;
    std::vector<size_t> branch_param_offset;
    std::vector<double> params;
};

extern "C" {

eshkol_sym_rule_t* eshkol_sym_rule_create(uint32_t input_dim) {
    if (input_dim == 0) return nullptr;
    auto* r = new eshkol_sym_rule();
    r->input_dim = input_dim;
    return r;
}

void eshkol_sym_rule_destroy(eshkol_sym_rule_t* rule) {
    delete rule;
}

bool eshkol_sym_rule_add_step_branch(eshkol_sym_rule_t* rule,
                                     const double* w_gate, double b_gate,
                                     const double* w_then, double b_then,
                                     const double* w_else, double b_else) {
    if (!rule || !w_gate || !w_then || !w_else) return false;
    BranchSpec br;
    br.is_hard_step = true;
    br.w_a.assign(w_gate, w_gate + rule->input_dim);
    br.b_a = b_gate;
    br.w_b.assign(w_then, w_then + rule->input_dim);
    br.b_b = b_then;
    br.w_c.assign(w_else, w_else + rule->input_dim);
    br.b_c = b_else;
    rule->branches.push_back(std::move(br));
    return true;
}

bool eshkol_sym_rule_add_smooth_branch(eshkol_sym_rule_t* rule,
                                       const double* w, double b,
                                       double alpha, double c) {
    if (!rule || !w) return false;
    BranchSpec br;
    br.is_hard_step = false;
    br.w_a.assign(w, w + rule->input_dim);
    br.b_a = b;
    br.b_b = alpha;
    br.b_c = c;
    rule->branches.push_back(std::move(br));
    return true;
}

double eshkol_sym_rule_eval(const eshkol_sym_rule_t* rule, const double* x) {
    if (!rule || !x) return 0.0;
    const uint32_t d = rule->input_dim;
    double total = 0.0;
    for (const BranchSpec& br : rule->branches) {
        double ga = br.b_a;
        for (uint32_t i = 0; i < d; ++i) ga += br.w_a[i] * x[i];
        if (br.is_hard_step) {
            if (ga > 0.0) {
                double tv = br.b_b;
                for (uint32_t i = 0; i < d; ++i) tv += br.w_b[i] * x[i];
                total += tv;
            } else {
                double ev = br.b_c;
                for (uint32_t i = 0; i < d; ++i) ev += br.w_c[i] * x[i];
                total += ev;
            }
        } else {
            total += br.b_b * std::tanh(ga) + br.b_c;
        }
    }
    return total;
}

uint32_t eshkol_sym_rule_input_dim(const eshkol_sym_rule_t* rule) {
    return rule ? rule->input_dim : 0;
}

eshkol_capsule_status_t eshkol_capsule_compile(
    const eshkol_sym_rule_t* rule,
    eshkol_capsule_mode_t mode,
    double tau,
    eshkol_program_capsule_t** out_capsule) {
    if (!rule || !out_capsule || rule->branches.empty()) {
        return ESHKOL_CAPSULE_ERR_INVALID_ARG;
    }
    *out_capsule = nullptr;

    if (mode == ESHKOL_CAPSULE_MODE_SMOOTH && (!std::isfinite(tau) || tau <= 0.0)) {
        return ESHKOL_CAPSULE_ERR_INVALID_ARG;
    }

    if (mode == ESHKOL_CAPSULE_MODE_STRICT) {
        for (const BranchSpec& br : rule->branches) {
            if (br.is_hard_step) {
                return ESHKOL_CAPSULE_ERR_NON_DIFFERENTIABLE;
            }
        }
    }

    auto* cap = new eshkol_program_capsule();
    cap->input_dim = rule->input_dim;
    cap->mode = mode;
    cap->tau = (mode == ESHKOL_CAPSULE_MODE_SMOOTH) ? tau : 1.0;

    for (const BranchSpec& br : rule->branches) {
        cap->branch_is_step.push_back(br.is_hard_step);
        cap->branch_param_offset.push_back(cap->params.size());
        if (br.is_hard_step) {
            /* Layout: [w_gate(d), b_gate, w_then(d), b_then, w_else(d), b_else] */
            cap->params.insert(cap->params.end(), br.w_a.begin(), br.w_a.end());
            cap->params.push_back(br.b_a);
            cap->params.insert(cap->params.end(), br.w_b.begin(), br.w_b.end());
            cap->params.push_back(br.b_b);
            cap->params.insert(cap->params.end(), br.w_c.begin(), br.w_c.end());
            cap->params.push_back(br.b_c);
        } else {
            /* Layout: [w(d), b, alpha, c] */
            cap->params.insert(cap->params.end(), br.w_a.begin(), br.w_a.end());
            cap->params.push_back(br.b_a);
            cap->params.push_back(br.b_b);
            cap->params.push_back(br.b_c);
        }
    }

    *out_capsule = cap;
    return ESHKOL_CAPSULE_OK;
}

void eshkol_capsule_destroy(eshkol_program_capsule_t* capsule) {
    delete capsule;
}

size_t eshkol_capsule_param_count(const eshkol_program_capsule_t* capsule) {
    return capsule ? capsule->params.size() : 0;
}

bool eshkol_capsule_get_param(const eshkol_program_capsule_t* capsule,
                              size_t idx, double* out_val) {
    if (!capsule || !out_val || idx >= capsule->params.size()) return false;
    *out_val = capsule->params[idx];
    return true;
}

bool eshkol_capsule_set_param(eshkol_program_capsule_t* capsule,
                              size_t idx, double new_val) {
    if (!capsule || idx >= capsule->params.size() || !std::isfinite(new_val)) return false;
    capsule->params[idx] = new_val;
    return true;
}

bool eshkol_capsule_eval_value_grad(const eshkol_program_capsule_t* capsule,
                                    const double* x,
                                    double* out_value,
                                    double* out_input_grads,
                                    double* out_param_grads,
                                    uint64_t* out_fd_evals) {
    if (!capsule || !x || !out_value) return false;
    if (out_fd_evals) *out_fd_evals = 0;

    const uint32_t d = capsule->input_dim;
    if (out_input_grads) {
        for (uint32_t i = 0; i < d; ++i) out_input_grads[i] = 0.0;
    }
    if (out_param_grads) {
        for (size_t i = 0; i < capsule->params.size(); ++i) out_param_grads[i] = 0.0;
    }

    double total = 0.0;
    const size_t n_branches = capsule->branch_is_step.size();

    for (size_t b = 0; b < n_branches; ++b) {
        const size_t off = capsule->branch_param_offset[b];
        if (capsule->branch_is_step[b]) {
            const size_t off_wg = off;
            const size_t off_bg = off + d;
            const size_t off_wt = off + d + 1;
            const size_t off_bt = off + 2 * d + 1;
            const size_t off_we = off + 2 * d + 2;
            const size_t off_be = off + 3 * d + 2;

            double g = capsule->params[off_bg];
            double t = capsule->params[off_bt];
            double e = capsule->params[off_be];
            for (uint32_t i = 0; i < d; ++i) {
                g += capsule->params[off_wg + i] * x[i];
                t += capsule->params[off_wt + i] * x[i];
                e += capsule->params[off_we + i] * x[i];
            }

            const double s = stable_sigmoid(g / capsule->tau);
            const double ds_dg = (s * (1.0 - s)) / capsule->tau;
            const double dy_dg = (t - e) * ds_dg;
            const double dy_dt = s;
            const double dy_de = 1.0 - s;

            total += s * t + (1.0 - s) * e;

            if (out_input_grads) {
                for (uint32_t i = 0; i < d; ++i) {
                    out_input_grads[i] += dy_dg * capsule->params[off_wg + i]
                                        + dy_dt * capsule->params[off_wt + i]
                                        + dy_de * capsule->params[off_we + i];
                }
            }
            if (out_param_grads) {
                for (uint32_t i = 0; i < d; ++i) {
                    out_param_grads[off_wg + i] += dy_dg * x[i];
                    out_param_grads[off_wt + i] += dy_dt * x[i];
                    out_param_grads[off_we + i] += dy_de * x[i];
                }
                out_param_grads[off_bg] += dy_dg;
                out_param_grads[off_bt] += dy_dt;
                out_param_grads[off_be] += dy_de;
            }
        } else {
            const size_t off_w = off;
            const size_t off_b = off + d;
            const size_t off_alpha = off + d + 1;
            const size_t off_c = off + d + 2;

            double z = capsule->params[off_b];
            for (uint32_t i = 0; i < d; ++i) z += capsule->params[off_w + i] * x[i];
            const double th = std::tanh(z);
            const double alpha = capsule->params[off_alpha];
            const double c = capsule->params[off_c];
            const double dy_dz = alpha * (1.0 - th * th);

            total += alpha * th + c;

            if (out_input_grads) {
                for (uint32_t i = 0; i < d; ++i) {
                    out_input_grads[i] += dy_dz * capsule->params[off_w + i];
                }
            }
            if (out_param_grads) {
                for (uint32_t i = 0; i < d; ++i) {
                    out_param_grads[off_w + i] += dy_dz * x[i];
                }
                out_param_grads[off_b] += dy_dz;
                out_param_grads[off_alpha] += th;
                out_param_grads[off_c] += 1.0;
            }
        }
    }

    *out_value = total;
    return true;
}

eshkol_capsule_status_t eshkol_capsule_relink_to_rule(
    const eshkol_program_capsule_t* capsule,
    eshkol_sym_rule_t* target_rule,
    bool verification_passed) {
    if (!capsule || !target_rule) return ESHKOL_CAPSULE_ERR_INVALID_ARG;
    if (!verification_passed) return ESHKOL_CAPSULE_ERR_UNVERIFIED_RELINK;
    if (capsule->input_dim != target_rule->input_dim ||
        capsule->branch_is_step.size() != target_rule->branches.size()) {
        return ESHKOL_CAPSULE_ERR_INVALID_ARG;
    }

    const uint32_t d = capsule->input_dim;
    for (size_t b = 0; b < capsule->branch_is_step.size(); ++b) {
        if (capsule->branch_is_step[b] != target_rule->branches[b].is_hard_step) {
            return ESHKOL_CAPSULE_ERR_INVALID_ARG;
        }
    }

    for (size_t b = 0; b < capsule->branch_is_step.size(); ++b) {
        const size_t off = capsule->branch_param_offset[b];
        BranchSpec& br = target_rule->branches[b];
        if (br.is_hard_step) {
            for (uint32_t i = 0; i < d; ++i) br.w_a[i] = capsule->params[off + i];
            br.b_a = capsule->params[off + d];
            for (uint32_t i = 0; i < d; ++i) br.w_b[i] = capsule->params[off + d + 1 + i];
            br.b_b = capsule->params[off + 2 * d + 1];
            for (uint32_t i = 0; i < d; ++i) br.w_c[i] = capsule->params[off + 2 * d + 2 + i];
            br.b_c = capsule->params[off + 3 * d + 2];
        } else {
            for (uint32_t i = 0; i < d; ++i) br.w_a[i] = capsule->params[off + i];
            br.b_a = capsule->params[off + d];
            br.b_b = capsule->params[off + d + 1];
            br.b_c = capsule->params[off + d + 2];
        }
    }
    return ESHKOL_CAPSULE_OK;
}

} /* extern "C" */
