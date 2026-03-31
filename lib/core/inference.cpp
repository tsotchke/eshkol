/*
 * Active Inference Implementation for Eshkol Consciousness Engine
 *
 * Implements: factor graphs, sum-product belief propagation (log-space),
 * variational free energy, expected free energy.
 * All allocations use arena_allocate_with_header (bignum pattern).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/inference.h>
#include <eshkol/eshkol.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* ===== Arena Forward Declarations ===== */

extern "C" void* arena_allocate_with_header(arena_t* arena, size_t data_size,
                                             uint8_t subtype, uint8_t flags);
extern "C" void* arena_allocate_aligned(arena_t* arena, size_t size, size_t alignment);

/* ===== Tensor Layout (mirrors LLVM tensor struct) ===== */
/* Tensor struct: {dimensions*, num_dims, elements*, total_elements} = 32 bytes
 * Elements are double bit patterns stored as int64_t. */
typedef struct {
    uint64_t* dimensions;
    uint64_t  num_dimensions;
    int64_t*  elements;       /* double bit patterns stored as int64 */
    uint64_t  total_elements;
} tensor_layout_t;

static inline double tensor_get(const tensor_layout_t* t, uint32_t idx) {
    union { int64_t i; double d; } u;
    u.i = t->elements[idx];
    return u.d;
}

static inline void tensor_set(tensor_layout_t* t, uint32_t idx, double val) {
    union { int64_t i; double d; } u;
    u.d = val;
    t->elements[idx] = u.i;
}

/* Extract tensor_layout_t from a tagged value (HEAP_PTR to tensor) */
static tensor_layout_t* extract_tensor(const eshkol_tagged_value_t* tv) {
    if (!tv || tv->type != ESHKOL_VALUE_HEAP_PTR || !tv->data.ptr_val) return NULL;
    return (tensor_layout_t*)tv->data.ptr_val;
}

/* ===== Log-Space Arithmetic ===== */

static const double LOG_ZERO = -1e30;  /* Approximate -infinity for log-space */

/* log-sum-exp: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|)) */
static double logsumexp2(double a, double b) {
    if (a == LOG_ZERO) return b;
    if (b == LOG_ZERO) return a;
    double m = (a > b) ? a : b;
    return m + log(1.0 + exp(-(fabs(a - b))));
}

/* log-sum-exp over an array */
static double logsumexp(const double* arr, uint32_t n) {
    if (n == 0) return LOG_ZERO;
    double m = arr[0];
    for (uint32_t i = 1; i < n; i++) {
        if (arr[i] > m) m = arr[i];
    }
    if (m == LOG_ZERO) return LOG_ZERO;
    double s = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        s += exp(arr[i] - m);
    }
    return m + log(s);
}

/* Normalize log-probability vector in-place (so exp sums to 1) */
static void log_normalize(double* arr, uint32_t n) {
    double z = logsumexp(arr, n);
    if (z != LOG_ZERO) {
        for (uint32_t i = 0; i < n; i++) {
            arr[i] -= z;
        }
    }
}

/* ===== Arena Allocation Helpers ===== */

static eshkol_factor_t* alloc_factor(arena_t* arena, uint32_t num_vars) {
    size_t data_size = sizeof(eshkol_factor_t) + num_vars * sizeof(uint32_t);
    /* Factor doesn't get its own heap subtype — it's part of the factor graph */
    eshkol_factor_t* f = (eshkol_factor_t*)arena_allocate_aligned(arena, data_size, 8);
    if (f) {
        memset(f, 0, data_size);
        f->num_vars = num_vars;
    }
    return f;
}

static double* alloc_doubles(arena_t* arena, uint32_t n) {
    return (double*)arena_allocate_aligned(arena, n * sizeof(double), 8);
}

static uint32_t* alloc_uint32s(arena_t* arena, uint32_t n) {
    return (uint32_t*)arena_allocate_aligned(arena, n * sizeof(uint32_t), 8);
}

/* ===== Factor Graph Construction ===== */

eshkol_factor_graph_t* eshkol_make_factor_graph(arena_t* arena,
    uint32_t num_vars, const uint32_t* var_dims) {
    if (!arena || num_vars == 0 || !var_dims) return NULL;

    eshkol_factor_graph_t* fg = (eshkol_factor_graph_t*)arena_allocate_with_header(
        arena, sizeof(eshkol_factor_graph_t), HEAP_SUBTYPE_FACTOR_GRAPH, 0);
    if (!fg) return NULL;

    fg->num_vars = num_vars;
    fg->num_factors = 0;
    fg->max_factors = 16;
    fg->_pad = 0;
    fg->total_messages = 0;

    /* Allocate variable dimensions array */
    fg->var_dims = alloc_uint32s(arena, num_vars);
    if (!fg->var_dims) return NULL;
    memcpy(fg->var_dims, var_dims, num_vars * sizeof(uint32_t));

    /* Allocate beliefs array (one log-prob vector per variable) */
    fg->beliefs = (double**)arena_allocate_aligned(arena, num_vars * sizeof(double*), 8);
    if (!fg->beliefs) return NULL;

    for (uint32_t i = 0; i < num_vars; i++) {
        fg->beliefs[i] = alloc_doubles(arena, var_dims[i]);
        if (!fg->beliefs[i]) return NULL;
        /* Initialize to uniform: log(1/dim) */
        double log_uniform = -log((double)var_dims[i]);
        for (uint32_t j = 0; j < var_dims[i]; j++) {
            fg->beliefs[i][j] = log_uniform;
        }
    }

    /* Allocate factor pointer array */
    fg->factors = (eshkol_factor_t**)arena_allocate_aligned(
        arena, fg->max_factors * sizeof(eshkol_factor_t*), 8);
    if (!fg->factors) return NULL;

    /* Message arrays will be allocated lazily when inference runs */
    fg->msg_fv = NULL;
    fg->msg_vf = NULL;

    return fg;
}

eshkol_factor_t* eshkol_make_factor(arena_t* arena,
    const uint32_t* var_indices, uint32_t num_vars,
    const double* cpt_data, const uint32_t* dims) {
    if (!arena || !var_indices || !cpt_data || !dims || num_vars == 0) return NULL;

    eshkol_factor_t* f = alloc_factor(arena, num_vars);
    if (!f) return NULL;

    /* Compute CPT size = product of all dimensions */
    uint32_t cpt_size = 1;
    for (uint32_t i = 0; i < num_vars; i++) {
        cpt_size *= dims[i];
    }
    f->cpt_size = cpt_size;

    /* Allocate and copy CPT data */
    f->cpt = alloc_doubles(arena, cpt_size);
    if (!f->cpt) return NULL;
    memcpy(f->cpt, cpt_data, cpt_size * sizeof(double));

    /* Allocate and copy dimensions */
    f->dims = alloc_uint32s(arena, num_vars);
    if (!f->dims) return NULL;
    memcpy(f->dims, dims, num_vars * sizeof(uint32_t));

    /* Copy variable indices */
    uint32_t* indices = FACTOR_VAR_INDICES(f);
    memcpy(indices, var_indices, num_vars * sizeof(uint32_t));

    return f;
}

void eshkol_fg_add_factor(arena_t* arena, eshkol_factor_graph_t* fg,
    eshkol_factor_t* factor) {
    if (!arena || !fg || !factor) return;

    /* Grow factor array if needed */
    if (fg->num_factors >= fg->max_factors) {
        uint32_t new_max = fg->max_factors * 2;
        eshkol_factor_t** new_factors = (eshkol_factor_t**)arena_allocate_aligned(
            arena, new_max * sizeof(eshkol_factor_t*), 8);
        if (!new_factors) return;
        memcpy(new_factors, fg->factors, fg->num_factors * sizeof(eshkol_factor_t*));
        fg->factors = new_factors;
        fg->max_factors = new_max;
    }

    fg->factors[fg->num_factors++] = factor;
    fg->total_messages += factor->num_vars;
}

/* ===== Belief Propagation (Sum-Product in Log-Space) ===== */

/*
 * Allocate message storage for belief propagation.
 * Messages are organized as msg[factor_idx * max_vars + var_within_factor][state].
 */
static bool allocate_messages(arena_t* arena, eshkol_factor_graph_t* fg) {
    if (fg->msg_fv && fg->msg_vf) return true; /* already allocated */

    /* Count total messages (one per factor-variable edge, in each direction) */
    uint32_t total_msgs = 0;
    for (uint32_t fi = 0; fi < fg->num_factors; fi++) {
        total_msgs += fg->factors[fi]->num_vars;
    }

    /* Allocate message pointer arrays */
    fg->msg_fv = (double**)arena_allocate_aligned(arena, total_msgs * sizeof(double*), 8);
    fg->msg_vf = (double**)arena_allocate_aligned(arena, total_msgs * sizeof(double*), 8);
    if (!fg->msg_fv || !fg->msg_vf) return false;

    /* Allocate individual message vectors */
    uint32_t msg_idx = 0;
    for (uint32_t fi = 0; fi < fg->num_factors; fi++) {
        const eshkol_factor_t* f = fg->factors[fi];
        const uint32_t* var_indices = FACTOR_VAR_INDICES(f);

        for (uint32_t vi = 0; vi < f->num_vars; vi++) {
            uint32_t var_id = var_indices[vi];
            uint32_t dim = fg->var_dims[var_id];

            fg->msg_fv[msg_idx] = alloc_doubles(arena, dim);
            fg->msg_vf[msg_idx] = alloc_doubles(arena, dim);
            if (!fg->msg_fv[msg_idx] || !fg->msg_vf[msg_idx]) return false;

            /* Initialize to uniform (log(1/dim)) */
            double log_uni = -log((double)dim);
            for (uint32_t s = 0; s < dim; s++) {
                fg->msg_fv[msg_idx][s] = log_uni;
                fg->msg_vf[msg_idx][s] = log_uni;
            }
            msg_idx++;
        }
    }

    return true;
}

/* Get the message index for factor fi, local variable position vi */
static uint32_t get_msg_idx(const eshkol_factor_graph_t* fg,
                             uint32_t fi, uint32_t vi) {
    uint32_t idx = 0;
    for (uint32_t i = 0; i < fi; i++) {
        idx += fg->factors[i]->num_vars;
    }
    return idx + vi;
}

/*
 * Compute factor-to-variable message using sum-product algorithm.
 *
 * msg_{f→v}(x_v) = sum_{x_\v} [f(x) * prod_{u ∈ ne(f)\v} msg_{u→f}(x_u)]
 *
 * In log-space: log-sum-exp over configurations of other variables.
 */
static void compute_factor_to_var_message(
    const eshkol_factor_graph_t* fg, uint32_t fi, uint32_t target_vi,
    double* out_msg) {
    const eshkol_factor_t* f = fg->factors[fi];
    const uint32_t* var_indices = FACTOR_VAR_INDICES(f);
    uint32_t target_var = var_indices[target_vi];
    uint32_t target_dim = fg->var_dims[target_var];

    /* Initialize output message */
    for (uint32_t s = 0; s < target_dim; s++) {
        out_msg[s] = LOG_ZERO;
    }

    /* Enumerate all configurations of the factor's variables */
    /* Using iterative index enumeration over the factor's state space */
    uint32_t* state = (uint32_t*)alloca(f->num_vars * sizeof(uint32_t));
    memset(state, 0, f->num_vars * sizeof(uint32_t));

    uint32_t total_configs = f->cpt_size;
    for (uint32_t config = 0; config < total_configs; config++) {
        /* Decode config into per-variable states */
        uint32_t remaining = config;
        for (int32_t k = (int32_t)f->num_vars - 1; k >= 0; k--) {
            state[k] = remaining % f->dims[k];
            remaining /= f->dims[k];
        }

        /* Compute log-potential for this configuration */
        double log_val = f->cpt[config];

        /* Add log-messages from all other variables to this factor */
        for (uint32_t vi = 0; vi < f->num_vars; vi++) {
            if (vi == target_vi) continue;
            uint32_t msg_idx = get_msg_idx(fg, fi, vi);
            log_val += fg->msg_vf[msg_idx][state[vi]];
        }

        /* Accumulate into target state's message (log-sum-exp) */
        uint32_t target_state = state[target_vi];
        out_msg[target_state] = logsumexp2(out_msg[target_state], log_val);
    }

    /* Normalize the message */
    log_normalize(out_msg, target_dim);
}

/*
 * Compute variable-to-factor message.
 * msg_{v→f}(x_v) = prod_{g ∈ ne(v)\f} msg_{g→v}(x_v)
 * In log-space: sum of all incoming factor-to-variable messages except from f.
 */
static void compute_var_to_factor_message(
    const eshkol_factor_graph_t* fg,
    uint32_t target_fi, uint32_t target_vi,
    double* out_msg) {
    const eshkol_factor_t* target_f = fg->factors[target_fi];
    const uint32_t* target_var_indices = FACTOR_VAR_INDICES(target_f);
    uint32_t var_id = target_var_indices[target_vi];
    uint32_t dim = fg->var_dims[var_id];

    /* Initialize to zero (neutral element for log-product = sum) */
    for (uint32_t s = 0; s < dim; s++) {
        out_msg[s] = 0.0;
    }

    /* Sum log-messages from all other factors connected to this variable */
    for (uint32_t fi = 0; fi < fg->num_factors; fi++) {
        if (fi == target_fi) continue;
        const eshkol_factor_t* f = fg->factors[fi];
        const uint32_t* var_indices = FACTOR_VAR_INDICES(f);

        for (uint32_t vi = 0; vi < f->num_vars; vi++) {
            if (var_indices[vi] == var_id) {
                uint32_t msg_idx = get_msg_idx(fg, fi, vi);
                for (uint32_t s = 0; s < dim; s++) {
                    out_msg[s] += fg->msg_fv[msg_idx][s];
                }
            }
        }
    }

    log_normalize(out_msg, dim);
}

bool eshkol_fg_infer(arena_t* arena, eshkol_factor_graph_t* fg,
    uint32_t max_iterations, double tolerance) {
    if (!arena || !fg || fg->num_factors == 0) return false;

    /* Allocate message storage if needed */
    if (!allocate_messages(arena, fg)) return false;

    bool converged = false;

    for (uint32_t iter = 0; iter < max_iterations; iter++) {
        double max_delta = 0.0;

        /* Update all variable-to-factor messages */
        for (uint32_t fi = 0; fi < fg->num_factors; fi++) {
            const eshkol_factor_t* f = fg->factors[fi];
            for (uint32_t vi = 0; vi < f->num_vars; vi++) {
                uint32_t msg_idx = get_msg_idx(fg, fi, vi);
                uint32_t var_id = FACTOR_VAR_INDICES(f)[vi];
                uint32_t dim = fg->var_dims[var_id];

                double* old_msg = fg->msg_vf[msg_idx];
                double* new_msg = (double*)alloca(dim * sizeof(double));
                compute_var_to_factor_message(fg, fi, vi, new_msg);

                for (uint32_t s = 0; s < dim; s++) {
                    double delta = fabs(new_msg[s] - old_msg[s]);
                    if (delta > max_delta) max_delta = delta;
                    old_msg[s] = new_msg[s];
                }
            }
        }

        /* Update all factor-to-variable messages */
        for (uint32_t fi = 0; fi < fg->num_factors; fi++) {
            const eshkol_factor_t* f = fg->factors[fi];
            for (uint32_t vi = 0; vi < f->num_vars; vi++) {
                uint32_t msg_idx = get_msg_idx(fg, fi, vi);
                uint32_t var_id = FACTOR_VAR_INDICES(f)[vi];
                uint32_t dim = fg->var_dims[var_id];

                double* old_msg = fg->msg_fv[msg_idx];
                double* new_msg = (double*)alloca(dim * sizeof(double));
                compute_factor_to_var_message(fg, fi, vi, new_msg);

                for (uint32_t s = 0; s < dim; s++) {
                    double delta = fabs(new_msg[s] - old_msg[s]);
                    if (delta > max_delta) max_delta = delta;
                    old_msg[s] = new_msg[s];
                }
            }
        }

        /* Update beliefs: b(x_v) = prod_{f ∈ ne(v)} msg_{f→v}(x_v).
         * Skip observed variables — their beliefs are clamped by fg-observe!.
         * Must match the VM path (vm_inference.c:388) for AOT/VM consistency. */
        for (uint32_t v = 0; v < fg->num_vars; v++) {
            if (fg->observed && fg->observed[v]) continue;  /* Clamped — skip */
            uint32_t dim = fg->var_dims[v];
            for (uint32_t s = 0; s < dim; s++) {
                fg->beliefs[v][s] = 0.0; /* log-product = sum */
            }

            for (uint32_t fi = 0; fi < fg->num_factors; fi++) {
                const eshkol_factor_t* f = fg->factors[fi];
                const uint32_t* var_indices = FACTOR_VAR_INDICES(f);
                for (uint32_t vi = 0; vi < f->num_vars; vi++) {
                    if (var_indices[vi] == v) {
                        uint32_t msg_idx = get_msg_idx(fg, fi, vi);
                        for (uint32_t s = 0; s < dim; s++) {
                            fg->beliefs[v][s] += fg->msg_fv[msg_idx][s];
                        }
                    }
                }
            }

            log_normalize(fg->beliefs[v], dim);
        }

        if (max_delta < tolerance) {
            converged = true;
            break;
        }
    }

    return converged;
}

/* ===== Free Energy ===== */

double eshkol_free_energy(const eshkol_factor_graph_t* fg,
    const double* observations, uint32_t num_obs) {
    if (!fg) return 0.0;

    /*
     * Variational Free Energy: F = E_q[ln q(s)] - E_q[ln p(o,s)]
     *
     * Term 1: Negative entropy of approximate posterior
     *   H_q = -sum_s q(s) * ln q(s)  (per variable, assuming mean-field)
     *
     * Term 2: Expected log-joint under q
     *   E_q[ln p(o,s)] ≈ sum over factors: E_q[ln f(x)]
     */

    double entropy = 0.0;
    double expected_log_joint = 0.0;

    /* Entropy of beliefs (mean-field approximation: sum of marginal entropies) */
    for (uint32_t v = 0; v < fg->num_vars; v++) {
        uint32_t dim = fg->var_dims[v];
        for (uint32_t s = 0; s < dim; s++) {
            double q = exp(fg->beliefs[v][s]);
            if (q > 1e-30) {
                entropy -= q * fg->beliefs[v][s]; /* -q*log(q) */
            }
        }
    }

    /* Expected log-joint: sum over factors of E_q[log f(x)] */
    for (uint32_t fi = 0; fi < fg->num_factors; fi++) {
        const eshkol_factor_t* f = fg->factors[fi];
        const uint32_t* var_indices = FACTOR_VAR_INDICES(f);

        /* For each configuration, compute q(config) * log f(config) */
        uint32_t total_configs = f->cpt_size;
        for (uint32_t config = 0; config < total_configs; config++) {
            /* Decode config */
            uint32_t remaining = config;
            double log_q_config = 0.0;

            for (int32_t k = (int32_t)f->num_vars - 1; k >= 0; k--) {
                uint32_t state = remaining % f->dims[k];
                remaining /= f->dims[k];
                uint32_t var_id = var_indices[k];
                log_q_config += fg->beliefs[var_id][state];
            }

            double q_config = exp(log_q_config);
            if (q_config > 1e-30) {
                expected_log_joint += q_config * f->cpt[config];
            }
        }
    }

    /* Apply observation clamping if provided */
    /* observations is pairs: [var_index, observed_state, var_index, observed_state, ...] */
    if (observations && num_obs > 0) {
        /* Observation term: add log p(o|s) where observed variables are clamped */
        for (uint32_t i = 0; i + 1 < num_obs * 2; i += 2) {
            uint32_t var_idx = (uint32_t)observations[i];
            uint32_t obs_state = (uint32_t)observations[i + 1];
            if (var_idx < fg->num_vars && obs_state < fg->var_dims[var_idx]) {
                /* Surprise from observation: -ln q(o) */
                expected_log_joint += fg->beliefs[var_idx][obs_state];
            }
        }
    }

    /* F = -H(q) - E_q[ln p(o,s)] = E_q[ln q] - E_q[ln p(o,s)] */
    double free_energy = -entropy - expected_log_joint;

    return free_energy;
}

double eshkol_expected_free_energy(arena_t* arena,
    const eshkol_factor_graph_t* fg,
    uint32_t action_var, uint32_t action_state) {
    if (!arena || !fg) return 0.0;
    if (action_var >= fg->num_vars) return 0.0;
    if (action_state >= fg->var_dims[action_var]) return 0.0;

    /*
     * Expected Free Energy: G(a) = E_q(o,s|a)[ln q(s|a) - ln p(o,s)]
     *
     * Simplified for discrete factor graphs:
     * G(a) ≈ -H_q(o|a) + E_q[D_KL(q(s|o,a) || q(s|a))]
     *
     * Pragmatic value (goal-seeking): -E_q[ln p(o|a)]
     * Epistemic value (uncertainty reduction): mutual information I(o;s|a)
     *
     * We approximate by conditioning beliefs on action and computing
     * the resulting free energy.
     */

    double efe = 0.0;

    /* Pragmatic value: expected surprise under current model */
    /* For each variable connected to action_var, compute expected log-prob */
    for (uint32_t fi = 0; fi < fg->num_factors; fi++) {
        const eshkol_factor_t* f = fg->factors[fi];
        const uint32_t* var_indices = FACTOR_VAR_INDICES(f);

        /* Check if this factor involves the action variable */
        bool involves_action = false;
        uint32_t action_pos = 0;
        for (uint32_t vi = 0; vi < f->num_vars; vi++) {
            if (var_indices[vi] == action_var) {
                involves_action = true;
                action_pos = vi;
                break;
            }
        }

        if (!involves_action) continue;

        /* Marginalize CPT over action state */
        uint32_t total_configs = f->cpt_size;
        for (uint32_t config = 0; config < total_configs; config++) {
            /* Check if this config has the right action state */
            uint32_t remaining = config;
            uint32_t* state = (uint32_t*)alloca(f->num_vars * sizeof(uint32_t));
            for (int32_t k = (int32_t)f->num_vars - 1; k >= 0; k--) {
                state[k] = remaining % f->dims[k];
                remaining /= f->dims[k];
            }

            if (state[action_pos] != action_state) continue;

            /* Compute q(config | action) */
            double log_q = 0.0;
            for (uint32_t vi = 0; vi < f->num_vars; vi++) {
                if (var_indices[vi] == action_var) continue;
                uint32_t var_id = var_indices[vi];
                log_q += fg->beliefs[var_id][state[vi]];
            }

            double q = exp(log_q);
            if (q > 1e-30) {
                /* Pragmatic: -q * ln p(config) */
                efe -= q * f->cpt[config];
                /* Epistemic: q * ln q */
                efe += q * log_q;
            }
        }
    }

    return efe;
}

/* ===== Tagged Value Dispatch ===== */

void eshkol_make_factor_graph_tagged(arena_t* arena,
    const eshkol_tagged_value_t* num_vars_tv,
    const eshkol_tagged_value_t* var_dims_tv,
    eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    if (!arena || !num_vars_tv || !var_dims_tv) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Extract num_vars from integer */
    uint32_t num_vars = 0;
    if (num_vars_tv->type == ESHKOL_VALUE_INT64) {
        num_vars = (uint32_t)num_vars_tv->data.int_val;
    } else {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Extract var_dims from tensor */
    tensor_layout_t* dims_tensor = extract_tensor(var_dims_tv);
    if (!dims_tensor || dims_tensor->total_elements < num_vars) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Convert tensor doubles to uint32_t array */
    uint32_t* var_dims = (uint32_t*)alloca(num_vars * sizeof(uint32_t));
    for (uint32_t i = 0; i < num_vars; i++) {
        var_dims[i] = (uint32_t)tensor_get(dims_tensor, i);
    }

    eshkol_factor_graph_t* fg = eshkol_make_factor_graph(arena, num_vars, var_dims);
    if (fg) {
        result->type = ESHKOL_VALUE_HEAP_PTR;
        result->data.ptr_val = (uint64_t)fg;
    } else {
        result->type = ESHKOL_VALUE_NULL;
    }
}

void eshkol_fg_add_factor_tagged(arena_t* arena,
    const eshkol_tagged_value_t* fg_tv,
    const eshkol_tagged_value_t* var_indices_tv,
    const eshkol_tagged_value_t* cpt_tv,
    eshkol_tagged_value_t* result) {
    if (result) {
        memset(result, 0, sizeof(*result));
        result->type = ESHKOL_VALUE_NULL;
    }

    if (!arena || !fg_tv || !var_indices_tv || !cpt_tv) return;

    /* Extract factor graph */
    if (fg_tv->type != ESHKOL_VALUE_HEAP_PTR || !fg_tv->data.ptr_val) return;
    eshkol_factor_graph_t* fg = (eshkol_factor_graph_t*)fg_tv->data.ptr_val;

    /* Extract var_indices from tensor */
    tensor_layout_t* idx_tensor = extract_tensor(var_indices_tv);
    if (!idx_tensor) return;
    uint32_t num_vars = (uint32_t)idx_tensor->total_elements;

    uint32_t* var_indices = (uint32_t*)alloca(num_vars * sizeof(uint32_t));
    for (uint32_t i = 0; i < num_vars; i++) {
        var_indices[i] = (uint32_t)tensor_get(idx_tensor, i);
    }

    /* Extract CPT from tensor */
    tensor_layout_t* cpt_tensor = extract_tensor(cpt_tv);
    if (!cpt_tensor) return;

    /* Build dims array from the factor graph's var_dims */
    uint32_t* dims = (uint32_t*)alloca(num_vars * sizeof(uint32_t));
    for (uint32_t i = 0; i < num_vars; i++) {
        if (var_indices[i] < fg->num_vars) {
            dims[i] = fg->var_dims[var_indices[i]];
        } else {
            return; /* invalid variable index */
        }
    }

    /* Extract CPT data as doubles */
    uint32_t cpt_size = (uint32_t)cpt_tensor->total_elements;
    double* cpt_data = (double*)alloca(cpt_size * sizeof(double));
    for (uint32_t i = 0; i < cpt_size; i++) {
        cpt_data[i] = tensor_get(cpt_tensor, i);
    }

    eshkol_factor_t* factor = eshkol_make_factor(arena, var_indices, num_vars, cpt_data, dims);
    if (factor) {
        eshkol_fg_add_factor(arena, fg, factor);
    }
}

void eshkol_fg_infer_tagged(arena_t* arena,
    const eshkol_tagged_value_t* fg_tv,
    const eshkol_tagged_value_t* max_iters_tv,
    eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    if (!arena || !fg_tv) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    if (fg_tv->type != ESHKOL_VALUE_HEAP_PTR || !fg_tv->data.ptr_val) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    eshkol_factor_graph_t* fg = (eshkol_factor_graph_t*)fg_tv->data.ptr_val;

    uint32_t max_iters = 20; /* default */
    if (max_iters_tv && max_iters_tv->type == ESHKOL_VALUE_INT64) {
        max_iters = (uint32_t)max_iters_tv->data.int_val;
    }

    eshkol_fg_infer(arena, fg, max_iters, 1e-6);

    /* Return beliefs as a tensor: flat array of all belief values */
    /* Total size = sum of all var_dims */
    uint32_t total_beliefs = 0;
    for (uint32_t v = 0; v < fg->num_vars; v++) {
        total_beliefs += fg->var_dims[v];
    }

    /* Allocate tensor for beliefs */
    size_t tensor_size = sizeof(tensor_layout_t);
    tensor_layout_t* tensor = (tensor_layout_t*)arena_allocate_with_header(
        arena, tensor_size, HEAP_SUBTYPE_TENSOR, 0);
    if (!tensor) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    tensor->num_dimensions = 1;
    tensor->total_elements = total_beliefs;
    tensor->dimensions = (uint64_t*)arena_allocate_aligned(arena, sizeof(uint64_t), 8);
    if (tensor->dimensions) {
        tensor->dimensions[0] = total_beliefs;
    }
    tensor->elements = (int64_t*)arena_allocate_aligned(
        arena, total_beliefs * sizeof(int64_t), 8);
    if (!tensor->elements) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Copy beliefs (converting from log-space to probability space) */
    uint32_t idx = 0;
    for (uint32_t v = 0; v < fg->num_vars; v++) {
        for (uint32_t s = 0; s < fg->var_dims[v]; s++) {
            double prob = exp(fg->beliefs[v][s]);
            tensor_set(tensor, idx++, prob);
        }
    }

    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->data.ptr_val = (uint64_t)tensor;
}

void eshkol_free_energy_tagged(
    const eshkol_tagged_value_t* fg_tv,
    const eshkol_tagged_value_t* obs_tv,
    eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    if (!fg_tv || fg_tv->type != ESHKOL_VALUE_HEAP_PTR || !fg_tv->data.ptr_val) {
        result->type = ESHKOL_VALUE_DOUBLE;
        result->flags = 0x20; /* INEXACT */
        result->data.double_val = 0.0;
        return;
    }

    const eshkol_factor_graph_t* fg = (const eshkol_factor_graph_t*)fg_tv->data.ptr_val;

    /* Extract observations from tensor if provided */
    double* obs_data = NULL;
    uint32_t num_obs = 0;
    if (obs_tv) {
        tensor_layout_t* obs_tensor = extract_tensor(obs_tv);
        if (obs_tensor && obs_tensor->total_elements > 0) {
            num_obs = (uint32_t)(obs_tensor->total_elements / 2);
            obs_data = (double*)alloca(obs_tensor->total_elements * sizeof(double));
            for (uint32_t i = 0; i < obs_tensor->total_elements; i++) {
                obs_data[i] = tensor_get(obs_tensor, i);
            }
        }
    }

    double fe = eshkol_free_energy(fg, obs_data, num_obs);

    result->type = ESHKOL_VALUE_DOUBLE;
    result->flags = 0x20; /* INEXACT */
    result->data.double_val = fe;
}

void eshkol_efe_tagged(arena_t* arena,
    const eshkol_tagged_value_t* fg_tv,
    const eshkol_tagged_value_t* action_var_tv,
    const eshkol_tagged_value_t* action_state_tv,
    eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    if (!arena || !fg_tv ||
        fg_tv->type != ESHKOL_VALUE_HEAP_PTR || !fg_tv->data.ptr_val) {
        result->type = ESHKOL_VALUE_DOUBLE;
        result->flags = 0x20;
        result->data.double_val = 0.0;
        return;
    }

    const eshkol_factor_graph_t* fg = (const eshkol_factor_graph_t*)fg_tv->data.ptr_val;

    uint32_t action_var = 0;
    uint32_t action_state = 0;
    if (action_var_tv && action_var_tv->type == ESHKOL_VALUE_INT64) {
        action_var = (uint32_t)action_var_tv->data.int_val;
    }
    if (action_state_tv && action_state_tv->type == ESHKOL_VALUE_INT64) {
        action_state = (uint32_t)action_state_tv->data.int_val;
    }

    double efe = eshkol_expected_free_energy(arena, fg, action_var, action_state);

    result->type = ESHKOL_VALUE_DOUBLE;
    result->flags = 0x20; /* INEXACT */
    result->data.double_val = efe;
}

/* ===== CPT Update ===== */

void eshkol_fg_update_cpt_tagged(arena_t* arena,
    const eshkol_tagged_value_t* fg_tv,
    const eshkol_tagged_value_t* factor_idx_tv,
    const eshkol_tagged_value_t* new_cpt_tv,
    eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    if (!arena || !fg_tv || !factor_idx_tv || !new_cpt_tv) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Extract factor graph */
    if (fg_tv->type != ESHKOL_VALUE_HEAP_PTR || !fg_tv->data.ptr_val) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }
    eshkol_factor_graph_t* fg = (eshkol_factor_graph_t*)fg_tv->data.ptr_val;

    /* Extract factor index */
    uint32_t factor_idx = 0;
    if (factor_idx_tv->type == ESHKOL_VALUE_INT64) {
        factor_idx = (uint32_t)factor_idx_tv->data.int_val;
    } else if (factor_idx_tv->type == ESHKOL_VALUE_DOUBLE) {
        factor_idx = (uint32_t)factor_idx_tv->data.double_val;
    } else {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Bounds check */
    if (factor_idx >= fg->num_factors) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Extract new CPT from tensor */
    tensor_layout_t* cpt_tensor = extract_tensor(new_cpt_tv);
    if (!cpt_tensor) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    eshkol_factor_t* f = fg->factors[factor_idx];
    if (!f) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Validate CPT size matches */
    if ((uint32_t)cpt_tensor->total_elements != f->cpt_size) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Copy new CPT data */
    for (uint32_t i = 0; i < f->cpt_size; i++) {
        f->cpt[i] = tensor_get(cpt_tensor, i);
    }

    /* Reset messages to force reconvergence on next fg-infer! */
    fg->msg_fv = NULL;
    fg->msg_vf = NULL;

    /* Return the factor graph (mutated in place) */
    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->data.ptr_val = (uint64_t)fg;
}

/* ===== Display ===== */

void eshkol_display_factor_graph(const eshkol_factor_graph_t* fg, void* file) {
    FILE* f = file ? (FILE*)file : stdout;
    if (!fg) {
        fprintf(f, "#<factor-graph: empty>");
        return;
    }
    fprintf(f, "#<factor-graph: %u factors, %u vars>",
            fg->num_factors, fg->num_vars);
}

/**
 * @brief fg-observe! runtime for AOT compilation.
 * Clamps a factor graph variable to an observed state.
 * After calling, run fg-infer! to propagate the evidence.
 * Ref: Standard factor graph evidence clamping (Kschischang et al. 2001).
 */
extern "C" void eshkol_fg_observe_tagged(arena_t* arena,
    const eshkol_tagged_value_t* fg_tv,
    const eshkol_tagged_value_t* var_tv,
    const eshkol_tagged_value_t* state_tv,
    eshkol_tagged_value_t* result) {
    (void)arena;
    if (!result) return;
    memset(result, 0, sizeof(*result));
    result->type = ESHKOL_VALUE_BOOL;
    result->data.int_val = 0;  /* false by default */

    if (!fg_tv || !var_tv || !state_tv) return;
    if (fg_tv->type != ESHKOL_VALUE_HEAP_PTR || !fg_tv->data.ptr_val) return;

    eshkol_factor_graph_t* fg = (eshkol_factor_graph_t*)fg_tv->data.ptr_val;

    /* Extract var_id */
    int var_id = 0;
    if (var_tv->type == ESHKOL_VALUE_INT64) var_id = (int)var_tv->data.int_val;
    else if (var_tv->type == ESHKOL_VALUE_DOUBLE) var_id = (int)var_tv->data.double_val;
    else return;

    /* Extract observed_state */
    int obs_state = 0;
    if (state_tv->type == ESHKOL_VALUE_INT64) obs_state = (int)state_tv->data.int_val;
    else if (state_tv->type == ESHKOL_VALUE_DOUBLE) obs_state = (int)state_tv->data.double_val;
    else return;

    /* Bounds check */
    if (var_id < 0 || (uint32_t)var_id >= fg->num_vars) return;
    if (obs_state < 0 || obs_state >= (int)fg->var_dims[var_id]) return;

    /* Clamp beliefs: observed state → log(1)=0, others → log(0)≈-1e30 */
    int dim = fg->var_dims[var_id];
    for (int s = 0; s < dim; s++) {
        fg->beliefs[var_id][s] = (s == obs_state) ? 0.0 : -1e30;
    }

    /* Mark as observed (allocate observed array if needed) */
    if (!fg->observed) {
        fg->observed = (bool*)calloc(fg->num_vars, sizeof(bool));
    }
    if (fg->observed) {
        fg->observed[var_id] = true;
    }

    result->data.int_val = 1;  /* true — observation applied */
}
