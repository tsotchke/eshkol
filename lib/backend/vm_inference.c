/**
 * @file vm_inference.c
 * @brief Probabilistic inference for the Eshkol bytecode VM consciousness engine.
 *
 * Implements core probabilistic inference primitives:
 *   - Factor graphs with tensor-based CPTs
 *   - Loopy belief propagation (sum-product algorithm in log-space)
 *   - Variational free energy computation
 *   - Expected free energy for action selection
 *
 * Ported from inference.h / inference.cpp (C++ w/ Eshkol arena) to pure C
 * using VmRegionStack / VmArena from vm_arena.h.
 *
 * Native call IDs: 520-539
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"   /* pulls in vm_arena.h + subtypes */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

/* ========================================================================
 * Log-space arithmetic
 * ======================================================================== */

static const double LOG_ZERO = -1e30; /* Approximate -infinity */

/* log-sum-exp of two values: log(exp(a) + exp(b)) */
static double logsumexp2(double a, double b) {
    if (a <= LOG_ZERO) return b;
    if (b <= LOG_ZERO) return a;
    double m = (a > b) ? a : b;
    return m + log(1.0 + exp(-(fabs(a - b))));
}

/* log-sum-exp over array */
static double logsumexp(const double* arr, int n) {
    if (n == 0) return LOG_ZERO;
    double m = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > m) m = arr[i];
    }
    if (m <= LOG_ZERO) return LOG_ZERO;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        s += exp(arr[i] - m);
    }
    return m + log(s);
}

/* Normalize log-probability vector in-place (so exp sums to 1) */
static void log_normalize(double* arr, int n) {
    double z = logsumexp(arr, n);
    if (z > LOG_ZERO) {
        for (int i = 0; i < n; i++) {
            arr[i] -= z;
        }
    }
}

/* ========================================================================
 * Factor
 * ======================================================================== */

typedef struct {
    int      num_vars;
    int*     var_indices;    /* which FG variables this factor connects */
    int*     dims;           /* state count per variable in this factor */
    double*  cpt;            /* conditional probability table (log-space) */
    int      cpt_size;       /* product of dims */
} VmFactor;

/* ========================================================================
 * Factor Graph
 * ======================================================================== */

typedef struct {
    int        num_vars;
    int        num_factors;
    int        max_factors;
    int*       var_dims;     /* states per variable */
    double**   beliefs;      /* beliefs[var][state] in log-space */
    VmFactor** factors;
    double**   msg_fv;       /* factor -> variable messages */
    double**   msg_vf;       /* variable -> factor messages */
    int        total_msgs;   /* total message count (sum of factor->num_vars) */
    bool*      observed;     /* observed[var] = true → clamped, skip during BP (Phase 5a) */
} VmFactorGraph;

/* ========================================================================
 * Construction
 * ======================================================================== */

/* 520: make-factor-graph */
static VmFactorGraph* vm_make_factor_graph(VmRegionStack* rs,
    int num_vars, const int* var_dims)
{
    if (!rs || num_vars <= 0 || !var_dims) return NULL;

    VmFactorGraph* fg = (VmFactorGraph*)vm_alloc_object(rs,
        VM_SUBTYPE_FG, sizeof(VmFactorGraph));
    if (!fg) return NULL;

    fg->num_vars    = num_vars;
    fg->num_factors = 0;
    fg->max_factors = 16;
    fg->total_msgs  = 0;

    /* Variable dimensions */
    fg->var_dims = (int*)vm_alloc(rs, (size_t)num_vars * sizeof(int));
    if (!fg->var_dims) return NULL;
    memcpy(fg->var_dims, var_dims, (size_t)num_vars * sizeof(int));

    /* Beliefs — one log-prob vector per variable, initialized to uniform */
    fg->beliefs = (double**)vm_alloc(rs, (size_t)num_vars * sizeof(double*));
    if (!fg->beliefs) return NULL;

    for (int i = 0; i < num_vars; i++) {
        fg->beliefs[i] = (double*)vm_alloc(rs, (size_t)var_dims[i] * sizeof(double));
        if (!fg->beliefs[i]) return NULL;
        double log_uniform = -log((double)var_dims[i]);
        for (int j = 0; j < var_dims[i]; j++) {
            fg->beliefs[i][j] = log_uniform;
        }
    }

    /* Factor pointer array */
    fg->factors = (VmFactor**)vm_alloc(rs, (size_t)fg->max_factors * sizeof(VmFactor*));
    if (!fg->factors) return NULL;

    /* Messages allocated lazily during inference */
    fg->msg_fv = NULL;
    fg->msg_vf = NULL;

    return fg;
}

/* 521: make-factor */
static VmFactor* vm_make_factor(VmRegionStack* rs,
    const int* var_indices, int num_vars,
    const double* cpt_data, const int* dims)
{
    if (!rs || !var_indices || !cpt_data || !dims || num_vars <= 0) return NULL;

    VmFactor* f = (VmFactor*)vm_alloc(rs, sizeof(VmFactor));
    if (!f) return NULL;

    f->num_vars = num_vars;

    /* Compute CPT size = product of dimensions */
    int cpt_size = 1;
    for (int i = 0; i < num_vars; i++) cpt_size *= dims[i];
    f->cpt_size = cpt_size;

    /* Allocate and copy var_indices */
    f->var_indices = (int*)vm_alloc(rs, (size_t)num_vars * sizeof(int));
    if (!f->var_indices) return NULL;
    memcpy(f->var_indices, var_indices, (size_t)num_vars * sizeof(int));

    /* Allocate and copy dims */
    f->dims = (int*)vm_alloc(rs, (size_t)num_vars * sizeof(int));
    if (!f->dims) return NULL;
    memcpy(f->dims, dims, (size_t)num_vars * sizeof(int));

    /* Allocate and copy CPT */
    f->cpt = (double*)vm_alloc(rs, (size_t)cpt_size * sizeof(double));
    if (!f->cpt) return NULL;
    memcpy(f->cpt, cpt_data, (size_t)cpt_size * sizeof(double));

    return f;
}

/* 522: fg-add-factor! */
static void vm_fg_add_factor(VmRegionStack* rs, VmFactorGraph* fg, VmFactor* factor) {
    if (!rs || !fg || !factor) return;

    /* Grow factor array if needed */
    if (fg->num_factors >= fg->max_factors) {
        int new_max = fg->max_factors * 2;
        VmFactor** new_arr = (VmFactor**)vm_alloc(rs, (size_t)new_max * sizeof(VmFactor*));
        if (!new_arr) return;
        memcpy(new_arr, fg->factors, (size_t)fg->num_factors * sizeof(VmFactor*));
        fg->factors     = new_arr;
        fg->max_factors = new_max;
    }

    fg->factors[fg->num_factors++] = factor;
    fg->total_msgs += factor->num_vars;

    /* Invalidate cached messages so they're re-allocated */
    fg->msg_fv = NULL;
    fg->msg_vf = NULL;
}

/* ========================================================================
 * Message allocation for BP
 * ======================================================================== */

static int allocate_messages(VmRegionStack* rs, VmFactorGraph* fg) {
    if (fg->msg_fv && fg->msg_vf) return 1; /* already allocated */

    int total = 0;
    for (int fi = 0; fi < fg->num_factors; fi++) {
        total += fg->factors[fi]->num_vars;
    }

    fg->msg_fv = (double**)vm_alloc(rs, (size_t)total * sizeof(double*));
    fg->msg_vf = (double**)vm_alloc(rs, (size_t)total * sizeof(double*));
    if (!fg->msg_fv || !fg->msg_vf) return 0;

    int idx = 0;
    for (int fi = 0; fi < fg->num_factors; fi++) {
        VmFactor* f = fg->factors[fi];
        for (int vi = 0; vi < f->num_vars; vi++) {
            int var_id = f->var_indices[vi];
            int dim = fg->var_dims[var_id];

            fg->msg_fv[idx] = (double*)vm_alloc(rs, (size_t)dim * sizeof(double));
            fg->msg_vf[idx] = (double*)vm_alloc(rs, (size_t)dim * sizeof(double));
            if (!fg->msg_fv[idx] || !fg->msg_vf[idx]) return 0;

            /* Initialize to uniform */
            double log_uni = -log((double)dim);
            for (int s = 0; s < dim; s++) {
                fg->msg_fv[idx][s] = log_uni;
                fg->msg_vf[idx][s] = log_uni;
            }
            idx++;
        }
    }

    return 1;
}

/* Get flat message index for factor fi, local variable position vi */
static int get_msg_idx(const VmFactorGraph* fg, int fi, int vi) {
    int idx = 0;
    for (int i = 0; i < fi; i++) idx += fg->factors[i]->num_vars;
    return idx + vi;
}

/* ========================================================================
 * Belief Propagation — Sum-Product in Log-Space
 * ======================================================================== */

/*
 * Factor-to-variable message:
 * msg_{f->v}(x_v) = sum_{x_\v} [ f(x) * prod_{u in ne(f)\v} msg_{u->f}(x_u) ]
 * In log-space: log-sum-exp over configs of non-target variables.
 */
static void compute_f2v_message(const VmFactorGraph* fg, int fi, int target_vi,
    double* out_msg)
{
    const VmFactor* f = fg->factors[fi];
    int target_var = f->var_indices[target_vi];
    int target_dim = fg->var_dims[target_var];

    /* Initialize output */
    for (int s = 0; s < target_dim; s++) out_msg[s] = LOG_ZERO;

    /* Enumerate all CPT configurations */
    int total_configs = f->cpt_size;

    /* We need per-variable states. Use stack allocation (safe: num_vars small). */
    int state[32]; /* max 32 vars per factor */
    if (f->num_vars > 32) return; /* safety */

    for (int config = 0; config < total_configs; config++) {
        /* Decode config into per-variable states (big-endian: var 0 is most significant) */
        int remaining = config;
        for (int k = f->num_vars - 1; k >= 0; k--) {
            state[k] = remaining % f->dims[k];
            remaining /= f->dims[k];
        }

        /* log-potential = CPT entry */
        double log_val = f->cpt[config];

        /* Add log-messages from all OTHER variables to this factor */
        for (int vi = 0; vi < f->num_vars; vi++) {
            if (vi == target_vi) continue;
            int msg_idx = get_msg_idx(fg, fi, vi);
            log_val += fg->msg_vf[msg_idx][state[vi]];
        }

        /* Accumulate into target state's slot via log-sum-exp */
        int ts = state[target_vi];
        out_msg[ts] = logsumexp2(out_msg[ts], log_val);
    }

    log_normalize(out_msg, target_dim);
}

/*
 * Variable-to-factor message:
 * msg_{v->f}(x_v) = prod_{g in ne(v)\f} msg_{g->v}(x_v)
 * In log-space: sum of all incoming f->v messages except from target factor.
 */
static void compute_v2f_message(const VmFactorGraph* fg, int target_fi, int target_vi,
    double* out_msg)
{
    const VmFactor* tf = fg->factors[target_fi];
    int var_id = tf->var_indices[target_vi];
    int dim = fg->var_dims[var_id];

    /* Initialize to 0 (neutral for log-sum = product) */
    for (int s = 0; s < dim; s++) out_msg[s] = 0.0;

    /* Sum messages from all OTHER factors connected to var_id */
    for (int fi = 0; fi < fg->num_factors; fi++) {
        if (fi == target_fi) continue;
        VmFactor* f = fg->factors[fi];
        for (int vi = 0; vi < f->num_vars; vi++) {
            if (f->var_indices[vi] == var_id) {
                int msg_idx = get_msg_idx(fg, fi, vi);
                for (int s = 0; s < dim; s++) {
                    out_msg[s] += fg->msg_fv[msg_idx][s];
                }
            }
        }
    }

    log_normalize(out_msg, dim);
}

/*
 * 523: fg-infer! — Loopy Belief Propagation
 * Returns 1 if converged, 0 otherwise.
 */
static int vm_fg_infer(VmRegionStack* rs, VmFactorGraph* fg,
    int max_iterations, double tolerance)
{
    if (!rs || !fg || fg->num_factors == 0) return 0;
    if (!allocate_messages(rs, fg)) return 0;

    int converged = 0;
    double new_msg[256]; /* max 256 states per variable — stack allocated */

    for (int iter = 0; iter < max_iterations; iter++) {
        double max_delta = 0.0;

        /* Update all variable-to-factor messages */
        for (int fi = 0; fi < fg->num_factors; fi++) {
            VmFactor* f = fg->factors[fi];
            for (int vi = 0; vi < f->num_vars; vi++) {
                int msg_idx = get_msg_idx(fg, fi, vi);
                int var_id = f->var_indices[vi];
                int dim = fg->var_dims[var_id];
                if (dim > 256) continue; /* safety */

                compute_v2f_message(fg, fi, vi, new_msg);

                double* old = fg->msg_vf[msg_idx];
                for (int s = 0; s < dim; s++) {
                    double d = fabs(new_msg[s] - old[s]);
                    if (d > max_delta) max_delta = d;
                    old[s] = new_msg[s];
                }
            }
        }

        /* Update all factor-to-variable messages */
        for (int fi = 0; fi < fg->num_factors; fi++) {
            VmFactor* f = fg->factors[fi];
            for (int vi = 0; vi < f->num_vars; vi++) {
                int msg_idx = get_msg_idx(fg, fi, vi);
                int var_id = f->var_indices[vi];
                int dim = fg->var_dims[var_id];
                if (dim > 256) continue; /* safety */

                compute_f2v_message(fg, fi, vi, new_msg);

                double* old = fg->msg_fv[msg_idx];
                for (int s = 0; s < dim; s++) {
                    double d = fabs(new_msg[s] - old[s]);
                    if (d > max_delta) max_delta = d;
                    old[s] = new_msg[s];
                }
            }
        }

        /* Update beliefs: b(x_v) = product of all incoming f->v messages.
         * Skip observed variables — their beliefs are clamped by fg-observe!. */
        for (int v = 0; v < fg->num_vars; v++) {
            if (fg->observed && fg->observed[v]) continue;  /* Clamped — skip */
            int dim = fg->var_dims[v];
            for (int s = 0; s < dim; s++) fg->beliefs[v][s] = 0.0;

            for (int fi = 0; fi < fg->num_factors; fi++) {
                VmFactor* f = fg->factors[fi];
                for (int vi = 0; vi < f->num_vars; vi++) {
                    if (f->var_indices[vi] == v) {
                        int msg_idx = get_msg_idx(fg, fi, vi);
                        for (int s = 0; s < dim; s++) {
                            fg->beliefs[v][s] += fg->msg_fv[msg_idx][s];
                        }
                    }
                }
            }

            log_normalize(fg->beliefs[v], dim);
        }

        if (max_delta < tolerance) {
            converged = 1;
            break;
        }
    }

    return converged;
}

/* ========================================================================
 * Free Energy
 *
 * F = E_q[ln q(s)] - E_q[ln p(o,s)]
 *   = -H(q) - E_q[ln p(o,s)]
 * ======================================================================== */

/*
 * 524: free-energy
 * observations: pairs of (var_index, observed_state) as flat double array.
 * num_obs: number of observation pairs (array has 2*num_obs elements).
 */
static double vm_free_energy(const VmFactorGraph* fg,
    const double* observations, int num_obs)
{
    if (!fg) return 0.0;

    double entropy = 0.0;
    double expected_log_joint = 0.0;

    /* Entropy of beliefs (mean-field: sum of marginal entropies) */
    for (int v = 0; v < fg->num_vars; v++) {
        int dim = fg->var_dims[v];
        for (int s = 0; s < dim; s++) {
            double q = exp(fg->beliefs[v][s]);
            if (q > 1e-30) {
                entropy -= q * fg->beliefs[v][s]; /* -q * log(q) */
            }
        }
    }

    /* Expected log-joint: sum over factors of E_q[log f(x)] */
    for (int fi = 0; fi < fg->num_factors; fi++) {
        const VmFactor* f = fg->factors[fi];
        int total_configs = f->cpt_size;

        int state[32];
        if (f->num_vars > 32) continue;

        for (int config = 0; config < total_configs; config++) {
            int remaining = config;
            double log_q_config = 0.0;

            for (int k = f->num_vars - 1; k >= 0; k--) {
                state[k] = remaining % f->dims[k];
                remaining /= f->dims[k];
                int var_id = f->var_indices[k];
                log_q_config += fg->beliefs[var_id][state[k]];
            }

            double q_config = exp(log_q_config);
            if (q_config > 1e-30) {
                expected_log_joint += q_config * f->cpt[config];
            }
        }
    }

    /* Observation clamping */
    if (observations && num_obs > 0) {
        for (int i = 0; i < num_obs; i++) {
            int var_idx   = (int)observations[i * 2];
            int obs_state = (int)observations[i * 2 + 1];
            if (var_idx >= 0 && var_idx < fg->num_vars &&
                obs_state >= 0 && obs_state < fg->var_dims[var_idx])
            {
                expected_log_joint += fg->beliefs[var_idx][obs_state];
            }
        }
    }

    /* F = -H(q) - E_q[ln p(o,s)] */
    return -entropy - expected_log_joint;
}

/* ========================================================================
 * Expected Free Energy
 *
 * G(a) = -E_q[ln p(o|a)] + E_q[ln q(s|a)]
 *       = pragmatic value + epistemic value
 * ======================================================================== */

/*
 * 525: expected-free-energy
 * Evaluates how good a particular action is by decomposing into:
 *   pragmatic: how well action achieves goals (-E_q[ln p(o|a)])
 *   epistemic: how much action reduces uncertainty (mutual information)
 */
static double vm_expected_free_energy(VmRegionStack* rs,
    const VmFactorGraph* fg, int action_var, int action_state)
{
    (void)rs; /* not used for allocation in this simplified version */
    if (!fg) return 0.0;
    if (action_var < 0 || action_var >= fg->num_vars) return 0.0;
    if (action_state < 0 || action_state >= fg->var_dims[action_var]) return 0.0;

    double efe = 0.0;

    for (int fi = 0; fi < fg->num_factors; fi++) {
        const VmFactor* f = fg->factors[fi];

        /* Check if this factor involves the action variable */
        int involves_action = 0;
        int action_pos = 0;
        for (int vi = 0; vi < f->num_vars; vi++) {
            if (f->var_indices[vi] == action_var) {
                involves_action = 1;
                action_pos = vi;
                break;
            }
        }
        if (!involves_action) continue;

        int total_configs = f->cpt_size;
        int state[32];
        if (f->num_vars > 32) continue;

        for (int config = 0; config < total_configs; config++) {
            /* Decode config */
            int remaining = config;
            for (int k = f->num_vars - 1; k >= 0; k--) {
                state[k] = remaining % f->dims[k];
                remaining /= f->dims[k];
            }

            /* Only configs where action variable = action_state */
            if (state[action_pos] != action_state) continue;

            /* Compute q(config | action) — beliefs of non-action vars */
            double log_q = 0.0;
            for (int vi = 0; vi < f->num_vars; vi++) {
                if (f->var_indices[vi] == action_var) continue;
                int var_id = f->var_indices[vi];
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

/* ========================================================================
 * CPT Update (for learning)
 * ======================================================================== */

/* 526: fg-update-cpt! — replace a factor's CPT and reset messages */
static int vm_fg_update_cpt(VmFactorGraph* fg, int factor_idx,
    const double* new_cpt, int cpt_size)
{
    if (!fg || factor_idx < 0 || factor_idx >= fg->num_factors) return 0;
    VmFactor* f = fg->factors[factor_idx];
    if (!f || f->cpt_size != cpt_size) return 0;
    memcpy(f->cpt, new_cpt, (size_t)cpt_size * sizeof(double));
    /* Reset messages to force reconvergence */
    fg->msg_fv = NULL;
    fg->msg_vf = NULL;
    return 1;
}

/* ========================================================================
 * Display
 * ======================================================================== */

static void vm_display_factor_graph(const VmFactorGraph* fg) {
    if (!fg) { printf("#<factor-graph: empty>"); return; }
    printf("#<factor-graph: %d factors, %d vars>", fg->num_factors, fg->num_vars);
}

static void vm_display_beliefs(const VmFactorGraph* fg) {
    if (!fg) return;
    for (int v = 0; v < fg->num_vars; v++) {
        printf("  var %d: [", v);
        for (int s = 0; s < fg->var_dims[v]; s++) {
            if (s > 0) printf(", ");
            printf("%.4f", exp(fg->beliefs[v][s]));
        }
        printf("]\n");
    }
}

/* ========================================================================
 * Self-tests
 * ======================================================================== */

#ifdef VM_INFERENCE_TEST
#include <assert.h>

static void check_beliefs_sum_to_one(const VmFactorGraph* fg, double tol) {
    for (int v = 0; v < fg->num_vars; v++) {
        double sum = 0.0;
        for (int s = 0; s < fg->var_dims[v]; s++) {
            sum += exp(fg->beliefs[v][s]);
        }
        if (fabs(sum - 1.0) > tol) {
            printf("  FAIL: var %d beliefs sum to %.6f (expected 1.0)\n", v, sum);
            assert(0);
        }
    }
}

int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    printf("=== vm_inference self-tests ===\n");

    /* --- Test 1: Factor graph construction --- */
    {
        int dims[2] = { 2, 3 };
        VmFactorGraph* fg = vm_make_factor_graph(&rs, 2, dims);
        assert(fg != NULL);
        assert(fg->num_vars == 2);
        assert(fg->num_factors == 0);
        assert(fg->var_dims[0] == 2);
        assert(fg->var_dims[1] == 3);

        /* Beliefs should be uniform */
        double expected0 = -log(2.0);
        double expected1 = -log(3.0);
        assert(fabs(fg->beliefs[0][0] - expected0) < 1e-10);
        assert(fabs(fg->beliefs[0][1] - expected0) < 1e-10);
        assert(fabs(fg->beliefs[1][0] - expected1) < 1e-10);

        printf("  [PASS] factor graph construction, uniform beliefs\n");
    }

    /* --- Test 2: Factor creation and addition --- */
    {
        int dims[2] = { 2, 2 };
        VmFactorGraph* fg = vm_make_factor_graph(&rs, 2, dims);

        /* Create a factor connecting vars 0 and 1 with a CPT.
         * CPT layout: [p(0,0), p(0,1), p(1,0), p(1,1)] in log-space
         * This represents p(X,Y) — biased towards X=Y */
        double cpt[4] = {
            log(0.4), log(0.1),   /* X=0: Y=0 likely, Y=1 unlikely */
            log(0.1), log(0.4)    /* X=1: Y=0 unlikely, Y=1 likely */
        };
        int var_indices[2] = { 0, 1 };
        int factor_dims[2] = { 2, 2 };

        VmFactor* f = vm_make_factor(&rs, var_indices, 2, cpt, factor_dims);
        assert(f != NULL);
        assert(f->num_vars == 2);
        assert(f->cpt_size == 4);

        vm_fg_add_factor(&rs, fg, f);
        assert(fg->num_factors == 1);

        printf("  [PASS] factor creation and addition\n");
    }

    /* --- Test 3: BP convergence on simple 2-var graph --- */
    {
        int dims[2] = { 2, 2 };
        VmFactorGraph* fg = vm_make_factor_graph(&rs, 2, dims);

        /* Joint: p(X=0,Y=0)=0.4, p(X=0,Y=1)=0.1, p(X=1,Y=0)=0.1, p(X=1,Y=1)=0.4 */
        double cpt[4] = { log(0.4), log(0.1), log(0.1), log(0.4) };
        int vi[2] = { 0, 1 };
        int fd[2] = { 2, 2 };

        VmFactor* f = vm_make_factor(&rs, vi, 2, cpt, fd);
        vm_fg_add_factor(&rs, fg, f);

        int converged = vm_fg_infer(&rs, fg, 50, 1e-8);
        assert(converged);

        /* Check beliefs sum to 1 */
        check_beliefs_sum_to_one(fg, 1e-6);

        /* Marginals: p(X=0)=0.5, p(X=1)=0.5 (symmetric joint) */
        double p0 = exp(fg->beliefs[0][0]);
        double p1 = exp(fg->beliefs[0][1]);
        assert(fabs(p0 - 0.5) < 0.05);
        assert(fabs(p1 - 0.5) < 0.05);

        printf("  [PASS] BP converges, beliefs sum to 1, marginals correct\n");
    }

    /* --- Test 4: BP with asymmetric CPT --- */
    {
        int dims[2] = { 2, 2 };
        VmFactorGraph* fg = vm_make_factor_graph(&rs, 2, dims);

        /* Asymmetric: p(X=0,Y=0)=0.7, p(X=0,Y=1)=0.1, p(X=1,Y=0)=0.1, p(X=1,Y=1)=0.1 */
        double cpt[4] = { log(0.7), log(0.1), log(0.1), log(0.1) };
        int vi[2] = { 0, 1 };
        int fd[2] = { 2, 2 };

        VmFactor* f = vm_make_factor(&rs, vi, 2, cpt, fd);
        vm_fg_add_factor(&rs, fg, f);

        int converged = vm_fg_infer(&rs, fg, 50, 1e-8);
        assert(converged);
        check_beliefs_sum_to_one(fg, 1e-6);

        /* p(X=0) = 0.7+0.1 = 0.8, p(X=1) = 0.1+0.1 = 0.2 */
        double px0 = exp(fg->beliefs[0][0]);
        double px1 = exp(fg->beliefs[0][1]);
        assert(fabs(px0 - 0.8) < 0.05);
        assert(fabs(px1 - 0.2) < 0.05);

        /* p(Y=0) = 0.7+0.1 = 0.8, p(Y=1) = 0.1+0.1 = 0.2 */
        double py0 = exp(fg->beliefs[1][0]);
        double py1 = exp(fg->beliefs[1][1]);
        assert(fabs(py0 - 0.8) < 0.05);
        assert(fabs(py1 - 0.2) < 0.05);

        printf("  [PASS] BP with asymmetric CPT, correct marginals\n");
    }

    /* --- Test 5: BP converges in < 10 iterations on simple graph --- */
    {
        int dims[2] = { 3, 3 };
        VmFactorGraph* fg = vm_make_factor_graph(&rs, 2, dims);

        /* 3x3 CPT: identity-like (diagonal entries high) */
        double cpt[9];
        for (int i = 0; i < 9; i++) {
            int r = i / 3, c = i % 3;
            cpt[i] = (r == c) ? log(0.3) : log(0.0333);
        }
        /* Normalize: sum = 3*0.3 + 6*0.0333 ≈ 1.1 — close enough for log-space */
        int vi[2] = { 0, 1 };
        int fd[2] = { 3, 3 };

        VmFactor* f = vm_make_factor(&rs, vi, 2, cpt, fd);
        vm_fg_add_factor(&rs, fg, f);

        /* Run with max 10 iterations */
        int converged = vm_fg_infer(&rs, fg, 10, 1e-6);
        assert(converged);
        check_beliefs_sum_to_one(fg, 1e-6);

        printf("  [PASS] BP converges in <= 10 iterations (3x3 graph)\n");
    }

    /* --- Test 6: Free energy is finite and negative --- */
    {
        int dims[2] = { 2, 2 };
        VmFactorGraph* fg = vm_make_factor_graph(&rs, 2, dims);

        double cpt[4] = { log(0.4), log(0.1), log(0.1), log(0.4) };
        int vi[2] = { 0, 1 };
        int fd[2] = { 2, 2 };

        VmFactor* f = vm_make_factor(&rs, vi, 2, cpt, fd);
        vm_fg_add_factor(&rs, fg, f);
        vm_fg_infer(&rs, fg, 50, 1e-8);

        /* No observations */
        double fe = vm_free_energy(fg, NULL, 0);
        assert(isfinite(fe));

        /* With observations: var 0 in state 0 */
        double obs[2] = { 0.0, 0.0 };
        double fe_obs = vm_free_energy(fg, obs, 1);
        assert(isfinite(fe_obs));

        printf("  [PASS] free energy finite (F=%.4f, F_obs=%.4f)\n", fe, fe_obs);
    }

    /* --- Test 7: Expected free energy --- */
    {
        int dims[3] = { 2, 2, 2 };
        VmFactorGraph* fg = vm_make_factor_graph(&rs, 3, dims);

        /* Factor: action (var 2) -> observation (var 0) */
        double cpt1[4] = {
            log(0.9), log(0.1),  /* action=0: obs=0 likely */
            log(0.1), log(0.9)   /* action=1: obs=1 likely */
        };
        int vi1[2] = { 0, 2 };
        int fd1[2] = { 2, 2 };
        vm_fg_add_factor(&rs, fg, vm_make_factor(&rs, vi1, 2, cpt1, fd1));

        /* Factor: action (var 2) -> state (var 1) */
        double cpt2[4] = {
            log(0.8), log(0.2),
            log(0.2), log(0.8)
        };
        int vi2[2] = { 1, 2 };
        int fd2[2] = { 2, 2 };
        vm_fg_add_factor(&rs, fg, vm_make_factor(&rs, vi2, 2, cpt2, fd2));

        vm_fg_infer(&rs, fg, 50, 1e-8);

        double efe0 = vm_expected_free_energy(&rs, fg, 2, 0);
        double efe1 = vm_expected_free_energy(&rs, fg, 2, 1);
        assert(isfinite(efe0));
        assert(isfinite(efe1));

        printf("  [PASS] EFE: action0=%.4f, action1=%.4f\n", efe0, efe1);
    }

    /* --- Test 8: CPT update and reconvergence --- */
    {
        int dims[2] = { 2, 2 };
        VmFactorGraph* fg = vm_make_factor_graph(&rs, 2, dims);

        double cpt_initial[4] = { log(0.4), log(0.1), log(0.1), log(0.4) };
        int vi[2] = { 0, 1 };
        int fd[2] = { 2, 2 };

        VmFactor* f = vm_make_factor(&rs, vi, 2, cpt_initial, fd);
        vm_fg_add_factor(&rs, fg, f);
        vm_fg_infer(&rs, fg, 50, 1e-8);

        double p0_before = exp(fg->beliefs[0][0]);

        /* Update CPT to heavily bias X=0 */
        double cpt_new[4] = { log(0.9), log(0.05), log(0.025), log(0.025) };
        int ok = vm_fg_update_cpt(fg, 0, cpt_new, 4);
        assert(ok);

        /* Re-run inference */
        vm_fg_infer(&rs, fg, 50, 1e-8);
        check_beliefs_sum_to_one(fg, 1e-6);

        double p0_after = exp(fg->beliefs[0][0]);
        /* After update, X=0 should be much more likely */
        assert(p0_after > p0_before);

        printf("  [PASS] CPT update + reconvergence (p0: %.2f -> %.2f)\n",
               p0_before, p0_after);
    }

    /* --- Test 9: 3-variable chain graph --- */
    {
        int dims[3] = { 2, 2, 2 };
        VmFactorGraph* fg = vm_make_factor_graph(&rs, 3, dims);

        /* Factor 0-1 */
        double cpt01[4] = { log(0.45), log(0.05), log(0.05), log(0.45) };
        int vi01[2] = { 0, 1 };
        int fd01[2] = { 2, 2 };
        vm_fg_add_factor(&rs, fg, vm_make_factor(&rs, vi01, 2, cpt01, fd01));

        /* Factor 1-2 */
        double cpt12[4] = { log(0.45), log(0.05), log(0.05), log(0.45) };
        int vi12[2] = { 1, 2 };
        int fd12[2] = { 2, 2 };
        vm_fg_add_factor(&rs, fg, vm_make_factor(&rs, vi12, 2, cpt12, fd12));

        int converged = vm_fg_infer(&rs, fg, 50, 1e-8);
        assert(converged);
        check_beliefs_sum_to_one(fg, 1e-6);

        /* All marginals should be approximately uniform due to symmetry */
        for (int v = 0; v < 3; v++) {
            double p = exp(fg->beliefs[v][0]);
            assert(fabs(p - 0.5) < 0.1);
        }

        printf("  [PASS] 3-variable chain, BP converges, symmetric beliefs\n");
    }

    /* --- Test 10: log-sum-exp correctness --- */
    {
        /* logsumexp2 */
        double r1 = logsumexp2(log(2.0), log(3.0));
        assert(fabs(exp(r1) - 5.0) < 1e-10);

        /* logsumexp2 with LOG_ZERO */
        double r2 = logsumexp2(LOG_ZERO, log(5.0));
        assert(fabs(exp(r2) - 5.0) < 1e-10);

        /* logsumexp array */
        double arr[3] = { log(1.0), log(2.0), log(3.0) };
        double r3 = logsumexp(arr, 3);
        assert(fabs(exp(r3) - 6.0) < 1e-10);

        /* log_normalize */
        double lp[3] = { log(2.0), log(3.0), log(5.0) };
        log_normalize(lp, 3);
        double sum = exp(lp[0]) + exp(lp[1]) + exp(lp[2]);
        assert(fabs(sum - 1.0) < 1e-10);
        assert(fabs(exp(lp[0]) - 0.2) < 1e-10);

        printf("  [PASS] log-sum-exp and log_normalize\n");
    }

    vm_region_stack_destroy(&rs);
    printf("vm_inference: ALL TESTS PASSED\n");
    return 0;
}
#endif /* VM_INFERENCE_TEST */
