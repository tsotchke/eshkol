/**
 * @file qllm_bridge_gradcheck_test.cpp
 * @brief End-to-end proof that the Eshkol <-> qLLM bridge is wired.
 *
 * The bridge has two halves. The backward half
 * (`lib/bridge/tensor_backward.cpp`, dispatched from
 * `lib/backend/tensor_backward.cpp`) shipped and compiled long before the
 * forward half existed, but it was unreachable: nothing in the tree ever
 * created an `AD_NODE_TENSOR_*` node for it to differentiate. This test
 * exercises the completed path in the only way that actually proves it — it
 * runs a real computation through the bridge's forward entry points, back-
 * propagates through the previously unreachable backward rules, and checks the
 * resulting gradients against a central finite-difference reference computed
 * through the same forward code.
 *
 * Method (the precedent is tests/backend/qllm_backward_gradcheck_test.c, the
 * SDNC weight-matrix gradient check from #335): aggregate L2 relative error
 *     ||num - ana||_2 / (||num||_2 + ||ana||_2)
 * over sampled input coordinates. Per-element ratios are avoided because they
 * blow up wherever the analytical gradient is ~0. Everything here is double
 * precision, so the finite-difference floor sits far below the 1e-6 bar the
 * SDNC gradient check already uses.
 *
 * Coverage: every bridge op that has an exact backward rule registered in
 * get_tensor_backward_fn — matmul, gelu, softmax, layernorm, rmsnorm, silu and
 * cross-entropy. Attention and embedding are deliberately excluded: their
 * backward rules raise an explicit unsupported-op error rather than return an
 * approximate gradient, which is the existing hard constraint in this area.
 *
 * Also covered: the float32 interop round-trip and the bridge lifecycle, which
 * are the other four symbols the header declared and nothing defined.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>

#include "eshkol/eshkol.h"
#include "eshkol/bridge/qllm_bridge.h"
#include "eshkol/backend/tensor_backward.h"

extern "C" {
    typedef struct arena arena_t;
    arena_t* get_global_arena(void);
    void* arena_allocate_zeroed(arena_t* arena, size_t size);
    ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity);
    ad_node_t* arena_allocate_ad_node(arena_t* arena);
}

#define TOLERANCE 1e-6   /* same bar as the SDNC weight-matrix gradient check */
#define STEP      1e-4   /* central-difference step (double regime) */

/* ---------------------------------------------------------------- helpers */

static double* arena_doubles(size_t n) {
    return (double*)arena_allocate_zeroed(get_global_arena(), n * sizeof(double));
}

/** @brief Build an input-variable node holding a copy of @p data. */
static ad_node_t* var_node(const double* data, const int64_t* shape, size_t ndim) {
    ad_node_t* n = arena_allocate_ad_node(get_global_arena());
    size_t count = 1;
    for (size_t i = 0; i < ndim; ++i) count *= (size_t)shape[i];
    double* buf = arena_doubles(count);
    std::memcpy(buf, data, count * sizeof(double));
    int64_t* sh = (int64_t*)arena_allocate_zeroed(get_global_arena(), ndim * sizeof(int64_t));
    std::memcpy(sh, shape, ndim * sizeof(int64_t));
    n->type = AD_NODE_VARIABLE;
    n->tensor_value = buf;
    n->shape = sh;
    n->ndim = ndim;
    return n;
}

/**
 * @brief Backpropagate a scalar loss node through the whole tape.
 *
 * Seeds the loss node's upstream gradient (both the scalar mirror, which
 * tensor_cross_entropy_backward reads, and the 1-element tensor gradient,
 * which its non-null guard requires) and then walks the tape in reverse
 * evaluation order, exactly as the runtime's reverse sweep does.
 */
static void backward(ad_tape_t* tape, ad_node_t* loss) {
    loss->gradient = 1.0;
    if (!loss->tensor_gradient) loss->tensor_gradient = arena_doubles(1);
    ((double*)loss->tensor_gradient)[0] = 1.0;
    for (size_t i = tape->num_nodes; i-- > 0;) {
        eshkol_tensor_backward_dispatch(tape->nodes[i]);
    }
}

/** @brief Aggregate L2 relative-error accumulator. */
struct RelAcc { double dn = 0.0, nn = 0.0, an = 0.0; };
static void relacc_add(RelAcc* a, double num, double ana) {
    double d = num - ana;
    a->dn += d * d; a->nn += num * num; a->an += ana * ana;
}
static double relacc_value(const RelAcc* a) {
    double den = std::sqrt(a->nn) + std::sqrt(a->an);
    return (den > 0.0) ? std::sqrt(a->dn) / den : 0.0;
}

static unsigned long g_rng = 0x9e3779b97f4a7c15UL;
static double urand(void) { /* deterministic uniform in [-1, 1] */
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return ((double)(g_rng >> 11) / (double)(1UL << 53)) * 2.0 - 1.0;
}

/* --------------------------------------------------------------- fixtures */

/* Shapes kept small so the finite-difference sweep stays cheap and exact. */
enum { ROWS = 2, KDIM = 4, COLS = 3 };

static double g_x[ROWS * KDIM];
static double g_w[KDIM * COLS];
static double g_t[ROWS * COLS];      /* cross-entropy targets (rows sum to 1) */
static double g_gamma[COLS];
static double g_beta[COLS];

static void init_fixtures(void) {
    for (int i = 0; i < ROWS * KDIM; ++i) g_x[i] = urand();
    for (int i = 0; i < KDIM * COLS; ++i) g_w[i] = urand();
    for (int i = 0; i < COLS; ++i) { g_gamma[i] = 1.0 + 0.3 * urand(); g_beta[i] = 0.2 * urand(); }
    for (int r = 0; r < ROWS; ++r) {
        double s = 0.0;
        for (int c = 0; c < COLS; ++c) { g_t[r * COLS + c] = 0.25 + 0.5 * (urand() + 1.0) * 0.5; s += g_t[r * COLS + c]; }
        for (int c = 0; c < COLS; ++c) g_t[r * COLS + c] /= s;
    }
}

/* Which pipeline a check runs. Each ends in cross-entropy so the loss is a
 * scalar and the finite-difference reference is well defined. */
enum Pipeline { P_MATMUL_GELU, P_LAYERNORM, P_RMSNORM, P_SILU, P_SOFTMAX };

/**
 * @brief Run one pipeline. With @p tape non-null the ops are recorded;
 *        with null only the forward runs, which is what the finite-difference
 *        reference uses. Returns the scalar loss; optionally hands back the
 *        input nodes so the caller can read their gradients.
 */
static double run_pipeline(Pipeline p, ad_tape_t* tape,
                           const double* x, const double* w,
                           ad_node_t** out_x, ad_node_t** out_w,
                           ad_node_t** out_gamma) {
    const int64_t sh_x[2]   = { ROWS, KDIM };
    const int64_t sh_w[2]   = { KDIM, COLS };
    const int64_t sh_out[2] = { ROWS, COLS };
    const int64_t sh_g[1]   = { COLS };

    ad_node_t* tgt = var_node(g_t, sh_out, 2);
    ad_node_t* h = nullptr;
    ad_node_t* xn = nullptr;
    ad_node_t* wn = nullptr;
    ad_node_t* gn = nullptr;

    if (p == P_MATMUL_GELU) {
        xn = var_node(x, sh_x, 2);
        wn = var_node(w, sh_w, 2);
        ad_node_t* mm = ad_tensor_matmul(tape, xn, wn);
        if (!mm) return NAN;
        h = ad_tensor_gelu(tape, mm);
    } else {
        /* The remaining ops are elementwise / row-wise: drive them directly
         * with a [ROWS, COLS] input so no matmul is in the path. */
        xn = var_node(x, sh_out, 2);
        switch (p) {
            case P_LAYERNORM: {
                gn = var_node(g_gamma, sh_g, 1);
                ad_node_t* bn = var_node(g_beta, sh_g, 1);
                h = ad_tensor_layernorm(tape, xn, gn, bn, 1e-5);
                break;
            }
            case P_RMSNORM:
                gn = var_node(g_gamma, sh_g, 1);
                h = ad_tensor_rmsnorm(tape, xn, gn, 1e-5);
                break;
            case P_SILU:    h = ad_tensor_silu(tape, xn); break;
            case P_SOFTMAX: h = ad_tensor_softmax(tape, xn, -1); break;
            default: break;
        }
    }
    if (!h) return NAN;

    ad_node_t* loss = ad_tensor_cross_entropy(tape, h, tgt);
    if (!loss) return NAN;

    if (out_x) *out_x = xn;
    if (out_w) *out_w = wn;
    if (out_gamma) *out_gamma = gn;
    return ((double*)loss->tensor_value)[0];
}

/** @brief Number of input elements the given pipeline's x carries. */
static size_t x_len(Pipeline p) {
    return (p == P_MATMUL_GELU) ? (size_t)(ROWS * KDIM) : (size_t)(ROWS * COLS);
}

/**
 * @brief Gradient-check one pipeline: analytical (bridge forward + bridge
 *        backward) vs. central finite differences of the same forward.
 */
static double check(Pipeline p, const char* name) {
    size_t nx = x_len(p);
    double x[ROWS * KDIM];
    std::memcpy(x, g_x, nx * sizeof(double));

    /* Analytical pass. */
    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 64);
    ad_node_t *xn = nullptr, *wn = nullptr, *gn = nullptr;
    double L = run_pipeline(p, tape, x, g_w, &xn, &wn, &gn);
    if (std::isnan(L)) {
        std::printf("  %-22s FORWARD REFUSED\n", name);
        return 1.0;
    }
    backward(tape, tape->nodes[tape->num_nodes - 1]);

    const double* dx = xn ? (const double*)xn->tensor_gradient : nullptr;
    if (!dx) {
        std::printf("  %-22s NO GRADIENT REACHED THE INPUT\n", name);
        return 1.0;
    }

    RelAcc acc;
    for (size_t i = 0; i < nx; ++i) {
        double save = x[i];
        x[i] = save + STEP;
        double Lp = run_pipeline(p, nullptr, x, g_w, nullptr, nullptr, nullptr);
        x[i] = save - STEP;
        double Lm = run_pipeline(p, nullptr, x, g_w, nullptr, nullptr, nullptr);
        x[i] = save;
        relacc_add(&acc, (Lp - Lm) / (2.0 * STEP), dx[i]);
    }

    /* Also check the weight gradient where the pipeline has weights. */
    if (p == P_MATMUL_GELU && wn && wn->tensor_gradient) {
        const double* dw = (const double*)wn->tensor_gradient;
        double w[KDIM * COLS];
        std::memcpy(w, g_w, sizeof(w));
        for (size_t i = 0; i < (size_t)(KDIM * COLS); ++i) {
            double save = w[i];
            w[i] = save + STEP;
            double Lp = run_pipeline(p, nullptr, x, w, nullptr, nullptr, nullptr);
            w[i] = save - STEP;
            double Lm = run_pipeline(p, nullptr, x, w, nullptr, nullptr, nullptr);
            w[i] = save;
            relacc_add(&acc, (Lp - Lm) / (2.0 * STEP), dw[i]);
        }
    }

    double err = relacc_value(&acc);
    std::printf("  %-22s L2 rel err = %.3e  (tol %.0e)  %s   [loss %.10f]\n",
                name, err, (double)TOLERANCE, err < TOLERANCE ? "PASS" : "FAIL", L);
    return err;
}

/* ------------------------------------------------------ conversion + init */

static bool check_conversion(void) {
    const size_t shape[2] = { 2, 3 };
    const double src[6] = { -1.5, 0.25, 3.75, 1e-3, -2.5, 0.0 };

    qllm_tensor_t* t = eshkol_to_qllm_tensor(src, shape, 2);
    if (!t) { std::printf("  conversion              eshkol_to_qllm_tensor returned NULL\n"); return false; }

    double back[6] = { 0 };
    size_t n = 0;
    bool ok = qllm_to_eshkol_tensor(t, back, &n);
    bool exact = ok && n == 6;
    double worst = 0.0;
    for (size_t i = 0; i < 6 && exact; ++i) {
        double d = std::fabs(back[i] - src[i]);
        if (d > worst) worst = d;
    }
    /* Values chosen to be exactly representable in float32 except 1e-3, which
     * must round-trip within float32 epsilon. */
    exact = exact && worst < 1e-9;
    std::free(t);   /* single contiguous allocation: one free() releases it */

    std::printf("  %-22s round-trip max abs err = %.3e  %s\n",
                "float32 conversion", worst, exact ? "PASS" : "FAIL");
    return exact;
}

static bool check_lifecycle(void) {
    bool ok = true;
    if (eshkol_qllm_bridge_ready()) { std::printf("    ready() true before init\n"); ok = false; }
    /* No qLLM runtime is present in a default build; init must report that
     * honestly rather than claim success. */
    if (eshkol_qllm_bridge_init("/nonexistent/libsemiclassical_qllm.dylib")) {
        std::printf("    init() claimed success for a missing library\n"); ok = false;
    }
    if (eshkol_qllm_bridge_ready()) { std::printf("    ready() true after failed init\n"); ok = false; }
    eshkol_qllm_bridge_shutdown();   /* must be a safe no-op */
    if (eshkol_qllm_bridge_ready()) { std::printf("    ready() true after shutdown\n"); ok = false; }

    std::printf("  %-22s init/ready/shutdown report honestly  %s\n",
                "bridge lifecycle", ok ? "PASS" : "FAIL");
    return ok;
}

/* ------------------------------------------------------------------- main */

int main(void) {
    std::printf("=== qLLM bridge end-to-end check ===\n");
    init_fixtures();

    int passed = 0, failed = 0;

    struct { Pipeline p; const char* name; } cases[] = {
        { P_MATMUL_GELU, "matmul+gelu+xent" },
        { P_LAYERNORM,   "layernorm+xent"   },
        { P_RMSNORM,     "rmsnorm+xent"     },
        { P_SILU,        "silu+xent"        },
        { P_SOFTMAX,     "softmax+xent"     },
    };
    for (auto& c : cases) {
        if (check(c.p, c.name) < TOLERANCE) ++passed; else ++failed;
    }

    if (check_conversion()) ++passed; else ++failed;
    if (check_lifecycle())  ++passed; else ++failed;

    std::printf("=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
