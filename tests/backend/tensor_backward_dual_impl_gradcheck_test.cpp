/**
 * @file tensor_backward_dual_impl_gradcheck_test.cpp
 * @brief SW-12: direct differential check between the TWO independent
 *        backward implementations for matmul / layernorm / transpose / sum /
 *        embedding / attention.
 *
 * `lib/backend/tensor_backward.cpp` defines `eshkol_backward_{matmul,
 * layernorm,transpose,sum,embedding,attention}` — raw-buffer kernels called
 * from the AD_NODE_{MATMUL,LAYERNORM,TRANSPOSE,SUM,EMBEDDING,ATTENTION}
 * dispatch arms (the "native" tensor-op path). `lib/bridge/tensor_backward.cpp`
 * defines `tensor_{matmul,layernorm,transpose,sum,embedding,attention}_backward`
 * — node-based rules called from the AD_NODE_TENSOR_{MATMUL,LAYERNORM,
 * TRANSPOSE,SUM,EMBEDDING,ATTENTION} arms (the qLLM-bridge path). Nothing in
 * the tree ever compared the two on identical inputs before this file; each
 * was only ever checked against finite differences of its OWN forward, which
 * cannot catch a bug both implementations happen to share (e.g. a
 * convention that is internally self-consistent but wrong), and cannot catch
 * the two disagreeing with each other while each looks locally fine.
 *
 * Method, per op: build IDENTICAL inputs and an identical upstream cotangent,
 * run both implementations, and diff their output gradients directly. Central
 * finite differences of an independently-written (third) forward function
 * serve as the tie-breaking oracle per the task's ruling: "finite differences
 * arbitrate; fix the wrong one".
 *
 *   matmul, layernorm, transpose, sum:  both sides have exact closed-form
 *     rules; this file asserts they agree with each other AND with FD.
 *   embedding:  already covered end-to-end (including a bridge-vs-native
 *     comparison) by tensor_embedding_backward_gradcheck_test.cpp; not
 *     duplicated here.
 *   attention:  the NATIVE rule (eshkol_backward_attention) is exact and is
 *     FD-validated here. The BRIDGE rule (tensor_attention_backward) is an
 *     unconditional refusal by design (see the comment at its definition) —
 *     its forward (ad_tensor_attention, lib/bridge/qllm_bridge.cpp) does not
 *     even retain the `causal` flag or the softmax weights the backward would
 *     need, so there is no "wrong value" to arbitrate: it is a genuine
 *     forward-plumbing gap, not a numeric disagreement. This file asserts the
 *     CURRENT contract (native exact + FD-checked, bridge loudly refuses)
 *     instead of silently excluding the op, and files the fix as
 *     NEEDS-DECISION (see the block comment at attention_check() below and
 *     the ledger entry).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "eshkol/eshkol.h"
#include "eshkol/backend/tensor_backward.h"
#include "eshkol/bridge/qllm_bridge.h"

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#define ESHKOL_HAVE_FORK_DEATH_TESTS 1
#endif

extern "C" {
    typedef struct arena arena_t;
    arena_t* get_global_arena(void);
    void* arena_allocate_zeroed(arena_t* arena, size_t size);
    ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity);
    ad_node_t* arena_allocate_ad_node(arena_t* arena);
}

namespace {

int g_passed = 0;
int g_failed = 0;

void report(const char* name, bool ok, const char* detail = nullptr) {
    std::printf("  %-46s %s", name, ok ? "PASS" : "FAIL");
    if (detail) std::printf("   [%s]", detail);
    std::printf("\n");
    if (ok) ++g_passed; else ++g_failed;
}

unsigned long g_rng = 0xD1B54A32D192ED03UL;
double urand() { /* deterministic uniform in [-1, 1] */
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return ((double)(g_rng >> 11) / (double)(1UL << 53)) * 2.0 - 1.0;
}

double max_abs_diff(const double* a, const double* b, size_t n) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

double* zeros(size_t n) {
    return (double*)arena_allocate_zeroed(get_global_arena(), n * sizeof(double));
}

/** @brief Build a leaf AD node wrapping a copy of @p data, shape @p shape. */
ad_node_t* var_node(const double* data, const int64_t* shape, size_t ndim) {
    ad_node_t* n = arena_allocate_ad_node(get_global_arena());
    size_t count = 1;
    for (size_t i = 0; i < ndim; ++i) count *= (size_t)shape[i];
    double* buf = zeros(count);
    std::memcpy(buf, data, count * sizeof(double));
    int64_t* sh = (int64_t*)arena_allocate_zeroed(get_global_arena(), ndim * sizeof(int64_t));
    std::memcpy(sh, shape, ndim * sizeof(int64_t));
    n->type = AD_NODE_VARIABLE;
    n->tensor_value = buf;
    n->shape = sh;
    n->ndim = ndim;
    return n;
}

/*******************************************************************************
 * MATMUL — Y[M,N] = X[M,K] @ W[K,N]
 ******************************************************************************/

void matmul_forward_ref(const double* X, const double* W, double* Y,
                        int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++) {
            double s = 0.0;
            for (int64_t k = 0; k < K; k++) s += X[i * K + k] * W[k * N + j];
            Y[i * N + j] = s;
        }
}

double matmul_probe(const double* X, const double* W, const double* cotan,
                    int64_t M, int64_t K, int64_t N) {
    std::vector<double> Y((size_t)(M * N));
    matmul_forward_ref(X, W, Y.data(), M, K, N);
    double L = 0.0;
    for (int64_t i = 0; i < M * N; i++) L += cotan[i] * Y[i];
    return L;
}

void matmul_check() {
    constexpr int64_t M = 3, K = 4, N = 2;
    double X[M * K], W[K * N], cotan[M * N];
    for (auto& v : X) v = urand();
    for (auto& v : W) v = urand();
    for (auto& v : cotan) v = urand();

    /* -- native -- */
    double dX_native[M * K] = {0}, dW_native[K * N] = {0};
    eshkol_backward_matmul(cotan, X, W, dX_native, dW_native, M, K, N);

    /* -- bridge -- */
    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 8);
    int64_t shX[2] = {M, K}, shW[2] = {K, N};
    ad_node_t* xn = var_node(X, shX, 2);
    ad_node_t* wn = var_node(W, shW, 2);
    ad_node_t* node = ad_tensor_matmul(tape, xn, wn);
    double* cotan_buf = zeros((size_t)(M * N));
    std::memcpy(cotan_buf, cotan, sizeof(cotan));
    node->tensor_gradient = cotan_buf;
    eshkol_tensor_backward_dispatch(node);
    const double* dX_bridge = (const double*)xn->tensor_gradient;
    const double* dW_bridge = (const double*)wn->tensor_gradient;

    /* -- direct native-vs-bridge -- */
    double d_dX = max_abs_diff(dX_native, dX_bridge, M * K);
    double d_dW = max_abs_diff(dW_native, dW_bridge, K * N);
    char detail[128];
    std::snprintf(detail, sizeof detail, "max|dX diff|=%.3e max|dW diff|=%.3e", d_dX, d_dW);
    report("matmul: native vs bridge", d_dX < 1e-9 && d_dW < 1e-9, detail);

    /* -- finite-difference oracle (against native; bridge already agrees) -- */
    const double h = 1e-6;
    std::vector<double> fdX((size_t)(M * K)), fdW((size_t)(K * N));
    for (int64_t i = 0; i < M * K; i++) {
        double save = X[i];
        X[i] = save + h; double Lp = matmul_probe(X, W, cotan, M, K, N);
        X[i] = save - h; double Lm = matmul_probe(X, W, cotan, M, K, N);
        X[i] = save;
        fdX[i] = (Lp - Lm) / (2.0 * h);
    }
    for (int64_t i = 0; i < K * N; i++) {
        double save = W[i];
        W[i] = save + h; double Lp = matmul_probe(X, W, cotan, M, K, N);
        W[i] = save - h; double Lm = matmul_probe(X, W, cotan, M, K, N);
        W[i] = save;
        fdW[i] = (Lp - Lm) / (2.0 * h);
    }
    double fd_dX = max_abs_diff(dX_native, fdX.data(), M * K);
    double fd_dW = max_abs_diff(dW_native, fdW.data(), K * N);
    std::snprintf(detail, sizeof detail, "max|dX-fd|=%.3e max|dW-fd|=%.3e", fd_dX, fd_dW);
    report("matmul: native vs finite differences", fd_dX < 1e-6 && fd_dW < 1e-6, detail);
}

/*******************************************************************************
 * LAYERNORM — per-row y = gamma*(x-mean)/sqrt(var+eps)   (beta omitted: the
 * bridge rule computes no dbeta at all, so there is nothing to compare there)
 ******************************************************************************/

void layernorm_forward_ref(const double* X, const double* G, double* Y,
                           int64_t rows, int64_t width, double eps) {
    for (int64_t r = 0; r < rows; r++) {
        const double* xr = &X[r * width];
        double* yr = &Y[r * width];
        double mean = 0.0;
        for (int64_t i = 0; i < width; i++) mean += xr[i];
        mean /= (double)width;
        double var = 0.0;
        for (int64_t i = 0; i < width; i++) { double d = xr[i] - mean; var += d * d; }
        var /= (double)width;
        double inv = 1.0 / std::sqrt(var + eps);
        for (int64_t i = 0; i < width; i++) yr[i] = (xr[i] - mean) * inv * G[i];
    }
}

double layernorm_probe(const double* X, const double* G, const double* cotan,
                       int64_t rows, int64_t width, double eps) {
    std::vector<double> Y((size_t)(rows * width));
    layernorm_forward_ref(X, G, Y.data(), rows, width, eps);
    double L = 0.0;
    for (int64_t i = 0; i < rows * width; i++) L += cotan[i] * Y[i];
    return L;
}

void layernorm_check() {
    constexpr int64_t rows = 3, width = 4;
    const double eps = 1e-5;
    double X[rows * width], G[width], cotan[rows * width];
    for (auto& v : X) v = urand();
    for (auto& v : G) v = 1.0 + 0.3 * urand();
    for (auto& v : cotan) v = urand();

    /* -- native -- */
    double saved_mean[rows], saved_inv_std[rows];
    for (int64_t r = 0; r < rows; r++) {
        const double* xr = &X[r * width];
        double mean = 0.0;
        for (int64_t i = 0; i < width; i++) mean += xr[i];
        mean /= (double)width;
        double var = 0.0;
        for (int64_t i = 0; i < width; i++) { double d = xr[i] - mean; var += d * d; }
        var /= (double)width;
        saved_mean[r] = mean;
        saved_inv_std[r] = 1.0 / std::sqrt(var + eps);
    }
    double dX_native[rows * width] = {0}, dG_native[width] = {0}, dBeta_scratch[width] = {0};
    eshkol_backward_layernorm(cotan, X, saved_mean, saved_inv_std, G,
                              dX_native, dG_native, dBeta_scratch, rows, width);

    /* -- bridge -- */
    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 8);
    int64_t shX[2] = {rows, width}, shG[1] = {width};
    ad_node_t* xn = var_node(X, shX, 2);
    ad_node_t* gn = var_node(G, shG, 1);
    ad_node_t* node = ad_tensor_layernorm(tape, xn, gn, nullptr, eps);
    double* cotan_buf = zeros((size_t)(rows * width));
    std::memcpy(cotan_buf, cotan, sizeof(cotan));
    node->tensor_gradient = cotan_buf;
    eshkol_tensor_backward_dispatch(node);
    const double* dX_bridge = (const double*)xn->tensor_gradient;
    const double* dG_bridge = (const double*)gn->tensor_gradient;

    double d_dX = max_abs_diff(dX_native, dX_bridge, rows * width);
    double d_dG = max_abs_diff(dG_native, dG_bridge, width);
    char detail[128];
    std::snprintf(detail, sizeof detail, "max|dX diff|=%.3e max|dGamma diff|=%.3e", d_dX, d_dG);
    report("layernorm: native vs bridge", d_dX < 1e-9 && d_dG < 1e-9, detail);

    const double h = 1e-6;
    std::vector<double> fdX((size_t)(rows * width)), fdG((size_t)width);
    for (int64_t i = 0; i < rows * width; i++) {
        double save = X[i];
        X[i] = save + h; double Lp = layernorm_probe(X, G, cotan, rows, width, eps);
        X[i] = save - h; double Lm = layernorm_probe(X, G, cotan, rows, width, eps);
        X[i] = save;
        fdX[i] = (Lp - Lm) / (2.0 * h);
    }
    for (int64_t i = 0; i < width; i++) {
        double save = G[i];
        G[i] = save + h; double Lp = layernorm_probe(X, G, cotan, rows, width, eps);
        G[i] = save - h; double Lm = layernorm_probe(X, G, cotan, rows, width, eps);
        G[i] = save;
        fdG[i] = (Lp - Lm) / (2.0 * h);
    }
    double fd_dX = max_abs_diff(dX_native, fdX.data(), rows * width);
    double fd_dG = max_abs_diff(dG_native, fdG.data(), width);
    std::snprintf(detail, sizeof detail, "max|dX-fd|=%.3e max|dGamma-fd|=%.3e", fd_dX, fd_dG);
    report("layernorm: native vs finite differences", fd_dX < 1e-6 && fd_dG < 1e-6, detail);
}

/*******************************************************************************
 * TRANSPOSE — Y[rows,cols] = X[cols,rows]^T, i.e. Y[i,j] = X[j,i].
 * AD_NODE_TENSOR_TRANSPOSE has no forward producer anywhere in the tree (only
 * the enum + backward dispatch entry exist), so the bridge side is exercised
 * by hand-building the node exactly per tensor_transpose_backward's contract
 * (same technique tests/backend/tensor_embedding_backward_gradcheck_test.cpp
 * uses for AD_NODE_TENSOR_EMBEDDING).
 ******************************************************************************/

void transpose_forward_ref(const double* X, double* Y, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; i++)
        for (int64_t j = 0; j < cols; j++)
            Y[i * cols + j] = X[j * rows + i];
}

double transpose_probe(const double* X, const double* cotan, int64_t rows, int64_t cols) {
    std::vector<double> Y((size_t)(rows * cols));
    transpose_forward_ref(X, Y.data(), rows, cols);
    double L = 0.0;
    for (int64_t i = 0; i < rows * cols; i++) L += cotan[i] * Y[i];
    return L;
}

void transpose_check() {
    constexpr int64_t rows = 3, cols = 2;   /* X: (cols,rows), Y: (rows,cols) */
    double X[cols * rows], cotan[rows * cols];
    for (auto& v : X) v = urand();
    for (auto& v : cotan) v = urand();

    /* -- native -- */
    double dX_native[cols * rows] = {0};
    eshkol_backward_transpose(cotan, dX_native, rows, cols);

    /* -- bridge: hand-built AD_NODE_TENSOR_TRANSPOSE (no producer exists) -- */
    ad_node_t in{};
    ad_node_t node{};
    node.type = AD_NODE_TENSOR_TRANSPOSE;
    node.input1 = &in;
    int64_t out_shape[2] = {rows, cols};
    node.shape = out_shape;
    node.ndim = 2;
    double* cotan_buf = zeros((size_t)(rows * cols));
    std::memcpy(cotan_buf, cotan, sizeof(cotan));
    node.tensor_gradient = cotan_buf;
    eshkol_tensor_backward_dispatch(&node);
    const double* dX_bridge = (const double*)in.tensor_gradient;

    double d_dX = dX_bridge ? max_abs_diff(dX_native, dX_bridge, cols * rows) : -1.0;
    char detail[96];
    std::snprintf(detail, sizeof detail, "max|dX diff|=%.3e", d_dX);
    report("transpose: native vs bridge", dX_bridge && d_dX < 1e-9, detail);

    const double h = 1e-6;
    std::vector<double> fdX((size_t)(cols * rows));
    for (int64_t i = 0; i < cols * rows; i++) {
        double save = X[i];
        X[i] = save + h; double Lp = transpose_probe(X, cotan, rows, cols);
        X[i] = save - h; double Lm = transpose_probe(X, cotan, rows, cols);
        X[i] = save;
        fdX[i] = (Lp - Lm) / (2.0 * h);
    }
    double fd_dX = max_abs_diff(dX_native, fdX.data(), cols * rows);
    std::snprintf(detail, sizeof detail, "max|dX-fd|=%.3e", fd_dX);
    report("transpose: native vs finite differences", fd_dX < 1e-6, detail);
}

/*******************************************************************************
 * SUM — full reduction to a scalar: Y = sum(X). AD_NODE_TENSOR_SUM also has
 * no forward producer; hand-built per tensor_sum_backward's contract.
 ******************************************************************************/

double sum_forward_ref(const double* X, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += X[i];
    return s;
}

void sum_check() {
    constexpr int64_t n = 5;
    double X[n];
    for (auto& v : X) v = urand();
    double cotan = urand();   /* upstream gradient of the scalar output */

    /* -- native -- */
    double dX_native[n] = {0};
    eshkol_backward_sum(cotan, dX_native, n);

    /* -- bridge -- */
    ad_node_t in{};
    int64_t in_shape[1] = {n};
    in.shape = in_shape;
    in.ndim = 1;
    ad_node_t node{};
    node.type = AD_NODE_TENSOR_SUM;
    node.input1 = &in;
    double cotan_buf[1] = {cotan};
    node.tensor_gradient = cotan_buf;
    eshkol_tensor_backward_dispatch(&node);
    const double* dX_bridge = (const double*)in.tensor_gradient;

    double d_dX = dX_bridge ? max_abs_diff(dX_native, dX_bridge, n) : -1.0;
    char detail[96];
    std::snprintf(detail, sizeof detail, "max|dX diff|=%.3e", d_dX);
    report("sum: native vs bridge", dX_bridge && d_dX < 1e-9, detail);

    const double h = 1e-6;
    std::vector<double> fdX((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        double save = X[i];
        X[i] = save + h; double Lp = cotan * sum_forward_ref(X, n);
        X[i] = save - h; double Lm = cotan * sum_forward_ref(X, n);
        X[i] = save;
        fdX[i] = (Lp - Lm) / (2.0 * h);
    }
    double fd_dX = max_abs_diff(dX_native, fdX.data(), n);
    std::snprintf(detail, sizeof detail, "max|dX-fd|=%.3e", fd_dX);
    report("sum: native vs finite differences", fd_dX < 1e-6, detail);
}

/*******************************************************************************
 * ATTENTION — single-head scaled dot-product attention.
 *
 * NATIVE (eshkol_backward_attention) is exact; FD-validated below.
 *
 * BRIDGE (tensor_attention_backward) unconditionally calls eshkol_fatal() —
 * see the comment at its definition in lib/bridge/tensor_backward.cpp. This
 * is not a numeric disagreement to arbitrate: ad_tensor_attention's forward
 * (lib/bridge/qllm_bridge.cpp) never retains the `causal` flag or the
 * softmax attention weights on the node, so an exact adjoint cannot be
 * computed from what the node carries even in principle — the gap is
 * upstream, in the forward's contract, not a bug in a formula. Implementing
 * it exactly needs a forward-side change (thread `causal` through
 * AD_NODE_TENSOR_ATTENTION's params, and either retain the attention-weight
 * matrix or recompute it in the backward), which is a real feature addition
 * beyond a differential-test fix.
 *
 * NEEDS-DECISION (recorded for the ledger): should the bridge attention
 * backward be implemented now (real work: extend the node contract +
 * mirror the native 5-step chain rule already proven exact below), or is the
 * existing hard refusal (loud, not silent) an acceptable placeholder because
 * `scaled-dot-attention` already provides an exact, differentiable,
 * scalar-decomposed attention path elsewhere in the tree? This file asserts
 * the CURRENT contract (native exact, bridge refuses) so a future change to
 * either side is caught, and the ledger records the finding without
 * guessing which of "implement it" or "leave it refusing" the maintainer
 * prefers.
 ******************************************************************************/

void attention_forward_ref(const double* Q, const double* K, const double* V,
                           double* attn, double* out,
                           int64_t seq_q, int64_t seq_k, int64_t d_k, int64_t d_v,
                           double scale) {
    for (int64_t i = 0; i < seq_q; i++) {
        double mx = -1e300;
        std::vector<double> row(seq_k);
        for (int64_t j = 0; j < seq_k; j++) {
            double dot = 0.0;
            for (int64_t d = 0; d < d_k; d++) dot += Q[i * d_k + d] * K[j * d_k + d];
            row[j] = dot * scale;
            if (row[j] > mx) mx = row[j];
        }
        double sum = 0.0;
        for (int64_t j = 0; j < seq_k; j++) { row[j] = std::exp(row[j] - mx); sum += row[j]; }
        for (int64_t j = 0; j < seq_k; j++) attn[i * seq_k + j] = row[j] / sum;
        for (int64_t d = 0; d < d_v; d++) {
            double acc = 0.0;
            for (int64_t j = 0; j < seq_k; j++) acc += attn[i * seq_k + j] * V[j * d_v + d];
            out[i * d_v + d] = acc;
        }
    }
}

double attention_probe(const double* Q, const double* K, const double* V,
                       const double* cotan,
                       int64_t seq_q, int64_t seq_k, int64_t d_k, int64_t d_v, double scale) {
    std::vector<double> attn((size_t)(seq_q * seq_k)), out((size_t)(seq_q * d_v));
    attention_forward_ref(Q, K, V, attn.data(), out.data(), seq_q, seq_k, d_k, d_v, scale);
    double L = 0.0;
    for (int64_t i = 0; i < seq_q * d_v; i++) L += cotan[i] * out[i];
    return L;
}

#if defined(ESHKOL_HAVE_FORK_DEATH_TESTS)
/** @brief Run @p body in a forked child; true iff it terminated abnormally
 *  (signal) or exited nonzero — i.e. it "refused" rather than returning a
 *  value. Mirrors tests/backend/tensor_embedding_backward_gradcheck_test.cpp. */
bool refuses(void (*body)()) {
    std::fflush(stdout);
    std::fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        FILE* devnull = std::freopen("/dev/null", "w", stderr);
        (void)devnull;
        body();
        std::_Exit(0);   /* reached only if the call did NOT refuse */
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return false;
    if (WIFSIGNALED(status)) return true;
    return WIFEXITED(status) && WEXITSTATUS(status) != 0;
}

constexpr int64_t kAttnSeqQ = 2, kAttnSeqK = 3, kAttnDK = 4, kAttnDV = 3;
double g_attn_Q[kAttnSeqQ * kAttnDK];
double g_attn_K[kAttnSeqK * kAttnDK];
double g_attn_V[kAttnSeqK * kAttnDV];
double g_attn_cotan[kAttnSeqQ * kAttnDV];

void body_bridge_attention_backward() {
    int64_t shQ[2] = {kAttnSeqQ, kAttnDK}, shK[2] = {kAttnSeqK, kAttnDK}, shV[2] = {kAttnSeqK, kAttnDV};
    ad_node_t qn{}; qn.type = AD_NODE_VARIABLE; qn.tensor_value = g_attn_Q; qn.shape = shQ; qn.ndim = 2;
    ad_node_t kn{}; kn.type = AD_NODE_VARIABLE; kn.tensor_value = g_attn_K; kn.shape = shK; kn.ndim = 2;
    ad_node_t vn{}; vn.type = AD_NODE_VARIABLE; vn.tensor_value = g_attn_V; vn.shape = shV; vn.ndim = 2;
    ad_node_t node{};
    node.type = AD_NODE_TENSOR_ATTENTION;
    node.input1 = &qn; node.input2 = &kn; node.input3 = &vn;
    int64_t shO[2] = {kAttnSeqQ, kAttnDV};
    node.shape = shO; node.ndim = 2;
    node.tensor_gradient = g_attn_cotan;
    node.params.attention_params.num_heads = 1;
    node.params.attention_params.head_dim = kAttnDK;
    eshkol_tensor_backward_dispatch(&node);
}
#endif  /* ESHKOL_HAVE_FORK_DEATH_TESTS */

void attention_check() {
    constexpr int64_t seq_q = 2, seq_k = 3, d_k = 4, d_v = 3;
    double Q[seq_q * d_k], K[seq_k * d_k], V[seq_k * d_v], cotan[seq_q * d_v];
    for (auto& v : Q) v = urand();
    for (auto& v : K) v = urand();
    for (auto& v : V) v = urand();
    for (auto& v : cotan) v = urand();
    double scale = 1.0 / std::sqrt((double)d_k);

    /* -- native: exact, needs the forward's saved attention weights -- */
    std::vector<double> attn((size_t)(seq_q * seq_k)), out((size_t)(seq_q * d_v));
    attention_forward_ref(Q, K, V, attn.data(), out.data(), seq_q, seq_k, d_k, d_v, scale);

    double dQ_native[seq_q * d_k] = {0}, dK_native[seq_k * d_k] = {0}, dV_native[seq_k * d_v] = {0};
    eshkol_backward_attention(cotan, Q, K, V, attn.data(), dQ_native, dK_native, dV_native,
                              seq_q, seq_k, d_k, d_v, scale);

    const double h = 1e-6;
    std::vector<double> fdQ((size_t)(seq_q * d_k)), fdK((size_t)(seq_k * d_k)), fdV((size_t)(seq_k * d_v));
    for (int64_t i = 0; i < seq_q * d_k; i++) {
        double save = Q[i];
        Q[i] = save + h; double Lp = attention_probe(Q, K, V, cotan, seq_q, seq_k, d_k, d_v, scale);
        Q[i] = save - h; double Lm = attention_probe(Q, K, V, cotan, seq_q, seq_k, d_k, d_v, scale);
        Q[i] = save;
        fdQ[i] = (Lp - Lm) / (2.0 * h);
    }
    for (int64_t i = 0; i < seq_k * d_k; i++) {
        double save = K[i];
        K[i] = save + h; double Lp = attention_probe(Q, K, V, cotan, seq_q, seq_k, d_k, d_v, scale);
        K[i] = save - h; double Lm = attention_probe(Q, K, V, cotan, seq_q, seq_k, d_k, d_v, scale);
        K[i] = save;
        fdK[i] = (Lp - Lm) / (2.0 * h);
    }
    for (int64_t i = 0; i < seq_k * d_v; i++) {
        double save = V[i];
        V[i] = save + h; double Lp = attention_probe(Q, K, V, cotan, seq_q, seq_k, d_k, d_v, scale);
        V[i] = save - h; double Lm = attention_probe(Q, K, V, cotan, seq_q, seq_k, d_k, d_v, scale);
        V[i] = save;
        fdV[i] = (Lp - Lm) / (2.0 * h);
    }
    double fd_dQ = max_abs_diff(dQ_native, fdQ.data(), seq_q * d_k);
    double fd_dK = max_abs_diff(dK_native, fdK.data(), seq_k * d_k);
    double fd_dV = max_abs_diff(dV_native, fdV.data(), seq_k * d_v);
    char detail[160];
    std::snprintf(detail, sizeof detail, "max|dQ-fd|=%.3e max|dK-fd|=%.3e max|dV-fd|=%.3e",
                 fd_dQ, fd_dK, fd_dV);
    report("attention: native vs finite differences", fd_dQ < 1e-6 && fd_dK < 1e-6 && fd_dV < 1e-6, detail);

#if defined(ESHKOL_HAVE_FORK_DEATH_TESTS)
    std::memcpy(g_attn_Q, Q, sizeof(g_attn_Q));
    std::memcpy(g_attn_K, K, sizeof(g_attn_K));
    std::memcpy(g_attn_V, V, sizeof(g_attn_V));
    std::memcpy(g_attn_cotan, cotan, sizeof(g_attn_cotan));
    report("attention: bridge backward refuses (NEEDS-DECISION, see block comment)",
          refuses(&body_bridge_attention_backward));
#else
    std::printf("  (bridge attention refusal check skipped: no fork on this platform)\n");
#endif
}

}  // namespace

int main() {
    std::printf("=== dual-backward differential check (SW-12) ===\n");

    matmul_check();
    layernorm_check();
    transpose_check();
    sum_check();
    attention_check();

    std::printf("(embedding: bridge-vs-native comparison lives in "
               "tensor_embedding_backward_gradcheck_test.cpp, checks 1/2/9)\n");

    std::printf("=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
