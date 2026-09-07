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
 *   attention:  both sides are now exact and are diffed against each other
 *     here, causal and non-causal, at two shapes, with FD as the third
 *     oracle. The bridge side used to be an unconditional eshkol_fatal
 *     refusal because its forward retained neither the softmax weights nor
 *     the causal flag; the forward now retains both (see
 *     ad_tensor_attention in lib/bridge/qllm_bridge.cpp) and the exact
 *     adjoint is implemented against that contract. The refusal is retained
 *     for the contract VIOLATION only — a node presented without the
 *     retained weights — and that is asserted below too, so the fix cannot
 *     regress into a silent zero.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
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
 * ATTENTION — multi-head scaled dot-product attention, causal and non-causal.
 *
 * Both implementations are exact, and this is the op the two were furthest
 * apart on: until the SW-12 fix the bridge rule was an unconditional
 * eshkol_fatal because ad_tensor_attention retained neither the softmax
 * weights nor the causal flag. The forward now retains the dense
 * [batch, num_heads, seq, seq] weight matrix (masked entries exactly zero) in
 * node->saved_tensors[0] and records [num_heads, head_dim, causal, scale_bits]
 * in params, and tensor_attention_backward computes the exact adjoint from
 * them.
 *
 * THREE oracles, per shape and per masking mode:
 *   bridge   ad_tensor_attention -> eshkol_tensor_backward_dispatch, i.e. the
 *            real producer and the real dispatch path, not a hand-built node.
 *   native   eshkol_backward_attention, applied per (batch, head) column slice
 *            of the [batch, seq, dim] operands, fed the attention weights from
 *            the reference forward below. Feeding it causal weights (zeros
 *            above the diagonal) is exactly how the causal adjoint falls out of
 *            an unmasked kernel: every term carries a factor attn[i][j].
 *   FD       central differences of the reference forward, which is written
 *            independently of both.
 *
 * The bridge-vs-native bar is the same 1e-9 the other five ops are held to.
 * The two do not share code — the native kernel is BLAS-backed where the
 * bridge rule is plain loops — so they are not expected to be bit-identical,
 * only to agree far inside that bar.
 ******************************************************************************/

/** @brief Reference forward for multi-head attention over [batch, seq, dim],
 *  written independently of both implementations. Fills @p attn with the dense
 *  [batch, heads, seq, seq] weight matrix (masked entries left at zero) and
 *  @p out with the [batch, seq, dim] output. */
void attention_forward_ref(const double* Q, const double* K, const double* V,
                           double* attn, double* out,
                           int64_t batch, int64_t heads, int64_t seq, int64_t head_dim,
                           bool causal) {
    const int64_t dim = heads * head_dim;
    const double scale = 1.0 / std::sqrt((double)head_dim);
    std::vector<double> row((size_t)seq);

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t hh = 0; hh < heads; hh++) {
            const int64_t off = hh * head_dim;
            double* Ah = &attn[(size_t)(((b * heads) + hh) * seq * seq)];
            for (int64_t i = 0; i < seq; i++) {
                const int64_t limit = causal ? (i + 1) : seq;
                double mx = -1e300;
                for (int64_t j = 0; j < limit; j++) {
                    double dot = 0.0;
                    for (int64_t d = 0; d < head_dim; d++)
                        dot += Q[(b * seq + i) * dim + off + d] * K[(b * seq + j) * dim + off + d];
                    row[(size_t)j] = dot * scale;
                    if (row[(size_t)j] > mx) mx = row[(size_t)j];
                }
                double sum = 0.0;
                for (int64_t j = 0; j < limit; j++) {
                    row[(size_t)j] = std::exp(row[(size_t)j] - mx);
                    sum += row[(size_t)j];
                }
                for (int64_t j = 0; j < limit; j++) Ah[i * seq + j] = row[(size_t)j] / sum;
                for (int64_t j = limit; j < seq; j++) Ah[i * seq + j] = 0.0;
                for (int64_t d = 0; d < head_dim; d++) {
                    double acc = 0.0;
                    for (int64_t j = 0; j < seq; j++)
                        acc += Ah[i * seq + j] * V[(b * seq + j) * dim + off + d];
                    out[(b * seq + i) * dim + off + d] = acc;
                }
            }
        }
    }
}

/** @brief Scalar probe L = <cotan, attention(Q,K,V)> for the FD oracle. */
double attention_probe(const double* Q, const double* K, const double* V,
                       const double* cotan,
                       int64_t batch, int64_t heads, int64_t seq, int64_t head_dim,
                       bool causal) {
    const int64_t dim = heads * head_dim;
    std::vector<double> attn((size_t)(batch * heads * seq * seq));
    std::vector<double> out((size_t)(batch * seq * dim));
    attention_forward_ref(Q, K, V, attn.data(), out.data(), batch, heads, seq, head_dim, causal);
    double L = 0.0;
    for (int64_t i = 0; i < batch * seq * dim; i++) L += cotan[i] * out[i];
    return L;
}

/** @brief Run the NATIVE kernel (eshkol_backward_attention) over the same
 *  [batch, seq, dim] layout the bridge uses, one (batch, head) slice at a
 *  time: gather the slice into the contiguous [seq, head_dim] buffers the
 *  kernel expects, run it, scatter the per-head gradients back. */
void attention_native_batched(const double* Q, const double* K, const double* V,
                              const double* attn, const double* cotan,
                              double* dQ, double* dK, double* dV,
                              int64_t batch, int64_t heads, int64_t seq, int64_t head_dim) {
    const int64_t dim = heads * head_dim;
    const double scale = 1.0 / std::sqrt((double)head_dim);
    const size_t slice = (size_t)(seq * head_dim);
    std::vector<double> qs(slice), ks(slice), vs(slice), gs(slice);
    std::vector<double> dq(slice), dk(slice), dv(slice);

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t hh = 0; hh < heads; hh++) {
            const int64_t off = hh * head_dim;
            for (int64_t i = 0; i < seq; i++)
                for (int64_t d = 0; d < head_dim; d++) {
                    size_t src = (size_t)((b * seq + i) * dim + off + d);
                    size_t dst = (size_t)(i * head_dim + d);
                    qs[dst] = Q[src]; ks[dst] = K[src]; vs[dst] = V[src]; gs[dst] = cotan[src];
                }
            std::fill(dq.begin(), dq.end(), 0.0);
            std::fill(dk.begin(), dk.end(), 0.0);
            std::fill(dv.begin(), dv.end(), 0.0);
            eshkol_backward_attention(gs.data(), qs.data(), ks.data(), vs.data(),
                                      &attn[(size_t)(((b * heads) + hh) * seq * seq)],
                                      dq.data(), dk.data(), dv.data(),
                                      seq, seq, head_dim, head_dim, scale);
            for (int64_t i = 0; i < seq; i++)
                for (int64_t d = 0; d < head_dim; d++) {
                    size_t dst = (size_t)((b * seq + i) * dim + off + d);
                    size_t src = (size_t)(i * head_dim + d);
                    dQ[dst] += dq[src]; dK[dst] += dk[src]; dV[dst] += dv[src];
                }
        }
    }
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

constexpr int64_t kBadSeq = 3, kBadHeadDim = 4;
double g_attn_Q[kBadSeq * kBadHeadDim];
double g_attn_K[kBadSeq * kBadHeadDim];
double g_attn_V[kBadSeq * kBadHeadDim];
double g_attn_cotan[kBadSeq * kBadHeadDim];

/** @brief A node built the way a producer that did NOT retain the attention
 *  weights would build it. The exact adjoint is unobtainable from that, so the
 *  rule must still refuse rather than reconstruct one — this is the half of
 *  the old unconditional refusal that stays. */
void body_bridge_attention_unretained() {
    int64_t sh[3] = {1, kBadSeq, kBadHeadDim};
    ad_node_t qn{}; qn.type = AD_NODE_VARIABLE; qn.tensor_value = g_attn_Q; qn.shape = sh; qn.ndim = 3;
    ad_node_t kn{}; kn.type = AD_NODE_VARIABLE; kn.tensor_value = g_attn_K; kn.shape = sh; kn.ndim = 3;
    ad_node_t vn{}; vn.type = AD_NODE_VARIABLE; vn.tensor_value = g_attn_V; vn.shape = sh; vn.ndim = 3;
    ad_node_t node{};
    node.type = AD_NODE_TENSOR_ATTENTION;
    node.input1 = &qn; node.input2 = &kn; node.input3 = &vn;
    node.shape = sh; node.ndim = 3;
    node.tensor_gradient = g_attn_cotan;
    node.params.attention_params.num_heads = 1;
    node.params.attention_params.head_dim = kBadHeadDim;
    node.saved_tensors = nullptr;   /* the whole point */
    node.num_saved = 0;
    eshkol_tensor_backward_dispatch(&node);
}
#endif  /* ESHKOL_HAVE_FORK_DEATH_TESTS */

/** @brief One shape x masking-mode cell: bridge vs native vs FD. */
void attention_case(int64_t batch, int64_t heads, int64_t seq, int64_t head_dim,
                    bool causal, const char* label) {
    const int64_t dim = heads * head_dim;
    const size_t n = (size_t)(batch * seq * dim);

    std::vector<double> Q(n), K(n), V(n), cotan(n);
    for (auto& x : Q) x = urand();
    for (auto& x : K) x = urand();
    for (auto& x : V) x = urand();
    for (auto& x : cotan) x = urand();

    /* -- reference forward: attention weights for the native kernel -- */
    std::vector<double> attn((size_t)(batch * heads * seq * seq)), out(n);
    attention_forward_ref(Q.data(), K.data(), V.data(), attn.data(), out.data(),
                          batch, heads, seq, head_dim, causal);

    /* -- native -- */
    std::vector<double> dQ_native(n, 0.0), dK_native(n, 0.0), dV_native(n, 0.0);
    attention_native_batched(Q.data(), K.data(), V.data(), attn.data(), cotan.data(),
                             dQ_native.data(), dK_native.data(), dV_native.data(),
                             batch, heads, seq, head_dim);

    /* -- bridge: real forward producer, real backward dispatch -- */
    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 8);
    int64_t sh[3] = {batch, seq, dim};
    ad_node_t* qn = var_node(Q.data(), sh, 3);
    ad_node_t* kn = var_node(K.data(), sh, 3);
    ad_node_t* vn = var_node(V.data(), sh, 3);
    ad_node_t* node = ad_tensor_attention(tape, qn, kn, vn, (int)heads, causal);

    char name[128], detail[192];
    if (!node) {
        std::snprintf(name, sizeof name, "attention[%s]: native vs bridge", label);
        report(name, false, "bridge forward refused");
        return;
    }

    /* The forward must also agree with the reference forward — a bridge
     * adjoint that matches an incorrect forward would prove nothing. */
    double fwd_diff = max_abs_diff(out.data(), (const double*)node->tensor_value, n);

    double* cotan_buf = zeros(n);
    std::memcpy(cotan_buf, cotan.data(), n * sizeof(double));
    node->tensor_gradient = cotan_buf;
    eshkol_tensor_backward_dispatch(node);

    const double* dQ_bridge = (const double*)qn->tensor_gradient;
    const double* dK_bridge = (const double*)kn->tensor_gradient;
    const double* dV_bridge = (const double*)vn->tensor_gradient;

    double d_dQ = dQ_bridge ? max_abs_diff(dQ_native.data(), dQ_bridge, n) : -1.0;
    double d_dK = dK_bridge ? max_abs_diff(dK_native.data(), dK_bridge, n) : -1.0;
    double d_dV = dV_bridge ? max_abs_diff(dV_native.data(), dV_bridge, n) : -1.0;

    std::snprintf(name, sizeof name, "attention[%s]: native vs bridge", label);
    std::snprintf(detail, sizeof detail,
                  "max|dQ diff|=%.3e max|dK diff|=%.3e max|dV diff|=%.3e fwd=%.3e",
                  d_dQ, d_dK, d_dV, fwd_diff);
    report(name, dQ_bridge && dK_bridge && dV_bridge &&
                 d_dQ < 1e-9 && d_dK < 1e-9 && d_dV < 1e-9 && fwd_diff < 1e-12, detail);

    /* -- finite differences, the third oracle -- */
    const double h = 1e-6;
    std::vector<double> fdQ(n), fdK(n), fdV(n);
    double* tensors[3] = { Q.data(), K.data(), V.data() };
    std::vector<double>* fds[3] = { &fdQ, &fdK, &fdV };
    for (int t = 0; t < 3; t++) {
        for (size_t i = 0; i < n; i++) {
            double save = tensors[t][i];
            tensors[t][i] = save + h;
            double Lp = attention_probe(Q.data(), K.data(), V.data(), cotan.data(),
                                        batch, heads, seq, head_dim, causal);
            tensors[t][i] = save - h;
            double Lm = attention_probe(Q.data(), K.data(), V.data(), cotan.data(),
                                        batch, heads, seq, head_dim, causal);
            tensors[t][i] = save;
            (*fds[t])[i] = (Lp - Lm) / (2.0 * h);
        }
    }
    double fd_dQ = dQ_bridge ? max_abs_diff(dQ_bridge, fdQ.data(), n) : 1.0;
    double fd_dK = dK_bridge ? max_abs_diff(dK_bridge, fdK.data(), n) : 1.0;
    double fd_dV = dV_bridge ? max_abs_diff(dV_bridge, fdV.data(), n) : 1.0;
    std::snprintf(name, sizeof name, "attention[%s]: bridge vs finite differences", label);
    std::snprintf(detail, sizeof detail, "max|dQ-fd|=%.3e max|dK-fd|=%.3e max|dV-fd|=%.3e",
                  fd_dQ, fd_dK, fd_dV);
    report(name, fd_dQ < 1e-6 && fd_dK < 1e-6 && fd_dV < 1e-6, detail);

    double nfd_dQ = max_abs_diff(dQ_native.data(), fdQ.data(), n);
    double nfd_dK = max_abs_diff(dK_native.data(), fdK.data(), n);
    double nfd_dV = max_abs_diff(dV_native.data(), fdV.data(), n);
    std::snprintf(name, sizeof name, "attention[%s]: native vs finite differences", label);
    std::snprintf(detail, sizeof detail, "max|dQ-fd|=%.3e max|dK-fd|=%.3e max|dV-fd|=%.3e",
                  nfd_dQ, nfd_dK, nfd_dV);
    report(name, nfd_dQ < 1e-6 && nfd_dK < 1e-6 && nfd_dV < 1e-6, detail);
}

/** @brief Uniform-score attention must have an exactly zero score adjoint when
 * every value row is equal. This exercises the cancellation-sensitive branch
 * at both exponent ends while comparing the independent native and bridge
 * implementations. */
void attention_uniform_scores_case(double magnitude) {
    constexpr int64_t batch = 1, heads = 1, seq = 3, head_dim = 1;
    constexpr int64_t dim = heads * head_dim;
    const size_t n = (size_t)(batch * seq * dim);
    const double reciprocal = 1.0 / magnitude;
    const double Q[3] = { reciprocal, reciprocal, reciprocal };
    const double K[3] = { magnitude, magnitude, magnitude };
    const double V[3] = { 2.0, 2.0, 2.0 };
    const double cotan[3] = { 1.0, -0.5, 0.25 };
    std::vector<double> attn((size_t)(heads * seq * seq));
    std::vector<double> out(n), dQ_native(n, 0.0), dK_native(n, 0.0), dV_native(n, 0.0);
    attention_forward_ref(Q, K, V, attn.data(), out.data(), batch, heads, seq,
                          head_dim, false);
    attention_native_batched(Q, K, V, attn.data(), cotan, dQ_native.data(),
                             dK_native.data(), dV_native.data(), batch, heads,
                             seq, head_dim);

    ad_tape_t* tape = arena_allocate_tape(get_global_arena(), 8);
    int64_t shape[3] = {batch, seq, dim};
    ad_node_t* qn = var_node(Q, shape, 3);
    ad_node_t* kn = var_node(K, shape, 3);
    ad_node_t* vn = var_node(V, shape, 3);
    ad_node_t* node = ad_tensor_attention(tape, qn, kn, vn, (int)heads, false);
    bool exact = node != nullptr;
    if (node) {
        std::memcpy(node->tensor_gradient, cotan, n * sizeof(double));
        eshkol_tensor_backward_dispatch(node);
        const double* dQ_bridge = (const double*)qn->tensor_gradient;
        const double* dK_bridge = (const double*)kn->tensor_gradient;
        exact = dQ_bridge && dK_bridge;
        for (size_t i = 0; exact && i < n; ++i)
            exact = dQ_native[i] == 0.0 && dK_native[i] == 0.0 &&
                    dQ_bridge[i] == 0.0 && dK_bridge[i] == 0.0;
    }
    char name[128];
    std::snprintf(name, sizeof name,
                  "attention.uniform_scores.zero_qk.magnitude_%.0e", magnitude);
    report(name, exact, "native and bridge Q/K adjoints are bit-exact zero");
}

void attention_check() {
    /* small: one batch, one head — the plain 2-D case both kernels reduce to.
     * moderate: several batches and heads, seq > head_dim, so a head-slicing
     * or stride error cannot cancel out. Both run masked and unmasked. */
    attention_case(1, 1, 3, 4, false, "small,noncausal");
    attention_case(1, 1, 3, 4, true,  "small,causal");
    attention_case(2, 3, 5, 4, false, "moderate,noncausal");
    attention_case(2, 3, 5, 4, true,  "moderate,causal");
    const double magnitudes[] = {1e-300, 1e-200, 1e-100, 1.0,
                                 1e100, 1e200, 1e300};
    for (double magnitude : magnitudes)
        attention_uniform_scores_case(magnitude);

#if defined(ESHKOL_HAVE_FORK_DEATH_TESTS)
    for (auto& x : g_attn_Q) x = urand();
    for (auto& x : g_attn_K) x = urand();
    for (auto& x : g_attn_V) x = urand();
    for (auto& x : g_attn_cotan) x = urand();
    report("attention: bridge refuses a node with no retained weights",
          refuses(&body_bridge_attention_unretained));
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
