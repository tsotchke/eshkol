/**
 * @file qllm_bridge.cpp
 * @brief Implementation of the Eshkol <-> qLLM bridge declared in
 *        `inc/eshkol/bridge/qllm_bridge.h`.
 *
 * This file supplies the forward half of the bridge. The backward half already
 * existed and was already compiled into the runtime
 * (`lib/bridge/tensor_backward.cpp`, dispatched from
 * `lib/backend/tensor_backward.cpp` under "qLLM Bridge Tensor Ops (67-79)"),
 * but nothing in the tree ever *created* an `AD_NODE_TENSOR_*` node, so those
 * gradient rules were unreachable. Every function here records a node of the
 * canonical `ad_node_type_t` the backward dispatch already keys on, which is
 * what makes that path live.
 *
 * Contract followed by every AD-aware entry point:
 *   - the forward value is always computed;
 *   - when `tape` is non-NULL the operation is additionally recorded, with the
 *     input wiring (`input1`/`input2`/`input3`) and `params` field that the
 *     matching backward rule in tensor_backward.cpp reads;
 *   - when `tape` is NULL only the forward pass runs (the node is still
 *     allocated so the caller has somewhere to read the result from).
 *
 * Tensors on AD nodes are row-major `double` buffers in `tensor_value`, with
 * `shape`/`ndim` describing them -- the representation `tensor_backward.cpp`
 * already reads. Buffers are arena-allocated so their lifetime matches the
 * tape's.
 *
 * The `qllm_tensor_t` interop container (float32) is defined here; it is the
 * boundary representation used by eshkol_to_qllm_tensor() /
 * qllm_to_eshkol_tensor(). Eshkol computes in double, qLLM infers in float32.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <mutex>

#include "eshkol/eshkol.h"
#include "eshkol/logger.h"
#include "eshkol/bridge/qllm_bridge.h"
#include "eshkol/backend/frechet_mean_core.h"
#include "eshkol/backend/riemannian_core.h"

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

/*******************************************************************************
 * Runtime allocation surface
 *
 * Declared locally (same convention as lib/bridge/tensor_backward.cpp) so the
 * bridge does not need to pull in the full arena/codegen headers.
 ******************************************************************************/

extern "C" {
    typedef struct arena arena_t;
    arena_t* get_global_arena(void);
    void* arena_allocate_zeroed(arena_t* arena, size_t size);
    ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity);
    ad_node_t* arena_allocate_ad_node(arena_t* arena);
    void arena_tape_add_node(ad_tape_t* tape, ad_node_t* node);
}

namespace {

/** @brief Product of a shape's dimensions (element count). */
size_t elem_count(const int64_t* shape, size_t ndim) {
    size_t n = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] <= 0) return 0;
        n *= (size_t)shape[i];
    }
    return n;
}

/** @brief Zero-initialised double buffer from the global arena. */
double* alloc_doubles(size_t n) {
    if (n == 0) return nullptr;
    return (double*)arena_allocate_zeroed(get_global_arena(), n * sizeof(double));
}

/** @brief Arena-allocated copy of a shape vector. */
int64_t* copy_shape(const int64_t* shape, size_t ndim) {
    if (ndim == 0) return nullptr;
    int64_t* out = (int64_t*)arena_allocate_zeroed(get_global_arena(),
                                                   ndim * sizeof(int64_t));
    if (out) std::memcpy(out, shape, ndim * sizeof(int64_t));
    return out;
}

void ensure_grad(ad_node_t* n);   /* defined below; see its comment for why */

/**
 * @brief Allocate a tensor-valued AD node and, when `tape` is non-NULL, record
 *        it on the tape in evaluation order.
 *
 * The node owns `value` (already arena-allocated by the caller) and a private
 * copy of `shape`. `tensor_gradient` is left NULL; the backward rules allocate
 * it on first accumulation.
 */
ad_node_t* make_node(ad_tape_t* tape,
                     ad_node_type_t type,
                     double* value,
                     const int64_t* shape,
                     size_t ndim,
                     ad_node_t* in1,
                     ad_node_t* in2,
                     ad_node_t* in3) {
    ad_node_t* node = arena_allocate_ad_node(get_global_arena());
    if (!node) {
        eshkol_error("qllm bridge: failed to allocate AD node (type %d)", (int)type);
        return nullptr;
    }
    node->type = type;
    node->tensor_value = (void*)value;
    node->shape = copy_shape(shape, ndim);
    node->ndim = ndim;
    node->input1 = in1;
    node->input2 = in2;
    node->input3 = in3;
    if (value && ndim > 0) {
        /* Keep the scalar mirror meaningful for 1-element tensors so scalar
         * consumers (and the cross-entropy backward, which reads node->gradient
         * as its upstream) behave sensibly. */
        if (elem_count(shape, ndim) == 1) node->value = value[0];
    }
    if (tape) {
        node->id = tape->num_nodes;
        arena_tape_add_node(tape, node);
        /* Only needed when the chain will actually be differentiated. Covers the
         * inputs too, so the bridge guarantees a persistent gradient buffer for
         * every node in the subgraph it records regardless of who created the
         * input variable nodes. See ensure_grad(). */
        ensure_grad(node);
        ensure_grad(in1);
        ensure_grad(in2);
        ensure_grad(in3);
    }
    return node;
}

/**
 * @brief Give a node a persistent, zero-filled gradient buffer.
 *
 * This has to happen at FORWARD time, and the reason is subtle enough to be
 * worth stating. eshkol_tensor_backward_dispatch() wraps each node's backward
 * rule in arena_push_scope()/arena_pop_scope(), and arena_pop_scope() rewinds
 * the arena's bump pointer to the mark taken at push. The backward rules in
 * lib/bridge/tensor_backward.cpp allocate a destination gradient buffer lazily
 * ("if (x_node->tensor_gradient == NULL) ... = alloc_grad(n)") from the same
 * arena -- so that buffer is reclaimed the moment the node's dispatch returns.
 * The next node upstream then finds a non-NULL pointer into freed arena space
 * and the entire chain silently produces zeros.
 *
 * Allocating the buffers here, outside any backward scope, is what makes a
 * recorded chain actually differentiable: every lazy branch in the backward
 * rules is already satisfied, so nothing is allocated inside the scope and
 * nothing is rewound. Buffers are zero-filled once, which is also what the
 * rules' "+=" accumulation assumes.
 *
 * The codegen path never hit this because it pre-allocates gradient buffers of
 * its own; the bridge path was unreachable, so nothing had ever exercised it.
 */
void ensure_grad(ad_node_t* n) {
    if (!n || n->tensor_gradient || !n->shape || n->ndim == 0) return;
    size_t count = elem_count(n->shape, n->ndim);
    if (count == 0) return;
    n->tensor_gradient = arena_allocate_zeroed(get_global_arena(),
                                               count * sizeof(double));
}

/** @brief Validate that a node carries a usable tensor payload. */
bool has_tensor(const ad_node_t* n) {
    return n && n->tensor_value && n->ndim > 0 && n->shape;
}

/** @brief Row count / row width split along the last dimension. */
void row_split(const ad_node_t* n, size_t* rows, size_t* width) {
    size_t total = elem_count(n->shape, n->ndim);
    size_t last = (size_t)n->shape[n->ndim - 1];
    *width = last;
    *rows = (last > 0) ? total / last : 0;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

/** @brief Euclidean norm squared. */
double norm_sq(const double* v, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += v[i] * v[i];
    return s;
}

/**
 * @brief Poincare-ball membership: is @p p strictly inside the ball of
 *        curvature -c, i.e. sqrt(c)|p| < 1?
 *
 * Every geometric op below asks this before it computes anything, and refuses
 * when the answer is no. The reason it must be asked is that the formulas do
 * NOT fail on their own outside the ball -- they keep returning finite
 * doubles. The conformal factor lambda = 2/(1 - c|x|^2) is negative outside
 * and infinite on the boundary; artanh's argument crosses 1; and each of those
 * still lands on some number a caller cannot tell from a real one. On the
 * boundary there is no tangent space and no finite log map, so there is no
 * value to return and no derivative to record: a substituted one is a
 * fabrication, and in the AD path it is a fabricated GRADIENT, which is the
 * silent-wrong class this ledger exists to keep out (SW-76).
 *
 * The strict `<` is the whole point -- `sn < 1.0` is also false for NaN, so a
 * NaN coordinate refuses here rather than propagating into the tape.
 *
 * This is the same test the VM path applies through
 * eshkol_rm_check_point()/eshkol_rm_log_map() in
 * inc/eshkol/backend/riemannian_core.h, and the same one
 * eshkol_frechet_log_map() in inc/eshkol/backend/frechet_mean_core.h and
 * FrechetGeometry::log_map_with_jacobians() in lib/bridge/tensor_backward.cpp
 * already enforce. Projection onto the ball is a DIFFERENT operation with a
 * documented radius (`manifold-project`); it is never applied here silently.
 *
 * @param out_sn  Receives sqrt(c)|p| so the caller can name the measured
 *                value in its refusal message. Written even when the point is
 *                rejected.
 * @return true iff sqrt(c)|p| < 1.
 */
bool poincare_in_ball(const double* p, double c, size_t n, double* out_sn) {
    double sn = std::sqrt(c) * std::sqrt(norm_sq(p, n));
    if (out_sn) *out_sn = sn;
    return eshkol_rm_check_point(p, -c, (int)n) == nullptr;
}

/**
 * @brief Convert a SECTIONAL CURVATURE K to the Poincare-ball parameter c = -K,
 *        refusing every K this surface does not implement.
 *
 * The sign of K names the manifold, and the three signs are three different
 * geometries: K < 0 is the Poincare ball of radius 1/sqrt(-K), K = 0 is
 * Euclidean space, K > 0 is the sphere of radius 1/sqrt(K). The ops in this
 * file implement the K < 0 branch and only that branch -- their forwards are
 * Mobius/artanh formulas and their reverse rules in lib/bridge/tensor_backward.
 * cpp are derived from those same formulas. So the honest contract is c = -K
 * for K < 0 and an explicit refusal otherwise.
 *
 * This replaces `c = (curvature == 0.0) ? 1.0 : fabs(curvature)`, which was
 * wrong in both directions and silent in both:
 *
 *   - K = 0 (Euclidean) selected c = 1, the UNIT HYPERBOLIC BALL. A caller
 *     asking for flat space got hyperbolic answers: d(0, 0.5) came back
 *     1.0986... instead of 0.5, exp_0(0.5) = 0.462..., log_0(0.5) = 0.549...
 *   - K > 0 (sphere) selected c = |K|, i.e. a hyperbolic ball, on the same
 *     inputs for which the VM's geometry opcodes select a SPHERE. Two surfaces
 *     of the same engine answered the same question from different manifolds.
 *
 * Neither said anything. Both are the fabricated-value class this file's other
 * refusals exist to exclude, so both refuse now (SW-76).
 *
 * Implementing the flat and spherical branches is a larger change than a fix:
 * the reverse rules would need matching branches, and a forward that supports a
 * geometry its backward does not is precisely how a fabricated gradient gets
 * created. They are refused rather than half-supported.
 *
 * NOTE ON THE CONFORMAL FACTOR. This function deliberately does NOT touch the
 * lambda = 2/(1 - c|x|^2) convention used throughout; a maintainer ruling on
 * the K -> 0 continuity of that factor is pending. Keeping the curvature
 * mapping in exactly this one place is what makes that ruling a one-line change
 * here rather than an edit spread over four call sites.
 *
 * @param op  Entry-point name, for the diagnostic.
 * @return true and writes c = -K when K < 0; false with a diagnostic otherwise
 *         (K >= 0, or NaN -- `K < 0.0` is false for NaN, so NaN refuses).
 */
bool poincare_curvature(double K, const char* op, double* out_c) {
    if (K < 0.0) { *out_c = -K; return true; }
    if (!(K == K)) {
        eshkol_error("qllm bridge: %s got curvature K = NaN; K must be < 0 "
                     "(the Poincare ball of radius 1/sqrt(-K))", op);
        return false;
    }
    eshkol_error("qllm bridge: %s needs a NEGATIVE sectional curvature K "
                 "(got %.17g). K < 0 is the Poincare ball of radius 1/sqrt(-K), "
                 "which is the only geometry this op implements; K = 0 is "
                 "Euclidean space and K > 0 is the sphere of radius 1/sqrt(K), "
                 "and both would need different formulas in the forward AND in "
                 "the reverse rule. Refusing rather than answering from the "
                 "hyperbolic ball as if it were the manifold you named.",
                 op, K);
    return false;
}

} /* namespace */

/*******************************************************************************
 * Type Conversion: Eshkol <-> qLLM
 *
 * The interop container is one contiguous allocation (header + shape + data)
 * so a caller can release it with a single free(), which is what the header's
 * "must be freed by the caller" contract implies.
 ******************************************************************************/

struct qllm_tensor {
    float*  data;   /**< float32 elements, row-major. */
    size_t* shape;  /**< Dimension sizes. */
    size_t  ndim;   /**< Number of dimensions. */
    size_t  size;   /**< Element count. */
};

extern "C" qllm_tensor_t* eshkol_to_qllm_tensor(const double* eshkol_data,
                                                const size_t* shape,
                                                size_t ndim) {
    if (!eshkol_data || !shape || ndim == 0) {
        eshkol_error("qllm bridge: eshkol_to_qllm_tensor called with null data/shape");
        return nullptr;
    }

    size_t n = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0) {
            eshkol_error("qllm bridge: eshkol_to_qllm_tensor got a zero-length dimension");
            return nullptr;
        }
        n *= shape[i];
    }

    /* Single block: [struct][shape][data], each suitably aligned. */
    size_t head = (sizeof(qllm_tensor_t) + 15u) & ~(size_t)15u;
    size_t shape_bytes = ((ndim * sizeof(size_t)) + 15u) & ~(size_t)15u;
    size_t total = head + shape_bytes + n * sizeof(float);

    unsigned char* block = (unsigned char*)std::malloc(total);
    if (!block) {
        eshkol_error("qllm bridge: out of memory converting %zu elements", n);
        return nullptr;
    }

    qllm_tensor_t* t = (qllm_tensor_t*)block;
    t->shape = (size_t*)(block + head);
    t->data  = (float*)(block + head + shape_bytes);
    t->ndim  = ndim;
    t->size  = n;
    std::memcpy(t->shape, shape, ndim * sizeof(size_t));
    for (size_t i = 0; i < n; ++i) t->data[i] = (float)eshkol_data[i];
    return t;
}

extern "C" bool qllm_to_eshkol_tensor(const qllm_tensor_t* tensor,
                                      double* out_data,
                                      size_t* out_size) {
    if (!tensor || !tensor->data || !out_data) {
        eshkol_error("qllm bridge: qllm_to_eshkol_tensor called with null argument");
        return false;
    }
    for (size_t i = 0; i < tensor->size; ++i) out_data[i] = (double)tensor->data[i];
    if (out_size) *out_size = tensor->size;
    return true;
}

/*******************************************************************************
 * AD-Aware Tensor Operations
 ******************************************************************************/

extern "C" ad_node_t* ad_tensor_matmul(ad_tape_t* tape, ad_node_t* a, ad_node_t* b) {
    if (!has_tensor(a) || !has_tensor(b)) {
        eshkol_error("qllm bridge: ad_tensor_matmul needs two tensor-valued nodes");
        return nullptr;
    }
    /* tensor_matmul_backward implements the 2-D rule; refuse anything it
     * cannot differentiate exactly rather than record a node whose gradient
     * would be wrong. */
    if (a->ndim != 2 || b->ndim != 2) {
        eshkol_error("qllm bridge: ad_tensor_matmul supports 2-D operands only "
                     "(got %zu-D and %zu-D)", a->ndim, b->ndim);
        return nullptr;
    }
    size_t m = (size_t)a->shape[0], k = (size_t)a->shape[1];
    size_t k2 = (size_t)b->shape[0], n = (size_t)b->shape[1];
    if (k != k2) {
        eshkol_error("qllm bridge: ad_tensor_matmul inner-dimension mismatch (%zu vs %zu)",
                     k, k2);
        return nullptr;
    }

    const double* A = (const double*)a->tensor_value;
    const double* B = (const double*)b->tensor_value;
    double* C = alloc_doubles(m * n);
    if (!C) return nullptr;

    for (size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k; ++p) {
            double aip = A[i * k + p];
            if (aip == 0.0) continue;
            for (size_t j = 0; j < n; ++j) C[i * n + j] += aip * B[p * n + j];
        }
    }

    int64_t shape[2] = { (int64_t)m, (int64_t)n };
    return make_node(tape, AD_NODE_TENSOR_MATMUL, C, shape, 2, a, b, nullptr);
}

extern "C" ad_node_t* ad_tensor_softmax(ad_tape_t* tape, ad_node_t* x, int dim) {
    if (!has_tensor(x)) {
        eshkol_error("qllm bridge: ad_tensor_softmax needs a tensor-valued node");
        return nullptr;
    }
    /* tensor_softmax_backward normalises along the LAST axis. Accept the
     * explicit last axis or the -1 shorthand; refuse other axes instead of
     * pairing a forward over axis d with a backward over the last axis. */
    int last = (int)x->ndim - 1;
    if (dim != -1 && dim != last) {
        eshkol_error("qllm bridge: ad_tensor_softmax supports the last axis only "
                     "(requested %d, last is %d)", dim, last);
        return nullptr;
    }

    size_t rows, width;
    row_split(x, &rows, &width);
    const double* X = (const double*)x->tensor_value;
    double* Y = alloc_doubles(rows * width);
    if (!Y) return nullptr;

    for (size_t r = 0; r < rows; ++r) {
        const double* xr = &X[r * width];
        double* yr = &Y[r * width];
        double mx = xr[0];
        for (size_t i = 1; i < width; ++i) if (xr[i] > mx) mx = xr[i];
        double sum = 0.0;
        for (size_t i = 0; i < width; ++i) { yr[i] = std::exp(xr[i] - mx); sum += yr[i]; }
        if (sum <= 0.0) sum = 1.0;
        for (size_t i = 0; i < width; ++i) yr[i] /= sum;
    }

    return make_node(tape, AD_NODE_TENSOR_SOFTMAX, Y, x->shape, x->ndim, x, nullptr, nullptr);
}

extern "C" ad_node_t* ad_tensor_layernorm(ad_tape_t* tape, ad_node_t* x,
                                          ad_node_t* gamma, ad_node_t* beta,
                                          double eps) {
    if (!has_tensor(x)) {
        eshkol_error("qllm bridge: ad_tensor_layernorm needs a tensor-valued input");
        return nullptr;
    }
    size_t rows, width;
    row_split(x, &rows, &width);

    const double* X = (const double*)x->tensor_value;
    const double* G = has_tensor(gamma) ? (const double*)gamma->tensor_value : nullptr;
    const double* Bt = has_tensor(beta) ? (const double*)beta->tensor_value : nullptr;

    double* Y = alloc_doubles(rows * width);
    if (!Y) return nullptr;

    for (size_t r = 0; r < rows; ++r) {
        const double* xr = &X[r * width];
        double* yr = &Y[r * width];
        double mean = 0.0;
        for (size_t i = 0; i < width; ++i) mean += xr[i];
        mean /= (double)width;
        double var = 0.0;
        for (size_t i = 0; i < width; ++i) { double d = xr[i] - mean; var += d * d; }
        var /= (double)width;
        double inv = 1.0 / std::sqrt(var + eps);
        for (size_t i = 0; i < width; ++i) {
            double xhat = (xr[i] - mean) * inv;
            yr[i] = xhat * (G ? G[i] : 1.0) + (Bt ? Bt[i] : 0.0);
        }
    }

    ad_node_t* node = make_node(tape, AD_NODE_TENSOR_LAYERNORM, Y,
                                x->shape, x->ndim, x, gamma, beta);
    /* tensor_layernorm_backward reads eps out of params.alpha. */
    if (node) node->params.alpha = eps;
    return node;
}

extern "C" ad_node_t* ad_tensor_rmsnorm(ad_tape_t* tape, ad_node_t* x,
                                        ad_node_t* gamma, double eps) {
    if (!has_tensor(x)) {
        eshkol_error("qllm bridge: ad_tensor_rmsnorm needs a tensor-valued input");
        return nullptr;
    }
    size_t rows, width;
    row_split(x, &rows, &width);

    const double* X = (const double*)x->tensor_value;
    const double* G = has_tensor(gamma) ? (const double*)gamma->tensor_value : nullptr;

    double* Y = alloc_doubles(rows * width);
    if (!Y) return nullptr;

    for (size_t r = 0; r < rows; ++r) {
        const double* xr = &X[r * width];
        double* yr = &Y[r * width];
        double sq = 0.0;
        for (size_t i = 0; i < width; ++i) sq += xr[i] * xr[i];
        double inv = 1.0 / std::sqrt(sq / (double)width + eps);
        for (size_t i = 0; i < width; ++i) yr[i] = xr[i] * inv * (G ? G[i] : 1.0);
    }

    ad_node_t* node = make_node(tape, AD_NODE_TENSOR_RMSNORM, Y,
                                x->shape, x->ndim, x, gamma, nullptr);
    if (node) node->params.alpha = eps;
    return node;
}

extern "C" ad_node_t* ad_tensor_gelu(ad_tape_t* tape, ad_node_t* x) {
    if (!has_tensor(x)) {
        eshkol_error("qllm bridge: ad_tensor_gelu needs a tensor-valued input");
        return nullptr;
    }
    size_t n = elem_count(x->shape, x->ndim);
    const double* X = (const double*)x->tensor_value;
    double* Y = alloc_doubles(n);
    if (!Y) return nullptr;

    /* Same tanh approximation the backward rule differentiates. */
    const double a = 0.7978845608; /* sqrt(2/pi) */
    const double b = 0.044715;
    for (size_t i = 0; i < n; ++i) {
        double xi = X[i];
        Y[i] = 0.5 * xi * (1.0 + std::tanh(a * (xi + b * xi * xi * xi)));
    }
    return make_node(tape, AD_NODE_TENSOR_GELU, Y, x->shape, x->ndim, x, nullptr, nullptr);
}

extern "C" ad_node_t* ad_tensor_silu(ad_tape_t* tape, ad_node_t* x) {
    if (!has_tensor(x)) {
        eshkol_error("qllm bridge: ad_tensor_silu needs a tensor-valued input");
        return nullptr;
    }
    size_t n = elem_count(x->shape, x->ndim);
    const double* X = (const double*)x->tensor_value;
    double* Y = alloc_doubles(n);
    if (!Y) return nullptr;
    for (size_t i = 0; i < n; ++i) Y[i] = X[i] * sigmoid(X[i]);
    return make_node(tape, AD_NODE_TENSOR_SILU, Y, x->shape, x->ndim, x, nullptr, nullptr);
}

extern "C" ad_node_t* ad_tensor_cross_entropy(ad_tape_t* tape,
                                              ad_node_t* logits,
                                              ad_node_t* targets) {
    if (!has_tensor(logits) || !has_tensor(targets)) {
        eshkol_error("qllm bridge: ad_tensor_cross_entropy needs logits and targets");
        return nullptr;
    }
    size_t batch, vocab;
    row_split(logits, &batch, &vocab);

    const double* L = (const double*)logits->tensor_value;
    const double* T = (const double*)targets->tensor_value;

    double loss = 0.0;
    for (size_t b = 0; b < batch; ++b) {
        const double* row = &L[b * vocab];
        const double* tgt = &T[b * vocab];
        double mx = row[0];
        for (size_t i = 1; i < vocab; ++i) if (row[i] > mx) mx = row[i];
        double sum = 0.0;
        for (size_t i = 0; i < vocab; ++i) sum += std::exp(row[i] - mx);
        double logZ = mx + std::log(sum);
        for (size_t i = 0; i < vocab; ++i) loss -= tgt[i] * (row[i] - logZ);
    }

    /* Scalar loss: a 1-element tensor so the node has a uniform representation,
     * with the scalar mirrored into node->value. tensor_cross_entropy_backward
     * takes its upstream from node->gradient. */
    double* out = alloc_doubles(1);
    if (!out) return nullptr;
    out[0] = loss;
    int64_t shape[1] = { 1 };
    return make_node(tape, AD_NODE_TENSOR_CROSS_ENTROPY, out, shape, 1,
                     logits, targets, nullptr);
}

/**
 * @brief Embedding lookup, recorded as AD_NODE_TENSOR_EMBEDDING.
 *
 * Forward: y[i, :] = W[idx[i], :] for i in [0, num_indices).
 *
 * The node contract this fills is documented in full at the head of
 * tensor_embedding_backward (lib/bridge/tensor_backward.cpp): input1 = the
 * weight node, input2 = the INDEX node, params as int64[6] =
 * [num_indices, d_model, vocab_size, 0, 0, 0]. Threading the index tensor onto
 * the node is the piece ESH-0230 named as missing; without it the backward
 * cannot know which rows to scatter into and refuses.
 *
 * VALIDATION IS DELIBERATELY STRICT AND HAPPENS HERE, in the forward. A
 * fractional index, a negative one, or one past the vocabulary is not rounded,
 * clamped or skipped — each is refused. The backward refuses on exactly the
 * same conditions, but by then the forward has already returned a value the
 * caller may have used; catching it at record time is what keeps a mis-wired
 * producer from being discovered only at backward time, and it makes the two
 * halves agree on what an index is.
 */
extern "C" ad_node_t* ad_tensor_embedding(ad_tape_t* tape,
                                          ad_node_t* weights,
                                          ad_node_t* indices) {
    if (!has_tensor(weights) || !has_tensor(indices)) {
        eshkol_error("qllm bridge: ad_tensor_embedding needs a weight matrix "
                     "and an index tensor");
        return nullptr;
    }
    if (weights->ndim != 2) {
        eshkol_error("qllm bridge: ad_tensor_embedding expects rank-2 weights "
                     "[vocab_size, d_model], got rank %zu", weights->ndim);
        return nullptr;
    }

    const int64_t vocab_size = weights->shape[0];
    const int64_t d_model    = weights->shape[1];
    if (vocab_size <= 0 || d_model <= 0) {
        eshkol_error("qllm bridge: ad_tensor_embedding got a degenerate weight "
                     "shape [%lld, %lld]",
                     (long long)vocab_size, (long long)d_model);
        return nullptr;
    }

    const size_t num_indices = elem_count(indices->shape, indices->ndim);
    if (num_indices == 0) {
        eshkol_error("qllm bridge: ad_tensor_embedding got an empty index tensor");
        return nullptr;
    }

    const double* W   = (const double*)weights->tensor_value;
    const double* idx = (const double*)indices->tensor_value;

    double* Y = alloc_doubles(num_indices * (size_t)d_model);
    if (!Y) return nullptr;

    for (size_t i = 0; i < num_indices; ++i) {
        double v = idx[i];
        /* Whole-number test, not a cast: (int64_t)2.7 is 2, and scattering
         * row 2's gradient for a lookup that was never row 2 is silent
         * corruption of the weight gradient. */
        double r = (v < 0.0) ? -std::floor(-v + 0.5) : std::floor(v + 0.5);
        if (!(v == v) || r != v) {
            eshkol_error("qllm bridge: ad_tensor_embedding index %zu is %.17g, "
                         "which is not a whole number; an embedding index must "
                         "be integral", i, v);
            return nullptr;
        }
        int64_t row = (int64_t)r;
        if (row < 0 || row >= vocab_size) {
            eshkol_error("qllm bridge: ad_tensor_embedding index %zu is %lld, "
                         "outside the vocabulary [0, %lld)",
                         i, (long long)row, (long long)vocab_size);
            return nullptr;
        }
        std::memcpy(&Y[i * (size_t)d_model], &W[(size_t)row * (size_t)d_model],
                    (size_t)d_model * sizeof(double));
    }

    int64_t shape[2] = { (int64_t)num_indices, d_model };
    ad_node_t* node = make_node(tape, AD_NODE_TENSOR_EMBEDDING, Y, shape, 2,
                                weights, indices, nullptr);
    if (node) {
        int64_t* p = (int64_t*)&node->params;
        p[0] = (int64_t)num_indices;
        p[1] = d_model;
        p[2] = vocab_size;
        p[3] = 0;
        p[4] = 0;
        p[5] = 0;
    }
    return node;
}

extern "C" ad_node_t* ad_tensor_attention(ad_tape_t* tape,
                                          ad_node_t* q, ad_node_t* k, ad_node_t* v,
                                          int num_heads, bool causal) {
    if (!has_tensor(q) || !has_tensor(k) || !has_tensor(v)) {
        eshkol_error("qllm bridge: ad_tensor_attention needs Q, K and V");
        return nullptr;
    }
    if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3) {
        eshkol_error("qllm bridge: ad_tensor_attention expects [batch, seq, dim] tensors");
        return nullptr;
    }
    if (num_heads <= 0) {
        eshkol_error("qllm bridge: ad_tensor_attention needs num_heads > 0");
        return nullptr;
    }
    size_t batch = (size_t)q->shape[0];
    size_t seq   = (size_t)q->shape[1];
    size_t dim   = (size_t)q->shape[2];
    /* K and V are indexed with Q's strides below, so a shape mismatch is an
     * out-of-bounds read, not a broadcast. Refuse it here. */
    for (size_t d = 0; d < 3; ++d) {
        if (k->shape[d] != q->shape[d] || v->shape[d] != q->shape[d]) {
            eshkol_error("qllm bridge: ad_tensor_attention needs Q, K and V to share "
                         "one [batch, seq, dim] shape (dimension %zu: Q=%lld K=%lld V=%lld)",
                         d, (long long)q->shape[d], (long long)k->shape[d],
                         (long long)v->shape[d]);
            return nullptr;
        }
    }
    if (batch == 0 || seq == 0 || dim == 0) {
        eshkol_error("qllm bridge: ad_tensor_attention got a degenerate shape "
                     "[%zu, %zu, %zu]", batch, seq, dim);
        return nullptr;
    }
    if (dim % (size_t)num_heads != 0) {
        eshkol_error("qllm bridge: ad_tensor_attention dim %zu not divisible by %d heads",
                     dim, num_heads);
        return nullptr;
    }
    size_t head_dim = dim / (size_t)num_heads;

    const double* Q = (const double*)q->tensor_value;
    const double* K = (const double*)k->tensor_value;
    const double* V = (const double*)v->tensor_value;
    double* O = alloc_doubles(batch * seq * dim);
    if (!O) return nullptr;

    /* RETAINED FOR THE ADJOINT (SW-12).
     *
     * tensor_attention_backward needs the softmax attention weights A and it
     * needs to know which entries the causal mask removed. Both are recorded
     * here, at forward time, in the mechanism the node struct already provides
     * for exactly this ("Saved tensors for backward pass") and that the
     * non-bridge AD_NODE_ATTENTION arm already uses for its own attention
     * weights (lib/backend/tensor_backward.cpp, saved_tensors[3]).
     *
     * The buffer is the DENSE [batch, num_heads, seq, seq] weight matrix, with
     * masked-out entries left at EXACTLY zero. Storing the mask as zeros in A
     * rather than as a separate bitmap is what makes the causal case fall out
     * of the same code path as the non-causal one: every term of the adjoint
     * below carries a factor of A[i][j], so a zero weight contributes exactly
     * zero to dQ/dK/dV without a single mask test. The `causal` flag is still
     * recorded in params so the backward can validate the retained weights
     * against the shape the forward actually ran (and so a future rule can
     * recompute rather than retain without guessing).
     *
     * Recomputing A in the backward instead was the alternative; it was
     * rejected because it would have to re-derive the softmax max-shift and
     * the mask, i.e. duplicate the forward, and any drift between the two
     * copies is precisely the silently-wrong-gradient class SW-12 exists to
     * close. */
    double* A = alloc_doubles(batch * (size_t)num_heads * seq * seq);
    if (!A) return nullptr;

    double scale = 1.0 / std::sqrt((double)head_dim);

    for (size_t b = 0; b < batch; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            size_t off = (size_t)h * head_dim;
            for (size_t i = 0; i < seq; ++i) {
                size_t qi = (b * seq + i) * dim + off;
                /* Row of A for this (b, h, i). Already zero-filled, so the
                 * masked tail stays exactly zero. */
                double* arow = &A[((b * (size_t)num_heads + (size_t)h) * seq + i) * seq];
                size_t limit = causal ? (i + 1) : seq;
                double mx = -HUGE_VAL;
                for (size_t j = 0; j < limit; ++j) {
                    size_t kj = (b * seq + j) * dim + off;
                    double dot = 0.0;
                    for (size_t d = 0; d < head_dim; ++d) dot += Q[qi + d] * K[kj + d];
                    arow[j] = dot * scale;
                    if (arow[j] > mx) mx = arow[j];
                }
                double sum = 0.0;
                for (size_t j = 0; j < limit; ++j) {
                    arow[j] = std::exp(arow[j] - mx);
                    sum += arow[j];
                }
                if (sum <= 0.0) sum = 1.0;
                for (size_t j = 0; j < limit; ++j) arow[j] /= sum;
                for (size_t d = 0; d < head_dim; ++d) {
                    double acc = 0.0;
                    for (size_t j = 0; j < limit; ++j) {
                        size_t vj = (b * seq + j) * dim + off;
                        acc += arow[j] * V[vj + d];
                    }
                    O[(b * seq + i) * dim + off + d] = acc;
                }
            }
        }
    }

    ad_node_t* node = make_node(tape, AD_NODE_TENSOR_ATTENTION, O,
                                q->shape, q->ndim, q, k, v);
    if (node) {
        node->input4 = nullptr;
        /* params as int64[6] — the layout tensor_attention_backward reads:
         *   [0] num_heads   [1] head_dim   [2] causal (0/1)
         *   [3] scale bit-cast from double (the "scale_bits" convention used
         *       by the AD_NODE_ATTENTION params and the Frechet rule)
         *   [4] [5] reserved, zero
         * [0]/[1] deliberately coincide with the named attention_params
         * fields so both spellings read the same slots. */
        int64_t* p = (int64_t*)&node->params;
        p[0] = (int64_t)num_heads;
        p[1] = (int64_t)head_dim;
        p[2] = causal ? 1 : 0;
        std::memcpy(&p[3], &scale, sizeof scale);
        p[4] = 0;
        p[5] = 0;

        double** saved = (double**)arena_allocate_zeroed(get_global_arena(),
                                                         sizeof(double*));
        if (!saved) {
            eshkol_error("qllm bridge: ad_tensor_attention could not retain the "
                         "attention weights; refusing to record a node whose "
                         "backward would have nothing exact to work from");
            return nullptr;
        }
        saved[0] = A;
        node->saved_tensors = (void**)saved;
        node->num_saved = 1;
    }
    return node;
}

/*******************************************************************************
 * Geometric AD Operations (Riemannian manifold)
 *
 * Poincare-ball formulas for curvature -c. At c = 1 the distance reduces to the
 * form quoted in the header:
 *     d(x,y) = acosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
 ******************************************************************************/

extern "C" ad_node_t* ad_hyperbolic_distance(ad_tape_t* tape, ad_node_t* x,
                                             ad_node_t* y, double curvature) {
    if (!has_tensor(x) || !has_tensor(y)) {
        eshkol_error("qllm bridge: ad_hyperbolic_distance needs two points");
        return nullptr;
    }
    size_t n = elem_count(x->shape, x->ndim);
    if (n != elem_count(y->shape, y->ndim)) {
        eshkol_error("qllm bridge: ad_hyperbolic_distance point dimensions differ");
        return nullptr;
    }
    const double* X = (const double*)x->tensor_value;
    const double* Y = (const double*)y->tensor_value;
    double c;
    if (!poincare_curvature(curvature, "ad_hyperbolic_distance", &c)) return nullptr;

    double snx = 0.0, sny = 0.0;
    if (!poincare_in_ball(X, c, n, &snx) || !poincare_in_ball(Y, c, n, &sny)) {
        eshkol_error("qllm bridge: ad_hyperbolic_distance got a point that is "
                     "not strictly inside the Poincare ball of curvature -%.17g "
                     "(radius 1/sqrt(c) = %.17g): sqrt(c)|x| = %.17g, "
                     "sqrt(c)|y| = %.17g, both of which must be < 1. The "
                     "distance diverges at the boundary, so there is no value "
                     "to return and no derivative to record; refusing rather "
                     "than substituting one. Project explicitly first if that "
                     "is the intent.",
                     c, 1.0 / std::sqrt(c), snx, sny);
        return nullptr;
    }
    double dist = 0.0;
    const char* why = eshkol_rm_distance(X, Y, -c, (int)n, &dist);
    if (why) {
        eshkol_error("qllm bridge: ad_hyperbolic_distance refused the operands: %s",
                     why);
        return nullptr;
    }

    double* out = alloc_doubles(1);
    if (!out) return nullptr;
    out[0] = dist;
    int64_t shape[1] = { 1 };
    ad_node_t* node = make_node(tape, AD_NODE_HYPERBOLIC_DISTANCE, out, shape, 1,
                                x, y, nullptr);
    if (node) node->params.curvature = c;
    return node;
}

extern "C" ad_node_t* ad_poincare_exp_map(ad_tape_t* tape, ad_node_t* x,
                                          ad_node_t* v, double curvature) {
    if (!has_tensor(x) || !has_tensor(v)) {
        eshkol_error("qllm bridge: ad_poincare_exp_map needs a base point and a tangent vector");
        return nullptr;
    }
    size_t n = elem_count(x->shape, x->ndim);
    if (n != elem_count(v->shape, v->ndim)) {
        eshkol_error("qllm bridge: ad_poincare_exp_map dimension mismatch");
        return nullptr;
    }
    const double* X = (const double*)x->tensor_value;
    const double* V = (const double*)v->tensor_value;
    double c;
    if (!poincare_curvature(curvature, "ad_poincare_exp_map", &c)) return nullptr;
    /* The shared exp-map implementation checks the base point before using
     * the conformal factor and checks that its computed result remains an
     * interior point. */
    double snx = 0.0;
    if (!poincare_in_ball(X, c, n, &snx)) {
        eshkol_error("qllm bridge: ad_poincare_exp_map base point is not "
                     "strictly inside the Poincare ball of curvature -%.17g "
                     "(radius 1/sqrt(c) = %.17g): sqrt(c)|x| = %.17g, which "
                     "must be < 1. There is no tangent space at or beyond the "
                     "boundary, so exp_x(v) has no value and no derivative "
                     "there; refusing rather than substituting one.",
                     c, 1.0 / std::sqrt(c), snx);
        return nullptr;
    }

    double* out = alloc_doubles(n);
    double* scratch = alloc_doubles(n);
    if (!out || !scratch) return nullptr;
    const char* why = eshkol_rm_exp_map(X, V, -c, (int)n, out, scratch);
    if (why) {
        eshkol_error("qllm bridge: ad_poincare_exp_map refused the operands: %s",
                     why);
        return nullptr;
    }

    ad_node_t* node = make_node(tape, AD_NODE_POINCARE_EXP_MAP, out,
                                x->shape, x->ndim, x, v, nullptr);
    if (node) node->params.curvature = c;
    return node;
}

extern "C" ad_node_t* ad_poincare_log_map(ad_tape_t* tape, ad_node_t* x,
                                          ad_node_t* y, double curvature) {
    if (!has_tensor(x) || !has_tensor(y)) {
        eshkol_error("qllm bridge: ad_poincare_log_map needs two points");
        return nullptr;
    }
    size_t n = elem_count(x->shape, x->ndim);
    if (n != elem_count(y->shape, y->ndim)) {
        eshkol_error("qllm bridge: ad_poincare_log_map dimension mismatch");
        return nullptr;
    }
    const double* X = (const double*)x->tensor_value;
    const double* Y = (const double*)y->tensor_value;
    double c;
    if (!poincare_curvature(curvature, "ad_poincare_log_map", &c)) return nullptr;
    double snx = 0.0, sny = 0.0;
    if (!poincare_in_ball(X, c, n, &snx) || !poincare_in_ball(Y, c, n, &sny)) {
        eshkol_error("qllm bridge: ad_poincare_log_map got a point that is not "
                     "strictly inside the Poincare ball of curvature -%.17g "
                     "(radius 1/sqrt(c) = %.17g): sqrt(c)|x| = %.17g, "
                     "sqrt(c)|y| = %.17g, both of which must be < 1. log_x(y) "
                     "is defined only between interior points; refusing rather "
                     "than substituting a fabricated tangent vector.",
                     c, 1.0 / std::sqrt(c), snx, sny);
        return nullptr;
    }

    double* out = alloc_doubles(n);
    if (!out) return nullptr;

    double* scratch = alloc_doubles(n);
    if (!scratch) return nullptr;
    const char* why = eshkol_rm_log_map(X, Y, -c, (int)n, out, scratch);
    if (why) {
        eshkol_error("qllm bridge: ad_poincare_log_map refused the operands: %s",
                     why);
        return nullptr;
    }

    ad_node_t* node = make_node(tape, AD_NODE_POINCARE_LOG_MAP, out,
                                x->shape, x->ndim, x, y, nullptr);
    if (node) node->params.curvature = c;
    return node;
}

extern "C" ad_node_t* ad_geodesic_attention(ad_tape_t* tape,
                                            ad_node_t* q, ad_node_t* k, ad_node_t* v,
                                            int num_heads, double curvature,
                                            bool causal) {
    if (!has_tensor(q) || !has_tensor(k) || !has_tensor(v)) {
        eshkol_error("qllm bridge: ad_geodesic_attention needs Q, K and V");
        return nullptr;
    }
    if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3) {
        eshkol_error("qllm bridge: ad_geodesic_attention expects [batch, seq, dim] tensors");
        return nullptr;
    }
    if (num_heads <= 0) {
        eshkol_error("qllm bridge: ad_geodesic_attention needs num_heads > 0");
        return nullptr;
    }
    size_t batch = (size_t)q->shape[0];
    size_t seq   = (size_t)q->shape[1];
    size_t dim   = (size_t)q->shape[2];
    if (dim % (size_t)num_heads != 0) {
        eshkol_error("qllm bridge: ad_geodesic_attention dim %zu not divisible by %d heads",
                     dim, num_heads);
        return nullptr;
    }
    size_t head_dim = dim / (size_t)num_heads;
    if (!(curvature == curvature)) {
        eshkol_error("qllm bridge: ad_geodesic_attention got curvature K = NaN; "
                     "the Euclidean, hyperbolic, and spherical branches require "
                     "a finite sectional curvature");
        return nullptr;
    }

    const double* Q = (const double*)q->tensor_value;
    const double* K = (const double*)k->tensor_value;
    const double* V = (const double*)v->tensor_value;

    /* Every Q and K head-slice is a point of the selected constant-curvature
     * manifold, and the score is a geodesic distance between two of them, so
     * each slice must be validated before any score is computed. This used to be
     * handled INSIDE the score loop by `if (dxq <= 0.0 || dxk <= 0.0) dist =
     * HUGE_VAL;` commented "outside the ball: unreachable", which silently
     * demoted an invalid point to an attention weight of exactly zero: the op
     * returned a full, finite, plausible attention output in which one key had
     * simply been dropped, with no diagnostic, and the backward then produced
     * gradients for that arrangement. "Unreachable" is a statement about
     * hyperbolic geometry between valid points; a point outside the ball is
     * not far away, it is not a point of the manifold at all. Refuse, and name
     * the slice, so the caller learns which row is off-manifold instead of
     * silently losing it. Checking up front also keeps the softmax honest: mx
     * is then finite, so `sum` is >= 1 by its own max term and the downstream
     * `if (sum <= 0.0) sum = 1.0;` guard becomes unreachable rather than
     * papering over an all-off-manifold NaN row. */
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq; ++t) {
            for (int h = 0; h < num_heads; ++h) {
                size_t off = (b * seq + t) * dim + (size_t)h * head_dim;
                const double norm_scale = curvature == 0.0
                                               ? 1.0
                                               : std::sqrt(std::fabs(curvature));
                double snq = norm_scale * std::sqrt(norm_sq(Q + off, head_dim));
                double snk = norm_scale * std::sqrt(norm_sq(K + off, head_dim));
                bool okq = eshkol_rm_check_point(Q + off, curvature,
                                                 (int)head_dim) == nullptr;
                bool okk = eshkol_rm_check_point(K + off, curvature,
                                                 (int)head_dim) == nullptr;
                if (!okq || !okk) {
                    const double radius = curvature > 0.0
                                              ? 1.0 / std::sqrt(curvature)
                                              : (curvature < 0.0
                                                     ? 1.0 / std::sqrt(-curvature)
                                                     : 0.0);
                    eshkol_error(
                        "qllm bridge: ad_geodesic_attention got a %s head-slice "
                        "that is not on the manifold for sectional curvature "
                        "K = %.17g (radius %.17g) at "
                        "batch %zu, position %zu, head %d: measured scaled "
                        "norm(q) = %.17g, scaled norm(k) = %.17g. The first "
                        "invalid query/key row "
                        "has no geodesic score; refusing before softmax rather "
                        "than creating a NaN or dropping it.",
                        okq ? "key" : "query", curvature, radius, b, t, h,
                        snq, snk);
                    return nullptr;
                }
            }
        }
    }

    double* O = alloc_doubles(batch * seq * dim);
    double* scores = alloc_doubles(seq);
    /* Retain the softmax weights for the backward (SW-65). Recomputing them in
     * the reverse pass would have to re-derive the max-shift and the causal
     * mask, i.e. duplicate this loop, and any drift between the two copies is
     * the silently-wrong-gradient class SW-12 records for ad_tensor_attention.
     * Zero-filled, so a masked tail stays exactly zero. */
    double* A = alloc_doubles(batch * (size_t)num_heads * seq * seq);
    if (!O || !scores || !A) return nullptr;

    /* Score by NEGATIVE geodesic distance (closer => higher attention), with
     * the same branch scale as the VM: hyperbolic uses 1/sqrt(-K), while the
     * Euclidean and spherical branches use one. */
    double metric_scale = curvature < 0.0 ? std::sqrt(-curvature) : 1.0;
    double scale = 1.0 / (metric_scale * std::sqrt((double)head_dim));

    for (size_t b = 0; b < batch; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            size_t off = (size_t)h * head_dim;
            for (size_t i = 0; i < seq; ++i) {
                size_t qi = (b * seq + i) * dim + off;
                size_t limit = causal ? (i + 1) : seq;
                double mx = -HUGE_VAL;
                for (size_t j = 0; j < limit; ++j) {
                    size_t kj = (b * seq + j) * dim + off;
                    double dist = 0.0;
                    const char* why = eshkol_rm_distance(
                        Q + qi, K + kj, curvature, (int)head_dim, &dist);
                    if (why) {
                        eshkol_error("qllm bridge: ad_geodesic_attention refused "
                                     "query/key row (%zu,%zu): %s", i, j, why);
                        return nullptr;
                    }
                    scores[j] = -dist * scale;
                    if (scores[j] > mx) mx = scores[j];
                }
                double sum = 0.0;
                for (size_t j = 0; j < limit; ++j) {
                    scores[j] = std::exp(scores[j] - mx);
                    sum += scores[j];
                }
                if (sum <= 0.0) sum = 1.0;
                double* arow =
                    &A[((b * (size_t)num_heads + (size_t)h) * seq + i) * seq];
                for (size_t j = 0; j < limit; ++j) arow[j] = scores[j] / sum;
                for (size_t d = 0; d < head_dim; ++d) {
                    double acc = 0.0;
                    for (size_t j = 0; j < limit; ++j) {
                        size_t vj = (b * seq + j) * dim + off;
                        acc += arow[j] * V[vj + d];
                    }
                    O[(b * seq + i) * dim + off + d] = acc;
                }
            }
        }
    }

    ad_node_t* node = make_node(tape, AD_NODE_GEODESIC_ATTENTION, O,
                                q->shape, q->ndim, q, k, v);
    if (node) {
        /* params as int64[6], the layout tensor_geodesic_attention_backward
         * reads:
         *   [0] num_heads   [1] head_dim   [2] causal (0/1)
         *   [3] sectional curvature K bit-cast from double (the "scale_bits" convention
         *       shared with ad_tensor_attention and the Frechet rule)
         *   [4] [5] reserved, zero
         * [0]/[1] deliberately coincide with the named attention_params fields
         * so both spellings read the same slots. Before this, `causal` and the
         * curvature were not recorded at all — the backward could not have
         * reconstructed the mask or the metric. */
        int64_t* p = (int64_t*)&node->params;
        p[0] = (int64_t)num_heads;
        p[1] = (int64_t)head_dim;
        p[2] = causal ? 1 : 0;
        std::memcpy(&p[3], &curvature, sizeof curvature);
        p[4] = 0;
        p[5] = 0;

        double** saved = (double**)arena_allocate_zeroed(get_global_arena(),
                                                         sizeof(double*));
        if (!saved) {
            eshkol_error("qllm bridge: ad_geodesic_attention could not retain the "
                         "attention weights; refusing to record a node whose "
                         "backward would have nothing exact to work from");
            return nullptr;
        }
        saved[0] = A;
        node->saved_tensors = (void**)saved;
        node->num_saved = 1;
    }
    return node;
}

/**
 * @brief Weighted Fréchet (Karcher) mean, recorded as AD_NODE_FRECHET_MEAN.
 *
 * The forward is the shared f64 Karcher iteration from
 * inc/eshkol/backend/frechet_mean_core.h — the same code the VM's `frechet-mean`
 * opcode runs. Sharing it is not housekeeping. The backward
 * (tensor_frechet_mean_backward) differentiates the stationarity condition
 *
 *     F(mu) = sum_i w_i log_mu(x_i) = 0
 *
 * implicitly, which is only valid AT the fixed point, so it RECOMPUTES the
 * relative residual and refuses above tolerance. Forward and backward therefore
 * have to agree on what "converged" means, down to the units the residual is
 * measured in (Riemannian, not ambient — see the header). Two copies of this
 * iteration would eventually disagree, and the failure mode is not a crash: it
 * is a mean that one gate accepts and the other refuses, or worse, two means
 * that both pass.
 *
 * WHY THIS REFUSES RATHER THAN RECORDS ON NON-CONVERGENCE. At a non-stationary
 * mu the implicit formulas still evaluate — they just return the derivative of
 * a condition that does not hold, which is a plausible wrong number. The
 * backward refuses there. Recording a node the backward will refuse would move
 * the failure from forward time (where the caller can still act) to backward
 * time (where a training loop has already consumed the value), so the forward
 * refuses first, on the same criterion.
 *
 * NODE CONTRACT (documented in full at tensor_frechet_mean_backward):
 *   input1        points node [n_points, dim]
 *   input2        weights node [n_points], or NULL for uniform
 *   tensor_value  the converged mean mu*, dim doubles
 *   params int64[6]  [n_points, dim, K bits, tol bits, 0, 0]
 * K and the tolerance travel bit-cast from double, the same "scale_bits"
 * convention ad_tensor_attention uses for its scale.
 */
extern "C" ad_node_t* ad_frechet_mean(ad_tape_t* tape,
                                      ad_node_t* points,
                                      ad_node_t* weights,
                                      double curvature,
                                      double tolerance) {
    if (!has_tensor(points)) {
        eshkol_error("qllm bridge: ad_frechet_mean needs a points tensor "
                     "[n_points, dim]");
        return nullptr;
    }
    if (points->ndim != 2) {
        eshkol_error("qllm bridge: ad_frechet_mean expects rank-2 points "
                     "[n_points, dim], got rank %zu", points->ndim);
        return nullptr;
    }
    const int64_t n_points = points->shape[0];
    const int64_t dim      = points->shape[1];
    if (n_points <= 0 || dim <= 0) {
        eshkol_error("qllm bridge: ad_frechet_mean got a degenerate points "
                     "shape [%lld, %lld]", (long long)n_points, (long long)dim);
        return nullptr;
    }
    if (!(curvature <= 0.0) || !(curvature == curvature)) {
        eshkol_error("qllm bridge: ad_frechet_mean needs sectional curvature "
                     "K <= 0 (got %.17g); the ball has radius 1/sqrt(-K)",
                     curvature);
        return nullptr;
    }

    /* Uniform weights are expressed by passing no weight node at all, which is
     * also what the backward reads: with input2 NULL it uses w_i = 1 and
     * produces no dL/dw. A node of ones would instead claim the weights were
     * differentiated and came out with some gradient, which is a different
     * statement. */
    const double* W = nullptr;
    int64_t n_w = 0;
    if (weights) {
        if (!has_tensor(weights)) {
            eshkol_error("qllm bridge: ad_frechet_mean got a weights node with "
                         "no tensor payload; pass NULL for uniform weights");
            return nullptr;
        }
        n_w = (int64_t)elem_count(weights->shape, weights->ndim);
        if (n_w < n_points) {
            eshkol_error("qllm bridge: ad_frechet_mean got %lld weights for "
                         "%lld points", (long long)n_w, (long long)n_points);
            return nullptr;
        }
        W = (const double*)weights->tensor_value;
    }

    double* mu = alloc_doubles((size_t)dim);
    double* scratch = alloc_doubles((size_t)dim * 4);
    if (!mu || !scratch) return nullptr;

    double resid = 0.0;
    const char* why = eshkol_frechet_mean_compute(
        (const double*)points->tensor_value, W, n_w,
        (int)n_points, (int)dim, curvature, mu, scratch, &resid);
    if (why) {
        eshkol_error("qllm bridge: ad_frechet_mean did not converge (%s); "
                     "n_points=%lld dim=%lld K=%.17g residual=%.3g bar=%.3g. "
                     "Refusing to record a node whose implicit derivative is "
                     "only defined at the fixed point.",
                     why, (long long)n_points, (long long)dim, curvature,
                     resid, ESHKOL_FRECHET_RESID_TOL);
        return nullptr;
    }

    int64_t shape[1] = { dim };
    ad_node_t* node = make_node(tape, AD_NODE_FRECHET_MEAN, mu, shape, 1,
                                points, weights, nullptr);
    if (node) {
        int64_t* p = (int64_t*)&node->params;
        p[0] = n_points;
        p[1] = dim;
        std::memcpy(&p[2], &curvature, sizeof curvature);
        std::memcpy(&p[3], &tolerance, sizeof tolerance);
        p[4] = 0;
        p[5] = 0;
    }
    return node;
}

/*******************************************************************************
 * Bridge Lifecycle
 *
 * The AD surface above is self-contained: it computes in portable C and records
 * onto Eshkol's own tape, matching the documented default build ("the portable
 * C reference matmul, no external dependency" -- docs/SDNC.md S13). The
 * lifecycle below manages the OPTIONAL qLLM tensor runtime: it loads
 * libsemiclassical_qllm and verifies it really is that library by resolving a
 * known entry point, so eshkol_qllm_bridge_ready() reports a fact rather than
 * an assumption.
 ******************************************************************************/

namespace {

std::mutex g_bridge_mutex;
void* g_bridge_handle = nullptr;

#if defined(_WIN32)
const char* kDefaultQllmLibrary = "semiclassical_qllm.dll";
#elif defined(__APPLE__)
const char* kDefaultQllmLibrary = "libsemiclassical_qllm.dylib";
#else
const char* kDefaultQllmLibrary = "libsemiclassical_qllm.so";
#endif

/** @brief Entry point every real qLLM tensor runtime exports; used to reject
 *  a file that loads but is not the qLLM library. */
const char* kProbeSymbol = "qllm_tensor_create";

void* load_library(const char* path) {
#if defined(_WIN32)
    return (void*)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* find_symbol(void* handle, const char* name) {
#if defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)handle, name);
#else
    return dlsym(handle, name);
#endif
}

void close_library(void* handle) {
#if defined(_WIN32)
    FreeLibrary((HMODULE)handle);
#else
    dlclose(handle);
#endif
}

} /* namespace */

extern "C" bool eshkol_qllm_bridge_init(const char* library_path) {
    std::lock_guard<std::mutex> lock(g_bridge_mutex);
    if (g_bridge_handle) return true; /* Already initialised: idempotent. */

    const char* path = (library_path && library_path[0]) ? library_path
                                                         : kDefaultQllmLibrary;
    void* handle = load_library(path);
    if (!handle) {
        eshkol_warn("qllm bridge: could not load '%s'; the qLLM tensor runtime is "
                    "unavailable (the bridge's own AD surface still works)", path);
        return false;
    }
    if (!find_symbol(handle, kProbeSymbol)) {
        eshkol_warn("qllm bridge: '%s' loaded but does not export %s; not a qLLM "
                    "tensor runtime", path, kProbeSymbol);
        close_library(handle);
        return false;
    }
    g_bridge_handle = handle;
    return true;
}

extern "C" void eshkol_qllm_bridge_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_bridge_mutex);
    if (!g_bridge_handle) return;
    close_library(g_bridge_handle);
    g_bridge_handle = nullptr;
}

extern "C" bool eshkol_qllm_bridge_ready(void) {
    std::lock_guard<std::mutex> lock(g_bridge_mutex);
    return g_bridge_handle != nullptr;
}
