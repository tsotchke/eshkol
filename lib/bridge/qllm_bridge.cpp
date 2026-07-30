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
 * @brief Mobius addition on the Poincare ball of curvature -c.
 *
 * x (+)_c y = ((1 + 2c<x,y> + c||y||^2) x + (1 - c||x||^2) y)
 *             / (1 + 2c<x,y> + c^2 ||x||^2 ||y||^2)
 */
void mobius_add(const double* x, const double* y, double c, size_t n, double* out) {
    double xy = 0.0;
    for (size_t i = 0; i < n; ++i) xy += x[i] * y[i];
    double x2 = norm_sq(x, n);
    double y2 = norm_sq(y, n);
    double num_x = 1.0 + 2.0 * c * xy + c * y2;
    double num_y = 1.0 - c * x2;
    double den = 1.0 + 2.0 * c * xy + c * c * x2 * y2;
    if (std::fabs(den) < 1e-15) den = (den < 0.0) ? -1e-15 : 1e-15;
    for (size_t i = 0; i < n; ++i) out[i] = (num_x * x[i] + num_y * y[i]) / den;
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

    double* scores = (double*)alloc_doubles(seq);
    if (!scores) return nullptr;
    double scale = 1.0 / std::sqrt((double)head_dim);

    for (size_t b = 0; b < batch; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            size_t off = (size_t)h * head_dim;
            for (size_t i = 0; i < seq; ++i) {
                size_t qi = (b * seq + i) * dim + off;
                size_t limit = causal ? (i + 1) : seq;
                double mx = -HUGE_VAL;
                for (size_t j = 0; j < limit; ++j) {
                    size_t kj = (b * seq + j) * dim + off;
                    double dot = 0.0;
                    for (size_t d = 0; d < head_dim; ++d) dot += Q[qi + d] * K[kj + d];
                    scores[j] = dot * scale;
                    if (scores[j] > mx) mx = scores[j];
                }
                double sum = 0.0;
                for (size_t j = 0; j < limit; ++j) {
                    scores[j] = std::exp(scores[j] - mx);
                    sum += scores[j];
                }
                if (sum <= 0.0) sum = 1.0;
                for (size_t d = 0; d < head_dim; ++d) {
                    double acc = 0.0;
                    for (size_t j = 0; j < limit; ++j) {
                        size_t vj = (b * seq + j) * dim + off;
                        acc += (scores[j] / sum) * V[vj + d];
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
        node->params.attention_params.num_heads = num_heads;
        node->params.attention_params.head_dim = (int64_t)head_dim;
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
    double c = (curvature == 0.0) ? 1.0 : std::fabs(curvature);

    double diff2 = 0.0;
    for (size_t i = 0; i < n; ++i) { double d = X[i] - Y[i]; diff2 += d * d; }
    double dx = 1.0 - c * norm_sq(X, n);
    double dy = 1.0 - c * norm_sq(Y, n);
    if (dx <= 0.0 || dy <= 0.0) {
        eshkol_error("qllm bridge: ad_hyperbolic_distance argument outside the Poincare ball");
        return nullptr;
    }
    double arg = 1.0 + 2.0 * c * diff2 / (dx * dy);
    if (arg < 1.0) arg = 1.0;
    double dist = std::acosh(arg) / std::sqrt(c);

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
    double c = (curvature == 0.0) ? 1.0 : std::fabs(curvature);
    double sc = std::sqrt(c);

    double* out = alloc_doubles(n);
    if (!out) return nullptr;

    double vn = std::sqrt(norm_sq(V, n));
    if (vn < 1e-15) {
        std::memcpy(out, X, n * sizeof(double));
    } else {
        double lam = 2.0 / (1.0 - c * norm_sq(X, n));
        double coef = std::tanh(sc * lam * vn / 2.0) / (sc * vn);
        double* scaled = (double*)alloc_doubles(n);
        if (!scaled) return nullptr;
        for (size_t i = 0; i < n; ++i) scaled[i] = coef * V[i];
        mobius_add(X, scaled, c, n, out);
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
    double c = (curvature == 0.0) ? 1.0 : std::fabs(curvature);
    double sc = std::sqrt(c);

    double* out = alloc_doubles(n);
    if (!out) return nullptr;

    double* neg = (double*)alloc_doubles(n);
    double* diff = (double*)alloc_doubles(n);
    if (!neg || !diff) return nullptr;
    for (size_t i = 0; i < n; ++i) neg[i] = -X[i];
    mobius_add(neg, Y, c, n, diff);

    double dn = std::sqrt(norm_sq(diff, n));
    if (dn >= 1e-15) {
        double lam = 2.0 / (1.0 - c * norm_sq(X, n));
        double t = sc * dn;
        if (t >= 1.0) t = 1.0 - 1e-12;
        double coef = (2.0 / (sc * lam)) * std::atanh(t) / dn;
        for (size_t i = 0; i < n; ++i) out[i] = coef * diff[i];
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
    double c = (curvature == 0.0) ? 1.0 : std::fabs(curvature);
    double sc = std::sqrt(c);

    const double* Q = (const double*)q->tensor_value;
    const double* K = (const double*)k->tensor_value;
    const double* V = (const double*)v->tensor_value;
    double* O = alloc_doubles(batch * seq * dim);
    double* scores = alloc_doubles(seq);
    if (!O || !scores) return nullptr;

    /* Score by NEGATIVE geodesic distance (closer => higher attention), with the
     * curvature-adaptive 1/sqrt(c * head_dim) scaling. */
    double scale = 1.0 / (sc * std::sqrt((double)head_dim));

    for (size_t b = 0; b < batch; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            size_t off = (size_t)h * head_dim;
            for (size_t i = 0; i < seq; ++i) {
                size_t qi = (b * seq + i) * dim + off;
                size_t limit = causal ? (i + 1) : seq;
                double mx = -HUGE_VAL;
                for (size_t j = 0; j < limit; ++j) {
                    size_t kj = (b * seq + j) * dim + off;
                    double diff2 = 0.0, qn = 0.0, kn = 0.0;
                    for (size_t d = 0; d < head_dim; ++d) {
                        double dd = Q[qi + d] - K[kj + d];
                        diff2 += dd * dd;
                        qn += Q[qi + d] * Q[qi + d];
                        kn += K[kj + d] * K[kj + d];
                    }
                    double dxq = 1.0 - c * qn, dxk = 1.0 - c * kn;
                    double dist;
                    if (dxq <= 0.0 || dxk <= 0.0) {
                        dist = HUGE_VAL; /* outside the ball: unreachable */
                    } else {
                        double arg = 1.0 + 2.0 * c * diff2 / (dxq * dxk);
                        if (arg < 1.0) arg = 1.0;
                        dist = std::acosh(arg) / sc;
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
                for (size_t d = 0; d < head_dim; ++d) {
                    double acc = 0.0;
                    for (size_t j = 0; j < limit; ++j) {
                        size_t vj = (b * seq + j) * dim + off;
                        acc += (scores[j] / sum) * V[vj + d];
                    }
                    O[(b * seq + i) * dim + off + d] = acc;
                }
            }
        }
    }

    ad_node_t* node = make_node(tape, AD_NODE_GEODESIC_ATTENTION, O,
                                q->shape, q->ndim, q, k, v);
    if (node) {
        node->params.attention_params.num_heads = num_heads;
        node->params.attention_params.head_dim = (int64_t)head_dim;
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
