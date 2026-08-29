/**
 * @file tensor_backward.cpp
 * @brief Backward pass implementations for tensor AD nodes.
 *
 * Provides gradient computation for each tensor operation recorded
 * on the AD tape. Called during backpropagation to compute dL/dx
 * for each input tensor.
 *
 * Each backward function follows the signature:
 *   void backward_<op>(ad_node_t* node)
 * where node->tensor_gradient contains dL/d(output) and the function
 * must propagate gradients to node->input1->tensor_gradient, etc.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>

/* Include the Eshkol AD types — eshkol.h is a C++ header, no extern "C" needed */
#include "eshkol/eshkol.h"
#include "eshkol/logger.h"
#include "eshkol/backend/riemannian_core.h"

/*******************************************************************************
 * Internal Helpers
 ******************************************************************************/

/** @brief Computes the total element count of a tensor from its shape
 *  (product of all dimension sizes). */
static size_t tensor_size(const int64_t* shape, size_t ndim) {
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= (size_t)shape[i];
    }
    return size;
}

/* Arena allocation for gradient tensors */
extern "C" {
    typedef struct arena arena_t;
    arena_t* get_global_arena(void);
    void* arena_allocate_zeroed(arena_t* arena, size_t size);
}

/** @brief Allocates a zero-initialized gradient buffer of n doubles from
 *  the global arena. */
static double* alloc_grad(size_t n) {
    return (double*)arena_allocate_zeroed(get_global_arena(), n * sizeof(double));
}

/*******************************************************************************
 * MatMul Backward
 *
 * Forward: C = A @ B where A:[m,k], B:[k,n] → C:[m,n]
 * Backward:
 *   dL/dA = dL/dC @ B^T   →  [m,n] @ [n,k] = [m,k]
 *   dL/dB = A^T @ dL/dC   →  [k,m] @ [m,n] = [k,n]
 ******************************************************************************/

/** @brief Backward pass for matrix multiply: propagates dL/dC into
 *  dL/dA = dL/dC @ B^T and dL/dB = A^T @ dL/dC (2D case). */
extern "C" void tensor_matmul_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;

    double* dC = (double*)node->tensor_gradient;
    ad_node_t* a_node = node->input1;
    ad_node_t* b_node = node->input2;

    if (!a_node || !b_node) return;

    double* A = (double*)a_node->tensor_value;
    double* B = (double*)b_node->tensor_value;

    /* Get dimensions: A is [m,k], B is [k,n], C is [m,n] */
    /* For simplicity, handle 2D case */
    size_t m = (a_node->ndim >= 1) ? (size_t)a_node->shape[0] : 1;
    size_t k = (a_node->ndim >= 2) ? (size_t)a_node->shape[1] : 1;
    size_t n = (b_node->ndim >= 2) ? (size_t)b_node->shape[1] : 1;

    /* dL/dA = dL/dC @ B^T */
    if (a_node->tensor_gradient == NULL) {
        a_node->tensor_gradient = alloc_grad(m * k);
    }
    double* dA = (double*)a_node->tensor_gradient;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
            double sum = 0.0;
            for (size_t l = 0; l < n; l++) {
                sum += dC[i * n + l] * B[j * n + l]; /* B^T[l,j] = B[j,l] */
            }
            dA[i * k + j] += sum;
        }
    }

    /* dL/dB = A^T @ dL/dC */
    if (b_node->tensor_gradient == NULL) {
        b_node->tensor_gradient = alloc_grad(k * n);
    }
    double* dB = (double*)b_node->tensor_gradient;
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t l = 0; l < m; l++) {
                sum += A[l * k + i] * dC[l * n + j]; /* A^T[i,l] = A[l,i] */
            }
            dB[i * n + j] += sum;
        }
    }
}

/*******************************************************************************
 * Softmax Backward
 *
 * Forward: y = softmax(x) where y[i] = exp(x[i]) / sum(exp(x[j]))
 * Backward:
 *   dL/dx[i] = y[i] * (dL/dy[i] - sum_j(dL/dy[j] * y[j]))
 ******************************************************************************/

/** @brief Backward pass for softmax: for each row along the last
 *  dimension, computes dx[i] = y[i] * (dy[i] - sum_j(dy[j] * y[j])). */
extern "C" void tensor_softmax_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;

    double* dy = (double*)node->tensor_gradient;
    double* y = (double*)node->tensor_value;
    ad_node_t* x_node = node->input1;

    if (!x_node || !y) return;

    size_t n = tensor_size(node->shape, node->ndim);

    if (x_node->tensor_gradient == NULL) {
        x_node->tensor_gradient = alloc_grad(n);
    }
    double* dx = (double*)x_node->tensor_gradient;

    /* For each row in the last dimension */
    size_t last_dim = (node->ndim > 0) ? (size_t)node->shape[node->ndim - 1] : n;
    size_t num_rows = n / last_dim;

    for (size_t r = 0; r < num_rows; r++) {
        double* y_row = &y[r * last_dim];
        double* dy_row = &dy[r * last_dim];
        double* dx_row = &dx[r * last_dim];

        /* dot = sum_j(dy[j] * y[j]) */
        double dot = 0.0;
        for (size_t j = 0; j < last_dim; j++) {
            dot += dy_row[j] * y_row[j];
        }

        /* dx[i] += y[i] * (dy[i] - dot) */
        for (size_t i = 0; i < last_dim; i++) {
            dx_row[i] += y_row[i] * (dy_row[i] - dot);
        }
    }
}

/*******************************************************************************
 * LayerNorm Backward
 *
 * Forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
 * Backward: complex chain rule through mean and variance
 ******************************************************************************/

/** @brief Backward pass for layer normalization: for each row along the
 *  last dimension, chains gradients back through the mean/variance
 *  normalization to dx, and accumulates dgamma when a gamma input is
 *  present. */
extern "C" void tensor_layernorm_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;

    double* dy = (double*)node->tensor_gradient;
    ad_node_t* x_node = node->input1;
    ad_node_t* gamma_node = node->input2;

    if (!x_node) return;

    double* x = (double*)x_node->tensor_value;
    double* gamma = gamma_node ? (double*)gamma_node->tensor_value : NULL;
    double eps = node->params.alpha; /* eps stored in params */

    size_t n = tensor_size(node->shape, node->ndim);
    size_t last_dim = (node->ndim > 0) ? (size_t)node->shape[node->ndim - 1] : n;
    size_t num_rows = n / last_dim;

    if (x_node->tensor_gradient == NULL) {
        x_node->tensor_gradient = alloc_grad(n);
    }
    double* dx = (double*)x_node->tensor_gradient;

    /* Gradient for gamma (accumulated across batch) */
    double* dgamma = NULL;
    if (gamma_node) {
        if (gamma_node->tensor_gradient == NULL) {
            gamma_node->tensor_gradient = alloc_grad(last_dim);
        }
        dgamma = (double*)gamma_node->tensor_gradient;
    }

    for (size_t r = 0; r < num_rows; r++) {
        const double* x_row = &x[r * last_dim];
        const double* dy_row = &dy[r * last_dim];
        double* dx_row = &dx[r * last_dim];

        /* Compute mean and variance */
        double mean = 0.0;
        for (size_t i = 0; i < last_dim; i++) mean += x_row[i];
        mean /= (double)last_dim;

        double var = 0.0;
        for (size_t i = 0; i < last_dim; i++) {
            double d = x_row[i] - mean;
            var += d * d;
        }
        var /= (double)last_dim;

        double inv_std = 1.0 / sqrt(var + eps);
        double N = (double)last_dim;

        /* Compute intermediate gradients */
        double sum_dy_xhat = 0.0;
        double sum_dy = 0.0;
        for (size_t i = 0; i < last_dim; i++) {
            double xhat = (x_row[i] - mean) * inv_std;
            double dy_scaled = dy_row[i] * (gamma ? gamma[i] : 1.0);
            sum_dy_xhat += dy_scaled * xhat;
            sum_dy += dy_scaled;

            if (dgamma) {
                dgamma[i] += dy_row[i] * xhat;
            }
        }

        /* dx = (1/N) * inv_std * (N * dy_scaled - sum_dy - xhat * sum_dy_xhat) */
        for (size_t i = 0; i < last_dim; i++) {
            double xhat = (x_row[i] - mean) * inv_std;
            double dy_scaled = dy_row[i] * (gamma ? gamma[i] : 1.0);
            dx_row[i] += inv_std * (dy_scaled - sum_dy / N - xhat * sum_dy_xhat / N);
        }
    }
}

/*******************************************************************************
 * RMSNorm Backward
 *
 * Forward: y = x * rsqrt(mean(x^2) + eps) * gamma
 * Backward: chain rule through RMS computation
 ******************************************************************************/

/** @brief Backward pass for RMS normalization: for each row along the
 *  last dimension, chains gradients back through the RMS computation to
 *  dx, and accumulates dgamma when a gamma input is present. */
extern "C" void tensor_rmsnorm_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;

    double* dy = (double*)node->tensor_gradient;
    ad_node_t* x_node = node->input1;
    ad_node_t* gamma_node = node->input2;

    if (!x_node) return;

    double* x = (double*)x_node->tensor_value;
    double* gamma = gamma_node ? (double*)gamma_node->tensor_value : NULL;
    double eps = node->params.alpha;

    size_t n = tensor_size(node->shape, node->ndim);
    size_t last_dim = (node->ndim > 0) ? (size_t)node->shape[node->ndim - 1] : n;
    size_t num_rows = n / last_dim;

    if (x_node->tensor_gradient == NULL) {
        x_node->tensor_gradient = alloc_grad(n);
    }
    double* dx = (double*)x_node->tensor_gradient;

    double* dgamma = NULL;
    if (gamma_node) {
        if (gamma_node->tensor_gradient == NULL) {
            gamma_node->tensor_gradient = alloc_grad(last_dim);
        }
        dgamma = (double*)gamma_node->tensor_gradient;
    }

    for (size_t r = 0; r < num_rows; r++) {
        const double* x_row = &x[r * last_dim];
        const double* dy_row = &dy[r * last_dim];
        double* dx_row = &dx[r * last_dim];

        /* RMS = sqrt(mean(x^2) + eps) */
        double sum_sq = 0.0;
        for (size_t i = 0; i < last_dim; i++) sum_sq += x_row[i] * x_row[i];
        double rms = sqrt(sum_sq / (double)last_dim + eps);
        double inv_rms = 1.0 / rms;
        double N = (double)last_dim;

        /* Gradient computation */
        double sum_dy_x = 0.0;
        for (size_t i = 0; i < last_dim; i++) {
            double dy_g = dy_row[i] * (gamma ? gamma[i] : 1.0);
            sum_dy_x += dy_g * x_row[i];
        }

        for (size_t i = 0; i < last_dim; i++) {
            double xhat = x_row[i] * inv_rms;
            double dy_g = dy_row[i] * (gamma ? gamma[i] : 1.0);
            /* y_i = x_i * inv * g_i with inv = (mean(x^2) + eps)^(-1/2), so
             *   dL/dx_k = inv * (dy_k g_k - inv^2 x_k * sum_i(dy_i g_i x_i) / N).
             * Since xhat = x_k * inv, the correction term is inv * xhat * sum / N
             * -- ONE factor of inv, not two. This previously read
             * "xhat * sum_dy_x * inv_rms * inv_rms", i.e. inv^3 x_k where the
             * derivative calls for inv^2 x_k, so every gradient was scaled by a
             * spurious extra 1/rms and was only correct when rms happened to be
             * 1. Nothing caught it because no code path created an
             * AD_NODE_TENSOR_RMSNORM node until the bridge forward half landed;
             * the gradient check now covers it (7.1e-01 -> ~1e-8). */
            dx_row[i] += inv_rms * (dy_g - xhat * sum_dy_x * inv_rms / N);

            if (dgamma) {
                dgamma[i] += dy_row[i] * xhat;
            }
        }
    }
}

/*******************************************************************************
 * GELU Backward
 *
 * gelu(x) = 0.5 * x * (1 + tanh(a * (x + b * x^3)))
 * where a = sqrt(2/pi), b = 0.044715
 ******************************************************************************/

/** @brief Backward pass for the tanh-approximation GELU activation:
 *  applies the derivative of gelu(x) = 0.5*x*(1+tanh(a*(x+b*x^3))) to
 *  scale the incoming gradient. */
extern "C" void tensor_gelu_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;

    double* dy = (double*)node->tensor_gradient;
    ad_node_t* x_node = node->input1;
    if (!x_node) return;

    double* x = (double*)x_node->tensor_value;
    size_t n = tensor_size(node->shape, node->ndim);

    if (x_node->tensor_gradient == NULL) {
        x_node->tensor_gradient = alloc_grad(n);
    }
    double* dx = (double*)x_node->tensor_gradient;

    const double a = 0.7978845608; /* sqrt(2/pi) */
    const double b = 0.044715;

    for (size_t i = 0; i < n; i++) {
        double xi = x[i];
        double inner = a * (xi + b * xi * xi * xi);
        double tanh_inner = tanh(inner);
        double sech2 = 1.0 - tanh_inner * tanh_inner;
        double d_inner = a * (1.0 + 3.0 * b * xi * xi);
        dx[i] += dy[i] * (0.5 * (1.0 + tanh_inner) + 0.5 * xi * sech2 * d_inner);
    }
}

/*******************************************************************************
 * SiLU/Swish Backward
 *
 * silu(x) = x * sigmoid(x)
 * silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
 ******************************************************************************/

/** @brief Backward pass for SiLU/Swish: applies the derivative
 *  silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))) to scale the
 *  incoming gradient. */
extern "C" void tensor_silu_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;

    double* dy = (double*)node->tensor_gradient;
    ad_node_t* x_node = node->input1;
    if (!x_node) return;

    double* x = (double*)x_node->tensor_value;
    size_t n = tensor_size(node->shape, node->ndim);

    if (x_node->tensor_gradient == NULL) {
        x_node->tensor_gradient = alloc_grad(n);
    }
    double* dx = (double*)x_node->tensor_gradient;

    for (size_t i = 0; i < n; i++) {
        double sig = 1.0 / (1.0 + exp(-x[i]));
        dx[i] += dy[i] * sig * (1.0 + x[i] * (1.0 - sig));
    }
}

/*******************************************************************************
 * Cross-Entropy Backward
 *
 * loss = -sum(target * log(softmax(logits)))
 * dL/dlogits = softmax(logits) - target  (numerically stable)
 ******************************************************************************/

/** @brief Backward pass for softmax cross-entropy loss: computes the
 *  numerically-stable gradient dL/dlogits = softmax(logits) - targets for
 *  each row of the batch. */
extern "C" void tensor_cross_entropy_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;

    ad_node_t* logits_node = node->input1;
    ad_node_t* targets_node = node->input2;

    if (!logits_node || !targets_node) return;

    double* logits = (double*)logits_node->tensor_value;
    double* targets = (double*)targets_node->tensor_value;

    size_t n = tensor_size(logits_node->shape, logits_node->ndim);
    size_t vocab = (logits_node->ndim > 0) ?
        (size_t)logits_node->shape[logits_node->ndim - 1] : n;
    size_t batch = n / vocab;

    if (logits_node->tensor_gradient == NULL) {
        logits_node->tensor_gradient = alloc_grad(n);
    }
    double* dlogits = (double*)logits_node->tensor_gradient;

    double loss_grad = node->gradient; /* dL/d(loss), usually 1.0 */

    for (size_t b = 0; b < batch; b++) {
        const double* row = &logits[b * vocab];
        double* drow = &dlogits[b * vocab];
        const double* tgt = &targets[b * vocab];

        /* Compute softmax */
        double max_val = row[0];
        for (size_t i = 1; i < vocab; i++) {
            if (row[i] > max_val) max_val = row[i];
        }
        double sum_exp = 0.0;
        for (size_t i = 0; i < vocab; i++) {
            sum_exp += exp(row[i] - max_val);
        }

        /* dL/dlogits = softmax - target */
        for (size_t i = 0; i < vocab; i++) {
            double prob = exp(row[i] - max_val) / sum_exp;
            drow[i] += loss_grad * (prob - tgt[i]);
        }
    }
}

/*******************************************************************************
 * Backward passes for the remaining AD_NODE_TENSOR_* ops that the
 * forward codegen can emit. Missing implementations here silently
 * zero-out gradients — models using attention / shape ops would train
 * with corrupt gradients and no error signal. Each function below is
 * intentionally conservative (propagate a copy of the output gradient
 * with the minimum correct operation for that op) so that, until a
 * mathematically exact version lands, we at least don't lose signal
 * magnitude.
 ******************************************************************************/

/** @brief Backward pass for transpose: dL/dX[i,j] = dL/dY[j,i]. Permutation
 *  is its own inverse for 2D; higher-rank transposes require the
 *  permutation vector which the forward codegen should set up on the
 *  node. */
extern "C" void tensor_transpose_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;
    ad_node_t* in = node->input1;
    if (!in) return;

    size_t m = (node->ndim >= 1) ? (size_t)node->shape[0] : 1;
    size_t n = (node->ndim >= 2) ? (size_t)node->shape[1] : 1;
    double* dY = (double*)node->tensor_gradient;

    size_t total = m * n;
    if (in->tensor_gradient == NULL) in->tensor_gradient = alloc_grad(total);
    double* dX = (double*)in->tensor_gradient;
    /* dX shape is (n, m); output shape is (m, n) */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            dX[j * m + i] += dY[i * n + j];
        }
    }
}

/** @brief Backward pass for a full sum reduction: the forward pass
 *  reduces to a scalar, so backward broadcasts the scalar dL/dy uniformly
 *  onto every element of the input. */
extern "C" void tensor_sum_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;
    ad_node_t* in = node->input1;
    if (!in) return;
    size_t n = tensor_size(in->shape, in->ndim);
    double dy = ((double*)node->tensor_gradient)[0];
    if (in->tensor_gradient == NULL) in->tensor_gradient = alloc_grad(n);
    double* dX = (double*)in->tensor_gradient;
    for (size_t i = 0; i < n; i++) dX[i] += dy;
}

/** @brief Backward pass for broadcast-add: forward is y[i] = a + b[i]
 *  (scalar a broadcast over tensor b) or y[i,j] = a[i] + b[i,j]. Splits
 *  the gradient along both inputs: dL/db = dL/dy (elementwise, sum-reduced
 *  if b is smaller than the output), dL/da = sum(dL/dy) across the
 *  broadcast axis. */
extern "C" void tensor_broadcast_add_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;
    ad_node_t* a = node->input1;
    ad_node_t* b = node->input2;
    size_t n_out = tensor_size(node->shape, node->ndim);
    double* dy = (double*)node->tensor_gradient;

    if (b) {
        size_t n_b = tensor_size(b->shape, b->ndim);
        if (b->tensor_gradient == NULL) b->tensor_gradient = alloc_grad(n_b);
        double* dB = (double*)b->tensor_gradient;
        /* b matches y shape → elementwise; if smaller, sum-reduce. */
        if (n_b == n_out) {
            for (size_t i = 0; i < n_b; i++) dB[i] += dy[i];
        } else if (n_b > 0) {
            size_t factor = n_out / n_b;
            for (size_t i = 0; i < n_b; i++) {
                double s = 0.0;
                for (size_t k = 0; k < factor; k++) s += dy[i * factor + k];
                dB[i] += s;
            }
        }
    }

    if (a) {
        size_t n_a = tensor_size(a->shape, a->ndim);
        if (a->tensor_gradient == NULL) a->tensor_gradient = alloc_grad(n_a == 0 ? 1 : n_a);
        double* dA = (double*)a->tensor_gradient;
        if (n_a == n_out) {
            for (size_t i = 0; i < n_a; i++) dA[i] += dy[i];
        } else {
            /* Scalar or smaller — sum all. */
            double s = 0.0;
            for (size_t i = 0; i < n_out; i++) s += dy[i];
            dA[0] += s;
        }
    }
}

/** @brief Backward pass for broadcast-multiply: y = a * b (one operand may
 *  be broadcast). Applies the product rule dL/da = sum(dL/dy * b),
 *  dL/db = sum(dL/dy * a), with the same broadcast-reduction handling as
 *  tensor_broadcast_add_backward(). */
extern "C" void tensor_broadcast_mul_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;
    ad_node_t* a = node->input1;
    ad_node_t* b = node->input2;
    size_t n_out = tensor_size(node->shape, node->ndim);
    double* dy = (double*)node->tensor_gradient;
    double* A = a ? (double*)a->tensor_value : NULL;
    double* B = b ? (double*)b->tensor_value : NULL;

    if (b && B) {
        size_t n_b = tensor_size(b->shape, b->ndim);
        if (b->tensor_gradient == NULL) b->tensor_gradient = alloc_grad(n_b);
        double* dB = (double*)b->tensor_gradient;
        double a_scalar = A ? A[0] : 0.0;  /* scalar case */
        if (n_b == n_out) {
            for (size_t i = 0; i < n_b; i++) dB[i] += dy[i] * (A ? A[i] : a_scalar);
        } else if (n_b > 0) {
            size_t factor = n_out / n_b;
            for (size_t i = 0; i < n_b; i++) {
                double s = 0.0;
                for (size_t k = 0; k < factor; k++) s += dy[i * factor + k] * (A ? A[i * factor + k] : a_scalar);
                dB[i] += s;
            }
        }
    }

    if (a && A) {
        size_t n_a = tensor_size(a->shape, a->ndim);
        if (a->tensor_gradient == NULL) a->tensor_gradient = alloc_grad(n_a == 0 ? 1 : n_a);
        double* dA = (double*)a->tensor_gradient;
        double b_scalar = B ? B[0] : 0.0;
        if (n_a == n_out) {
            for (size_t i = 0; i < n_a; i++) dA[i] += dy[i] * (B ? B[i] : b_scalar);
        } else {
            double s = 0.0;
            for (size_t i = 0; i < n_out; i++) s += dy[i] * (B ? B[i] : b_scalar);
            dA[0] += s;
        }
    }
}

/*******************************************************************************
 * Embedding Backward — exact indexed scatter-add (ESH-0230)
 *
 * Forward:   y[i, :] = W[idx[i], :]      i = 0 .. num_indices-1
 * Backward:  dL/dW[idx[i], :] += dL/dy[i, :]
 *
 * The lookup is a gather, so its adjoint is a scatter-add. Two properties of
 * that adjoint are the whole reason this needs its own rule:
 *
 *   1. Rows of W never looked up receive EXACTLY zero — the gradient is
 *      genuinely sparse, not merely small. Any rule that spreads the upstream
 *      gradient over other rows is wrong, not approximate.
 *   2. A row looked up k times receives the SUM of all k upstream rows.
 *      `+=` (not `=`) is load-bearing: overwriting is the classic scatter-add
 *      bug and it silently under-counts every repeated token — the most common
 *      case in real text, where frequent tokens repeat within one sequence.
 *
 * NODE CONTRACT (this is the threading ESH-0230 asked for)
 *   node->input1        weight node W, shape [vocab_size, d_model]
 *   node->input2        index node,   tensor_value = num_indices f64 lookup
 *                       indices (Eshkol tensors are f64 throughout, so the
 *                       indices arrive as exactly-representable whole doubles)
 *   node->shape/ndim    output shape [num_indices, d_model]
 *   node->params as int64[6]  [num_indices, d_model, vocab_size, 0, 0, 0]
 *                       (matches the layout documented in
 *                       inc/eshkol/backend/tensor_backward.h). Zero or absent
 *                       entries are recovered from the input/output shapes.
 *
 * The index tensor is an integer-valued operand, so it carries no gradient:
 * d y / d idx does not exist (the map is piecewise constant in idx). input2's
 * tensor_gradient is therefore deliberately left untouched rather than being
 * seeded with a zero that would read as "differentiated, came out zero".
 ******************************************************************************/

/** @brief Round an f64 lookup index to int64, rejecting anything that is not a
 *  whole number. A fractional index means the producer passed something that
 *  was never an index; silently truncating it would scatter the gradient into
 *  the wrong row of W, which is the exact failure mode this rule exists to
 *  prevent. Returns false if @p v is not integral or not finite. */
static bool exact_index_of(double v, int64_t* out) {
    if (!(v == v) || v > 9.007199254740992e15 || v < -9.007199254740992e15)
        return false;                       /* NaN/inf, or beyond exact f64 ints */
    double r = (v < 0.0) ? -std::floor(-v + 0.5) : std::floor(v + 0.5);
    if (r != v) return false;               /* fractional — not an index */
    *out = (int64_t)r;
    return true;
}

/** @brief Backward pass for embedding lookup: scatters each upstream output
 *  row dL/dy[i, :] into dL/dW[idx[i], :], accumulating when an index repeats.
 *  See the block comment above for the node contract and why `+=` matters. */
extern "C" void tensor_embedding_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;

    ad_node_t* w_node   = node->input1;
    ad_node_t* idx_node = node->input2;

    /* HARD CONSTRAINT (exact AD or an explicit error, never a silent/plausible
     * zero). Before ESH-0230 this rule could not run at all because the index
     * operand was not on the node; now that it is part of the contract, a
     * missing index operand means the *producer* is at fault. Refusing here is
     * what keeps a mis-wired forward from quietly training on a wrong
     * gradient. */
    if (!idx_node || !idx_node->tensor_value) {
        eshkol_fatal("embedding backward: node->input2 must carry the lookup-index "
                     "tensor (ESH-0230 contract: input1 = weights [vocab, d_model], "
                     "input2 = f64 indices [num_indices], params = "
                     "[num_indices, d_model, vocab_size]); refusing to guess an "
                     "index and scatter the gradient into the wrong rows.");
        return;
    }
    if (!w_node) return;   /* weights are a constant — nothing to propagate into */

    const double* dy  = (const double*)node->tensor_gradient;
    const double* idx = (const double*)idx_node->tensor_value;

    /* Dimensions: prefer the declared params, fall back to the shapes. */
    const int64_t* p = (const int64_t*)&node->params;
    int64_t num_indices = p[0];
    int64_t d_model     = p[1];
    int64_t vocab_size  = p[2];

    if (num_indices <= 0)
        num_indices = (node->ndim >= 1 && node->shape) ? node->shape[0]
                    : (int64_t)tensor_size(idx_node->shape, idx_node->ndim);
    if (d_model <= 0)
        d_model = (node->ndim >= 2 && node->shape) ? node->shape[1]
                : ((w_node->ndim >= 2 && w_node->shape) ? w_node->shape[1] : 1);
    if (vocab_size <= 0)
        vocab_size = (w_node->ndim >= 1 && w_node->shape) ? w_node->shape[0] : 0;

    if (num_indices <= 0 || d_model <= 0 || vocab_size <= 0) {
        eshkol_fatal("embedding backward: degenerate shape "
                     "(num_indices=%lld, d_model=%lld, vocab_size=%lld); the "
                     "producer must set params [num_indices, d_model, vocab_size] "
                     "or shapes that imply them.",
                     (long long)num_indices, (long long)d_model,
                     (long long)vocab_size);
        return;
    }

    /* The index tensor must be long enough to cover every output row, or some
     * row's gradient would be read from uninitialised memory. */
    size_t idx_len = tensor_size(idx_node->shape, idx_node->ndim);
    if (idx_node->ndim > 0 && idx_len < (size_t)num_indices) {
        eshkol_fatal("embedding backward: index tensor holds %zu entries but the "
                     "output has %lld rows; refusing to read past the index "
                     "tensor.", idx_len, (long long)num_indices);
        return;
    }

    size_t w_total = (size_t)vocab_size * (size_t)d_model;
    if (w_node->tensor_gradient == NULL)
        w_node->tensor_gradient = alloc_grad(w_total);
    double* dW = (double*)w_node->tensor_gradient;
    if (!dW) return;

    /* THE SCATTER-ADD.  `+=` on both axes: `+=` into dW accumulates across
     * repeated indices within this call, and dW itself is the node's running
     * gradient so it also accumulates across multiple uses of W on the tape.
     * Rows of W that no index selects are left untouched at zero — that is the
     * correct sparse adjoint of a gather, not a dropped gradient. */
    for (int64_t i = 0; i < num_indices; i++) {
        int64_t row;
        if (!exact_index_of(idx[i], &row)) {
            eshkol_fatal("embedding backward: lookup index %lld is %.17g, which is "
                         "not a whole number; an embedding index must be integral "
                         "(rounding it would scatter the gradient into the wrong "
                         "row of the weight matrix).", (long long)i, idx[i]);
            return;
        }
        if (row < 0 || row >= vocab_size) {
            /* Out of range is a forward-pass bug that already produced garbage
             * in y. Skipping it here (the old native behaviour) would return a
             * gradient that is wrong by exactly the contribution of that row —
             * a plausible number. Refuse. */
            eshkol_fatal("embedding backward: lookup index %lld is %lld, outside "
                         "[0, %lld) for a weight matrix with %lld rows; refusing "
                         "to drop its gradient contribution silently.",
                         (long long)i, (long long)row, (long long)vocab_size,
                         (long long)vocab_size);
            return;
        }
        const double* dy_row = dy + (size_t)i * (size_t)d_model;
        double*       dW_row = dW + (size_t)row * (size_t)d_model;
        for (int64_t d = 0; d < d_model; d++)
            dW_row[d] += dy_row[d];
    }
}

/*******************************************************************************
 * Fréchet mean backward — implicit differentiation at the fixed point
 *
 * The weighted Fréchet (Karcher) mean of points x_1..x_n on the Poincaré ball
 * with weights w_1..w_n is the minimiser of the weighted variance
 *
 *     mu* = argmin_mu  sum_i w_i d(mu, x_i)^2
 *
 * and is therefore defined IMPLICITLY, as the stationary point of that
 * functional:
 *
 *     F(mu; X, w) := sum_i w_i log_mu(x_i) = 0.                            (*)
 *
 * WHY IMPLICIT DIFFERENTIATION AND NOT UNROLLING.  The forward is computed by a
 * fixed-point iteration, so there are two different things one could
 * differentiate: the mathematical object mu*(X, w) defined by (*), or the
 * particular finite iteration that approximates it. They do not agree — the
 * unrolled derivative carries the iteration's own transient, converges to the
 * implicit one only as the iterate converges, and depends on the starting point
 * and the iteration count, none of which are properties of the Fréchet mean.
 * The derivative of the mathematical object is the implicit one, so that is what
 * this rule computes: differentiate (*) and solve.
 *
 * Differentiating (*) totally at mu = mu*:
 *
 *     A dmu + sum_j (dF/dx_j) dx_j + sum_j (dF/dw_j) dw_j = 0,
 *     A := dF/dmu = sum_i w_i * d log_mu(x_i)/d mu           (d x d)
 *
 * so d mu/d x_j = -A^{-1} (w_j * d log_mu(x_j)/d x_j) and
 *    d mu/d w_j = -A^{-1} log_mu(x_j).
 *
 * In reverse mode we are handed g = dL/dmu and want dL/dx_j and dL/dw_j, which
 * needs only ONE linear solve regardless of n:
 *
 *     solve  A^T z = g
 *     dL/dx_j = -w_j (d log_mu(x_j)/d x_j)^T z
 *     dL/dw_j = -<log_mu(x_j), z>
 *
 * THE RESIDUAL GATE IS NOT OPTIONAL.  Every line above assumes F(mu*) = 0. At a
 * point that has not converged, F != 0, the implicit function theorem does not
 * apply, and the formulas still return a smooth, plausible, WRONG vector — the
 * worst failure class there is, because nothing downstream can tell it from a
 * correct gradient. So the rule recomputes the residual from the retained mu*,
 * points and weights and refuses if it is not at the fixed point. Recomputing
 * rather than trusting a residual stored by the forward is deliberate: a stored
 * residual can be stale with respect to the operands actually on the node.
 *
 * NODE CONTRACT
 *   node->input1        points node, shape [n_points, dim]
 *   node->input2        weights node, shape [n_points]
 *   node->tensor_value  the converged mean mu*, dim doubles
 *   node->tensor_gradient  upstream dL/dmu, dim doubles
 *   node->params as int64[6]
 *       [0] n_points
 *       [1] dim
 *       [2] sectional curvature K <= 0, bit-cast from double
 *           (the "scale_bits" convention already used by the attention params)
 *       [3] residual tolerance, bit-cast from double; <= 0 or non-finite
 *           selects the default
 *       [4] [5] reserved, zero
 *
 * The ball has radius 1/sqrt(c) for c = -K; K = 0 is the Euclidean case, where
 * the mean is linear in the points and the implicit machinery degenerates to the
 * exact closed form.
 ******************************************************************************/

/** @brief Default relative tolerance for the stationarity residual. The
 *  residual is a tangent vector at mu whose scale is set by the spread of the
 *  data, so the gate is relative to sum_i w_i and to the largest |log_mu(x_i)|
 *  seen — an absolute bar would be meaningless for tightly or widely spread
 *  point sets alike. */
static const double kFrechetResidualTol = 1e-9;

/** @brief Bit-cast an int64 params slot back to the double it was stored from
 *  (the attention rule's "scale_bits" convention). */
static double double_from_bits(int64_t bits) {
    double d;
    std::memcpy(&d, &bits, sizeof d);
    return d;
}

namespace {

/** @brief Small dense workspace for the Poincaré-ball geometry of one Fréchet
 *  mean backward. All buffers are plain std::vector: the enclosing dispatcher
 *  runs inside an arena scope that is popped on return, so scratch must not come
 *  from the arena, and the persistent gradient buffers must not come from here.
 */
struct FrechetGeometry {
    double  c;        /**< -K, so the ball has radius 1/sqrt(c); c > 0 */
    double  s;        /**< sqrt(c) */
    int64_t d;        /**< ambient dimension */

    /* Scratch reused per point. */
    std::vector<double> u;         /**< (-mu) (+)_c x                       */
    std::vector<double> du_dmu;    /**< d x d                              */
    std::vector<double> du_dx;     /**< d x d                              */
    std::vector<double> dlog_dmu;  /**< d x d                              */
    std::vector<double> dlog_dx;   /**< d x d                              */
    std::vector<double> logv;      /**< log_mu(x)                          */

    FrechetGeometry(double curvature_K, int64_t dim)
        : c(-curvature_K), s(std::sqrt(-curvature_K)), d(dim),
          u((size_t)dim), du_dmu((size_t)(dim * dim)), du_dx((size_t)(dim * dim)),
          dlog_dmu((size_t)(dim * dim)), dlog_dx((size_t)(dim * dim)),
          logv((size_t)dim) {}

    static double dot(const double* a, const double* b, int64_t n) {
        double t = 0.0;
        for (int64_t i = 0; i < n; i++) t += a[i] * b[i];
        return t;
    }

    /**
     * @brief Möbius addition u = a (+)_c x together with its Jacobians in a and
     *        in x, all from the one shared set of intermediates.
     *
     *   u = (A1 a + B1 x) / D
     *   A1 = 1 + 2c<a,x> + c|x|^2      B1 = 1 - c|a|^2
     *   D  = 1 + 2c<a,x> + c^2 |a|^2 |x|^2
     *
     * Writing the quotient rule out once for both arguments keeps the two
     * Jacobians consistent by construction; deriving them separately is how a
     * sign error in one of them survives a gradient check on the other.
     */
    void mobius_add_with_jacobians(const double* a, const double* x,
                                   double* out,          /* d      */
                                   double* dout_da,      /* d x d  */
                                   double* dout_dx)      /* d x d  */
    {
        const int64_t n = d;
        double ax = dot(a, x, n);
        double aa = dot(a, a, n);
        double xx = dot(x, x, n);

        double A1 = 1.0 + 2.0 * c * ax + c * xx;
        double B1 = 1.0 - c * aa;
        double D  = 1.0 + 2.0 * c * ax + c * c * aa * xx;
        double invD = 1.0 / D;

        for (int64_t i = 0; i < n; i++) out[i] = (A1 * a[i] + B1 * x[i]) * invD;

        /* Row gradients of the scalars. */
        for (int64_t j = 0; j < n; j++) {
            double dA1_da = 2.0 * c * x[j];
            double dB1_da = -2.0 * c * a[j];
            double dD_da  = 2.0 * c * x[j] + 2.0 * c * c * xx * a[j];

            double dA1_dx = 2.0 * c * a[j] + 2.0 * c * x[j];
            /* dB1/dx = 0 */
            double dD_dx  = 2.0 * c * a[j] + 2.0 * c * c * aa * x[j];

            for (int64_t i = 0; i < n; i++) {
                double kron = (i == j) ? 1.0 : 0.0;
                dout_da[i * n + j] =
                    (A1 * kron + a[i] * dA1_da + x[i] * dB1_da) * invD
                    - out[i] * dD_da * invD;
                dout_dx[i * n + j] =
                    (a[i] * dA1_dx + B1 * kron) * invD
                    - out[i] * dD_dx * invD;
            }
        }
    }

    /**
     * @brief log_mu(x) and its Jacobians in mu and in x.
     *
     *   log_mu(x) = k(mu) * phi(r) * u,     u = (-mu) (+)_c x,  r = |u|
     *   k(mu)     = (1 - c|mu|^2)/sqrt(c)   ( = 2/(sqrt(c) lambda_mu) )
     *   phi(r)    = artanh(sqrt(c) r) / r
     *
     * @return false if the pair is outside the differentiable domain (x on the
     *         ball boundary, or u degenerate), in which case no Jacobian exists
     *         and the caller must refuse rather than substitute a limit.
     */
    bool log_map_with_jacobians(const double* mu, const double* x) {
        const int64_t n = d;

        std::vector<double> neg_mu((size_t)n);
        for (int64_t i = 0; i < n; i++) neg_mu[(size_t)i] = -mu[i];

        mobius_add_with_jacobians(neg_mu.data(), x, u.data(),
                                  du_dmu.data(), du_dx.data());
        /* d u / d mu = (d u / d a) * (d a / d mu) = -(d u / d a). */
        for (size_t t = 0; t < du_dmu.size(); t++) du_dmu[t] = -du_dmu[t];

        double r2 = dot(u.data(), u.data(), n);
        double r  = std::sqrt(r2);

        double mumu = dot(mu, mu, n);
        double k    = (1.0 - c * mumu) / s;

        /* r == 0 means x == mu exactly: log_mu(mu) = 0 and the Jacobian in x is
         * k*sqrt(c)... times the identity in the limit, but the u/|u| direction
         * is undefined at the point itself. Handle the limit exactly rather than
         * dividing by zero: phi(r) -> sqrt(c) and phi'(r) -> 0 as r -> 0. */
        double phi, dphi;
        if (r <= 0.0) {
            phi  = s;
            dphi = 0.0;
        } else {
            double sr = s * r;
            if (!(sr < 1.0)) return false;      /* x at/outside the boundary */
            double at = std::atanh(sr);
            phi  = at / r;
            /* d/dr [ artanh(s r)/r ] = s/(r (1 - s^2 r^2)) - artanh(s r)/r^2 */
            dphi = s / (r * (1.0 - sr * sr)) - at / r2;
        }

        for (int64_t i = 0; i < n; i++) logv[(size_t)i] = k * phi * u[(size_t)i];

        /* log = k(mu) * phi(r) * u, with r = |u| and u = u(mu, x):
         *
         *   d log/d mu = phi * u (dk/dmu) + k [ phi' u u^T/r + phi I ] du/dmu
         *   d log/d x  =                    k [ phi' u u^T/r + phi I ] du/dx
         *
         * dk/dmu = -2 c mu^T / s. The bracket is the same operator in both, so
         * it is applied once and reused. */
        double inv_r = (r > 0.0) ? (1.0 / r) : 0.0;
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j < n; j++) {
                /* Apply M = phi' u u^T/r + phi I to column j of du/dmu and
                 * du/dx. Same operator for both, built once per (i, m). */
                double acc_mu = 0.0, acc_x = 0.0;
                for (int64_t m = 0; m < n; m++) {
                    double M_im = dphi * u[(size_t)i] * u[(size_t)m] * inv_r
                                + ((i == m) ? phi : 0.0);
                    acc_mu += M_im * du_dmu[(size_t)(m * n + j)];
                    acc_x  += M_im * du_dx[(size_t)(m * n + j)];
                }
                double dk_dmu_j = -2.0 * c * mu[j] / s;
                dlog_dmu[(size_t)(i * n + j)] =
                    phi * u[(size_t)i] * dk_dmu_j + k * acc_mu;
                dlog_dx[(size_t)(i * n + j)] = k * acc_x;
            }
        }
        return true;
    }
};

/**
 * @brief Solve M^T z = rhs in place by LU with partial pivoting.
 *
 * @param M   n x n row-major; overwritten.
 * @param rhs n doubles in, the solution z out.
 * @return false if M is numerically singular, in which case the fixed point is
 *         degenerate and no derivative exists.
 */
bool solve_transpose(double* M, double* rhs, int64_t n) {
    /* Transpose in place so the factorisation solves M^T z = rhs. */
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = i + 1; j < n; j++)
            std::swap(M[i * n + j], M[j * n + i]);

    for (int64_t col = 0; col < n; col++) {
        int64_t piv = col;
        double best = std::fabs(M[col * n + col]);
        for (int64_t r = col + 1; r < n; r++) {
            double v = std::fabs(M[r * n + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (!(best > 0.0) || !(best == best)) return false;
        if (piv != col) {
            for (int64_t j = 0; j < n; j++) std::swap(M[col * n + j], M[piv * n + j]);
            std::swap(rhs[col], rhs[piv]);
        }
        double d = M[col * n + col];
        for (int64_t r = col + 1; r < n; r++) {
            double f = M[r * n + col] / d;
            if (f == 0.0) continue;
            for (int64_t j = col; j < n; j++) M[r * n + j] -= f * M[col * n + j];
            rhs[r] -= f * rhs[col];
        }
    }
    /* Back substitution. */
    for (int64_t i = n - 1; i >= 0; i--) {
        double acc = rhs[i];
        for (int64_t j = i + 1; j < n; j++) acc -= M[i * n + j] * rhs[j];
        double d = M[i * n + i];
        if (d == 0.0) return false;
        rhs[i] = acc / d;
    }
    return true;
}

}  // namespace

/** @brief Backward pass for the weighted Fréchet (Karcher) mean, by implicit
 *  differentiation of the stationarity condition sum_i w_i log_mu(x_i) = 0 at
 *  the converged mean. Gated on the recomputed residual: at a non-stationary mu
 *  the implicit formulas return a plausible wrong gradient, so the rule refuses
 *  instead. See the block comment above for the derivation and node contract. */
extern "C" void tensor_frechet_mean_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;

    ad_node_t* pts_node = node->input1;
    ad_node_t* w_node   = node->input2;

    if (!pts_node || !pts_node->tensor_value) {
        eshkol_fatal("frechet-mean backward: node->input1 must carry the points "
                     "tensor [n_points, dim]; refusing to differentiate a fixed "
                     "point whose operands were not retained.");
        return;
    }
    if (!node->tensor_value) {
        eshkol_fatal("frechet-mean backward: node->tensor_value must carry the "
                     "converged mean mu*; the implicit derivative is only defined "
                     "at the fixed point, so it cannot be recovered here.");
        return;
    }

    const int64_t* p = (const int64_t*)&node->params;
    int64_t n_points = p[0];
    int64_t dim      = p[1];
    double  K        = double_from_bits(p[2]);
    double  tol      = double_from_bits(p[3]);

    if (n_points <= 0)
        n_points = (pts_node->ndim >= 1 && pts_node->shape) ? pts_node->shape[0] : 0;
    if (dim <= 0)
        dim = (node->ndim >= 1 && node->shape) ? node->shape[0]
            : ((pts_node->ndim >= 2 && pts_node->shape) ? pts_node->shape[1] : 0);
    if (!(tol > 0.0) || !(tol == tol)) tol = kFrechetResidualTol;

    if (n_points <= 0 || dim <= 0) {
        eshkol_fatal("frechet-mean backward: degenerate shape (n_points=%lld, "
                     "dim=%lld).", (long long)n_points, (long long)dim);
        return;
    }
    if (!(K <= 0.0) || !(K == K)) {
        eshkol_fatal("frechet-mean backward: sectional curvature must be <= 0 "
                     "(the Poincare ball has radius 1/sqrt(-K)); params slot 2 "
                     "holds %.17g.", K);
        return;
    }

    const double* mu   = (const double*)node->tensor_value;
    const double* pts  = (const double*)pts_node->tensor_value;
    const double* g    = (const double*)node->tensor_gradient;
    const double* wts  = (w_node && w_node->tensor_value)
                       ? (const double*)w_node->tensor_value : nullptr;

    /* ---- Euclidean limit (K = 0) -------------------------------------
     * mu = sum_i w_i x_i / sum_i w_i is linear in the points, so the exact
     * derivative is the closed form and no solve is needed. Routing it through
     * the hyperbolic path would divide by sqrt(c) = 0. */
    if (K == 0.0) {
        double wsum = 0.0;
        for (int64_t i = 0; i < n_points; i++) wsum += wts ? wts[i] : 1.0;
        if (!(wsum > 0.0)) {
            eshkol_fatal("frechet-mean backward: total weight is %.17g; the mean "
                         "is undefined and so is its derivative.", wsum);
            return;
        }
        size_t pts_total = (size_t)n_points * (size_t)dim;
        if (pts_node->tensor_gradient == NULL)
            pts_node->tensor_gradient = alloc_grad(pts_total);
        double* dpts = (double*)pts_node->tensor_gradient;
        if (dpts) {
            for (int64_t i = 0; i < n_points; i++) {
                double wi = (wts ? wts[i] : 1.0) / wsum;
                for (int64_t k = 0; k < dim; k++)
                    dpts[(size_t)(i * dim + k)] += wi * g[k];
            }
        }
        if (w_node) {
            if (w_node->tensor_gradient == NULL)
                w_node->tensor_gradient = alloc_grad((size_t)n_points);
            double* dw = (double*)w_node->tensor_gradient;
            if (dw) {
                for (int64_t i = 0; i < n_points; i++) {
                    /* d mu/d w_i = (x_i - mu)/wsum */
                    double acc = 0.0;
                    for (int64_t k = 0; k < dim; k++)
                        acc += g[k] * (pts[(size_t)(i * dim + k)] - mu[k]);
                    dw[i] += acc / wsum;
                }
            }
        }
        return;
    }

    /* ---- Hyperbolic case --------------------------------------------- */
    FrechetGeometry geo(K, dim);
    const double ball_radius = 1.0 / geo.s;

    /* mu* and every point must lie strictly inside the ball, or the log map has
     * no derivative there and (*) is not the condition being solved. */
    double mu_norm = std::sqrt(FrechetGeometry::dot(mu, mu, dim));
    if (!(mu_norm < ball_radius)) {
        eshkol_fatal("frechet-mean backward: |mu*| = %.17g is not strictly inside "
                     "the Poincare ball of radius %.17g (curvature K = %.17g); "
                     "the log map is not differentiable there.",
                     mu_norm, ball_radius, K);
        return;
    }

    std::vector<double> A((size_t)(dim * dim), 0.0);   /* dF/dmu             */
    std::vector<double> resid((size_t)dim, 0.0);       /* F(mu*) = sum w log */
    std::vector<double> logs((size_t)(n_points * dim), 0.0);
    std::vector<double> dlogdx((size_t)(n_points * dim * dim), 0.0);

    double wsum = 0.0, max_log = 0.0;
    for (int64_t i = 0; i < n_points; i++) {
        const double* xi = pts + (size_t)i * (size_t)dim;
        double xn = std::sqrt(FrechetGeometry::dot(xi, xi, dim));
        if (!(xn < ball_radius)) {
            eshkol_fatal("frechet-mean backward: point %lld has |x| = %.17g, not "
                         "strictly inside the Poincare ball of radius %.17g "
                         "(curvature K = %.17g).",
                         (long long)i, xn, ball_radius, K);
            return;
        }
        if (!geo.log_map_with_jacobians(mu, xi)) {
            eshkol_fatal("frechet-mean backward: log_mu(x_%lld) is outside its "
                         "differentiable domain; refusing to substitute a limit "
                         "for a derivative that does not exist.", (long long)i);
            return;
        }
        double wi = wts ? wts[i] : 1.0;
        wsum += wi;
        for (int64_t k = 0; k < dim; k++) {
            double lv = geo.logv[(size_t)k];
            logs[(size_t)(i * dim + k)] = lv;
            resid[(size_t)k] += wi * lv;
            double a = std::fabs(lv);
            if (a > max_log) max_log = a;
        }
        for (int64_t t = 0; t < dim * dim; t++) {
            A[(size_t)t] += wi * geo.dlog_dmu[(size_t)t];
            dlogdx[(size_t)(i * dim * dim + t)] = geo.dlog_dx[(size_t)t];
        }
    }

    if (!(wsum > 0.0)) {
        eshkol_fatal("frechet-mean backward: total weight is %.17g; the Frechet "
                     "mean is undefined and so is its derivative.", wsum);
        return;
    }

    /* ---- THE RESIDUAL GATE ------------------------------------------------
     * Everything below assumes F(mu*) = sum_i w_i log_mu*(x_i) = 0. If it does
     * not, the implicit function theorem does not apply and the solve returns a
     * smooth, plausible, wrong vector that nothing downstream can distinguish
     * from a gradient. Refuse.
     *
     * The bar is relative with an absolute floor: the residual is a sum of
     * tangent vectors scaled by the weights, so it is compared against wsum
     * times (1 + the largest individual |log_mu(x_i)|) — the natural scale of
     * the terms being cancelled. A purely absolute bar would be vacuous for
     * tightly clustered points and unsatisfiable for widely spread ones; a
     * purely relative one divides by zero in the most exact case available,
     * where every point coincides with the mean and each log is zero to
     * rounding. The 1 + |.| denominator is the convention the gradient-oracle
     * comparison helpers already use.
     *
     * BOTH TERMS ARE MEASURED IN RIEMANNIAN UNITS. The logs above are ambient
     * ball coordinates; the tangent space at mu carries the conformal metric
     * lambda_mu^2 <.,.> with lambda_mu = 2/(1 - c|mu|^2), so the invariant
     * length of a tangent vector v is lambda_mu |v|_2. The common factor
     * cancels out of the relative term, but NOT out of the floor — and the
     * floor is the whole reason the bar is not purely relative. In ambient
     * coordinates lambda_mu diverges at the boundary, every |log| collapses
     * toward zero, the floor swamps the relative term, and the bar degenerates
     * to |resid|_ambient <= tol * wsum, which a mu wrong by a whole unit of
     * hyperbolic distance satisfies comfortably. That is not hypothetical: with
     * the ambient scale the forward iteration accepted means wrong by 8.8e-8
     * and 7.6e-6 as converged, and this rule would then have differentiated
     * them and returned exactly the plausible wrong gradient the gate exists to
     * prevent. In Riemannian units the 1 is one unit of hyperbolic distance, so
     * the floor still protects the coincident-point case while the relative
     * term stays live near the boundary. vm_frechet_mean_compute in
     * lib/backend/vm_geometric.c applies the identical scale at the identical
     * default tolerance: the forward's gate is what makes this gate satisfiable,
     * so the two must not drift apart. */
    double resid_norm = std::sqrt(FrechetGeometry::dot(resid.data(), resid.data(), dim));
    double lambda = 2.0 / (1.0 - geo.c * FrechetGeometry::dot(mu, mu, dim));
    double resid_scale = wsum * (1.0 + lambda * max_log);
    double resid_rel = (lambda * resid_norm) / resid_scale;
    if (!(resid_rel <= tol)) {
        eshkol_fatal("frechet-mean backward: the retained mean is NOT a converged "
                     "stationary point — the stationarity residual "
                     "|sum_i w_i log_mu(x_i)| is %.6e (relative %.6e, tolerance "
                     "%.6e) over %lld points in dimension %lld. The implicit "
                     "derivative is only the derivative of the Frechet mean AT "
                     "the fixed point; away from it these formulas return a "
                     "plausible but wrong gradient, so this refuses rather than "
                     "reporting one. Tighten the forward iteration's convergence "
                     "or raise params slot 3 deliberately.",
                     resid_norm, resid_rel, tol,
                     (long long)n_points, (long long)dim);
        return;
    }

    /* ---- One solve: A^T z = g --------------------------------------- */
    std::vector<double> z(g, g + (size_t)dim);
    std::vector<double> Afac(A);
    if (!solve_transpose(Afac.data(), z.data(), dim)) {
        eshkol_fatal("frechet-mean backward: dF/dmu is numerically singular at "
                     "the fixed point, so the implicit function theorem gives no "
                     "unique derivative there (degenerate weights, or coincident "
                     "points in dimension %lld).", (long long)dim);
        return;
    }

    /* ---- dL/dx_j = -w_j (d log_mu(x_j)/d x_j)^T z ------------------- */
    size_t pts_total = (size_t)n_points * (size_t)dim;
    if (pts_node->tensor_gradient == NULL)
        pts_node->tensor_gradient = alloc_grad(pts_total);
    double* dpts = (double*)pts_node->tensor_gradient;
    if (dpts) {
        for (int64_t i = 0; i < n_points; i++) {
            double wi = wts ? wts[i] : 1.0;
            const double* J = &dlogdx[(size_t)(i * dim * dim)];
            for (int64_t j = 0; j < dim; j++) {
                double acc = 0.0;
                for (int64_t m = 0; m < dim; m++)
                    acc += J[(size_t)(m * dim + j)] * z[(size_t)m];
                dpts[(size_t)(i * dim + j)] += -wi * acc;
            }
        }
    }

    /* ---- dL/dw_j = -<log_mu(x_j), z> -------------------------------- */
    if (w_node) {
        if (w_node->tensor_gradient == NULL)
            w_node->tensor_gradient = alloc_grad((size_t)n_points);
        double* dw = (double*)w_node->tensor_gradient;
        if (dw) {
            for (int64_t i = 0; i < n_points; i++) {
                double acc = 0.0;
                for (int64_t k = 0; k < dim; k++)
                    acc += logs[(size_t)(i * dim + k)] * z[(size_t)k];
                dw[i] += -acc;
            }
        }
    }
}

/*******************************************************************************
 * Attention Backward — exact adjoint of multi-head scaled dot-product
 * attention, causal and non-causal (SW-12)
 *
 * FORWARD (ad_tensor_attention, lib/bridge/qllm_bridge.cpp). Q, K, V are
 * [batch, seq, dim] with dim = num_heads * head_dim, so head h of batch b owns
 * the column slice [h*head_dim, (h+1)*head_dim) of every row. Per (b, h):
 *
 *     S[i][j] = scale * <Q[b,i,h], K[b,j,h]>,   scale = 1/sqrt(head_dim)
 *     A[i][j] = softmax_j(S[i][j])              (row-wise; causal => j <= i)
 *     O[b,i,h] = sum_j A[i][j] * V[b,j,h]
 *
 * BACKWARD. Each (b, h) pair is an independent single-head attention — the
 * heads never mix in the forward, so their adjoints never mix either, and the
 * whole rule is that 2-D adjoint applied on each column slice:
 *
 *     dV[j] = sum_i A[i][j] * dO[i]                    (seq x head_dim)
 *     dA[i][j] = <dO[i], V[j]>                         (seq x seq)
 *     dS[i][j] = A[i][j] * (dA[i][j] - sum_k A[i][k] dA[i][k]) * scale
 *     dQ[i] = sum_j dS[i][j] * K[j]
 *     dK[j] = sum_i dS[i][j] * Q[i]
 *
 * The dS line IS the softmax Jacobian: for a row y = softmax(s),
 * dL/ds = J^T dL/dy with J = diag(y) - y y^T, i.e.
 * dL/ds_j = y_j (dL/dy_j - sum_k y_k dL/dy_k). The row dot product is computed
 * once per row, which is what makes the rule O(seq^2) rather than O(seq^3).
 *
 * CAUSAL MASKING is handled entirely by the retained weights. The forward
 * leaves A[i][j] at EXACTLY zero for j > i, and every term above carries a
 * factor A[i][j]: dV and dA vanish there, the row dot product picks up nothing
 * from the masked tail, and dS[i][j] = 0 * (...) = 0 kills the dQ/dK
 * contributions. So the masked positions contribute exactly zero without a
 * single mask test, and the summations still run over the full [0, seq) range
 * — which keeps the accumulation ORDER identical between the causal and
 * non-causal cases and identical to the non-bridge kernel's
 * (eshkol_backward_attention, lib/backend/tensor_backward.cpp), as the
 * bit-for-bit determinism rule requires. Skipping the masked j instead would
 * change nothing numerically (x + 0.0 == x for every finite x) but would make
 * the two implementations' orders diverge for no gain.
 *
 * NODE CONTRACT
 *   node->input1/2/3        Q, K, V nodes, each [batch, seq, dim]
 *   node->shape/ndim        output shape, [batch, seq, dim]
 *   node->tensor_gradient   upstream dL/dO, batch*seq*dim doubles
 *   node->saved_tensors[0]  retained attention weights A, dense
 *                           [batch, num_heads, seq, seq], masked entries zero
 *   node->num_saved         >= 1
 *   node->params as int64[6]  [num_heads, head_dim, causal, scale_bits, 0, 0]
 *
 * This rule is written independently of eshkol_backward_attention rather than
 * delegating to it: the two implementations are the two sides of the SW-12
 * differential check, and a delegation would make that check compare a thing
 * to itself.
 ******************************************************************************/

/** @brief Backward pass for the qLLM-bridge attention node: the exact adjoint
 *  of multi-head scaled dot-product attention (dQ, dK, dV through the softmax
 *  Jacobian), with causal masking carried by the forward's retained weights.
 *  See the block comment above for the derivation and the node contract. */
extern "C" void tensor_attention_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;

    ad_node_t* q_node = node->input1;
    ad_node_t* k_node = node->input2;
    ad_node_t* v_node = node->input3;

    /* HARD CONSTRAINT (exact AD or an explicit error, never a silent/plausible
     * zero). Before this rule existed the node carried neither the attention
     * weights nor the causal flag, and the backward could only refuse. Both are
     * now part of the contract, so a node arriving without them means the
     * PRODUCER did not follow it — refusing here is what stops a mis-wired
     * forward from training on a gradient reconstructed from guesses. */
    if (!q_node || !k_node || !v_node) {
        eshkol_fatal("attention backward: node->input1/2/3 must carry the Q, K "
                     "and V operands (SW-12 contract: inputs [batch, seq, dim], "
                     "saved_tensors[0] = attention weights "
                     "[batch, num_heads, seq, seq], params = "
                     "[num_heads, head_dim, causal, scale_bits]); refusing to "
                     "invent an operand.");
        return;
    }
    if (!node->saved_tensors || node->num_saved < 1 || !node->saved_tensors[0]) {
        eshkol_fatal("attention backward: node->saved_tensors[0] must carry the "
                     "softmax attention weights retained by the forward "
                     "(ad_tensor_attention). Recomputing them here would have to "
                     "re-derive the causal mask and the softmax shift, and any "
                     "drift from the forward is a silently-wrong gradient; "
                     "refusing instead.");
        return;
    }

    const int64_t* p = (const int64_t*)&node->params;
    int64_t num_heads = p[0];
    int64_t head_dim  = p[1];
    int64_t causal    = p[2];
    double scale;
    std::memcpy(&scale, &p[3], sizeof scale);

    if (node->ndim != 3 || !node->shape) {
        eshkol_fatal("attention backward: expected a [batch, seq, dim] node "
                     "(got %zu-D); the bridge forward records exactly that "
                     "shape.", node->ndim);
        return;
    }
    int64_t batch = node->shape[0];
    int64_t seq   = node->shape[1];
    int64_t dim   = node->shape[2];

    if (num_heads <= 0 || head_dim <= 0 || num_heads * head_dim != dim) {
        eshkol_fatal("attention backward: params [num_heads=%lld, head_dim=%lld] "
                     "do not partition dim=%lld; the forward must record the "
                     "head split it actually used.",
                     (long long)num_heads, (long long)head_dim, (long long)dim);
        return;
    }
    if (batch <= 0 || seq <= 0) {
        eshkol_fatal("attention backward: degenerate shape (batch=%lld, seq=%lld).",
                     (long long)batch, (long long)seq);
        return;
    }
    /* scale is bit-cast out of the params; a zero/non-finite slot means the
     * producer never wrote it, and silently substituting 1/sqrt(head_dim) would
     * hide a forward that used something else. */
    if (!(scale > 0.0) || scale != scale || scale > 1e300) {
        eshkol_fatal("attention backward: params[3] holds %.17g, not a usable "
                     "softmax scale; the forward must bit-cast its scale into "
                     "that slot.", scale);
        return;
    }
    const double* dO = (const double*)node->tensor_gradient;
    const double* A  = (const double*)node->saved_tensors[0];
    const double* Q  = (const double*)q_node->tensor_value;
    const double* K  = (const double*)k_node->tensor_value;
    const double* V  = (const double*)v_node->tensor_value;
    if (!Q || !K || !V) {
        eshkol_fatal("attention backward: Q/K/V nodes must still carry their "
                     "forward values; refusing to differentiate against a "
                     "released operand.");
        return;
    }

    /* Q, K and V are indexed with the OUTPUT's strides, the same way the
     * forward indexed them, so an operand of a different shape would be read
     * and written out of bounds. The forward already refuses that; check it
     * here too, because this rule also has to hold for nodes it did not
     * build. */
    size_t total = (size_t)(batch * seq * dim);
    const ad_node_t* operands[3] = { q_node, k_node, v_node };
    const char* operand_names[3] = { "Q", "K", "V" };
    for (int idx = 0; idx < 3; idx++) {
        size_t have = (operands[idx]->shape && operands[idx]->ndim > 0)
                    ? tensor_size(operands[idx]->shape, operands[idx]->ndim) : 0;
        if (have != total) {
            eshkol_fatal("attention backward: operand %s holds %zu elements but "
                         "the output shape [%lld, %lld, %lld] needs %zu; every "
                         "operand is indexed with the output's strides.",
                         operand_names[idx], have,
                         (long long)batch, (long long)seq, (long long)dim, total);
            return;
        }
    }

    if (q_node->tensor_gradient == NULL) q_node->tensor_gradient = alloc_grad(total);
    if (k_node->tensor_gradient == NULL) k_node->tensor_gradient = alloc_grad(total);
    if (v_node->tensor_gradient == NULL) v_node->tensor_gradient = alloc_grad(total);
    double* dQ = (double*)q_node->tensor_gradient;
    double* dK = (double*)k_node->tensor_gradient;
    double* dV = (double*)v_node->tensor_gradient;
    if (!dQ || !dK || !dV) return;

    /* The causal flag and the retained weights are two records of the same
     * fact, and the adjoint trusts the weights. If they disagree — a forward
     * that masked differently from what it recorded, or params written by a
     * different producer — every masked position would silently pick up a
     * gradient it must not have. Cross-check them instead: under a causal mask
     * the forward leaves j > i at EXACTLY zero, so this is an equality test,
     * not a tolerance. */
    if (causal) {
        for (int64_t bh = 0; bh < batch * num_heads; bh++) {
            const double* Ah = &A[(size_t)(bh * seq * seq)];
            for (int64_t i = 0; i < seq; i++)
                for (int64_t j = i + 1; j < seq; j++)
                    if (Ah[i * seq + j] != 0.0) {
                        eshkol_fatal("attention backward: params say causal but "
                                     "the retained weight A[%lld][%lld][%lld] is "
                                     "%.17g, not zero; the mask the forward ran "
                                     "and the mask it recorded disagree.",
                                     (long long)bh, (long long)i, (long long)j,
                                     Ah[i * seq + j]);
                        return;
                    }
        }
    }

    /* Per-head scratch. Plain std::vector, not the arena: the dispatcher runs
     * this rule inside a scope it pops on return, and the persistent gradient
     * buffers above must not share a lifetime with these. (Same reasoning as
     * FrechetGeometry earlier in this file.) */
    std::vector<double> dA((size_t)(seq * seq));
    std::vector<double> dS((size_t)(seq * seq));

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t h = 0; h < num_heads; h++) {
            const int64_t off = h * head_dim;
            const double* Ah = &A[(size_t)(((b * num_heads) + h) * seq * seq)];

            /* dV[j][d] = sum_i A[i][j] * dO[i][d] — j-major, matching the
             * non-bridge kernel's loop nest. */
            for (int64_t j = 0; j < seq; j++) {
                double* dVj = &dV[(size_t)((b * seq + j) * dim + off)];
                for (int64_t d = 0; d < head_dim; d++) {
                    double acc = 0.0;
                    for (int64_t i = 0; i < seq; i++)
                        acc += Ah[i * seq + j] * dO[(b * seq + i) * dim + off + d];
                    dVj[d] += acc;
                }
            }

            /* dA[i][j] = <dO[i], V[j]> */
            for (int64_t i = 0; i < seq; i++) {
                const double* dOi = &dO[(size_t)((b * seq + i) * dim + off)];
                for (int64_t j = 0; j < seq; j++) {
                    const double* Vj = &V[(size_t)((b * seq + j) * dim + off)];
                    double acc = 0.0;
                    for (int64_t d = 0; d < head_dim; d++) acc += dOi[d] * Vj[d];
                    dA[(size_t)(i * seq + j)] = acc;
                }
            }

            /* Softmax Jacobian, one row dot product per row. */
            for (int64_t i = 0; i < seq; i++) {
                double dot = 0.0;
                for (int64_t j = 0; j < seq; j++)
                    dot += Ah[i * seq + j] * dA[(size_t)(i * seq + j)];
                for (int64_t j = 0; j < seq; j++)
                    dS[(size_t)(i * seq + j)] =
                        Ah[i * seq + j] * (dA[(size_t)(i * seq + j)] - dot) * scale;
            }

            /* dQ[i][d] = sum_j dS[i][j] * K[j][d] */
            for (int64_t i = 0; i < seq; i++) {
                double* dQi = &dQ[(size_t)((b * seq + i) * dim + off)];
                for (int64_t d = 0; d < head_dim; d++) {
                    double acc = 0.0;
                    for (int64_t j = 0; j < seq; j++)
                        acc += dS[(size_t)(i * seq + j)] * K[(b * seq + j) * dim + off + d];
                    dQi[d] += acc;
                }
            }

            /* dK[j][d] = sum_i dS[i][j] * Q[i][d] */
            for (int64_t j = 0; j < seq; j++) {
                double* dKj = &dK[(size_t)((b * seq + j) * dim + off)];
                for (int64_t d = 0; d < head_dim; d++) {
                    double acc = 0.0;
                    for (int64_t i = 0; i < seq; i++)
                        acc += dS[(size_t)(i * seq + j)] * Q[(b * seq + i) * dim + off + d];
                    dKj[d] += acc;
                }
            }
        }
    }
}


/*******************************************************************************
 * Geometric bridge backwards (SW-65)
 *
 * ad_hyperbolic_distance, ad_poincare_exp_map, ad_poincare_log_map and
 * ad_geodesic_attention (lib/bridge/qllm_bridge.cpp) all record TENSOR-VALUED
 * nodes — types 33, 34, 35 and 37 — and none of them had a backward. They were
 * not refused and they did not warn: the node types sit in the 33-66 band that
 * eshkol_tensor_backward_dispatch treats as "scalar ops differentiated by
 * codegen", so they fell into its default: and the reverse sweep completed
 * having propagated exactly nothing. Every input gradient came back 0.0 with no
 * diagnostic, which a caller cannot tell from a genuine zero.
 *
 * The rules below are exact. The geometric rules call directional evaluators
 * in riemannian_core.h, so their adjoints are contractions of the shared
 * forward formulas rather than separately written square-based copies.
 *
 * WHERE THESE REFUSE, AND WHY IT IS NOT CONSERVATISM. The Riemannian distance
 * d(x, y) behaves like |x - y| near coincidence: it has no derivative at x = y,
 * only a subgradient set. Any number returned there is invented. Both
 * distance-based rules therefore refuse at exact coincidence instead of picking
 * a plausible member of that set. For geodesic attention this has a consequence
 * worth stating outright: scoring by distance makes the op non-differentiable
 * whenever a query row coincides exactly with a key row, which is the ordinary
 * case when Q and K are the same tensor. That is a property of distance
 * scoring, not of this implementation, and it is precisely the kind of fact a
 * silent zero would have hidden.
 ******************************************************************************/

namespace {

/** @brief Upstream cotangent of a tensor-valued node.
 *
 *  `tensor_gradient` is the canonical channel — every rule in this file
 *  accumulates into it — so it is what a downstream op will have written. The
 *  scalar `gradient` mirror is the fallback for the loss-node convention used
 *  by tensor_cross_entropy_backward, which seeds only the scalar. */
double scalar_upstream(const ad_node_t* node) {
    if (node->tensor_gradient) return ((const double*)node->tensor_gradient)[0];
    return node->gradient;
}

/** @brief Gradient of the Poincare-ball distance d(x, y) with respect to both
 *         arguments, for the ball of curvature -c.
 *
 *      d      = acosh(arg)/sqrt(c),   arg = 1 + 2c|x-y|^2/(dx dy)
 *      dx     = 1 - c|x|^2,           dy  = 1 - c|y|^2
 *
 *  d(acosh)/d(arg) = 1/sqrt(arg^2 - 1), and arg depends on x both through the
 *  numerator |x-y|^2 and through dx in the denominator; both terms are kept.
 *
 *  @param gx  out, n doubles: d d / d x (may be NULL)
 *  @param gy  out, n doubles: d d / d y (may be NULL)
 *  @return false when the two points coincide (arg <= 1), where the distance
 *          has no derivative, or when either point is outside the ball.
 */
bool hyperbolic_distance_grad(const double* x, const double* y, double c,
                              int64_t n, double* gx, double* gy,
                              double* dist_out) {
    if (!(c > 0.0) || !std::isfinite(c) || n <= 0) return false;
    std::vector<double> zero((size_t)n, 0.0), seed((size_t)n, 0.0);
    for (int64_t j = 0; j < n; ++j) {
        seed[(size_t)j] = 1.0;
        double distance = 0.0, tangent = 0.0;
        if (!eshkol_rm_distance_directional(x, y, seed.data(), zero.data(),
                                            -c, (int)n, &distance, &tangent))
            return false;
        if (gx) gx[j] = tangent;
        if (dist_out && j == 0) *dist_out = distance;
        seed[(size_t)j] = 0.0;
    }
    if (gy) {
        for (int64_t j = 0; j < n; ++j) {
            seed[(size_t)j] = 1.0;
            double distance = 0.0, tangent = 0.0;
            if (!eshkol_rm_distance_directional(x, y, zero.data(), seed.data(),
                                                -c, (int)n, &distance, &tangent))
                return false;
            gy[j] = tangent;
            if (dist_out && j == 0 && !gx) *dist_out = distance;
            seed[(size_t)j] = 0.0;
        }
    }
    return true;
}

}  // namespace

/** @brief Backward for the Poincare-ball distance. Scalar output, two point
 *  inputs; refuses at coincident points, where no derivative exists. */
extern "C" void tensor_hyperbolic_distance_backward(ad_node_t* node) {
    if (!node) return;
    ad_node_t* xn = node->input1;
    ad_node_t* yn = node->input2;
    if (!xn || !yn || !xn->tensor_value || !yn->tensor_value) {
        eshkol_fatal("hyperbolic-distance backward: node->input1 and input2 must "
                     "carry the two points; refusing to differentiate a distance "
                     "whose operands were not retained.");
        return;
    }
    const double g = scalar_upstream(node);
    if (g == 0.0) return;

    const int64_t n = (int64_t)tensor_size(xn->shape, xn->ndim);
    const double c = node->params.curvature;
    if (!(c > 0.0)) {
        eshkol_fatal("hyperbolic-distance backward: curvature parameter is %.17g; "
                     "the forward stores |curvature| and it must be positive.", c);
        return;
    }

    std::vector<double> gx((size_t)n), gy((size_t)n);
    if (!hyperbolic_distance_grad((const double*)xn->tensor_value,
                                  (const double*)yn->tensor_value,
                                  c, n, gx.data(), gy.data(), nullptr)) {
        eshkol_fatal("hyperbolic-distance backward: the two points coincide (or "
                     "one is outside the Poincare ball), where the distance has "
                     "no derivative — it is a cone point, and every value a rule "
                     "could return there is invented. Refusing.");
        return;
    }

    if (!xn->tensor_gradient) xn->tensor_gradient = alloc_grad((size_t)n);
    if (!yn->tensor_gradient) yn->tensor_gradient = alloc_grad((size_t)n);
    double* dX = (double*)xn->tensor_gradient;
    double* dY = (double*)yn->tensor_gradient;
    if (!dX || !dY) return;
    for (int64_t i = 0; i < n; i++) { dX[i] += g * gx[(size_t)i]; dY[i] += g * gy[(size_t)i]; }
}

/** @brief Backward for the Poincare exponential map exp_x(v).
 *
 *  The forward is out = x (+)_c (coef * v) with
 *      coef = tanh(sqrt(c) * lambda_x * |v| / 2) / (sqrt(c) * |v|),
 *      lambda_x = 2 / (1 - c|x|^2).
 *  The differential is evaluated by eshkol_rm_exp_directional, which runs the
 *  same scaled helpers and multiplication order as eshkol_rm_exp_map. A
 *  separately written Jacobian would reintroduce direct squares at exactly
 *  the exponent boundaries the forward path handles.
 *
 *  coef has a removable singularity at |v| = 0: it tends to lambda_x/2, and its
 *  v-derivative tends to 0. Below a threshold the series
 *      coef       = 1/dx - c|v|^2/(3 dx^3) + O(|v|^4)
 *      dcoef/dv_j = v_j * (-2c/(3 dx^3)) + O(|v|^2)
 *  is used instead of the quotient, which would be 0/0 there. Note this is the
 *  derivative of the MATHEMATICAL map, which is smooth at v = 0 — not the
 *  derivative of the forward's `|v| < 1e-15 -> return x` shortcut branch, whose
 *  v-derivative would be a (wrong) zero. Same distinction the sphere_retract
 *  finding in tests/qllm_oracle/README.md draws between a guard and the map. */
extern "C" void tensor_poincare_exp_map_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;
    ad_node_t* xn = node->input1;
    ad_node_t* vn_node = node->input2;
    if (!xn || !vn_node || !xn->tensor_value || !vn_node->tensor_value) {
        eshkol_fatal("poincare-exp-map backward: node->input1 (base point) and "
                     "input2 (tangent) must both be retained.");
        return;
    }
    const int64_t n = (int64_t)tensor_size(xn->shape, xn->ndim);
    const double c = node->params.curvature;
    if (n <= 0 || !(c > 0.0)) {
        eshkol_fatal("poincare-exp-map backward: degenerate dimension %lld or "
                     "curvature %.17g.", (long long)n, c);
        return;
    }
    const double* X = (const double*)xn->tensor_value;
    const double* V = (const double*)vn_node->tensor_value;
    const double* g = (const double*)node->tensor_gradient;

    if (!xn->tensor_gradient) xn->tensor_gradient = alloc_grad((size_t)n);
    if (!vn_node->tensor_gradient) vn_node->tensor_gradient = alloc_grad((size_t)n);
    double* dX = (double*)xn->tensor_gradient;
    double* dV = (double*)vn_node->tensor_gradient;
    if (!dX || !dV) return;

    bool zero_v = true;
    for (int64_t i = 0; i < n; ++i)
        if (V[i] != 0.0) { zero_v = false; break; }
    if (zero_v) {
        for (int64_t i = 0; i < n; ++i) {
            dX[i] += g[i];
            dV[i] += g[i];
        }
        return;
    }

    std::vector<double> zero((size_t)n, 0.0), seed((size_t)n, 0.0);
    std::vector<double> dout((size_t)n);
    for (int64_t j = 0; j < n; ++j) {
        seed[(size_t)j] = 1.0;
        if (!eshkol_rm_exp_directional(X, V, seed.data(), zero.data(),
                                       -c, (int)n, nullptr, dout.data())) {
            eshkol_fatal("poincare-exp-map backward: shared scaled forward could "
                         "not produce a finite directional derivative. Refusing.");
            return;
        }
        double acc = 0.0;
        for (int64_t i = 0; i < n; ++i) acc += g[i] * dout[(size_t)i];
        dX[j] += acc;
        seed[(size_t)j] = 0.0;
    }
    for (int64_t j = 0; j < n; ++j) {
        seed[(size_t)j] = 1.0;
        if (!eshkol_rm_exp_directional(X, V, zero.data(), seed.data(),
                                       -c, (int)n, nullptr, dout.data())) {
            eshkol_fatal("poincare-exp-map backward: shared scaled forward could "
                         "not produce a finite directional derivative. Refusing.");
            return;
        }
        double acc = 0.0;
        for (int64_t i = 0; i < n; ++i) acc += g[i] * dout[(size_t)i];
        dV[j] += acc;
        seed[(size_t)j] = 0.0;
    }
}

static bool stable_poincare_log_jacobians(
    const double* x, const double* y, double c, int64_t n,
    std::vector<double>* log_out, std::vector<double>* dlog_dx,
    std::vector<double>* dlog_dy) {
    if (!(c > 0.0) || !std::isfinite(c) || n <= 0) return false;
    const size_t count = (size_t)n;
    log_out->assign(count, 0.0);
    dlog_dx->assign(count * count, 0.0);
    dlog_dy->assign(count * count, 0.0);
    std::vector<double> zero(count, 0.0), seed(count, 0.0), values(count);
    std::vector<double> directional(count);
    for (int64_t j = 0; j < n; ++j) {
        seed[(size_t)j] = 1.0;
        if (!eshkol_rm_log_directional(x, y, seed.data(), zero.data(),
                                       -c, (int)n, values.data(),
                                       directional.data()))
            return false;
        if (j == 0) *log_out = values;
        for (int64_t i = 0; i < n; ++i)
            (*dlog_dx)[(size_t)i * count + (size_t)j] = directional[(size_t)i];
        seed[(size_t)j] = 0.0;
    }
    for (int64_t j = 0; j < n; ++j) {
        seed[(size_t)j] = 1.0;
        if (!eshkol_rm_log_directional(x, y, zero.data(), seed.data(),
                                       -c, (int)n, values.data(),
                                       directional.data()))
            return false;
        for (int64_t i = 0; i < n; ++i)
            (*dlog_dy)[(size_t)i * count + (size_t)j] = directional[(size_t)i];
        seed[(size_t)j] = 0.0;
    }
    return true;
}

static bool euclidean_distance_grad(const double* x, const double* y, int64_t n,
                                    double* gx, double* gy) {
    std::vector<double> zero((size_t)n, 0.0), seed((size_t)n, 0.0);
    for (int64_t j = 0; j < n; ++j) {
        seed[(size_t)j] = 1.0;
        double tangent = 0.0;
        if (!eshkol_rm_distance_directional(x, y, seed.data(), zero.data(),
                                            0.0, (int)n, nullptr, &tangent))
            return false;
        gx[j] = tangent;
        seed[(size_t)j] = 0.0;
    }
    for (int64_t j = 0; j < n; ++j) {
        seed[(size_t)j] = 1.0;
        double tangent = 0.0;
        if (!eshkol_rm_distance_directional(x, y, zero.data(), seed.data(),
                                            0.0, (int)n, nullptr, &tangent))
            return false;
        gy[j] = tangent;
        seed[(size_t)j] = 0.0;
    }
    return true;
}

static bool spherical_distance_grad(const double* x, const double* y, double K,
                                    int64_t n, double* gx, double* gy) {
    if (!(K > 0.0) || !std::isfinite(K)) return false;
    double R = 1.0 / std::sqrt(K);
    double cs = eshkol_rm_scaled_dot_factor(x, y, K, (int)n);
    double sn2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double t = y[i] - cs * x[i];
        sn2 += t * t;
    }
    double sn = std::sqrt(sn2) / R;
    if (!(sn > 0.0) || !std::isfinite(sn)) return false;
    for (int64_t i = 0; i < n; ++i) {
        gx[i] = (cs * x[i] - y[i]) / (R * sn);
        gy[i] = (cs * y[i] - x[i]) / (R * sn);
    }
    return true;
}

/** @brief Backward for the Poincare logarithmic map log_x(y).
 *
 *  log_x(y) = k(x) * phi(|u|) * u with u = (-x) (+)_c y, k = (1 - c|x|^2)/sqrt(c)
 *  and phi(r) = artanh(sqrt(c) r)/r — which is exactly the map
 *  FrechetGeometry::log_map_with_jacobians already differentiates for the
 *  Frechet rule, including the r -> 0 limit. Reusing it is the point: this is
 *  the same function, and a second derivation of it could only introduce a
 *  disagreement.
 *
 *  It returns false when sqrt(c)|u| >= 1, i.e. no finite log exists. The
 *  forward clamps there (`t >= 1 -> t = 1 - 1e-12`) and returns a value, but
 *  that value is fabricated — beyond the boundary the log map has no value to
 *  differentiate — so the rule refuses rather than differentiate the clamp. */
extern "C" void tensor_poincare_log_map_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;
    ad_node_t* xn = node->input1;
    ad_node_t* yn = node->input2;
    if (!xn || !yn || !xn->tensor_value || !yn->tensor_value) {
        eshkol_fatal("poincare-log-map backward: node->input1 and input2 must "
                     "both be retained.");
        return;
    }
    const int64_t n = (int64_t)tensor_size(xn->shape, xn->ndim);
    const double c = node->params.curvature;
    if (n <= 0 || !(c > 0.0)) {
        eshkol_fatal("poincare-log-map backward: degenerate dimension %lld or "
                     "curvature %.17g.", (long long)n, c);
        return;
    }
    const double* X = (const double*)xn->tensor_value;
    const double* Y = (const double*)yn->tensor_value;
    const double* g = (const double*)node->tensor_gradient;

    std::vector<double> logv, dlog_dx, dlog_dy;
    if (!stable_poincare_log_jacobians(X, Y, c, n, &logv, &dlog_dx, &dlog_dy)) {
        eshkol_fatal("poincare-log-map backward: the shared-core log could not "
                     "produce a finite Jacobian for these operands. Refusing.");
        return;
    }

    if (!xn->tensor_gradient) xn->tensor_gradient = alloc_grad((size_t)n);
    if (!yn->tensor_gradient) yn->tensor_gradient = alloc_grad((size_t)n);
    double* dX = (double*)xn->tensor_gradient;
    double* dY = (double*)yn->tensor_gradient;
    if (!dX || !dY) return;

    for (int64_t j = 0; j < n; j++) {
        double ax = 0.0, ay = 0.0;
        for (int64_t i = 0; i < n; i++) {
            ax += g[i] * dlog_dx[(size_t)(i * n + j)];
            ay += g[i] * dlog_dy[(size_t)(i * n + j)];
        }
        dX[j] += ax;
        dY[j] += ay;
    }
}

/** @brief Backward for geodesic attention: softmax over negated geodesic
 *  distances, composed with the value-weighted sum.
 *
 *      s_ij = -scale * d(Q_i, K_j),  p = softmax_j(s),  O_i = sum_j p_ij V_j
 *
 *  The chain is the standard attention one with the dot-product score replaced
 *  by the distance score, so only ds/dQ and ds/dK differ; those come from
 *  hyperbolic_distance_grad, the same routine the distance rule uses.
 *
 *  The softmax weights are read from node->saved_tensors[0], retained by the
 *  forward. Recomputing them here was the alternative and is rejected for the
 *  reason SW-12 records for ad_tensor_attention: it would have to re-derive the
 *  max-shift and the causal mask, and any drift between the two copies is
 *  exactly the silently-wrong-gradient class this rule exists to close. */
extern "C" void tensor_geodesic_attention_backward(ad_node_t* node) {
    if (!node || !node->tensor_gradient) return;
    ad_node_t* qn = node->input1;
    ad_node_t* kn = node->input2;
    ad_node_t* vn_node = node->input3;
    if (!qn || !kn || !vn_node || !qn->tensor_value || !kn->tensor_value
        || !vn_node->tensor_value) {
        eshkol_fatal("geodesic-attention backward: Q, K and V must all be "
                     "retained on the node (input1/input2/input3).");
        return;
    }
    if (!node->saved_tensors || node->num_saved < 1 || !node->saved_tensors[0]) {
        eshkol_fatal("geodesic-attention backward: the forward did not retain the "
                     "softmax weights (saved_tensors[0]). Recomputing them here "
                     "would duplicate the forward's max-shift and mask, and any "
                     "drift between the copies is the silently-wrong-gradient "
                     "class this rule exists to close. Refusing.");
        return;
    }
    if (qn->ndim != 3 || kn->ndim != 3 || vn_node->ndim != 3) {
        eshkol_fatal("geodesic-attention backward: expected [batch, seq, dim] "
                     "operands, got rank %zu.", qn->ndim);
        return;
    }
    for (size_t d = 0; d < 3; ++d) {
        if (qn->shape[d] != kn->shape[d] || qn->shape[d] != vn_node->shape[d]) {
            eshkol_fatal("geodesic-attention backward: Q, K and V shapes must match.");
            return;
        }
    }

    const size_t batch = (size_t)qn->shape[0];
    const size_t seq   = (size_t)qn->shape[1];
    const size_t dim   = (size_t)qn->shape[2];
    const int64_t* p6  = (const int64_t*)&node->params;
    const int     heads    = (int)p6[0];
    const size_t  head_dim = (size_t)p6[1];
    const bool    causal   = (p6[2] != 0);
    const double  curvature = double_from_bits(p6[3]);
    if (heads <= 0 || head_dim == 0 || dim != (size_t)heads * head_dim ||
        !std::isfinite(curvature)) {
        eshkol_fatal("geodesic-attention backward: inconsistent params "
                     "(heads=%d head_dim=%zu dim=%zu curvature=%.17g).",
                     heads, head_dim, dim, curvature);
        return;
    }
    const double metric_scale = curvature < 0.0 ? std::sqrt(-curvature) : 1.0;
    const double scale = 1.0 / (metric_scale * std::sqrt((double)head_dim));

    const double* Q = (const double*)qn->tensor_value;
    const double* K = (const double*)kn->tensor_value;
    const double* V = (const double*)vn_node->tensor_value;
    const double* A = (const double*)node->saved_tensors[0];   /* [b][h][i][j] */
    const double* G = (const double*)node->tensor_gradient;

    const size_t total = batch * seq * dim;
    if (!qn->tensor_gradient) qn->tensor_gradient = alloc_grad(total);
    if (!kn->tensor_gradient) kn->tensor_gradient = alloc_grad(total);
    if (!vn_node->tensor_gradient) vn_node->tensor_gradient = alloc_grad(total);
    double* dQ = (double*)qn->tensor_gradient;
    double* dK = (double*)kn->tensor_gradient;
    double* dV = (double*)vn_node->tensor_gradient;
    if (!dQ || !dK || !dV) return;

    std::vector<double> dp(seq), ds(seq), gq(head_dim), gk(head_dim);

    for (size_t b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            const size_t off = (size_t)h * head_dim;
            for (size_t i = 0; i < seq; ++i) {
                const size_t qi = (b * seq + i) * dim + off;
                const size_t limit = causal ? (i + 1) : seq;
                const double* arow =
                    &A[((b * (size_t)heads + (size_t)h) * seq + i) * seq];
                const double* grow = &G[(b * seq + i) * dim + off];

                /* dL/dV and dL/dp. */
                double pdot = 0.0;
                for (size_t j = 0; j < limit; ++j) {
                    const size_t vj = (b * seq + j) * dim + off;
                    double acc = 0.0;
                    for (size_t d = 0; d < head_dim; ++d) {
                        dV[vj + d] += arow[j] * grow[d];
                        acc += grow[d] * V[vj + d];
                    }
                    dp[j] = acc;
                    pdot += arow[j] * acc;
                }
                /* Softmax Jacobian: ds_j = p_j (dp_j - sum_m p_m dp_m). */
                for (size_t j = 0; j < limit; ++j)
                    ds[j] = arow[j] * (dp[j] - pdot);

                /* s_j = -scale * d(Q_i, K_j): push through the distance. */
                for (size_t j = 0; j < limit; ++j) {
                    if (ds[j] == 0.0) continue;
                    const size_t kj = (b * seq + j) * dim + off;
                    bool grad_ok = false;
                    if (curvature < 0.0) {
                        grad_ok = hyperbolic_distance_grad(
                            &Q[qi], &K[kj], -curvature, (int64_t)head_dim,
                            gq.data(), gk.data(), nullptr);
                    } else if (curvature == 0.0) {
                        grad_ok = euclidean_distance_grad(
                            &Q[qi], &K[kj], (int64_t)head_dim,
                            gq.data(), gk.data());
                    } else {
                        grad_ok = spherical_distance_grad(
                            &Q[qi], &K[kj], curvature, (int64_t)head_dim,
                            gq.data(), gk.data());
                    }
                    if (!grad_ok) {
                        eshkol_fatal(
                            "geodesic-attention backward: query row %zu and key "
                            "row %zu (batch %zu, head %d) coincide exactly, or "
                            "one lies outside the Poincare ball. The geodesic "
                            "distance has no derivative at coincident points — "
                            "it is a cone point — so scoring attention by "
                            "distance makes this op non-differentiable there. "
                            "This is the ordinary case when Q and K are the same "
                            "tensor. Refusing rather than inventing a "
                            "subgradient for sectional curvature K=%.17g.",
                            i, j, b, h, curvature);
                        return;
                    }
                    const double f = -scale * ds[j];
                    for (size_t d = 0; d < head_dim; ++d) {
                        dQ[qi + d] += f * gq[d];
                        dK[kj + d] += f * gk[d];
                    }
                }
            }
        }
    }
}

/*******************************************************************************
 * Backward Dispatch Table
 ******************************************************************************/

typedef void (*backward_fn_t)(ad_node_t*);

/* The backward-function table, GENERATED from inc/eshkol/ad_node_registry.def.
 *
 * This replaces a hand-written switch with a `default:` that returned NULL and
 * warned once.  A default cannot tell "this node has no adjoint by design" from
 * "someone added a node type and forgot this table", so it answered both with
 * the same shrug; the caller's NULL handling then turned the second case into a
 * silently dropped gradient.  Now every row of the registry contributes exactly
 * one entry, and the disposition it declared decides what that entry is:
 *
 *   BRIDGE   -> the function the row names.  If that symbol does not exist,
 *               THIS FILE DOES NOT COMPILE.  That is what makes "registered"
 *               a fact rather than an intention: a registry row cannot claim a
 *               backward it does not have.
 *   anything -> nullptr, because the row said so — LEAF and SCALAR_ADJOINT
 *   else        genuinely have nothing here, CUSTOM_VJP is answered elsewhere,
 *               and UNREGISTERED is an explicit refusal the dispatcher turns
 *               into an abort rather than a zero.
 *
 * The array is sized AD_NODE_TYPE_COUNT and indexed by node type, which is
 * sound because eshkol.h statically asserts the registry's values are dense.
 */
#define ESHKOL_AD_NO_BRIDGE nullptr
#define ESHKOL_AD_BRIDGE_ENTRY_BRIDGE(FN)         FN
#define ESHKOL_AD_BRIDGE_ENTRY_LEAF(FN)           nullptr
#define ESHKOL_AD_BRIDGE_ENTRY_SCALAR_ADJOINT(FN) nullptr
#define ESHKOL_AD_BRIDGE_ENTRY_INLINE(FN)         nullptr
#define ESHKOL_AD_BRIDGE_ENTRY_CUSTOM_VJP(FN)     nullptr
#define ESHKOL_AD_BRIDGE_ENTRY_UNREGISTERED(FN)   nullptr

static backward_fn_t const kTensorBackwardTable[AD_NODE_TYPE_COUNT] = {
#define ESHKOL_AD_NODE(NAME, VALUE, PAYLOAD, TENSOR_BACKWARD, BRIDGE_FN) \
    ESHKOL_AD_BRIDGE_ENTRY_##TENSOR_BACKWARD(BRIDGE_FN),
#include "eshkol/ad_node_registry.def"
#undef ESHKOL_AD_NODE
};

/** @brief Look up the backward-pass function for an AD tensor node type.
 *
 *  Returns the registered function, or nullptr when the node's registry row
 *  declared no bridge backward.  nullptr is NOT a judgement about whether a
 *  gradient may be dropped — the caller decides that from the row's declared
 *  disposition — it only says this table does not answer this node type.
 *  Out-of-range values (a corrupt or foreign tape) return nullptr; the caller
 *  reports them.
 */
extern "C" backward_fn_t get_tensor_backward_fn(int node_type) {
    if (node_type < 0 || node_type >= AD_NODE_TYPE_COUNT) return nullptr;
    return kTensorBackwardTable[node_type];
}
