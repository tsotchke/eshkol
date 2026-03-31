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

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>

/* Include the Eshkol AD types — eshkol.h is a C++ header, no extern "C" needed */
#include "eshkol/eshkol.h"

/*******************************************************************************
 * Internal Helpers
 ******************************************************************************/

static size_t tensor_size(const int64_t* shape, size_t ndim) {
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= (size_t)shape[i];
    }
    return size;
}

/* Allocate zero-initialized gradient tensor */
static double* alloc_grad(size_t n) {
    double* g = (double*)calloc(n, sizeof(double));
    return g;
}

/* Accumulate gradient: dst += src */
static void accumulate_grad(double* dst, const double* src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] += src[i];
    }
}

/*******************************************************************************
 * MatMul Backward
 *
 * Forward: C = A @ B where A:[m,k], B:[k,n] → C:[m,n]
 * Backward:
 *   dL/dA = dL/dC @ B^T   →  [m,n] @ [n,k] = [m,k]
 *   dL/dB = A^T @ dL/dC   →  [k,m] @ [m,n] = [k,n]
 ******************************************************************************/

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
            dx_row[i] += inv_rms * (dy_g - xhat * sum_dy_x * inv_rms * inv_rms / N);

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
 * Backward Dispatch Table
 ******************************************************************************/

typedef void (*backward_fn_t)(ad_node_t*);

extern "C" backward_fn_t get_tensor_backward_fn(int node_type) {
    switch ((ad_node_type_t)node_type) {
        case AD_NODE_TENSOR_MATMUL:       return tensor_matmul_backward;
        case AD_NODE_TENSOR_SOFTMAX:      return tensor_softmax_backward;
        case AD_NODE_TENSOR_LAYERNORM:    return tensor_layernorm_backward;
        case AD_NODE_TENSOR_RMSNORM:      return tensor_rmsnorm_backward;
        case AD_NODE_TENSOR_GELU:         return tensor_gelu_backward;
        case AD_NODE_TENSOR_SILU:         return tensor_silu_backward;
        case AD_NODE_TENSOR_CROSS_ENTROPY: return tensor_cross_entropy_backward;
        default:                          return NULL;
    }
}
