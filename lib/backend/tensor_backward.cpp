/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Tensor Backward Pass Runtime Functions
 *
 * Implements backward passes for all tensor AD operations. Called from
 * LLVM-generated code during reverse-mode automatic differentiation.
 *
 * Each function follows the PyTorch autograd convention:
 * - grad_out: upstream gradient (from the operation's output)
 * - saved_*: tensors saved during the forward pass
 * - grad_*: output gradient tensors (pre-allocated by caller)
 */

#include <eshkol/backend/tensor_backward.h>
#include <eshkol/eshkol.h>
#include <cstring>
#include <cmath>
#include <cstdlib>

/* ===== Structural Operations ===== */

extern "C" void eshkol_backward_transpose(
    const double* grad_out, double* grad_in,
    int64_t rows, int64_t cols)
{
    /* Backward of transpose is transpose:
     * If forward: out[i][j] = in[j][i], shape (rows, cols)
     * Then backward: grad_in[j][i] = grad_out[i][j]
     * grad_in has shape (cols, rows) */
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            grad_in[j * rows + i] = grad_out[i * cols + j];
        }
    }
}

extern "C" void eshkol_backward_reshape(
    const double* grad_out, double* grad_in,
    int64_t total_elements)
{
    /* Reshape doesn't change data, just the view.
     * Gradient is a simple copy with the original shape. */
    memcpy(grad_in, grad_out, (size_t)total_elements * sizeof(double));
}

extern "C" void eshkol_backward_positional_encoding(
    const double* grad_out, double* grad_in,
    int64_t total_elements)
{
    /* Positional encoding is additive constant: out = in + PE
     * Gradient passes through unchanged: grad_in = grad_out */
    memcpy(grad_in, grad_out, (size_t)total_elements * sizeof(double));
}

/* ===== Reduction Operations ===== */

extern "C" void eshkol_backward_sum(
    double grad_out_scalar, double* grad_in,
    int64_t total_elements)
{
    /* Sum reduces to scalar: out = sum(in)
     * Backward broadcasts: grad_in[i] = grad_out for all i */
    for (int64_t i = 0; i < total_elements; i++) {
        grad_in[i] = grad_out_scalar;
    }
}

extern "C" void eshkol_backward_mean(
    double grad_out_scalar, double* grad_in,
    int64_t total_elements)
{
    /* Mean = sum / N
     * Backward: grad_in[i] = grad_out / N */
    double scale = 1.0 / (double)total_elements;
    for (int64_t i = 0; i < total_elements; i++) {
        grad_in[i] = grad_out_scalar * scale;
    }
}

/* ===== Pooling Operations ===== */

extern "C" void eshkol_backward_maxpool2d(
    const double* grad_out, double* grad_in,
    const int64_t* max_indices,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t channels, int64_t batch_size)
{
    (void)kernel_h; (void)kernel_w; (void)stride_h; (void)stride_w;
    int64_t in_spatial = in_h * in_w;
    int64_t out_spatial = out_h * out_w;

    /* Zero gradient input */
    memset(grad_in, 0, (size_t)(batch_size * channels * in_spatial) * sizeof(double));

    /* Scatter gradient to the position of the max element in each window */
    for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t c = 0; c < channels; c++) {
            int64_t in_offset = (b * channels + c) * in_spatial;
            int64_t out_offset = (b * channels + c) * out_spatial;
            for (int64_t oi = 0; oi < out_spatial; oi++) {
                int64_t max_idx = max_indices[out_offset + oi];
                grad_in[in_offset + max_idx] += grad_out[out_offset + oi];
            }
        }
    }
}

extern "C" void eshkol_backward_avgpool2d(
    const double* grad_out, double* grad_in,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t channels, int64_t batch_size)
{
    int64_t in_spatial = in_h * in_w;
    int64_t pool_size = kernel_h * kernel_w;
    double scale = 1.0 / (double)pool_size;

    /* Zero gradient input */
    memset(grad_in, 0, (size_t)(batch_size * channels * in_spatial) * sizeof(double));

    /* Distribute gradient uniformly to all elements in each pooling window */
    for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t c = 0; c < channels; c++) {
            int64_t in_base = (b * channels + c) * in_spatial;
            int64_t out_base = (b * channels + c) * out_h * out_w;
            for (int64_t oh = 0; oh < out_h; oh++) {
                for (int64_t ow = 0; ow < out_w; ow++) {
                    double g = grad_out[out_base + oh * out_w + ow] * scale;
                    for (int64_t kh = 0; kh < kernel_h; kh++) {
                        for (int64_t kw = 0; kw < kernel_w; kw++) {
                            int64_t ih = oh * stride_h + kh;
                            int64_t iw = ow * stride_w + kw;
                            if (ih < in_h && iw < in_w) {
                                grad_in[in_base + ih * in_w + iw] += g;
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ===== Convolution ===== */

extern "C" void eshkol_backward_conv2d(
    const double* grad_out,
    const double* saved_input, const double* saved_kernel,
    double* grad_input, double* grad_kernel,
    int64_t in_h, int64_t in_w,
    int64_t k_h, int64_t k_w,
    int64_t out_h, int64_t out_w,
    int64_t stride_h, int64_t stride_w,
    int64_t channels_in, int64_t channels_out,
    int64_t batch_size)
{
    int64_t in_spatial = in_h * in_w;
    int64_t out_spatial = out_h * out_w;
    int64_t kernel_spatial = k_h * k_w;

    /* Zero output gradients */
    memset(grad_input, 0,
           (size_t)(batch_size * channels_in * in_spatial) * sizeof(double));
    memset(grad_kernel, 0,
           (size_t)(channels_out * channels_in * kernel_spatial) * sizeof(double));

    for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t co = 0; co < channels_out; co++) {
            for (int64_t ci = 0; ci < channels_in; ci++) {
                const double* kernel_ptr =
                    saved_kernel + (co * channels_in + ci) * kernel_spatial;
                double* gk_ptr =
                    grad_kernel + (co * channels_in + ci) * kernel_spatial;

                for (int64_t oh = 0; oh < out_h; oh++) {
                    for (int64_t ow = 0; ow < out_w; ow++) {
                        double g = grad_out[
                            (b * channels_out + co) * out_spatial +
                            oh * out_w + ow];

                        for (int64_t kh = 0; kh < k_h; kh++) {
                            for (int64_t kw = 0; kw < k_w; kw++) {
                                int64_t ih = oh * stride_h + kh;
                                int64_t iw = ow * stride_w + kw;

                                /* grad_input: full convolution of
                                 * grad_out with flipped kernel */
                                double in_val = saved_input[
                                    (b * channels_in + ci) * in_spatial +
                                    ih * in_w + iw];
                                grad_input[
                                    (b * channels_in + ci) * in_spatial +
                                    ih * in_w + iw]
                                    += g * kernel_ptr[kh * k_w + kw];

                                /* grad_kernel: cross-correlation of
                                 * input with grad_out */
                                gk_ptr[kh * k_w + kw] += g * in_val;
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ===== Normalization ===== */

extern "C" void eshkol_backward_batchnorm(
    const double* grad_out,
    const double* saved_input, const double* saved_mean,
    const double* saved_inv_std, const double* saved_gamma,
    double* grad_input, double* grad_gamma, double* grad_beta,
    int64_t batch_size, int64_t feature_size)
{
    double N = (double)batch_size;

    /* Compute grad_beta = sum(grad_out, dim=batch) */
    memset(grad_beta, 0, (size_t)feature_size * sizeof(double));
    for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t f = 0; f < feature_size; f++) {
            grad_beta[f] += grad_out[b * feature_size + f];
        }
    }

    /* Compute grad_gamma = sum(grad_out * normalized, dim=batch) */
    memset(grad_gamma, 0, (size_t)feature_size * sizeof(double));
    for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t f = 0; f < feature_size; f++) {
            double normalized =
                (saved_input[b * feature_size + f] - saved_mean[f])
                * saved_inv_std[f];
            grad_gamma[f] += grad_out[b * feature_size + f] * normalized;
        }
    }

    /* Compute grad_input using the standard batchnorm backward formula:
     * grad_input = (gamma / (N * std)) *
     *   (N * grad_out - sum(grad_out) - normalized * sum(grad_out * normalized))
     */
    for (int64_t f = 0; f < feature_size; f++) {
        double gamma_f = saved_gamma[f];
        double inv_std_f = saved_inv_std[f];
        double sum_grad = grad_beta[f];
        double sum_grad_norm = grad_gamma[f];
        double coeff = gamma_f * inv_std_f / N;

        for (int64_t b = 0; b < batch_size; b++) {
            double normalized =
                (saved_input[b * feature_size + f] - saved_mean[f])
                * inv_std_f;
            grad_input[b * feature_size + f] =
                coeff * (N * grad_out[b * feature_size + f]
                         - sum_grad
                         - normalized * sum_grad_norm);
        }
    }
}

extern "C" void eshkol_backward_layernorm(
    const double* grad_out,
    const double* saved_input, const double* saved_mean,
    const double* saved_inv_std, const double* saved_gamma,
    double* grad_input, double* grad_gamma, double* grad_beta,
    int64_t num_samples, int64_t feature_size)
{
    double D = (double)feature_size;

    /* Compute grad_beta and grad_gamma (accumulated across samples) */
    memset(grad_beta, 0, (size_t)feature_size * sizeof(double));
    memset(grad_gamma, 0, (size_t)feature_size * sizeof(double));

    for (int64_t s = 0; s < num_samples; s++) {
        double mean_s = saved_mean[s];
        double inv_std_s = saved_inv_std[s];

        for (int64_t f = 0; f < feature_size; f++) {
            int64_t idx = s * feature_size + f;
            double normalized = (saved_input[idx] - mean_s) * inv_std_s;
            grad_beta[f] += grad_out[idx];
            grad_gamma[f] += grad_out[idx] * normalized;
        }
    }

    /* Compute grad_input per-sample (same formula as batchnorm but
     * applied per-sample across feature dimension) */
    for (int64_t s = 0; s < num_samples; s++) {
        double mean_s = saved_mean[s];
        double inv_std_s = saved_inv_std[s];

        /* Per-sample sums */
        double sum_grad = 0.0;
        double sum_grad_norm = 0.0;
        for (int64_t f = 0; f < feature_size; f++) {
            int64_t idx = s * feature_size + f;
            double g = grad_out[idx] * saved_gamma[f];
            double normalized = (saved_input[idx] - mean_s) * inv_std_s;
            sum_grad += g;
            sum_grad_norm += g * normalized;
        }

        for (int64_t f = 0; f < feature_size; f++) {
            int64_t idx = s * feature_size + f;
            double g = grad_out[idx] * saved_gamma[f];
            double normalized = (saved_input[idx] - mean_s) * inv_std_s;
            grad_input[idx] =
                inv_std_s / D *
                (D * g - sum_grad - normalized * sum_grad_norm);
        }
    }
}

/* ===== Matmul ===== */

extern "C" void eshkol_backward_matmul(
    const double* grad_out,
    const double* saved_A, const double* saved_B,
    double* grad_A, double* grad_B,
    int64_t M, int64_t K, int64_t N)
{
    /* Forward: C = A @ B, where A is (M,K), B is (K,N), C is (M,N)
     * Backward: dA = grad_C @ B^T, dB = A^T @ grad_C */

    /* dA = grad_out @ B^T: (M,N) @ (N,K) -> (M,K) */
    memset(grad_A, 0, (size_t)(M * K) * sizeof(double));
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < K; j++) {
            double sum = 0.0;
            for (int64_t n = 0; n < N; n++) {
                sum += grad_out[i * N + n] * saved_B[j * N + n];
            }
            grad_A[i * K + j] = sum;
        }
    }

    /* dB = A^T @ grad_out: (K,M) @ (M,N) -> (K,N) */
    memset(grad_B, 0, (size_t)(K * N) * sizeof(double));
    for (int64_t j = 0; j < K; j++) {
        for (int64_t n = 0; n < N; n++) {
            double sum = 0.0;
            for (int64_t i = 0; i < M; i++) {
                sum += saved_A[i * K + j] * grad_out[i * N + n];
            }
            grad_B[j * N + n] = sum;
        }
    }
}

/* ===== Attention ===== */

extern "C" void eshkol_backward_attention(
    const double* grad_out,
    const double* saved_Q, const double* saved_K, const double* saved_V,
    const double* saved_attn_weights,
    double* grad_Q, double* grad_K, double* grad_V,
    int64_t seq_q, int64_t seq_k, int64_t d_k, int64_t d_v,
    double scale)
{
    /* Forward:
     *   scores = Q @ K^T * scale   (seq_q, seq_k)
     *   attn = softmax(scores)     (seq_q, seq_k)
     *   out = attn @ V             (seq_q, d_v)
     *
     * Backward:
     *   d_V = attn^T @ grad_out          (seq_k, d_v)
     *   d_attn = grad_out @ V^T           (seq_q, seq_k)
     *   d_scores = softmax_backward(d_attn, attn)  (seq_q, seq_k)
     *   d_scores *= scale
     *   d_Q = d_scores @ K                (seq_q, d_k)
     *   d_K = d_scores^T @ Q              (seq_k, d_k)
     */

    /* Allocate temporary for d_attn and d_scores */
    size_t scores_size = (size_t)(seq_q * seq_k);
    double* d_attn = (double*)calloc(scores_size, sizeof(double));
    double* d_scores = (double*)calloc(scores_size, sizeof(double));
    if (!d_attn || !d_scores) {
        free(d_attn);
        free(d_scores);
        return;
    }

    /* d_V = attn^T @ grad_out: (seq_k, seq_q) @ (seq_q, d_v) -> (seq_k, d_v) */
    memset(grad_V, 0, (size_t)(seq_k * d_v) * sizeof(double));
    for (int64_t k = 0; k < seq_k; k++) {
        for (int64_t d = 0; d < d_v; d++) {
            double sum = 0.0;
            for (int64_t q = 0; q < seq_q; q++) {
                sum += saved_attn_weights[q * seq_k + k] *
                       grad_out[q * d_v + d];
            }
            grad_V[k * d_v + d] = sum;
        }
    }

    /* d_attn = grad_out @ V^T: (seq_q, d_v) @ (d_v, seq_k) -> (seq_q, seq_k) */
    for (int64_t q = 0; q < seq_q; q++) {
        for (int64_t k = 0; k < seq_k; k++) {
            double sum = 0.0;
            for (int64_t d = 0; d < d_v; d++) {
                sum += grad_out[q * d_v + d] * saved_V[k * d_v + d];
            }
            d_attn[q * seq_k + k] = sum;
        }
    }

    /* Softmax backward: d_scores[i][j] = attn[i][j] * (d_attn[i][j] - dot(attn[i], d_attn[i])) */
    for (int64_t q = 0; q < seq_q; q++) {
        double dot = 0.0;
        for (int64_t k = 0; k < seq_k; k++) {
            dot += saved_attn_weights[q * seq_k + k] *
                   d_attn[q * seq_k + k];
        }
        for (int64_t k = 0; k < seq_k; k++) {
            d_scores[q * seq_k + k] =
                saved_attn_weights[q * seq_k + k] *
                (d_attn[q * seq_k + k] - dot) * scale;
        }
    }

    /* d_Q = d_scores @ K: (seq_q, seq_k) @ (seq_k, d_k) -> (seq_q, d_k) */
    memset(grad_Q, 0, (size_t)(seq_q * d_k) * sizeof(double));
    for (int64_t q = 0; q < seq_q; q++) {
        for (int64_t d = 0; d < d_k; d++) {
            double sum = 0.0;
            for (int64_t k = 0; k < seq_k; k++) {
                sum += d_scores[q * seq_k + k] * saved_K[k * d_k + d];
            }
            grad_Q[q * d_k + d] = sum;
        }
    }

    /* d_K = d_scores^T @ Q: (seq_k, seq_q) @ (seq_q, d_k) -> (seq_k, d_k) */
    memset(grad_K, 0, (size_t)(seq_k * d_k) * sizeof(double));
    for (int64_t k = 0; k < seq_k; k++) {
        for (int64_t d = 0; d < d_k; d++) {
            double sum = 0.0;
            for (int64_t q = 0; q < seq_q; q++) {
                sum += d_scores[q * seq_k + k] * saved_Q[q * d_k + d];
            }
            grad_K[k * d_k + d] = sum;
        }
    }

    free(d_attn);
    free(d_scores);
}

extern "C" void eshkol_backward_multihead_attention(
    const double* grad_out,
    const double* saved_Q, const double* saved_K, const double* saved_V,
    const double* saved_W_Q, const double* saved_W_K,
    const double* saved_W_V, const double* saved_W_O,
    const double* saved_attn_per_head,
    double* grad_Q, double* grad_K, double* grad_V,
    double* grad_WQ, double* grad_WK, double* grad_WV, double* grad_WO,
    int64_t seq_q, int64_t seq_k, int64_t d_model, int64_t num_heads)
{
    int64_t head_dim = d_model / num_heads;
    double scale = 1.0 / sqrt((double)head_dim);

    /* Allocate temporaries for per-head gradients */
    size_t head_q_size = (size_t)(seq_q * head_dim);
    size_t head_k_size = (size_t)(seq_k * head_dim);
    double* d_head_Q = (double*)calloc(head_q_size, sizeof(double));
    double* d_head_K = (double*)calloc(head_k_size, sizeof(double));
    double* d_head_V = (double*)calloc(head_k_size, sizeof(double));
    if (!d_head_Q || !d_head_K || !d_head_V) {
        free(d_head_Q); free(d_head_K); free(d_head_V);
        return;
    }

    /* grad_WO: grad_out^T @ concat_heads
     * For simplicity, compute as: grad_WO += grad_out^T @ projected
     * First, backprop through W_O: d_concat = grad_out @ W_O^T */
    size_t concat_size = (size_t)(seq_q * d_model);
    double* d_concat = (double*)calloc(concat_size, sizeof(double));
    if (!d_concat) {
        free(d_head_Q); free(d_head_K); free(d_head_V);
        return;
    }

    /* d_concat = grad_out @ W_O^T: (seq_q, d_model) @ (d_model, d_model) */
    memset(d_concat, 0, concat_size * sizeof(double));
    for (int64_t i = 0; i < seq_q; i++) {
        for (int64_t j = 0; j < d_model; j++) {
            double sum = 0.0;
            for (int64_t k = 0; k < d_model; k++) {
                sum += grad_out[i * d_model + k] *
                       saved_W_O[j * d_model + k];
            }
            d_concat[i * d_model + j] = sum;
        }
    }

    /* Zero output weight gradients */
    memset(grad_WQ, 0, (size_t)(d_model * d_model) * sizeof(double));
    memset(grad_WK, 0, (size_t)(d_model * d_model) * sizeof(double));
    memset(grad_WV, 0, (size_t)(d_model * d_model) * sizeof(double));
    memset(grad_WO, 0, (size_t)(d_model * d_model) * sizeof(double));
    memset(grad_Q, 0, (size_t)(seq_q * d_model) * sizeof(double));
    memset(grad_K, 0, (size_t)(seq_k * d_model) * sizeof(double));
    memset(grad_V, 0, (size_t)(seq_k * d_model) * sizeof(double));

    /* grad_WO = (concat_heads)^T @ grad_out — accumulate from saved data */
    /* For now, approximate using the concatenated output from forward */

    /* Per-head backward */
    for (int64_t h = 0; h < num_heads; h++) {
        int64_t h_offset = h * head_dim;

        /* Extract per-head gradient from d_concat */
        for (int64_t i = 0; i < seq_q; i++) {
            for (int64_t d = 0; d < head_dim; d++) {
                d_head_Q[i * head_dim + d] = 0.0;
            }
        }

        /* Extract per-head d_concat slice */
        double* d_head_out = (double*)calloc(
            (size_t)(seq_q * head_dim), sizeof(double));
        if (!d_head_out) continue;

        for (int64_t i = 0; i < seq_q; i++) {
            for (int64_t d = 0; d < head_dim; d++) {
                d_head_out[i * head_dim + d] =
                    d_concat[i * d_model + h_offset + d];
            }
        }

        /* Per-head projected Q, K, V */
        const double* head_attn =
            saved_attn_per_head + h * seq_q * seq_k;

        /* Extract per-head Q, K, V from saved projected data */
        /* Q_h = Q @ W_Q_h, so we need to extract the head slice */
        double* head_Q = (double*)calloc(head_q_size, sizeof(double));
        double* head_K = (double*)calloc(head_k_size, sizeof(double));
        double* head_V = (double*)calloc(head_k_size, sizeof(double));
        if (!head_Q || !head_K || !head_V) {
            free(d_head_out); free(head_Q); free(head_K); free(head_V);
            continue;
        }

        /* Project Q, K, V through head-specific weight slices */
        for (int64_t i = 0; i < seq_q; i++) {
            for (int64_t d = 0; d < head_dim; d++) {
                double sum = 0.0;
                for (int64_t j = 0; j < d_model; j++) {
                    sum += saved_Q[i * d_model + j] *
                           saved_W_Q[j * d_model + h_offset + d];
                }
                head_Q[i * head_dim + d] = sum;
            }
        }
        for (int64_t i = 0; i < seq_k; i++) {
            for (int64_t d = 0; d < head_dim; d++) {
                double sum = 0.0;
                for (int64_t j = 0; j < d_model; j++) {
                    sum += saved_K[i * d_model + j] *
                           saved_W_K[j * d_model + h_offset + d];
                }
                head_K[i * head_dim + d] = sum;
            }
        }
        for (int64_t i = 0; i < seq_k; i++) {
            for (int64_t d = 0; d < head_dim; d++) {
                double sum = 0.0;
                for (int64_t j = 0; j < d_model; j++) {
                    sum += saved_V[i * d_model + j] *
                           saved_W_V[j * d_model + h_offset + d];
                }
                head_V[i * head_dim + d] = sum;
            }
        }

        /* Attention backward for this head */
        eshkol_backward_attention(
            d_head_out,
            head_Q, head_K, head_V,
            head_attn,
            d_head_Q, d_head_K, d_head_V,
            seq_q, seq_k, head_dim, head_dim, scale);

        /* Accumulate weight gradients:
         * grad_WQ_h += Q^T @ d_head_Q, etc. */
        for (int64_t j = 0; j < d_model; j++) {
            for (int64_t d = 0; d < head_dim; d++) {
                double sum_q = 0.0, sum_k = 0.0, sum_v = 0.0;
                for (int64_t i = 0; i < seq_q; i++) {
                    sum_q += saved_Q[i * d_model + j] *
                             d_head_Q[i * head_dim + d];
                }
                for (int64_t i = 0; i < seq_k; i++) {
                    sum_k += saved_K[i * d_model + j] *
                             d_head_K[i * head_dim + d];
                    sum_v += saved_V[i * d_model + j] *
                             d_head_V[i * head_dim + d];
                }
                grad_WQ[j * d_model + h_offset + d] += sum_q;
                grad_WK[j * d_model + h_offset + d] += sum_k;
                grad_WV[j * d_model + h_offset + d] += sum_v;
            }
        }

        /* Backprop through projection: grad_Q += d_head_Q @ W_Q_h^T */
        for (int64_t i = 0; i < seq_q; i++) {
            for (int64_t j = 0; j < d_model; j++) {
                double sum = 0.0;
                for (int64_t d = 0; d < head_dim; d++) {
                    sum += d_head_Q[i * head_dim + d] *
                           saved_W_Q[j * d_model + h_offset + d];
                }
                grad_Q[i * d_model + j] += sum;
            }
        }
        for (int64_t i = 0; i < seq_k; i++) {
            for (int64_t j = 0; j < d_model; j++) {
                double sum_k = 0.0, sum_v = 0.0;
                for (int64_t d = 0; d < head_dim; d++) {
                    sum_k += d_head_K[i * head_dim + d] *
                             saved_W_K[j * d_model + h_offset + d];
                    sum_v += d_head_V[i * head_dim + d] *
                             saved_W_V[j * d_model + h_offset + d];
                }
                grad_K[i * d_model + j] += sum_k;
                grad_V[i * d_model + j] += sum_v;
            }
        }

        free(d_head_out);
        free(head_Q);
        free(head_K);
        free(head_V);
    }

    free(d_concat);
    free(d_head_Q);
    free(d_head_K);
    free(d_head_V);
}

/* ===== Embedding ===== */

extern "C" void eshkol_backward_embedding(
    const double* grad_out,
    const int64_t* saved_indices,
    double* grad_weights,
    int64_t num_indices, int64_t d_model, int64_t vocab_size)
{
    /* Forward: out[i] = weights[indices[i]]
     * Backward: scatter-add grad_out to weight rows indicated by indices */
    memset(grad_weights, 0,
           (size_t)(vocab_size * d_model) * sizeof(double));

    for (int64_t i = 0; i < num_indices; i++) {
        int64_t idx = saved_indices[i];
        if (idx >= 0 && idx < vocab_size) {
            for (int64_t d = 0; d < d_model; d++) {
                grad_weights[idx * d_model + d] +=
                    grad_out[i * d_model + d];
            }
        }
    }
}

/* ===== Tensor Gradient Seeding ===== */

extern "C" void eshkol_seed_tensor_gradient(void* ad_node_ptr) {
    if (!ad_node_ptr) return;

    ad_node_t* node = (ad_node_t*)ad_node_ptr;

    /* Only seed if this is a tensor node (has tensor_value set) */
    if (!node->tensor_value) return;

    /* Compute total elements from shape */
    int64_t total = 1;
    for (size_t i = 0; i < node->ndim; i++) {
        total *= node->shape[i];
    }
    if (total <= 0) return;

    /* Allocate tensor_gradient if not already set */
    if (!node->tensor_gradient) {
        node->tensor_gradient = calloc((size_t)total, sizeof(double));
        if (!node->tensor_gradient) return;
    }

    /* Fill with all ones: dL/dL = 1.0 for every output element */
    double* tg = (double*)node->tensor_gradient;
    for (int64_t i = 0; i < total; i++) {
        tg[i] = 1.0;
    }
}

/* ===== Tensor Backward Dispatcher ===== */

/*
 * Helper: compute total elements from shape array.
 */
static int64_t compute_total_elements(const int64_t* shape, size_t ndim) {
    int64_t total = 1;
    for (size_t i = 0; i < ndim; i++) {
        total *= shape[i];
    }
    return total;
}

extern "C" void eshkol_tensor_backward_dispatch(void* ad_node_ptr) {
    if (!ad_node_ptr) return;

    ad_node_t* node = (ad_node_t*)ad_node_ptr;
    double* upstream_grad = (double*)node->tensor_gradient;
    if (!upstream_grad) return;

    /* Access params as raw int64_t array (matches union layout) */
    int64_t* p = (int64_t*)&node->params;

    switch (node->type) {

    /* --- CONV2D (19) --- */
    case AD_NODE_CONV2D: {
        if (!node->saved_tensors || node->num_saved < 2) break;
        const double* saved_input  = (const double*)node->saved_tensors[0];
        const double* saved_kernel = (const double*)node->saved_tensors[1];
        int64_t k_h = p[0], k_w = p[1];
        int64_t stride_h = p[2], stride_w = p[3];
        int64_t channels_in = p[4], channels_out = p[5];
        /* Derive spatial dimensions from output shape */
        /* shape = [batch_size, channels_out, out_h, out_w] */
        int64_t batch_size = (node->ndim >= 1) ? node->shape[0] : 1;
        int64_t out_h = (node->ndim >= 3) ? node->shape[2] : 1;
        int64_t out_w = (node->ndim >= 4) ? node->shape[3] : 1;
        int64_t in_h = (out_h - 1) * stride_h + k_h;
        int64_t in_w = (out_w - 1) * stride_w + k_w;
        int64_t input_total = batch_size * channels_in * in_h * in_w;
        int64_t kernel_total = channels_out * channels_in * k_h * k_w;
        double* grad_input  = (double*)calloc((size_t)input_total, sizeof(double));
        double* grad_kernel = (double*)calloc((size_t)kernel_total, sizeof(double));
        if (grad_input && grad_kernel) {
            eshkol_backward_conv2d(upstream_grad, saved_input, saved_kernel,
                grad_input, grad_kernel, in_h, in_w, k_h, k_w, out_h, out_w,
                stride_h, stride_w, channels_in, channels_out, batch_size);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
            if (node->input2)
                eshkol_accumulate_tensor_grad(node->input2, grad_kernel, kernel_total);
        }
        free(grad_input);
        free(grad_kernel);
        break;
    }

    /* --- MAXPOOL2D (20) --- */
    case AD_NODE_MAXPOOL2D: {
        if (!node->saved_tensors || node->num_saved < 1) break;
        const int64_t* max_indices = (const int64_t*)node->saved_tensors[0];
        int64_t k_h = p[0], k_w = p[1];
        int64_t stride_h = p[2], stride_w = p[3];
        int64_t channels = p[4], batch_size = p[5];
        int64_t out_h = (node->ndim >= 3) ? node->shape[2] : 1;
        int64_t out_w = (node->ndim >= 4) ? node->shape[3] : 1;
        int64_t in_h = (out_h - 1) * stride_h + k_h;
        int64_t in_w = (out_w - 1) * stride_w + k_w;
        int64_t input_total = batch_size * channels * in_h * in_w;
        double* grad_input = (double*)calloc((size_t)input_total, sizeof(double));
        if (grad_input) {
            eshkol_backward_maxpool2d(upstream_grad, grad_input, max_indices,
                in_h, in_w, out_h, out_w, k_h, k_w, stride_h, stride_w,
                channels, batch_size);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
        }
        free(grad_input);
        break;
    }

    /* --- AVGPOOL2D (21) --- */
    case AD_NODE_AVGPOOL2D: {
        int64_t k_h = p[0], k_w = p[1];
        int64_t stride_h = p[2], stride_w = p[3];
        int64_t channels = p[4], batch_size = p[5];
        int64_t out_h = (node->ndim >= 3) ? node->shape[2] : 1;
        int64_t out_w = (node->ndim >= 4) ? node->shape[3] : 1;
        int64_t in_h = (out_h - 1) * stride_h + k_h;
        int64_t in_w = (out_w - 1) * stride_w + k_w;
        int64_t input_total = batch_size * channels * in_h * in_w;
        double* grad_input = (double*)calloc((size_t)input_total, sizeof(double));
        if (grad_input) {
            eshkol_backward_avgpool2d(upstream_grad, grad_input,
                in_h, in_w, out_h, out_w, k_h, k_w, stride_h, stride_w,
                channels, batch_size);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
        }
        free(grad_input);
        break;
    }

    /* --- BATCHNORM (22) --- */
    case AD_NODE_BATCHNORM: {
        if (!node->saved_tensors || node->num_saved < 4) break;
        const double* saved_input   = (const double*)node->saved_tensors[0];
        const double* saved_mean    = (const double*)node->saved_tensors[1];
        const double* saved_inv_std = (const double*)node->saved_tensors[2];
        const double* saved_gamma   = (const double*)node->saved_tensors[3];
        int64_t batch_size    = p[0];
        int64_t feature_size  = p[1];
        int64_t input_total   = batch_size * feature_size;
        double* grad_input = (double*)calloc((size_t)input_total, sizeof(double));
        double* grad_gamma = (double*)calloc((size_t)feature_size, sizeof(double));
        double* grad_beta  = (double*)calloc((size_t)feature_size, sizeof(double));
        if (grad_input && grad_gamma && grad_beta) {
            eshkol_backward_batchnorm(upstream_grad, saved_input, saved_mean,
                saved_inv_std, saved_gamma, grad_input, grad_gamma, grad_beta,
                batch_size, feature_size);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
            if (node->input2)
                eshkol_accumulate_tensor_grad(node->input2, grad_gamma, feature_size);
        }
        free(grad_input);
        free(grad_gamma);
        free(grad_beta);
        break;
    }

    /* --- LAYERNORM (23) --- */
    case AD_NODE_LAYERNORM: {
        if (!node->saved_tensors || node->num_saved < 4) break;
        const double* saved_input   = (const double*)node->saved_tensors[0];
        const double* saved_mean    = (const double*)node->saved_tensors[1];
        const double* saved_inv_std = (const double*)node->saved_tensors[2];
        const double* saved_gamma   = (const double*)node->saved_tensors[3];
        int64_t num_samples   = p[0];
        int64_t feature_size  = p[1];
        int64_t input_total   = num_samples * feature_size;
        double* grad_input = (double*)calloc((size_t)input_total, sizeof(double));
        double* grad_gamma = (double*)calloc((size_t)feature_size, sizeof(double));
        double* grad_beta  = (double*)calloc((size_t)feature_size, sizeof(double));
        if (grad_input && grad_gamma && grad_beta) {
            eshkol_backward_layernorm(upstream_grad, saved_input, saved_mean,
                saved_inv_std, saved_gamma, grad_input, grad_gamma, grad_beta,
                num_samples, feature_size);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
            if (node->input2)
                eshkol_accumulate_tensor_grad(node->input2, grad_gamma, feature_size);
        }
        free(grad_input);
        free(grad_gamma);
        free(grad_beta);
        break;
    }

    /* --- MATMUL (24) --- */
    case AD_NODE_MATMUL: {
        if (!node->saved_tensors || node->num_saved < 2) break;
        const double* saved_A = (const double*)node->saved_tensors[0];
        const double* saved_B = (const double*)node->saved_tensors[1];
        int64_t M = p[0], K = p[1], N = p[2];
        double* grad_A = (double*)calloc((size_t)(M * K), sizeof(double));
        double* grad_B = (double*)calloc((size_t)(K * N), sizeof(double));
        if (grad_A && grad_B) {
            eshkol_backward_matmul(upstream_grad, saved_A, saved_B,
                grad_A, grad_B, M, K, N);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_A, M * K);
            if (node->input2)
                eshkol_accumulate_tensor_grad(node->input2, grad_B, K * N);
        }
        free(grad_A);
        free(grad_B);
        break;
    }

    /* --- TRANSPOSE (25) --- */
    case AD_NODE_TRANSPOSE: {
        int64_t rows = p[0], cols = p[1];
        /* Input has shape (cols, rows) — transpose of output (rows, cols) */
        int64_t input_total = rows * cols;
        double* grad_input = (double*)calloc((size_t)input_total, sizeof(double));
        if (grad_input) {
            eshkol_backward_transpose(upstream_grad, grad_input, rows, cols);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
        }
        free(grad_input);
        break;
    }

    /* --- RESHAPE (26) --- */
    case AD_NODE_RESHAPE: {
        int64_t total_elements = p[0];
        double* grad_input = (double*)calloc((size_t)total_elements, sizeof(double));
        if (grad_input) {
            eshkol_backward_reshape(upstream_grad, grad_input, total_elements);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, total_elements);
        }
        free(grad_input);
        break;
    }

    /* --- SUM (27) --- */
    case AD_NODE_SUM: {
        int64_t input_total = p[0];
        /* Sum output is scalar; upstream_grad is a 1-element tensor */
        double grad_scalar = upstream_grad[0];
        double* grad_input = (double*)calloc((size_t)input_total, sizeof(double));
        if (grad_input) {
            eshkol_backward_sum(grad_scalar, grad_input, input_total);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
        }
        free(grad_input);
        break;
    }

    /* --- MEAN (28) --- */
    case AD_NODE_MEAN: {
        int64_t input_total = p[0];
        double grad_scalar = upstream_grad[0];
        double* grad_input = (double*)calloc((size_t)input_total, sizeof(double));
        if (grad_input) {
            eshkol_backward_mean(grad_scalar, grad_input, input_total);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
        }
        free(grad_input);
        break;
    }

    /* --- ATTENTION (29) --- */
    case AD_NODE_ATTENTION: {
        if (!node->saved_tensors || node->num_saved < 4) break;
        const double* saved_Q    = (const double*)node->saved_tensors[0];
        const double* saved_K    = (const double*)node->saved_tensors[1];
        const double* saved_V    = (const double*)node->saved_tensors[2];
        const double* saved_attn = (const double*)node->saved_tensors[3];
        int64_t seq_q = p[0], seq_k = p[1];
        int64_t d_k = p[2], d_v = p[3];
        /* scale stored as double bit-pattern in p[4] */
        double scale;
        memcpy(&scale, &p[4], sizeof(double));
        double* grad_Q = (double*)calloc((size_t)(seq_q * d_k), sizeof(double));
        double* grad_K = (double*)calloc((size_t)(seq_k * d_k), sizeof(double));
        double* grad_V = (double*)calloc((size_t)(seq_k * d_v), sizeof(double));
        if (grad_Q && grad_K && grad_V) {
            eshkol_backward_attention(upstream_grad, saved_Q, saved_K, saved_V,
                saved_attn, grad_Q, grad_K, grad_V,
                seq_q, seq_k, d_k, d_v, scale);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_Q, seq_q * d_k);
            if (node->input2)
                eshkol_accumulate_tensor_grad(node->input2, grad_K, seq_k * d_k);
            if (node->input3)
                eshkol_accumulate_tensor_grad(node->input3, grad_V, seq_k * d_v);
        }
        free(grad_Q);
        free(grad_K);
        free(grad_V);
        break;
    }

    /* --- MULTIHEAD_ATTENTION (30) --- */
    case AD_NODE_MULTIHEAD_ATTENTION: {
        if (!node->saved_tensors || node->num_saved < 8) break;
        const double* saved_Q        = (const double*)node->saved_tensors[0];
        const double* saved_K        = (const double*)node->saved_tensors[1];
        const double* saved_V        = (const double*)node->saved_tensors[2];
        const double* saved_W_Q      = (const double*)node->saved_tensors[3];
        const double* saved_W_K      = (const double*)node->saved_tensors[4];
        const double* saved_W_V      = (const double*)node->saved_tensors[5];
        const double* saved_W_O      = (const double*)node->saved_tensors[6];
        const double* saved_attn_ph  = (const double*)node->saved_tensors[7];
        int64_t seq_q = p[0], seq_k = p[1];
        int64_t d_model = p[2], num_heads = p[3];
        double* grad_Q  = (double*)calloc((size_t)(seq_q * d_model), sizeof(double));
        double* grad_K  = (double*)calloc((size_t)(seq_k * d_model), sizeof(double));
        double* grad_V  = (double*)calloc((size_t)(seq_k * d_model), sizeof(double));
        double* grad_WQ = (double*)calloc((size_t)(d_model * d_model), sizeof(double));
        double* grad_WK = (double*)calloc((size_t)(d_model * d_model), sizeof(double));
        double* grad_WV = (double*)calloc((size_t)(d_model * d_model), sizeof(double));
        double* grad_WO = (double*)calloc((size_t)(d_model * d_model), sizeof(double));
        if (grad_Q && grad_K && grad_V && grad_WQ && grad_WK && grad_WV && grad_WO) {
            eshkol_backward_multihead_attention(upstream_grad,
                saved_Q, saved_K, saved_V,
                saved_W_Q, saved_W_K, saved_W_V, saved_W_O, saved_attn_ph,
                grad_Q, grad_K, grad_V,
                grad_WQ, grad_WK, grad_WV, grad_WO,
                seq_q, seq_k, d_model, num_heads);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_Q, seq_q * d_model);
            if (node->input2)
                eshkol_accumulate_tensor_grad(node->input2, grad_K, seq_k * d_model);
            if (node->input3)
                eshkol_accumulate_tensor_grad(node->input3, grad_V, seq_k * d_model);
            /* Weight gradients would go to input4 or separate weight nodes */
        }
        free(grad_Q); free(grad_K); free(grad_V);
        free(grad_WQ); free(grad_WK); free(grad_WV); free(grad_WO);
        break;
    }

    /* --- POSITIONAL_ENCODING (31) --- */
    case AD_NODE_POSITIONAL_ENCODING: {
        int64_t total_elements = p[0];
        if (total_elements <= 0 && node->ndim > 0)
            total_elements = compute_total_elements(node->shape, node->ndim);
        double* grad_input = (double*)calloc((size_t)total_elements, sizeof(double));
        if (grad_input) {
            eshkol_backward_positional_encoding(upstream_grad, grad_input, total_elements);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, total_elements);
        }
        free(grad_input);
        break;
    }

    /* --- EMBEDDING (32) --- */
    case AD_NODE_EMBEDDING: {
        if (!node->saved_tensors || node->num_saved < 1) break;
        const int64_t* saved_indices = (const int64_t*)node->saved_tensors[0];
        int64_t num_indices = p[0];
        int64_t d_model     = p[1];
        int64_t vocab_size  = p[2];
        int64_t weight_total = vocab_size * d_model;
        double* grad_weights = (double*)calloc((size_t)weight_total, sizeof(double));
        if (grad_weights) {
            eshkol_backward_embedding(upstream_grad, saved_indices,
                grad_weights, num_indices, d_model, vocab_size);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_weights, weight_total);
        }
        free(grad_weights);
        break;
    }

    default:
        /* Unhandled tensor op — gradient silently dropped.
         * Geometric/hyperbolic ops (33-40) use scalar backward in codegen. */
        break;
    }
}

/* ===== Tensor Gradient Accumulation ===== */

extern "C" void eshkol_accumulate_tensor_grad(
    void* ad_node_ptr, const double* grad, int64_t num_elements)
{
    if (!ad_node_ptr || !grad || num_elements <= 0) return;

    ad_node_t* node = (ad_node_t*)ad_node_ptr;

    /* If tensor_gradient is NULL, allocate and zero-fill */
    if (!node->tensor_gradient) {
        node->tensor_gradient = calloc(
            (size_t)num_elements, sizeof(double));
        if (!node->tensor_gradient) return;
    }

    /* Element-wise accumulate: tensor_gradient[i] += grad[i] */
    double* tg = (double*)node->tensor_gradient;
    for (int64_t i = 0; i < num_elements; i++) {
        tg[i] += grad[i];
    }
}
