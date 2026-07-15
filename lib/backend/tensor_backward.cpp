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
#include <eshkol/logger.h>
#include <cstring>
#include <cmath>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <eshkol/backend/gpu/gpu_memory.h>
#include "../../lib/core/arena_memory.h"

/*
 * Safe integer multiplication for allocation sizes.
 * Returns false if the result would overflow, preventing heap buffer overflows.
 */
static bool safe_mul(int64_t a, int64_t b, int64_t* result) {
    if (a == 0 || b == 0) { *result = 0; return true; }
    if (a < 0 || b < 0) return false;
    if (a > INT64_MAX / b) return false;
    *result = a * b;
    return true;
}

/** @brief Overflow-checked three-operand product; see safe_mul(). */
static bool safe_mul3(int64_t a, int64_t b, int64_t c, int64_t* result) {
    int64_t ab;
    if (!safe_mul(a, b, &ab)) return false;
    return safe_mul(ab, c, result);
}

/** @brief Overflow-checked four-operand product; see safe_mul(). */
static bool safe_mul4(int64_t a, int64_t b, int64_t c, int64_t d, int64_t* result) {
    int64_t abc;
    if (!safe_mul3(a, b, c, &abc)) return false;
    return safe_mul(abc, d, result);
}

/*
 * Arena-based zero-initialized allocation for backward pass temporaries.
 * Uses the global arena with a scope push/pop pattern — the caller must
 * call backward_scope_end() when done. Returns nullptr on failure.
 *
 * OpenMP per-thread buffers also use the global arena — the arena is
 * only accessed from the main thread's scope, and per-thread buffers
 * are allocated before the parallel region begins.
 */
static arena_t* backward_scope_begin() {
    arena_t* arena = get_global_arena();
    if (arena) arena_push_scope(arena);
    return arena;
}

/** @brief Pop the arena scope opened by backward_scope_begin(), bulk-freeing
 *         the backward pass's temporary gradient buffers. */
static void backward_scope_end(arena_t* arena) {
    if (arena) arena_pop_scope(arena);
}

/** @brief Zero-initialized allocation of @p size bytes from @p arena; returns
 *         nullptr if the arena is absent or the size is zero. */
static void* arena_calloc(arena_t* arena, size_t size) {
    if (!arena || size == 0) return nullptr;
    return arena_allocate_zeroed(arena, size);
}

#ifdef ESHKOL_BLAS_ENABLED
#include <eshkol/backend/cblas_compat.h>
#endif

/* Forward declaration: bridge backward dispatch from lib/bridge/tensor_backward.cpp */
typedef void (*bridge_backward_fn_t)(ad_node_t*);
extern "C" bridge_backward_fn_t get_tensor_backward_fn(int node_type);

/* ===== Structural Operations ===== */

/** @brief Backward pass for tensor transpose: transposes grad_out back into
 *         grad_in's (cols, rows) layout. */
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

/** @brief Backward pass for tensor reshape: the gradient is a plain copy
 *         since reshape only changes the logical view, not the data. */
extern "C" void eshkol_backward_reshape(
    const double* grad_out, double* grad_in,
    int64_t total_elements)
{
    /* Reshape doesn't change data, just the view.
     * Gradient is a simple copy with the original shape. */
    memcpy(grad_in, grad_out, (size_t)total_elements * sizeof(double));
}

/** @brief Backward pass for additive positional encoding: gradient passes
 *         through unchanged (out = in + PE, a constant offset). */
extern "C" void eshkol_backward_positional_encoding(
    const double* grad_out, double* grad_in,
    int64_t total_elements)
{
    /* Positional encoding is additive constant: out = in + PE
     * Gradient passes through unchanged: grad_in = grad_out */
    memcpy(grad_in, grad_out, (size_t)total_elements * sizeof(double));
}

/* ===== Reduction Operations ===== */

/** @brief Backward pass for tensor sum reduction: broadcasts the scalar
 *         upstream gradient to every element of grad_in. */
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

/** @brief Backward pass for tensor mean reduction: distributes the scalar
 *         upstream gradient evenly (scaled by 1/N) to every input element. */
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

/** @brief Backward pass for 2D max pooling: scatters each output gradient to
 *         the single input position (recorded in @p max_indices during the
 *         forward pass) that produced the max in its window; all other
 *         input positions receive zero gradient. */
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

/** @brief Backward pass for 2D average pooling: distributes each output
 *         gradient uniformly (scaled by 1/kernel_size) across every input
 *         position that contributed to its pooling window. */
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

/** @brief Backward pass for 2D convolution: computes grad_input (full
 *         correlation of grad_out with the kernel) and grad_kernel
 *         (correlation of the saved input with grad_out). Dispatches to GPU
 *         when the op is large enough (eshkol_gpu_should_use()), falling
 *         back to CPU on GPU failure; the CPU path parallelizes over the
 *         batch dimension with OpenMP, using per-thread kernel-gradient
 *         buffers that are reduced at the end to avoid write contention. */
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

    /* Try GPU dispatch for large convolutions */
    size_t total_ops = (size_t)(batch_size * channels_out * channels_in * out_h * out_w * k_h * k_w);
    if (eshkol_gpu_should_use((int64_t)total_ops)) {
        EshkolGPUBuffer buf_go, buf_kw, buf_gi, buf_si, buf_gk;
        size_t go_size = (size_t)(batch_size * channels_out * out_spatial) * sizeof(double);
        size_t gi_size = (size_t)(batch_size * channels_in * in_spatial) * sizeof(double);
        size_t kw_size = (size_t)(channels_out * channels_in * kernel_spatial) * sizeof(double);

        if (eshkol_gpu_wrap_host((void*)grad_out, go_size, &buf_go) == 0 &&
            eshkol_gpu_wrap_host((void*)saved_kernel, kw_size, &buf_kw) == 0 &&
            eshkol_gpu_wrap_host((void*)grad_input, gi_size, &buf_gi) == 0 &&
            eshkol_gpu_wrap_host((void*)saved_input, gi_size, &buf_si) == 0 &&
            eshkol_gpu_wrap_host((void*)grad_kernel, kw_size, &buf_gk) == 0) {

            int rc1 = eshkol_gpu_conv2d_backward_input_f64(
                &buf_go, &buf_kw, &buf_gi,
                in_h, in_w, k_h, k_w, out_h, out_w,
                stride_h, stride_w, channels_in, channels_out, batch_size);
            int rc2 = eshkol_gpu_conv2d_backward_kernel_f64(
                &buf_go, &buf_si, &buf_gk,
                in_h, in_w, k_h, k_w, out_h, out_w,
                stride_h, stride_w, channels_in, channels_out, batch_size);

            eshkol_gpu_free(&buf_go);
            eshkol_gpu_free(&buf_kw);
            eshkol_gpu_free(&buf_gi);
            eshkol_gpu_free(&buf_si);
            eshkol_gpu_free(&buf_gk);

            if (rc1 == 0 && rc2 == 0) return; /* GPU succeeded */
        }
        /* GPU failed — fall through to CPU */
    }

    /* Parallel backward: batch dimension is embarrassingly parallel.
     * grad_input[b] is independent per batch element.
     * grad_kernel is shared — use per-thread buffers and reduce. */
    size_t kernel_total = (size_t)(channels_out * channels_in * kernel_spatial);

#if defined(_OPENMP)
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif

    if (batch_size > 1 && n_threads > 1) {
        /* Allocate per-thread kernel gradient buffers */
        arena_t* conv_arena = get_global_arena();
        double** thread_gk = (double**)arena_allocate_zeroed(conv_arena, (size_t)n_threads * sizeof(double*));
        if (!thread_gk) return;
        for (int t = 0; t < n_threads; t++) {
            thread_gk[t] = (double*)arena_allocate_zeroed(conv_arena, kernel_total * sizeof(double));
        }

#if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic)
#endif
        for (int64_t b = 0; b < batch_size; b++) {
#if defined(_OPENMP)
            int tid = omp_get_thread_num();
#else
            int tid = 0;
#endif
            double* local_gk = thread_gk[tid];

            for (int64_t co = 0; co < channels_out; co++) {
                for (int64_t ci = 0; ci < channels_in; ci++) {
                    const double* kernel_ptr =
                        saved_kernel + (co * channels_in + ci) * kernel_spatial;
                    double* lgk_ptr =
                        local_gk + (co * channels_in + ci) * kernel_spatial;

                    for (int64_t oh = 0; oh < out_h; oh++) {
                        for (int64_t ow = 0; ow < out_w; ow++) {
                            double g = grad_out[
                                (b * channels_out + co) * out_spatial +
                                oh * out_w + ow];

                            for (int64_t kh = 0; kh < k_h; kh++) {
                                for (int64_t kw = 0; kw < k_w; kw++) {
                                    int64_t ih = oh * stride_h + kh;
                                    int64_t iw = ow * stride_w + kw;

                                    double in_val = saved_input[
                                        (b * channels_in + ci) * in_spatial +
                                        ih * in_w + iw];
                                    /* grad_input: per-batch, no contention */
                                    grad_input[
                                        (b * channels_in + ci) * in_spatial +
                                        ih * in_w + iw]
                                        += g * kernel_ptr[kh * k_w + kw];

                                    /* grad_kernel: per-thread buffer */
                                    lgk_ptr[kh * k_w + kw] += g * in_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        /* Reduce per-thread kernel gradients */
        for (int t = 0; t < n_threads; t++) {
            for (size_t i = 0; i < kernel_total; i++) {
                grad_kernel[i] += thread_gk[t][i];
            }
        }
    } else {
        /* Single-thread fallback (batch_size == 1 or no OpenMP) */
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

                                    double in_val = saved_input[
                                        (b * channels_in + ci) * in_spatial +
                                        ih * in_w + iw];
                                    grad_input[
                                        (b * channels_in + ci) * in_spatial +
                                        ih * in_w + iw]
                                        += g * kernel_ptr[kh * k_w + kw];

                                    gk_ptr[kh * k_w + kw] += g * in_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ===== Normalization ===== */

/** @brief Backward pass for batch normalization: computes grad_gamma and
 *         grad_beta by reducing over the batch dimension per feature, then
 *         computes grad_input using the standard batchnorm backward formula
 *         (scaled by gamma/std, corrected by the per-feature gradient and
 *         gradient-times-normalized-input sums). Parallelized across
 *         features with OpenMP. */
extern "C" void eshkol_backward_batchnorm(
    const double* grad_out,
    const double* saved_input, const double* saved_mean,
    const double* saved_inv_std, const double* saved_gamma,
    double* grad_input, double* grad_gamma, double* grad_beta,
    int64_t batch_size, int64_t feature_size)
{
    double N = (double)batch_size;

    /* Compute grad_beta and grad_gamma with parallel reduction across features.
     * Feature dimension is parallelized — each feature's sum is independent. */
    memset(grad_beta, 0, (size_t)feature_size * sizeof(double));
    memset(grad_gamma, 0, (size_t)feature_size * sizeof(double));

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(feature_size > 64)
#endif
    for (int64_t f = 0; f < feature_size; f++) {
        double sum_g = 0.0, sum_gn = 0.0;
        for (int64_t b = 0; b < batch_size; b++) {
            int64_t idx = b * feature_size + f;
            double g = grad_out[idx];
            double normalized = (saved_input[idx] - saved_mean[f]) * saved_inv_std[f];
            sum_g += g;
            sum_gn += g * normalized;
        }
        grad_beta[f] = sum_g;
        grad_gamma[f] = sum_gn;
    }

    /* Compute grad_input — parallelized across features */
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(feature_size > 64)
#endif
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

/** @brief Backward pass for layer normalization: computes grad_gamma and
 *         grad_beta by reducing over samples per feature, then computes
 *         grad_input per-sample using the standard layernorm backward
 *         formula. Parallelized across samples with OpenMP. */
extern "C" void eshkol_backward_layernorm(
    const double* grad_out,
    const double* saved_input, const double* saved_mean,
    const double* saved_inv_std, const double* saved_gamma,
    double* grad_input, double* grad_gamma, double* grad_beta,
    int64_t num_samples, int64_t feature_size)
{
    double D = (double)feature_size;

    /* Compute grad_beta and grad_gamma — reduce across samples per feature.
     * Each feature's sum is independent → parallelizable. */
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

    /* Compute grad_input per-sample — each sample is independent.
     * Parallelized across samples. */
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if(num_samples > 16)
#endif
    for (int64_t s = 0; s < num_samples; s++) {
        double mean_s = saved_mean[s];
        double inv_std_s = saved_inv_std[s];

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

/* Dispatched backward matmul — uses BLAS/GPU/SIMD via blas_backend.cpp */
extern "C" void eshkol_matmul_backward_f64(
    const double*, const double*, const double*,
    double*, double*, uint64_t, uint64_t, uint64_t);

/** @brief Backward pass for matrix multiplication (out = A @ B): thin
 *         wrapper that forwards to the dispatched eshkol_matmul_backward_f64()
 *         (BLAS/GPU/SIMD) to compute grad_A and grad_B. */
extern "C" void eshkol_backward_matmul(
    const double* grad_out,
    const double* saved_A, const double* saved_B,
    double* grad_A, double* grad_B,
    int64_t M, int64_t K, int64_t N)
{
    eshkol_matmul_backward_f64(grad_out, saved_A, saved_B,
                                grad_A, grad_B,
                                (uint64_t)M, (uint64_t)K, (uint64_t)N);
}

/* ===== Attention ===== */

/** @brief Backward pass for single-head scaled dot-product attention
 *         (scores = Q@K^T*scale, attn = softmax(scores), out = attn@V).
 *         Computes grad_V (attn^T @ grad_out), grad_Q/grad_K via a softmax
 *         backward through the attention weights, using BLAS (cblas_dgemm)
 *         when ESHKOL_BLAS_ENABLED, else naive triple loops. */
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

    /* Allocate temporary for d_attn and d_scores (arena-scoped) */
    arena_t* attn_arena = get_global_arena();
    size_t scores_size = (size_t)(seq_q * seq_k);
    double* d_attn = (double*)arena_allocate_zeroed(attn_arena, scores_size * sizeof(double));
    double* d_scores = (double*)arena_allocate_zeroed(attn_arena, scores_size * sizeof(double));
    if (!d_attn || !d_scores) return;

    /* d_V = attn^T @ grad_out: use dispatched backward matmul pattern
     * attn is (seq_q, seq_k), grad_out is (seq_q, d_v) → need attn^T @ grad_out = (seq_k, d_v)
     * This is: eshkol_matmul_backward_f64 treats this as the A^T @ grad_out case */
#ifdef ESHKOL_BLAS_ENABLED
    memset(grad_V, 0, (size_t)(seq_k * d_v) * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                (int)seq_k, (int)d_v, (int)seq_q,
                1.0, saved_attn_weights, (int)seq_k,
                grad_out, (int)d_v,
                0.0, grad_V, (int)d_v);
#else
    memset(grad_V, 0, (size_t)(seq_k * d_v) * sizeof(double));
    for (int64_t k = 0; k < seq_k; k++)
        for (int64_t d = 0; d < d_v; d++)
            for (int64_t q = 0; q < seq_q; q++)
                grad_V[k * d_v + d] += saved_attn_weights[q * seq_k + k] * grad_out[q * d_v + d];
#endif

    /* d_attn = grad_out @ V^T: (seq_q, d_v) @ (d_v, seq_k) -> (seq_q, seq_k) */
#ifdef ESHKOL_BLAS_ENABLED
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)seq_q, (int)seq_k, (int)d_v,
                1.0, grad_out, (int)d_v,
                saved_V, (int)d_v,
                0.0, d_attn, (int)seq_k);
#else
    for (int64_t q = 0; q < seq_q; q++)
        for (int64_t k = 0; k < seq_k; k++) {
            double sum = 0.0;
            for (int64_t d = 0; d < d_v; d++)
                sum += grad_out[q * d_v + d] * saved_V[k * d_v + d];
            d_attn[q * seq_k + k] = sum;
        }
#endif

    /* Softmax backward: d_scores[i][j] = attn[i][j] * (d_attn[i][j] - dot(attn[i], d_attn[i])) */
    for (int64_t q = 0; q < seq_q; q++) {
        double dot = 0.0;
        for (int64_t k = 0; k < seq_k; k++)
            dot += saved_attn_weights[q * seq_k + k] * d_attn[q * seq_k + k];
        for (int64_t k = 0; k < seq_k; k++)
            d_scores[q * seq_k + k] = saved_attn_weights[q * seq_k + k] * (d_attn[q * seq_k + k] - dot) * scale;
    }

    /* d_Q = d_scores @ K: (seq_q, seq_k) @ (seq_k, d_k) -> (seq_q, d_k) */
#ifdef ESHKOL_BLAS_ENABLED
    memset(grad_Q, 0, (size_t)(seq_q * d_k) * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)seq_q, (int)d_k, (int)seq_k,
                1.0, d_scores, (int)seq_k,
                saved_K, (int)d_k,
                0.0, grad_Q, (int)d_k);
#else
    memset(grad_Q, 0, (size_t)(seq_q * d_k) * sizeof(double));
    for (int64_t q = 0; q < seq_q; q++)
        for (int64_t d = 0; d < d_k; d++)
            for (int64_t k = 0; k < seq_k; k++)
                grad_Q[q * d_k + d] += d_scores[q * seq_k + k] * saved_K[k * d_k + d];
#endif

    /* d_K = d_scores^T @ Q: (seq_k, seq_q) @ (seq_q, d_k) -> (seq_k, d_k) */
#ifdef ESHKOL_BLAS_ENABLED
    memset(grad_K, 0, (size_t)(seq_k * d_k) * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                (int)seq_k, (int)d_k, (int)seq_q,
                1.0, d_scores, (int)seq_k,
                saved_Q, (int)d_k,
                0.0, grad_K, (int)d_k);
#else
    memset(grad_K, 0, (size_t)(seq_k * d_k) * sizeof(double));
    for (int64_t k = 0; k < seq_k; k++)
        for (int64_t d = 0; d < d_k; d++)
            for (int64_t q = 0; q < seq_q; q++)
                grad_K[k * d_k + d] += d_scores[q * seq_k + k] * saved_Q[q * d_k + d];
#endif

}

/** @brief Backward pass for multi-head attention. Backprops grad_out through
 *         the output projection W_O to get d_concat, then for each head
 *         slices out its portion, re-derives that head's projected Q/K/V
 *         from the saved pre-projection tensors and per-head weight slices,
 *         calls eshkol_backward_attention() for that head, and accumulates
 *         the resulting per-head gradients into grad_Q/K/V and the
 *         grad_W{Q,K,V} projection-weight gradients (grad_WO accumulation
 *         is not yet implemented here — see inline TODO). Uses BLAS
 *         (cblas_dgemm) when ESHKOL_BLAS_ENABLED, else naive loops. */
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

    /* Allocate temporaries for per-head gradients (arena-scoped) */
    arena_t* mha_arena = get_global_arena();
    size_t head_q_size = (size_t)(seq_q * head_dim);
    size_t head_k_size = (size_t)(seq_k * head_dim);
    double* d_head_Q = (double*)arena_allocate_zeroed(mha_arena, head_q_size * sizeof(double));
    double* d_head_K = (double*)arena_allocate_zeroed(mha_arena, head_k_size * sizeof(double));
    double* d_head_V = (double*)arena_allocate_zeroed(mha_arena, head_k_size * sizeof(double));
    if (!d_head_Q || !d_head_K || !d_head_V) return;

    size_t concat_size = (size_t)(seq_q * d_model);
    double* d_concat = (double*)arena_allocate_zeroed(mha_arena, concat_size * sizeof(double));
    if (!d_concat) return;

    /* d_concat = grad_out @ W_O^T: (seq_q, d_model) @ (d_model, d_model) */
    memset(d_concat, 0, concat_size * sizeof(double));
#ifdef ESHKOL_BLAS_ENABLED
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)seq_q, (int)d_model, (int)d_model,
                1.0, grad_out, (int)d_model,
                saved_W_O, (int)d_model,
                0.0, d_concat, (int)d_model);
#else
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
#endif

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
        double* d_head_out = (double*)arena_allocate_zeroed(mha_arena,
            (size_t)(seq_q * head_dim) * sizeof(double));
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
        double* head_Q = (double*)arena_allocate_zeroed(mha_arena, head_q_size * sizeof(double));
        double* head_K = (double*)arena_allocate_zeroed(mha_arena, head_k_size * sizeof(double));
        double* head_V = (double*)arena_allocate_zeroed(mha_arena, head_k_size * sizeof(double));
        if (!head_Q || !head_K || !head_V) continue;

        /* Project Q, K, V through head-specific weight slices
         * head_Q = saved_Q @ W_Q[:, h_offset:h_offset+head_dim]
         * W_Q is (d_model, d_model) row-major; slice has lda=d_model */
#ifdef ESHKOL_BLAS_ENABLED
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)seq_q, (int)head_dim, (int)d_model,
                    1.0, saved_Q, (int)d_model,
                    saved_W_Q + h_offset, (int)d_model,
                    0.0, head_Q, (int)head_dim);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)seq_k, (int)head_dim, (int)d_model,
                    1.0, saved_K, (int)d_model,
                    saved_W_K + h_offset, (int)d_model,
                    0.0, head_K, (int)head_dim);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)seq_k, (int)head_dim, (int)d_model,
                    1.0, saved_V, (int)d_model,
                    saved_W_V + h_offset, (int)d_model,
                    0.0, head_V, (int)head_dim);
#else
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
#endif

        /* Attention backward for this head */
        eshkol_backward_attention(
            d_head_out,
            head_Q, head_K, head_V,
            head_attn,
            d_head_Q, d_head_K, d_head_V,
            seq_q, seq_k, head_dim, head_dim, scale);

        /* Accumulate weight gradients:
         * grad_WQ_h += Q^T @ d_head_Q, etc.
         * grad_WQ is (d_model, d_model); slice at col h_offset has lda=d_model
         * beta=1.0 to accumulate across heads */
#ifdef ESHKOL_BLAS_ENABLED
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    (int)d_model, (int)head_dim, (int)seq_q,
                    1.0, saved_Q, (int)d_model,
                    d_head_Q, (int)head_dim,
                    1.0, grad_WQ + h_offset, (int)d_model);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    (int)d_model, (int)head_dim, (int)seq_k,
                    1.0, saved_K, (int)d_model,
                    d_head_K, (int)head_dim,
                    1.0, grad_WK + h_offset, (int)d_model);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    (int)d_model, (int)head_dim, (int)seq_k,
                    1.0, saved_V, (int)d_model,
                    d_head_V, (int)head_dim,
                    1.0, grad_WV + h_offset, (int)d_model);
#else
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
#endif

        /* Backprop through projection: grad_Q += d_head_Q @ W_Q_h^T
         * d_head_Q is (seq_q, head_dim), W_Q_h slice is (d_model, head_dim) with lda=d_model
         * Result: (seq_q, d_model), beta=1.0 to accumulate across heads */
#ifdef ESHKOL_BLAS_ENABLED
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)seq_q, (int)d_model, (int)head_dim,
                    1.0, d_head_Q, (int)head_dim,
                    saved_W_Q + h_offset, (int)d_model,
                    1.0, grad_Q, (int)d_model);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)seq_k, (int)d_model, (int)head_dim,
                    1.0, d_head_K, (int)head_dim,
                    saved_W_K + h_offset, (int)d_model,
                    1.0, grad_K, (int)d_model);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)seq_k, (int)d_model, (int)head_dim,
                    1.0, d_head_V, (int)head_dim,
                    saved_W_V + h_offset, (int)d_model,
                    1.0, grad_V, (int)d_model);
#else
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
#endif

    }

}

/* ===== Embedding ===== */

/** @brief Backward pass for embedding lookup: scatter-adds each row of
 *         grad_out into the weight-gradient row indicated by the
 *         corresponding saved index (out-of-range indices are skipped). */
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

/** @brief Seed the gradient of a tensor-valued AD node with all-ones
 *         (dL/dL = 1 for every output element), allocating tensor_gradient
 *         from the global arena on first use. No-op if the node isn't a
 *         tensor node or its shape/total-element count is invalid. */
extern "C" void eshkol_seed_tensor_gradient(void* ad_node_ptr) {
    if (!ad_node_ptr) return;

    ad_node_t* node = (ad_node_t*)ad_node_ptr;

    /* Only seed if this is a tensor node (has tensor_value set) */
    if (!node->tensor_value) return;

    /* Compute total elements from shape (with overflow check) */
    if (node->ndim > 0 && !node->shape) return;
    int64_t total = 1;
    for (size_t i = 0; i < node->ndim; i++) {
        if (!safe_mul(total, node->shape[i], &total)) return;
    }
    if (total <= 0) return;

    /* Allocate tensor_gradient if not already set */
    if (!node->tensor_gradient) {
        node->tensor_gradient = arena_allocate_zeroed(get_global_arena(), (size_t)total * sizeof(double));
        if (!node->tensor_gradient) return;
    }

    /* Fill with all ones: dL/dL = 1.0 for every output element */
    double* tg = (double*)node->tensor_gradient;
    for (int64_t i = 0; i < total; i++) {
        tg[i] = 1.0;
    }
}

/* ===== Tensor Backward Dispatcher ===== */

/** @brief Compute the overflow-checked product of a shape array's
 *         dimensions; returns 0 on invalid input or overflow. */
static int64_t compute_total_elements(const int64_t* shape, size_t ndim) {
    if (ndim > 0 && !shape) return 0;
    int64_t total = 1;
    for (size_t i = 0; i < ndim; i++) {
        if (!safe_mul(total, shape[i], &total)) return 0;
    }
    return total;
}

/** @brief Central reverse-mode AD dispatcher for tensor operation nodes:
 *         switches on @p ad_node_ptr's node type (conv2d, pooling, norm
 *         layers, matmul, transpose, reshape, reductions, attention,
 *         embedding, and qLLM-bridge tensor ops), reconstructs the needed
 *         shapes/params from the node, invokes the matching
 *         eshkol_backward_* function (or the bridge dispatch table for
 *         AD_NODE_TENSOR_* ops), and accumulates the resulting input
 *         gradients via eshkol_accumulate_tensor_grad(). All temporary
 *         gradient buffers are allocated from an arena scope that is popped
 *         before returning; node types without a backward implementation
 *         silently drop the gradient. */
extern "C" void eshkol_tensor_backward_dispatch(void* ad_node_ptr) {
    if (!ad_node_ptr) return;

    ad_node_t* node = (ad_node_t*)ad_node_ptr;
    double* upstream_grad = (double*)node->tensor_gradient;
    if (!upstream_grad) return;

    /* Access params as raw int64_t array (matches union layout) */
    int64_t* p = (int64_t*)&node->params;

    /* Arena-scoped allocation: all gradient temporaries below are allocated from
     * the global arena within a pushed scope. When we pop at function exit, the
     * memory is bulk-freed. Gradients are accumulated into AD nodes (which live
     * outside this scope) via eshkol_accumulate_tensor_grad before the pop. */
    arena_t* bwd_arena = backward_scope_begin();
    if (!bwd_arena) return;

    switch (node->type) {

    /* --- CONV2D (19) --- */
    case AD_NODE_CONV2D: {
        if (!node->saved_tensors || node->num_saved < 2) break;
        if (node->ndim > 0 && !node->shape) break;
        const double* saved_input  = (const double*)node->saved_tensors[0];
        const double* saved_kernel = (const double*)node->saved_tensors[1];
        int64_t k_h = p[0], k_w = p[1];
        int64_t stride_h = p[2], stride_w = p[3];
        int64_t channels_in = p[4], channels_out = p[5];
        int64_t batch_size = (node->ndim >= 1) ? node->shape[0] : 1;
        int64_t out_h = (node->ndim >= 3) ? node->shape[2] : 1;
        int64_t out_w = (node->ndim >= 4) ? node->shape[3] : 1;
        int64_t in_h = (out_h - 1) * stride_h + k_h;
        int64_t in_w = (out_w - 1) * stride_w + k_w;
        int64_t input_total, kernel_total;
        if (!safe_mul4(batch_size, channels_in, in_h, in_w, &input_total)) break;
        if (!safe_mul4(channels_out, channels_in, k_h, k_w, &kernel_total)) break;
        double* grad_input  = (double*)arena_calloc(bwd_arena, (size_t)input_total * sizeof(double));
        double* grad_kernel = (double*)arena_calloc(bwd_arena, (size_t)kernel_total * sizeof(double));
        if (grad_input && grad_kernel) {
            eshkol_backward_conv2d(upstream_grad, saved_input, saved_kernel,
                grad_input, grad_kernel, in_h, in_w, k_h, k_w, out_h, out_w,
                stride_h, stride_w, channels_in, channels_out, batch_size);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
            if (node->input2)
                eshkol_accumulate_tensor_grad(node->input2, grad_kernel, kernel_total);
        }
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
        double* grad_input = (double*)arena_calloc(bwd_arena,(size_t)input_total * sizeof(double));
        if (grad_input) {
            eshkol_backward_maxpool2d(upstream_grad, grad_input, max_indices,
                in_h, in_w, out_h, out_w, k_h, k_w, stride_h, stride_w,
                channels, batch_size);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
        }
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
        double* grad_input = (double*)arena_calloc(bwd_arena,(size_t)input_total * sizeof(double));
        if (grad_input) {
            eshkol_backward_avgpool2d(upstream_grad, grad_input,
                in_h, in_w, out_h, out_w, k_h, k_w, stride_h, stride_w,
                channels, batch_size);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
        }
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
        double* grad_input = (double*)arena_calloc(bwd_arena,(size_t)input_total * sizeof(double));
        double* grad_gamma = (double*)arena_calloc(bwd_arena,(size_t)feature_size * sizeof(double));
        double* grad_beta  = (double*)arena_calloc(bwd_arena,(size_t)feature_size * sizeof(double));
        if (grad_input && grad_gamma && grad_beta) {
            eshkol_backward_batchnorm(upstream_grad, saved_input, saved_mean,
                saved_inv_std, saved_gamma, grad_input, grad_gamma, grad_beta,
                batch_size, feature_size);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
            if (node->input2)
                eshkol_accumulate_tensor_grad(node->input2, grad_gamma, feature_size);
        }
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
        double* grad_input = (double*)arena_calloc(bwd_arena,(size_t)input_total * sizeof(double));
        double* grad_gamma = (double*)arena_calloc(bwd_arena,(size_t)feature_size * sizeof(double));
        double* grad_beta  = (double*)arena_calloc(bwd_arena,(size_t)feature_size * sizeof(double));
        if (grad_input && grad_gamma && grad_beta) {
            eshkol_backward_layernorm(upstream_grad, saved_input, saved_mean,
                saved_inv_std, saved_gamma, grad_input, grad_gamma, grad_beta,
                num_samples, feature_size);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
            if (node->input2)
                eshkol_accumulate_tensor_grad(node->input2, grad_gamma, feature_size);
        }
        break;
    }

    /* --- MATMUL (24) --- */
    case AD_NODE_MATMUL: {
        if (!node->saved_tensors || node->num_saved < 2) break;
        const double* saved_A = (const double*)node->saved_tensors[0];
        const double* saved_B = (const double*)node->saved_tensors[1];
        int64_t M = p[0], K = p[1], N = p[2];
        double* grad_A = (double*)arena_calloc(bwd_arena,(size_t)(M * K) * sizeof(double));
        double* grad_B = (double*)arena_calloc(bwd_arena,(size_t)(K * N) * sizeof(double));
        if (grad_A && grad_B) {
            eshkol_backward_matmul(upstream_grad, saved_A, saved_B,
                grad_A, grad_B, M, K, N);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_A, M * K);
            if (node->input2)
                eshkol_accumulate_tensor_grad(node->input2, grad_B, K * N);
        }
        break;
    }

    /* --- TRANSPOSE (25) --- */
    case AD_NODE_TRANSPOSE: {
        int64_t rows = p[0], cols = p[1];
        /* Input has shape (cols, rows) — transpose of output (rows, cols) */
        int64_t input_total = rows * cols;
        double* grad_input = (double*)arena_calloc(bwd_arena,(size_t)input_total * sizeof(double));
        if (grad_input) {
            eshkol_backward_transpose(upstream_grad, grad_input, rows, cols);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
        }
        break;
    }

    /* --- RESHAPE (26) --- */
    case AD_NODE_RESHAPE: {
        int64_t total_elements = p[0];
        double* grad_input = (double*)arena_calloc(bwd_arena,(size_t)total_elements * sizeof(double));
        if (grad_input) {
            eshkol_backward_reshape(upstream_grad, grad_input, total_elements);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, total_elements);
        }
        break;
    }

    /* --- SUM (27) --- */
    case AD_NODE_SUM: {
        int64_t input_total = p[0];
        /* Sum output is scalar; upstream_grad is a 1-element tensor */
        double grad_scalar = upstream_grad[0];
        double* grad_input = (double*)arena_calloc(bwd_arena,(size_t)input_total * sizeof(double));
        if (grad_input) {
            eshkol_backward_sum(grad_scalar, grad_input, input_total);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
        }
        break;
    }

    /* --- MEAN (28) --- */
    case AD_NODE_MEAN: {
        int64_t input_total = p[0];
        double grad_scalar = upstream_grad[0];
        double* grad_input = (double*)arena_calloc(bwd_arena,(size_t)input_total * sizeof(double));
        if (grad_input) {
            eshkol_backward_mean(grad_scalar, grad_input, input_total);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, input_total);
        }
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
        double* grad_Q = (double*)arena_calloc(bwd_arena,(size_t)(seq_q * d_k) * sizeof(double));
        double* grad_K = (double*)arena_calloc(bwd_arena,(size_t)(seq_k * d_k) * sizeof(double));
        double* grad_V = (double*)arena_calloc(bwd_arena,(size_t)(seq_k * d_v) * sizeof(double));
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
        double* grad_Q  = (double*)arena_calloc(bwd_arena,(size_t)(seq_q * d_model) * sizeof(double));
        double* grad_K  = (double*)arena_calloc(bwd_arena,(size_t)(seq_k * d_model) * sizeof(double));
        double* grad_V  = (double*)arena_calloc(bwd_arena,(size_t)(seq_k * d_model) * sizeof(double));
        double* grad_WQ = (double*)arena_calloc(bwd_arena,(size_t)(d_model * d_model) * sizeof(double));
        double* grad_WK = (double*)arena_calloc(bwd_arena,(size_t)(d_model * d_model) * sizeof(double));
        double* grad_WV = (double*)arena_calloc(bwd_arena,(size_t)(d_model * d_model) * sizeof(double));
        double* grad_WO = (double*)arena_calloc(bwd_arena,(size_t)(d_model * d_model) * sizeof(double));
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
        break;
    }

    /* --- POSITIONAL_ENCODING (31) --- */
    case AD_NODE_POSITIONAL_ENCODING: {
        int64_t total_elements = p[0];
        if (total_elements <= 0 && node->ndim > 0)
            total_elements = compute_total_elements(node->shape, node->ndim);
        double* grad_input = (double*)arena_calloc(bwd_arena,(size_t)total_elements * sizeof(double));
        if (grad_input) {
            eshkol_backward_positional_encoding(upstream_grad, grad_input, total_elements);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_input, total_elements);
        }
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
        double* grad_weights = (double*)arena_calloc(bwd_arena,(size_t)weight_total * sizeof(double));
        if (grad_weights) {
            eshkol_backward_embedding(upstream_grad, saved_indices,
                grad_weights, num_indices, d_model, vocab_size);
            if (node->input1)
                eshkol_accumulate_tensor_grad(node->input1, grad_weights, weight_total);
        }
        break;
    }

    /* ===== qLLM Bridge Tensor Ops (67-79) =====
     * These delegate to the self-contained backward functions in
     * lib/bridge/tensor_backward.cpp. Each function reads the node's
     * tensor_value, tensor_gradient, input1/input2 and propagates
     * gradients internally — no manual buffer management needed.  */

    case AD_NODE_TENSOR_MATMUL:
    case AD_NODE_TENSOR_SOFTMAX:
    case AD_NODE_TENSOR_LAYERNORM:
    case AD_NODE_TENSOR_RMSNORM:
    case AD_NODE_TENSOR_GELU:
    case AD_NODE_TENSOR_SILU:
    case AD_NODE_TENSOR_CROSS_ENTROPY:
    /* Previously these fell into a "gradient passthrough" case that silently
     * dropped the gradient even though get_tensor_backward_fn HAS a registered
     * backward for each. Route them through the dispatch table like the ops
     * above. attention/embedding backends now raise an explicit unsupported-op
     * error instead of returning a plausible-but-wrong gradient (hard
     * constraint: exact AD or an explicit error, never a silent zero). */
    case AD_NODE_TENSOR_ATTENTION:
    case AD_NODE_TENSOR_TRANSPOSE:
    case AD_NODE_TENSOR_SUM:
    case AD_NODE_TENSOR_BROADCAST_ADD:
    case AD_NODE_TENSOR_BROADCAST_MUL:
    case AD_NODE_TENSOR_EMBEDDING: {
        /* Use the bridge dispatch table to find the right backward fn */
        bridge_backward_fn_t fn = get_tensor_backward_fn((int)node->type);
        if (!fn) {
            eshkol_fatal("unsupported AD op: no backward registered for tensor "
                         "bridge node type %d; refusing to drop the gradient "
                         "silently", (int)node->type);
        }
        fn(node);
        break;
    }

    case AD_NODE_FRECHET_MEAN:
        /* No exact backward for the Riemannian center-of-mass node yet. Refuse
         * rather than leave the gradient at a silent zero. */
        eshkol_fatal("unsupported AD op: exact backward for the Frechet-mean "
                     "tensor node is not implemented; refusing to drop the "
                     "gradient silently.");
        break;

    default:
        /* Scalar activation / geometric / hyperbolic ops (12-18, 33-66) are
         * differentiated by their scalar backward emitted in codegen and
         * legitimately reach here with nothing to do. Every TENSOR op
         * (19-32 and the qLLM bridge ops 67-80) has an explicit case above, so
         * a tensor-op gradient can never silently fall through this default. */
        break;
    }

    /* Pop arena scope — all temporary gradient buffers are bulk-freed here.
     * Accumulated gradients in AD nodes (via eshkol_accumulate_tensor_grad)
     * live in the unscoped global arena and survive this pop. */
    backward_scope_end(bwd_arena);
}

/* ===== Tensor Gradient Accumulation ===== */

/** @brief Element-wise accumulate @p grad into an AD node's tensor_gradient
 *         buffer (allocating and zero-filling it from the global arena on
 *         first use). */
extern "C" void eshkol_accumulate_tensor_grad(
    void* ad_node_ptr, const double* grad, int64_t num_elements)
{
    if (!ad_node_ptr || !grad || num_elements <= 0) return;

    ad_node_t* node = (ad_node_t*)ad_node_ptr;

    /* If tensor_gradient is NULL, allocate and zero-fill */
    if (!node->tensor_gradient) {
        node->tensor_gradient = arena_allocate_zeroed(get_global_arena(),
            (size_t)num_elements * sizeof(double));
        if (!node->tensor_gradient) return;
    }

    /* Element-wise accumulate: tensor_gradient[i] += grad[i] */
    double* tg = (double*)node->tensor_gradient;
    for (int64_t i = 0; i < num_elements; i++) {
        tg[i] += grad[i];
    }
}
