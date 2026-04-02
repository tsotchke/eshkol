/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Tensor Backward Pass Runtime Functions
 *
 * These extern "C" functions implement the backward passes for tensor
 * operations. They are called from LLVM-generated code during reverse-mode
 * automatic differentiation. Each function computes gradients for its inputs
 * given the upstream gradient (grad_output).
 *
 * Architecture: Forward passes in tensor_codegen.cpp record AD nodes with
 * tensor data on the tape. During backpropagation, propagateGradient()
 * dispatches to these runtime functions which operate on raw double arrays.
 * This matches PyTorch's autograd architecture.
 */
#ifndef ESHKOL_BACKEND_TENSOR_BACKWARD_H
#define ESHKOL_BACKEND_TENSOR_BACKWARD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ===== Structural Operations ===== */

/* Transpose backward: grad_in[j*rows + i] = grad_out[i*cols + j] */
void eshkol_backward_transpose(
    const double* grad_out, double* grad_in,
    int64_t rows, int64_t cols);

/* Reshape backward: gradient shape changes but values are identical */
void eshkol_backward_reshape(
    const double* grad_out, double* grad_in,
    int64_t total_elements);

/* Positional encoding backward: additive constant, gradient passes through */
void eshkol_backward_positional_encoding(
    const double* grad_out, double* grad_in,
    int64_t total_elements);

/* ===== Reduction Operations ===== */

/* Sum backward: broadcast scalar gradient to all elements */
void eshkol_backward_sum(
    double grad_out_scalar, double* grad_in,
    int64_t total_elements);

/* Mean backward: distribute scalar gradient equally */
void eshkol_backward_mean(
    double grad_out_scalar, double* grad_in,
    int64_t total_elements);

/* ===== Pooling Operations ===== */

/* MaxPool2d backward: scatter gradient to max element positions */
void eshkol_backward_maxpool2d(
    const double* grad_out, double* grad_in,
    const int64_t* max_indices,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t channels, int64_t batch_size);

/* AvgPool2d backward: distribute gradient uniformly across pooling window */
void eshkol_backward_avgpool2d(
    const double* grad_out, double* grad_in,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t channels, int64_t batch_size);

/* ===== Convolution ===== */

/* Conv2d backward: compute grad_input and grad_kernel */
void eshkol_backward_conv2d(
    const double* grad_out,
    const double* saved_input, const double* saved_kernel,
    double* grad_input, double* grad_kernel,
    int64_t in_h, int64_t in_w,
    int64_t k_h, int64_t k_w,
    int64_t out_h, int64_t out_w,
    int64_t stride_h, int64_t stride_w,
    int64_t channels_in, int64_t channels_out,
    int64_t batch_size);

/* ===== Normalization ===== */

/* BatchNorm backward: compute grad_input, grad_gamma, grad_beta */
void eshkol_backward_batchnorm(
    const double* grad_out,
    const double* saved_input, const double* saved_mean,
    const double* saved_inv_std, const double* saved_gamma,
    double* grad_input, double* grad_gamma, double* grad_beta,
    int64_t batch_size, int64_t feature_size);

/* LayerNorm backward: per-sample normalization gradients */
void eshkol_backward_layernorm(
    const double* grad_out,
    const double* saved_input, const double* saved_mean,
    const double* saved_inv_std, const double* saved_gamma,
    double* grad_input, double* grad_gamma, double* grad_beta,
    int64_t num_samples, int64_t feature_size);

/* ===== Attention ===== */

/* Single-head attention backward: compute dQ, dK, dV */
void eshkol_backward_attention(
    const double* grad_out,
    const double* saved_Q, const double* saved_K, const double* saved_V,
    const double* saved_attn_weights,
    double* grad_Q, double* grad_K, double* grad_V,
    int64_t seq_q, int64_t seq_k, int64_t d_k, int64_t d_v,
    double scale);

/* Multi-head attention backward: compute gradients for all projections */
void eshkol_backward_multihead_attention(
    const double* grad_out,
    const double* saved_Q, const double* saved_K, const double* saved_V,
    const double* saved_W_Q, const double* saved_W_K,
    const double* saved_W_V, const double* saved_W_O,
    const double* saved_attn_per_head,
    double* grad_Q, double* grad_K, double* grad_V,
    double* grad_WQ, double* grad_WK, double* grad_WV, double* grad_WO,
    int64_t seq_q, int64_t seq_k, int64_t d_model, int64_t num_heads);

/* ===== Embedding ===== */

/* Embedding backward: scatter-add gradients to weight matrix */
void eshkol_backward_embedding(
    const double* grad_out,
    const int64_t* saved_indices,
    double* grad_weights,
    int64_t num_indices, int64_t d_model, int64_t vocab_size);

/* ===== Matmul ===== */

/* Matmul backward: dA = grad_C @ B^T, dB = A^T @ grad_C */
void eshkol_backward_matmul(
    const double* grad_out,
    const double* saved_A, const double* saved_B,
    double* grad_A, double* grad_B,
    int64_t M, int64_t K, int64_t N);

/* ===== Tensor Gradient Accumulation ===== */

/* Accumulate gradient into an AD node's tensor_gradient field.
 * If tensor_gradient is NULL, allocates and zero-fills it first.
 * Thread-safe for single-threaded backward pass. */
void eshkol_accumulate_tensor_grad(
    void* ad_node_ptr, const double* grad, int64_t num_elements);

/* ===== Tensor Gradient Seeding ===== */

/* Seed the output node's tensor gradient for backpropagation.
 * If the node has tensor_value (non-null), allocates tensor_gradient
 * and fills it with all ones (dL/dL = 1 for every element).
 * Called at the start of backpropagate() on the output node. */
void eshkol_seed_tensor_gradient(void* ad_node_ptr);

/* ===== Tensor Backward Dispatcher ===== */

/* Main entry point called from LLVM-generated backward pass.
 * Reads the AD node's type, tensor_gradient (upstream grad),
 * saved_tensors, params, shape/ndim, then dispatches to the
 * appropriate eshkol_backward_* function and accumulates
 * gradients on input nodes.
 *
 * Convention for params (stored as 6 x int64_t):
 *   Conv2d:     [kernel_h, kernel_w, stride_h, stride_w, channels_in, channels_out]
 *   MaxPool2d:  [kernel_h, kernel_w, stride_h, stride_w, channels, batch_size]
 *   AvgPool2d:  [kernel_h, kernel_w, stride_h, stride_w, channels, batch_size]
 *   BatchNorm:  [batch_size, feature_size, 0, 0, 0, 0]
 *   LayerNorm:  [num_samples, feature_size, 0, 0, 0, 0]
 *   Matmul:     [M, K, N, 0, 0, 0]
 *   Transpose:  [rows, cols, 0, 0, 0, 0]
 *   Reshape:    [total_elements, 0, 0, 0, 0, 0]
 *   Sum:        [input_total_elements, 0, 0, 0, 0, 0]
 *   Mean:       [input_total_elements, 0, 0, 0, 0, 0]
 *   Attention:  [seq_q, seq_k, d_k, d_v, scale_bits, 0]
 *   MHA:        [seq_q, seq_k, d_model, num_heads, 0, 0]
 *   PosEnc:     [total_elements, 0, 0, 0, 0, 0]
 *   Embedding:  [num_indices, d_model, vocab_size, 0, 0, 0]
 */
void eshkol_tensor_backward_dispatch(void* ad_node_ptr);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_BACKEND_TENSOR_BACKWARD_H */
