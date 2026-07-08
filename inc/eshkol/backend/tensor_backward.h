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

/**
 * @brief Transpose backward pass: grad_in[j*rows + i] = grad_out[i*cols + j].
 * @param grad_out Upstream gradient (rows x cols, row-major)
 * @param grad_in Output gradient buffer (cols x rows, row-major)
 * @param rows Rows of the original (forward) input
 * @param cols Columns of the original (forward) input
 */
void eshkol_backward_transpose(
    const double* grad_out, double* grad_in,
    int64_t rows, int64_t cols);

/**
 * @brief Reshape backward pass: gradient shape changes but values are identical.
 * @param grad_out Upstream gradient (flat)
 * @param grad_in Output gradient buffer (flat, same total size)
 * @param total_elements Number of elements in both buffers
 */
void eshkol_backward_reshape(
    const double* grad_out, double* grad_in,
    int64_t total_elements);

/**
 * @brief Positional-encoding backward pass: additive constant, gradient passes through unchanged.
 * @param grad_out Upstream gradient
 * @param grad_in Output gradient buffer
 * @param total_elements Number of elements
 */
void eshkol_backward_positional_encoding(
    const double* grad_out, double* grad_in,
    int64_t total_elements);

/* ===== Reduction Operations ===== */

/**
 * @brief Sum backward pass: broadcast the scalar upstream gradient to all input elements.
 * @param grad_out_scalar Upstream scalar gradient
 * @param grad_in Output gradient buffer (one entry per input element)
 * @param total_elements Number of input elements
 */
void eshkol_backward_sum(
    double grad_out_scalar, double* grad_in,
    int64_t total_elements);

/**
 * @brief Mean backward pass: distribute the scalar upstream gradient equally across inputs.
 * @param grad_out_scalar Upstream scalar gradient
 * @param grad_in Output gradient buffer (one entry per input element)
 * @param total_elements Number of input elements
 */
void eshkol_backward_mean(
    double grad_out_scalar, double* grad_in,
    int64_t total_elements);

/* ===== Pooling Operations ===== */

/**
 * @brief MaxPool2d backward pass: scatter the upstream gradient to each window's max element.
 * @param grad_out Upstream gradient over the pooled output
 * @param grad_in Output gradient buffer over the pre-pooling input (zero-filled elsewhere)
 * @param max_indices Flat index of the max element chosen in each pooling window (forward pass)
 * @param in_h Input height
 * @param in_w Input width
 * @param out_h Output height
 * @param out_w Output width
 * @param kernel_h Pooling kernel height
 * @param kernel_w Pooling kernel width
 * @param stride_h Pooling stride (rows)
 * @param stride_w Pooling stride (columns)
 * @param channels Number of channels
 * @param batch_size Batch size
 */
void eshkol_backward_maxpool2d(
    const double* grad_out, double* grad_in,
    const int64_t* max_indices,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t channels, int64_t batch_size);

/**
 * @brief AvgPool2d backward pass: distribute the upstream gradient uniformly across each pooling window.
 * @param grad_out Upstream gradient over the pooled output
 * @param grad_in Output gradient buffer over the pre-pooling input
 * @param in_h Input height
 * @param in_w Input width
 * @param out_h Output height
 * @param out_w Output width
 * @param kernel_h Pooling kernel height
 * @param kernel_w Pooling kernel width
 * @param stride_h Pooling stride (rows)
 * @param stride_w Pooling stride (columns)
 * @param channels Number of channels
 * @param batch_size Batch size
 */
void eshkol_backward_avgpool2d(
    const double* grad_out, double* grad_in,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t channels, int64_t batch_size);

/* ===== Convolution ===== */

/**
 * @brief Conv2d backward pass: computes grad_input and grad_kernel.
 * @param grad_out Upstream gradient over the convolution output
 * @param saved_input Forward-pass input (saved for the kernel gradient)
 * @param saved_kernel Forward-pass kernel weights (saved for the input gradient)
 * @param grad_input Output gradient buffer with respect to the input
 * @param grad_kernel Output gradient buffer with respect to the kernel weights
 * @param in_h Input height
 * @param in_w Input width
 * @param k_h Kernel height
 * @param k_w Kernel width
 * @param out_h Output height
 * @param out_w Output width
 * @param stride_h Convolution stride (rows)
 * @param stride_w Convolution stride (columns)
 * @param channels_in Input channel count
 * @param channels_out Output channel count
 * @param batch_size Batch size
 */
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

/**
 * @brief BatchNorm backward pass: computes grad_input, grad_gamma, and grad_beta.
 * @param grad_out Upstream gradient over the normalized output
 * @param saved_input Forward-pass input
 * @param saved_mean Forward-pass batch mean
 * @param saved_inv_std Forward-pass inverse standard deviation
 * @param saved_gamma Forward-pass scale parameter
 * @param grad_input Output gradient buffer with respect to the input
 * @param grad_gamma Output gradient buffer with respect to gamma
 * @param grad_beta Output gradient buffer with respect to beta
 * @param batch_size Batch size
 * @param feature_size Number of features per sample
 */
void eshkol_backward_batchnorm(
    const double* grad_out,
    const double* saved_input, const double* saved_mean,
    const double* saved_inv_std, const double* saved_gamma,
    double* grad_input, double* grad_gamma, double* grad_beta,
    int64_t batch_size, int64_t feature_size);

/**
 * @brief LayerNorm backward pass: per-sample normalization gradients.
 * @param grad_out Upstream gradient over the normalized output
 * @param saved_input Forward-pass input
 * @param saved_mean Forward-pass per-sample mean
 * @param saved_inv_std Forward-pass per-sample inverse standard deviation
 * @param saved_gamma Forward-pass scale parameter
 * @param grad_input Output gradient buffer with respect to the input
 * @param grad_gamma Output gradient buffer with respect to gamma
 * @param grad_beta Output gradient buffer with respect to beta
 * @param num_samples Number of samples
 * @param feature_size Number of features per sample
 */
void eshkol_backward_layernorm(
    const double* grad_out,
    const double* saved_input, const double* saved_mean,
    const double* saved_inv_std, const double* saved_gamma,
    double* grad_input, double* grad_gamma, double* grad_beta,
    int64_t num_samples, int64_t feature_size);

/* ===== Attention ===== */

/**
 * @brief Single-head attention backward pass: computes dQ, dK, dV.
 * @param grad_out Upstream gradient over the attention output
 * @param saved_Q Forward-pass query matrix
 * @param saved_K Forward-pass key matrix
 * @param saved_V Forward-pass value matrix
 * @param saved_attn_weights Forward-pass softmax attention weights
 * @param grad_Q Output gradient buffer with respect to Q
 * @param grad_K Output gradient buffer with respect to K
 * @param grad_V Output gradient buffer with respect to V
 * @param seq_q Query sequence length
 * @param seq_k Key/value sequence length
 * @param d_k Key/query feature dimension
 * @param d_v Value feature dimension
 * @param scale Scaling factor applied to Q@K^T in the forward pass
 */
void eshkol_backward_attention(
    const double* grad_out,
    const double* saved_Q, const double* saved_K, const double* saved_V,
    const double* saved_attn_weights,
    double* grad_Q, double* grad_K, double* grad_V,
    int64_t seq_q, int64_t seq_k, int64_t d_k, int64_t d_v,
    double scale);

/**
 * @brief Multi-head attention backward pass: computes gradients for Q, K, V and all projection weights.
 * @param grad_out Upstream gradient over the MHA output
 * @param saved_Q Forward-pass query input
 * @param saved_K Forward-pass key input
 * @param saved_V Forward-pass value input
 * @param saved_W_Q Forward-pass query projection weights
 * @param saved_W_K Forward-pass key projection weights
 * @param saved_W_V Forward-pass value projection weights
 * @param saved_W_O Forward-pass output projection weights
 * @param saved_attn_per_head Forward-pass per-head attention weights
 * @param grad_Q Output gradient buffer with respect to Q
 * @param grad_K Output gradient buffer with respect to K
 * @param grad_V Output gradient buffer with respect to V
 * @param grad_WQ Output gradient buffer with respect to W_Q
 * @param grad_WK Output gradient buffer with respect to W_K
 * @param grad_WV Output gradient buffer with respect to W_V
 * @param grad_WO Output gradient buffer with respect to W_O
 * @param seq_q Query sequence length
 * @param seq_k Key/value sequence length
 * @param d_model Model (embedding) dimension
 * @param num_heads Number of attention heads
 */
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

/**
 * @brief Embedding backward pass: scatter-adds gradients into the weight matrix rows selected by saved_indices.
 * @param grad_out Upstream gradient over the embedded output
 * @param saved_indices Forward-pass token indices used to look up rows
 * @param grad_weights Output gradient buffer with respect to the embedding weight matrix
 * @param num_indices Number of indices looked up
 * @param d_model Embedding dimension
 * @param vocab_size Vocabulary size (number of rows in the weight matrix)
 */
void eshkol_backward_embedding(
    const double* grad_out,
    const int64_t* saved_indices,
    double* grad_weights,
    int64_t num_indices, int64_t d_model, int64_t vocab_size);

/* ===== Matmul ===== */

/**
 * @brief Matmul backward pass: dA = grad_C @ B^T, dB = A^T @ grad_C.
 * @param grad_out Upstream gradient over C (M x N)
 * @param saved_A Forward-pass A matrix (M x K)
 * @param saved_B Forward-pass B matrix (K x N)
 * @param grad_A Output gradient buffer with respect to A
 * @param grad_B Output gradient buffer with respect to B
 * @param M Rows of A and C
 * @param K Columns of A / rows of B
 * @param N Columns of B and C
 */
void eshkol_backward_matmul(
    const double* grad_out,
    const double* saved_A, const double* saved_B,
    double* grad_A, double* grad_B,
    int64_t M, int64_t K, int64_t N);

/* ===== Tensor Gradient Accumulation ===== */

/**
 * @brief Accumulate a gradient into an AD node's tensor_gradient field.
 *
 * If tensor_gradient is NULL, allocates and zero-fills it first.
 * Thread-safe for single-threaded backward pass.
 *
 * @param ad_node_ptr Target AD node (as eshkol_ad_node_t*, opaque here)
 * @param grad Gradient values to add
 * @param num_elements Number of elements in grad
 */
void eshkol_accumulate_tensor_grad(
    void* ad_node_ptr, const double* grad, int64_t num_elements);

/* ===== Tensor Gradient Seeding ===== */

/**
 * @brief Seed the output node's tensor gradient for backpropagation.
 *
 * If the node has tensor_value (non-null), allocates tensor_gradient
 * and fills it with all ones (dL/dL = 1 for every element).
 * Called at the start of backpropagate() on the output node.
 *
 * @param ad_node_ptr Output AD node (as eshkol_ad_node_t*, opaque here)
 */
void eshkol_seed_tensor_gradient(void* ad_node_ptr);

/* ===== Tensor Backward Dispatcher ===== */

/**
 * @brief Main entry point called from LLVM-generated backward pass.
 *
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
 *
 * @param ad_node_ptr AD node whose backward pass should be dispatched (as eshkol_ad_node_t*, opaque here)
 */
void eshkol_tensor_backward_dispatch(void* ad_node_ptr);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_BACKEND_TENSOR_BACKWARD_H */
