/**
 * @file qllm_bridge.h
 * @brief Eshkol ↔ qLLM bridge for tensor operations with automatic differentiation.
 *
 * Provides type conversion between Eshkol tagged values and qLLM tensors,
 * plus AD-aware tensor operations that record onto Eshkol's gradient tape.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef ESHKOL_QLLM_BRIDGE_H
#define ESHKOL_QLLM_BRIDGE_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations — avoid including full headers */
typedef struct qllm_tensor qllm_tensor_t;
typedef struct ad_tape ad_tape_t;
typedef struct ad_node ad_node_t;

/*******************************************************************************
 * Type Conversion: Eshkol ↔ qLLM
 ******************************************************************************/

/**
 * @brief Convert an Eshkol tensor (double) to a qLLM tensor (float32).
 *
 * Eshkol uses double precision internally; qLLM uses float32 for inference.
 * The returned tensor is newly allocated and must be freed by the caller.
 *
 * @param eshkol_data  Pointer to double array
 * @param shape        Tensor shape
 * @param ndim         Number of dimensions
 * @return qLLM tensor (float32), or NULL on failure
 */
qllm_tensor_t* eshkol_to_qllm_tensor(
    const double* eshkol_data,
    const size_t* shape,
    size_t ndim
);

/**
 * @brief Convert a qLLM tensor (float32) to Eshkol doubles.
 *
 * @param tensor       qLLM tensor
 * @param out_data     Output: double array (must be pre-allocated)
 * @param out_size     Output: number of elements
 * @return true on success
 */
bool qllm_to_eshkol_tensor(
    const qllm_tensor_t* tensor,
    double* out_data,
    size_t* out_size
);

/*******************************************************************************
 * AD-Aware Tensor Operations
 *
 * Each function performs the forward pass and records onto the AD tape
 * when tape is non-NULL. When tape is NULL, just the forward pass runs.
 ******************************************************************************/

/**
 * @brief Matrix multiply with AD recording.
 *
 * Forward: C = A @ B
 * Backward: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
 */
ad_node_t* ad_tensor_matmul(
    ad_tape_t* tape,
    ad_node_t* a,
    ad_node_t* b
);

/**
 * @brief Softmax with AD recording.
 *
 * Forward: y = softmax(x, dim)
 * Backward: dL/dx[i] = y[i] * (dL/dy[i] - sum_j(dL/dy[j] * y[j]))
 */
ad_node_t* ad_tensor_softmax(
    ad_tape_t* tape,
    ad_node_t* x,
    int dim
);

/**
 * @brief Layer normalization with AD recording.
 *
 * Forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
 * Backward: chain rule through mean/variance computation
 */
ad_node_t* ad_tensor_layernorm(
    ad_tape_t* tape,
    ad_node_t* x,
    ad_node_t* gamma,
    ad_node_t* beta,
    double eps
);

/**
 * @brief RMS normalization with AD recording.
 *
 * Forward: y = x * rsqrt(mean(x^2) + eps) * gamma
 * Backward: chain rule through RMS computation
 */
ad_node_t* ad_tensor_rmsnorm(
    ad_tape_t* tape,
    ad_node_t* x,
    ad_node_t* gamma,
    double eps
);

/**
 * @brief Multi-head scaled dot-product attention with AD recording.
 *
 * Forward: attn = softmax(Q @ K^T / sqrt(d_k)) @ V
 * Backward: 5-step chain rule through output proj → attn@V → softmax → scale → Q@K^T
 *
 * @param tape     AD tape (NULL for forward-only)
 * @param q        Query tensor [batch, seq, dim]
 * @param k        Key tensor [batch, seq, dim]
 * @param v        Value tensor [batch, seq, dim]
 * @param num_heads Number of attention heads
 * @param causal   Whether to apply causal masking
 * @return Output tensor node
 */
ad_node_t* ad_tensor_attention(
    ad_tape_t* tape,
    ad_node_t* q,
    ad_node_t* k,
    ad_node_t* v,
    int num_heads,
    bool causal
);

/**
 * @brief GELU activation with AD recording.
 *
 * Forward: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Backward: chain rule through tanh approximation
 */
ad_node_t* ad_tensor_gelu(
    ad_tape_t* tape,
    ad_node_t* x
);

/**
 * @brief SiLU/Swish activation with AD recording.
 *
 * Forward: silu(x) = x * sigmoid(x)
 * Backward: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
 */
ad_node_t* ad_tensor_silu(
    ad_tape_t* tape,
    ad_node_t* x
);

/**
 * @brief Cross-entropy loss with AD recording.
 *
 * Forward: loss = -sum(target * log(softmax(logits)))
 * Backward: dL/dlogits = softmax(logits) - target (numerically stable)
 */
ad_node_t* ad_tensor_cross_entropy(
    ad_tape_t* tape,
    ad_node_t* logits,
    ad_node_t* targets
);

/*******************************************************************************
 * Geometric AD Operations (Riemannian manifold)
 ******************************************************************************/

/**
 * @brief Hyperbolic distance in the Poincare ball model.
 *
 * d(x, y) = acosh(1 + 2 * ||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
 */
ad_node_t* ad_hyperbolic_distance(
    ad_tape_t* tape,
    ad_node_t* x,
    ad_node_t* y,
    double curvature
);

/**
 * @brief Poincare exponential map.
 *
 * Maps a tangent vector at x to a point on the manifold.
 */
ad_node_t* ad_poincare_exp_map(
    ad_tape_t* tape,
    ad_node_t* x,
    ad_node_t* v,
    double curvature
);

/**
 * @brief Poincare logarithmic map.
 *
 * Maps a point y back to the tangent space at x.
 */
ad_node_t* ad_poincare_log_map(
    ad_tape_t* tape,
    ad_node_t* x,
    ad_node_t* y,
    double curvature
);

/**
 * @brief Geodesic attention with curvature-adaptive scaling.
 *
 * Replaces dot-product with geodesic distance in attention scores.
 */
ad_node_t* ad_geodesic_attention(
    ad_tape_t* tape,
    ad_node_t* q,
    ad_node_t* k,
    ad_node_t* v,
    int num_heads,
    double curvature,
    bool causal
);

/*******************************************************************************
 * Bridge Lifecycle
 ******************************************************************************/

/**
 * @brief Initialize the qLLM bridge.
 *
 * Loads libsemiclassical_qllm and initializes the tensor runtime.
 *
 * @param library_path  Path to libsemiclassical_qllm.dylib (NULL for default)
 * @return true on success
 */
bool eshkol_qllm_bridge_init(const char* library_path);

/**
 * @brief Shutdown the qLLM bridge.
 */
void eshkol_qllm_bridge_shutdown(void);

/**
 * @brief Check if the qLLM bridge is initialized.
 */
bool eshkol_qllm_bridge_ready(void);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_QLLM_BRIDGE_H */
