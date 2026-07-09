/*
 * Native EAGLE linear-head training helpers.
 *
 * This is intentionally a small C ABI around the core operation EAGLE needs:
 * y = W.x, dW = dL/dy outer x, SGD update.  Eshkol can drive it through FFI
 * without relying on the currently-buggy vector-valued gradient path.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t in_dim;
    int64_t out_dim;
    double* weights; /* out_dim x in_dim, row-major */
    double* grads;   /* out_dim x in_dim, row-major */
    double* input;   /* in_dim */
    double* target;  /* out_dim */
    double* pred;    /* out_dim */
} qllm_ffi_linear_t;

void qllm_ffi_linear_destroy(void* handle);

/**
 * @brief Validates that a linear layer's input/output dimensions are positive,
 * bounded, and won't overflow when multiplied.
 *
 * @param in_dim Proposed input dimension.
 * @param out_dim Proposed output dimension.
 * @return 1 if the dimensions are usable, 0 otherwise.
 */
static int qllm_ffi_linear_valid_dims(int64_t in_dim, int64_t out_dim) {
    const int64_t max_dim = 1 << 20;
    if (in_dim <= 0 || out_dim <= 0) return 0;
    if (in_dim > max_dim || out_dim > max_dim) return 0;
    if (out_dim > INT64_MAX / in_dim) return 0;
    return 1;
}

/**
 * @brief Checks that @p layer is non-NULL and (@p out, @p in) is a valid
 * weight-matrix coordinate for it.
 *
 * @param layer Layer to validate against (may be NULL).
 * @param out Output-dimension index to check.
 * @param in Input-dimension index to check.
 * @return 1 if @p layer is non-NULL and both indices are in range, 0 otherwise.
 */
static int qllm_ffi_linear_bounds(qllm_ffi_linear_t* layer, int64_t out, int64_t in) {
    if (!layer) return 0;
    if (out < 0 || out >= layer->out_dim) return 0;
    if (in < 0 || in >= layer->in_dim) return 0;
    return 1;
}

/**
 * @brief Allocates a linear layer with zero-initialized weights, gradients,
 * input, target, and prediction buffers.
 *
 * On any allocation failure, all partially-allocated buffers are released
 * via qllm_ffi_linear_destroy() before returning NULL.
 *
 * @param in_dim Input dimension (must be positive and bounded; see
 *        qllm_ffi_linear_valid_dims()).
 * @param out_dim Output dimension (same constraints as @p in_dim).
 * @return Opaque layer handle on success, or NULL on invalid dimensions or OOM.
 */
void* qllm_ffi_linear_create(int64_t in_dim, int64_t out_dim) {
    if (!qllm_ffi_linear_valid_dims(in_dim, out_dim)) return NULL;

    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)calloc(1, sizeof(qllm_ffi_linear_t));
    if (!layer) return NULL;

    const size_t n_weights = (size_t)(in_dim * out_dim);
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->weights = (double*)calloc(n_weights, sizeof(double));
    layer->grads = (double*)calloc(n_weights, sizeof(double));
    layer->input = (double*)calloc((size_t)in_dim, sizeof(double));
    layer->target = (double*)calloc((size_t)out_dim, sizeof(double));
    layer->pred = (double*)calloc((size_t)out_dim, sizeof(double));

    if (!layer->weights || !layer->grads || !layer->input ||
        !layer->target || !layer->pred) {
        qllm_ffi_linear_destroy(layer);
        return NULL;
    }

    return layer;
}

/**
 * @brief Frees a linear layer and all of its internal buffers.
 *
 * @param handle Layer handle from qllm_ffi_linear_create(). No-op if NULL.
 */
void qllm_ffi_linear_destroy(void* handle) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!layer) return;
    free(layer->weights);
    free(layer->grads);
    free(layer->input);
    free(layer->target);
    free(layer->pred);
    free(layer);
}

/**
 * @brief Sets a single entry of the layer's row-major weight matrix.
 *
 * @param handle Layer handle from qllm_ffi_linear_create().
 * @param out Output-dimension (row) index.
 * @param in Input-dimension (column) index.
 * @param value New weight value.
 * @return 0 on success, -1 if @p handle is invalid or the indices are out of range.
 */
int32_t qllm_ffi_linear_set_weight(void* handle, int64_t out, int64_t in, double value) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!qllm_ffi_linear_bounds(layer, out, in)) return -1;
    layer->weights[(size_t)(out * layer->in_dim + in)] = value;
    return 0;
}

/**
 * @brief Reads a single entry of the layer's row-major weight matrix.
 *
 * @param handle Layer handle from qllm_ffi_linear_create().
 * @param out Output-dimension (row) index.
 * @param in Input-dimension (column) index.
 * @return The weight value, or 0.0 if @p handle is invalid or the indices
 *         are out of range.
 */
double qllm_ffi_linear_get_weight(void* handle, int64_t out, int64_t in) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!qllm_ffi_linear_bounds(layer, out, in)) return 0.0;
    return layer->weights[(size_t)(out * layer->in_dim + in)];
}

/**
 * @brief Sets one component of the layer's input vector x.
 *
 * @param handle Layer handle from qllm_ffi_linear_create().
 * @param in Input-dimension index.
 * @param value New input value.
 * @return 0 on success, -1 if @p handle is invalid or @p in is out of range.
 */
int32_t qllm_ffi_linear_set_input(void* handle, int64_t in, double value) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!layer || in < 0 || in >= layer->in_dim) return -1;
    layer->input[in] = value;
    return 0;
}

/**
 * @brief Sets one component of the layer's training target vector.
 *
 * @param handle Layer handle from qllm_ffi_linear_create().
 * @param out Output-dimension index.
 * @param value New target value.
 * @return 0 on success, -1 if @p handle is invalid or @p out is out of range.
 */
int32_t qllm_ffi_linear_set_target(void* handle, int64_t out, double value) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!layer || out < 0 || out >= layer->out_dim) return -1;
    layer->target[out] = value;
    return 0;
}

/**
 * @brief Computes the forward pass y = W.x, storing the result in the layer's
 * prediction buffer.
 *
 * @param handle Layer handle from qllm_ffi_linear_create().
 * @return 0 on success, -1 if @p handle is NULL.
 */
int32_t qllm_ffi_linear_forward(void* handle) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!layer) return -1;

    for (int64_t out = 0; out < layer->out_dim; out++) {
        double sum = 0.0;
        const double* wrow = layer->weights + (size_t)(out * layer->in_dim);
        for (int64_t in = 0; in < layer->in_dim; in++) {
            sum += wrow[in] * layer->input[in];
        }
        layer->pred[out] = sum;
    }
    return 0;
}

/**
 * @brief Reads one component of the layer's most recent prediction (from
 * qllm_ffi_linear_forward()).
 *
 * @param handle Layer handle from qllm_ffi_linear_create().
 * @param out Output-dimension index.
 * @return The predicted value, or 0.0 if @p handle is invalid or @p out is
 *         out of range.
 */
double qllm_ffi_linear_pred(void* handle, int64_t out) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!layer || out < 0 || out >= layer->out_dim) return 0.0;
    return layer->pred[out];
}

/**
 * @brief Computes the mean squared error between the layer's prediction and
 * target vectors.
 *
 * @param handle Layer handle from qllm_ffi_linear_create().
 * @return Mean squared error over all output dimensions, or 0.0 if @p handle
 *         is NULL.
 */
double qllm_ffi_linear_loss(void* handle) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!layer) return 0.0;
    double loss = 0.0;
    for (int64_t out = 0; out < layer->out_dim; out++) {
        const double d = layer->pred[out] - layer->target[out];
        loss += d * d;
    }
    return loss / (double)layer->out_dim;
}

/**
 * @brief Computes the MSE-loss gradient dW = dL/dy outer x and stores it in
 * the layer's gradient buffer.
 *
 * Requires qllm_ffi_linear_forward() to have already populated the
 * prediction buffer for the current input/target.
 *
 * @param handle Layer handle from qllm_ffi_linear_create().
 * @return 0 on success, -1 if @p handle is NULL.
 */
int32_t qllm_ffi_linear_backward(void* handle) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!layer) return -1;

    const size_t n_weights = (size_t)(layer->in_dim * layer->out_dim);
    memset(layer->grads, 0, n_weights * sizeof(double));

    for (int64_t out = 0; out < layer->out_dim; out++) {
        const double dloss_dpred =
            2.0 * (layer->pred[out] - layer->target[out]) / (double)layer->out_dim;
        double* grow = layer->grads + (size_t)(out * layer->in_dim);
        for (int64_t in = 0; in < layer->in_dim; in++) {
            grow[in] = dloss_dpred * layer->input[in];
        }
    }

    return 0;
}

/**
 * @brief Reads a single entry of the layer's gradient matrix (from
 * qllm_ffi_linear_backward()).
 *
 * @param handle Layer handle from qllm_ffi_linear_create().
 * @param out Output-dimension (row) index.
 * @param in Input-dimension (column) index.
 * @return The gradient value, or 0.0 if @p handle is invalid or the indices
 *         are out of range.
 */
double qllm_ffi_linear_grad(void* handle, int64_t out, int64_t in) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!qllm_ffi_linear_bounds(layer, out, in)) return 0.0;
    return layer->grads[(size_t)(out * layer->in_dim + in)];
}

/**
 * @brief Applies one vanilla SGD update: weights -= lr * grads, elementwise.
 *
 * @param handle Layer handle from qllm_ffi_linear_create().
 * @param lr Learning rate.
 * @return 0 on success, -1 if @p handle is NULL.
 */
int32_t qllm_ffi_linear_sgd_step(void* handle, double lr) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!layer) return -1;

    const size_t n_weights = (size_t)(layer->in_dim * layer->out_dim);
    for (size_t i = 0; i < n_weights; i++) {
        layer->weights[i] -= lr * layer->grads[i];
    }
    return 0;
}

/**
 * @brief Runs one full training step: forward pass, backward pass, then an
 * SGD weight update.
 *
 * @param handle Layer handle from qllm_ffi_linear_create().
 * @param lr Learning rate passed through to qllm_ffi_linear_sgd_step().
 * @return 0 on success, -1 if any sub-step fails (invalid @p handle).
 */
int32_t qllm_ffi_linear_train_step(void* handle, double lr) {
    if (qllm_ffi_linear_forward(handle) != 0) return -1;
    if (qllm_ffi_linear_backward(handle) != 0) return -1;
    return qllm_ffi_linear_sgd_step(handle, lr);
}
