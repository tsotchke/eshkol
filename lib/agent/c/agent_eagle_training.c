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

static int qllm_ffi_linear_valid_dims(int64_t in_dim, int64_t out_dim) {
    const int64_t max_dim = 1 << 20;
    if (in_dim <= 0 || out_dim <= 0) return 0;
    if (in_dim > max_dim || out_dim > max_dim) return 0;
    if (out_dim > INT64_MAX / in_dim) return 0;
    return 1;
}

static int qllm_ffi_linear_bounds(qllm_ffi_linear_t* layer, int64_t out, int64_t in) {
    if (!layer) return 0;
    if (out < 0 || out >= layer->out_dim) return 0;
    if (in < 0 || in >= layer->in_dim) return 0;
    return 1;
}

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

int32_t qllm_ffi_linear_set_weight(void* handle, int64_t out, int64_t in, double value) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!qllm_ffi_linear_bounds(layer, out, in)) return -1;
    layer->weights[(size_t)(out * layer->in_dim + in)] = value;
    return 0;
}

double qllm_ffi_linear_get_weight(void* handle, int64_t out, int64_t in) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!qllm_ffi_linear_bounds(layer, out, in)) return 0.0;
    return layer->weights[(size_t)(out * layer->in_dim + in)];
}

int32_t qllm_ffi_linear_set_input(void* handle, int64_t in, double value) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!layer || in < 0 || in >= layer->in_dim) return -1;
    layer->input[in] = value;
    return 0;
}

int32_t qllm_ffi_linear_set_target(void* handle, int64_t out, double value) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!layer || out < 0 || out >= layer->out_dim) return -1;
    layer->target[out] = value;
    return 0;
}

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

double qllm_ffi_linear_pred(void* handle, int64_t out) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!layer || out < 0 || out >= layer->out_dim) return 0.0;
    return layer->pred[out];
}

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

double qllm_ffi_linear_grad(void* handle, int64_t out, int64_t in) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!qllm_ffi_linear_bounds(layer, out, in)) return 0.0;
    return layer->grads[(size_t)(out * layer->in_dim + in)];
}

int32_t qllm_ffi_linear_sgd_step(void* handle, double lr) {
    qllm_ffi_linear_t* layer = (qllm_ffi_linear_t*)handle;
    if (!layer) return -1;

    const size_t n_weights = (size_t)(layer->in_dim * layer->out_dim);
    for (size_t i = 0; i < n_weights; i++) {
        layer->weights[i] -= lr * layer->grads[i];
    }
    return 0;
}

int32_t qllm_ffi_linear_train_step(void* handle, double lr) {
    if (qllm_ffi_linear_forward(handle) != 0) return -1;
    if (qllm_ffi_linear_backward(handle) != 0) return -1;
    return qllm_ffi_linear_sgd_step(handle, lr);
}
