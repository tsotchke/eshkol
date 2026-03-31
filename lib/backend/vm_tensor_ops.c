/**
 * @file vm_tensor_ops.c
 * @brief Tensor operations for the Eshkol bytecode VM.
 *
 * Implements broadcasting, element-wise ops, matmul, reductions,
 * activation functions, loss functions, and conv2d.
 * All data is arena-allocated via vm_arena.h. No GC.
 *
 * Native call IDs: 440-489
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_tensor.c"
#include <float.h>

/* ══════════════════════════════════════════════════════════════════════════════
 *  Broadcasting
 * ══════════════════════════════════════════════════════════════════════════════*/

/* Compute broadcast-compatible output shape from two input shapes.
 * Returns 0 on success, -1 if shapes are incompatible. */
static int vm_broadcast_shapes(const int64_t* a_shape, int a_dims,
                               const int64_t* b_shape, int b_dims,
                               int64_t* out_shape, int* out_dims) {
    *out_dims = a_dims > b_dims ? a_dims : b_dims;
    if (*out_dims > VM_TENSOR_MAX_DIMS) return -1;

    for (int i = 0; i < *out_dims; i++) {
        int64_t a = (i < a_dims) ? a_shape[a_dims - 1 - i] : 1;
        int64_t b = (i < b_dims) ? b_shape[b_dims - 1 - i] : 1;
        if (a != b && a != 1 && b != 1) return -1;
        out_shape[*out_dims - 1 - i] = (a > b) ? a : b;
    }
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Element-wise binary operations with broadcasting
 * ══════════════════════════════════════════════════════════════════════════════*/

typedef double (*VmBinaryFn)(double, double);

/* Convert a flat output index to source flat index, clamping broadcast dims to 0. */
static int64_t vm_broadcast_src_index(int64_t flat, const int64_t* out_shape, int out_dims,
                                      const int64_t* src_shape, int src_dims,
                                      const int64_t* src_strides) {
    int64_t indices[VM_TENSOR_MAX_DIMS];
    vm_tensor_unravel(flat, out_shape, out_dims, indices);

    int64_t off = 0;
    for (int i = 0; i < src_dims; i++) {
        int out_i = out_dims - src_dims + i;
        int64_t idx = indices[out_i];
        if (src_shape[i] == 1) idx = 0;  /* broadcast: clamp to 0 */
        off += idx * src_strides[i];
    }
    return off;
}

/* 440: Generic element-wise binary operation with broadcasting. */
static VmTensor* vm_tensor_binary_op(VmRegionStack* rs, const VmTensor* a,
                                     const VmTensor* b, VmBinaryFn op) {
    if (!a || !b || !op) return NULL;

    int64_t out_shape[VM_TENSOR_MAX_DIMS];
    int out_dims;
    if (vm_broadcast_shapes(a->shape, a->n_dims, b->shape, b->n_dims,
                            out_shape, &out_dims) != 0) {
        return NULL;
    }

    VmTensor* out = vm_tensor_new(rs, out_shape, out_dims);
    if (!out) return NULL;

    for (int64_t i = 0; i < out->total; i++) {
        int64_t ai = vm_broadcast_src_index(i, out_shape, out_dims,
                                            a->shape, a->n_dims, a->strides);
        int64_t bi = vm_broadcast_src_index(i, out_shape, out_dims,
                                            b->shape, b->n_dims, b->strides);
        out->data[i] = op(a->data[ai], b->data[bi]);
    }
    return out;
}

/* Scalar binary op helpers */
static double vm_op_add(double a, double b) { return a + b; }
static double vm_op_sub(double a, double b) { return a - b; }
static double vm_op_mul(double a, double b) { return a * b; }
static double vm_op_div(double a, double b) { return b != 0.0 ? a / b : 0.0; }
static double vm_op_pow(double a, double b) { return pow(a, b); }
static double vm_op_max(double a, double b) { return a > b ? a : b; }
static double vm_op_min(double a, double b) { return a < b ? a : b; }

/* 441-447: Convenience wrappers */
static VmTensor* vm_tensor_add(VmRegionStack* rs, const VmTensor* a, const VmTensor* b) {
    return vm_tensor_binary_op(rs, a, b, vm_op_add);
}
static VmTensor* vm_tensor_sub(VmRegionStack* rs, const VmTensor* a, const VmTensor* b) {
    return vm_tensor_binary_op(rs, a, b, vm_op_sub);
}
static VmTensor* vm_tensor_mul(VmRegionStack* rs, const VmTensor* a, const VmTensor* b) {
    return vm_tensor_binary_op(rs, a, b, vm_op_mul);
}
static VmTensor* vm_tensor_div(VmRegionStack* rs, const VmTensor* a, const VmTensor* b) {
    return vm_tensor_binary_op(rs, a, b, vm_op_div);
}
static VmTensor* vm_tensor_pow(VmRegionStack* rs, const VmTensor* a, const VmTensor* b) {
    return vm_tensor_binary_op(rs, a, b, vm_op_pow);
}
static VmTensor* vm_tensor_maximum(VmRegionStack* rs, const VmTensor* a, const VmTensor* b) {
    return vm_tensor_binary_op(rs, a, b, vm_op_max);
}
static VmTensor* vm_tensor_minimum(VmRegionStack* rs, const VmTensor* a, const VmTensor* b) {
    return vm_tensor_binary_op(rs, a, b, vm_op_min);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Element-wise unary operations
 * ══════════════════════════════════════════════════════════════════════════════*/

typedef double (*VmUnaryFn)(double);

/* 448: Generic element-wise unary operation. */
static VmTensor* vm_tensor_unary_op(VmRegionStack* rs, const VmTensor* t, VmUnaryFn op) {
    if (!t || !op) return NULL;

    VmTensor* out = vm_tensor_new(rs, t->shape, t->n_dims);
    if (!out) return NULL;

    /* Handle non-contiguous source */
    int64_t expected_strides[VM_TENSOR_MAX_DIMS];
    vm_tensor_compute_strides(t->shape, t->n_dims, expected_strides);
    int is_contiguous = 1;
    for (int i = 0; i < t->n_dims; i++) {
        if (t->strides[i] != expected_strides[i]) { is_contiguous = 0; break; }
    }

    if (is_contiguous) {
        for (int64_t i = 0; i < out->total; i++) {
            out->data[i] = op(t->data[i]);
        }
    } else {
        int64_t indices[VM_TENSOR_MAX_DIMS];
        for (int64_t i = 0; i < out->total; i++) {
            vm_tensor_unravel(i, t->shape, t->n_dims, indices);
            int64_t src_off = vm_tensor_flat_offset(t, indices, t->n_dims);
            out->data[i] = op(t->data[src_off]);
        }
    }
    return out;
}

static double vm_op_neg(double x) { return -x; }
static double vm_op_abs(double x) { return fabs(x); }
static double vm_op_sqrt(double x) { return sqrt(x); }
static double vm_op_exp(double x) { return exp(x); }
static double vm_op_log(double x) { return x > 0.0 ? log(x) : -INFINITY; }
static double vm_op_sin(double x) { return sin(x); }
static double vm_op_cos(double x) { return cos(x); }

/* 449-455: Convenience wrappers */
static VmTensor* vm_tensor_neg(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_neg);
}
static VmTensor* vm_tensor_abs(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_abs);
}
static VmTensor* vm_tensor_sqrt_op(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_sqrt);
}
static VmTensor* vm_tensor_exp_op(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_exp);
}
static VmTensor* vm_tensor_log_op(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_log);
}
static VmTensor* vm_tensor_sin_op(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_sin);
}
static VmTensor* vm_tensor_cos_op(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_cos);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Scale (scalar * tensor)
 * ══════════════════════════════════════════════════════════════════════════════*/

/* 456: Multiply every element by a scalar. */
static VmTensor* vm_tensor_scale(VmRegionStack* rs, const VmTensor* t, double s) {
    if (!t) return NULL;
    VmTensor* out = vm_tensor_new(rs, t->shape, t->n_dims);
    if (!out) return NULL;
    for (int64_t i = 0; i < out->total; i++) {
        out->data[i] = t->data[i] * s;
    }
    return out;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Matrix Multiplication
 * ══════════════════════════════════════════════════════════════════════════════*/

/* 457: matmul — 2D matrix multiplication (MxK) @ (KxN) → (MxN).
 * Uses scalar triple loop. Optional BLAS dispatch point (not linked here). */
static VmTensor* vm_tensor_matmul(VmRegionStack* rs, const VmTensor* a, const VmTensor* b) {
    if (!a || !b) return NULL;
    if (a->n_dims != 2 || b->n_dims != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;

    int64_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
    int64_t out_shape[2] = { M, N };
    VmTensor* out = vm_tensor_zeros(rs, out_shape, 2);
    if (!out) return NULL;

    for (int64_t i = 0; i < M; i++) {
        for (int64_t k = 0; k < K; k++) {
            double a_ik = a->data[i * K + k];
            for (int64_t j = 0; j < N; j++) {
                out->data[i * N + j] += a_ik * b->data[k * N + j];
            }
        }
    }
    return out;
}

/* 458: Batched matmul — (..., M, K) @ (..., K, N) → (..., M, N).
 * Handles 3D tensors as batches of 2D matrices. */
static VmTensor* vm_tensor_batch_matmul(VmRegionStack* rs, const VmTensor* a, const VmTensor* b) {
    if (!a || !b) return NULL;
    if (a->n_dims != 3 || b->n_dims != 3) return NULL;
    if (a->shape[0] != b->shape[0]) return NULL;  /* batch size must match */
    if (a->shape[2] != b->shape[1]) return NULL;  /* K must match */

    int64_t B = a->shape[0], M = a->shape[1], K = a->shape[2], N = b->shape[2];
    int64_t out_shape[3] = { B, M, N };
    VmTensor* out = vm_tensor_zeros(rs, out_shape, 3);
    if (!out) return NULL;

    for (int64_t batch = 0; batch < B; batch++) {
        const double* a_ptr = a->data + batch * M * K;
        const double* b_ptr = b->data + batch * K * N;
        double* o_ptr = out->data + batch * M * N;
        for (int64_t i = 0; i < M; i++) {
            for (int64_t k = 0; k < K; k++) {
                double a_ik = a_ptr[i * K + k];
                for (int64_t j = 0; j < N; j++) {
                    o_ptr[i * N + j] += a_ik * b_ptr[k * N + j];
                }
            }
        }
    }
    return out;
}

/* 459: Dot product — 1D inner product. */
static double vm_tensor_dot(const VmTensor* a, const VmTensor* b) {
    if (!a || !b || a->n_dims != 1 || b->n_dims != 1) return 0.0;
    if (a->total != b->total) return 0.0;
    double sum = 0.0;
    for (int64_t i = 0; i < a->total; i++) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Reduction Operations
 * ══════════════════════════════════════════════════════════════════════════════*/

typedef enum {
    VM_REDUCE_SUM,
    VM_REDUCE_MEAN,
    VM_REDUCE_MAX,
    VM_REDUCE_MIN,
    VM_REDUCE_PROD
} VmReduceOp;

/* 460: Reduce along a single axis.
 * Output shape has the specified axis removed. */
static VmTensor* vm_tensor_reduce(VmRegionStack* rs, const VmTensor* t,
                                  int axis, VmReduceOp op) {
    if (!t) return NULL;
    if (axis < 0) axis += t->n_dims;
    if (axis < 0 || axis >= t->n_dims) return NULL;

    /* Build output shape (remove axis) */
    int64_t out_shape[VM_TENSOR_MAX_DIMS];
    int out_dims = 0;
    int scalar_reduce = 0;  /* flag: 1D input reduced to scalar */
    for (int i = 0; i < t->n_dims; i++) {
        if (i != axis) out_shape[out_dims++] = t->shape[i];
    }
    if (out_dims == 0) {
        /* Scalar result from 1D input: return 1-element tensor */
        out_shape[0] = 1;
        out_dims = 1;
        scalar_reduce = 1;
    }

    VmTensor* out = vm_tensor_new(rs, out_shape, out_dims);
    if (!out) return NULL;

    /* Initialize based on op */
    for (int64_t i = 0; i < out->total; i++) {
        switch (op) {
            case VM_REDUCE_SUM:  out->data[i] = 0.0; break;
            case VM_REDUCE_MEAN: out->data[i] = 0.0; break;
            case VM_REDUCE_MAX:  out->data[i] = -DBL_MAX; break;
            case VM_REDUCE_MIN:  out->data[i] = DBL_MAX; break;
            case VM_REDUCE_PROD: out->data[i] = 1.0; break;
        }
    }

    /* Iterate over all elements of t */
    int64_t t_indices[VM_TENSOR_MAX_DIMS];
    for (int64_t flat = 0; flat < t->total; flat++) {
        vm_tensor_unravel(flat, t->shape, t->n_dims, t_indices);
        int64_t src_off = vm_tensor_flat_offset(t, t_indices, t->n_dims);
        double val = t->data[src_off];

        /* Compute output flat index (skip axis dimension) */
        int64_t out_flat = 0;
        if (!scalar_reduce) {
            int64_t out_indices[VM_TENSOR_MAX_DIMS];
            int oi = 0;
            for (int i = 0; i < t->n_dims; i++) {
                if (i != axis) out_indices[oi++] = t_indices[i];
            }
            int64_t out_strides[VM_TENSOR_MAX_DIMS];
            vm_tensor_compute_strides(out_shape, out_dims, out_strides);
            for (int i = 0; i < out_dims; i++) {
                out_flat += out_indices[i] * out_strides[i];
            }
        }
        /* else: scalar_reduce → out_flat stays 0 (all elements accumulate into single slot) */

        switch (op) {
            case VM_REDUCE_SUM:
            case VM_REDUCE_MEAN:
                out->data[out_flat] += val;
                break;
            case VM_REDUCE_MAX:
                if (val > out->data[out_flat]) out->data[out_flat] = val;
                break;
            case VM_REDUCE_MIN:
                if (val < out->data[out_flat]) out->data[out_flat] = val;
                break;
            case VM_REDUCE_PROD:
                out->data[out_flat] *= val;
                break;
        }
    }

    /* For mean: divide by axis length */
    if (op == VM_REDUCE_MEAN) {
        double axis_len = (double)t->shape[axis];
        for (int64_t i = 0; i < out->total; i++) {
            out->data[i] /= axis_len;
        }
    }

    return out;
}

/* 461: Full reduction (all axes) to a single scalar value. */
static double vm_tensor_reduce_all(const VmTensor* t, VmReduceOp op) {
    if (!t || t->total == 0) return 0.0;

    double acc;
    switch (op) {
        case VM_REDUCE_SUM:  acc = 0.0; break;
        case VM_REDUCE_MEAN: acc = 0.0; break;
        case VM_REDUCE_MAX:  acc = -DBL_MAX; break;
        case VM_REDUCE_MIN:  acc = DBL_MAX; break;
        case VM_REDUCE_PROD: acc = 1.0; break;
        default:             acc = 0.0; break;
    }

    for (int64_t i = 0; i < t->total; i++) {
        double v = t->data[i];
        switch (op) {
            case VM_REDUCE_SUM:
            case VM_REDUCE_MEAN:
                acc += v; break;
            case VM_REDUCE_MAX:
                if (v > acc) acc = v; break;
            case VM_REDUCE_MIN:
                if (v < acc) acc = v; break;
            case VM_REDUCE_PROD:
                acc *= v; break;
        }
    }

    if (op == VM_REDUCE_MEAN) acc /= (double)t->total;
    return acc;
}

/* Convenience wrappers */
static VmTensor* vm_tensor_sum(VmRegionStack* rs, const VmTensor* t, int axis) {
    return vm_tensor_reduce(rs, t, axis, VM_REDUCE_SUM);
}
static VmTensor* vm_tensor_mean(VmRegionStack* rs, const VmTensor* t, int axis) {
    return vm_tensor_reduce(rs, t, axis, VM_REDUCE_MEAN);
}
static VmTensor* vm_tensor_max(VmRegionStack* rs, const VmTensor* t, int axis) {
    return vm_tensor_reduce(rs, t, axis, VM_REDUCE_MAX);
}
static VmTensor* vm_tensor_min(VmRegionStack* rs, const VmTensor* t, int axis) {
    return vm_tensor_reduce(rs, t, axis, VM_REDUCE_MIN);
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Activation Functions
 * ══════════════════════════════════════════════════════════════════════════════*/

static double vm_op_relu(double x) { return x > 0.0 ? x : 0.0; }
static double vm_op_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double vm_op_tanh_act(double x) {
    double ep = exp(x), en = exp(-x);
    return (ep - en) / (ep + en);
}
static double vm_op_leaky_relu(double x) { return x > 0.0 ? x : 0.01 * x; }
static double vm_op_elu(double x) { return x > 0.0 ? x : exp(x) - 1.0; }
static double vm_op_gelu(double x) {
    /* Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    double c = 0.7978845608028654; /* sqrt(2/pi) */
    double inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}
static double vm_op_swish(double x) { return x / (1.0 + exp(-x)); }

/* 462-468: Activation function wrappers */
static VmTensor* vm_tensor_relu(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_relu);
}
static VmTensor* vm_tensor_sigmoid(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_sigmoid);
}
static VmTensor* vm_tensor_tanh_act(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_tanh_act);
}
static VmTensor* vm_tensor_leaky_relu(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_leaky_relu);
}
static VmTensor* vm_tensor_elu(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_elu);
}
static VmTensor* vm_tensor_gelu(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_gelu);
}
static VmTensor* vm_tensor_swish(VmRegionStack* rs, const VmTensor* t) {
    return vm_tensor_unary_op(rs, t, vm_op_swish);
}

/* 469: Softmax along specified axis.
 * softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
 * Numerically stable: subtract max before exponentiating. */
static VmTensor* vm_tensor_softmax(VmRegionStack* rs, const VmTensor* t, int axis) {
    if (!t) return NULL;
    if (axis < 0) axis += t->n_dims;
    if (axis < 0 || axis >= t->n_dims) return NULL;

    VmTensor* out = vm_tensor_new(rs, t->shape, t->n_dims);
    if (!out) return NULL;

    /* Copy source data, respecting non-contiguous strides */
    {
        int64_t indices[VM_TENSOR_MAX_DIMS];
        for (int64_t i = 0; i < out->total; i++) {
            vm_tensor_unravel(i, t->shape, t->n_dims, indices);
            int64_t src_off = vm_tensor_flat_offset(t, indices, t->n_dims);
            out->data[i] = t->data[src_off];
        }
    }

    /* Now out is contiguous. Iterate over all positions except the softmax axis. */
    /* For each "slice" along the axis, compute max, subtract, exp, normalize. */

    /* Number of independent softmax operations = total / shape[axis] */
    int64_t axis_len = t->shape[axis];
    int64_t n_slices = out->total / axis_len;

    /* Compute the stride for the axis dimension in the output (contiguous) */
    int64_t out_strides[VM_TENSOR_MAX_DIMS];
    vm_tensor_compute_strides(out->shape, out->n_dims, out_strides);
    int64_t axis_stride = out_strides[axis];

    /* For each slice perpendicular to axis */
    for (int64_t s = 0; s < n_slices; s++) {
        /* Compute the base flat index for this slice.
         * We decompose s into indices for all non-axis dimensions. */
        int64_t base = 0;
        {
            int64_t remaining = s;
            for (int d = out->n_dims - 1; d >= 0; d--) {
                if (d == axis) continue;
                int64_t dim_size = out->shape[d];
                int64_t idx = remaining % dim_size;
                remaining /= dim_size;
                base += idx * out_strides[d];
            }
        }

        /* Pass 1: find max */
        double max_val = -DBL_MAX;
        for (int64_t k = 0; k < axis_len; k++) {
            double v = out->data[base + k * axis_stride];
            if (v > max_val) max_val = v;
        }

        /* Pass 2: exp(x - max) */
        double sum_exp = 0.0;
        for (int64_t k = 0; k < axis_len; k++) {
            double e = exp(out->data[base + k * axis_stride] - max_val);
            out->data[base + k * axis_stride] = e;
            sum_exp += e;
        }

        /* Pass 3: normalize */
        if (sum_exp > 0.0) {
            double inv = 1.0 / sum_exp;
            for (int64_t k = 0; k < axis_len; k++) {
                out->data[base + k * axis_stride] *= inv;
            }
        }
    }

    return out;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Loss Functions
 * ══════════════════════════════════════════════════════════════════════════════*/

/* 470: Mean Squared Error — mean((pred - target)^2) */
static double vm_tensor_mse_loss(const VmTensor* pred, const VmTensor* target) {
    if (!pred || !target || pred->total != target->total || pred->total == 0) return 0.0;
    double sum = 0.0;
    for (int64_t i = 0; i < pred->total; i++) {
        double d = pred->data[i] - target->data[i];
        sum += d * d;
    }
    return sum / (double)pred->total;
}

/* 471: Cross-entropy — -sum(target * log(pred + eps)) */
static double vm_tensor_cross_entropy_loss(const VmTensor* pred, const VmTensor* target) {
    if (!pred || !target || pred->total != target->total || pred->total == 0) return 0.0;
    double eps = 1e-12;
    double sum = 0.0;
    for (int64_t i = 0; i < pred->total; i++) {
        double p = pred->data[i];
        if (p < eps) p = eps;
        sum -= target->data[i] * log(p);
    }
    return sum;
}

/* 472: Binary Cross-Entropy — -mean(t*log(p) + (1-t)*log(1-p)) */
static double vm_tensor_bce_loss(const VmTensor* pred, const VmTensor* target) {
    if (!pred || !target || pred->total != target->total || pred->total == 0) return 0.0;
    double eps = 1e-12;
    double sum = 0.0;
    for (int64_t i = 0; i < pred->total; i++) {
        double p = pred->data[i];
        double t_val = target->data[i];
        /* Clamp to avoid log(0) */
        if (p < eps) p = eps;
        if (p > 1.0 - eps) p = 1.0 - eps;
        sum -= t_val * log(p) + (1.0 - t_val) * log(1.0 - p);
    }
    return sum / (double)pred->total;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Convolution
 * ══════════════════════════════════════════════════════════════════════════════*/

/* 473: Conv2d — naive implementation (no padding, stride=1).
 *   Input:  [batch, in_channels, H, W]
 *   Kernel: [out_channels, in_channels, kH, kW]
 *   Output: [batch, out_channels, H-kH+1, W-kW+1]
 */
static VmTensor* vm_tensor_conv2d(VmRegionStack* rs, const VmTensor* input,
                                  const VmTensor* kernel) {
    if (!input || !kernel) return NULL;
    if (input->n_dims != 4 || kernel->n_dims != 4) return NULL;
    if (input->shape[1] != kernel->shape[1]) return NULL;  /* in_channels must match */

    int64_t batch = input->shape[0];
    int64_t in_c  = input->shape[1];
    int64_t H     = input->shape[2];
    int64_t W     = input->shape[3];
    int64_t out_c = kernel->shape[0];
    int64_t kH    = kernel->shape[2];
    int64_t kW    = kernel->shape[3];

    int64_t oH = H - kH + 1;
    int64_t oW = W - kW + 1;
    if (oH <= 0 || oW <= 0) return NULL;

    int64_t out_shape[4] = { batch, out_c, oH, oW };
    VmTensor* out = vm_tensor_zeros(rs, out_shape, 4);
    if (!out) return NULL;

    /* Strides for 4D indexing (contiguous, row-major) */
    int64_t in_s0 = in_c * H * W, in_s1 = H * W, in_s2 = W;
    int64_t k_s0 = in_c * kH * kW, k_s1 = kH * kW, k_s2 = kW;
    int64_t o_s0 = out_c * oH * oW, o_s1 = oH * oW, o_s2 = oW;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t oc = 0; oc < out_c; oc++) {
            for (int64_t oh = 0; oh < oH; oh++) {
                for (int64_t ow = 0; ow < oW; ow++) {
                    double sum = 0.0;
                    for (int64_t ic = 0; ic < in_c; ic++) {
                        for (int64_t kh = 0; kh < kH; kh++) {
                            for (int64_t kw = 0; kw < kW; kw++) {
                                double iv = input->data[b*in_s0 + ic*in_s1 + (oh+kh)*in_s2 + (ow+kw)];
                                double kv = kernel->data[oc*k_s0 + ic*k_s1 + kh*k_s2 + kw];
                                sum += iv * kv;
                            }
                        }
                    }
                    out->data[b*o_s0 + oc*o_s1 + oh*o_s2 + ow] = sum;
                }
            }
        }
    }
    return out;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Pooling
 * ══════════════════════════════════════════════════════════════════════════════*/

/* 474: Max pooling 2D — [batch, channels, H, W] with kernel_size, stride. */
static VmTensor* vm_tensor_maxpool2d(VmRegionStack* rs, const VmTensor* input,
                                     int64_t pool_h, int64_t pool_w, int64_t stride) {
    if (!input || input->n_dims != 4) return NULL;
    if (pool_h <= 0 || pool_w <= 0 || stride <= 0) return NULL;

    int64_t batch = input->shape[0];
    int64_t chans = input->shape[1];
    int64_t H     = input->shape[2];
    int64_t W     = input->shape[3];

    int64_t oH = (H - pool_h) / stride + 1;
    int64_t oW = (W - pool_w) / stride + 1;
    if (oH <= 0 || oW <= 0) return NULL;

    int64_t out_shape[4] = { batch, chans, oH, oW };
    VmTensor* out = vm_tensor_new(rs, out_shape, 4);
    if (!out) return NULL;

    int64_t in_s0 = chans * H * W, in_s1 = H * W, in_s2 = W;
    int64_t o_s0 = chans * oH * oW, o_s1 = oH * oW, o_s2 = oW;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t c = 0; c < chans; c++) {
            for (int64_t oh = 0; oh < oH; oh++) {
                for (int64_t ow = 0; ow < oW; ow++) {
                    double max_val = -DBL_MAX;
                    for (int64_t ph = 0; ph < pool_h; ph++) {
                        for (int64_t pw = 0; pw < pool_w; pw++) {
                            int64_t ih = oh * stride + ph;
                            int64_t iw = ow * stride + pw;
                            double v = input->data[b*in_s0 + c*in_s1 + ih*in_s2 + iw];
                            if (v > max_val) max_val = v;
                        }
                    }
                    out->data[b*o_s0 + c*o_s1 + oh*o_s2 + ow] = max_val;
                }
            }
        }
    }
    return out;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Normalization
 * ══════════════════════════════════════════════════════════════════════════════*/

/* 475: Layer normalization along last axis.
 * out[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
 * gamma and beta can be NULL for no affine transform. */
static VmTensor* vm_tensor_layer_norm(VmRegionStack* rs, const VmTensor* t,
                                      const VmTensor* gamma, const VmTensor* beta) {
    if (!t || t->n_dims < 1) return NULL;

    VmTensor* out = vm_tensor_new(rs, t->shape, t->n_dims);
    if (!out) return NULL;

    int64_t last_dim = t->shape[t->n_dims - 1];
    int64_t n_slices = t->total / last_dim;
    double eps = 1e-5;

    for (int64_t s = 0; s < n_slices; s++) {
        double* src = t->data + s * last_dim;
        double* dst = out->data + s * last_dim;

        /* Compute mean */
        double mean = 0.0;
        for (int64_t i = 0; i < last_dim; i++) mean += src[i];
        mean /= (double)last_dim;

        /* Compute variance */
        double var = 0.0;
        for (int64_t i = 0; i < last_dim; i++) {
            double d = src[i] - mean;
            var += d * d;
        }
        var /= (double)last_dim;

        double inv_std = 1.0 / sqrt(var + eps);

        /* Normalize and apply affine */
        for (int64_t i = 0; i < last_dim; i++) {
            double norm = (src[i] - mean) * inv_std;
            if (gamma) norm *= gamma->data[i];
            if (beta)  norm += beta->data[i];
            dst[i] = norm;
        }
    }

    return out;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Comparison
 * ══════════════════════════════════════════════════════════════════════════════*/

/* 476: Element-wise equality check (returns 1.0 if equal within eps, 0.0 otherwise). */
static int vm_tensor_allclose(const VmTensor* a, const VmTensor* b, double atol) {
    if (!a || !b) return 0;
    if (a->total != b->total) return 0;
    if (a->n_dims != b->n_dims) return 0;
    for (int i = 0; i < a->n_dims; i++) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    for (int64_t i = 0; i < a->total; i++) {
        if (fabs(a->data[i] - b->data[i]) > atol) return 0;
    }
    return 1;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Concatenation
 * ══════════════════════════════════════════════════════════════════════════════*/

/* 477: Concatenate two tensors along specified axis. */
static VmTensor* vm_tensor_concat(VmRegionStack* rs, const VmTensor* a,
                                  const VmTensor* b, int axis) {
    if (!a || !b) return NULL;
    if (a->n_dims != b->n_dims) return NULL;
    if (axis < 0) axis += a->n_dims;
    if (axis < 0 || axis >= a->n_dims) return NULL;

    /* All dims except axis must match */
    for (int i = 0; i < a->n_dims; i++) {
        if (i != axis && a->shape[i] != b->shape[i]) return NULL;
    }

    int64_t out_shape[VM_TENSOR_MAX_DIMS];
    for (int i = 0; i < a->n_dims; i++) {
        out_shape[i] = (i == axis) ? a->shape[i] + b->shape[i] : a->shape[i];
    }

    VmTensor* out = vm_tensor_new(rs, out_shape, a->n_dims);
    if (!out) return NULL;

    /* Copy elements: iterate output, map to source */
    int64_t indices[VM_TENSOR_MAX_DIMS];
    for (int64_t flat = 0; flat < out->total; flat++) {
        vm_tensor_unravel(flat, out_shape, a->n_dims, indices);

        const VmTensor* src;
        int64_t src_indices[VM_TENSOR_MAX_DIMS];
        memcpy(src_indices, indices, (size_t)a->n_dims * sizeof(int64_t));

        if (indices[axis] < a->shape[axis]) {
            src = a;
        } else {
            src = b;
            src_indices[axis] -= a->shape[axis];
        }

        int64_t src_off = vm_tensor_flat_offset(src, src_indices, src->n_dims);
        out->data[flat] = src->data[src_off];
    }

    return out;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Outer product
 * ══════════════════════════════════════════════════════════════════════════════*/

/* 478: Outer product of two 1D tensors → 2D matrix. */
static VmTensor* vm_tensor_outer(VmRegionStack* rs, const VmTensor* a, const VmTensor* b) {
    if (!a || !b || a->n_dims != 1 || b->n_dims != 1) return NULL;

    int64_t M = a->total, N = b->total;
    int64_t out_shape[2] = { M, N };
    VmTensor* out = vm_tensor_new(rs, out_shape, 2);
    if (!out) return NULL;

    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            out->data[i * N + j] = a->data[i] * b->data[j];
        }
    }
    return out;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Clamp
 * ══════════════════════════════════════════════════════════════════════════════*/

/* 479: Clamp all elements to [lo, hi]. */
static VmTensor* vm_tensor_clamp(VmRegionStack* rs, const VmTensor* t, double lo, double hi) {
    if (!t) return NULL;
    VmTensor* out = vm_tensor_new(rs, t->shape, t->n_dims);
    if (!out) return NULL;
    for (int64_t i = 0; i < t->total; i++) {
        double v = t->data[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out->data[i] = v;
    }
    return out;
}

/* ══════════════════════════════════════════════════════════════════════════════
 *  Self-Test
 * ══════════════════════════════════════════════════════════════════════════════*/

#ifdef VM_TENSOR_OPS_TEST
#include <assert.h>

/* Helper: check double equality within tolerance */
static void assert_near(double a, double b, double tol, const char* msg) {
    if (fabs(a - b) > tol) {
        fprintf(stderr, "FAIL: %s: expected %.10f, got %.10f (diff %.2e)\n",
                msg, b, a, fabs(a - b));
        assert(0);
    }
}

int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    /* --- element-wise add --- */
    {
        double a_data[] = { 1, 2, 3 };
        double b_data[] = { 4, 5, 6 };
        int64_t shape[] = { 3 };
        VmTensor* a = vm_tensor_from_data(&rs, a_data, shape, 1);
        VmTensor* b = vm_tensor_from_data(&rs, b_data, shape, 1);
        VmTensor* c = vm_tensor_add(&rs, a, b);
        assert(c && c->total == 3);
        assert(c->data[0] == 5.0 && c->data[1] == 7.0 && c->data[2] == 9.0);
        printf("  add:         OK\n");
    }

    /* --- element-wise mul --- */
    {
        double a_data[] = { 2, 3 };
        double b_data[] = { 4, 5 };
        int64_t shape[] = { 2 };
        VmTensor* a = vm_tensor_from_data(&rs, a_data, shape, 1);
        VmTensor* b = vm_tensor_from_data(&rs, b_data, shape, 1);
        VmTensor* c = vm_tensor_mul(&rs, a, b);
        assert(c->data[0] == 8.0 && c->data[1] == 15.0);
        printf("  mul:         OK\n");
    }

    /* --- broadcasting: [3] + [[1],[2]] → [[4,5,6],[5,6,7]] --- */
    {
        double a_data[] = { 3, 4, 5 };
        int64_t a_shape[] = { 3 };
        VmTensor* a = vm_tensor_from_data(&rs, a_data, a_shape, 1);

        double b_data[] = { 1, 2 };
        int64_t b_shape[] = { 2, 1 };
        VmTensor* b = vm_tensor_from_data(&rs, b_data, b_shape, 2);

        VmTensor* c = vm_tensor_add(&rs, a, b);
        assert(c != NULL);
        assert(c->n_dims == 2 && c->shape[0] == 2 && c->shape[1] == 3);
        /* row 0: 3+1=4, 4+1=5, 5+1=6 */
        assert(c->data[0] == 4.0 && c->data[1] == 5.0 && c->data[2] == 6.0);
        /* row 1: 3+2=5, 4+2=6, 5+2=7 */
        assert(c->data[3] == 5.0 && c->data[4] == 6.0 && c->data[5] == 7.0);
        printf("  broadcast:   OK\n");
    }

    /* --- broadcasting incompatible shapes fail --- */
    {
        int64_t s1[] = { 3 }, s2[] = { 2 };
        VmTensor* a = vm_tensor_new(&rs, s1, 1);
        VmTensor* b = vm_tensor_new(&rs, s2, 1);
        VmTensor* c = vm_tensor_add(&rs, a, b);
        assert(c == NULL);
        printf("  broadcast-bad:OK\n");
    }

    /* --- matmul: [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]] --- */
    {
        double a_data[] = { 1, 2, 3, 4 };
        double b_data[] = { 5, 6, 7, 8 };
        int64_t shape[] = { 2, 2 };
        VmTensor* a = vm_tensor_from_data(&rs, a_data, shape, 2);
        VmTensor* b = vm_tensor_from_data(&rs, b_data, shape, 2);
        VmTensor* c = vm_tensor_matmul(&rs, a, b);
        assert(c != NULL);
        assert(c->data[0] == 19.0);
        assert(c->data[1] == 22.0);
        assert(c->data[2] == 43.0);
        assert(c->data[3] == 50.0);
        printf("  matmul:      OK\n");
    }

    /* --- matmul dimension mismatch --- */
    {
        int64_t s1[] = { 2, 3 }, s2[] = { 2, 2 };
        VmTensor* a = vm_tensor_new(&rs, s1, 2);
        VmTensor* b = vm_tensor_new(&rs, s2, 2);
        assert(vm_tensor_matmul(&rs, a, b) == NULL);
        printf("  matmul-bad:  OK\n");
    }

    /* --- matmul non-square --- */
    {
        double a_data[] = { 1, 2, 3, 4, 5, 6 }; /* 2x3 */
        double b_data[] = { 1, 2, 3, 4, 5, 6 }; /* 3x2 */
        int64_t s_a[] = { 2, 3 }, s_b[] = { 3, 2 };
        VmTensor* a = vm_tensor_from_data(&rs, a_data, s_a, 2);
        VmTensor* b = vm_tensor_from_data(&rs, b_data, s_b, 2);
        VmTensor* c = vm_tensor_matmul(&rs, a, b);
        assert(c && c->shape[0] == 2 && c->shape[1] == 2);
        /* [1,2,3].[1,3,5] = 1+6+15 = 22 */
        assert(c->data[0] == 22.0);
        /* [1,2,3].[2,4,6] = 2+8+18 = 28 */
        assert(c->data[1] == 28.0);
        printf("  matmul-rect: OK\n");
    }

    /* --- dot product --- */
    {
        double a_data[] = { 1, 2, 3 };
        double b_data[] = { 4, 5, 6 };
        int64_t shape[] = { 3 };
        VmTensor* a = vm_tensor_from_data(&rs, a_data, shape, 1);
        VmTensor* b = vm_tensor_from_data(&rs, b_data, shape, 1);
        double d = vm_tensor_dot(a, b);
        assert(d == 32.0); /* 4+10+18 */
        printf("  dot:         OK\n");
    }

    /* --- sum along axis 0 --- */
    {
        double data[] = { 1, 2, 3, 4, 5, 6 }; /* 2x3 */
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* s = vm_tensor_sum(&rs, t, 0);
        assert(s != NULL);
        assert(s->shape[0] == 3);
        assert(s->data[0] == 5.0);  /* 1+4 */
        assert(s->data[1] == 7.0);  /* 2+5 */
        assert(s->data[2] == 9.0);  /* 3+6 */
        printf("  sum-axis0:   OK\n");
    }

    /* --- sum along axis 1 --- */
    {
        double data[] = { 1, 2, 3, 4, 5, 6 }; /* 2x3 */
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* s = vm_tensor_sum(&rs, t, 1);
        assert(s != NULL);
        assert(s->shape[0] == 2);
        assert(s->data[0] == 6.0);   /* 1+2+3 */
        assert(s->data[1] == 15.0);  /* 4+5+6 */
        printf("  sum-axis1:   OK\n");
    }

    /* --- mean --- */
    {
        double data[] = { 2, 4, 6, 8 };
        int64_t shape[] = { 2, 2 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* m = vm_tensor_mean(&rs, t, 0);
        assert(m && m->data[0] == 4.0 && m->data[1] == 6.0); /* (2+6)/2, (4+8)/2 */
        printf("  mean:        OK\n");
    }

    /* --- max along axis --- */
    {
        double data[] = { 1, 5, 3, 4, 2, 6 }; /* 2x3 */
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* m = vm_tensor_max(&rs, t, 0);
        assert(m && m->data[0] == 4.0 && m->data[1] == 5.0 && m->data[2] == 6.0);
        printf("  max-axis0:   OK\n");
    }

    /* --- min along axis --- */
    {
        double data[] = { 1, 5, 3, 4, 2, 6 };
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* m = vm_tensor_min(&rs, t, 1);
        assert(m && m->data[0] == 1.0 && m->data[1] == 2.0);
        printf("  min-axis1:   OK\n");
    }

    /* --- reduce_all --- */
    {
        double data[] = { 1, 2, 3, 4 };
        int64_t shape[] = { 2, 2 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        assert(vm_tensor_reduce_all(t, VM_REDUCE_SUM) == 10.0);
        assert(vm_tensor_reduce_all(t, VM_REDUCE_MAX) == 4.0);
        assert(vm_tensor_reduce_all(t, VM_REDUCE_MIN) == 1.0);
        assert_near(vm_tensor_reduce_all(t, VM_REDUCE_MEAN), 2.5, 1e-12, "reduce_all mean");
        assert(vm_tensor_reduce_all(t, VM_REDUCE_PROD) == 24.0);
        printf("  reduce_all:  OK\n");
    }

    /* --- relu --- */
    {
        double data[] = { -1, 0, 1, -0.5, 2 };
        int64_t shape[] = { 5 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 1);
        VmTensor* r = vm_tensor_relu(&rs, t);
        assert(r != NULL);
        assert(r->data[0] == 0.0);
        assert(r->data[1] == 0.0);
        assert(r->data[2] == 1.0);
        assert(r->data[3] == 0.0);
        assert(r->data[4] == 2.0);
        printf("  relu:        OK\n");
    }

    /* --- sigmoid --- */
    {
        double data[] = { 0.0 };
        int64_t shape[] = { 1 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 1);
        VmTensor* s = vm_tensor_sigmoid(&rs, t);
        assert_near(s->data[0], 0.5, 1e-10, "sigmoid(0)");
        printf("  sigmoid:     OK\n");
    }

    /* --- tanh --- */
    {
        double data[] = { 0.0 };
        int64_t shape[] = { 1 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 1);
        VmTensor* s = vm_tensor_tanh_act(&rs, t);
        assert_near(s->data[0], 0.0, 1e-10, "tanh(0)");
        printf("  tanh:        OK\n");
    }

    /* --- softmax --- */
    {
        double data[] = { 1, 2, 3 };
        int64_t shape[] = { 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 1);
        VmTensor* s = vm_tensor_softmax(&rs, t, 0);
        assert(s != NULL);
        /* softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652] */
        assert_near(s->data[0], 0.0900306, 1e-4, "softmax[0]");
        assert_near(s->data[1], 0.2447285, 1e-4, "softmax[1]");
        assert_near(s->data[2], 0.6652409, 1e-4, "softmax[2]");
        /* Sum must be 1 */
        double sm = s->data[0] + s->data[1] + s->data[2];
        assert_near(sm, 1.0, 1e-10, "softmax sum");
        printf("  softmax:     OK\n");
    }

    /* --- softmax 2D along axis 1 --- */
    {
        double data[] = { 1, 2, 3, 1, 2, 3 };
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* s = vm_tensor_softmax(&rs, t, 1);
        assert(s != NULL);
        /* Each row should softmax independently */
        double row0_sum = s->data[0] + s->data[1] + s->data[2];
        double row1_sum = s->data[3] + s->data[4] + s->data[5];
        assert_near(row0_sum, 1.0, 1e-10, "softmax 2D row0 sum");
        assert_near(row1_sum, 1.0, 1e-10, "softmax 2D row1 sum");
        printf("  softmax-2D:  OK\n");
    }

    /* --- leaky relu --- */
    {
        double data[] = { -2, 0, 3 };
        int64_t shape[] = { 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 1);
        VmTensor* r = vm_tensor_leaky_relu(&rs, t);
        assert_near(r->data[0], -0.02, 1e-10, "leaky_relu(-2)");
        assert(r->data[1] == 0.0);
        assert(r->data[2] == 3.0);
        printf("  leaky_relu:  OK\n");
    }

    /* --- MSE loss --- */
    {
        double p[] = { 1, 2, 3 };
        double t_data[] = { 1.5, 2.5, 3.5 };
        int64_t shape[] = { 3 };
        VmTensor* pred = vm_tensor_from_data(&rs, p, shape, 1);
        VmTensor* target = vm_tensor_from_data(&rs, t_data, shape, 1);
        double mse = vm_tensor_mse_loss(pred, target);
        /* mean(0.25, 0.25, 0.25) = 0.25 */
        assert_near(mse, 0.25, 1e-10, "MSE");
        printf("  MSE:         OK\n");
    }

    /* --- cross-entropy loss --- */
    {
        double p[] = { 0.7, 0.2, 0.1 };
        double t_data[] = { 1, 0, 0 };
        int64_t shape[] = { 3 };
        VmTensor* pred = vm_tensor_from_data(&rs, p, shape, 1);
        VmTensor* target = vm_tensor_from_data(&rs, t_data, shape, 1);
        double ce = vm_tensor_cross_entropy_loss(pred, target);
        assert_near(ce, -log(0.7), 1e-10, "cross-entropy");
        printf("  cross-ent:   OK\n");
    }

    /* --- BCE loss --- */
    {
        double p[] = { 0.9, 0.1 };
        double t_data[] = { 1, 0 };
        int64_t shape[] = { 2 };
        VmTensor* pred = vm_tensor_from_data(&rs, p, shape, 1);
        VmTensor* target = vm_tensor_from_data(&rs, t_data, shape, 1);
        double bce = vm_tensor_bce_loss(pred, target);
        /* -mean(1*log(0.9) + 0, 0 + 1*log(0.9)) = -log(0.9) */
        assert_near(bce, -log(0.9), 1e-10, "BCE");
        printf("  BCE:         OK\n");
    }

    /* --- conv2d --- */
    {
        /* Simple 1-batch, 1-channel, 3x3 input, 1-output-channel, 2x2 kernel */
        double in_data[] = { 1,2,3, 4,5,6, 7,8,9 };
        int64_t in_shape[] = { 1, 1, 3, 3 };
        VmTensor* input = vm_tensor_from_data(&rs, in_data, in_shape, 4);

        double k_data[] = { 1,0, 0,1 };
        int64_t k_shape[] = { 1, 1, 2, 2 };
        VmTensor* kernel = vm_tensor_from_data(&rs, k_data, k_shape, 4);

        VmTensor* out = vm_tensor_conv2d(&rs, input, kernel);
        assert(out != NULL);
        assert(out->shape[0] == 1 && out->shape[1] == 1);
        assert(out->shape[2] == 2 && out->shape[3] == 2);
        /* out[0,0] = 1*1 + 2*0 + 4*0 + 5*1 = 6 */
        assert(out->data[0] == 6.0);
        /* out[0,1] = 2*1 + 3*0 + 5*0 + 6*1 = 8 */
        assert(out->data[1] == 8.0);
        /* out[1,0] = 4*1 + 5*0 + 7*0 + 8*1 = 12 */
        assert(out->data[2] == 12.0);
        /* out[1,1] = 5*1 + 6*0 + 8*0 + 9*1 = 14 */
        assert(out->data[3] == 14.0);
        printf("  conv2d:      OK\n");
    }

    /* --- maxpool2d --- */
    {
        double data[] = { 1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16 };
        int64_t shape[] = { 1, 1, 4, 4 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 4);
        VmTensor* p = vm_tensor_maxpool2d(&rs, t, 2, 2, 2);
        assert(p && p->shape[2] == 2 && p->shape[3] == 2);
        assert(p->data[0] == 6.0);   /* max(1,2,5,6) */
        assert(p->data[1] == 8.0);   /* max(3,4,7,8) */
        assert(p->data[2] == 14.0);  /* max(9,10,13,14) */
        assert(p->data[3] == 16.0);  /* max(11,12,15,16) */
        printf("  maxpool2d:   OK\n");
    }

    /* --- layer norm --- */
    {
        double data[] = { 1, 2, 3, 4, 5, 6 };
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* out = vm_tensor_layer_norm(&rs, t, NULL, NULL);
        assert(out != NULL);
        /* Row 0: mean=2, var=2/3, std=sqrt(2/3+1e-5)
         * (1-2)/std ≈ -1.2247, (2-2)/std ≈ 0, (3-2)/std ≈ 1.2247 */
        assert_near(out->data[1], 0.0, 1e-3, "layer_norm center");
        assert_near(out->data[0], -out->data[2], 1e-3, "layer_norm symmetry");
        printf("  layer_norm:  OK\n");
    }

    /* --- scale --- */
    {
        double data[] = { 1, 2, 3 };
        int64_t shape[] = { 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 1);
        VmTensor* s = vm_tensor_scale(&rs, t, 2.5);
        assert(s->data[0] == 2.5 && s->data[1] == 5.0 && s->data[2] == 7.5);
        printf("  scale:       OK\n");
    }

    /* --- neg --- */
    {
        double data[] = { 1, -2, 3 };
        int64_t shape[] = { 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 1);
        VmTensor* n = vm_tensor_neg(&rs, t);
        assert(n->data[0] == -1.0 && n->data[1] == 2.0 && n->data[2] == -3.0);
        printf("  neg:         OK\n");
    }

    /* --- allclose --- */
    {
        double a_data[] = { 1.0, 2.0, 3.0 };
        double b_data[] = { 1.0, 2.0, 3.00001 };
        int64_t shape[] = { 3 };
        VmTensor* a = vm_tensor_from_data(&rs, a_data, shape, 1);
        VmTensor* b = vm_tensor_from_data(&rs, b_data, shape, 1);
        assert(vm_tensor_allclose(a, b, 1e-4) == 1);
        assert(vm_tensor_allclose(a, b, 1e-6) == 0);
        printf("  allclose:    OK\n");
    }

    /* --- concat --- */
    {
        double a_data[] = { 1, 2, 3 };
        double b_data[] = { 4, 5, 6 };
        int64_t shape[] = { 3 };
        VmTensor* a = vm_tensor_from_data(&rs, a_data, shape, 1);
        VmTensor* b = vm_tensor_from_data(&rs, b_data, shape, 1);
        VmTensor* c = vm_tensor_concat(&rs, a, b, 0);
        assert(c && c->total == 6);
        assert(c->data[3] == 4.0 && c->data[5] == 6.0);
        printf("  concat:      OK\n");
    }

    /* --- outer product --- */
    {
        double a_data[] = { 1, 2, 3 };
        double b_data[] = { 4, 5 };
        int64_t s1[] = { 3 }, s2[] = { 2 };
        VmTensor* a = vm_tensor_from_data(&rs, a_data, s1, 1);
        VmTensor* b = vm_tensor_from_data(&rs, b_data, s2, 1);
        VmTensor* o = vm_tensor_outer(&rs, a, b);
        assert(o && o->shape[0] == 3 && o->shape[1] == 2);
        assert(o->data[0] == 4.0 && o->data[1] == 5.0);
        assert(o->data[4] == 12.0 && o->data[5] == 15.0);
        printf("  outer:       OK\n");
    }

    /* --- clamp --- */
    {
        double data[] = { -5, 0, 3, 10 };
        int64_t shape[] = { 4 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 1);
        VmTensor* c = vm_tensor_clamp(&rs, t, 0.0, 5.0);
        assert(c->data[0] == 0.0 && c->data[1] == 0.0);
        assert(c->data[2] == 3.0 && c->data[3] == 5.0);
        printf("  clamp:       OK\n");
    }

    /* --- batch matmul --- */
    {
        /* 2 batches of 2x2 matrices */
        double a_data[] = { 1,0,0,1,  2,0,0,2 };
        double b_data[] = { 3,4,5,6,  7,8,9,10 };
        int64_t shape[] = { 2, 2, 2 };
        VmTensor* a = vm_tensor_from_data(&rs, a_data, shape, 3);
        VmTensor* b = vm_tensor_from_data(&rs, b_data, shape, 3);
        VmTensor* c = vm_tensor_batch_matmul(&rs, a, b);
        assert(c != NULL);
        /* Batch 0: identity * [[3,4],[5,6]] = [[3,4],[5,6]] */
        assert(c->data[0] == 3.0 && c->data[1] == 4.0);
        /* Batch 1: 2I * [[7,8],[9,10]] = [[14,16],[18,20]] */
        assert(c->data[4] == 14.0 && c->data[7] == 20.0);
        printf("  batch_matmul:OK\n");
    }

    /* --- unary ops: exp, log, sqrt --- */
    {
        double data[] = { 1.0 };
        int64_t shape[] = { 1 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 1);

        VmTensor* e = vm_tensor_exp_op(&rs, t);
        assert_near(e->data[0], exp(1.0), 1e-10, "exp(1)");

        VmTensor* l = vm_tensor_log_op(&rs, e);
        assert_near(l->data[0], 1.0, 1e-10, "log(e)");

        double data4[] = { 4.0 };
        VmTensor* t4 = vm_tensor_from_data(&rs, data4, shape, 1);
        VmTensor* sq = vm_tensor_sqrt_op(&rs, t4);
        assert_near(sq->data[0], 2.0, 1e-10, "sqrt(4)");

        printf("  unary ops:   OK\n");
    }

    /* --- GELU approximation --- */
    {
        double data[] = { 0.0, 1.0, -1.0 };
        int64_t shape[] = { 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 1);
        VmTensor* g = vm_tensor_gelu(&rs, t);
        assert_near(g->data[0], 0.0, 1e-4, "gelu(0)");
        assert_near(g->data[1], 0.8413, 1e-3, "gelu(1)");
        printf("  gelu:        OK\n");
    }

    /* --- swish --- */
    {
        double data[] = { 0.0 };
        int64_t shape[] = { 1 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 1);
        VmTensor* s = vm_tensor_swish(&rs, t);
        assert_near(s->data[0], 0.0, 1e-10, "swish(0)");
        printf("  swish:       OK\n");
    }

    /* --- 1D sum reduction --- */
    {
        double data[] = { 1, 2, 3, 4 };
        int64_t shape[] = { 4 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 1);
        VmTensor* s = vm_tensor_sum(&rs, t, 0);
        assert(s != NULL && s->total == 1);
        assert(s->data[0] == 10.0);
        printf("  1D-sum:      OK\n");
    }

    /* --- product reduction --- */
    {
        double data[] = { 1, 2, 3, 4, 5, 6 };
        int64_t shape[] = { 2, 3 };
        VmTensor* t = vm_tensor_from_data(&rs, data, shape, 2);
        VmTensor* p = vm_tensor_reduce(&rs, t, 1, VM_REDUCE_PROD);
        assert(p && p->total == 2);
        assert(p->data[0] == 6.0);   /* 1*2*3 */
        assert(p->data[1] == 120.0); /* 4*5*6 */
        printf("  prod-reduce: OK\n");
    }

    vm_region_stack_destroy(&rs);
    printf("vm_tensor_ops: ALL TESTS PASSED\n");
    return 0;
}
#endif
