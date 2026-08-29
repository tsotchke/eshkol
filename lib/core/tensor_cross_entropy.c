#include "eshkol/tensor_cross_entropy.h"

#include <math.h>
#include <string.h>

static double ce_read(const void* data, uint64_t index, int data_is_double_bits) {
    if (!data_is_double_bits) return ((const double*)data)[index];
    uint64_t bits = 0;
    memcpy(&bits, (const unsigned char*)data + index * sizeof(bits), sizeof(bits));
    double value = 0.0;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int ce_product(const uint64_t* shape, uint64_t ndim, uint64_t* out) {
    uint64_t total = 1;
    if (!shape || ndim == 0) return 0;
    for (uint64_t i = 0; i < ndim; ++i) {
        if (shape[i] == 0 || total > UINT64_MAX / shape[i]) return 0;
        total *= shape[i];
    }
    *out = total;
    return 1;
}

static int ce_same_shape(const uint64_t* a, uint64_t an,
                         const uint64_t* b, uint64_t bn) {
    if (!a || !b || an != bn) return 0;
    for (uint64_t i = 0; i < an; ++i)
        if (a[i] != b[i]) return 0;
    return 1;
}

static int ce_classify(const uint64_t* logits_shape, uint64_t logits_ndim,
                       const uint64_t* targets_shape, uint64_t targets_ndim,
                       uint64_t* total_out, uint64_t* rows_out,
                       uint64_t* classes_out, int* dense_out) {
    uint64_t total = 0;
    if (!ce_product(logits_shape, logits_ndim, &total))
        return ESHKOL_CROSS_ENTROPY_LOGITS_SHAPE;
    uint64_t classes = logits_shape[logits_ndim - 1];
    uint64_t rows = total / classes;
    if (ce_same_shape(logits_shape, logits_ndim, targets_shape, targets_ndim)) {
        if (total_out) *total_out = total;
        if (rows_out) *rows_out = rows;
        if (classes_out) *classes_out = classes;
        if (dense_out) *dense_out = 1;
        return ESHKOL_CROSS_ENTROPY_OK;
    }

    int indexed = 0;
    if (logits_ndim == 1) {
        indexed = targets_ndim == 1 && targets_shape && targets_shape[0] == 1;
    } else if (targets_ndim + 1 == logits_ndim && targets_shape) {
        indexed = 1;
        for (uint64_t i = 0; i < targets_ndim; ++i) {
            if (targets_shape[i] != logits_shape[i]) {
                indexed = 0;
                break;
            }
        }
    }
    if (!indexed) return ESHKOL_CROSS_ENTROPY_TARGET_SHAPE;
    if (total_out) *total_out = total;
    if (rows_out) *rows_out = rows;
    if (classes_out) *classes_out = classes;
    if (dense_out) *dense_out = 0;
    return ESHKOL_CROSS_ENTROPY_OK;
}

static int ce_validate(const void* logits_data, const uint64_t* logits_shape,
                       uint64_t logits_ndim, const void* targets_data,
                       const uint64_t* targets_shape, uint64_t targets_ndim,
                       int data_is_double_bits, uint64_t* total_out,
                       uint64_t* rows_out, uint64_t* classes_out,
                       int* dense_out) {
    if (!logits_data || !targets_data || !logits_shape || !targets_shape ||
        logits_ndim == 0 || targets_ndim == 0)
        return ESHKOL_CROSS_ENTROPY_TARGET_SHAPE;
    int status = ce_classify(logits_shape, logits_ndim, targets_shape,
                             targets_ndim, total_out, rows_out, classes_out,
                             dense_out);
    if (status != ESHKOL_CROSS_ENTROPY_OK) return status;

    uint64_t total = *total_out;
    uint64_t rows = *rows_out;
    uint64_t classes = *classes_out;
    int dense = *dense_out;
    for (uint64_t i = 0; i < total; ++i) {
        if (!isfinite(ce_read(logits_data, i, data_is_double_bits)))
            return ESHKOL_CROSS_ENTROPY_LOGITS_VALUE;
    }
    if (!dense) {
        for (uint64_t r = 0; r < rows; ++r) {
            double index = ce_read(targets_data, r, data_is_double_bits);
            if (!isfinite(index) || floor(index) != index ||
                index < 0.0 || index >= (double)classes)
                return ESHKOL_CROSS_ENTROPY_TARGET_INDEX;
        }
    } else {
        const double tolerance = 1e-9 * (classes > 1 ? (double)classes : 1.0);
        for (uint64_t r = 0; r < rows; ++r) {
            double sum = 0.0;
            for (uint64_t i = 0; i < classes; ++i) {
                double probability = ce_read(targets_data, r * classes + i,
                                              data_is_double_bits);
                if (!isfinite(probability) || probability < 0.0)
                    return ESHKOL_CROSS_ENTROPY_TARGET_PROBABILITY;
                sum += probability;
            }
            if (!isfinite(sum) || fabs(sum - 1.0) > tolerance)
                return ESHKOL_CROSS_ENTROPY_TARGET_PROBABILITY;
        }
    }
    return ESHKOL_CROSS_ENTROPY_OK;
}

int eshkol_cross_entropy_forward(const void* logits_data,
                                 const uint64_t* logits_shape,
                                 uint64_t logits_ndim,
                                 const void* targets_data,
                                 const uint64_t* targets_shape,
                                 uint64_t targets_ndim,
                                 int data_is_double_bits,
                                 double* loss_out) {
    uint64_t total = 0, rows = 0, classes = 0;
    int dense = 0;
    int status = ce_validate(logits_data, logits_shape, logits_ndim,
                             targets_data, targets_shape, targets_ndim,
                             data_is_double_bits, &total, &rows, &classes,
                             &dense);
    if (status != ESHKOL_CROSS_ENTROPY_OK) return status;

    double loss = 0.0;
    for (uint64_t r = 0; r < rows; ++r) {
        double maximum = ce_read(logits_data, r * classes, data_is_double_bits);
        for (uint64_t i = 1; i < classes; ++i) {
            double value = ce_read(logits_data, r * classes + i, data_is_double_bits);
            if (value > maximum) maximum = value;
        }
        double exp_sum = 0.0;
        for (uint64_t i = 0; i < classes; ++i)
            exp_sum += exp(ce_read(logits_data, r * classes + i,
                                   data_is_double_bits) - maximum);
        double log_sum_exp = maximum + log(exp_sum);
        if (dense) {
            for (uint64_t i = 0; i < classes; ++i)
                loss -= ce_read(targets_data, r * classes + i, data_is_double_bits) *
                        (ce_read(logits_data, r * classes + i, data_is_double_bits) -
                         log_sum_exp);
        } else {
            uint64_t index = (uint64_t)ce_read(targets_data, r, data_is_double_bits);
            loss -= ce_read(logits_data, r * classes + index, data_is_double_bits) -
                    log_sum_exp;
        }
    }
    if (loss_out) *loss_out = loss / (double)rows;
    return ESHKOL_CROSS_ENTROPY_OK;
}

int eshkol_cross_entropy_backward(const double* logits_data,
                                  const uint64_t* logits_shape,
                                  uint64_t logits_ndim,
                                  const double* targets_data,
                                  const uint64_t* targets_shape,
                                  uint64_t targets_ndim,
                                  double upstream,
                                  double* dlogits_out) {
    uint64_t total = 0, rows = 0, classes = 0;
    int dense = 0;
    int status = ce_validate(logits_data, logits_shape, logits_ndim,
                             targets_data, targets_shape, targets_ndim, 0,
                             &total, &rows, &classes, &dense);
    if (status != ESHKOL_CROSS_ENTROPY_OK || !dlogits_out) return status;

    for (uint64_t r = 0; r < rows; ++r) {
        double maximum = logits_data[r * classes];
        for (uint64_t i = 1; i < classes; ++i)
            if (logits_data[r * classes + i] > maximum)
                maximum = logits_data[r * classes + i];
        double exp_sum = 0.0;
        for (uint64_t i = 0; i < classes; ++i)
            exp_sum += exp(logits_data[r * classes + i] - maximum);
        for (uint64_t i = 0; i < classes; ++i) {
            double probability = exp(logits_data[r * classes + i] - maximum) / exp_sum;
            double target = dense ? targets_data[r * classes + i] : 0.0;
            if (!dense && i == (uint64_t)targets_data[r]) target = 1.0;
            dlogits_out[r * classes + i] =
                upstream * (probability - target) / (double)rows;
        }
    }
    return ESHKOL_CROSS_ENTROPY_OK;
}

const char* eshkol_cross_entropy_status_message(int status) {
    switch (status) {
        case ESHKOL_CROSS_ENTROPY_LOGITS_SHAPE:
            return "cross-entropy-loss: logits must be a non-empty rank-1-or-higher tensor";
        case ESHKOL_CROSS_ENTROPY_TARGET_SHAPE:
            return "cross-entropy-loss: targets must match logits shape or omit the class axis";
        case ESHKOL_CROSS_ENTROPY_TARGET_INDEX:
            return "cross-entropy-loss: indexed targets must be integral and in class range";
        case ESHKOL_CROSS_ENTROPY_TARGET_PROBABILITY:
            return "cross-entropy-loss: probability targets must be finite, non-negative, and sum to 1 per row";
        case ESHKOL_CROSS_ENTROPY_LOGITS_VALUE:
            return "cross-entropy-loss: logits must be finite";
        default:
            return "cross-entropy-loss: invalid targets";
    }
}
