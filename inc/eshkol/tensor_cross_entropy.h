#ifndef ESHKOL_TENSOR_CROSS_ENTROPY_H
#define ESHKOL_TENSOR_CROSS_ENTROPY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ESHKOL_CROSS_ENTROPY_OK = 0,
    ESHKOL_CROSS_ENTROPY_LOGITS_SHAPE = 1,
    ESHKOL_CROSS_ENTROPY_TARGET_SHAPE = 2,
    ESHKOL_CROSS_ENTROPY_TARGET_INDEX = 3,
    ESHKOL_CROSS_ENTROPY_TARGET_PROBABILITY = 4,
    ESHKOL_CROSS_ENTROPY_LOGITS_VALUE = 5,
    ESHKOL_CROSS_ENTROPY_GAMMA_VALUE = 6
} eshkol_cross_entropy_status_t;

/* data_is_double_bits is non-zero when data points at int64_t IEEE-754
 * bit-pattern storage, as used by eshkol_tensor_t. */
int eshkol_cross_entropy_forward(const void* logits_data,
                                 const uint64_t* logits_shape,
                                 uint64_t logits_ndim,
                                 const void* targets_data,
                                 const uint64_t* targets_shape,
                                 uint64_t targets_ndim,
                                 int data_is_double_bits,
                                 double* loss_out);

/* Focal loss (Lin et al., 2017) over the same logits/targets contract as
 * eshkol_cross_entropy_forward, and validated by the same rules.
 *
 *     L = mean_rows -(1 - p_t)^gamma * sum_classes t_i * log(softmax(x)_i)
 *
 * where p_t is the true-class probability under the row's softmax (for a
 * one-hot or indexed target, the probability of the labelled class; for a
 * general probability row, its expectation under the target distribution).
 * At gamma = 0 the modulating factor is 1 and this is exactly
 * eshkol_cross_entropy_forward. gamma must be finite and non-negative. */
int eshkol_focal_loss_forward(const void* logits_data,
                              const uint64_t* logits_shape,
                              uint64_t logits_ndim,
                              const void* targets_data,
                              const uint64_t* targets_shape,
                              uint64_t targets_ndim,
                              int data_is_double_bits,
                              double gamma,
                              double* loss_out);

/* Computes the local derivative of the mean loss. dlogits_out is written,
 * not accumulated, so callers can add it to an existing reverse-mode buffer. */
int eshkol_cross_entropy_backward(const double* logits_data,
                                  const uint64_t* logits_shape,
                                  uint64_t logits_ndim,
                                  const double* targets_data,
                                  const uint64_t* targets_shape,
                                  uint64_t targets_ndim,
                                  double upstream,
                                  double* dlogits_out);

const char* eshkol_cross_entropy_status_message(int status);

#ifdef __cplusplus
}
#endif

#endif
