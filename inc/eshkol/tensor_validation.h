#ifndef ESHKOL_TENSOR_VALIDATION_H
#define ESHKOL_TENSOR_VALIDATION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return the checked product of a positive-rank shape, or -1 when the shape
 * is malformed or its product cannot be represented by int64_t. Zero extents
 * are valid and produce an empty tensor. */
int64_t eshkol_tensor_shape_total(const int64_t* dims, int64_t ndim);

/* Validate a tensor descriptor's shape, element count, and backing storage. */
int eshkol_tensor_metadata_valid(const int64_t* dims, int64_t ndim,
                                 const void* elements, int64_t total);

/* Compute a NumPy-style broadcast shape and checked element count. */
int eshkol_tensor_broadcast_shape(const int64_t* a_dims, int64_t a_ndim,
                                  const int64_t* b_dims, int64_t b_ndim,
                                  int64_t* out_dims, int64_t* out_ndim,
                                  int64_t* out_total);

/* Validate a complete or partial row-major index and return its offset. */
int eshkol_tensor_index_offset(const int64_t* dims, int64_t ndim,
                               const int64_t* indices, int64_t n_indices,
                               int64_t* offset, int64_t* slice_total);

#ifdef __cplusplus
}
#endif

#endif
