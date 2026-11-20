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

/* eshkol_xla_reduce axis operations 0..4 retain their unseeded ABI. The
 * seeded lowering combines its initial value with the returned slices. */
enum { ESHKOL_REDUCE_SEEDED_AXIS = 8 };

/* Products on either side of the selected axis. The output element count
 * must fit a double buffer even when the input has a zero extent. */
int eshkol_tensor_axis_geometry(const uint64_t* shape, int64_t rank, int64_t axis,
                                int64_t* outer, int64_t* inner);

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
