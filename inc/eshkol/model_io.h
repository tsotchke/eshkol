#ifndef ESHKOL_MODEL_IO_H
#define ESHKOL_MODEL_IO_H

#include <eshkol/eshkol.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque arena allocator handle used by all the tagged model/tensor
 *        I/O entry points below.
 *
 * Values written to the `result` out-parameter of these functions (tensors,
 * cons cells, strings) are allocated from this arena and share its lifetime.
 */
typedef struct arena arena_t;

/**
 * @brief Save a single tensor to a binary checkpoint file (ESKM format).
 *
 * Writes the tensor's dimensions and flattened float64 elements to @p path
 * using the same little-endian, CRC32-checked container format shared with
 * eshkol_model_save_tagged() (magic "ESKM", format version 1). The tensor is
 * stored under an empty name field since this call always writes exactly one
 * record.
 *
 * @param arena     Unused by the current implementation; accepted for
 *                  interface symmetry with the other tagged model-I/O entry
 *                  points.
 * @param path_tv   Tagged value that must be a string naming the output file.
 * @param tensor_tv Tagged value that must reference a tensor; any other type
 *                  causes the call to fail.
 * @param result    Output: tagged boolean set to true on success, false if
 *                  @p path_tv / @p tensor_tv are invalid or the write fails.
 *                  Must not be NULL.
 */
void eshkol_tensor_save_tagged(arena_t* arena,
                               const eshkol_tagged_value_t* path_tv,
                               const eshkol_tagged_value_t* tensor_tv,
                               eshkol_tagged_value_t* result);

/**
 * @brief Load a single-tensor checkpoint file previously written by
 *        eshkol_tensor_save_tagged().
 *
 * @param arena   Arena used to allocate the resulting tensor and its
 *                backing storage.
 * @param path_tv Tagged value that must be a string naming the input file.
 * @param result  Output: tagged tensor value on success, or the tagged null
 *                value if the file is missing, fails its CRC32 check, has an
 *                unsupported format version, or does not contain exactly one
 *                tensor record. Must not be NULL.
 */
void eshkol_tensor_load_tagged(arena_t* arena,
                               const eshkol_tagged_value_t* path_tv,
                               eshkol_tagged_value_t* result);

/**
 * @brief Save a named collection of tensors (a model checkpoint) to disk.
 *
 * @param arena      Unused by the current implementation; accepted for
 *                   interface symmetry with the other tagged model-I/O entry
 *                   points.
 * @param path_tv    Tagged value that must be a string naming the output
 *                   file.
 * @param entries_tv Tagged value referencing a proper list of (name . tensor)
 *                   pairs, where each name is a tagged string/symbol and each
 *                   tensor is a tagged tensor value. A malformed list, or any
 *                   entry whose name/tensor is the wrong type, causes the
 *                   call to fail.
 * @param result     Output: tagged boolean set to true on success, false on
 *                   any validation or I/O failure. Must not be NULL.
 */
void eshkol_model_save_tagged(arena_t* arena,
                              const eshkol_tagged_value_t* path_tv,
                              const eshkol_tagged_value_t* entries_tv,
                              eshkol_tagged_value_t* result);

/**
 * @brief Load a model checkpoint previously written by
 *        eshkol_model_save_tagged() (or eshkol_tensor_save_tagged()).
 *
 * @param arena   Arena used to allocate the resulting list, pairs, name
 *                strings, and tensors.
 * @param path_tv Tagged value that must be a string naming the input file.
 * @param result  Output: tagged value bound to a proper list of
 *                (name . tensor) pairs in file order, or the tagged null
 *                value if the file is missing, fails its CRC32 check, or has
 *                an unsupported format version. Must not be NULL.
 */
void eshkol_model_load_tagged(arena_t* arena,
                              const eshkol_tagged_value_t* path_tv,
                              eshkol_tagged_value_t* result);

/* Numeric batch-norm / layer-norm kernel. gamma_tv / beta_tv are decoded as
 * either a scalar (broadcast) or a per-feature tensor. group_len / inner_stride
 * describe the normalization grouping (see model_io.cpp). Returns a new tensor
 * (eshkol_tensor_t*) or NULL on error. */
void* eshkol_tensor_normalize_apply(arena_t* arena,
                                    const eshkol_tagged_value_t* input_tv,
                                    int64_t group_len,
                                    int64_t inner_stride,
                                    const eshkol_tagged_value_t* gamma_tv,
                                    const eshkol_tagged_value_t* beta_tv,
                                    double epsilon);

/* Elementwise base ^ scalar exponent. Returns a new tensor or NULL. */
void* eshkol_tensor_pow_scalar(arena_t* arena,
                               const eshkol_tagged_value_t* base_tv,
                               double exponent);

/* Elementwise libm unary map over a tensor (op codes mirror
 * codegenMathFunction). Returns a new tensor or NULL. */
void* eshkol_tensor_map_libm(arena_t* arena,
                             const eshkol_tagged_value_t* in_tv,
                             int32_t op);

#ifdef __cplusplus
}
#endif

#endif
