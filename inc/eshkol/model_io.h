#ifndef ESHKOL_MODEL_IO_H
#define ESHKOL_MODEL_IO_H

#include <eshkol/eshkol.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct arena arena_t;

void eshkol_tensor_save_tagged(arena_t* arena,
                               const eshkol_tagged_value_t* path_tv,
                               const eshkol_tagged_value_t* tensor_tv,
                               eshkol_tagged_value_t* result);

void eshkol_tensor_load_tagged(arena_t* arena,
                               const eshkol_tagged_value_t* path_tv,
                               eshkol_tagged_value_t* result);

void eshkol_model_save_tagged(arena_t* arena,
                              const eshkol_tagged_value_t* path_tv,
                              const eshkol_tagged_value_t* entries_tv,
                              eshkol_tagged_value_t* result);

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
