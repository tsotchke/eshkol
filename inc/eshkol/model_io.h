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

#ifdef __cplusplus
}
#endif

#endif
