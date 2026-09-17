#ifndef ESHKOL_TENSOR_OBSERVATION_H
#define ESHKOL_TENSOR_OBSERVATION_H

#include <eshkol/tensor_validation.h>
#include <string.h>

/* ESKM rank-0 scalars may be inspected without relaxing the shape rules for
 * arithmetic, destinations, or general tensor construction. Keep the native
 * and VM operand guards on the same narrow observation contract. */
static inline int eshkol_tensor_operand_metadata_valid(
    const int64_t* dims, int64_t ndim, const void* elements, int64_t total,
    const char* operation) {
    if (ndim != 0) return eshkol_tensor_metadata_valid(dims, ndim, elements, total);
    return total == 1 && elements != NULL && operation != NULL &&
           (strcmp(operation, "tensor-shape") == 0 ||
            strcmp(operation, "tensor-data") == 0 ||
            strcmp(operation, "tensor->vector") == 0 ||
            strcmp(operation, "tensor-length") == 0);
}

#endif
