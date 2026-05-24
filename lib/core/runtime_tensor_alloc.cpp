/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Freestanding-safe tensor allocation helpers.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" {

eshkol_tensor_t* arena_allocate_tensor_with_header(arena_t* arena) {
    if (!arena) {
        eshkol_error("Invalid arena for tensor allocation");
        return nullptr;
    }

    const size_t data_size = sizeof(eshkol_tensor_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 63) & ~((size_t)63);

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 64);
    if (!mem) {
        eshkol_error("Failed to allocate tensor with header");
        return nullptr;
    }

    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_TENSOR;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    eshkol_tensor_t* tensor =
        (eshkol_tensor_t*)(mem + sizeof(eshkol_object_header_t));
    tensor->dimensions = nullptr;
    tensor->num_dimensions = 0;
    tensor->elements = nullptr;
    tensor->total_elements = 0;

    return tensor;
}

eshkol_tensor_t* arena_allocate_tensor_full(
    arena_t* arena, uint64_t num_dims, uint64_t total_elements) {
    if (!arena) {
        eshkol_error("Invalid arena for tensor allocation");
        return nullptr;
    }

    eshkol_tensor_t* tensor = arena_allocate_tensor_with_header(arena);
    if (!tensor) {
        return nullptr;
    }

    if (num_dims > 0) {
        if (num_dims > SIZE_MAX / sizeof(uint64_t)) {
            eshkol_error("Tensor dimensions allocation overflow (num_dims=%llu)",
                         (unsigned long long)num_dims);
            return nullptr;
        }

        tensor->dimensions = (uint64_t*)arena_allocate_aligned(
            arena, (size_t)num_dims * sizeof(uint64_t), 64);
        if (!tensor->dimensions) {
            eshkol_error("Failed to allocate tensor dimensions array");
            return nullptr;
        }
    }

    if (total_elements > 0) {
        if (total_elements > SIZE_MAX / sizeof(int64_t)) {
            eshkol_error("Tensor elements allocation overflow (total_elements=%llu)",
                         (unsigned long long)total_elements);
            return nullptr;
        }

        const size_t elem_size = (size_t)total_elements * sizeof(int64_t);
        tensor->elements = (int64_t*)arena_allocate_aligned(arena, elem_size, 64);
        if (!tensor->elements) {
            eshkol_error("Failed to allocate tensor elements array");
            return nullptr;
        }
        std::memset(tensor->elements, 0, elem_size);
    }

    tensor->num_dimensions = num_dims;
    tensor->total_elements = total_elements;

    return tensor;
}

}  // extern "C"
