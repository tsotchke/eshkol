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

/* Raise a catchable type error reporting the operand's actual runtime type.
 * Declared here (rather than including runtime.h) to keep this freestanding-
 * adjacent translation unit's include surface small. ABI-stable symbol. */
void eshkol_type_error_with_operand(const char* proc_name,
                                    const char* expected_type,
                                    const eshkol_tagged_value_t* actual);

/*
 * Centralized, type-checked tensor-operand unpack (ESH-0069).
 *
 * Every tensor op (activations, conv/pool, reductions, shape ops, …) must route
 * its primary tensor operand through this single helper instead of blindly
 * reinterpreting the operand pointer as an eshkol_tensor_t*. Behavior:
 *
 *   (a) operand is already a tensor (HEAP_SUBTYPE_TENSOR or legacy TENSOR_PTR)
 *       -> return its data pointer unchanged (zero-copy, hot path).
 *   (b) operand is a homogeneous numeric vector (HEAP_SUBTYPE_VECTOR whose every
 *       element is an int64 or double) -> coerce to a fresh 1-D tensor.
 *   (c) anything else (int, string, bool, null, pair, non-numeric/heterogeneous
 *       vector, …) -> raise a clean, catchable type error via
 *       eshkol_type_error_with_operand and never touch the struct.
 *
 * This makes it structurally impossible for a tensor op to segfault on a
 * wrong-typed operand: it either gets a valid tensor or the program sees a
 * catchable condition. `op_name` is used only for the error message.
 *
 * Returns the eshkol_tensor_t* (as void*) on success; on the error path it does
 * not return (the type error raises). The trailing `return nullptr` keeps the
 * compiler happy and is never reached.
 */
void* eshkol_tensor_operand_checked(const eshkol_tagged_value_t* val,
                                    const char* op_name) {
    if (val) {
        /* Consolidated HEAP_PTR form: dispatch on the object-header subtype. */
        if (val->type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            void* ptr = (void*)(uintptr_t)val->data.ptr_val;
            const eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(ptr);
            if (hdr) {
                if (hdr->subtype == HEAP_SUBTYPE_TENSOR) {
                    return ptr;  /* already a tensor — zero-copy fast path */
                }
                if (hdr->subtype == HEAP_SUBTYPE_VECTOR) {
                    /* Coerce a *homogeneous numeric* vector to a 1-D tensor.
                     * Layout: [len:i64][eshkol_tagged_value_t elems...]. */
                    char* v = (char*)ptr;
                    int64_t len = *(int64_t*)v;
                    if (len < 0) len = 0;
                    const eshkol_tagged_value_t* elems =
                        (const eshkol_tagged_value_t*)(v + sizeof(int64_t));
                    for (int64_t i = 0; i < len; i++) {
                        uint8_t bt = (uint8_t)(elems[i].type & 0x0F);
                        if (bt != ESHKOL_VALUE_INT64 && bt != ESHKOL_VALUE_DOUBLE) {
                            /* heterogeneous / non-numeric vector — not coercible */
                            eshkol_type_error_with_operand(
                                op_name, "tensor or numeric vector", val);
                            return nullptr;  /* not reached */
                        }
                    }
                    arena_t* arena = get_global_arena();
                    eshkol_tensor_t* t =
                        arena_allocate_tensor_full(arena, 1, (uint64_t)len);
                    if (!t) return nullptr;
                    if (t->dimensions) t->dimensions[0] = (uint64_t)len;
                    for (int64_t i = 0; i < len; i++) {
                        const eshkol_tagged_value_t* e = &elems[i];
                        double d = ((e->type & 0x0F) == ESHKOL_VALUE_DOUBLE)
                                       ? e->data.double_val
                                       : (double)e->data.int_val;
                        std::memcpy(&t->elements[i], &d, sizeof(double));
                    }
                    return t;
                }
            }
        }
        /* Legacy direct TENSOR_PTR type tag. */
        if (val->type == ESHKOL_VALUE_TENSOR_PTR && val->data.ptr_val) {
            return (void*)(uintptr_t)val->data.ptr_val;
        }
    }

    /* Not a tensor and not a coercible numeric vector: clean, catchable error
     * instead of a segfault from misreading the struct. */
    eshkol_type_error_with_operand(op_name, "tensor", val);
    return nullptr;  /* not reached (type error raises) */
}

/**
 * @brief Allocate an empty tensor object (header only, no dimensions/elements).
 *
 * Allocates `sizeof(eshkol_tensor_t)` bytes plus a header, 64-byte aligned,
 * tags the header with HEAP_SUBTYPE_TENSOR, and initializes the tensor to
 * the zero/empty state: no dimensions array, no elements array,
 * `num_dimensions = 0`, `total_elements = 0`, and dtype defaulted to
 * ESHKOL_TENSOR_DTYPE_F64. Callers that need a populated tensor should use
 * arena_allocate_tensor_full instead, which calls this and then allocates
 * the dimensions/elements storage. The tensor is arena-owned and lives
 * until the arena is freed/reset.
 *
 * @param arena  Arena to allocate from (must not be null).
 * @return       Newly allocated empty tensor, or nullptr on failure.
 */
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
    tensor->dtype = ESHKOL_TENSOR_DTYPE_F64;  // default precision

    return tensor;
}

/**
 * @brief Allocate a fully-populated tensor: header, dimensions array, and elements array.
 *
 * Calls arena_allocate_tensor_with_header for the tensor struct itself, then
 * (if `num_dims > 0`) allocates a 64-byte-aligned `uint64_t[num_dims]`
 * dimensions array, and (if `total_elements > 0`) allocates a 64-byte-aligned
 * `int64_t[total_elements]` elements array zero-initialized via memset
 * (elements store IEEE-754 double bit patterns, per eshkol_tensor_t's
 * layout note — a zeroed element therefore represents 0.0). Both allocation
 * sizes are overflow-checked before allocating. On success, sets
 * `tensor->num_dimensions` and `tensor->total_elements` to the requested
 * values. Everything allocated is arena-owned and lives until the arena is
 * freed/reset; a failure partway through leaves the already-allocated
 * pieces arena-owned garbage (not individually freed) since the arena
 * doesn't support partial frees.
 *
 * @param arena           Arena to allocate from (must not be null).
 * @param num_dims        Number of dimensions (rank); 0 allocates no dimensions array.
 * @param total_elements  Total element count (product of dimensions); 0 allocates no elements array.
 * @return                Newly allocated, populated tensor, or nullptr on failure.
 */
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
