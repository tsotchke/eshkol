/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Freestanding-safe tensor index helpers.
 *
 * These helpers normalize tagged scalar/list index arguments and compute
 * tensor row-major offsets. They only inspect tagged values and tensor layout;
 * they do not allocate, raise, log, use files, read environment state, or touch
 * host process/thread/signal APIs.
 */

#include <eshkol/eshkol.h>

#include <cstdint>
#include <cstring>

extern "C" {

/** Freestanding-safe forward declaration of arena_memory.h's
 *  arena_tagged_cons_get_tagged_value(): reads the complete tagged value
 *  from the car (or cdr, if @p is_cdr) of the tagged cons @p cell. */
extern eshkol_tagged_value_t arena_tagged_cons_get_tagged_value(
    const void* cell, bool is_cdr);

/** @brief Strip flag bits from a tagged-value type byte, returning the base type (types >= 8 have no flag bits and pass through unchanged). */
static inline uint8_t eshkol_base_type(uint8_t type) {
    return type < 8 ? (type & 0x0F) : type;
}

/**
 * @brief Coerce a tagged scalar value to an int64 index.
 *
 * If the value's base type is ESHKOL_VALUE_DOUBLE, reinterprets the payload
 * bits as a double and truncates toward zero; otherwise reads the payload
 * directly as int_val (covers INT64/BOOL/CHAR and similar int-storage
 * types).
 *
 * @param value Tagged scalar value to convert.
 * @return      Integer index derived from @p value.
 */
static inline int64_t tagged_to_int64(const eshkol_tagged_value_t& value) {
    const uint8_t base = eshkol_base_type(value.type);
    if (base == ESHKOL_VALUE_DOUBLE) {
        double d;
        std::memcpy(&d, &value.data, sizeof(double));
        return (int64_t)d;
    }
    return (int64_t)value.data.int_val;
}

/**
 * @brief Extract a single int64 index from a tagged value that may be a scalar or a 1-element cons list.
 *
 * If @p tv_in is a HEAP_PTR whose object header reports HEAP_SUBTYPE_CONS,
 * unwraps one level by taking the cell's car (so `(vector-ref v (list 2))`
 * behaves like `(vector-ref v 2)`); otherwise treats @p tv_in as the index
 * directly. Only unwraps one cons level regardless of nesting.
 *
 * @param tv_in Tagged index argument (scalar or single-element list).
 * @return      Integer index, or 0 if @p tv_in is NULL.
 */
int64_t eshkol_unwrap_list_index(const eshkol_tagged_value_t* tv_in) {
    if (!tv_in) return 0;

    const eshkol_tagged_value_t tv = *tv_in;
    const uint8_t base_type = eshkol_base_type(tv.type);
    if (base_type == ESHKOL_VALUE_HEAP_PTR && tv.data.ptr_val != 0) {
        const eshkol_object_header_t* header =
            ESHKOL_GET_HEADER((void*)tv.data.ptr_val);
        if (header->subtype == HEAP_SUBTYPE_CONS) {
            const eshkol_tagged_value_t car =
                arena_tagged_cons_get_tagged_value((const void*)tv.data.ptr_val, false);
            return tagged_to_int64(car);
        }
    }

    return tagged_to_int64(tv);
}

/**
 * @brief Compute a row-major linear offset into a tensor from a tagged index argument.
 *
 * If @p tv_in is a cons list, walks it (up to @p ndim elements) treating each
 * element as one dimension's index, accumulating the standard row-major
 * offset: `linear = i0`, then `linear = linear * dims[k] + i_k` for each
 * subsequent index `i_k` (0-based dimension `k`). If fewer than @p ndim
 * indices are supplied (a partial/sub-tensor index, e.g. indexing only the
 * outer dimensions of a multi-dim tensor), the accumulated linear offset is
 * scaled by the product of the remaining (unindexed) dimension sizes so the
 * result points at the start of the corresponding sub-block. If @p tv_in is
 * not a cons list, it is treated as a single scalar index (via
 * tagged_to_int64()).
 *
 * @param tv_in Tagged index argument: a cons list of per-dimension indices,
 *              or a bare scalar index.
 * @param dims  Array of @p ndim dimension sizes (row-major, outermost
 *              first).
 * @param ndim  Number of dimensions in @p dims.
 * @return      Linear (flat) row-major offset, or 0 if @p tv_in is NULL.
 */
int64_t eshkol_tensor_linear_from_index_arg(
    const eshkol_tagged_value_t* tv_in,
    const int64_t* dims,
    int64_t ndim) {
    if (!tv_in) return 0;

    const eshkol_tagged_value_t tv = *tv_in;
    const uint8_t base_type = eshkol_base_type(tv.type);
    if (base_type == ESHKOL_VALUE_HEAP_PTR && tv.data.ptr_val != 0) {
        const eshkol_object_header_t* header =
            ESHKOL_GET_HEADER((void*)tv.data.ptr_val);
        if (header->subtype == HEAP_SUBTYPE_CONS) {
            int64_t linear = 0;
            int64_t count = 0;
            eshkol_tagged_value_t current = tv;

            while (true) {
                const uint8_t current_base = eshkol_base_type(current.type);
                if (current_base != ESHKOL_VALUE_HEAP_PTR ||
                    current.data.ptr_val == 0) {
                    break;
                }
                const eshkol_object_header_t* current_header =
                    ESHKOL_GET_HEADER((void*)current.data.ptr_val);
                if (current_header->subtype != HEAP_SUBTYPE_CONS) break;
                if (count >= ndim) break;

                const eshkol_tagged_value_t car =
                    arena_tagged_cons_get_tagged_value(
                        (const void*)current.data.ptr_val, false);
                const eshkol_tagged_value_t cdr =
                    arena_tagged_cons_get_tagged_value(
                        (const void*)current.data.ptr_val, true);
                const int64_t value = tagged_to_int64(car);

                if (count == 0) {
                    linear = value;
                } else {
                    linear = linear * dims[count] + value;
                }
                count++;
                current = cdr;
            }

            for (int64_t k = count; k < ndim; k++) {
                linear *= dims[k];
            }
            return linear;
        }
    }

    return tagged_to_int64(tv);
}

/**
 * @brief Compute the index to use for a `(vector-ref v idx)`-style access, tensor-aware.
 *
 * If @p vec_tv_in is a HEAP_PTR to an object whose header reports
 * HEAP_SUBTYPE_TENSOR, reinterprets it as a tensor layout (dimensions
 * pointer, dimension count, elements pointer, total element count) and
 * delegates to eshkol_tensor_linear_from_index_arg() so a multi-dimensional
 * index list is converted into the correct row-major linear offset.
 * Otherwise (plain Scheme vector or any non-tensor target), falls back to
 * eshkol_unwrap_list_index() to unwrap a possibly list-wrapped scalar index.
 *
 * @param vec_tv_in Tagged value for the collection being indexed.
 * @param idx_tv_in Tagged index argument (scalar or index list).
 * @return          Index/offset to use for the access; falls back to
 *                  eshkol_unwrap_list_index(idx_tv_in) if either argument is
 *                  NULL or @p vec_tv_in is not a tensor.
 */
int64_t eshkol_vref_unwrap_index(
    const eshkol_tagged_value_t* vec_tv_in,
    const eshkol_tagged_value_t* idx_tv_in) {
    if (!vec_tv_in || !idx_tv_in) return eshkol_unwrap_list_index(idx_tv_in);

    const eshkol_tagged_value_t vec = *vec_tv_in;
    const uint8_t base_type = eshkol_base_type(vec.type);
    if (base_type == ESHKOL_VALUE_HEAP_PTR && vec.data.ptr_val != 0) {
        const eshkol_object_header_t* header =
            ESHKOL_GET_HEADER((void*)vec.data.ptr_val);
        if (header->subtype == HEAP_SUBTYPE_TENSOR) {
            struct tensor_layout {
                uint64_t* dimensions;
                uint64_t num_dimensions;
                int64_t* elements;
                uint64_t total_elements;
            };
            const tensor_layout* tensor =
                (const tensor_layout*)(uintptr_t)vec.data.ptr_val;
            return eshkol_tensor_linear_from_index_arg(
                idx_tv_in,
                (const int64_t*)tensor->dimensions,
                (int64_t)tensor->num_dimensions);
        }
    }

    return eshkol_unwrap_list_index(idx_tv_in);
}

/**
 * @brief Count how many index dimensions a tagged index argument supplies.
 *
 * If @p tv_in is a cons list, counts its length by walking cdrs while each
 * node's header reports HEAP_SUBTYPE_CONS, stopping at the first non-cons
 * tail; a zero-length walk (e.g. an unrecognized structure) still reports 1,
 * since a single scalar index is always assumed to supply at least one
 * dimension. Any non-cons @p tv_in is treated as a single scalar index.
 *
 * @param tv_in Tagged index argument (index list or scalar).
 * @return      Number of index dimensions supplied (at least 1), or 1 if
 *              @p tv_in is NULL.
 */
int64_t eshkol_tensor_index_arg_count(const eshkol_tagged_value_t* tv_in) {
    if (!tv_in) return 1;

    const eshkol_tagged_value_t tv = *tv_in;
    const uint8_t base_type = eshkol_base_type(tv.type);
    if (base_type == ESHKOL_VALUE_HEAP_PTR && tv.data.ptr_val != 0) {
        const eshkol_object_header_t* header =
            ESHKOL_GET_HEADER((void*)tv.data.ptr_val);
        if (header->subtype == HEAP_SUBTYPE_CONS) {
            int64_t count = 0;
            eshkol_tagged_value_t current = tv;
            while (true) {
                const uint8_t current_base = eshkol_base_type(current.type);
                if (current_base != ESHKOL_VALUE_HEAP_PTR ||
                    current.data.ptr_val == 0) {
                    break;
                }
                const eshkol_object_header_t* current_header =
                    ESHKOL_GET_HEADER((void*)current.data.ptr_val);
                if (current_header->subtype != HEAP_SUBTYPE_CONS) break;
                count++;
                current = arena_tagged_cons_get_tagged_value(
                    (const void*)current.data.ptr_val, true);
            }
            return count > 0 ? count : 1;
        }
    }

    return 1;
}

}  // extern "C"
