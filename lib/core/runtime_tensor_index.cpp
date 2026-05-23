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

extern eshkol_tagged_value_t arena_tagged_cons_get_tagged_value(
    const void* cell, bool is_cdr);

static inline uint8_t eshkol_base_type(uint8_t type) {
    return type < 8 ? (type & 0x0F) : type;
}

static inline int64_t tagged_to_int64(const eshkol_tagged_value_t& value) {
    const uint8_t base = eshkol_base_type(value.type);
    if (base == ESHKOL_VALUE_DOUBLE) {
        double d;
        std::memcpy(&d, &value.data, sizeof(double));
        return (int64_t)d;
    }
    return (int64_t)value.data.int_val;
}

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
