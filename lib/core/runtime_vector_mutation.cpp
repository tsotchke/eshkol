/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Representation-aware R7RS vector mutation helpers.
 */

#include "arena_memory.h"

#include <cstdint>
#include <cstring>

namespace {

uint8_t subtype_of(const void* ptr) {
    if (!ptr) return 0xFF;
    return ESHKOL_GET_HEADER(const_cast<void*>(ptr))->subtype;
}

uint8_t base_type(uint8_t type) {
    return type >= 8 ? type : static_cast<uint8_t>(type & 0x0F);
}

int64_t sequence_length(const void* ptr, uint8_t subtype) {
    if (subtype == HEAP_SUBTYPE_VECTOR) {
        return *reinterpret_cast<const int64_t*>(ptr);
    }
    if (subtype == HEAP_SUBTYPE_TENSOR) {
        const auto* tensor = reinterpret_cast<const eshkol_tensor_t*>(ptr);
        return tensor->total_elements > static_cast<uint64_t>(INT64_MAX)
            ? -1 : static_cast<int64_t>(tensor->total_elements);
    }
    return -1;
}

eshkol_tagged_value_t* vector_elements(void* ptr) {
    return reinterpret_cast<eshkol_tagged_value_t*>(
        reinterpret_cast<uint8_t*>(ptr) + sizeof(int64_t));
}

const eshkol_tagged_value_t* vector_elements(const void* ptr) {
    return reinterpret_cast<const eshkol_tagged_value_t*>(
        reinterpret_cast<const uint8_t*>(ptr) + sizeof(int64_t));
}

eshkol_tagged_value_t tagged_double(double value) {
    eshkol_tagged_value_t out{};
    out.type = ESHKOL_VALUE_DOUBLE;
    out.flags = ESHKOL_VALUE_INEXACT_FLAG;
    out.data.double_val = value;
    return out;
}

bool tagged_numeric_to_double(const eshkol_tagged_value_t& value, double* out) {
    const uint8_t type = base_type(value.type);
    if (type == ESHKOL_VALUE_INT64) {
        *out = static_cast<double>(value.data.int_val);
        return true;
    }
    if (type == ESHKOL_VALUE_DOUBLE) {
        *out = value.data.double_val;
        return true;
    }
    return false;
}

}  // namespace

extern "C" int32_t eshkol_vector_copy_mutating(void* dst, int64_t at,
                                                const void* src, int64_t start,
                                                int64_t end) {
    if (!dst || !src) return ESHKOL_VECTOR_COPY_NULL;

    const uint8_t dst_subtype = subtype_of(dst);
    const uint8_t src_subtype = subtype_of(src);
    if ((dst_subtype != HEAP_SUBTYPE_VECTOR && dst_subtype != HEAP_SUBTYPE_TENSOR) ||
        (src_subtype != HEAP_SUBTYPE_VECTOR && src_subtype != HEAP_SUBTYPE_TENSOR)) {
        return ESHKOL_VECTOR_COPY_TYPE;
    }

    const int64_t dst_len = sequence_length(dst, dst_subtype);
    const int64_t src_len = sequence_length(src, src_subtype);
    if (end == -1) end = src_len;
    if (dst_len < 0 || src_len < 0 || at < 0 || start < 0 || end < start ||
        end > src_len || at > dst_len || end - start > dst_len - at) {
        return ESHKOL_VECTOR_COPY_BOUNDS;
    }
    const int64_t count = end - start;
    if (count == 0) return ESHKOL_VECTOR_COPY_OK;

    if (dst_subtype == HEAP_SUBTYPE_VECTOR && src_subtype == HEAP_SUBTYPE_VECTOR) {
        auto* dst_values = vector_elements(dst) + at;
        const auto* src_values = vector_elements(src) + start;
        std::memmove(dst_values, src_values,
                     static_cast<size_t>(count) * sizeof(eshkol_tagged_value_t));
        eshkol_region_write_barrier_range(dst, dst_values, static_cast<uint64_t>(count));
        return ESHKOL_VECTOR_COPY_OK;
    }

    if (dst_subtype == HEAP_SUBTYPE_TENSOR && src_subtype == HEAP_SUBTYPE_TENSOR) {
        auto* dst_tensor = reinterpret_cast<eshkol_tensor_t*>(dst);
        const auto* src_tensor = reinterpret_cast<const eshkol_tensor_t*>(src);
        const bool dst_dual = dst_tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL;
        const bool src_dual = src_tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL;
        if (dst_dual != src_dual) return ESHKOL_VECTOR_COPY_TYPE;
        if (dst_dual) {
            auto* dst_values = reinterpret_cast<eshkol_tagged_value_t*>(dst_tensor->elements) + at;
            const auto* src_values =
                reinterpret_cast<const eshkol_tagged_value_t*>(src_tensor->elements) + start;
            std::memmove(dst_values, src_values,
                         static_cast<size_t>(count) * sizeof(eshkol_tagged_value_t));
            eshkol_region_write_barrier_range(
                dst, dst_values, static_cast<uint64_t>(count));
            return ESHKOL_VECTOR_COPY_OK;
        }
        if (dst_tensor->dtype == src_tensor->dtype) {
            std::memmove(dst_tensor->elements + at, src_tensor->elements + start,
                         static_cast<size_t>(count) * sizeof(int64_t));
            return ESHKOL_VECTOR_COPY_OK;
        }
        const auto* src_values = reinterpret_cast<const double*>(src_tensor->elements);
        auto* dst_values = reinterpret_cast<double*>(dst_tensor->elements);
        for (int64_t i = 0; i < count; ++i) {
            dst_values[at + i] = eshkol_tensor_reduce_precision_value(
                src_values[start + i], static_cast<int64_t>(dst_tensor->dtype));
        }
        return ESHKOL_VECTOR_COPY_OK;
    }

    if (dst_subtype == HEAP_SUBTYPE_VECTOR) {
        auto* dst_values = vector_elements(dst) + at;
        const auto* src_tensor = reinterpret_cast<const eshkol_tensor_t*>(src);
        if (src_tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL) {
            const auto* src_values =
                reinterpret_cast<const eshkol_tagged_value_t*>(src_tensor->elements) + start;
            std::memcpy(dst_values, src_values,
                        static_cast<size_t>(count) * sizeof(eshkol_tagged_value_t));
        } else {
            const auto* src_values = reinterpret_cast<const double*>(src_tensor->elements);
            for (int64_t i = 0; i < count; ++i) {
                dst_values[i] = tagged_double(src_values[start + i]);
            }
        }
        eshkol_region_write_barrier_range(dst, dst_values, static_cast<uint64_t>(count));
        return ESHKOL_VECTOR_COPY_OK;
    }

    auto* dst_tensor = reinterpret_cast<eshkol_tensor_t*>(dst);
    const auto* src_values = vector_elements(src) + start;
    if (dst_tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL) {
        auto* dst_values = reinterpret_cast<eshkol_tagged_value_t*>(dst_tensor->elements) + at;
        std::memcpy(dst_values, src_values,
                    static_cast<size_t>(count) * sizeof(eshkol_tagged_value_t));
        eshkol_region_write_barrier_range(
            dst, dst_values, static_cast<uint64_t>(count));
        return ESHKOL_VECTOR_COPY_OK;
    }
    // Validate the complete source range before mutating the destination so a
    // type error cannot leave a partially copied tensor behind.
    for (int64_t i = 0; i < count; ++i) {
        double ignored = 0.0;
        if (!tagged_numeric_to_double(src_values[i], &ignored)) {
            return ESHKOL_VECTOR_COPY_TYPE;
        }
    }
    auto* dst_values = reinterpret_cast<double*>(dst_tensor->elements);
    for (int64_t i = 0; i < count; ++i) {
        double numeric = 0.0;
        (void)tagged_numeric_to_double(src_values[i], &numeric);
        dst_values[at + i] = eshkol_tensor_reduce_precision_value(
            numeric, static_cast<int64_t>(dst_tensor->dtype));
    }
    return ESHKOL_VECTOR_COPY_OK;
}
