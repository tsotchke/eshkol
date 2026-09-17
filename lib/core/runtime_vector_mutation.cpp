/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Representation-aware R7RS vector mutation helpers.
 *
 * This file is the container slot store boundary (ADR-0016). A sequence is
 * either a Scheme vector (inline 16-byte tagged slots) or a tensor (a dense
 * numeric carrier whose slot representation is declared by its dtype). Every
 * mutator -- vector-set!, vector-fill!, vector-copy!, tensor-set! -- reaches a
 * tensor slot through encode_tensor_slot() below and nowhere else, so a value
 * is always converted to the slot's declared representation or refused; its
 * payload bits are never reinterpreted.
 */

#include "arena_memory.h"

#include <eshkol/core/bignum.h>
#include <eshkol/core/rational.h>

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

bool is_ad_node(const eshkol_tagged_value_t& value, uint8_t type) {
    if (type == ESHKOL_VALUE_AD_NODE_PTR) return value.data.ptr_val != 0;
    if (type != ESHKOL_VALUE_CALLABLE || value.data.ptr_val == 0) return false;
    return subtype_of(reinterpret_cast<const void*>(value.data.ptr_val)) ==
           CALLABLE_SUBTYPE_AD_NODE;
}

// The real-number projection of the numeric tower: exactly the conversion
// `inexact` performs, and the one tensor construction applies to each element.
bool tagged_real_to_double(const eshkol_tagged_value_t& value, double* out) {
    const uint8_t type = base_type(value.type);
    if (type == ESHKOL_VALUE_DOUBLE) {
        *out = value.data.double_val;
        return true;
    }
    if (type == ESHKOL_VALUE_INT64) {
        *out = static_cast<double>(value.data.int_val);
        return true;
    }
    if (type == ESHKOL_VALUE_HEAP_PTR && value.data.ptr_val != 0) {
        void* object = reinterpret_cast<void*>(value.data.ptr_val);
        const uint8_t subtype = subtype_of(object);
        if (subtype == HEAP_SUBTYPE_RATIONAL) {
            *out = eshkol_rational_to_double(object);
            return true;
        }
        if (subtype == HEAP_SUBTYPE_BIGNUM) {
            *out = eshkol_bignum_to_double(
                reinterpret_cast<const eshkol_bignum_t*>(object));
            return true;
        }
    }
    return false;
}

// The single encoder for one tensor slot. `slot` addresses the slot itself:
// 8 bytes for every numeric dtype, a 16-byte tagged value for a dual tensor.
//
//  - A real number of any exactness becomes the dtype-reduced f64 the slot
//    declares.
//  - A reverse-mode AD node keeps the in-tensor carrier encoding the
//    differentiation operators already read back (its pointer, in an f64
//    tensor), so a store inside a differentiated function stays on the tape.
//  - A dual tensor holds tagged jets, so it takes the tagged value unchanged.
//  - Anything else has no representation in the slot and is refused before
//    the destination is touched.
bool tensor_slot_accepts(const eshkol_tensor_t* tensor,
                         const eshkol_tagged_value_t& value) {
    const uint8_t type = base_type(value.type);
    if (tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL) {
        double ignored = 0.0;
        return type == ESHKOL_VALUE_DUAL_NUMBER ||
               tagged_real_to_double(value, &ignored);
    }
    double ignored = 0.0;
    if (tagged_real_to_double(value, &ignored)) return true;
    return tensor->dtype == ESHKOL_TENSOR_DTYPE_F64 && is_ad_node(value, type);
}

void encode_tensor_slot(eshkol_tensor_t* tensor, int64_t index,
                        const eshkol_tagged_value_t& value) {
    if (tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL) {
        auto* slots = reinterpret_cast<eshkol_tagged_value_t*>(tensor->elements);
        double numeric = 0.0;
        slots[index] = base_type(value.type) == ESHKOL_VALUE_DUAL_NUMBER
            ? value
            : (tagged_real_to_double(value, &numeric), tagged_double(numeric));
        eshkol_region_write_barrier_range(tensor, slots + index, 1);
        return;
    }
    double numeric = 0.0;
    if (tagged_real_to_double(value, &numeric)) {
        reinterpret_cast<double*>(tensor->elements)[index] =
            eshkol_tensor_reduce_precision_value(
                numeric, static_cast<int64_t>(tensor->dtype));
        return;
    }
    tensor->elements[index] = static_cast<int64_t>(value.data.ptr_val);
}

// Resolve a tagged operand to a sequence object, or null when it is neither a
// Scheme vector nor a tensor.
void* sequence_object(const eshkol_tagged_value_t* sequence, uint8_t* subtype) {
    if (!sequence) return nullptr;
    const uint8_t type = base_type(sequence->type);
    const bool heap = type == ESHKOL_VALUE_HEAP_PTR ||
                      type == ESHKOL_VALUE_VECTOR_PTR ||
                      type == ESHKOL_VALUE_TENSOR_PTR;
    if (!heap || sequence->data.ptr_val == 0) return nullptr;
    void* object = reinterpret_cast<void*>(sequence->data.ptr_val);
    *subtype = subtype_of(object);
    if (*subtype != HEAP_SUBTYPE_VECTOR && *subtype != HEAP_SUBTYPE_TENSOR) {
        return nullptr;
    }
    return object;
}

}  // namespace

extern "C" int32_t eshkol_vector_copy_mutating(void* dst, int64_t at,
                                                const void* src, int64_t start,
                                                int64_t end) {
    if (!dst || !src) return ESHKOL_SLOT_STORE_NULL;

    const uint8_t dst_subtype = subtype_of(dst);
    const uint8_t src_subtype = subtype_of(src);
    if ((dst_subtype != HEAP_SUBTYPE_VECTOR && dst_subtype != HEAP_SUBTYPE_TENSOR) ||
        (src_subtype != HEAP_SUBTYPE_VECTOR && src_subtype != HEAP_SUBTYPE_TENSOR)) {
        return ESHKOL_SLOT_STORE_CONTAINER;
    }

    const int64_t dst_len = sequence_length(dst, dst_subtype);
    const int64_t src_len = sequence_length(src, src_subtype);
    if (end == -1) end = src_len;
    if (dst_len < 0 || src_len < 0 || at < 0 || start < 0 || end < start ||
        end > src_len || at > dst_len || end - start > dst_len - at) {
        return ESHKOL_SLOT_STORE_BOUNDS;
    }
    const int64_t count = end - start;
    if (count == 0) return ESHKOL_SLOT_STORE_OK;

    if (dst_subtype == HEAP_SUBTYPE_VECTOR && src_subtype == HEAP_SUBTYPE_VECTOR) {
        auto* dst_values = vector_elements(dst) + at;
        const auto* src_values = vector_elements(src) + start;
        std::memmove(dst_values, src_values,
                     static_cast<size_t>(count) * sizeof(eshkol_tagged_value_t));
        eshkol_region_write_barrier_range(dst, dst_values, static_cast<uint64_t>(count));
        return ESHKOL_SLOT_STORE_OK;
    }

    if (dst_subtype == HEAP_SUBTYPE_TENSOR && src_subtype == HEAP_SUBTYPE_TENSOR) {
        auto* dst_tensor = reinterpret_cast<eshkol_tensor_t*>(dst);
        const auto* src_tensor = reinterpret_cast<const eshkol_tensor_t*>(src);
        const bool dst_dual = dst_tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL;
        const bool src_dual = src_tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL;
        if (dst_dual != src_dual) return ESHKOL_SLOT_STORE_VALUE;
        if (dst_dual) {
            auto* dst_values = reinterpret_cast<eshkol_tagged_value_t*>(dst_tensor->elements) + at;
            const auto* src_values =
                reinterpret_cast<const eshkol_tagged_value_t*>(src_tensor->elements) + start;
            std::memmove(dst_values, src_values,
                         static_cast<size_t>(count) * sizeof(eshkol_tagged_value_t));
            eshkol_region_write_barrier_range(
                dst, dst_values, static_cast<uint64_t>(count));
            return ESHKOL_SLOT_STORE_OK;
        }
        if (dst_tensor->dtype == src_tensor->dtype) {
            std::memmove(dst_tensor->elements + at, src_tensor->elements + start,
                         static_cast<size_t>(count) * sizeof(int64_t));
            return ESHKOL_SLOT_STORE_OK;
        }
        const auto* src_values = reinterpret_cast<const double*>(src_tensor->elements);
        auto* dst_values = reinterpret_cast<double*>(dst_tensor->elements);
        for (int64_t i = 0; i < count; ++i) {
            dst_values[at + i] = eshkol_tensor_reduce_precision_value(
                src_values[start + i], static_cast<int64_t>(dst_tensor->dtype));
        }
        return ESHKOL_SLOT_STORE_OK;
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
        return ESHKOL_SLOT_STORE_OK;
    }

    auto* dst_tensor = reinterpret_cast<eshkol_tensor_t*>(dst);
    const auto* src_values = vector_elements(src) + start;
    if (dst_tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL) {
        auto* dst_values = reinterpret_cast<eshkol_tagged_value_t*>(dst_tensor->elements) + at;
        std::memcpy(dst_values, src_values,
                    static_cast<size_t>(count) * sizeof(eshkol_tagged_value_t));
        eshkol_region_write_barrier_range(
            dst, dst_values, static_cast<uint64_t>(count));
        return ESHKOL_SLOT_STORE_OK;
    }
    // Validate the complete source range before mutating the destination so a
    // refused value cannot leave a partially copied tensor behind.
    for (int64_t i = 0; i < count; ++i) {
        if (!tensor_slot_accepts(dst_tensor, src_values[i])) {
            return ESHKOL_SLOT_STORE_VALUE;
        }
    }
    for (int64_t i = 0; i < count; ++i) {
        encode_tensor_slot(dst_tensor, at + i, src_values[i]);
    }
    return ESHKOL_SLOT_STORE_OK;
}

extern "C" int32_t eshkol_tensor_slot_store(void* tensor_object, int64_t index,
                                             const eshkol_tagged_value_t* value) {
    if (!tensor_object || !value) return ESHKOL_SLOT_STORE_NULL;
    if (subtype_of(tensor_object) != HEAP_SUBTYPE_TENSOR) {
        return ESHKOL_SLOT_STORE_CONTAINER;
    }
    auto* tensor = reinterpret_cast<eshkol_tensor_t*>(tensor_object);
    const int64_t length = sequence_length(tensor_object, HEAP_SUBTYPE_TENSOR);
    if (index < 0 || index >= length) return ESHKOL_SLOT_STORE_BOUNDS;
    if (!tensor_slot_accepts(tensor, *value)) return ESHKOL_SLOT_STORE_VALUE;
    encode_tensor_slot(tensor, index, *value);
    return ESHKOL_SLOT_STORE_OK;
}

extern "C" int32_t eshkol_sequence_slot_store(const eshkol_tagged_value_t* sequence,
                                               int64_t index,
                                               const eshkol_tagged_value_t* value) {
    if (!value) return ESHKOL_SLOT_STORE_NULL;
    uint8_t subtype = 0xFF;
    void* object = sequence_object(sequence, &subtype);
    if (!object) return ESHKOL_SLOT_STORE_CONTAINER;
    if (subtype == HEAP_SUBTYPE_TENSOR) {
        return eshkol_tensor_slot_store(object, index, value);
    }
    const int64_t length = sequence_length(object, subtype);
    if (index < 0 || index >= length) return ESHKOL_SLOT_STORE_BOUNDS;
    auto* slot = vector_elements(object) + index;
    *slot = *value;
    eshkol_region_write_barrier_range(object, slot, 1);
    return ESHKOL_SLOT_STORE_OK;
}

extern "C" int32_t eshkol_sequence_fill(const eshkol_tagged_value_t* sequence,
                                         const eshkol_tagged_value_t* value) {
    if (!value) return ESHKOL_SLOT_STORE_NULL;
    uint8_t subtype = 0xFF;
    void* object = sequence_object(sequence, &subtype);
    if (!object) return ESHKOL_SLOT_STORE_CONTAINER;
    const int64_t length = sequence_length(object, subtype);
    if (length < 0) return ESHKOL_SLOT_STORE_BOUNDS;
    if (subtype == HEAP_SUBTYPE_TENSOR) {
        auto* tensor = reinterpret_cast<eshkol_tensor_t*>(object);
        if (!tensor_slot_accepts(tensor, *value)) return ESHKOL_SLOT_STORE_VALUE;
        for (int64_t i = 0; i < length; ++i) encode_tensor_slot(tensor, i, *value);
        return ESHKOL_SLOT_STORE_OK;
    }
    auto* slots = vector_elements(object);
    for (int64_t i = 0; i < length; ++i) slots[i] = *value;
    if (length > 0) {
        eshkol_region_write_barrier_range(object, slots, static_cast<uint64_t>(length));
    }
    return ESHKOL_SLOT_STORE_OK;
}
