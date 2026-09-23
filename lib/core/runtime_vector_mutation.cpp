/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Representation-aware R7RS vector mutation helpers.
 *
 * This file is the container slot store boundary (ADR-0020). A sequence is
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
#include <new>
#include <vector>

// Defined in runtime_autodiff.cpp: is `bits` a live reverse-mode node pointer?
extern "C" int eshkol_ad_node_probe(const arena_t* arena, uint64_t bits, int32_t expect_type);
// Defined in runtime_taylor.c: does the value reference a Taylor tower?
extern "C" int eshkol_is_taylor_tagged(const eshkol_tagged_value_t* tv);

namespace {

// ── #713: the order of a tagged store ────────────────────────────────────
//
// Stage, promote, then store. A value bound for a slot that outlives the
// value's region is promoted by the region write barrier BEFORE it reaches the
// slot, and the barrier is all-or-nothing: when the promotion cannot complete
// it raises with the staged values unchanged, so the store below it never runs
// and the destination keeps its previous contents. Storing first and fixing the
// slot up afterwards -- the order these helpers replace -- left the young
// pointer in the destination whenever the fix-up failed.

// One tagged value into one slot.
void store_tagged_value(const void* dst_object, eshkol_tagged_value_t* slot,
                        const eshkol_tagged_value_t& value) {
    eshkol_tagged_value_t staged;
    eshkol_region_write_barrier_into(&staged, dst_object, &value);
    *slot = staged;
}

// Stage n values bound for @p dst_object and promote them in ONE transaction,
// so structure they share stays shared and either all of them are promoted or
// the call raises. Returns the promoted values: @p src itself when no region is
// active (nothing can need promotion), otherwise a per-thread staging buffer
// that stays valid until the next store on this thread (a store never
// re-enters another store).
const eshkol_tagged_value_t* promote_staged(const void* dst_object,
                                            const eshkol_tagged_value_t* src,
                                            size_t n) {
    if (n == 0 || __region_stack_depth == 0) return src;
    static thread_local std::vector<eshkol_tagged_value_t> staging;
    bool grown = true;
    try {
        if (staging.size() < n) staging.resize(n);
    } catch (const std::bad_alloc&) {
        grown = false;
    }
    if (!grown) {
        eshkol_raise_allocation_failure("vector store", n * sizeof(eshkol_tagged_value_t));
    }
    std::memcpy(staging.data(), src, n * sizeof(eshkol_tagged_value_t));
    eshkol_region_write_barrier_range(dst_object, staging.data(), n);
    return staging.data();
}

// n tagged values into n slots, with memmove semantics (src and dst may
// overlap): either all of them are stored, promoted, or none is.
void store_tagged_values(const void* dst_object, eshkol_tagged_value_t* dst,
                         const eshkol_tagged_value_t* src, size_t n) {
    if (n == 0) return;
    const eshkol_tagged_value_t* values = promote_staged(dst_object, src, n);
    std::memmove(dst, values, n * sizeof(eshkol_tagged_value_t));
}

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
//  - A dual (jet) tensor holds tagged forward-mode carriers -- dual jets and
//    Taylor towers -- so it takes such a value unchanged; a real stored beside
//    them is reduced as above. A carrier arriving at a numeric tensor widens it
//    to a jet tensor first (widen_tensor_for).
//  - Anything else has no representation in the slot and is refused before
//    the destination is touched.
// A forward-mode derivative carrier: a first-order dual jet, or a Taylor tower
// (the exact tier's carrier, and derivative-n's). Both are whole values that a
// jet tensor slot holds unchanged; neither has an f64 representation.
bool is_forward_jet(const eshkol_tagged_value_t& value) {
    if (value.data.ptr_val == 0) return false;
    if (base_type(value.type) == ESHKOL_VALUE_DUAL_NUMBER) return true;
    return eshkol_is_taylor_tagged(&value) != 0;
}

bool tensor_slot_accepts(const eshkol_tensor_t* tensor,
                         const eshkol_tagged_value_t& value) {
    const uint8_t type = base_type(value.type);
    // A promoted carrier holds any value, like the Scheme vector it now is.
    if (tensor->dtype == ESHKOL_TENSOR_DTYPE_BOXED) return true;
    if (tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL) {
        double ignored = 0.0;
        return is_forward_jet(value) || tagged_real_to_double(value, &ignored);
    }
    double ignored = 0.0;
    if (tagged_real_to_double(value, &ignored)) return true;
    return tensor->dtype == ESHKOL_TENSOR_DTYPE_F64 && is_ad_node(value, type);
}

// Promote a numeric carrier to the general boxed representation, in place.
//
// ADR-0020. A numeric `#(...)` literal materialises as a tensor natively while
// the bytecode VM materialises it as a Scheme vector, and R7RS vectors are
// heterogeneous. Rather than refuse a string where the VM accepts one, the
// vector API widens the carrier: a fresh tagged-value buffer replaces the f64
// buffer, every existing element is carried over as the number it was, and the
// dtype records the new representation. The descriptor itself keeps its
// address, so every alias of the vector sees the promotion.
//
// The promoted carrier is no longer a numeric tensor: `tensor?` answers #f and
// every tensor kernel refuses it through the operand check, which is why the
// tensor API (`tensor-set!`) refuses the store instead of promoting.
// Re-point a numeric tensor's elements at a tagged-value buffer of `dtype`,
// carrying every existing element over as the number it was. A slot holding a
// reverse-mode node pointer (the f64 carrier encoding) has no tagged-number
// reading, so such a tensor is not widened and the store is refused loudly.
bool promote_numeric_tensor_to_tagged(eshkol_tensor_t* tensor, uint64_t dtype) {
    const uint64_t n = tensor->total_elements;
    arena_t* arena = get_global_arena();
    const auto* bits = reinterpret_cast<const uint64_t*>(tensor->elements);
    for (uint64_t i = 0; i < n; ++i) {
        if (eshkol_ad_node_probe(arena, bits[i], -1)) return false;
    }
    auto* slots = static_cast<eshkol_tagged_value_t*>(
        arena_allocate(arena, (size_t)n * sizeof(eshkol_tagged_value_t)));
    if (!slots && n != 0) return false;
    const auto* numeric = reinterpret_cast<const double*>(tensor->elements);
    for (uint64_t i = 0; i < n; ++i) slots[i] = tagged_double(numeric[i]);
    tensor->elements = reinterpret_cast<int64_t*>(slots);
    tensor->dtype = dtype;
    return true;
}

bool promote_tensor_to_boxed(eshkol_tensor_t* tensor) {
    if (tensor->dtype == ESHKOL_TENSOR_DTYPE_BOXED) return true;
    if (tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL) return false;
    return promote_numeric_tensor_to_tagged(tensor, ESHKOL_TENSOR_DTYPE_BOXED);
}

// Widen a tensor so it can hold `value`, or report that it cannot.
//
// ADR-0020 amendment 2. A forward-mode derivative carrier stored into a
// numeric tensor -- by `(tensor ...)` construction, a collection coerced to a
// tensor operand, or a mutator -- makes it a jet tensor (dtype DUAL): the same
// representation the forward-mode tensor kernels already produce and consume,
// so the derivative survives instead of being refused or read as its primal.
// This is independent of `allow_boxed`: a jet tensor is still a numeric
// tensor, so the tensor API widens to it too. Only the vector API may widen to
// the general boxed carrier for a value that is not a number at all.
bool widen_tensor_for(eshkol_tensor_t* tensor, const eshkol_tagged_value_t& value,
                      bool allow_boxed) {
    if (tensor_slot_accepts(tensor, value)) return true;
    if (is_forward_jet(value)) {
        if (eshkol_tensor_dtype_is_tagged(tensor->dtype)) return false;
        return promote_numeric_tensor_to_tagged(tensor, ESHKOL_TENSOR_DTYPE_DUAL) &&
               tensor_slot_accepts(tensor, value);
    }
    return allow_boxed && promote_tensor_to_boxed(tensor) &&
           tensor_slot_accepts(tensor, value);
}

void encode_tensor_slot(eshkol_tensor_t* tensor, int64_t index,
                        const eshkol_tagged_value_t& value) {
    if (tensor->dtype == ESHKOL_TENSOR_DTYPE_BOXED) {
        auto* slots = reinterpret_cast<eshkol_tagged_value_t*>(tensor->elements);
        store_tagged_value(tensor, slots + index, value);
        return;
    }
    if (tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL) {
        auto* slots = reinterpret_cast<eshkol_tagged_value_t*>(tensor->elements);
        double numeric = 0.0;
        const eshkol_tagged_value_t encoded =
            is_forward_jet(value)
                ? value
                : (tagged_real_to_double(value, &numeric), tagged_double(numeric));
        store_tagged_value(tensor, slots + index, encoded);
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
        store_tagged_values(dst, dst_values, src_values, static_cast<size_t>(count));
        return ESHKOL_SLOT_STORE_OK;
    }

    if (dst_subtype == HEAP_SUBTYPE_TENSOR && src_subtype == HEAP_SUBTYPE_TENSOR) {
        auto* dst_tensor = reinterpret_cast<eshkol_tensor_t*>(dst);
        const auto* src_tensor = reinterpret_cast<const eshkol_tensor_t*>(src);
        // A boxed source carries values a numeric destination cannot hold, so
        // the destination is promoted first (ADR-0020).
        if (src_tensor->dtype == ESHKOL_TENSOR_DTYPE_BOXED &&
            dst_tensor->dtype != ESHKOL_TENSOR_DTYPE_BOXED &&
            !promote_tensor_to_boxed(dst_tensor)) {
            return ESHKOL_SLOT_STORE_VALUE;
        }
        const bool dst_tagged = eshkol_tensor_dtype_is_tagged(dst_tensor->dtype);
        const bool src_tagged = eshkol_tensor_dtype_is_tagged(src_tensor->dtype);
        if (dst_tagged != src_tagged) {
            // A numeric source into a promoted destination: carry each element
            // over as the number it is.
            if (dst_tagged) {
                auto* dst_values = reinterpret_cast<eshkol_tagged_value_t*>(dst_tensor->elements);
                const auto* src_values = reinterpret_cast<const double*>(src_tensor->elements);
                // Plain doubles carry no pointer: nothing to promote.
                for (int64_t i = 0; i < count; ++i)
                    dst_values[at + i] = tagged_double(src_values[start + i]);
                return ESHKOL_SLOT_STORE_OK;
            }
            return ESHKOL_SLOT_STORE_VALUE;
        }
        const bool dst_dual = dst_tagged;
        if (dst_dual) {
            auto* dst_values = reinterpret_cast<eshkol_tagged_value_t*>(dst_tensor->elements) + at;
            const auto* src_values =
                reinterpret_cast<const eshkol_tagged_value_t*>(src_tensor->elements) + start;
            store_tagged_values(dst, dst_values, src_values, static_cast<size_t>(count));
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
        if (eshkol_tensor_dtype_is_tagged(src_tensor->dtype)) {
            const auto* src_values =
                reinterpret_cast<const eshkol_tagged_value_t*>(src_tensor->elements) + start;
            store_tagged_values(dst, dst_values, src_values, static_cast<size_t>(count));
        } else {
            // Plain doubles carry no pointer: nothing to promote.
            const auto* src_values = reinterpret_cast<const double*>(src_tensor->elements);
            for (int64_t i = 0; i < count; ++i) {
                dst_values[i] = tagged_double(src_values[start + i]);
            }
        }
        return ESHKOL_SLOT_STORE_OK;
    }

    auto* dst_tensor = reinterpret_cast<eshkol_tensor_t*>(dst);
    const auto* src_values = vector_elements(src) + start;
    if (dst_tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL) {
        auto* dst_values = reinterpret_cast<eshkol_tagged_value_t*>(dst_tensor->elements) + at;
        store_tagged_values(dst, dst_values, src_values, static_cast<size_t>(count));
        return ESHKOL_SLOT_STORE_OK;
    }
    // Validate the complete source range before mutating the destination so a
    // refused value cannot leave a partially copied tensor behind; a value the
    // numeric carrier cannot hold promotes it once (ADR-0020).
    for (int64_t i = 0; i < count; ++i) {
        if (!widen_tensor_for(dst_tensor, src_values[i], /*allow_boxed=*/true)) {
            return ESHKOL_SLOT_STORE_VALUE;
        }
    }
    for (int64_t i = 0; i < count; ++i) {
        if (!tensor_slot_accepts(dst_tensor, src_values[i])) {
            return ESHKOL_SLOT_STORE_VALUE;
        }
    }
    // Promote the whole source range before the first slot is written, so a
    // failed promotion leaves the destination exactly as it was.
    const eshkol_tagged_value_t* values =
        promote_staged(dst_tensor, src_values, static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        encode_tensor_slot(dst_tensor, at + i, values[i]);
    }
    return ESHKOL_SLOT_STORE_OK;
}

// promote == false is the tensor API: a tensor stays a numeric carrier, so a
// value with no numeric representation is refused rather than widening it.
static int32_t tensor_slot_store(void* tensor_object, int64_t index,
                                 const eshkol_tagged_value_t* value, bool promote) {
    if (!tensor_object || !value) return ESHKOL_SLOT_STORE_NULL;
    if (subtype_of(tensor_object) != HEAP_SUBTYPE_TENSOR) {
        return ESHKOL_SLOT_STORE_CONTAINER;
    }
    auto* tensor = reinterpret_cast<eshkol_tensor_t*>(tensor_object);
    const int64_t length = sequence_length(tensor_object, HEAP_SUBTYPE_TENSOR);
    if (index < 0 || index >= length) return ESHKOL_SLOT_STORE_BOUNDS;
    if (!widen_tensor_for(tensor, *value, promote)) return ESHKOL_SLOT_STORE_VALUE;
    encode_tensor_slot(tensor, index, *value);
    return ESHKOL_SLOT_STORE_OK;
}

extern "C" int32_t eshkol_tensor_slot_store(void* tensor_object, int64_t index,
                                             const eshkol_tagged_value_t* value) {
    return tensor_slot_store(tensor_object, index, value, /*promote=*/false);
}

// The vector API's tensor path: a value the numeric carrier cannot hold
// promotes it (ADR-0020) instead of being refused.
extern "C" int32_t eshkol_vector_slot_store(void* tensor_object, int64_t index,
                                             const eshkol_tagged_value_t* value) {
    return tensor_slot_store(tensor_object, index, value, /*promote=*/true);
}

extern "C" int32_t eshkol_sequence_slot_store(const eshkol_tagged_value_t* sequence,
                                               int64_t index,
                                               const eshkol_tagged_value_t* value) {
    if (!value) return ESHKOL_SLOT_STORE_NULL;
    uint8_t subtype = 0xFF;
    void* object = sequence_object(sequence, &subtype);
    if (!object) return ESHKOL_SLOT_STORE_CONTAINER;
    if (subtype == HEAP_SUBTYPE_TENSOR) {
        return tensor_slot_store(object, index, value, /*promote=*/true);
    }
    const int64_t length = sequence_length(object, subtype);
    if (index < 0 || index >= length) return ESHKOL_SLOT_STORE_BOUNDS;
    store_tagged_value(object, vector_elements(object) + index, *value);
    return ESHKOL_SLOT_STORE_OK;
}

// Construction: store values[k] into slot indices[k] of a tensor object for
// k in [0, n), in order, through the one encoder. Each value may widen the
// tensor (a forward-mode carrier makes it a jet tensor, carrying the numbers
// already stored over); a value that is not a number is refused. Used by the
// `(tensor ...)` literal lowering for the elements whose type is decided at
// run time; the rest it stores inline as doubles before this call.
extern "C" int32_t eshkol_tensor_store_indexed(void* tensor_object,
                                               const int64_t* indices,
                                               const eshkol_tagged_value_t* values,
                                               int64_t n) {
    if (!tensor_object || ((!values || !indices) && n > 0)) return ESHKOL_SLOT_STORE_NULL;
    if (subtype_of(tensor_object) != HEAP_SUBTYPE_TENSOR) return ESHKOL_SLOT_STORE_CONTAINER;
    auto* tensor = reinterpret_cast<eshkol_tensor_t*>(tensor_object);
    const int64_t length = sequence_length(tensor_object, HEAP_SUBTYPE_TENSOR);
    for (int64_t k = 0; k < n; ++k) {
        if (indices[k] < 0 || indices[k] >= length) return ESHKOL_SLOT_STORE_BOUNDS;
        if (!widen_tensor_for(tensor, values[k], /*allow_boxed=*/false)) {
            return ESHKOL_SLOT_STORE_VALUE;
        }
        encode_tensor_slot(tensor, indices[k], values[k]);
    }
    return ESHKOL_SLOT_STORE_OK;
}

// Can this value be an element of a numeric tensor? A real number of any
// exactness or a forward-mode derivative carrier. The flat-collection operand
// coercion asks this before building a tensor, so it and the constructors
// share one answer.
extern "C" int32_t eshkol_tensor_leaf_is_storable(const eshkol_tagged_value_t* value) {
    if (!value) return 0;
    double ignored = 0.0;
    return (tagged_real_to_double(*value, &ignored) || is_forward_jet(*value)) ? 1 : 0;
}

extern "C" int32_t eshkol_tensor_fill_slots(void* tensor_object,
                                             const eshkol_tagged_value_t* value) {
    if (!tensor_object || !value) return ESHKOL_SLOT_STORE_NULL;
    if (subtype_of(tensor_object) != HEAP_SUBTYPE_TENSOR) return ESHKOL_SLOT_STORE_CONTAINER;
    auto* tensor = reinterpret_cast<eshkol_tensor_t*>(tensor_object);
    const int64_t length = sequence_length(tensor_object, HEAP_SUBTYPE_TENSOR);
    if (length < 0) return ESHKOL_SLOT_STORE_BOUNDS;
    if (!widen_tensor_for(tensor, *value, /*allow_boxed=*/false)) return ESHKOL_SLOT_STORE_VALUE;
    for (int64_t i = 0; i < length; ++i) encode_tensor_slot(tensor, i, *value);
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
        if (!widen_tensor_for(tensor, *value, /*allow_boxed=*/true)) {
            return ESHKOL_SLOT_STORE_VALUE;
        }
        for (int64_t i = 0; i < length; ++i) encode_tensor_slot(tensor, i, *value);
        return ESHKOL_SLOT_STORE_OK;
    }
    // One value fills every slot: promote it once, then store it everywhere.
    eshkol_tagged_value_t staged;
    eshkol_region_write_barrier_into(&staged, object, value);
    auto* slots = vector_elements(object);
    for (int64_t i = 0; i < length; ++i) slots[i] = staged;
    return ESHKOL_SLOT_STORE_OK;
}
