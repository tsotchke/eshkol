/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Tensor dtype runtime helpers (ESH-0020).
 *
 * Tensor element storage is always f64 bit patterns (int64) for ABI
 * compatibility with the rest of the runtime and codegen. The dtype field in
 * the tensor header records the *logical* element precision. Casting to a
 * lower-precision dtype reduces each element through that precision (so an f16
 * cast loses mantissa bits exactly as the GPU path will), while keeping the
 * storage as f64. This lets matmul/tensor-* dispatch on dtype and lets host
 * code interrogate precision without changing the 40-byte tensor layout.
 *
 * These helpers are called from LLVM-generated code through extern "C" names.
 * The JIT resolves them via in-process dlsym, and AOT links them from the
 * aggregate runtime archive, so no explicit symbol registration is required.
 */

#include "arena_memory.h"

#include <cmath>
#include <cstdint>
#include <cstring>

// Forward declaration of the C ABI intern helper (symbol_intern.cpp). Returns
// the interned symbol object pointer, which is exactly what a HEAP_PTR-tagged
// symbol value carries (see wrap_interned in introspection.cpp).
extern "C" void* eshkol_intern_symbol_lookup(const char* name);

namespace {

// ---- precision reduction primitives -------------------------------------

// IEEE-754 half (binary16) round-trip through f64. Round-to-nearest-even.
double f64_to_f16_to_f64(double x) {
    float f = static_cast<float>(x);
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;  // unbiased
    uint32_t mant = bits & 0x7FFFFFu;

    uint16_t h;
    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        // Inf / NaN
        h = static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    } else if (exp >= 16) {
        // Overflow to inf
        h = static_cast<uint16_t>(sign | 0x7C00u);
    } else if (exp >= -14) {
        // Normalised half: biased exponent in [1,30]
        uint32_t hexp = static_cast<uint32_t>(exp + 15);
        uint32_t hmant = mant >> 13;
        uint32_t rem = mant & 0x1FFFu;
        h = static_cast<uint16_t>(sign | (hexp << 10) | hmant);
        // round-to-nearest-even on the dropped 13 bits
        if (rem > 0x1000u || (rem == 0x1000u && (hmant & 1u))) {
            h++;  // carry naturally propagates into exponent
        }
    } else if (exp >= -24) {
        // Subnormal half
        mant |= 0x800000u;  // restore implicit leading 1
        int32_t shift = -exp - 1;  // shift in [10,23]
        uint32_t hmant = mant >> (shift + 1);
        uint32_t rem = mant & ((1u << (shift + 1)) - 1u);
        uint32_t halfway = 1u << shift;
        h = static_cast<uint16_t>(sign | hmant);
        if (rem > halfway || (rem == halfway && (hmant & 1u))) {
            h++;
        }
    } else {
        // Underflow to signed zero
        h = static_cast<uint16_t>(sign);
    }

    // Decode the half back to f64.
    uint32_t hsign = (h & 0x8000u) >> 15;
    uint32_t hexp = (h & 0x7C00u) >> 10;
    uint32_t hmant = h & 0x03FFu;
    double sgn = hsign ? -1.0 : 1.0;
    if (hexp == 0) {
        if (hmant == 0) return sgn * 0.0;
        return sgn * std::ldexp(static_cast<double>(hmant), -24);  // subnormal
    }
    if (hexp == 0x1F) {
        return hmant ? (x != x ? x : std::nan("")) : sgn * HUGE_VAL;
    }
    double m = 1.0 + static_cast<double>(hmant) / 1024.0;
    return sgn * std::ldexp(m, static_cast<int>(hexp) - 15);
}

// bfloat16 round-trip through f64. Round-to-nearest-even on the low 16 bits of
// the f32 representation (bf16 shares f32's 8-bit exponent).
double f64_to_bf16_to_f64(double x) {
    float f = static_cast<float>(x);
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        return static_cast<double>(f);  // preserve Inf/NaN
    }
    uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;
    bits &= 0xFFFF0000u;
    std::memcpy(&f, &bits, sizeof(f));
    return static_cast<double>(f);
}

/**
 * @brief Reduces a double through the numeric precision implied by a tensor dtype code.
 *
 * Storage always stays f64 (per this file's dtype convention); this applies
 * the precision loss a given logical dtype would incur: F32 round-trips
 * through `float`; F16/BF16 round-trip through their respective binary
 * formats (round-to-nearest-even) via f64_to_f16_to_f64() /
 * f64_to_bf16_to_f64(); I8 rounds to the nearest integer, clamps to
 * [-128, 127], and maps NaN to 0; F64 (and any unrecognized code) passes the
 * value through unchanged.
 *
 * @param v     Value to reduce.
 * @param dtype Target dtype code (eshkol_tensor_dtype_t).
 * @return      `v` reduced through the target dtype's precision, still stored as double.
 */
double reduce_precision(double v, uint64_t dtype) {
    switch (dtype) {
        case ESHKOL_TENSOR_DTYPE_F32:
            return static_cast<double>(static_cast<float>(v));
        case ESHKOL_TENSOR_DTYPE_F16:
            return f64_to_f16_to_f64(v);
        case ESHKOL_TENSOR_DTYPE_BF16:
            return f64_to_bf16_to_f64(v);
        case ESHKOL_TENSOR_DTYPE_I8: {
            if (!(v == v)) return 0.0;  // NaN -> 0
            double r = std::nearbyint(v);
            if (r < -128.0) r = -128.0;
            if (r > 127.0) r = 127.0;
            return static_cast<double>(static_cast<int8_t>(r));
        }
        case ESHKOL_TENSOR_DTYPE_F64:
        default:
            return v;
    }
}

/**
 * @brief Maps a tensor dtype code to its canonical lowercase name string.
 *
 * @param dtype Dtype code (eshkol_tensor_dtype_t).
 * @return      Static string naming the dtype ("f32", "f16", "bf16", "i8", or
 *              "f64" for ESHKOL_TENSOR_DTYPE_F64 and any unrecognized code).
 */
const char* dtype_name(uint64_t dtype) {
    switch (dtype) {
        case ESHKOL_TENSOR_DTYPE_F32:  return "f32";
        case ESHKOL_TENSOR_DTYPE_F16:  return "f16";
        case ESHKOL_TENSOR_DTYPE_BF16: return "bf16";
        case ESHKOL_TENSOR_DTYPE_I8:   return "i8";
        case ESHKOL_TENSOR_DTYPE_F64:
        default:                       return "f64";
    }
}

}  // namespace

extern "C" {

// arena_allocate_tensor_full lives in runtime_tensor_alloc.cpp.
eshkol_tensor_t* arena_allocate_tensor_full(arena_t* arena, uint64_t num_dims,
                                            uint64_t total_elements);

double eshkol_tensor_reduce_precision_value(double value, int64_t dtype) {
    return reduce_precision(value, static_cast<uint64_t>(dtype));
}

// Returns the dtype code (eshkol_tensor_dtype_t) of a tensor.
int64_t eshkol_tensor_dtype_code(void* tensor_ptr) {
    if (!tensor_ptr) return ESHKOL_TENSOR_DTYPE_F64;
    return static_cast<int64_t>(reinterpret_cast<eshkol_tensor_t*>(tensor_ptr)->dtype);
}

// Returns the interned symbol object pointer naming a tensor's dtype, suitable
// for packing directly as a HEAP_PTR symbol tagged value.
void* eshkol_tensor_dtype_symbol(void* tensor_ptr) {
    uint64_t dt = ESHKOL_TENSOR_DTYPE_F64;
    if (tensor_ptr) dt = reinterpret_cast<eshkol_tensor_t*>(tensor_ptr)->dtype;
    return eshkol_intern_symbol_lookup(dtype_name(dt));
}

// Apply a dtype to an existing tensor in place: reduce every element through
// the target precision and record the dtype. Returns the same pointer so the
// caller can chain. Used by (make-tensor ... :dtype 'f16).
void* eshkol_tensor_apply_dtype(void* tensor_ptr, int64_t dtype) {
    if (!tensor_ptr) return tensor_ptr;
    eshkol_tensor_t* t = reinterpret_cast<eshkol_tensor_t*>(tensor_ptr);
    uint64_t dt = static_cast<uint64_t>(dtype);
    if (t->elements && dt != ESHKOL_TENSOR_DTYPE_F64) {
        double* data = reinterpret_cast<double*>(t->elements);
        for (uint64_t i = 0; i < t->total_elements; i++) {
            data[i] = reduce_precision(data[i], dt);
        }
    }
    t->dtype = dt;
    return tensor_ptr;
}

// Allocate a new tensor with the given dtype, copying dims and casting every
// element through the target precision. Returns the new tensor data pointer
// (HEAP_PTR payload), or null on failure. Used by (tensor-cast t 'f16).
void* eshkol_tensor_cast_alloc(void* arena, void* tensor_ptr, int64_t dtype) {
    if (!arena || !tensor_ptr) return nullptr;
    eshkol_tensor_t* src = reinterpret_cast<eshkol_tensor_t*>(tensor_ptr);
    eshkol_tensor_t* dst = arena_allocate_tensor_full(
        reinterpret_cast<arena_t*>(arena), src->num_dimensions, src->total_elements);
    if (!dst) return nullptr;

    for (uint64_t i = 0; i < src->num_dimensions; i++) {
        dst->dimensions[i] = src->dimensions[i];
    }
    uint64_t dt = static_cast<uint64_t>(dtype);
    const double* sdata = reinterpret_cast<const double*>(src->elements);
    double* ddata = reinterpret_cast<double*>(dst->elements);
    for (uint64_t i = 0; i < src->total_elements; i++) {
        ddata[i] = reduce_precision(sdata[i], dt);
    }
    dst->dtype = dt;
    return dst;
}

// Promote two dtype codes to the result dtype of a binary tensor op.
// Rule (precision-preserving): f64 dominates; then f32; mixed 16-bit
// (f16+bf16) widen to f32 to avoid silent cross-format loss; equal dtypes are
// preserved; i8 combined with any float yields that float. This lets matmul/
// elementwise ops keep the logical precision flowing toward the GPU dispatch.
static int64_t promote_dtype(uint64_t a, uint64_t b) {
    if (a == b) return (int64_t)a;
    if (a == ESHKOL_TENSOR_DTYPE_F64 || b == ESHKOL_TENSOR_DTYPE_F64)
        return ESHKOL_TENSOR_DTYPE_F64;
    if (a == ESHKOL_TENSOR_DTYPE_F32 || b == ESHKOL_TENSOR_DTYPE_F32)
        return ESHKOL_TENSOR_DTYPE_F32;
    // Remaining combinations are among {f16, bf16, i8}, all distinct here.
    // f16 vs bf16 -> f32; i8 vs (f16|bf16) -> the float dtype.
    if (a == ESHKOL_TENSOR_DTYPE_I8) return (int64_t)b;
    if (b == ESHKOL_TENSOR_DTYPE_I8) return (int64_t)a;
    return ESHKOL_TENSOR_DTYPE_F32;  // f16 vs bf16
}

// Set a result tensor's dtype from two input tensors (binary op). Null inputs
// are treated as f64. Returns the result pointer for chaining.
void* eshkol_tensor_result_dtype_binary(void* result, void* a, void* b) {
    if (!result) return result;
    uint64_t da = a ? reinterpret_cast<eshkol_tensor_t*>(a)->dtype : ESHKOL_TENSOR_DTYPE_F64;
    uint64_t db = b ? reinterpret_cast<eshkol_tensor_t*>(b)->dtype : ESHKOL_TENSOR_DTYPE_F64;
    reinterpret_cast<eshkol_tensor_t*>(result)->dtype =
        (uint64_t)promote_dtype(da, db);
    return result;
}

// Set a result tensor's dtype from a single input tensor (unary op / view).
void* eshkol_tensor_result_dtype_unary(void* result, void* a) {
    if (!result) return result;
    reinterpret_cast<eshkol_tensor_t*>(result)->dtype =
        a ? reinterpret_cast<eshkol_tensor_t*>(a)->dtype : ESHKOL_TENSOR_DTYPE_F64;
    return result;
}

// Map a dtype symbol name to its code (used by codegen when the dtype name is
// only known as a runtime string). Returns f64 for unknown names.
int64_t eshkol_tensor_dtype_from_name(const char* name) {
    if (!name) return ESHKOL_TENSOR_DTYPE_F64;
    if (std::strcmp(name, "f32") == 0)  return ESHKOL_TENSOR_DTYPE_F32;
    if (std::strcmp(name, "f16") == 0)  return ESHKOL_TENSOR_DTYPE_F16;
    if (std::strcmp(name, "bf16") == 0) return ESHKOL_TENSOR_DTYPE_BF16;
    if (std::strcmp(name, "i8") == 0)   return ESHKOL_TENSOR_DTYPE_I8;
    return ESHKOL_TENSOR_DTYPE_F64;
}

}  // extern "C"
