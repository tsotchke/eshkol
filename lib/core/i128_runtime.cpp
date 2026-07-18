/*
 * Native 128-bit Integer (i128) — arena/tagged runtime for the native, JIT,
 * and AOT backends.
 *
 * This file is the boxing + error-signalling layer for the native-code
 * substrate. All actual arithmetic is delegated to the pure, allocation-free
 * core in <eshkol/core/i128.h>, which the bytecode VM shares verbatim so the
 * two paths compute bit-identical results.
 *
 * i128 is a DISTINCT fixed-width, two's-complement type that lives OFF the
 * exact numeric tower: arithmetic WRAPS modulo 2^128 (never promotes to
 * bignum), and every crossing to/from the tower is an explicit conversion.
 *
 * Representation:  [eshkol_object_header_t][eshkol_i128_abi {lo,hi}]
 *   subtype = HEAP_SUBTYPE_I128 (25); tagged value type = ESHKOL_VALUE_HEAP_PTR.
 * The 16-byte payload layout deliberately matches the planned esk_i128_abi.
 *
 * Every function here takes/returns tagged values through pointers and
 * allocates results on the arena, mirroring eshkol_bignum_binary_tagged() so
 * the codegen can drive them all through one generic dispatch helper.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/eshkol.h"
#include "eshkol/core/i128.h"
#include "eshkol/core/i128_runtime.h"
#include <cstring>
#include <cstdio>
#include <cstdint>

/* Arena + exception primitives (defined elsewhere in the runtime archive). */
extern "C" {
    void* arena_allocate_with_header(arena_t* arena, size_t data_size,
                                     uint8_t subtype, uint8_t flags);
    char* arena_allocate_string_with_header(arena_t* arena, size_t length);
}

namespace {

/* ---- tagged-value helpers ----
 * The exact/inexact flag is sometimes folded into the type byte (the combined
 * ESHKOL_VALUE_EXACT_INT64 == INT64|EXACT_FLAG tag), depending on the codegen
 * path, so type-byte comparisons mask those two flag bits before matching. */

constexpr uint8_t TYPE_FLAG_MASK =
    (uint8_t)~(ESHKOL_VALUE_EXACT_FLAG | ESHKOL_VALUE_INEXACT_FLAG);

inline uint8_t base_type(const eshkol_tagged_value_t* v) {
    return (uint8_t)(v->type & TYPE_FLAG_MASK);
}

inline bool tagged_is_int64(const eshkol_tagged_value_t* v) {
    return v && base_type(v) == ESHKOL_VALUE_INT64;
}

inline bool tagged_is_heap_ptr(const eshkol_tagged_value_t* v) {
    return v && v->data.ptr_val != 0 && base_type(v) == ESHKOL_VALUE_HEAP_PTR;
}

inline bool tagged_is_i128(const eshkol_tagged_value_t* v) {
    if (!tagged_is_heap_ptr(v)) return false;
    const eshkol_object_header_t* h = ESHKOL_GET_HEADER((void*)v->data.ptr_val);
    return h->subtype == HEAP_SUBTYPE_I128;
}

inline __int128 tagged_unbox_i128(const eshkol_tagged_value_t* v) {
    const eshkol_i128_abi* pl = (const eshkol_i128_abi*)(void*)v->data.ptr_val;
    return eshkol_i128_from_abi(*pl);
}

inline eshkol_tagged_value_t box_i128(arena_t* arena, __int128 value) {
    eshkol_tagged_value_t out;
    std::memset(&out, 0, sizeof(out));
    void* p = arena_allocate_with_header(arena, sizeof(eshkol_i128_abi),
                                         HEAP_SUBTYPE_I128, 0);
    if (!p) { out.type = ESHKOL_VALUE_NULL; return out; }
    *(eshkol_i128_abi*)p = eshkol_i128_to_abi(value);
    out.type = ESHKOL_VALUE_HEAP_PTR;
    out.data.ptr_val = (uint64_t)p;
    return out;
}

inline eshkol_tagged_value_t make_bool_tv(bool b) {
    eshkol_tagged_value_t out;
    std::memset(&out, 0, sizeof(out));
    out.type = ESHKOL_VALUE_BOOL;
    out.data.int_val = b ? 1 : 0;
    return out;
}

inline eshkol_tagged_value_t make_int_tv(int64_t v) {
    eshkol_tagged_value_t out;
    std::memset(&out, 0, sizeof(out));
    out.type = ESHKOL_VALUE_INT64;
    out.flags = ESHKOL_VALUE_EXACT_FLAG;
    out.data.int_val = v;
    return out;
}

[[noreturn]] inline void raise(eshkol_exception_type_t type, const char* msg) {
    eshkol_raise(eshkol_make_exception(type, msg));
    __builtin_unreachable();
}

/* Coerce an argument to a native i128. i128 boxes pass through; exact fixnums
 * widen (an explicit conversion is still required at the surface, but the
 * arithmetic builtins accept a fixnum so idioms like (i128-add x 1) are not a
 * trap — the fixnum is the value 1, not tower-promoted). Anything else is a
 * type error. */
inline __int128 coerce_i128(const eshkol_tagged_value_t* v, const char* who) {
    if (tagged_is_i128(v)) return tagged_unbox_i128(v);
    if (tagged_is_int64(v)) return (__int128)v->data.int_val;
    raise(ESHKOL_EXCEPTION_TYPE_ERROR, who);
}

} // namespace

extern "C" {

/* (i128 x) / (int->i128 x): construct from an exact fixnum. */
void eshkol_i128_from_int_tagged(arena_t* arena,
                                 const eshkol_tagged_value_t* x,
                                 eshkol_tagged_value_t* out) {
    if (tagged_is_i128(x)) { *out = *x; return; }
    if (!tagged_is_int64(x))
        raise(ESHKOL_EXCEPTION_TYPE_ERROR, "i128: argument must be an exact integer");
    *out = box_i128(arena, (__int128)x->data.int_val);
}

/* (string->i128 s): parse the full signed 128-bit range, including -2^127. */
void eshkol_i128_from_string_tagged(arena_t* arena,
                                    const eshkol_tagged_value_t* s,
                                    eshkol_tagged_value_t* out) {
    if (!tagged_is_heap_ptr(s))
        raise(ESHKOL_EXCEPTION_TYPE_ERROR, "string->i128: argument must be a string");
    const eshkol_object_header_t* h = ESHKOL_GET_HEADER((void*)s->data.ptr_val);
    if (h->subtype != HEAP_SUBTYPE_STRING)
        raise(ESHKOL_EXCEPTION_TYPE_ERROR, "string->i128: argument must be a string");
    const char* data = (const char*)(void*)s->data.ptr_val;
    size_t len = (size_t)h->size;
    /* Strings are stored with a trailing NUL (size excludes it); trim any
     * accidental NUL inside the counted span so strlen-style callers agree. */
    while (len > 0 && data[len - 1] == '\0') len--;
    __int128 value;
    if (!eshkol_i128_parse(data, len, &value))
        raise(ESHKOL_EXCEPTION_READ_ERROR,
              "string->i128: not a valid 128-bit integer literal");
    *out = box_i128(arena, value);
}

/* (i128? x) */
void eshkol_i128_predicate_tagged(const eshkol_tagged_value_t* x,
                                  eshkol_tagged_value_t* out) {
    *out = make_bool_tv(tagged_is_i128(x));
}

/* Binary arithmetic. op: 0=add 1=sub 2=mul 3=quotient 4=remainder. */
void eshkol_i128_binary_tagged(arena_t* arena,
                               const eshkol_tagged_value_t* a,
                               const eshkol_tagged_value_t* b,
                               int32_t op,
                               eshkol_tagged_value_t* out) {
    __int128 x = coerce_i128(a, "i128 arithmetic: argument must be an i128");
    __int128 y = coerce_i128(b, "i128 arithmetic: argument must be an i128");
    __int128 r;
    switch (op) {
        case 0: r = eshkol_i128_add(x, y); break;
        case 1: r = eshkol_i128_sub(x, y); break;
        case 2: r = eshkol_i128_mul(x, y); break;
        case 3:
            if (y == 0) raise(ESHKOL_EXCEPTION_DIVIDE_BY_ZERO, "i128-quotient: division by zero");
            r = eshkol_i128_quotient(x, y);
            break;
        case 4:
            if (y == 0) raise(ESHKOL_EXCEPTION_DIVIDE_BY_ZERO, "i128-remainder: division by zero");
            r = eshkol_i128_remainder(x, y);
            break;
        default:
            raise(ESHKOL_EXCEPTION_ERROR, "i128 arithmetic: bad op selector");
    }
    *out = box_i128(arena, r);
}

/* (i128-neg n) */
void eshkol_i128_neg_tagged(arena_t* arena,
                            const eshkol_tagged_value_t* a,
                            eshkol_tagged_value_t* out) {
    __int128 x = coerce_i128(a, "i128-neg: argument must be an i128");
    *out = box_i128(arena, eshkol_i128_neg(x));
}

/* Shifts. op: 0=shl 1=ashr 2=lshr. Count must be an exact fixnum in [0,127]. */
void eshkol_i128_shift_tagged(arena_t* arena,
                              const eshkol_tagged_value_t* a,
                              const eshkol_tagged_value_t* count,
                              int32_t op,
                              eshkol_tagged_value_t* out) {
    __int128 x = coerce_i128(a, "i128 shift: value must be an i128");
    if (!tagged_is_int64(count))
        raise(ESHKOL_EXCEPTION_TYPE_ERROR, "i128 shift: count must be an exact integer");
    int64_t c = count->data.int_val;
    if (c < 0 || c > 127)
        raise(ESHKOL_EXCEPTION_RANGE_ERROR, "i128 shift: count out of range [0,127]");
    __int128 r;
    switch (op) {
        case 0: r = eshkol_i128_shl(x, (unsigned)c); break;
        case 1: r = eshkol_i128_ashr(x, (unsigned)c); break;
        case 2: r = eshkol_i128_lshr(x, (unsigned)c); break;
        default: raise(ESHKOL_EXCEPTION_ERROR, "i128 shift: bad op selector");
    }
    *out = box_i128(arena, r);
}

/* Comparisons. op: 0='=' 1='<' 2='>' 3='<=' 4='>='. */
void eshkol_i128_compare_tagged(const eshkol_tagged_value_t* a,
                                const eshkol_tagged_value_t* b,
                                int32_t op,
                                eshkol_tagged_value_t* out) {
    __int128 x = coerce_i128(a, "i128 compare: argument must be an i128");
    __int128 y = coerce_i128(b, "i128 compare: argument must be an i128");
    int c = eshkol_i128_cmp(x, y);
    bool r;
    switch (op) {
        case 0: r = (c == 0); break;
        case 1: r = (c < 0);  break;
        case 2: r = (c > 0);  break;
        case 3: r = (c <= 0); break;
        case 4: r = (c >= 0); break;
        default: raise(ESHKOL_EXCEPTION_ERROR, "i128 compare: bad op selector");
    }
    *out = make_bool_tv(r);
}

/* (i128->string n) */
void eshkol_i128_to_string_tagged(arena_t* arena,
                                  const eshkol_tagged_value_t* a,
                                  eshkol_tagged_value_t* out) {
    __int128 x = coerce_i128(a, "i128->string: argument must be an i128");
    char buf[ESHKOL_I128_STR_MAX];
    size_t len = eshkol_i128_format(x, buf);
    std::memset(out, 0, sizeof(*out));
    char* dst = arena_allocate_string_with_header(arena, len);
    if (!dst) { out->type = ESHKOL_VALUE_NULL; return; }
    std::memcpy(dst, buf, len);
    dst[len] = '\0';
    out->type = ESHKOL_VALUE_HEAP_PTR;
    out->data.ptr_val = (uint64_t)dst;
}

/* (i128->int n): narrow to an exact fixnum, raising when out of int64 range. */
void eshkol_i128_to_int_tagged(const eshkol_tagged_value_t* a,
                               eshkol_tagged_value_t* out) {
    __int128 x = coerce_i128(a, "i128->int: argument must be an i128");
    if (x < (__int128)INT64_MIN || x > (__int128)INT64_MAX)
        raise(ESHKOL_EXCEPTION_RANGE_ERROR, "i128->int: value does not fit in a fixnum");
    *out = make_int_tv((int64_t)x);
}

/* Display/write support: format a boxed i128 payload straight to a FILE*. */
void eshkol_i128_display(const void* payload, void* stream) {
    char buf[ESHKOL_I128_STR_MAX];
    __int128 v = eshkol_i128_from_abi(*(const eshkol_i128_abi*)payload);
    eshkol_i128_format(v, buf);
    fputs(buf, (FILE*)stream);
}

} // extern "C"
