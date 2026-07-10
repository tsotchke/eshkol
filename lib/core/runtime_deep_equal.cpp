/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Deep structural equality runtime helpers.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/core/bignum.h"
#include "../../inc/eshkol/core/rational.h"

#include <stdint.h>
#include <string.h>

// Runtime helper for deep structural equality of tagged values.
// Takes pointers to avoid struct-by-value ABI issues.
bool eshkol_deep_equal(const eshkol_tagged_value_t* val1,
                       const eshkol_tagged_value_t* val2) {
    if (!val1 || !val2) {
        return val1 == val2;  // Both null -> equal, one null -> not equal.
    }

    auto get_base_type = [](uint8_t t) -> uint8_t {
        if (t >= 8) return t;  // Legacy, consolidated, or multimedia types.
        return t & 0x0F;       // Immediate types: strip exactness flags.
    };
    uint8_t type1 = get_base_type(val1->type);
    uint8_t type2 = get_base_type(val2->type);

    auto is_cons = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_CONS_PTR) return true;
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_CONS;
        }
        return false;
    };

    auto is_empty_list = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_NULL) return true;
        if (type == ESHKOL_VALUE_CONS_PTR && val->data.ptr_val == 0) return true;
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val == 0) return true;
        return false;
    };

    auto is_string = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_STRING_PTR) return true;
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_STRING;
        }
        return false;
    };

    auto is_symbol = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_SYMBOL;
        }
        return false;
    };

    bool empty1 = is_empty_list(type1, val1);
    bool empty2 = is_empty_list(type2, val2);
    if (empty1 && empty2) return true;
    if (empty1 || empty2) return false;

    bool is_cons1 = is_cons(type1, val1);
    bool is_cons2 = is_cons(type2, val2);
    if (is_cons1 && is_cons2) {
        arena_tagged_cons_cell_t* cell1 = (arena_tagged_cons_cell_t*)val1->data.ptr_val;
        arena_tagged_cons_cell_t* cell2 = (arena_tagged_cons_cell_t*)val2->data.ptr_val;

        if (!cell1 || !cell2) return cell1 == cell2;

        eshkol_tagged_value_t car1 = arena_tagged_cons_get_tagged_value(cell1, false);
        eshkol_tagged_value_t car2 = arena_tagged_cons_get_tagged_value(cell2, false);
        if (!eshkol_deep_equal(&car1, &car2)) return false;

        eshkol_tagged_value_t cdr1 = arena_tagged_cons_get_tagged_value(cell1, true);
        eshkol_tagged_value_t cdr2 = arena_tagged_cons_get_tagged_value(cell2, true);
        return eshkol_deep_equal(&cdr1, &cdr2);
    }

    bool is_str1 = is_string(type1, val1);
    bool is_str2 = is_string(type2, val2);
    if (is_str1 && is_str2) {
        if (val1->data.ptr_val == val2->data.ptr_val) return true;
        if (!val1->data.ptr_val || !val2->data.ptr_val) return false;
        return strcmp((const char*)val1->data.ptr_val, (const char*)val2->data.ptr_val) == 0;
    }

    bool is_sym1 = is_symbol(type1, val1);
    bool is_sym2 = is_symbol(type2, val2);
    if (is_sym1 && is_sym2) {
        if (val1->data.ptr_val == val2->data.ptr_val) return true;
        if (!val1->data.ptr_val || !val2->data.ptr_val) return false;
        return strcmp((const char*)val1->data.ptr_val, (const char*)val2->data.ptr_val) == 0;
    }

    if ((type1 == ESHKOL_VALUE_INT64 && type2 == ESHKOL_VALUE_DOUBLE) ||
        (type1 == ESHKOL_VALUE_DOUBLE && type2 == ESHKOL_VALUE_INT64)) {
        double d1 = (type1 == ESHKOL_VALUE_DOUBLE) ? val1->data.double_val : (double)val1->data.int_val;
        double d2 = (type2 == ESHKOL_VALUE_DOUBLE) ? val2->data.double_val : (double)val2->data.int_val;
        return d1 == d2;
    }

    auto is_bignum = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_BIGNUM;
        }
        return false;
    };
    bool is_bn1 = is_bignum(type1, val1);
    bool is_bn2 = is_bignum(type2, val2);
    if (is_bn1 && is_bn2) {
        return eshkol_bignum_compare((const eshkol_bignum_t*)val1->data.ptr_val,
                                     (const eshkol_bignum_t*)val2->data.ptr_val) == 0;
    }
    if (is_bn1 && type2 == ESHKOL_VALUE_INT64) {
        return eshkol_bignum_compare_int64((const eshkol_bignum_t*)val1->data.ptr_val,
                                           val2->data.int_val) == 0;
    }
    if (type1 == ESHKOL_VALUE_INT64 && is_bn2) {
        return eshkol_bignum_compare_int64((const eshkol_bignum_t*)val2->data.ptr_val,
                                           val1->data.int_val) == 0;
    }

    auto is_tensor = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_TENSOR;
        }
        return false;
    };
    bool is_t1 = is_tensor(type1, val1);
    bool is_t2 = is_tensor(type2, val2);
    if (is_t1 && is_t2) {
        if (val1->data.ptr_val == val2->data.ptr_val) return true;
        eshkol_tensor_t* t1 = (eshkol_tensor_t*)val1->data.ptr_val;
        eshkol_tensor_t* t2 = (eshkol_tensor_t*)val2->data.ptr_val;
        if (!t1 || !t2) return t1 == t2;
        if (t1->num_dimensions != t2->num_dimensions) return false;
        if (t1->total_elements != t2->total_elements) return false;
        for (uint64_t i = 0; i < t1->num_dimensions; i++) {
            if (t1->dimensions[i] != t2->dimensions[i]) return false;
        }
        for (uint64_t i = 0; i < t1->total_elements; i++) {
            union { int64_t i; double d; } u1, u2;
            u1.i = t1->elements[i];
            u2.i = t2->elements[i];
            if (u1.d != u2.d) return false;
        }
        return true;
    }

    auto is_vector = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_VECTOR;
        }
        return false;
    };
    bool is_v1 = is_vector(type1, val1);
    bool is_v2 = is_vector(type2, val2);
    if (is_v1 && is_v2) {
        if (val1->data.ptr_val == val2->data.ptr_val) return true;
        int64_t len1 = *(int64_t*)(uintptr_t)val1->data.ptr_val;
        int64_t len2 = *(int64_t*)(uintptr_t)val2->data.ptr_val;
        if (len1 != len2) return false;
        eshkol_tagged_value_t* elems1 =
            (eshkol_tagged_value_t*)((uint8_t*)(uintptr_t)val1->data.ptr_val + 8);
        eshkol_tagged_value_t* elems2 =
            (eshkol_tagged_value_t*)((uint8_t*)(uintptr_t)val2->data.ptr_val + 8);
        for (int64_t i = 0; i < len1; i++) {
            if (!eshkol_deep_equal(&elems1[i], &elems2[i])) return false;
        }
        return true;
    }

    // Exact rationals (ESH-0114): rationals are always stored in reduced form
    // with a positive denominator, so value equality reduces to comparing the
    // numerator/denominator fields. (equal? (/ 1 3) (/ 2 6)) must be #t.
    auto is_rational = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_HEAP_PTR && val->data.ptr_val) {
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER((void*)val->data.ptr_val);
            return hdr->subtype == HEAP_SUBTYPE_RATIONAL;
        }
        return false;
    };
    bool is_rat1 = is_rational(type1, val1);
    bool is_rat2 = is_rational(type2, val2);
    if (is_rat1 && is_rat2) {
        // Canonical value equality across both the int64 and bignum
        // representations (ESH-0105/ESH-0114).
        return eshkol_rational_equal((void*)val1->data.ptr_val,
                                     (void*)val2->data.ptr_val) != 0;
    }

    // Complex numbers (ESH-0114): compare real and imaginary components by
    // value. Complex tagged values are inexact (double components).
    if (type1 == ESHKOL_VALUE_COMPLEX && type2 == ESHKOL_VALUE_COMPLEX) {
        if (val1->data.ptr_val == val2->data.ptr_val) return true;
        const eshkol_complex_number_t* c1 =
            (const eshkol_complex_number_t*)val1->data.ptr_val;
        const eshkol_complex_number_t* c2 =
            (const eshkol_complex_number_t*)val2->data.ptr_val;
        if (!c1 || !c2) return c1 == c2;
        return c1->real == c2->real && c1->imag == c2->imag;
    }

    if (type1 != type2) return false;

    switch (type1) {
        case ESHKOL_VALUE_INT64:
        case ESHKOL_VALUE_BOOL:
            return val1->data.int_val == val2->data.int_val;

        case ESHKOL_VALUE_DOUBLE:
            return val1->data.double_val == val2->data.double_val;

        case ESHKOL_VALUE_STRING_PTR:
            if (val1->data.ptr_val == val2->data.ptr_val) return true;
            if (!val1->data.ptr_val || !val2->data.ptr_val) return false;
            return strcmp((const char*)val1->data.ptr_val, (const char*)val2->data.ptr_val) == 0;

        case ESHKOL_VALUE_CLOSURE_PTR:
        case ESHKOL_VALUE_LAMBDA_SEXPR:
            return val1->data.ptr_val == val2->data.ptr_val;

        default:
            return val1->data.int_val == val2->data.int_val;
    }
}
