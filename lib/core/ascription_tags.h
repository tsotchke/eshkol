#ifndef ESHKOL_ASCRIPTION_TAGS_H
#define ESHKOL_ASCRIPTION_TAGS_H

#include "value_type_names.h"
#include <stdint.h>
#include <stddef.h>

typedef uint64_t EshkolTagSet;

static inline EshkolTagSet eshkol_tag_bit(eshkol_type_name_id_t tag) {
    return UINT64_C(1) << (unsigned)tag;
}

static inline int eshkol_type_spelling_equal(const char* a, const char* b) {
    while (*a && *b) {
        char x = *a++, y = *b++;
        if (x >= 'A' && x <= 'Z') x += 'a' - 'A';
        if (x != y) return 0;
    }
    return *a == *b;
}

// Runtime tags shared by native values and bytecode values.
static inline EshkolTagSet eshkol_ground_type_tags(const char* name) {
    if (!name) return 0;
    const EshkolTagSet integer = eshkol_tag_bit(ESHKOL_TYPE_NAME_INTEGER) |
                                 eshkol_tag_bit(ESHKOL_TYPE_NAME_I128);
    const EshkolTagSet real = integer | eshkol_tag_bit(ESHKOL_TYPE_NAME_REAL) |
                              eshkol_tag_bit(ESHKOL_TYPE_NAME_RATIONAL);
    if (eshkol_type_spelling_equal(name, "number"))
        return real | eshkol_tag_bit(ESHKOL_TYPE_NAME_COMPLEX) |
               eshkol_tag_bit(ESHKOL_TYPE_NAME_DUAL_NUMBER) | eshkol_tag_bit(ESHKOL_TYPE_NAME_AD_NODE);
    if (eshkol_type_spelling_equal(name, "any") || eshkol_type_spelling_equal(name, "value"))
        return (UINT64_C(1) << ESHKOL_TYPE_NAME_COUNT) - 1;
    if (eshkol_type_spelling_equal(name, "real")) return real;
    if (eshkol_type_spelling_equal(name, "vector"))
        return eshkol_tag_bit(ESHKOL_TYPE_NAME_VECTOR) | eshkol_tag_bit(ESHKOL_TYPE_NAME_TENSOR);
    if (eshkol_type_spelling_equal(name, "procedure") || eshkol_type_spelling_equal(name, "closure"))
        return eshkol_tag_bit(ESHKOL_TYPE_NAME_PROCEDURE) | eshkol_tag_bit(ESHKOL_TYPE_NAME_CONTINUATION);
    if (eshkol_type_spelling_equal(name, "list"))
        return eshkol_tag_bit(ESHKOL_TYPE_NAME_NULL_LIST) | eshkol_tag_bit(ESHKOL_TYPE_NAME_PAIR);
    static const char* const integers[] = {"integer","int","i8","int8","i16","int16",
        "i32","int32","i64","int64","isize","u8","uint8","u16","uint16","u32","uint32",
        "u64","uint64","usize","nat","natural","bigint"};
    for (size_t i = 0; i < sizeof(integers)/sizeof(*integers); ++i)
        if (eshkol_type_spelling_equal(name, integers[i])) return integer;
    static const struct { const char* name; eshkol_type_name_id_t tag; } aliases[] = {
        {"str",ESHKOL_TYPE_NAME_STRING}, {"bool",ESHKOL_TYPE_NAME_BOOLEAN},
        {"nil",ESHKOL_TYPE_NAME_NULL_LIST}, {"character",ESHKOL_TYPE_NAME_CHAR},
        {"float",ESHKOL_TYPE_NAME_REAL}, {"double",ESHKOL_TYPE_NAME_REAL},
        {"float32",ESHKOL_TYPE_NAME_REAL}, {"float64",ESHKOL_TYPE_NAME_REAL},
        {"hashtable",ESHKOL_TYPE_NAME_HASH_TABLE}, {"dual",ESHKOL_TYPE_NAME_DUAL_NUMBER},
        {"adnode",ESHKOL_TYPE_NAME_AD_NODE}
    };
    for (size_t i = 0; i < sizeof(aliases)/sizeof(*aliases); ++i)
        if (eshkol_type_spelling_equal(name, aliases[i].name)) return eshkol_tag_bit(aliases[i].tag);
    for (unsigned i = 0; i < ESHKOL_TYPE_NAME_COUNT; ++i)
        if (eshkol_type_spelling_equal(name, eshkol_type_name_spelling((eshkol_type_name_id_t)i)))
            return UINT64_C(1) << i;
    return 0;
}

#endif
