/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted runtime error helpers.
 *
 * This file owns the current stderr/logger/exit-backed error path. The symbol
 * names are still part of the generated-code runtime ABI, but the implementation
 * is hosted until the freestanding panic/error hook ABI is introduced.
 */

#include <eshkol/core/runtime.h>
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

/*
 * Current error source location (v1.3 source-span errors).
 *
 * Codegen emits a call to eshkol_set_error_location(file, line, col) on the
 * type-error branch immediately before raising, so the formatter can prefix
 * the message with "file:line:col:". Thread-local so concurrent workers don't
 * clobber one another. The file pointer is a static global string emitted by
 * codegen (stable for the program's lifetime); we hold the pointer, not a copy.
 *
 * Hot paths never touch this: it is only written on an error branch that is
 * about to abort, and only read inside the error formatter.
 */
#ifdef __clang__
#  define ESHKOL_TLS __thread
#elif defined(__GNUC__)
#  define ESHKOL_TLS __thread
#elif defined(_MSC_VER)
#  define ESHKOL_TLS __declspec(thread)
#else
#  define ESHKOL_TLS
#endif

static ESHKOL_TLS const char* g_error_loc_file = nullptr;
static ESHKOL_TLS uint32_t g_error_loc_line = 0;
static ESHKOL_TLS uint32_t g_error_loc_column = 0;

extern "C" {

void eshkol_set_error_location(const char* file, uint32_t line, uint32_t column) {
    g_error_loc_file = file;
    g_error_loc_line = line;
    g_error_loc_column = column;
}

void eshkol_clear_error_location(void) {
    g_error_loc_file = nullptr;
    g_error_loc_line = 0;
    g_error_loc_column = 0;
}

/* Render the current "file:line:col: " prefix into `buf`. Returns the number
 * of bytes written (0 if no location is set). The trailing space is included
 * so callers can concatenate the message directly. */
static size_t eshkol_format_error_location_prefix(char* buf, size_t buflen) {
    if (g_error_loc_line == 0 || buflen == 0) {
        return 0;
    }
    int n;
    const char* file = g_error_loc_file ? g_error_loc_file : "<unknown>";
    if (g_error_loc_column > 0) {
        n = std::snprintf(buf, buflen, "%s:%u:%u: ",
                          file, g_error_loc_line, g_error_loc_column);
    } else {
        n = std::snprintf(buf, buflen, "%s:%u: ", file, g_error_loc_line);
    }
    if (n < 0) return 0;
    return (size_t)n < buflen ? (size_t)n : buflen - 1;
}

void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    std::fprintf(stderr, "%s\n", buf);

    eshkol_exception_t* exc = eshkol_make_exception(type, buf);
    if (exc) {
        eshkol_raise(exc);
        // If eshkol_raise returns, do not let callers continue past a fatal
        // runtime condition.
    }
    std::exit(1);
}

void eshkol_type_error(const char* proc_name, const char* expected_type) {
    eshkol_error("Type error in %s: expected %s",
                 proc_name ? proc_name : "<unknown>",
                 expected_type ? expected_type : "<type>");

    eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
                         "Type error in %s: expected %s",
                         proc_name ? proc_name : "<unknown>",
                         expected_type ? expected_type : "<type>");
}

void eshkol_type_error_with_value(const char* proc_name, const char* expected_type,
                                  const char* actual_type) {
    char prefix[320];
    eshkol_format_error_location_prefix(prefix, sizeof(prefix));

    eshkol_error("%sType error in %s: expected %s, got %s",
                 prefix,
                 proc_name ? proc_name : "<unknown>",
                 expected_type ? expected_type : "<type>",
                 actual_type ? actual_type : "<unknown>");

    eshkol_runtime_fatal(ESHKOL_EXCEPTION_TYPE_ERROR,
                         "%sType error in %s: expected %s, got %s",
                         prefix,
                         proc_name ? proc_name : "<unknown>",
                         expected_type ? expected_type : "<type>",
                         actual_type ? actual_type : "<unknown>");
}

/* Map a tagged value's runtime type to a human-readable type name. */
const char* eshkol_format_value_type_tag(eshkol_tagged_value_t v) {
    uint8_t base_type = (uint8_t)(v.type & 0x0F);
    switch (base_type) {
        case ESHKOL_VALUE_NULL:        return "null";
        case ESHKOL_VALUE_INT64:       return "integer";
        case ESHKOL_VALUE_DOUBLE:      return "double";
        case ESHKOL_VALUE_BOOL:        return "boolean";
        case ESHKOL_VALUE_CHAR:        return "character";
        case ESHKOL_VALUE_SYMBOL:      return "symbol";
        case ESHKOL_VALUE_DUAL_NUMBER: return "dual-number";
        case ESHKOL_VALUE_COMPLEX:     return "complex";
        case ESHKOL_VALUE_LOGIC_VAR:   return "logic-var";
        case ESHKOL_VALUE_CALLABLE: {
            void* p = (void*)v.data.ptr_val;
            if (!p) return "procedure";
            uint8_t sub = *((uint8_t*)p - 8);
            switch (sub) {
                case 2: return "ad-node";
                case 3: return "continuation";
                default: return "procedure";
            }
        }
        case ESHKOL_VALUE_HEAP_PTR: {
            void* p = (void*)v.data.ptr_val;
            if (!p) return "null";
            uint8_t sub = *((uint8_t*)p - 8);
            switch (sub) {
                case HEAP_SUBTYPE_CONS:           return "pair";
                case HEAP_SUBTYPE_STRING:         return "string";
                case HEAP_SUBTYPE_VECTOR:         return "vector";
                case HEAP_SUBTYPE_TENSOR:         return "tensor";
                case HEAP_SUBTYPE_MULTI_VALUE:    return "values";
                case HEAP_SUBTYPE_HASH:           return "hash-table";
                case HEAP_SUBTYPE_EXCEPTION:      return "exception";
                case HEAP_SUBTYPE_RECORD:         return "record";
                case HEAP_SUBTYPE_BYTEVECTOR:     return "bytevector";
                case HEAP_SUBTYPE_PORT:           return "port";
                case HEAP_SUBTYPE_SYMBOL:         return "symbol";
                case HEAP_SUBTYPE_BIGNUM:         return "bignum";
                case HEAP_SUBTYPE_SUBSTITUTION:   return "substitution";
                case HEAP_SUBTYPE_FACT:           return "fact";
                case HEAP_SUBTYPE_KNOWLEDGE_BASE: return "knowledge-base";
                case HEAP_SUBTYPE_FACTOR_GRAPH:   return "factor-graph";
                case HEAP_SUBTYPE_WORKSPACE:      return "workspace";
                case HEAP_SUBTYPE_PROMISE:        return "promise";
                case HEAP_SUBTYPE_RATIONAL:       return "rational";
                case HEAP_SUBTYPE_PRNG:           return "prng";
                default:                          return "heap-object";
            }
        }
        case ESHKOL_VALUE_CONS_PTR:    return "pair";
        case ESHKOL_VALUE_STRING_PTR:  return "string";
        case ESHKOL_VALUE_VECTOR_PTR:  return "vector";
        case ESHKOL_VALUE_TENSOR_PTR:  return "tensor";
        case ESHKOL_VALUE_HASH_PTR:    return "hash-table";
        case ESHKOL_VALUE_EXCEPTION:   return "exception";
        default:                       return "unknown-type";
    }
}

/* Raise a type error reporting the operand's actual runtime type
 * alongside the expected one. Generated codegen sites that have the
 * operand in scope should call this so users see "got <actual-type>"
 * instead of guessing which operand was wrong.
 */
void eshkol_type_error_with_operand(const char* proc_name,
                                    const char* expected_type,
                                    const eshkol_tagged_value_t* actual) {
    eshkol_tagged_value_t val;
    if (actual) {
        val = *actual;
    } else {
        val.type = ESHKOL_VALUE_NULL;
        val.flags = 0;
        val.reserved = 0;
        val.data.int_val = 0;
    }
    eshkol_type_error_with_value(proc_name, expected_type,
                                 eshkol_format_value_type_tag(val));
}

}  // extern "C"
