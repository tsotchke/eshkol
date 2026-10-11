/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * type-name -- classify a runtime value by the shared type-name vocabulary
 * (value_type_names.h) and answer with an interned symbol.
 *
 * This lives in a runtime-eligible translation unit, not in
 * lib/core/introspection.cpp: `type-name` is a native-codegen builtin that
 * generated programs call directly, and introspection.cpp is compiled only
 * into the compiler/tool aggregate, never into libeshkol-runtime.a (the same
 * reason runtime_gensym.cpp and symbol_intern.cpp are separate TUs).
 *
 * Declared in inc/eshkol/core/introspection.h (eshkol_type_of); the codegen
 * entry point eshkol_type_name_into is declared next to it.
 */

#include <eshkol/eshkol.h>
#include <eshkol/core/introspection.h>
#include <eshkol/logger.h>
#include <eshkol/exhaustive_dispatch.h>

#include "value_type_names.h"

#include <cstdint>

extern "C" void* eshkol_intern_symbol_lookup(const char* name);

/* The interned symbol spelling @p id, in the consolidated HEAP_PTR encoding.
 * Interning (rather than tagging the static spelling) gives the symbol a real
 * object header, so (eq? (type-name x) 'string) holds. */
static eshkol_tagged_value_t interned_type_name(eshkol_type_name_id_t id) {
    eshkol_tagged_value_t tv;
    tv.type = ESHKOL_VALUE_HEAP_PTR;
    tv.flags = 0;
    tv.reserved = 0;
    tv.data.ptr_val = reinterpret_cast<uint64_t>(
        eshkol_intern_symbol_lookup(eshkol_type_name_spelling(id)));
    if (!tv.data.ptr_val) {
        tv.type = ESHKOL_VALUE_BOOL;   // interning failed: #f, as eshkol_procedure_name does
        tv.data.int_val = 0;
    }
    return tv;
}

/**
 * @brief Classify a value as one of the shared runtime type names
 *        (value_type_names.h).
 *
 * Dispatches on the tagged value's immediate type, and for HEAP_PTR /
 * CALLABLE values further dispatches on the object header's subtype to
 * distinguish pairs, strings, vectors, tensors, hash tables, bignums,
 * rationals, closures, primitives, continuations, etc. Legacy pointer
 * type tags are recognized for backward compatibility.
 */
static eshkol_type_name_id_t eshkol_value_type_name_id(eshkol_tagged_value_t value) {
    eshkol_type_name_id_t type_name = ESHKOL_TYPE_NAME_COUNT;  // "not yet classified"

    // A port carries its direction/binary bits in the type byte next to
    // HEAP_PTR (eshkol.h, ESHKOL_PORT_*_FLAG).
    if ((value.type & 0x0F) == ESHKOL_VALUE_HEAP_PTR &&
        (value.type & ESHKOL_PORT_ANY_FLAG) != 0) {
        return ESHKOL_TYPE_NAME_PORT;
    }

    // Check immediate types first
    switch (value.type) {
        case ESHKOL_VALUE_NULL:
            type_name = ESHKOL_TYPE_NAME_NULL_LIST;
            break;
        case ESHKOL_VALUE_UNSPECIFIED:
            type_name = ESHKOL_TYPE_NAME_UNSPECIFIED;
            break;
        case ESHKOL_VALUE_INT64:
            type_name = ESHKOL_TYPE_NAME_INTEGER;
            break;
        case ESHKOL_VALUE_DOUBLE:
            type_name = ESHKOL_TYPE_NAME_REAL;
            break;
        case ESHKOL_VALUE_BOOL:
            type_name = ESHKOL_TYPE_NAME_BOOLEAN;
            break;
        case ESHKOL_VALUE_CHAR:
            type_name = ESHKOL_TYPE_NAME_CHAR;
            break;
        case ESHKOL_VALUE_SYMBOL:
            type_name = ESHKOL_TYPE_NAME_SYMBOL;
            break;
        case ESHKOL_VALUE_DUAL_NUMBER:
            type_name = ESHKOL_TYPE_NAME_DUAL_NUMBER;
            break;
        case ESHKOL_VALUE_COMPLEX:
            type_name = ESHKOL_TYPE_NAME_COMPLEX;
            break;
        case ESHKOL_VALUE_LOGIC_VAR:
            type_name = ESHKOL_TYPE_NAME_LOGIC_VAR;
            break;
        case ESHKOL_VALUE_HANDLE:
            type_name = ESHKOL_TYPE_NAME_HANDLE;
            break;
        case ESHKOL_VALUE_BUFFER:
            type_name = ESHKOL_TYPE_NAME_BUFFER;
            break;
        case ESHKOL_VALUE_STREAM:
            type_name = ESHKOL_TYPE_NAME_STREAM;
            break;
        case ESHKOL_VALUE_EVENT:
            type_name = ESHKOL_TYPE_NAME_EVENT;
            break;
        case ESHKOL_VALUE_HEAP_PTR: {
            // Need to check subtype from header
            void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(value.data.ptr_val));
            if (ptr) {
                eshkol_object_header_t* header = ESHKOL_GET_HEADER(ptr);
                if (header && !eshkol_heap_subtype_is_declared(header->subtype)) {
                    // Open-set half: the subtype byte is not a declared member
                    // at all. Loud, and named, because it means the header is
                    // not one of ours.
                    eshkol_warn("undeclared heap subtype: %d", header->subtype);
                    type_name = ESHKOL_TYPE_NAME_HEAP_OBJECT;
                } else if (header) {
                    // Closed-set half: exhaustive, no default. `type-of` is the
                    // program's answer to "what is this"; a subtype absorbed by
                    // a default answers "heap-object", which is uninformative
                    // in exactly the case where the information mattered.
                    ESHKOL_EXHAUSTIVE_SWITCH_BEGIN
                    switch ((heap_subtype_t)header->subtype) {
                        case HEAP_SUBTYPE_CONS:
                            type_name = ESHKOL_TYPE_NAME_PAIR;
                            break;
                        case HEAP_SUBTYPE_STRING:
                            type_name = ESHKOL_TYPE_NAME_STRING;
                            break;
                        case HEAP_SUBTYPE_SYMBOL:
                            type_name = ESHKOL_TYPE_NAME_SYMBOL;
                            break;
                        case HEAP_SUBTYPE_VECTOR:
                            type_name = ESHKOL_TYPE_NAME_VECTOR;
                            break;
                        case HEAP_SUBTYPE_TENSOR:
                            type_name = ESHKOL_TYPE_NAME_TENSOR;
                            break;
                        case HEAP_SUBTYPE_HASH:
                            type_name = ESHKOL_TYPE_NAME_HASH_TABLE;
                            break;
                        case HEAP_SUBTYPE_EXCEPTION:
                            type_name = ESHKOL_TYPE_NAME_EXCEPTION;
                            break;
                        case HEAP_SUBTYPE_PORT:
                            type_name = ESHKOL_TYPE_NAME_PORT;
                            break;
                        case HEAP_SUBTYPE_BIGNUM:
                            type_name = ESHKOL_TYPE_NAME_INTEGER;
                            break;
                        case HEAP_SUBTYPE_BYTEVECTOR:
                            type_name = ESHKOL_TYPE_NAME_BYTEVECTOR;
                            break;
                        case HEAP_SUBTYPE_RATIONAL:
                            type_name = ESHKOL_TYPE_NAME_RATIONAL;
                            break;
                        case HEAP_SUBTYPE_SUBSTITUTION:
                            type_name = ESHKOL_TYPE_NAME_SUBSTITUTION;
                            break;
                        case HEAP_SUBTYPE_FACT:
                            type_name = ESHKOL_TYPE_NAME_FACT;
                            break;
                        case HEAP_SUBTYPE_KNOWLEDGE_BASE:
                            type_name = ESHKOL_TYPE_NAME_KNOWLEDGE_BASE;
                            break;
                        case HEAP_SUBTYPE_FACTOR_GRAPH:
                            type_name = ESHKOL_TYPE_NAME_FACTOR_GRAPH;
                            break;
                        case HEAP_SUBTYPE_WORKSPACE:
                            type_name = ESHKOL_TYPE_NAME_WORKSPACE;
                            break;
                        case HEAP_SUBTYPE_PRNG:
                            type_name = ESHKOL_TYPE_NAME_PRNG;
                            break;
                        case HEAP_SUBTYPE_PARAMETER:
                            type_name = ESHKOL_TYPE_NAME_PARAMETER;
                            break;
                        case HEAP_SUBTYPE_MULTI_VALUE:
                            type_name = ESHKOL_TYPE_NAME_VALUES;
                            break;
                        case HEAP_SUBTYPE_RECORD:
                            // Records allocate through the vector allocator, so
                            // a live record normally arrives stamped VECTOR and
                            // answers "vector" above. Named so a future record
                            // allocator stamping subtype 7 does not silently
                            // start answering "heap-object".
                            type_name = ESHKOL_TYPE_NAME_RECORD;
                            break;
                        case HEAP_SUBTYPE_PROMISE:
                            type_name = ESHKOL_TYPE_NAME_PROMISE;
                            break;
                        case HEAP_SUBTYPE_DNC:
                            type_name = ESHKOL_TYPE_NAME_DNC;
                            break;
                        case HEAP_SUBTYPE_SDNC:
                            type_name = ESHKOL_TYPE_NAME_SDNC;
                            break;
                        case HEAP_SUBTYPE_TAYLOR:
                            type_name = ESHKOL_TYPE_NAME_TAYLOR;
                            break;
                        case HEAP_SUBTYPE_I128:
                            type_name = ESHKOL_TYPE_NAME_I128;
                            break;
                    }
                    ESHKOL_EXHAUSTIVE_SWITCH_END
                }
            }
            if (type_name == ESHKOL_TYPE_NAME_COUNT) type_name = ESHKOL_TYPE_NAME_HEAP_OBJECT;
            break;
        }
        case ESHKOL_VALUE_CALLABLE: {
            // Need to check subtype from header
            void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(value.data.ptr_val));
            if (ptr) {
                eshkol_object_header_t* header = ESHKOL_GET_HEADER(ptr);
                if (header && !eshkol_callable_subtype_is_declared(header->subtype)) {
                    eshkol_warn("undeclared callable subtype: %d", header->subtype);
                    type_name = ESHKOL_TYPE_NAME_PROCEDURE;
                } else if (header) {
                    ESHKOL_EXHAUSTIVE_SWITCH_BEGIN
                    switch ((callable_subtype_t)header->subtype) {
                        case CALLABLE_SUBTYPE_CLOSURE:
                            type_name = ESHKOL_TYPE_NAME_PROCEDURE;
                            break;
                        case CALLABLE_SUBTYPE_LAMBDA_SEXPR:
                            type_name = ESHKOL_TYPE_NAME_PROCEDURE;
                            break;
                        case CALLABLE_SUBTYPE_AD_NODE:
                            type_name = ESHKOL_TYPE_NAME_AD_NODE;
                            break;
                        case CALLABLE_SUBTYPE_PRIMITIVE:
                            type_name = ESHKOL_TYPE_NAME_PROCEDURE;
                            break;
                        case CALLABLE_SUBTYPE_CONTINUATION:
                            type_name = ESHKOL_TYPE_NAME_CONTINUATION;
                            break;
                    }
                    ESHKOL_EXHAUSTIVE_SWITCH_END
                }
            }
            if (type_name == ESHKOL_TYPE_NAME_COUNT) type_name = ESHKOL_TYPE_NAME_PROCEDURE;
            break;
        }
        // Handle legacy types for backward compatibility
        case ESHKOL_VALUE_CONS_PTR:
            type_name = ESHKOL_TYPE_NAME_PAIR;
            break;
        case ESHKOL_VALUE_STRING_PTR:
            type_name = ESHKOL_TYPE_NAME_STRING;
            break;
        case ESHKOL_VALUE_VECTOR_PTR:
            type_name = ESHKOL_TYPE_NAME_VECTOR;
            break;
        case ESHKOL_VALUE_TENSOR_PTR:
            type_name = ESHKOL_TYPE_NAME_TENSOR;
            break;
        case ESHKOL_VALUE_CLOSURE_PTR:
            type_name = ESHKOL_TYPE_NAME_PROCEDURE;
            break;
        default:
            type_name = ESHKOL_TYPE_NAME_UNKNOWN;
            break;
    }

    // Return as interned symbol in the consolidated HEAP_PTR encoding. The raw
    // type_name pointer here is a static string literal — tagging it as a
    // symbol directly would leave ESHKOL_GET_HEADER reading program data, so
    // we route through the symbol-intern path which allocates a proper symbol
    // header and ensures (eq? (type-of x) 'string) works.
    return type_name;
}

/**
 * @brief Get the type of a value as an interned symbol.
 *
 * The spelling comes from the shared vocabulary in value_type_names.h
 * (e.g. 'integer, 'real, 'string, 'pair, 'tensor, 'closure, 'unknown).
 *
 * @param value Value to inspect.
 * @return Interned type-name symbol.
 */
eshkol_tagged_value_t eshkol_type_of(eshkol_tagged_value_t value) {
    // Return as interned symbol in the consolidated HEAP_PTR encoding. The raw
    // spelling is a static string literal — tagging it as a symbol directly
    // would leave ESHKOL_GET_HEADER reading program data, so we route through
    // the symbol-intern path which allocates a proper symbol header and
    // ensures (eq? (type-name x) 'string) works.
    return interned_type_name(eshkol_value_type_name_id(value));
}

/**
 * @brief `type-name` entry point for generated code: writes the type-name
 *        symbol of @p value to @p out. Pointer arguments keep the tagged
 *        value off the by-value struct ABI.
 */
void eshkol_type_name_into(const eshkol_tagged_value_t* value,
                                      eshkol_tagged_value_t* out) {
    if (!out) return;
    if (!value) {
        *out = interned_type_name(ESHKOL_TYPE_NAME_UNKNOWN);
        return;
    }
    *out = eshkol_type_of(*value);
}
