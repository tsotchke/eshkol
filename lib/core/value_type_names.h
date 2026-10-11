/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * value_type_names.h -- the one vocabulary of runtime type names.
 *
 * `type-name`, the REPL's machine-mode `value_type` field and the native
 * runtime's eshkol_type_of() all answer "what kind of value is this?" with a
 * symbol from this list. The native runtime and the bytecode VM represent
 * values differently (eshkol_tagged_value_t + object-header subtype versus the
 * VM's ValType), so each engine maps its own representation onto an
 * eshkol_type_name_id_t; the spellings themselves exist only here, so the two
 * engines cannot drift apart on what a value is called.
 *
 * Procedures are named 'procedure whatever their representation (a compiled
 * closure, a lambda that keeps its S-expression, a builtin): which of those a
 * given procedure is depends on the engine and on how it was compiled, not on
 * the program, so it is not part of the value's type.
 *
 * Dependency-free C so the freestanding VM (including its WASM build) can
 * include it alongside the hosted C++ runtime.
 */
#ifndef ESHKOL_VALUE_TYPE_NAMES_H
#define ESHKOL_VALUE_TYPE_NAMES_H

#ifdef __cplusplus
extern "C" {
#endif

/* X(identifier, spelling) */
#define ESHKOL_TYPE_NAME_LIST(X)                  \
    X(NULL_LIST,       "null")                    \
    X(UNSPECIFIED,     "unspecified")             \
    X(INTEGER,         "integer")                 \
    X(RATIONAL,        "rational")                \
    X(REAL,            "real")                    \
    X(COMPLEX,         "complex")                 \
    X(I128,            "i128")                    \
    X(DUAL_NUMBER,     "dual-number")             \
    X(BOOLEAN,         "boolean")                 \
    X(CHAR,            "char")                    \
    X(SYMBOL,          "symbol")                  \
    X(STRING,          "string")                  \
    X(PAIR,            "pair")                    \
    X(VECTOR,          "vector")                  \
    X(TENSOR,          "tensor")                  \
    X(BYTEVECTOR,      "bytevector")              \
    X(HASH_TABLE,      "hash-table")              \
    X(RECORD,          "record")                  \
    X(VALUES,          "values")                  \
    X(EXCEPTION,       "exception")               \
    X(PORT,            "port")                    \
    X(EOF_OBJECT,      "eof-object")              \
    X(PROMISE,         "promise")                 \
    X(PARAMETER,       "parameter")               \
    X(PRNG,            "prng")                    \
    X(PROCEDURE,       "procedure")               \
    X(CONTINUATION,    "continuation")            \
    X(AD_NODE,         "ad-node")                 \
    X(AD_TAPE,         "ad-tape")                 \
    X(TAYLOR,          "taylor")                  \
    X(LOGIC_VAR,       "logic-var")               \
    X(SUBSTITUTION,    "substitution")            \
    X(FACT,            "fact")                    \
    X(KNOWLEDGE_BASE,  "knowledge-base")          \
    X(FACTOR_GRAPH,    "factor-graph")            \
    X(WORKSPACE,       "workspace")               \
    X(DNC,             "dnc")                     \
    X(SDNC,            "sdnc")                    \
    X(MANIFOLD,        "manifold")                \
    X(FUTURE,          "future")                  \
    X(HANDLE,          "handle")                  \
    X(BUFFER,          "buffer")                  \
    X(STREAM,          "stream")                  \
    X(EVENT,           "event")                   \
    X(HEAP_OBJECT,     "heap-object")             \
    X(UNKNOWN,         "unknown")

typedef enum {
#define ESHKOL_TYPE_NAME_ENUM_(id, spelling) ESHKOL_TYPE_NAME_##id,
    ESHKOL_TYPE_NAME_LIST(ESHKOL_TYPE_NAME_ENUM_)
#undef ESHKOL_TYPE_NAME_ENUM_
    ESHKOL_TYPE_NAME_COUNT
} eshkol_type_name_id_t;

/** Spelling of @p id; "unknown" for an id outside the list. */
static inline const char* eshkol_type_name_spelling(eshkol_type_name_id_t id) {
    static const char* const spellings[] = {
#define ESHKOL_TYPE_NAME_STR_(id, spelling) spelling,
        ESHKOL_TYPE_NAME_LIST(ESHKOL_TYPE_NAME_STR_)
#undef ESHKOL_TYPE_NAME_STR_
    };
    if ((unsigned)id >= (unsigned)ESHKOL_TYPE_NAME_COUNT) return "unknown";
    return spellings[id];
}

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_VALUE_TYPE_NAMES_H */
