/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Direct S-expression to AST Conversion
 *
 * High-performance direct conversion from runtime S-expressions (cons cells)
 * to AST nodes, avoiding the serialize→parse overhead. This enables efficient
 * runtime eval/compile for metaprogramming and self-modifying code.
 */

#include <eshkol/core/sexp_to_ast.h>
#include "eshkol/core/reader_limits.h"
#include <eshkol/frontend/ast_strings.h>
#include <eshkol/core/introspection.h>
#include <eshkol/logger.h>
#include "arena_memory.h"
#include "datum_cycles.h"
#include "eval_syntax.h"
#include "../frontend/token_type.h"
#include <eshkol/frontend/syntax_color.h>
#include <eshkol/frontend/syntax_datum.h>
#include <eshkol/core/string_escape.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <new>
#include <set>
#include <string>

extern "C" int64_t eshkol_string_byte_length(const char* s);

namespace {

// ============================================================================
// Helper Functions
// ============================================================================

// Create a null value
inline eshkol_tagged_value_t make_null() {
    eshkol_tagged_value_t value;
    std::memset(&value, 0, sizeof(value));
    value.type = ESHKOL_VALUE_NULL;
    return value;
}

// Check if value is null
inline bool is_null(eshkol_tagged_value_t value) {
    return value.type == ESHKOL_VALUE_NULL;
}

// Check if value is a pair/cons cell
inline bool is_pair(eshkol_tagged_value_t value) {
    return ESHKOL_IS_CONS_COMPAT(value);
}

// Check if value is a symbol (legacy ESHKOL_VALUE_SYMBOL or consolidated
// HEAP_PTR with HEAP_SUBTYPE_SYMBOL header). Mirrors ESHKOL_IS_CONS_COMPAT.
inline bool is_symbol(eshkol_tagged_value_t value) {
    if (value.type == ESHKOL_VALUE_SYMBOL) return true;
    if (value.type == ESHKOL_VALUE_HEAP_PTR && value.data.ptr_val != 0) {
        eshkol_object_header_t* hdr =
            ESHKOL_GET_HEADER((void*)(uintptr_t)value.data.ptr_val);
        return hdr->subtype == HEAP_SUBTYPE_SYMBOL;
    }
    return false;
}

// Check if value is a string
inline bool is_string(eshkol_tagged_value_t value) {
    return ESHKOL_IS_STRING_COMPAT(value);
}

// Get raw pointer from tagged value
inline void* get_ptr(eshkol_tagged_value_t value) {
    return reinterpret_cast<void*>(value.data.ptr_val);
}

// Get CAR of a pair
inline eshkol_tagged_value_t pair_car(eshkol_tagged_value_t pair) {
    if (!is_pair(pair)) {
        return make_null();
    }
    arena_tagged_cons_cell_t* cell = static_cast<arena_tagged_cons_cell_t*>(get_ptr(pair));
    return cell ? cell->car : make_null();
}

// Get CDR of a pair
inline eshkol_tagged_value_t pair_cdr(eshkol_tagged_value_t pair) {
    if (!is_pair(pair)) {
        return make_null();
    }
    arena_tagged_cons_cell_t* cell = static_cast<arena_tagged_cons_cell_t*>(get_ptr(pair));
    return cell ? cell->cdr : make_null();
}

// Get symbol name from tagged value (returns pointer to interned string)
inline const char* get_symbol_name(eshkol_tagged_value_t value) {
    if (!is_symbol(value)) {
        return nullptr;
    }
    return static_cast<const char*>(get_ptr(value));
}

// Get string value from tagged value
inline const char* get_string_value(eshkol_tagged_value_t value) {
    if (!is_string(value)) {
        return nullptr;
    }
    return static_cast<const char*>(get_ptr(value));
}

// Count elements in a proper list
size_t list_length(eshkol_tagged_value_t list) {
    size_t count = 0;
    eshkol_tagged_value_t current = list;
    while (is_pair(current)) {
        count++;
        current = pair_cdr(current);
    }
    return count;
}

// Get nth element of a list (0-indexed)
eshkol_tagged_value_t list_ref(eshkol_tagged_value_t list, size_t index) {
    eshkol_tagged_value_t current = list;
    for (size_t i = 0; i < index && is_pair(current); i++) {
        current = pair_cdr(current);
    }
    return is_pair(current) ? pair_car(current) : make_null();
}

// Check if symbol matches a name
inline bool symbol_eq(eshkol_tagged_value_t value, const char* name) {
    const char* sym_name = get_symbol_name(value);
    return sym_name && strcmp(sym_name, name) == 0;
}

void set_error(eshkol::EvalConversionError& error, eshkol_exception_type_t type,
               const char* text) {
    error.type = type;
    snprintf(error.message, sizeof(error.message), "%s", text);
}

// Spell one atom as the token the reader would have produced for it.
bool spell_atom(eshkol_tagged_value_t value, eshkol::SyntaxToken& token,
                eshkol::EvalConversionError& error) {
    if (const char* name = get_symbol_name(value)) {
        const size_t length = strlen(name);
        if (eshkol_syntax_source_name_reserved(name, length)) {
            error.type = ESHKOL_EXCEPTION_SYNTAX_ERROR;
            snprintf(error.message, sizeof(error.message),
                     "the symbol |%s| is spelled the way the compiler spells the "
                     "bindings it generates, which no program text can name",
                     name);
            return false;
        }
        token.token_type = TOKEN_SYMBOL;
        token.text.assign(name, length);
        token.verbatim = token.text == ".";
        return true;
    }
    if (const char* bytes = get_string_value(value)) {
        token.token_type = TOKEN_STRING;
        token.text.assign(bytes, eshkol_string_byte_length(bytes));
        return true;
    }
    switch (value.type) {
        case ESHKOL_VALUE_INT64:
            token.token_type = TOKEN_NUMBER;
            token.text = std::to_string(value.data.int_val);
            return true;
        case ESHKOL_VALUE_DOUBLE: {
            char buffer[64];
            eshkol_format_double(buffer, sizeof(buffer), value.data.double_val);
            token.token_type = TOKEN_NUMBER;
            token.text = buffer;
            return true;
        }
        case ESHKOL_VALUE_BOOL:
            token.token_type = TOKEN_BOOLEAN;
            token.text = value.data.int_val ? "#t" : "#f";
            return true;
        case ESHKOL_VALUE_CHAR: {
            unsigned char bytes[4];
            size_t length;
            if (value.data.int_val < 0 || value.data.int_val > 0x10ffff ||
                !eshkol_string_escape_append_utf8(value.data.int_val, bytes, &length)) {
                set_error(error, ESHKOL_EXCEPTION_SYNTAX_ERROR,
                          "a character outside the Unicode scalar values is not "
                          "expression syntax");
                return false;
            }
            token.token_type = TOKEN_CHAR;
            token.text.assign(reinterpret_cast<const char*>(bytes), length);
            return true;
        }
        default:
            error.type = ESHKOL_EXCEPTION_SYNTAX_ERROR;
            snprintf(error.message, sizeof(error.message),
                     "a value of type %s is not expression syntax",
                     eshkol_format_value_type_tag(value));
            return false;
    }
}

// Spell a runtime datum as reader tokens, in reading order. Nesting is held on
// an explicit work stack on the heap, so a datum converts at any depth the
// reader could have produced it at; the only limit is the one input-size cap
// (inc/eshkol/core/reader_limits.h), which the spelling is weighed against as
// it grows. The datum is known to be acyclic (validate_eval_datum).
bool spell_runtime_datum(eshkol_tagged_value_t datum,
                         std::vector<eshkol::SyntaxToken>& tokens,
                         eshkol::EvalConversionError& error) {
    enum class Step : uint8_t { Value, Close, Dot };
    struct Work {
        Step step;
        eshkol_tagged_value_t value;
    };
    const size_t cap = eshkol_reader_max_source_bytes();
    size_t bytes = 0;
    std::vector<Work> pending;
    std::vector<eshkol_tagged_value_t> elements;
    auto emit = [&](int type, const char* text) {
        eshkol::SyntaxToken token;
        token.token_type = type;
        token.text = text;
        bytes += token.text.size() + 1;
        tokens.push_back(std::move(token));
    };
    pending.push_back({Step::Value, datum});
    while (!pending.empty()) {
        if (cap && bytes > cap) {
            (void)eshkol_reader_source_over_cap(bytes, error.message, sizeof(error.message));
            error.type = ESHKOL_EXCEPTION_ERROR;
            return false;
        }
        const Work work = pending.back();
        pending.pop_back();
        if (work.step == Step::Close) {
            emit(TOKEN_RPAREN, ")");
            continue;
        }
        if (work.step == Step::Dot) {
            emit(TOKEN_SYMBOL, ".");
            continue;
        }
        eshkol_tagged_value_t value = work.value;
        if (is_pair(value)) {
            const eshkol_tagged_value_t rest = pair_cdr(value);
            if (symbol_eq(pair_car(value), "quote") && is_pair(rest) &&
                is_null(pair_cdr(rest))) {
                emit(TOKEN_QUOTE, "'");
                pending.push_back({Step::Value, pair_car(rest)});
                continue;
            }
        }
        if (is_pair(value) || is_null(value)) {
            emit(TOKEN_LPAREN, "(");
            elements.clear();
            while (is_pair(value)) {
                elements.push_back(pair_car(value));
                value = pair_cdr(value);
            }
            pending.push_back({Step::Close, make_null()});
            if (!is_null(value)) {
                pending.push_back({Step::Value, value});
                pending.push_back({Step::Dot, make_null()});
            }
            for (auto it = elements.rbegin(); it != elements.rend(); ++it)
                pending.push_back({Step::Value, *it});
            continue;
        }
        eshkol::SyntaxToken token;
        if (!spell_atom(value, token, error)) return false;
        bytes += token.text.size() + 1;
        tokens.push_back(std::move(token));
    }
    if (cap && bytes > cap) {
        (void)eshkol_reader_source_over_cap(bytes, error.message, sizeof(error.message));
        error.type = ESHKOL_EXCEPTION_ERROR;
        return false;
    }
    return true;
}

eshkol_ast_t* convert_datum(eshkol_tagged_value_t datum,
                            const std::set<std::string>& macro_names,
                            eshkol::EvalConversionError& error) {
    // The empty list evaluates to itself here, as it always has for eval.
    if (is_null(datum)) {
        eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
        ast->type = ESHKOL_NULL;
        return ast;
    }
    try {
        std::vector<eshkol::SyntaxToken> tokens;
        if (!spell_runtime_datum(datum, tokens, error)) return nullptr;
        eshkol_ast_t parsed = eshkol::parse_syntax_tokens(std::move(tokens), macro_names);
        if (parsed.type == ESHKOL_INVALID) {
            set_error(error, ESHKOL_EXCEPTION_SYNTAX_ERROR,
                      "the datum is not a well-formed expression");
            return nullptr;
        }
        eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
        *ast = parsed;
        return ast;
    } catch (const std::bad_alloc&) {
        set_error(error, ESHKOL_EXCEPTION_ERROR,
                  "converting the datum to an expression needs more memory than "
                  "this process can allocate");
        return nullptr;
    }
}

eshkol::EvalMacroNames eval_macro_names = nullptr;

} // anonymous namespace

// ============================================================================
// Public API Implementation
// ============================================================================

namespace eshkol {

void register_eval_macro_names(EvalMacroNames provider) {
    eval_macro_names = provider;
}

eshkol_ast_t* eval_datum_to_ast(eshkol_tagged_value_t datum, void* jit,
                                EvalConversionError& error) {
    validate_eval_datum(datum);
    error.type = ESHKOL_EXCEPTION_ERROR;
    error.message[0] = '\0';
    std::set<std::string> names;
    if (jit && eval_macro_names) names = eval_macro_names(jit);
    ToplevelFormParseScope toplevel;
    return convert_datum(datum, names, error);
}

} // namespace eshkol

extern "C" {

/**
 * @brief Convert a runtime S-expression to an AST node.
 *
 * The same conversion eval uses (eshkol::eval_datum_to_ast), outside any
 * evaluation context's macro scope.
 *
 * @param sexp The S-expression to convert (cons cell structure).
 * @return Pointer to allocated AST node, or NULL (with the reason logged).
 */
eshkol_ast_t* eshkol_sexp_to_ast(eshkol_tagged_value_t sexp) {
    eshkol::EvalConversionError error;
    eshkol_ast_t* ast = eshkol::eval_datum_to_ast(sexp, nullptr, error);
    if (!ast) eshkol_error("%s", error.message);
    return ast;
}

/**
 * @brief Free an AST node created by eshkol_sexp_to_ast.
 *
 * Recursively frees all sub-nodes via the standard AST cleanup path.
 * Note: the JIT compiler may take ownership of AST nodes, in which case
 * this should not be called on them.
 *
 * @param ast The AST node to free (may be NULL).
 */
void eshkol_free_sexp_ast(eshkol_ast_t* ast) {
    // Use the standard AST cleanup
    if (ast) {
        eshkol_ast_clean(ast);
    }
}

/**
 * @brief Check whether an S-expression is headed by a recognized special-form symbol.
 *
 * @param sexp The S-expression to check.
 * @return true if @p sexp is a pair whose head is a symbol matching one of
 *         the known special forms (lambda, define, if, cond, case, match,
 *         let, let-star, letrec, letrec-star, and, or, when, unless, begin,
 *         quote, quasiquote, set!, define-syntax).
 */
bool eshkol_sexp_is_special_form(eshkol_tagged_value_t sexp) {
    if (!is_pair(sexp)) {
        return false;
    }

    eshkol_tagged_value_t head = pair_car(sexp);
    if (!is_symbol(head)) {
        return false;
    }

    const char* name = get_symbol_name(head);
    if (!name) {
        return false;
    }

    // Check against known special forms
    static const char* special_forms[] = {
        "lambda", "define", "if", "cond", "case", "match",
        "let", "let*", "letrec", "letrec*",
        "and", "or", "when", "unless",
        "begin", "quote", "quasiquote",
        "set!", "define-syntax",
        nullptr
    };

    for (const char** sf = special_forms; *sf; sf++) {
        if (strcmp(name, *sf) == 0) {
            return true;
        }
    }

    return false;
}

/**
 * @brief Get the head symbol name of an S-expression list.
 *
 * @param sexp The S-expression to examine.
 * @return The head symbol's name, or NULL if @p sexp is not a pair whose car is a symbol.
 */
const char* eshkol_sexp_head_symbol(eshkol_tagged_value_t sexp) {
    if (!is_pair(sexp)) {
        return nullptr;
    }

    eshkol_tagged_value_t head = pair_car(sexp);
    return get_symbol_name(head);
}

/** Count the number of elements in an S-expression list (0 if not a proper list). */
size_t eshkol_sexp_list_length(eshkol_tagged_value_t sexp) {
    return list_length(sexp);
}

/** Get the nth (zero-based) element of an S-expression list, or a null value if out of bounds. */
eshkol_tagged_value_t eshkol_sexp_list_ref(eshkol_tagged_value_t sexp, size_t index) {
    return list_ref(sexp, index);
}

// eshkol_compile_with_env is implemented in introspection.cpp
// because it requires access to the JIT infrastructure

} // extern "C"
