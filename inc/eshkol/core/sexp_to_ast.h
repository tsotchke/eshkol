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
 *
 * Performance: O(n) single-pass traversal of the S-expression structure.
 */
#ifndef ESHKOL_CORE_SEXP_TO_AST_H
#define ESHKOL_CORE_SEXP_TO_AST_H

#include <eshkol/eshkol.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Core S-expression to AST Conversion
// ============================================================================

/**
 * @brief Convert a runtime S-expression to an AST node.
 *
 * Performs direct, single-pass conversion from cons-cell representation
 * to AST structure. Recognizes all standard Scheme special forms:
 *   - lambda, define, if, cond, case, match
 *   - let, let*, letrec, letrec*
 *   - and, or, when, unless
 *   - begin, quote, quasiquote
 *   - set!, define-syntax
 *
 * The returned AST is heap-allocated and must be freed by the caller
 * when no longer needed (unless passed to the JIT which takes ownership).
 *
 * @param sexp The S-expression to convert (cons cell structure)
 * @return Pointer to allocated AST node, or NULL on error
 *
 * @example
 *   // Convert '(+ 1 2) to AST
 *   eshkol_ast_t* ast = eshkol_sexp_to_ast(sexp);
 *   if (ast) {
 *       // Use AST with JIT compiler
 *       jit->executeTagged(ast);
 *   }
 */
eshkol_ast_t* eshkol_sexp_to_ast(eshkol_tagged_value_t sexp);

/**
 * @brief Free an AST node created by eshkol_sexp_to_ast.
 *
 * Recursively frees all sub-nodes and associated memory.
 * Note: The JIT compiler may take ownership of AST nodes, in which
 * case you should NOT call this function.
 *
 * @param ast The AST node to free (may be NULL)
 */
void eshkol_free_sexp_ast(eshkol_ast_t* ast);

// ============================================================================
// Environment-aware Compilation
// ============================================================================

/**
 * @brief Environment structure for compile-with-bindings.
 *
 * Allows passing pre-defined bindings to the compiler, enabling
 * closures compiled at runtime to access values from the calling context.
 */
typedef struct eshkol_compile_env {
    const char** names;           /**< Array of binding names */
    eshkol_tagged_value_t* values; /**< Array of binding values */
    size_t count;                 /**< Number of bindings */
} eshkol_compile_env_t;

/**
 * @brief Compile an S-expression with environment bindings.
 *
 * Compiles the S-expression, making the provided environment bindings
 * available as free variables in the resulting closure.
 *
 * @param sexp The S-expression to compile
 * @param env Environment bindings (may be NULL for no bindings)
 * @param arena Arena for result allocation
 * @return Compiled procedure, or NULL value on error
 *
 * @example
 *   // Compile (lambda (y) (+ x y)) where x is bound to 10
 *   const char* names[] = {"x"};
 *   eshkol_tagged_value_t values[] = {eshkol_make_int64(10, true)};
 *   eshkol_compile_env_t env = {names, values, 1};
 *   eshkol_tagged_value_t closure = eshkol_compile_with_env(lambda_sexp, &env, arena);
 */
eshkol_tagged_value_t eshkol_compile_with_env(
    eshkol_tagged_value_t sexp,
    const eshkol_compile_env_t* env,
    void* arena
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Check if an S-expression is a special form.
 *
 * Returns true if the S-expression starts with a recognized special form
 * symbol (lambda, define, if, let, etc.).
 *
 * @param sexp The S-expression to check
 * @return true if special form, false otherwise
 */
bool eshkol_sexp_is_special_form(eshkol_tagged_value_t sexp);

/**
 * @brief Get the head symbol of an S-expression list.
 *
 * If the S-expression is a non-empty list starting with a symbol,
 * returns that symbol's name. Otherwise returns NULL.
 *
 * @param sexp The S-expression to examine
 * @return Symbol name string, or NULL if not a symbol-headed list
 */
const char* eshkol_sexp_head_symbol(eshkol_tagged_value_t sexp);

/**
 * @brief Count the number of elements in an S-expression list.
 *
 * @param sexp The S-expression list
 * @return Number of elements, or 0 if not a proper list
 */
size_t eshkol_sexp_list_length(eshkol_tagged_value_t sexp);

/**
 * @brief Get the nth element of an S-expression list.
 *
 * @param sexp The S-expression list
 * @param index Zero-based index
 * @return The element at index, or NULL value if out of bounds
 */
eshkol_tagged_value_t eshkol_sexp_list_ref(eshkol_tagged_value_t sexp, size_t index);

#ifdef __cplusplus
}
#endif

#endif // ESHKOL_CORE_SEXP_TO_AST_H
