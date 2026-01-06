/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol Introspection API
 *
 * Provides runtime introspection capabilities for:
 * - Closure/procedure inspection (arity, captures, source code)
 * - Symbol manipulation (gensym, symbol->string)
 * - Code serialization (procedure->sexp)
 * - Runtime evaluation (eval)
 *
 * These features enable self-aware, self-modifying machine intelligence
 * and full R7RS Scheme compatibility.
 */
#ifndef ESHKOL_CORE_INTROSPECTION_H
#define ESHKOL_CORE_INTROSPECTION_H

#include <eshkol/eshkol.h>

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// ============================================================================
// Common Constants
// ============================================================================

// Null tagged value (type = NULL)
#ifndef ESHKOL_MAKE_NULL_VALUE
#define ESHKOL_MAKE_NULL_VALUE() \
    ((eshkol_tagged_value_t){ .type = ESHKOL_VALUE_NULL, .flags = 0, .reserved = 0, .data = {.raw_val = 0} })
#endif

// False tagged value
#ifndef ESHKOL_MAKE_FALSE_VALUE
#define ESHKOL_MAKE_FALSE_VALUE() \
    ((eshkol_tagged_value_t){ .type = ESHKOL_VALUE_BOOL, .flags = 0, .reserved = 0, .data = {.int_val = 0} })
#endif

// True tagged value
#ifndef ESHKOL_MAKE_TRUE_VALUE
#define ESHKOL_MAKE_TRUE_VALUE() \
    ((eshkol_tagged_value_t){ .type = ESHKOL_VALUE_BOOL, .flags = 0, .reserved = 0, .data = {.int_val = 1} })
#endif

// Closure flags (if not defined elsewhere)
#ifndef ESHKOL_CLOSURE_FLAG_VARIADIC
#define ESHKOL_CLOSURE_FLAG_VARIADIC  0x01  // Closure accepts variable arguments
#endif

#ifndef ESHKOL_CLOSURE_FLAG_NAMED
#define ESHKOL_CLOSURE_FLAG_NAMED     0x02  // Closure has a bound name
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Closure Introspection API
// ============================================================================

/**
 * @brief Check if a value is a closure/procedure.
 *
 * Returns true for any callable type (closure, primitive, continuation).
 *
 * @param value Tagged value to check
 * @return true if callable, false otherwise
 */
bool eshkol_is_procedure(eshkol_tagged_value_t value);

/**
 * @brief Check if a value is specifically a closure (has captures).
 *
 * Returns true only for closures, not primitives or continuations.
 *
 * @param value Tagged value to check
 * @return true if closure, false otherwise
 */
bool eshkol_is_closure(eshkol_tagged_value_t value);

/**
 * @brief Get the arity (number of parameters) of a procedure.
 *
 * For variadic procedures, returns the minimum required arguments.
 * Use eshkol_procedure_is_variadic() to check for variadicity.
 *
 * @param value Procedure to inspect
 * @return Arity, or -1 if not a procedure
 */
int eshkol_procedure_arity(eshkol_tagged_value_t value);

/**
 * @brief Check if a procedure is variadic (accepts variable arguments).
 *
 * @param value Procedure to inspect
 * @return true if variadic, false otherwise
 */
bool eshkol_procedure_is_variadic(eshkol_tagged_value_t value);

/**
 * @brief Get the number of captured values in a closure.
 *
 * @param value Closure to inspect
 * @return Number of captures, or 0 if not a closure or no captures
 */
size_t eshkol_closure_capture_count(eshkol_tagged_value_t value);

/**
 * @brief Get a specific captured value from a closure by index.
 *
 * @param value Closure to inspect
 * @param index Zero-based index of the capture
 * @return The captured value, or NULL_VALUE if invalid
 */
eshkol_tagged_value_t eshkol_closure_capture_ref(eshkol_tagged_value_t value, size_t index);

/**
 * @brief Get all captured values from a closure as a list.
 *
 * The returned list contains all captured values in capture order.
 * The list is allocated in the current arena.
 *
 * @param value Closure to inspect
 * @param arena Arena for list allocation
 * @return List of captured values, or empty list if no captures
 */
eshkol_tagged_value_t eshkol_closure_captures(eshkol_tagged_value_t value, void* arena);

/**
 * @brief Get the raw closure pointer from a tagged value.
 *
 * @param value Tagged value containing closure
 * @return Pointer to closure structure, or NULL if not a closure
 */
eshkol_closure_t* eshkol_get_closure(eshkol_tagged_value_t value);

// ============================================================================
// Symbol Manipulation API
// ============================================================================

/**
 * @brief Generate a unique symbol.
 *
 * Creates a new symbol that is guaranteed to be unique (not interned).
 * Format: G<counter> (e.g., G1, G2, G3, ...)
 *
 * @param arena Arena for symbol string allocation
 * @return New unique symbol as tagged value
 */
eshkol_tagged_value_t eshkol_gensym(void* arena);

/**
 * @brief Generate a unique symbol with a prefix.
 *
 * Creates a new symbol with the given prefix.
 * Format: <prefix><counter> (e.g., tmp1, tmp2, ...)
 *
 * @param prefix Prefix for the symbol name
 * @param arena Arena for symbol string allocation
 * @return New unique symbol as tagged value
 */
eshkol_tagged_value_t eshkol_gensym_prefix(const char* prefix, void* arena);

/**
 * @brief Convert a symbol to its string name.
 *
 * @param symbol Symbol value
 * @param arena Arena for string allocation (if needed)
 * @return String representation, or NULL_VALUE if not a symbol
 */
eshkol_tagged_value_t eshkol_symbol_to_string(eshkol_tagged_value_t symbol, void* arena);

/**
 * @brief Convert a string to an interned symbol.
 *
 * If a symbol with this name already exists, returns that symbol.
 * Otherwise, creates and interns a new symbol.
 *
 * @param str String value
 * @return Symbol with the given name, or NULL_VALUE if not a string
 */
eshkol_tagged_value_t eshkol_string_to_symbol(eshkol_tagged_value_t str);

/**
 * @brief Check if two symbols are equal.
 *
 * @param sym1 First symbol
 * @param sym2 Second symbol
 * @return true if equal, false otherwise
 */
bool eshkol_symbol_equal(eshkol_tagged_value_t sym1, eshkol_tagged_value_t sym2);

// ============================================================================
// Code Serialization API
// ============================================================================

/**
 * @brief Get the S-expression representation of a procedure.
 *
 * Returns the source code of the procedure as an S-expression.
 * For closures, this is the original lambda form.
 * For primitives, returns a descriptive form like (primitive <name>).
 *
 * @param proc Procedure to serialize
 * @param arena Arena for allocation
 * @return S-expression representation, or NULL_VALUE if not available
 */
eshkol_tagged_value_t eshkol_procedure_to_sexp(eshkol_tagged_value_t proc, void* arena);

/**
 * @brief Get just the body of a closure's lambda.
 *
 * Returns the body expression(s) of the lambda, without the
 * (lambda (params...) ...) wrapper.
 *
 * @param closure Closure to inspect
 * @param arena Arena for allocation
 * @return Body S-expression, or NULL_VALUE if not available
 */
eshkol_tagged_value_t eshkol_closure_body(eshkol_tagged_value_t closure, void* arena);

/**
 * @brief Get the parameter list of a closure.
 *
 * Returns the parameter list from the lambda form.
 *
 * @param closure Closure to inspect
 * @param arena Arena for allocation
 * @return Parameter list, or NULL_VALUE if not available
 */
eshkol_tagged_value_t eshkol_closure_params(eshkol_tagged_value_t closure, void* arena);

/**
 * @brief Get the name of a procedure (if available).
 *
 * Returns the bound name of the procedure, if it was defined with (define).
 * For anonymous lambdas, returns #f.
 *
 * @param proc Procedure to inspect
 * @return Symbol name, or FALSE if anonymous
 */
eshkol_tagged_value_t eshkol_procedure_name(eshkol_tagged_value_t proc);

// ============================================================================
// Runtime Evaluation API
// ============================================================================

/**
 * @brief Evaluate an S-expression at runtime.
 *
 * Compiles and executes the given S-expression using the JIT compiler.
 * The expression has access to the current global environment.
 *
 * @param sexp S-expression to evaluate
 * @param arena Arena for result allocation
 * @return Result of evaluation, or raises an exception on error
 */
eshkol_tagged_value_t eshkol_eval(eshkol_tagged_value_t sexp, void* arena);

/**
 * @brief Evaluate an S-expression with a custom environment.
 *
 * The environment is an association list mapping symbols to values.
 *
 * @param sexp S-expression to evaluate
 * @param env Environment (alist of (symbol . value) pairs)
 * @param arena Arena for result allocation
 * @return Result of evaluation
 */
eshkol_tagged_value_t eshkol_eval_env(eshkol_tagged_value_t sexp,
                                       eshkol_tagged_value_t env,
                                       void* arena);

/**
 * @brief Compile an S-expression to a procedure without executing.
 *
 * The S-expression should be a lambda form. The resulting procedure
 * can be called later.
 *
 * @param sexp Lambda S-expression to compile
 * @param arena Arena for allocation
 * @return Compiled procedure, or NULL_VALUE on error
 */
eshkol_tagged_value_t eshkol_compile(eshkol_tagged_value_t sexp, void* arena);

/**
 * @brief Convert an S-expression to a procedure.
 *
 * Wrapper around eshkol_compile for lambda forms.
 * Validates that the sexp is a lambda before compiling.
 *
 * @param sexp Lambda S-expression
 * @param arena Arena for allocation
 * @return Compiled procedure
 */
eshkol_tagged_value_t eshkol_sexp_to_procedure(eshkol_tagged_value_t sexp, void* arena);

/**
 * @brief Parse and evaluate a string as Eshkol code.
 *
 * Convenience function that parses the string and evaluates it.
 *
 * @param str String containing Eshkol code
 * @param arena Arena for allocation
 * @return Result of evaluation
 */
eshkol_tagged_value_t eshkol_eval_string(const char* str, void* arena);

// ============================================================================
// Reflection API
// ============================================================================

/**
 * @brief Get the type of a value.
 *
 * Returns a symbol representing the value's type:
 * 'integer, 'real, 'boolean, 'string, 'symbol, 'pair, 'vector,
 * 'procedure, 'closure, 'primitive, 'null, 'void, etc.
 *
 * @param value Value to inspect
 * @return Type symbol
 */
eshkol_tagged_value_t eshkol_type_of(eshkol_tagged_value_t value);

/**
 * @brief Get source location information for a procedure.
 *
 * Returns a list (filename line column) or #f if not available.
 *
 * @param proc Procedure to inspect
 * @param arena Arena for list allocation
 * @return Source location list or FALSE
 */
eshkol_tagged_value_t eshkol_source_location(eshkol_tagged_value_t proc, void* arena);

#ifdef __cplusplus
}
#endif

// ============================================================================
// C++ Helpers
// ============================================================================

#ifdef __cplusplus

namespace eshkol {

/**
 * @brief RAII wrapper for gensym with automatic prefix.
 */
class GensymGenerator {
public:
    explicit GensymGenerator(const char* prefix = "G")
        : prefix_(prefix ? prefix : "G") {}

    eshkol_tagged_value_t generate(void* arena) {
        return eshkol_gensym_prefix(prefix_, arena);
    }

private:
    const char* prefix_;
};

/**
 * @brief Helper to extract closure captures as a vector.
 */
inline std::vector<eshkol_tagged_value_t> getCapturesVector(eshkol_tagged_value_t closure) {
    std::vector<eshkol_tagged_value_t> result;
    size_t count = eshkol_closure_capture_count(closure);
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(eshkol_closure_capture_ref(closure, i));
    }
    return result;
}

} // namespace eshkol

#endif // __cplusplus

#endif // ESHKOL_CORE_INTROSPECTION_H
