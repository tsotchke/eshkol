/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol Introspection Implementation
 *
 * Provides runtime introspection capabilities for closures, symbols,
 * code serialization, and eval - enabling self-aware machine intelligence
 * and full R7RS Scheme compatibility.
 */

#include <eshkol/core/introspection.h>
#include <eshkol/core/sexp_to_ast.h>
#include <eshkol/logger.h>
#include "arena_memory.h"

// For eval/JIT integration
#include "../repl/repl_jit.h"

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <string>
#include <cstring>
#include <cstdio>
#include <sstream>

namespace {

// ============================================================================
// Global State
// ============================================================================

// Gensym counter for unique symbol generation
std::atomic<uint64_t> g_gensym_counter{1};

// Symbol interning table
std::mutex g_symbol_mutex;
std::unordered_map<std::string, eshkol_tagged_value_t> g_interned_symbols;

// Eval JIT context (lazy-initialized singleton)
std::mutex g_eval_jit_mutex;
std::unique_ptr<eshkol::ReplJITContext> g_eval_jit_context;

// Get or create the eval JIT context
eshkol::ReplJITContext* get_eval_jit() {
    std::lock_guard<std::mutex> lock(g_eval_jit_mutex);
    if (!g_eval_jit_context) {
        try {
            g_eval_jit_context = std::make_unique<eshkol::ReplJITContext>();
            eshkol_info("Eval JIT context initialized");
        } catch (const std::exception& e) {
            eshkol_error("Failed to initialize eval JIT: %s", e.what());
            return nullptr;
        }
    }
    return g_eval_jit_context.get();
}

// Serialize a tagged value (S-expression) to a string buffer
// Returns a malloc'd string that the caller must free
char* serialize_sexp_to_string(eshkol_tagged_value_t value) {
    char* buffer = nullptr;
    size_t size = 0;

    // Use open_memstream to create a FILE* that writes to a buffer
    FILE* memstream = open_memstream(&buffer, &size);
    if (!memstream) {
        eshkol_error("serialize_sexp_to_string: Failed to create memstream");
        return nullptr;
    }

    // Set up display options to write to the memstream
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    opts.output = memstream;
    opts.quote_strings = 1;  // Use 'write' semantics for proper serialization

    // Display the value to the memstream
    eshkol_display_value_opts(&value, &opts);

    // Close the stream to finalize the buffer
    fclose(memstream);

    return buffer;
}

// ============================================================================
// Helper Functions
// ============================================================================

// Create a null value
inline eshkol_tagged_value_t make_null() {
    return ESHKOL_MAKE_NULL_VALUE();
}

// Create a false value
inline eshkol_tagged_value_t make_false() {
    return ESHKOL_MAKE_FALSE_VALUE();
}

// Check if value is null
inline bool is_null(eshkol_tagged_value_t value) {
    return value.type == ESHKOL_VALUE_NULL;
}

// Check if value is a callable (closure, primitive, etc.)
inline bool is_callable(eshkol_tagged_value_t value) {
    return ESHKOL_IS_ANY_CALLABLE_TYPE(value.type);
}

// Check if value is specifically a closure
inline bool is_closure(eshkol_tagged_value_t value) {
    return ESHKOL_IS_CLOSURE_COMPAT(value);
}

// Check if value is a pair/cons cell
inline bool is_pair(eshkol_tagged_value_t value) {
    return ESHKOL_IS_CONS_COMPAT(value);
}

// Check if value is a symbol
inline bool is_symbol(eshkol_tagged_value_t value) {
    return value.type == ESHKOL_VALUE_SYMBOL;
}

// Check if value is a string
inline bool is_string(eshkol_tagged_value_t value) {
    return ESHKOL_IS_STRING_COMPAT(value);
}

// Get raw pointer from tagged value
inline void* get_ptr(eshkol_tagged_value_t value) {
    return reinterpret_cast<void*>(value.data.ptr_val);
}

// Create a cons cell (pair)
eshkol_tagged_value_t make_pair(eshkol_tagged_value_t car,
                                  eshkol_tagged_value_t cdr,
                                  void* arena) {
    if (!arena) {
        return make_null();
    }

    arena_t* a = static_cast<arena_t*>(arena);
    arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(a);

    if (!cell) {
        return make_null();
    }

    cell->car = car;
    cell->cdr = cdr;

    // Create tagged value pointing to the cons cell
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_HEAP_PTR;  // Use consolidated type
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = reinterpret_cast<uint64_t>(cell);

    return result;
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

// Check if value is a primitive function (builtin)
bool is_primitive(eshkol_tagged_value_t value) {
    uint8_t base_type = value.type & 0x3F;

    // Check consolidated CALLABLE type with PRIMITIVE subtype
    if (base_type == ESHKOL_VALUE_CALLABLE) {
        uint64_t ptr = value.data.ptr_val;
        if (!ptr) return false;
        eshkol_object_header_t* header = ESHKOL_GET_HEADER(reinterpret_cast<void*>(ptr));
        return header && header->subtype == CALLABLE_SUBTYPE_PRIMITIVE;
    }

    return false;
}

// Get primitive structure from tagged value
eshkol_primitive_t* get_primitive(eshkol_tagged_value_t value) {
    if (!is_primitive(value)) {
        return nullptr;
    }
    return reinterpret_cast<eshkol_primitive_t*>(value.data.ptr_val);
}

} // anonymous namespace

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

// ----------------------------------------------------------------------------
// Closure Introspection API
// ----------------------------------------------------------------------------

bool eshkol_is_procedure(eshkol_tagged_value_t value) {
    return is_callable(value);
}

bool eshkol_is_closure(eshkol_tagged_value_t value) {
    return is_closure(value);
}

int eshkol_procedure_arity(eshkol_tagged_value_t value) {
    if (!is_callable(value)) {
        return -1;
    }

    if (is_closure(value)) {
        eshkol_closure_t* closure = eshkol_get_closure(value);
        if (closure) {
            return closure->input_arity;
        }
    }

    if (is_primitive(value)) {
        eshkol_primitive_t* prim = get_primitive(value);
        if (prim) {
            return prim->input_arity;
        }
    }

    // For continuations, arity is always 1 (they accept one value)
    void* ptr = get_ptr(value);
    if (ptr) {
        eshkol_object_header_t* header = ESHKOL_GET_HEADER(ptr);
        if (header && header->subtype == CALLABLE_SUBTYPE_CONTINUATION) {
            return 1;
        }
    }

    return -1;
}

bool eshkol_procedure_is_variadic(eshkol_tagged_value_t value) {
    if (!is_callable(value)) {
        return false;
    }

    if (is_closure(value)) {
        eshkol_closure_t* closure = eshkol_get_closure(value);
        if (closure) {
            // Check variadic flag in closure flags
            return (closure->flags & ESHKOL_CLOSURE_FLAG_VARIADIC) != 0;
        }
    }

    if (is_primitive(value)) {
        eshkol_primitive_t* prim = get_primitive(value);
        if (prim) {
            return PRIMITIVE_IS_VARIADIC(prim);
        }
    }

    // Continuations are not variadic - they accept exactly one value
    return false;
}

size_t eshkol_closure_capture_count(eshkol_tagged_value_t value) {
    if (!is_closure(value)) {
        return 0;
    }

    eshkol_closure_t* closure = eshkol_get_closure(value);
    if (!closure || !closure->env) {
        return 0;
    }

    // num_captures is packed: num_captures | (fixed_params << 16) | (is_variadic << 63)
    // Extract just the lower 16 bits for the capture count
    return closure->env->num_captures & 0xFFFF;
}

eshkol_tagged_value_t eshkol_closure_capture_ref(eshkol_tagged_value_t value, size_t index) {
    if (!is_closure(value)) {
        return make_null();
    }

    eshkol_closure_t* closure = eshkol_get_closure(value);
    if (!closure || !closure->env) {
        return make_null();
    }

    // Get actual capture count (lower 16 bits of num_captures)
    size_t count = closure->env->num_captures & 0xFFFF;
    if (index >= count) {
        return make_null();
    }

    // Captures are stored directly as eshkol_tagged_value_t in the flexible array
    return closure->env->captures[index];
}

eshkol_tagged_value_t eshkol_closure_captures(eshkol_tagged_value_t value, void* arena) {
    if (!arena) {
        return make_null();
    }

    size_t count = eshkol_closure_capture_count(value);
    if (count == 0) {
        return make_null();
    }

    // Build list in reverse order, then it will be in correct order
    eshkol_tagged_value_t result = make_null();

    for (size_t i = count; i > 0; --i) {
        eshkol_tagged_value_t capture = eshkol_closure_capture_ref(value, i - 1);
        result = make_pair(capture, result, arena);
    }

    return result;
}

eshkol_closure_t* eshkol_get_closure(eshkol_tagged_value_t value) {
    if (!is_closure(value)) {
        return nullptr;
    }
    return static_cast<eshkol_closure_t*>(get_ptr(value));
}

// ----------------------------------------------------------------------------
// Symbol Manipulation API
// ----------------------------------------------------------------------------

eshkol_tagged_value_t eshkol_gensym(void* arena) {
    return eshkol_gensym_prefix("G", arena);
}

eshkol_tagged_value_t eshkol_gensym_prefix(const char* prefix, void* arena) {
    if (!arena) {
        return make_null();
    }

    uint64_t counter = g_gensym_counter.fetch_add(1, std::memory_order_relaxed);

    // Format: <prefix><counter>
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "%s%llu",
             prefix ? prefix : "G",
             (unsigned long long)counter);

    // Allocate symbol string in arena
    arena_t* a = static_cast<arena_t*>(arena);
    size_t len = strlen(buffer);
    char* sym_str = static_cast<char*>(
        arena_allocate_aligned(a, len + 1, 1)
    );

    if (!sym_str) {
        return make_null();
    }

    memcpy(sym_str, buffer, len + 1);

    // Create uninterned symbol (gensyms are unique, not interned)
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_SYMBOL;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = reinterpret_cast<uint64_t>(sym_str);

    return result;
}

eshkol_tagged_value_t eshkol_symbol_to_string(eshkol_tagged_value_t symbol, void* arena) {
    if (!is_symbol(symbol)) {
        return make_null();
    }

    const char* sym_name = static_cast<const char*>(get_ptr(symbol));
    if (!sym_name) {
        return make_null();
    }

    if (!arena) {
        // Return the symbol's string directly as a string value
        // Note: This shares the string data with the symbol
        eshkol_tagged_value_t result;
        result.type = ESHKOL_VALUE_HEAP_PTR;  // Strings are heap objects
        result.flags = 0;
        result.reserved = 0;
        result.data.ptr_val = symbol.data.ptr_val;
        return result;
    }

    // Copy the string to the arena with proper header
    arena_t* a = static_cast<arena_t*>(arena);
    size_t len = strlen(sym_name);
    char* str_copy = arena_allocate_string_with_header(a, len);

    if (!str_copy) {
        return make_null();
    }

    memcpy(str_copy, sym_name, len + 1);

    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_HEAP_PTR;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = reinterpret_cast<uint64_t>(str_copy);

    return result;
}

eshkol_tagged_value_t eshkol_string_to_symbol(eshkol_tagged_value_t str) {
    if (!is_string(str)) {
        return make_null();
    }

    const char* str_value = static_cast<const char*>(get_ptr(str));
    if (!str_value) {
        return make_null();
    }

    std::string key(str_value);

    // Check if symbol is already interned
    {
        std::lock_guard<std::mutex> lock(g_symbol_mutex);
        auto it = g_interned_symbols.find(key);
        if (it != g_interned_symbols.end()) {
            return it->second;
        }
    }

    // Create new interned symbol
    // Note: For interned symbols, we keep the original string pointer
    // In a production implementation, we'd want to manage symbol lifetime
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_SYMBOL;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = str.data.ptr_val;

    // Intern the symbol
    {
        std::lock_guard<std::mutex> lock(g_symbol_mutex);
        g_interned_symbols[key] = result;
    }

    return result;
}

bool eshkol_symbol_equal(eshkol_tagged_value_t sym1, eshkol_tagged_value_t sym2) {
    if (!is_symbol(sym1) || !is_symbol(sym2)) {
        return false;
    }

    // For interned symbols, pointer comparison suffices
    if (sym1.data.ptr_val == sym2.data.ptr_val) {
        return true;
    }

    // Fall back to string comparison for uninterned symbols (gensyms)
    const char* s1 = static_cast<const char*>(get_ptr(sym1));
    const char* s2 = static_cast<const char*>(get_ptr(sym2));

    if (!s1 || !s2) {
        return false;
    }

    return strcmp(s1, s2) == 0;
}

// ----------------------------------------------------------------------------
// Code Serialization API
// ----------------------------------------------------------------------------

eshkol_tagged_value_t eshkol_procedure_to_sexp(eshkol_tagged_value_t proc, void* arena) {
    if (!is_callable(proc)) {
        return make_null();
    }

    if (is_closure(proc)) {
        eshkol_closure_t* closure = eshkol_get_closure(proc);
        if (closure && closure->sexpr_ptr) {
            // The closure stores its S-expression representation
            // Return it directly (it's already a tagged value or S-expression)
            eshkol_tagged_value_t sexp;
            sexp.type = ESHKOL_VALUE_HEAP_PTR;  // S-expressions are lists (cons cells)
            sexp.flags = 0;
            sexp.reserved = 0;
            sexp.data.ptr_val = closure->sexpr_ptr;
            return sexp;
        }
    }

    // Check for primitive
    if (ESHKOL_IS_CALLABLE_TYPE(proc.type)) {
        void* ptr = get_ptr(proc);
        if (ptr) {
            eshkol_object_header_t* header = ESHKOL_GET_HEADER(ptr);
            if (header && header->subtype == CALLABLE_SUBTYPE_PRIMITIVE) {
                // For primitives, return (primitive <name>)
                if (!arena) {
                    return make_null();
                }

                // Create symbol 'primitive
                eshkol_tagged_value_t prim_sym;
                prim_sym.type = ESHKOL_VALUE_SYMBOL;
                prim_sym.flags = 0;
                prim_sym.reserved = 0;
                prim_sym.data.ptr_val = reinterpret_cast<uint64_t>("primitive");

                // Create (primitive)
                eshkol_tagged_value_t result = make_pair(prim_sym, make_null(), arena);
                return result;
            }
        }
    }

    return make_null();
}

eshkol_tagged_value_t eshkol_closure_body(eshkol_tagged_value_t closure, void* arena) {
    eshkol_tagged_value_t sexp = eshkol_procedure_to_sexp(closure, arena);

    if (is_null(sexp)) {
        return make_null();
    }

    // S-expression format: (lambda (params...) body...)
    // Skip 'lambda symbol
    eshkol_tagged_value_t rest = pair_cdr(sexp);
    if (is_null(rest)) {
        return make_null();
    }

    // Skip params list
    rest = pair_cdr(rest);
    if (is_null(rest)) {
        return make_null();
    }

    // Return the body (rest of the list)
    return rest;
}

eshkol_tagged_value_t eshkol_closure_params(eshkol_tagged_value_t closure, void* arena) {
    eshkol_tagged_value_t sexp = eshkol_procedure_to_sexp(closure, arena);

    if (is_null(sexp)) {
        return make_null();
    }

    // S-expression format: (lambda (params...) body...)
    // Skip 'lambda symbol
    eshkol_tagged_value_t rest = pair_cdr(sexp);
    if (is_null(rest)) {
        return make_null();
    }

    // Return params list (car of rest)
    return pair_car(rest);
}

eshkol_tagged_value_t eshkol_procedure_name(eshkol_tagged_value_t proc) {
    if (!is_callable(proc)) {
        return make_false();
    }

    // Check for closure - closures have a name field set when defined via (define name ...)
    if (is_closure(proc)) {
        eshkol_closure_t* closure = eshkol_get_closure(proc);
        if (closure && closure->name) {
            // Return the closure name as a symbol
            eshkol_tagged_value_t result;
            result.type = ESHKOL_VALUE_SYMBOL;
            result.flags = 0;
            result.reserved = 0;
            result.data.ptr_val = reinterpret_cast<uint64_t>(closure->name);
            return result;
        }
    }

    // Check for primitive - primitives have a name field
    if (is_primitive(proc)) {
        eshkol_primitive_t* prim = get_primitive(proc);
        if (prim && prim->name) {
            // Return the primitive name as a symbol
            eshkol_tagged_value_t result;
            result.type = ESHKOL_VALUE_SYMBOL;
            result.flags = 0;
            result.reserved = 0;
            result.data.ptr_val = reinterpret_cast<uint64_t>(prim->name);
            return result;
        }
    }

    // Anonymous lambdas and continuations without names return #f
    return make_false();
}

// ----------------------------------------------------------------------------
// Runtime Evaluation API
// ----------------------------------------------------------------------------

eshkol_tagged_value_t eshkol_eval(eshkol_tagged_value_t sexp, void* arena) {
    // Direct S-expression to AST conversion (O(n) single-pass)
    // Avoids serialize→parse overhead
    eshkol_ast_t* ast = eshkol_sexp_to_ast(sexp);
    if (!ast) {
        eshkol_error("eshkol_eval: Failed to convert S-expression to AST");
        return make_null();
    }

    // Get or create the eval JIT context
    eshkol::ReplJITContext* jit = get_eval_jit();
    if (!jit) {
        eshkol_free_sexp_ast(ast);
        eshkol_error("eshkol_eval: Failed to initialize JIT context");
        return make_null();
    }

    // Execute the AST through the JIT
    eshkol_tagged_value_t result = jit->executeTagged(ast);

    // Free the AST (JIT may have taken ownership of parts, but we allocated the root)
    eshkol_free_sexp_ast(ast);

    (void)arena;  // Arena passed for potential allocation needs
    return result;
}

eshkol_tagged_value_t eshkol_eval_env(eshkol_tagged_value_t sexp,
                                       eshkol_tagged_value_t env,
                                       void* arena) {
    (void)arena;  // Arena passed for potential allocation needs

    // Convert the S-expression to AST directly (O(n) single-pass)
    eshkol_ast_t* body_ast = eshkol_sexp_to_ast(sexp);
    if (!body_ast) {
        eshkol_error("eshkol_eval_env: Failed to convert S-expression to AST");
        return make_null();
    }

    eshkol_ast_t* final_ast = body_ast;

    // If there are environment bindings, wrap in a let-expression
    if (!is_null(env)) {
        // Count bindings in the alist: ((sym1 . val1) (sym2 . val2) ...)
        size_t num_bindings = 0;
        eshkol_tagged_value_t current = env;
        while (is_pair(current)) {
            num_bindings++;
            current = pair_cdr(current);
        }

        if (num_bindings > 0) {
            // Create let bindings array - each binding is a (var, value) pair
            eshkol_ast_t* bindings = static_cast<eshkol_ast_t*>(
                malloc(num_bindings * 2 * sizeof(eshkol_ast_t)));

            current = env;
            size_t i = 0;
            while (is_pair(current)) {
                eshkol_tagged_value_t binding = pair_car(current);
                if (is_pair(binding)) {
                    eshkol_tagged_value_t sym = pair_car(binding);
                    eshkol_tagged_value_t val = pair_cdr(binding);

                    if (is_symbol(sym)) {
                        const char* sym_name = static_cast<const char*>(get_ptr(sym));

                        // Variable name
                        bindings[i * 2].type = ESHKOL_VAR;
                        bindings[i * 2].variable.id = strdup(sym_name);
                        bindings[i * 2].variable.data = nullptr;

                        // Value - convert the S-expression value to AST
                        eshkol_ast_t* val_ast = eshkol_sexp_to_ast(val);
                        if (val_ast) {
                            bindings[i * 2 + 1] = *val_ast;
                            free(val_ast);
                        } else {
                            // Fallback: create a null value
                            bindings[i * 2 + 1].type = ESHKOL_NULL;
                        }
                        i++;
                    }
                }
                current = pair_cdr(current);
            }

            // Create the let AST node
            eshkol_ast_t* let_ast = eshkol_alloc_symbolic_ast();
            let_ast->type = ESHKOL_OP;
            let_ast->operation.op = ESHKOL_LET_OP;
            let_ast->operation.let_op.bindings = bindings;
            let_ast->operation.let_op.num_bindings = i;  // Actual count of valid bindings
            let_ast->operation.let_op.body = body_ast;
            let_ast->operation.let_op.name = nullptr;  // Not a named let
            let_ast->operation.let_op.binding_types = nullptr;

            final_ast = let_ast;
        }
    }

    // Get or create the eval JIT context
    eshkol::ReplJITContext* jit = get_eval_jit();
    if (!jit) {
        eshkol_free_sexp_ast(final_ast);
        eshkol_error("eshkol_eval_env: Failed to initialize JIT context");
        return make_null();
    }

    // Execute the AST through the JIT
    eshkol_tagged_value_t result = jit->executeTagged(final_ast);

    // Free the AST
    eshkol_free_sexp_ast(final_ast);

    return result;
}

eshkol_tagged_value_t eshkol_compile(eshkol_tagged_value_t sexp, void* arena) {
    // Direct S-expression to AST conversion (O(n) single-pass)
    // Avoids serialize→parse overhead
    eshkol_ast_t* ast = eshkol_sexp_to_ast(sexp);
    if (!ast) {
        eshkol_error("eshkol_compile: Failed to convert S-expression to AST");
        return make_null();
    }

    // Get or create the eval JIT context
    eshkol::ReplJITContext* jit = get_eval_jit();
    if (!jit) {
        eshkol_free_sexp_ast(ast);
        eshkol_error("eshkol_compile: Failed to initialize JIT context");
        return make_null();
    }

    // The AST might be ESHKOL_FUNC for bare lambda, or ESHKOL_OP for operations.
    // We proceed with execution regardless of type since the JIT will handle
    // type validation and return an appropriate result or error.

    // Execute the lambda expression through the JIT
    // When evaluating a bare lambda expression like (lambda (x) (* x x)),
    // the result IS the closure object itself - it doesn't get called
    eshkol_tagged_value_t result = jit->executeTagged(ast);

    // Free the AST (JIT may have taken ownership of parts, but we allocated the root)
    eshkol_free_sexp_ast(ast);

    (void)arena;  // Arena passed for potential allocation needs
    return result;
}

eshkol_tagged_value_t eshkol_sexp_to_procedure(eshkol_tagged_value_t sexp, void* arena) {
    // Validate that sexp is a lambda form
    if (!is_pair(sexp)) {
        eshkol_error("eshkol_sexp_to_procedure: Expected lambda S-expression");
        return make_null();
    }

    // Check for 'lambda at head
    eshkol_tagged_value_t head = pair_car(sexp);
    if (!is_symbol(head)) {
        eshkol_error("eshkol_sexp_to_procedure: Expected lambda symbol");
        return make_null();
    }

    const char* head_name = static_cast<const char*>(get_ptr(head));
    if (!head_name || strcmp(head_name, "lambda") != 0) {
        eshkol_error("eshkol_sexp_to_procedure: Expected 'lambda', got '%s'",
                     head_name ? head_name : "(null)");
        return make_null();
    }

    // Compile the lambda
    return eshkol_compile(sexp, arena);
}

eshkol_tagged_value_t eshkol_compile_with_env(
    eshkol_tagged_value_t sexp,
    const eshkol_compile_env_t* env,
    void* arena
) {
    (void)arena;  // Arena passed for potential allocation needs

    // Convert the S-expression to AST
    eshkol_ast_t* body_ast = eshkol_sexp_to_ast(sexp);
    if (!body_ast) {
        eshkol_error("eshkol_compile_with_env: Failed to convert S-expression to AST");
        return make_null();
    }

    eshkol_ast_t* final_ast = body_ast;

    // If there are environment bindings, wrap in a let-expression
    if (env && env->count > 0) {
        // Create let bindings array - each binding is a (var, value) pair
        eshkol_ast_t* bindings = static_cast<eshkol_ast_t*>(
            malloc(env->count * 2 * sizeof(eshkol_ast_t)));

        for (size_t i = 0; i < env->count; i++) {
            // Variable name
            bindings[i * 2].type = ESHKOL_VAR;
            bindings[i * 2].variable.id = strdup(env->names[i]);
            bindings[i * 2].variable.data = nullptr;

            // Value - convert the tagged value to an AST representation
            eshkol_tagged_value_t val = env->values[i];
            eshkol_ast_t val_ast;
            memset(&val_ast, 0, sizeof(val_ast));

            // Handle strings first since they use multiple type representations
            if (is_string(val)) {
                const char* str = static_cast<const char*>(get_ptr(val));
                val_ast.type = ESHKOL_STRING;
                val_ast.str_val.ptr = strdup(str ? str : "");
                val_ast.str_val.size = str ? strlen(str) : 0;
            } else {
                switch (val.type) {
                    case ESHKOL_VALUE_INT64:
                        val_ast.type = ESHKOL_INT64;
                        val_ast.int64_val = val.data.int_val;
                        break;
                    case ESHKOL_VALUE_DOUBLE:
                        val_ast.type = ESHKOL_DOUBLE;
                        val_ast.double_val = val.data.double_val;
                        break;
                    case ESHKOL_VALUE_BOOL:
                        val_ast.type = ESHKOL_BOOL;
                        val_ast.int64_val = val.data.int_val;
                        break;
                    case ESHKOL_VALUE_CHAR:
                        val_ast.type = ESHKOL_CHAR;
                        val_ast.int64_val = val.data.int_val;
                        break;
                    case ESHKOL_VALUE_NULL:
                        val_ast.type = ESHKOL_NULL;
                        break;
                    default:
                        // For complex types (closures, cons cells, etc.),
                        // we need to quote them to preserve them as values
                        // This is a limitation - complex runtime values can't be directly embedded
                        eshkol_error("eshkol_compile_with_env: Cannot embed complex type %d in environment", val.type);
                        free(bindings);
                        eshkol_free_sexp_ast(body_ast);
                        return make_null();
                }
            }

            bindings[i * 2 + 1] = val_ast;
        }

        // Create the let AST node
        eshkol_ast_t* let_ast = eshkol_alloc_symbolic_ast();
        let_ast->type = ESHKOL_OP;
        let_ast->operation.op = ESHKOL_LET_OP;
        let_ast->operation.let_op.bindings = bindings;
        let_ast->operation.let_op.num_bindings = env->count;
        let_ast->operation.let_op.body = body_ast;
        let_ast->operation.let_op.name = nullptr;  // Not a named let
        let_ast->operation.let_op.binding_types = nullptr;

        final_ast = let_ast;
    }

    // Get or create the eval JIT context
    eshkol::ReplJITContext* jit = get_eval_jit();
    if (!jit) {
        eshkol_free_sexp_ast(final_ast);
        eshkol_error("eshkol_compile_with_env: Failed to initialize JIT context");
        return make_null();
    }

    // Execute the expression through the JIT to compile it
    eshkol_tagged_value_t result = jit->executeTagged(final_ast);

    // Free the AST
    eshkol_free_sexp_ast(final_ast);

    return result;
}

eshkol_tagged_value_t eshkol_eval_string(const char* str, void* arena) {
    if (!str || !*str) {
        return make_null();
    }

    // Get or create the eval JIT context
    eshkol::ReplJITContext* jit = get_eval_jit();
    if (!jit) {
        eshkol_error("eshkol_eval_string: Failed to initialize JIT context");
        return make_null();
    }

    // Parse the string to an AST using istringstream
    std::istringstream input_stream(str);
    eshkol_ast_t ast = eshkol_parse_next_ast_from_stream(input_stream);

    // Check if parsing succeeded (ESHKOL_INVALID = 0)
    if (ast.type == ESHKOL_INVALID) {
        eshkol_error("eshkol_eval_string: Failed to parse expression");
        return make_null();
    }

    // Execute the AST using the JIT with proper type conversion
    // executeTagged() analyzes the AST's inferred_hott_type field to return
    // a properly typed tagged value instead of a raw pointer
    eshkol_tagged_value_t result = jit->executeTagged(&ast);

    (void)arena;  // Arena passed for potential allocation needs
    return result;
}

// ----------------------------------------------------------------------------
// Reflection API
// ----------------------------------------------------------------------------

eshkol_tagged_value_t eshkol_type_of(eshkol_tagged_value_t value) {
    const char* type_name = nullptr;

    // Check immediate types first
    switch (value.type) {
        case ESHKOL_VALUE_NULL:
            type_name = "null";
            break;
        case ESHKOL_VALUE_INT64:
            type_name = "integer";
            break;
        case ESHKOL_VALUE_DOUBLE:
            type_name = "real";
            break;
        case ESHKOL_VALUE_BOOL:
            type_name = "boolean";
            break;
        case ESHKOL_VALUE_CHAR:
            type_name = "char";
            break;
        case ESHKOL_VALUE_SYMBOL:
            type_name = "symbol";
            break;
        case ESHKOL_VALUE_DUAL_NUMBER:
            type_name = "dual-number";
            break;
        case ESHKOL_VALUE_HEAP_PTR: {
            // Need to check subtype from header
            void* ptr = get_ptr(value);
            if (ptr) {
                eshkol_object_header_t* header = ESHKOL_GET_HEADER(ptr);
                if (header) {
                    switch (header->subtype) {
                        case HEAP_SUBTYPE_CONS:
                            type_name = "pair";
                            break;
                        case HEAP_SUBTYPE_STRING:
                            type_name = "string";
                            break;
                        case HEAP_SUBTYPE_VECTOR:
                            type_name = "vector";
                            break;
                        case HEAP_SUBTYPE_TENSOR:
                            type_name = "tensor";
                            break;
                        case HEAP_SUBTYPE_HASH:
                            type_name = "hash-table";
                            break;
                        case HEAP_SUBTYPE_EXCEPTION:
                            type_name = "exception";
                            break;
                        case HEAP_SUBTYPE_PORT:
                            type_name = "port";
                            break;
                        default:
                            type_name = "heap-object";
                            break;
                    }
                }
            }
            if (!type_name) type_name = "heap-object";
            break;
        }
        case ESHKOL_VALUE_CALLABLE: {
            // Need to check subtype from header
            void* ptr = get_ptr(value);
            if (ptr) {
                eshkol_object_header_t* header = ESHKOL_GET_HEADER(ptr);
                if (header) {
                    switch (header->subtype) {
                        case CALLABLE_SUBTYPE_CLOSURE:
                            type_name = "closure";
                            break;
                        case CALLABLE_SUBTYPE_LAMBDA_SEXPR:
                            type_name = "lambda-sexpr";
                            break;
                        case CALLABLE_SUBTYPE_AD_NODE:
                            type_name = "ad-node";
                            break;
                        case CALLABLE_SUBTYPE_PRIMITIVE:
                            type_name = "primitive";
                            break;
                        case CALLABLE_SUBTYPE_CONTINUATION:
                            type_name = "continuation";
                            break;
                        default:
                            type_name = "procedure";
                            break;
                    }
                }
            }
            if (!type_name) type_name = "procedure";
            break;
        }
        // Handle legacy types for backward compatibility
        case ESHKOL_VALUE_CONS_PTR:
            type_name = "pair";
            break;
        case ESHKOL_VALUE_STRING_PTR:
            type_name = "string";
            break;
        case ESHKOL_VALUE_VECTOR_PTR:
            type_name = "vector";
            break;
        case ESHKOL_VALUE_TENSOR_PTR:
            type_name = "tensor";
            break;
        case ESHKOL_VALUE_CLOSURE_PTR:
            type_name = "closure";
            break;
        default:
            type_name = "unknown";
            break;
    }

    // Return as interned symbol
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_SYMBOL;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = reinterpret_cast<uint64_t>(type_name);

    return result;
}

eshkol_tagged_value_t eshkol_source_location(eshkol_tagged_value_t proc, void* arena) {
    if (!is_callable(proc)) {
        return make_false();
    }

    // Source locations are tracked at compile-time in the AST but are not currently
    // preserved in runtime closure structures. The closure structure prioritizes
    // execution-relevant data (func_ptr, env, sexpr_ptr, type info) over debug metadata.
    //
    // To enable source location retrieval at runtime, the following would be needed:
    // 1. Add source_location field to eshkol_closure_t structure
    // 2. Capture location during lambda compilation in CodeGenerator
    // 3. Store location in the closure when allocating
    //
    // Returns #f to indicate source location is not available at runtime.

    (void)arena;
    return make_false();
}

} // extern "C"
