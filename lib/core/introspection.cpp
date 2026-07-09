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
#include <eshkol/core/eval_bridge.h>
#include <eshkol/llvm_backend.h>
#include <eshkol/logger.h>
#include "arena_memory.h"

// NOTE: repl_jit.h is deliberately NOT included here. introspection.cpp
// lives in eshkol-static; ReplJITContext lives only in eshkol-repl-lib.
// JIT-dependent operations route through eshkol_eval_jit_* function
// pointers declared in eval_bridge.h. See lib/core/eval_bridge.cpp for
// the design rationale and lib/repl/eval_bridge_impl.cpp for the real
// implementations that override the weak defaults.

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <string>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <sstream>

namespace {

// ============================================================================
// Global State
// ============================================================================

// Gensym counter for unique symbol generation
std::atomic<uint64_t> g_gensym_counter{1};

// Eval JIT context acquired through the bridge. The pointer is opaque —
// only eshkol-repl-lib's strong implementation of eshkol_eval_jit_*
// knows how to dereference it. If that implementation isn't linked (i.e.
// this eshkol-static is in a user binary without the REPL runtime),
// acquire() returns nullptr and the caller emits a clear error instead
// of crashing on a dangling ReplJITContext call.
void* get_eval_jit() {
    eshkol_eval_jit_acquire_fn_t acquire = eshkol_eval_jit_get_acquire();
    if (!acquire) {
        eshkol_eval_jit_warn_missing("eval");
        return nullptr;
    }
    return acquire();
}

/* execute helper — funnel all jit->executeTagged sites through here so
 * the function pointer lookup happens in one place. If the getter
 * returns null (no REPL runtime linked) we emit a NULL tagged value
 * rather than crashing on a null function-pointer call. */
static inline eshkol_tagged_value_t jit_exec(void* jit, eshkol_ast_t* ast) {
    eshkol_eval_jit_execute_fn_t exec = eshkol_eval_jit_get_execute();
    if (!exec) {
        eshkol_tagged_value_t out;
        std::memset(&out, 0, sizeof(out));
        out.type = ESHKOL_VALUE_NULL;
        return out;
    }
    return exec(jit, ast);
}

/** Look up a compiled symbol's address in the JIT context, or 0 if no eval JIT is linked. */
static inline uint64_t jit_lookup(void* jit, const char* name) {
    eshkol_eval_jit_lookup_fn_t lookup = eshkol_eval_jit_get_lookup();
    return lookup ? lookup(jit, name) : 0;
}

/**
 * @brief Build the mangled JIT symbol name for a REPL variable's private storage slot.
 *
 * Prefixes with "__repl_storage_" and percent-encodes (as "_XX" hex) every
 * character of @p name that is not alphanumeric, so arbitrary Scheme
 * identifier characters produce a valid JIT symbol name.
 *
 * @param name Scheme variable name.
 * @return Mangled storage symbol name.
 */
static std::string repl_var_storage_symbol_name(const char* name) {
    std::string out = "__repl_storage_";
    char encoded[4];
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(name);
         p && *p;
         ++p) {
        unsigned char ch = *p;
        bool alnum = (ch >= '0' && ch <= '9') ||
                     (ch >= 'A' && ch <= 'Z') ||
                     (ch >= 'a' && ch <= 'z');
        if (alnum) {
            out.push_back(static_cast<char>(ch));
        } else {
            std::snprintf(encoded, sizeof(encoded), "_%02X", ch);
            out += encoded;
        }
    }
    return out;
}

/** Look up only the REPL private-storage symbol for @p name (no fallback), or 0 if not found. */
static uint64_t jit_lookup_repl_storage_only(void* jit, const char* name) {
    std::string storage_name = repl_var_storage_symbol_name(name);
    return jit_lookup(jit, storage_name.c_str());
}

/**
 * @brief Look up a REPL variable's storage address, checking private storage first.
 *
 * Tries the REPL private-storage symbol name first; if not found, falls
 * back to the plain symbol name (covers values `eval` captured via a
 * generated top-level define rather than REPL variable storage).
 *
 * @param jit Eval JIT context.
 * @param name Scheme variable name.
 * @return Storage address, or 0 if not found under either name.
 */
static uint64_t jit_lookup_repl_variable_storage(void* jit, const char* name) {
    uint64_t addr = jit_lookup_repl_storage_only(jit, name);
    if (addr != 0) return addr;

    // REPL hot-reload keeps user variables in a private storage namespace so
    // they can shadow functions/builtins. `eval` captures expression results
    // via a generated top-level define, so read back that storage symbol too.
    return jit_lookup(jit, name);
}

/**
 * @brief After evaluating a top-level (define name value), re-register the
 *        REPL variable's storage address so later lookups see the new value.
 *
 * No-op unless @p ast is a non-function ESHKOL_DEFINE_OP with a name and a
 * live JIT context, and its REPL storage symbol actually resolves.
 *
 * @param jit Eval JIT context.
 * @param ast The just-evaluated top-level AST node.
 */
static void refresh_repl_define_storage(void* jit, const eshkol_ast_t* ast) {
    if (!jit || !ast || ast->type != ESHKOL_OP ||
        ast->operation.op != ESHKOL_DEFINE_OP ||
        ast->operation.define_op.is_function ||
        !ast->operation.define_op.name) {
        return;
    }

    const char* name = ast->operation.define_op.name;
    uint64_t addr = jit_lookup_repl_storage_only(jit, name);
    if (addr == 0) {
        return;
    }

    eshkol_repl_register_symbol(name, addr);
    eshkol_repl_mark_user_variable(name);
}

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

// Create a false value
inline eshkol_tagged_value_t make_false() {
    eshkol_tagged_value_t value;
    std::memset(&value, 0, sizeof(value));
    value.type = ESHKOL_VALUE_BOOL;
    value.data.int_val = 0;
    return value;
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

// Check if value is a symbol.
//
// Symbols exist in two tagged-value encodings:
//   legacy:       type == ESHKOL_VALUE_SYMBOL, data.ptr_val -> raw char*
//   consolidated: type == ESHKOL_VALUE_HEAP_PTR with header.subtype == HEAP_SUBTYPE_SYMBOL
//
// New code produces the consolidated form so symbols participate in the
// unified heap-object protocol (headers, equal?, GC); legacy readers stay
// supported so cross-module boundaries (JIT, kb-load, quoted forms from
// older modules) continue to work.
inline bool is_symbol(eshkol_tagged_value_t value) {
    if (value.type == ESHKOL_VALUE_SYMBOL) return true;
    if (value.type == ESHKOL_VALUE_HEAP_PTR && value.data.ptr_val) {
        eshkol_object_header_t* hdr =
            ESHKOL_GET_HEADER(reinterpret_cast<void*>(value.data.ptr_val));
        return hdr && hdr->subtype == HEAP_SUBTYPE_SYMBOL;
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

/**
 * @brief Check if a value is a closure/procedure.
 *
 * @param value Tagged value to check.
 * @return true for any callable type (closure, primitive, continuation).
 */
bool eshkol_is_procedure(eshkol_tagged_value_t value) {
    return is_callable(value);
}

/**
 * @brief Check if a value is specifically a closure (has captures).
 *
 * @param value Tagged value to check.
 * @return true only for closures, not primitives or continuations.
 */
bool eshkol_is_closure(eshkol_tagged_value_t value) {
    return is_closure(value);
}

/**
 * @brief Get the arity (minimum required argument count) of a procedure.
 *
 * Closures and primitives report their declared input arity. Continuations
 * always report arity 1. Use eshkol_procedure_is_variadic() to check
 * whether additional arguments are also accepted.
 *
 * @param value Procedure to inspect.
 * @return Arity, or -1 if @p value is not a procedure.
 */
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

/**
 * @brief Check if a procedure is variadic (accepts variable arguments).
 *
 * Closures report their ESHKOL_CLOSURE_FLAG_VARIADIC flag; primitives
 * report their PRIMITIVE_IS_VARIADIC flag; continuations are never
 * variadic (they accept exactly one value).
 *
 * @param value Procedure to inspect.
 * @return true if variadic, false otherwise (including if not a procedure).
 */
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

/**
 * @brief Get the number of captured values in a closure.
 *
 * @param value Closure to inspect.
 * @return Number of captures, or 0 if @p value is not a closure or has no environment.
 */
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

/**
 * @brief Get a specific captured value from a closure by index.
 *
 * @param value Closure to inspect.
 * @param index Zero-based index of the capture.
 * @return The captured value, or a null tagged value if @p value is not a
 *         closure or @p index is out of range.
 */
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

/**
 * @brief Get all captured values from a closure as a freshly-consed list.
 *
 * @param value Closure to inspect.
 * @param arena Arena used to allocate the resulting cons cells.
 * @return List of captured values in capture order, or a null value if
 *         @p arena is NULL or the closure has no captures.
 */
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

/** Get the raw closure pointer from a tagged value, or NULL if @p value is not a closure. */
eshkol_closure_t* eshkol_get_closure(eshkol_tagged_value_t value) {
    if (!is_closure(value)) {
        return nullptr;
    }
    return static_cast<eshkol_closure_t*>(get_ptr(value));
}

// ----------------------------------------------------------------------------
// Symbol Manipulation API
// ----------------------------------------------------------------------------

/**
 * @brief Generate a unique (uninterned) symbol with the default "G" prefix.
 *
 * @param arena Arena for symbol string allocation.
 * @return New unique symbol as a tagged value (format G<counter>).
 */
eshkol_tagged_value_t eshkol_gensym(void* arena) {
    return eshkol_gensym_prefix("G", arena);
}

/**
 * @brief Generate a unique (uninterned) symbol with a caller-supplied prefix.
 *
 * Appends a process-wide monotonically increasing counter to @p prefix and
 * allocates the resulting string with a proper symbol object header (so
 * later ESHKOL_GET_HEADER dispatch correctly identifies it as
 * HEAP_SUBTYPE_SYMBOL) in the consolidated tagged-value encoding.
 *
 * @param prefix Prefix for the symbol name (defaults to "G" if NULL).
 * @param arena Arena for symbol string allocation.
 * @return New unique symbol as a tagged value, or a null value if @p arena is NULL or allocation fails.
 */
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

    // Allocate symbol string with a proper object header so ESHKOL_GET_HEADER
    // can identify the subtype. The previous headerless allocation made
    // ESHKOL_GET_HEADER read arena bookkeeping bytes as the header, producing
    // garbage subtype values and occasional crashes in introspection code.
    arena_t* a = static_cast<arena_t*>(arena);
    size_t len = strlen(buffer);
    char* sym_str = static_cast<char*>(
        arena_allocate_symbol_with_header(a, len)
    );

    if (!sym_str) {
        return make_null();
    }

    memcpy(sym_str, buffer, len + 1);

    // Gensym produces fresh (uninterned) symbols in the consolidated encoding
    // so header.subtype == HEAP_SUBTYPE_SYMBOL is authoritative for readers.
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_HEAP_PTR;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = reinterpret_cast<uint64_t>(sym_str);

    return result;
}

/**
 * @brief Convert a symbol to its string name.
 *
 * If @p arena is NULL, returns a string value that shares the symbol's
 * underlying character data directly rather than copying it. Otherwise
 * copies the name into a freshly arena-allocated, properly-headered string.
 *
 * @param symbol Symbol value to convert.
 * @param arena Arena for string allocation (see above), or NULL to alias the symbol's data.
 * @return String representation, or a null value if @p symbol is not a symbol.
 */
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

/* Forward declaration of the C ABI intern helper (symbol_intern.cpp). */
void* eshkol_intern_symbol_lookup(const char* name);

/**
 * @brief Wrap the result of an interned-symbol lookup as a tagged value.
 *
 * @param name Symbol name to intern/look up (may be NULL).
 * @param null_on_fail If true, a failed lookup yields a null tagged value; otherwise it yields #f.
 * @return Tagged HEAP_PTR symbol value on success, else the configured failure value.
 */
static eshkol_tagged_value_t wrap_interned(const char* name, bool null_on_fail) {
    void* ptr = name ? eshkol_intern_symbol_lookup(name) : nullptr;
    if (!ptr) {
        return null_on_fail ? make_null() : make_false();
    }
    eshkol_tagged_value_t tv;
    tv.type = ESHKOL_VALUE_HEAP_PTR;
    tv.flags = 0;
    tv.reserved = 0;
    tv.data.ptr_val = reinterpret_cast<uint64_t>(ptr);
    return tv;
}

/**
 * @brief Convert a string to an interned symbol, interning a new one if needed.
 *
 * @param str String value to convert.
 * @return Symbol with the given name, or a null value if @p str is not a string.
 */
eshkol_tagged_value_t eshkol_string_to_symbol(eshkol_tagged_value_t str) {
    if (!is_string(str)) {
        return make_null();
    }

    const char* str_value = static_cast<const char*>(get_ptr(str));
    return wrap_interned(str_value, /*null_on_fail=*/true);
}

/**
 * @brief Check if two symbols are equal.
 *
 * Interned symbols are compared by pointer identity first; falls back to
 * a string comparison so uninterned symbols (gensyms) with matching names
 * still compare equal.
 *
 * @param sym1 First symbol.
 * @param sym2 Second symbol.
 * @return true if both are symbols and equal, false otherwise.
 */
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

/**
 * @brief Get the S-expression representation of a procedure.
 *
 * For closures, returns the stored lambda S-expression captured at
 * creation time. For primitives, synthesizes a descriptive `(primitive)`
 * form (requires @p arena). Continuations and anonymous non-closure
 * callables have no representation and yield a null value.
 *
 * @param proc Procedure to serialize.
 * @param arena Arena for allocation (only needed for the primitive case).
 * @return S-expression representation, or a null value if unavailable.
 */
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

/**
 * @brief Get just the body of a closure's lambda, without the `(lambda (params...) ...)` wrapper.
 *
 * Obtains the closure's stored S-expression via eshkol_procedure_to_sexp()
 * and strips the leading `lambda` symbol and parameter list.
 *
 * @param closure Closure to inspect.
 * @param arena Arena for allocation.
 * @return Body S-expression (rest of the list after params), or a null value if not available.
 */
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

/**
 * @brief Get the parameter list of a closure's lambda form.
 *
 * Obtains the closure's stored S-expression via eshkol_procedure_to_sexp()
 * and returns the parameter list that follows the leading `lambda` symbol.
 *
 * @param closure Closure to inspect.
 * @param arena Arena for allocation.
 * @return Parameter list, or a null value if not available.
 */
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

// eshkol_intern_symbol_lookup lives in symbol_intern.cpp — codegen-emitted
// calls for symbol literals must not force introspection.cpp.o to be linked
// into user binaries (ReplJITContext resolves only in eshkol-repl-lib).

/** Intern @p name and wrap it as a tagged symbol value, yielding #f (not null) if interning fails. */
static eshkol_tagged_value_t intern_symbol_from_cstr(const char* name) {
    return wrap_interned(name, /*null_on_fail=*/false);
}

/**
 * @brief Get the bound name of a procedure, if it has one.
 *
 * Closures report the name field set by (define name ...); primitives
 * report their builtin name. Anonymous lambdas and continuations have no
 * name.
 *
 * @param proc Procedure to inspect.
 * @return Interned symbol name, or #f if anonymous or not a procedure.
 */
eshkol_tagged_value_t eshkol_procedure_name(eshkol_tagged_value_t proc) {
    if (!is_callable(proc)) {
        return make_false();
    }

    // Check for closure - closures have a name field set when defined via (define name ...)
    if (is_closure(proc)) {
        eshkol_closure_t* closure = eshkol_get_closure(proc);
        if (closure && closure->name) {
            return intern_symbol_from_cstr(closure->name);
        }
    }

    // Check for primitive - primitives have a name field
    if (is_primitive(proc)) {
        eshkol_primitive_t* prim = get_primitive(proc);
        if (prim && prim->name) {
            return intern_symbol_from_cstr(prim->name);
        }
    }

    // Anonymous lambdas and continuations without names return #f
    return make_false();
}

// ----------------------------------------------------------------------------
// Runtime Evaluation API
// ----------------------------------------------------------------------------

// Per-process counter for unique eval-result global names. Each eshkol_eval
// call wraps its AST in a (define __eshkol_eval_result_<N>__ <expr>) so the
// JIT entry function (which otherwise discards expression values as exit-code-
// only i32 returns) stores the value in a named global we can read back.
static std::atomic<uint64_t> g_eval_result_counter{0};

/**
 * @brief Evaluate an S-expression at runtime using the JIT compiler.
 *
 * Top-level define/import/require/provide forms are executed for effect
 * and yield a null value. Any other expression is wrapped in a synthetic
 * `(define __eshkol_eval_result_N__ <expr>)` so its value can be read back
 * from the named global afterwards (the JIT entry point otherwise only
 * exposes an i32 exit code, not the expression's actual value).
 *
 * @param sexp S-expression to evaluate.
 * @param arena Unused (reserved for future allocation needs).
 * @return Result of evaluation, or a null value on conversion/JIT failure.
 */
eshkol_tagged_value_t eshkol_eval(eshkol_tagged_value_t sexp, void* arena) {
    (void)arena;

    // Direct S-expression to AST conversion (O(n) single-pass)
    eshkol_ast_t* inner = eshkol_sexp_to_ast(sexp);
    if (!inner) {
        eshkol_error("eshkol_eval: Failed to convert S-expression to AST");
        return make_null();
    }

    void* jit = get_eval_jit();
    if (!jit) {
        eshkol_free_sexp_ast(inner);
        eshkol_error("eshkol_eval: Failed to initialize JIT context");
        return make_null();
    }

    // If the user expression itself is a top-level DEFINE (e.g., eval'ing
    // `(define foo ...)`) just execute it as-is and return null — there's no
    // expression value to capture. Same for imports/requires.
    if (inner->type == ESHKOL_OP &&
        (inner->operation.op == ESHKOL_DEFINE_OP ||
         inner->operation.op == ESHKOL_IMPORT_OP ||
         inner->operation.op == ESHKOL_REQUIRE_OP ||
         inner->operation.op == ESHKOL_PROVIDE_OP)) {
        jit_exec(jit, inner);
        eshkol_free_sexp_ast(inner);
        return make_null();
    }

    // Wrap in (define __eshkol_eval_result_N__ <expr>) so the JIT main writes
    // the expression value into a named global we can read back afterwards.
    // The JIT's current entry function returns i32 exit code; without the
    // wrap we'd always see 0 regardless of the real expression value.
    uint64_t id = g_eval_result_counter.fetch_add(1);
    char name_buf[64];
    snprintf(name_buf, sizeof(name_buf), "__eshkol_eval_result_%llu__",
             (unsigned long long)id);

    eshkol_ast_t* wrapper = eshkol_alloc_symbolic_ast();
    if (!wrapper) {
        eshkol_free_sexp_ast(inner);
        return make_null();
    }
    wrapper->type = ESHKOL_OP;
    wrapper->operation.op = ESHKOL_DEFINE_OP;
    wrapper->operation.define_op.name = strdup(name_buf);
    wrapper->operation.define_op.value = inner;
    wrapper->operation.define_op.is_function = 0;
    wrapper->operation.define_op.parameters = nullptr;
    wrapper->operation.define_op.num_params = 0;
    wrapper->operation.define_op.is_variadic = 0;
    wrapper->operation.define_op.rest_param = nullptr;
    wrapper->operation.define_op.is_external = 0;
    wrapper->operation.define_op.return_type = nullptr;
    wrapper->operation.define_op.param_types = nullptr;

    // Execute — sets the global to the expression value as a side effect.
    jit_exec(jit, wrapper);

    // Read back the value from the named global's memory.
    uint64_t addr = jit_lookup_repl_variable_storage(jit, name_buf);
    eshkol_tagged_value_t result;
    memset(&result, 0, sizeof(result));
    if (addr != 0) {
        result = *reinterpret_cast<eshkol_tagged_value_t*>(addr);
    } else {
        result.type = ESHKOL_VALUE_NULL;
    }


    // wrapper owns `inner` via define_op.value — freeing wrapper frees the tree.
    // define_op.name was strdup'd; free_sexp_ast should handle it, but play safe.
    eshkol_free_sexp_ast(wrapper);

    return result;
}

// SRET wrapper — used by LLVM JIT-compiled call sites to avoid struct-return
// ABI mismatches between JIT-emitted IR and the C runtime. Passing pointers
// on both sides sidesteps the aggregate-return calling convention entirely.
extern "C" void eshkol_eval_sret(eshkol_tagged_value_t* result,
                                  const eshkol_tagged_value_t* sexp,
                                  void* arena) {
    if (!result) return;
    if (!sexp) { *result = make_null(); return; }
    *result = eshkol_eval(*sexp, arena);
}

/** SRET wrapper for eshkol_eval_env(), avoiding struct-return ABI mismatches at JIT call sites. */
extern "C" void eshkol_eval_env_sret(eshkol_tagged_value_t* result,
                                      const eshkol_tagged_value_t* sexp,
                                      const eshkol_tagged_value_t* env,
                                      void* arena) {
    if (!result) return;
    if (!sexp || !env) { *result = make_null(); return; }
    *result = eshkol_eval_env(*sexp, *env, arena);
}

/**
 * @brief Evaluate an S-expression with a custom environment.
 *
 * The environment is an association list mapping symbols to values. If
 * non-empty, the converted AST is wrapped in an ESHKOL_LET_OP binding each
 * alist entry before being executed via the JIT.
 *
 * @param sexp S-expression to evaluate.
 * @param env Environment (alist of (symbol . value) pairs), or a null value for no bindings.
 * @param arena Unused (reserved for future allocation needs).
 * @return Result of evaluation, or a null value on conversion/JIT failure.
 */
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
                arena_allocate(get_global_arena(),num_bindings * 2 * sizeof(eshkol_ast_t)));

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
                            eshkol_ast_make_null(&bindings[i * 2 + 1]);
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
    void* jit = get_eval_jit();
    if (!jit) {
        eshkol_free_sexp_ast(final_ast);
        eshkol_error("eshkol_eval_env: Failed to initialize JIT context");
        return make_null();
    }

    // Execute the AST through the JIT
    eshkol_tagged_value_t result = jit_exec(jit, final_ast);

    // Free the AST
    eshkol_free_sexp_ast(final_ast);

    return result;
}

/**
 * @brief Compile an S-expression to a procedure without executing it as a top-level call.
 *
 * The S-expression is typically a lambda form; the JIT executes it once to
 * produce the closure value itself (evaluating a bare lambda yields the
 * closure object, not a call result).
 *
 * @param sexp Lambda S-expression to compile.
 * @param arena Unused (reserved for future allocation needs).
 * @return Compiled procedure (closure) value, or a null value on error.
 */
eshkol_tagged_value_t eshkol_compile(eshkol_tagged_value_t sexp, void* arena) {
    // Direct S-expression to AST conversion (O(n) single-pass)
    // Avoids serialize→parse overhead
    eshkol_ast_t* ast = eshkol_sexp_to_ast(sexp);
    if (!ast) {
        eshkol_error("eshkol_compile: Failed to convert S-expression to AST");
        return make_null();
    }

    // Get or create the eval JIT context
    void* jit = get_eval_jit();
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
    eshkol_tagged_value_t result = jit_exec(jit, ast);

    // Free the AST (JIT may have taken ownership of parts, but we allocated the root)
    eshkol_free_sexp_ast(ast);

    (void)arena;  // Arena passed for potential allocation needs
    return result;
}

/**
 * @brief Convert an S-expression to a procedure, validating it is a lambda form first.
 *
 * Thin wrapper around eshkol_compile() that rejects anything not headed by
 * the `lambda` symbol.
 *
 * @param sexp Lambda S-expression.
 * @param arena Arena forwarded to eshkol_compile().
 * @return Compiled procedure, or a null value if @p sexp is not a lambda form.
 */
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

/**
 * @brief Compile an S-expression with environment bindings available as free variables.
 *
 * Converts @p sexp to AST, and if @p env has bindings, wraps it in an
 * ESHKOL_LET_OP binding each (name, value) pair — each value is converted
 * to the matching literal AST node type (int64/double/bool/char/null/
 * string); complex runtime values (closures, cons cells, etc.) cannot
 * currently be embedded this way and cause the call to fail. Executes the
 * resulting AST through the JIT to compile the closure.
 *
 * @param sexp The S-expression to compile.
 * @param env Environment bindings (may be NULL for no bindings).
 * @param arena Unused (reserved for future allocation needs).
 * @return Compiled procedure, or a null value on error (including an unembeddable complex-typed binding).
 */
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
            arena_allocate(get_global_arena(),env->count * 2 * sizeof(eshkol_ast_t)));

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
                eshkol_ast_make_string(&val_ast, strdup(str ? str : ""), str ? strlen(str) : 0);
            } else {
                switch (val.type) {
                    case ESHKOL_VALUE_INT64:
                        eshkol_ast_make_int64(&val_ast, val.data.int_val);
                        break;
                    case ESHKOL_VALUE_DOUBLE:
                        eshkol_ast_make_double(&val_ast, val.data.double_val);
                        break;
                    case ESHKOL_VALUE_BOOL:
                        eshkol_ast_make_bool(&val_ast, val.data.int_val != 0);
                        break;
                    case ESHKOL_VALUE_CHAR:
                        eshkol_ast_make_char(&val_ast, val.data.int_val);
                        break;
                    case ESHKOL_VALUE_NULL:
                        eshkol_ast_make_null(&val_ast);
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
    void* jit = get_eval_jit();
    if (!jit) {
        eshkol_free_sexp_ast(final_ast);
        eshkol_error("eshkol_compile_with_env: Failed to initialize JIT context");
        return make_null();
    }

    // Execute the expression through the JIT to compile it
    eshkol_tagged_value_t result = jit_exec(jit, final_ast);

    // Free the AST
    eshkol_free_sexp_ast(final_ast);

    return result;
}

/**
 * @brief Parse and evaluate a string as Eshkol code.
 *
 * Parses a single expression from @p str and executes it through the JIT,
 * refreshing REPL variable storage bookkeeping for a top-level define.
 *
 * @param str String containing Eshkol code.
 * @param arena Unused (reserved for future allocation needs).
 * @return Result of evaluation, or a null value if @p str is empty, fails to parse, or the JIT is unavailable.
 */
eshkol_tagged_value_t eshkol_eval_string(const char* str, void* arena) {
    if (!str || !*str) {
        return make_null();
    }

    // Get or create the eval JIT context
    void* jit = get_eval_jit();
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
    eshkol_tagged_value_t result = jit_exec(jit, &ast);
    refresh_repl_define_storage(jit, &ast);

    (void)arena;  // Arena passed for potential allocation needs
    return result;
}

// ----------------------------------------------------------------------------
// Reflection API
// ----------------------------------------------------------------------------

/**
 * @brief Get the type of a value as an interned symbol.
 *
 * Dispatches on the tagged value's immediate type, and for HEAP_PTR /
 * CALLABLE values further dispatches on the object header's subtype to
 * distinguish pairs, strings, vectors, tensors, hash tables, bignums,
 * rationals, closures, primitives, continuations, etc. Legacy pointer
 * type tags are recognized for backward compatibility.
 *
 * @param value Value to inspect.
 * @return Interned type-name symbol (e.g. 'integer, 'string, 'closure, 'unknown).
 */
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
                        case HEAP_SUBTYPE_SYMBOL:
                            type_name = "symbol";
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
                        case HEAP_SUBTYPE_BIGNUM:
                            type_name = "integer";
                            break;
                        case HEAP_SUBTYPE_BYTEVECTOR:
                            type_name = "bytevector";
                            break;
                        case HEAP_SUBTYPE_RATIONAL:
                            type_name = "rational";
                            break;
                        case HEAP_SUBTYPE_SUBSTITUTION:
                            type_name = "substitution";
                            break;
                        case HEAP_SUBTYPE_FACT:
                            type_name = "fact";
                            break;
                        case HEAP_SUBTYPE_KNOWLEDGE_BASE:
                            type_name = "knowledge-base";
                            break;
                        case HEAP_SUBTYPE_FACTOR_GRAPH:
                            type_name = "factor-graph";
                            break;
                        case HEAP_SUBTYPE_WORKSPACE:
                            type_name = "workspace";
                            break;
                        default:
                            eshkol_warn("unknown heap subtype: %d", header->subtype);
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
                            eshkol_warn("unknown callable subtype: %d", header->subtype);
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

    // Return as interned symbol in the consolidated HEAP_PTR encoding. The raw
    // type_name pointer here is a static string literal — tagging it as a
    // symbol directly would leave ESHKOL_GET_HEADER reading program data, so
    // we route through the symbol-intern path which allocates a proper symbol
    // header and ensures (eq? (type-of x) 'string) works.
    return intern_symbol_from_cstr(type_name);
}

/**
 * @brief Get source location information for a procedure.
 *
 * Always returns #f: source locations are tracked at compile-time in the
 * AST but are not currently preserved in runtime closure structures (see
 * the in-body comment for what would be needed to support this).
 *
 * @param proc Procedure to inspect.
 * @param arena Unused (reserved for future allocation needs).
 * @return #f (not currently available at runtime).
 */
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
