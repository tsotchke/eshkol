/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Runtime closure reflection and lambda registry support.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"

#include <cstdlib>
#include <cstring>

// ===== CLOSURE REFLECTION =====
// Reflection helpers for procedure-arity / procedure-name / etc.
//
// The closure struct (inc/eshkol/eshkol.h) already records input_arity
// (as part of return_type_info packed by codegen) and a CLOSURE_FLAG_VARIADIC
// flag. These helpers expose those fields to user code.

/**
 * @brief Return a closure's declared argument arity (`procedure-arity` support).
 * @param closure  Closure to inspect.
 * @return         Number of expected arguments (0-255), or -1 if `closure` is null.
 */
extern "C" int64_t eshkol_closure_get_arity(const eshkol_closure_t* closure) {
    if (!closure) return -1;
    return (int64_t)closure->input_arity;
}

/**
 * @brief Test whether a closure accepts variadic arguments.
 * @param closure  Closure to inspect.
 * @return         Nonzero if CLOSURE_FLAG_VARIADIC is set; 0 if not set or `closure` is null.
 */
extern "C" int eshkol_closure_is_variadic_fn(const eshkol_closure_t* closure) {
    if (!closure) return 0;
    return (closure->flags & CLOSURE_FLAG_VARIADIC) ? 1 : 0;
}

// Returns a fresh arena-allocated Eshkol string (with subtype header) so
// the result can be passed to display/write/string-* directly. Returning
// the raw closure->name pointer would print as "#<heap:NN>" since display
// expects HEAP_PTR with a HEAP_SUBTYPE_STRING object header at ptr-8.
extern "C" char* eshkol_closure_get_name(const eshkol_closure_t* closure) {
    arena_t* arena = get_global_arena();
    const char* src = (closure && closure->name) ? closure->name : "";
    size_t len = strlen(src);
    char* dst = arena_allocate_string_with_header(arena, len);
    if (!dst) return nullptr;
    memcpy(dst, src, len);
    dst[len] = '\0';
    return dst;
}

// ===== LAMBDA REGISTRY IMPLEMENTATION =====
// Runtime table for mapping function pointers to S-expressions (homoiconicity)

eshkol_lambda_registry_t* g_lambda_registry = nullptr;

/**
 * @brief Lazily initialize the global lambda registry (`g_lambda_registry`).
 *
 * Idempotent: returns immediately if already initialized. Allocates the
 * registry struct and an initial 64-entry table via malloc (not the arena),
 * since the registry is process-lifetime state independent of any one arena's
 * scope. Logs and leaves the registry null on allocation failure so callers
 * degrade gracefully (lookups simply miss) instead of crashing.
 */
void eshkol_lambda_registry_init(void) {
    if (g_lambda_registry) {
        return;  // Already initialized
    }

    g_lambda_registry = (eshkol_lambda_registry_t*)malloc(sizeof(eshkol_lambda_registry_t));
    if (!g_lambda_registry) {
        eshkol_error("Failed to allocate lambda registry");
        return;
    }

    g_lambda_registry->capacity = 64;  // Initial capacity
    g_lambda_registry->count = 0;
    g_lambda_registry->entries = (eshkol_lambda_entry_t*)malloc(
        sizeof(eshkol_lambda_entry_t) * g_lambda_registry->capacity);

    if (!g_lambda_registry->entries) {
        eshkol_error("Failed to allocate lambda registry entries");
        free(g_lambda_registry);
        g_lambda_registry = nullptr;
        return;
    }

    eshkol_debug("Lambda registry initialized with capacity %zu", g_lambda_registry->capacity);
}

/**
 * @brief Free the global lambda registry and its entry table, resetting `g_lambda_registry` to null.
 *
 * Safe to call when the registry was never initialized (no-op).
 */
void eshkol_lambda_registry_destroy(void) {
    if (!g_lambda_registry) {
        return;
    }

    if (g_lambda_registry->entries) {
        free(g_lambda_registry->entries);
    }
    free(g_lambda_registry);
    g_lambda_registry = nullptr;
}

/**
 * @brief Register (or update) the S-expression associated with a lambda's function pointer.
 *
 * Supports Eshkol's homoiconicity: displaying a closure/lambda looks up its
 * source S-expression here when the closure struct doesn't already carry an
 * embedded `sexpr_ptr`. Lazily initializes the registry if needed. If
 * `func_ptr` is already registered, updates its `sexpr_ptr`/`name` in place;
 * otherwise grows the entry table (doubling capacity via realloc) and appends
 * a new entry. Silently no-ops if the registry could not be allocated or grown.
 *
 * @param func_ptr    Address of the compiled lambda function (registry key).
 * @param sexpr_ptr   Pointer (as a tagged cons-list address) to the lambda's source S-expression.
 * @param name        Bound name for debug logging, or null/"(anonymous)" for anonymous lambdas.
 */
void eshkol_lambda_registry_add(uint64_t func_ptr, uint64_t sexpr_ptr, const char* name) {
    if (!g_lambda_registry) {
        eshkol_lambda_registry_init();
    }

    if (!g_lambda_registry) {
        return;  // Init failed
    }

    // Check if already registered (update if so)
    for (size_t i = 0; i < g_lambda_registry->count; i++) {
        if (g_lambda_registry->entries[i].func_ptr == func_ptr) {
            g_lambda_registry->entries[i].sexpr_ptr = sexpr_ptr;
            g_lambda_registry->entries[i].name = name;
            eshkol_debug("Updated lambda registry entry for %s at %p -> sexpr %p",
                        name ? name : "(anonymous)", (void*)func_ptr, (void*)sexpr_ptr);
            return;
        }
    }

    // Grow if needed
    if (g_lambda_registry->count >= g_lambda_registry->capacity) {
        size_t new_capacity = g_lambda_registry->capacity * 2;
        eshkol_lambda_entry_t* new_entries = (eshkol_lambda_entry_t*)realloc(
            g_lambda_registry->entries,
            sizeof(eshkol_lambda_entry_t) * new_capacity);

        if (!new_entries) {
            eshkol_error("Failed to grow lambda registry");
            return;
        }

        g_lambda_registry->entries = new_entries;
        g_lambda_registry->capacity = new_capacity;
    }

    // Add new entry
    size_t idx = g_lambda_registry->count++;
    g_lambda_registry->entries[idx].func_ptr = func_ptr;
    g_lambda_registry->entries[idx].sexpr_ptr = sexpr_ptr;
    g_lambda_registry->entries[idx].name = name;

    eshkol_debug("Lambda registry: added %s func=%p sexpr=%p",
                name ? name : "(anon)", (void*)func_ptr, (void*)sexpr_ptr);
}

/**
 * @brief Look up the S-expression registered for a lambda's function pointer.
 * @param func_ptr  Address of the compiled lambda function.
 * @return          The registered S-expression pointer, or 0 if not found or the registry is uninitialized.
 */
uint64_t eshkol_lambda_registry_lookup(uint64_t func_ptr) {
    if (!g_lambda_registry) {
        return 0;
    }

    for (size_t i = 0; i < g_lambda_registry->count; i++) {
        if (g_lambda_registry->entries[i].func_ptr == func_ptr) {
            return g_lambda_registry->entries[i].sexpr_ptr;
        }
    }

    return 0;  // Not found
}

// ===== END LAMBDA REGISTRY IMPLEMENTATION =====
