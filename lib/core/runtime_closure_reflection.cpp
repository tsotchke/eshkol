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

extern "C" int64_t eshkol_closure_get_arity(const eshkol_closure_t* closure) {
    if (!closure) return -1;
    return (int64_t)closure->input_arity;
}

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
