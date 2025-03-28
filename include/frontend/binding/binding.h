/**
 * @file binding.h
 * @brief Binding system for Eshkol
 * 
 * This file defines the binding system for the Eshkol language, which tracks
 * variable bindings and scopes.
 */

#ifndef ESHKOL_BINDING_H
#define ESHKOL_BINDING_H

#include "core/memory.h"
#include "core/string_table.h"
#include "core/diagnostics.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief String identifier type
 * 
 * This is a pointer to an interned string in the string table.
 */
typedef const char* StringId;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Binding table structure
 */
typedef struct {
    StringId* names;           /**< Binding names */
    uint64_t* binding_ids;     /**< Binding IDs */
    uint64_t* scope_ids;       /**< Scope IDs where bindings are defined */
    bool* is_mutable;          /**< Whether the binding is mutable */
    size_t count;              /**< Number of bindings */
    size_t capacity;           /**< Capacity of the binding table */
} BindingTable;

/**
 * @brief Scope stack structure
 */
typedef struct {
    uint64_t* ids;             /**< Scope IDs */
    uint64_t* parent_ids;      /**< Parent scope IDs */
    size_t count;              /**< Number of scopes */
    size_t capacity;           /**< Capacity of the scope stack */
} ScopeStack;

/**
 * @brief Binding system structure
 */
typedef struct {
    Arena* arena;              /**< Arena allocator */
    DiagnosticContext* diag;   /**< Diagnostic context */
    BindingTable binding_table; /**< Binding table */
    ScopeStack scope_stack;    /**< Scope stack */
    uint64_t current_scope_id; /**< Current scope ID */
    uint64_t next_binding_id;  /**< Next binding ID to assign */
    uint64_t next_scope_id;    /**< Next scope ID to assign */
} BindingSystem;

/**
 * @brief Create a new binding system
 * 
 * @param arena Arena allocator
 * @param diag Diagnostic context
 * @return A new binding system, or NULL on failure
 */
BindingSystem* binding_system_create(Arena* arena, DiagnosticContext* diag);

/**
 * @brief Enter a new scope
 * 
 * @param system The binding system
 * @return The ID of the new scope, or 0 on failure
 */
uint64_t binding_system_enter_scope(BindingSystem* system);

/**
 * @brief Exit the current scope
 * 
 * @param system The binding system
 * @return true on success, false on failure
 */
bool binding_system_exit_scope(BindingSystem* system);

/**
 * @brief Add a binding to the current scope
 * 
 * @param system The binding system
 * @param name The name of the binding
 * @param is_mutable Whether the binding is mutable
 * @return The ID of the new binding, or 0 on failure
 */
uint64_t binding_system_add_binding(BindingSystem* system, StringId name, bool is_mutable);

/**
 * @brief Resolve a binding in the current scope or any parent scope
 * 
 * @param system The binding system
 * @param name The name to resolve
 * @return The ID of the binding, or 0 if not found
 */
uint64_t binding_system_resolve_binding(BindingSystem* system, StringId name);

/**
 * @brief Get the name of a binding by ID
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return The name of the binding, or NULL if not found
 */
StringId binding_system_get_binding_name(BindingSystem* system, uint64_t binding_id);

/**
 * @brief Get the scope of a binding by ID
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return The scope ID of the binding, or 0 if not found
 */
uint64_t binding_system_get_binding_scope(BindingSystem* system, uint64_t binding_id);

/**
 * @brief Check if a binding is mutable
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return true if the binding is mutable, false otherwise
 */
bool binding_system_is_binding_mutable(BindingSystem* system, uint64_t binding_id);

/**
 * @brief Check if a binding is in the current scope
 * 
 * @param system The binding system
 * @param name The name to check
 * @return true if the binding is in the current scope, false otherwise
 */
bool binding_system_binding_in_current_scope(BindingSystem* system, StringId name);

/**
 * @brief Check if a scope is a descendant of another scope
 * 
 * @param system The binding system
 * @param scope_id The scope ID to check
 * @param ancestor_id The potential ancestor scope ID
 * @return true if scope_id is a descendant of ancestor_id, false otherwise
 */
bool binding_system_is_descendant_scope(BindingSystem* system, uint64_t scope_id, uint64_t ancestor_id);

/**
 * @brief Get the parent scope of a scope
 * 
 * @param system The binding system
 * @param scope_id The scope ID
 * @return The parent scope ID, or 0 if not found or if the scope is the global scope
 */
uint64_t binding_system_get_parent_scope(BindingSystem* system, uint64_t scope_id);

/**
 * @brief Get the current scope ID
 * 
 * @param system The binding system
 * @return The current scope ID
 */
uint64_t binding_system_get_current_scope(BindingSystem* system);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_BINDING_H */
