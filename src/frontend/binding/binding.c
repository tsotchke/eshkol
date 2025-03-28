/**
 * @file binding.c
 * @brief Implementation of the binding system for Eshkol
 */

#include "frontend/binding/binding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Create a new binding system
 * 
 * @param arena Arena allocator
 * @param diag Diagnostic context
 * @return A new binding system, or NULL on failure
 */
BindingSystem* binding_system_create(Arena* arena, DiagnosticContext* diag) {
    assert(arena != NULL);
    assert(diag != NULL);
    
    // Allocate binding system
    BindingSystem* system = arena_alloc(arena, sizeof(BindingSystem));
    if (!system) {
        return NULL;
    }
    
    // Initialize binding system
    system->arena = arena;
    system->diag = diag;
    
    // Initialize binding table
    system->binding_table.names = NULL;
    system->binding_table.binding_ids = NULL;
    system->binding_table.scope_ids = NULL;
    system->binding_table.is_mutable = NULL;
    system->binding_table.count = 0;
    system->binding_table.capacity = 0;
    
    // Initialize scope stack
    system->scope_stack.ids = NULL;
    system->scope_stack.parent_ids = NULL;
    system->scope_stack.count = 0;
    system->scope_stack.capacity = 0;
    
    // Initialize IDs
    system->current_scope_id = 0;
    system->next_binding_id = 1; // Start at 1, 0 is reserved for "not found"
    system->next_scope_id = 1;   // Start at 1, 0 is reserved for "global scope"
    
    // Create global scope
    if (!binding_system_enter_scope(system)) {
        return NULL;
    }
    
    return system;
}

/**
 * @brief Enter a new scope
 * 
 * @param system The binding system
 * @return The ID of the new scope, or 0 on failure
 */
uint64_t binding_system_enter_scope(BindingSystem* system) {
    assert(system != NULL);
    
    // Check if we need to resize the scope stack
    if (system->scope_stack.count >= system->scope_stack.capacity) {
        // Calculate new capacity
        size_t new_capacity = system->scope_stack.capacity == 0 ? 8 : system->scope_stack.capacity * 2;
        
        // Allocate new arrays
        uint64_t* new_ids = arena_alloc(system->arena, sizeof(uint64_t) * new_capacity);
        uint64_t* new_parent_ids = arena_alloc(system->arena, sizeof(uint64_t) * new_capacity);
        
        if (!new_ids || !new_parent_ids) {
            return 0;
        }
        
        // Copy old data
        for (size_t i = 0; i < system->scope_stack.count; i++) {
            new_ids[i] = system->scope_stack.ids[i];
            new_parent_ids[i] = system->scope_stack.parent_ids[i];
        }
        
        // Update scope stack
        system->scope_stack.ids = new_ids;
        system->scope_stack.parent_ids = new_parent_ids;
        system->scope_stack.capacity = new_capacity;
    }
    
    // Create new scope
    uint64_t scope_id = system->next_scope_id++;
    uint64_t parent_id = system->current_scope_id;
    
    // Add to scope stack
    system->scope_stack.ids[system->scope_stack.count] = scope_id;
    system->scope_stack.parent_ids[system->scope_stack.count] = parent_id;
    system->scope_stack.count++;
    
    // Update current scope
    system->current_scope_id = scope_id;
    
    return scope_id;
}

/**
 * @brief Exit the current scope
 * 
 * @param system The binding system
 * @return true on success, false on failure
 */
bool binding_system_exit_scope(BindingSystem* system) {
    assert(system != NULL);
    
    // Check if we're in the global scope
    if (system->scope_stack.count <= 1) {
        // Can't exit global scope
        return false;
    }
    
    // Get parent scope
    uint64_t parent_id = system->scope_stack.parent_ids[system->scope_stack.count - 1];
    
    // Update current scope
    system->current_scope_id = parent_id;
    
    // Remove from scope stack
    system->scope_stack.count--;
    
    return true;
}

/**
 * @brief Add a binding to the current scope
 * 
 * @param system The binding system
 * @param name The name of the binding
 * @param is_mutable Whether the binding is mutable
 * @return The ID of the new binding, or 0 on failure
 */
uint64_t binding_system_add_binding(BindingSystem* system, StringId name, bool is_mutable) {
    assert(system != NULL);
    assert(name != NULL);
    
    // Check if we need to resize the binding table
    if (system->binding_table.count >= system->binding_table.capacity) {
        // Calculate new capacity
        size_t new_capacity = system->binding_table.capacity == 0 ? 16 : system->binding_table.capacity * 2;
        
        // Allocate new arrays
        StringId* new_names = arena_alloc(system->arena, sizeof(StringId) * new_capacity);
        uint64_t* new_binding_ids = arena_alloc(system->arena, sizeof(uint64_t) * new_capacity);
        uint64_t* new_scope_ids = arena_alloc(system->arena, sizeof(uint64_t) * new_capacity);
        bool* new_is_mutable = arena_alloc(system->arena, sizeof(bool) * new_capacity);
        
        if (!new_names || !new_binding_ids || !new_scope_ids || !new_is_mutable) {
            return 0;
        }
        
        // Copy old data
        for (size_t i = 0; i < system->binding_table.count; i++) {
            new_names[i] = system->binding_table.names[i];
            new_binding_ids[i] = system->binding_table.binding_ids[i];
            new_scope_ids[i] = system->binding_table.scope_ids[i];
            new_is_mutable[i] = system->binding_table.is_mutable[i];
        }
        
        // Update binding table
        system->binding_table.names = new_names;
        system->binding_table.binding_ids = new_binding_ids;
        system->binding_table.scope_ids = new_scope_ids;
        system->binding_table.is_mutable = new_is_mutable;
        system->binding_table.capacity = new_capacity;
    }
    
    // Check if the binding already exists in the current scope
    for (size_t i = 0; i < system->binding_table.count; i++) {
        if (system->binding_table.scope_ids[i] == system->current_scope_id &&
            strcmp(system->binding_table.names[i], name) == 0) {
            // Binding already exists in the current scope
            return system->binding_table.binding_ids[i];
        }
    }
    
    // Create new binding
    uint64_t binding_id = system->next_binding_id++;
    
    // Add to binding table
    system->binding_table.names[system->binding_table.count] = name;
    system->binding_table.binding_ids[system->binding_table.count] = binding_id;
    system->binding_table.scope_ids[system->binding_table.count] = system->current_scope_id;
    system->binding_table.is_mutable[system->binding_table.count] = is_mutable;
    system->binding_table.count++;
    
    return binding_id;
}

/**
 * @brief Resolve a binding in the current scope or any parent scope
 * 
 * @param system The binding system
 * @param name The name to resolve
 * @return The ID of the binding, or 0 if not found
 */
uint64_t binding_system_resolve_binding(BindingSystem* system, StringId name) {
    assert(system != NULL);
    assert(name != NULL);
    
    // Get current scope
    uint64_t scope_id = system->current_scope_id;
    
    // Search for binding in current scope and parent scopes
    while (scope_id != 0) {
        // Search for binding in current scope
        for (size_t i = 0; i < system->binding_table.count; i++) {
            if (system->binding_table.scope_ids[i] == scope_id &&
                strcmp(system->binding_table.names[i], name) == 0) {
                // Found binding
                return system->binding_table.binding_ids[i];
            }
        }
        
        // Get parent scope
        scope_id = binding_system_get_parent_scope(system, scope_id);
    }
    
    // Binding not found
    return 0;
}

/**
 * @brief Get the name of a binding by ID
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return The name of the binding, or NULL if not found
 */
StringId binding_system_get_binding_name(BindingSystem* system, uint64_t binding_id) {
    assert(system != NULL);
    
    // Search for binding
    for (size_t i = 0; i < system->binding_table.count; i++) {
        if (system->binding_table.binding_ids[i] == binding_id) {
            // Found binding
            return system->binding_table.names[i];
        }
    }
    
    // Binding not found
    return NULL;
}

/**
 * @brief Get the scope of a binding by ID
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return The scope ID of the binding, or 0 if not found
 */
uint64_t binding_system_get_binding_scope(BindingSystem* system, uint64_t binding_id) {
    assert(system != NULL);
    
    // Search for binding
    for (size_t i = 0; i < system->binding_table.count; i++) {
        if (system->binding_table.binding_ids[i] == binding_id) {
            // Found binding
            return system->binding_table.scope_ids[i];
        }
    }
    
    // Binding not found
    return 0;
}

/**
 * @brief Check if a binding is mutable
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return true if the binding is mutable, false otherwise
 */
bool binding_system_is_binding_mutable(BindingSystem* system, uint64_t binding_id) {
    assert(system != NULL);
    
    // Search for binding
    for (size_t i = 0; i < system->binding_table.count; i++) {
        if (system->binding_table.binding_ids[i] == binding_id) {
            // Found binding
            return system->binding_table.is_mutable[i];
        }
    }
    
    // Binding not found
    return false;
}

/**
 * @brief Check if a binding is in the current scope
 * 
 * @param system The binding system
 * @param name The name to check
 * @return true if the binding is in the current scope, false otherwise
 */
bool binding_system_binding_in_current_scope(BindingSystem* system, StringId name) {
    assert(system != NULL);
    assert(name != NULL);
    
    // Search for binding in current scope
    for (size_t i = 0; i < system->binding_table.count; i++) {
        if (system->binding_table.scope_ids[i] == system->current_scope_id &&
            strcmp(system->binding_table.names[i], name) == 0) {
            // Found binding
            return true;
        }
    }
    
    // Binding not found in current scope
    return false;
}

/**
 * @brief Check if a scope is a descendant of another scope
 * 
 * @param system The binding system
 * @param scope_id The scope ID to check
 * @param ancestor_id The potential ancestor scope ID
 * @return true if scope_id is a descendant of ancestor_id, false otherwise
 */
bool binding_system_is_descendant_scope(BindingSystem* system, uint64_t scope_id, uint64_t ancestor_id) {
    assert(system != NULL);
    
    // Check if scope_id is the same as ancestor_id
    if (scope_id == ancestor_id) {
        return true;
    }
    
    // Check if scope_id is a descendant of ancestor_id
    uint64_t parent_id = binding_system_get_parent_scope(system, scope_id);
    while (parent_id != 0) {
        if (parent_id == ancestor_id) {
            return true;
        }
        
        parent_id = binding_system_get_parent_scope(system, parent_id);
    }
    
    // Not a descendant
    return false;
}

/**
 * @brief Get the parent scope of a scope
 * 
 * @param system The binding system
 * @param scope_id The scope ID
 * @return The parent scope ID, or 0 if not found or if the scope is the global scope
 */
uint64_t binding_system_get_parent_scope(BindingSystem* system, uint64_t scope_id) {
    assert(system != NULL);
    
    // Search for scope
    for (size_t i = 0; i < system->scope_stack.count; i++) {
        if (system->scope_stack.ids[i] == scope_id) {
            // Found scope
            return system->scope_stack.parent_ids[i];
        }
    }
    
    // Scope not found
    return 0;
}

/**
 * @brief Get the current scope ID
 * 
 * @param system The binding system
 * @return The current scope ID
 */
uint64_t binding_system_get_current_scope(BindingSystem* system) {
    assert(system != NULL);
    
    return system->current_scope_id;
}
