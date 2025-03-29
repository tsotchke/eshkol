/**
 * @file binding_management.c
 * @brief Binding management for the binding system
 */

#include "frontend/binding/binding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

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
 * @brief Resolve a binding in a specific scope or any parent scope
 * 
 * @param system The binding system
 * @param name The name to resolve
 * @param scope_id The scope ID to start the search from
 * @return The ID of the binding, or 0 if not found
 */
uint64_t binding_system_resolve_binding_in_scope(BindingSystem* system, StringId name, uint64_t scope_id) {
    assert(system != NULL);
    assert(name != NULL);
    
    // Search for binding in specified scope and parent scopes
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
