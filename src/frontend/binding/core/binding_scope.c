/**
 * @file binding_scope.c
 * @brief Scope management for the binding system
 */

#include "frontend/binding/binding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

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
        ScopeType* new_types = arena_alloc(system->arena, sizeof(ScopeType) * new_capacity);
        uint64_t* new_lambda_ids = arena_alloc(system->arena, sizeof(uint64_t) * new_capacity);
        
        if (!new_ids || !new_parent_ids || !new_types || !new_lambda_ids) {
            return 0;
        }
        
        // Copy old data
        for (size_t i = 0; i < system->scope_stack.count; i++) {
            new_ids[i] = system->scope_stack.ids[i];
            new_parent_ids[i] = system->scope_stack.parent_ids[i];
            if (system->scope_stack.types) {
                new_types[i] = system->scope_stack.types[i];
            } else {
                new_types[i] = SCOPE_LOCAL; // Default for existing scopes
            }
            if (system->scope_stack.lambda_ids) {
                new_lambda_ids[i] = system->scope_stack.lambda_ids[i];
            } else {
                new_lambda_ids[i] = 0; // Default for existing scopes
            }
        }
        
        // Update scope stack
        system->scope_stack.ids = new_ids;
        system->scope_stack.parent_ids = new_parent_ids;
        system->scope_stack.types = new_types;
        system->scope_stack.lambda_ids = new_lambda_ids;
        system->scope_stack.capacity = new_capacity;
    }
    
    // Create new scope
    uint64_t scope_id = system->next_scope_id++;
    uint64_t parent_id = system->current_scope_id;
    
    // Add to scope stack
    system->scope_stack.ids[system->scope_stack.count] = scope_id;
    system->scope_stack.parent_ids[system->scope_stack.count] = parent_id;
    
    // Set scope type and lambda ID
    if (system->scope_stack.types) {
        system->scope_stack.types[system->scope_stack.count] = system->scope_stack.count == 0 ? SCOPE_GLOBAL : SCOPE_LOCAL;
    }
    if (system->scope_stack.lambda_ids) {
        system->scope_stack.lambda_ids[system->scope_stack.count] = 0;
    }
    
    system->scope_stack.count++;
    
    // Update current scope
    system->current_scope_id = scope_id;
    
    return scope_id;
}

/**
 * @brief Enter a new lambda scope
 * 
 * @param system The binding system
 * @return The ID of the new scope, or 0 on failure
 */
uint64_t binding_system_enter_lambda_scope(BindingSystem* system) {
    assert(system != NULL);
    
    // Create a new scope
    uint64_t scope_id = binding_system_enter_scope(system);
    if (scope_id == 0) {
        return 0;
    }
    
    // Set scope type to lambda
    if (system->scope_stack.types) {
        for (size_t i = 0; i < system->scope_stack.count; i++) {
            if (system->scope_stack.ids[i] == scope_id) {
                system->scope_stack.types[i] = SCOPE_LAMBDA;
                break;
            }
        }
    }
    
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
 * @brief Check if a scope is an ancestor of another scope
 * 
 * @param system The binding system
 * @param scope_id The scope ID to check
 * @param descendant_id The potential descendant scope ID
 * @return true if scope_id is an ancestor of descendant_id, false otherwise
 */
bool binding_system_is_ancestor_scope(BindingSystem* system, uint64_t scope_id, uint64_t descendant_id) {
    assert(system != NULL);
    
    // This is just the inverse of is_descendant_scope
    return binding_system_is_descendant_scope(system, descendant_id, scope_id);
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
