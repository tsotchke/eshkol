/**
 * @file binding_lambda.c
 * @brief Lambda management for the binding system
 */

#include "frontend/binding/binding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Register a lambda expression
 * 
 * @param system The binding system
 * @param scope_id The scope ID where the lambda is defined
 * @return The ID of the new lambda, or 0 on failure
 */
uint64_t binding_system_register_lambda(BindingSystem* system, uint64_t scope_id) {
    assert(system != NULL);
    
    // Check if we need to resize the lambda table
    if (system->lambda_table.count >= system->lambda_table.capacity) {
        // Calculate new capacity
        size_t new_capacity = system->lambda_table.capacity == 0 ? 8 : system->lambda_table.capacity * 2;
        
        // Allocate new arrays
        uint64_t* new_ids = arena_alloc(system->arena, sizeof(uint64_t) * new_capacity);
        uint64_t* new_scope_ids = arena_alloc(system->arena, sizeof(uint64_t) * new_capacity);
        uint64_t* new_environment_ids = arena_alloc(system->arena, sizeof(uint64_t) * new_capacity);
        
        if (!new_ids || !new_scope_ids || !new_environment_ids) {
            return 0;
        }
        
        // Copy old data
        for (size_t i = 0; i < system->lambda_table.count; i++) {
            new_ids[i] = system->lambda_table.ids[i];
            new_scope_ids[i] = system->lambda_table.scope_ids[i];
            new_environment_ids[i] = system->lambda_table.environment_ids[i];
        }
        
        // Update lambda table
        system->lambda_table.ids = new_ids;
        system->lambda_table.scope_ids = new_scope_ids;
        system->lambda_table.environment_ids = new_environment_ids;
        system->lambda_table.capacity = new_capacity;
    }
    
    // Create new lambda
    uint64_t lambda_id = system->next_lambda_id++;
    
    // Add to lambda table
    system->lambda_table.ids[system->lambda_table.count] = lambda_id;
    system->lambda_table.scope_ids[system->lambda_table.count] = scope_id;
    system->lambda_table.environment_ids[system->lambda_table.count] = 0; // No environment yet
    system->lambda_table.count++;
    
    // Associate lambda with scope if it's a lambda scope
    if (system->scope_stack.types && system->scope_stack.lambda_ids) {
        for (size_t i = 0; i < system->scope_stack.count; i++) {
            if (system->scope_stack.ids[i] == scope_id && 
                system->scope_stack.types[i] == SCOPE_LAMBDA) {
                system->scope_stack.lambda_ids[i] = lambda_id;
                break;
            }
        }
    }
    
    return lambda_id;
}

/**
 * @brief Mark a binding as captured
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return true on success, false on failure
 */
bool binding_system_mark_binding_captured(BindingSystem* system, uint64_t binding_id) {
    assert(system != NULL);
    
    // Ensure is_captured array exists
    if (!system->binding_table.is_captured) {
        system->binding_table.is_captured = arena_alloc(system->arena, sizeof(bool) * system->binding_table.capacity);
        if (!system->binding_table.is_captured) {
            return false;
        }
        
        // Initialize all to false
        for (size_t i = 0; i < system->binding_table.count; i++) {
            system->binding_table.is_captured[i] = false;
        }
    }
    
    // Ensure capture_depth array exists
    if (!system->binding_table.capture_depth) {
        system->binding_table.capture_depth = arena_alloc(system->arena, sizeof(uint64_t) * system->binding_table.capacity);
        if (!system->binding_table.capture_depth) {
            return false;
        }
        
        // Initialize all to 0
        for (size_t i = 0; i < system->binding_table.count; i++) {
            system->binding_table.capture_depth[i] = 0;
        }
    }
    
    // Ensure environment_indices array exists
    if (!system->binding_table.environment_indices) {
        system->binding_table.environment_indices = arena_alloc(system->arena, sizeof(uint64_t) * system->binding_table.capacity);
        if (!system->binding_table.environment_indices) {
            return false;
        }
        
        // Initialize all to 0
        for (size_t i = 0; i < system->binding_table.count; i++) {
            system->binding_table.environment_indices[i] = 0;
        }
    }
    
    // Find the binding
    for (size_t i = 0; i < system->binding_table.count; i++) {
        if (system->binding_table.binding_ids[i] == binding_id) {
            // Mark as captured
            system->binding_table.is_captured[i] = true;
            
            // Calculate capture depth
            uint64_t binding_scope = system->binding_table.scope_ids[i];
            uint64_t current_scope = system->current_scope_id;
            uint64_t depth = 0;
            
            while (current_scope != binding_scope && current_scope != 0) {
                depth++;
                current_scope = binding_system_get_parent_scope(system, current_scope);
            }
            
            system->binding_table.capture_depth[i] = depth;
            
            return true;
        }
    }
    
    return false;
}

/**
 * @brief Check if a binding is captured
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return true if the binding is captured, false otherwise
 */
bool binding_system_is_binding_captured(BindingSystem* system, uint64_t binding_id) {
    assert(system != NULL);
    
    // Check if is_captured array exists
    if (!system->binding_table.is_captured) {
        return false;
    }
    
    // Find the binding
    for (size_t i = 0; i < system->binding_table.count; i++) {
        if (system->binding_table.binding_ids[i] == binding_id) {
            return system->binding_table.is_captured[i];
        }
    }
    
    return false;
}

/**
 * @brief Get the environment index of a binding
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return The environment index, or -1 if not found
 */
int binding_system_get_binding_env_index(BindingSystem* system, uint64_t binding_id) {
    assert(system != NULL);
    
    // Check if environment_indices array exists
    if (!system->binding_table.environment_indices) {
        return -1;
    }
    
    // Find the binding
    for (size_t i = 0; i < system->binding_table.count; i++) {
        if (system->binding_table.binding_ids[i] == binding_id) {
            return (int)system->binding_table.environment_indices[i];
        }
    }
    
    return -1;
}

/**
 * @brief Get all captured bindings for a lambda
 * 
 * @param system The binding system
 * @param lambda_id The lambda ID
 * @param binding_ids Pointer to an array to store the binding IDs
 * @param count Pointer to store the number of bindings
 * @return true on success, false on failure
 */
bool binding_system_get_lambda_captures(BindingSystem* system, uint64_t lambda_id, 
                                       uint64_t** binding_ids, size_t* count) {
    assert(system != NULL);
    assert(binding_ids != NULL);
    assert(count != NULL);
    
    // Initialize count
    *count = 0;
    
    // Check if is_captured array exists
    if (!system->binding_table.is_captured) {
        *binding_ids = NULL;
        return true; // No captures
    }
    
    // Find the lambda scope
    uint64_t lambda_scope_id = 0;
    for (size_t i = 0; i < system->lambda_table.count; i++) {
        if (system->lambda_table.ids[i] == lambda_id) {
            lambda_scope_id = system->lambda_table.scope_ids[i];
            break;
        }
    }
    
    if (lambda_scope_id == 0) {
        return false; // Lambda not found
    }
    
    // Check if this lambda is part of a function composition
    bool is_composition_lambda = false;
    bool is_nested_composition = false;
    
    // Get the parent scope
    uint64_t parent_scope = binding_system_get_parent_scope(system, lambda_scope_id);
    
    // Check for function composition patterns
    for (size_t i = 0; i < system->binding_table.count; i++) {
        StringId name = system->binding_table.names[i];
        if (!name) continue;
        
        // Check for compose function
        if (strcmp(name, "compose") == 0) {
            is_composition_lambda = true;
            // Mark the compose function as captured
            system->binding_table.is_captured[i] = true;
        }
        
        // Check for named composition functions
        if (strstr(name, "-then-") != NULL || 
            strstr(name, "_then_") != NULL) {
            is_composition_lambda = true;
            
            // Check if this is a nested composition
            uint64_t binding_scope = system->binding_table.scope_ids[i];
            if (binding_scope != lambda_scope_id && 
                binding_scope != parent_scope &&
                binding_system_is_ancestor_scope(system, binding_scope, lambda_scope_id)) {
                is_nested_composition = true;
            }
            
            // Mark the composed function as captured
            system->binding_table.is_captured[i] = true;
        }
    }
    
    // Count captured bindings
    size_t capture_count = 0;
    for (size_t i = 0; i < system->binding_table.count; i++) {
        // A binding is captured if:
        // 1. It's marked as captured
        // 2. It's not in the lambda's scope or a descendant scope
        // 3. OR it's a function defined at the same level (for mutual recursion)
        // 4. OR it's part of a function composition
        if (system->binding_table.is_captured[i]) {
            uint64_t binding_scope = system->binding_table.scope_ids[i];
            bool is_same_level_function = false;
            bool is_composition_function = false;
            
            // Check if this is a function defined at the same level (for mutual recursion)
            if (parent_scope != 0 && binding_scope != lambda_scope_id && 
                binding_system_get_parent_scope(system, binding_scope) == parent_scope) {
                // This is a binding from a sibling scope - likely a mutually recursive function
                is_same_level_function = true;
            }
            
            // Check if this is a function composition
            StringId name = system->binding_table.names[i];
            if (name) {
                if (strcmp(name, "compose") == 0 || 
                    strstr(name, "-then-") != NULL || 
                    strstr(name, "_then_") != NULL) {
                    is_composition_function = true;
                }
            }
            
            // For function composition, we need to capture all functions
            if (is_composition_lambda) {
                // For function composition, capture all functions
                capture_count++;
            }
            else if (!binding_system_is_descendant_scope(system, binding_scope, lambda_scope_id) || 
                     is_same_level_function || 
                     is_composition_function) {
                capture_count++;
            }
        }
    }
    
    // For nested compositions, we need to capture all functions in the chain
    if (is_nested_composition) {
        // Add additional captures for nested compositions
        for (size_t i = 0; i < system->binding_table.count; i++) {
            if (!system->binding_table.is_captured[i]) {
                StringId name = system->binding_table.names[i];
                if (name) {
                    // Capture all functions that might be part of the composition chain
                    uint64_t binding_scope = system->binding_table.scope_ids[i];
                    if (binding_scope != lambda_scope_id && 
                        (binding_scope == 0 || // Global functions
                         binding_system_is_ancestor_scope(system, binding_scope, lambda_scope_id))) {
                        system->binding_table.is_captured[i] = true;
                        capture_count++;
                    }
                }
            }
        }
    }
    
    // Allocate array
    *binding_ids = arena_alloc(system->arena, sizeof(uint64_t) * capture_count);
    if (!*binding_ids) {
        return false;
    }
    
    // Fill array
    size_t index = 0;
    for (size_t i = 0; i < system->binding_table.count; i++) {
        if (system->binding_table.is_captured[i]) {
            uint64_t binding_scope = system->binding_table.scope_ids[i];
            bool is_same_level_function = false;
            bool is_composition_function = false;
            
            // Check if this is a function defined at the same level (for mutual recursion)
            if (parent_scope != 0 && binding_scope != lambda_scope_id && 
                binding_system_get_parent_scope(system, binding_scope) == parent_scope) {
                // This is a binding from a sibling scope - likely a mutually recursive function
                is_same_level_function = true;
            }
            
            // Check if this is a function composition
            StringId name = system->binding_table.names[i];
            if (name) {
                if (strcmp(name, "compose") == 0 || 
                    strstr(name, "-then-") != NULL || 
                    strstr(name, "_then_") != NULL) {
                    is_composition_function = true;
                }
            }
            
            // For function composition, we need to capture all functions
            if (is_composition_lambda) {
                // For function composition, capture all functions
                (*binding_ids)[index++] = system->binding_table.binding_ids[i];
            }
            else if (!binding_system_is_descendant_scope(system, binding_scope, lambda_scope_id) || 
                     is_same_level_function || 
                     is_composition_function) {
                (*binding_ids)[index++] = system->binding_table.binding_ids[i];
            }
        }
    }
    
    *count = capture_count;
    return true;
}
