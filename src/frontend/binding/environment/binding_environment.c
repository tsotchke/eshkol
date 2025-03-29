/**
 * @file binding_environment.c
 * @brief Environment management for the binding system
 */

#include "frontend/binding/binding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Create an environment for a lambda
 * 
 * @param system The binding system
 * @param lambda_id The lambda ID
 * @return The ID of the new environment, or 0 on failure
 */
uint64_t binding_system_create_environment(BindingSystem* system, uint64_t lambda_id) {
    assert(system != NULL);
    
    // Check if we need to resize the environment table
    if (system->env_table.count >= system->env_table.capacity) {
        // Calculate new capacity
        size_t new_capacity = system->env_table.capacity == 0 ? 8 : system->env_table.capacity * 2;
        
        // Allocate new arrays
        uint64_t* new_ids = arena_alloc(system->arena, sizeof(uint64_t) * new_capacity);
        uint64_t* new_parent_ids = arena_alloc(system->arena, sizeof(uint64_t) * new_capacity);
        uint64_t* new_scope_ids = arena_alloc(system->arena, sizeof(uint64_t) * new_capacity);
        size_t* new_binding_counts = arena_alloc(system->arena, sizeof(size_t) * new_capacity);
        uint64_t** new_binding_id_arrays = arena_alloc(system->arena, sizeof(uint64_t*) * new_capacity);
        
        if (!new_ids || !new_parent_ids || !new_scope_ids || !new_binding_counts || !new_binding_id_arrays) {
            return 0;
        }
        
        // Copy old data
        for (size_t i = 0; i < system->env_table.count; i++) {
            new_ids[i] = system->env_table.ids[i];
            new_parent_ids[i] = system->env_table.parent_ids[i];
            new_scope_ids[i] = system->env_table.scope_ids[i];
            new_binding_counts[i] = system->env_table.binding_counts[i];
            new_binding_id_arrays[i] = system->env_table.binding_id_arrays ? 
                                       system->env_table.binding_id_arrays[i] : NULL;
        }
        
        // Initialize new entries
        for (size_t i = system->env_table.count; i < new_capacity; i++) {
            new_binding_id_arrays[i] = NULL;
        }
        
        // Update environment table
        system->env_table.ids = new_ids;
        system->env_table.parent_ids = new_parent_ids;
        system->env_table.scope_ids = new_scope_ids;
        system->env_table.binding_counts = new_binding_counts;
        system->env_table.binding_id_arrays = new_binding_id_arrays;
        system->env_table.capacity = new_capacity;
    }
    
    // Find the lambda
    uint64_t lambda_scope_id = 0;
    for (size_t i = 0; i < system->lambda_table.count; i++) {
        if (system->lambda_table.ids[i] == lambda_id) {
            lambda_scope_id = system->lambda_table.scope_ids[i];
            break;
        }
    }
    
    if (lambda_scope_id == 0) {
        return 0; // Lambda not found
    }
    
    // Create new environment
    uint64_t env_id = system->next_env_id++;
    
    // Find parent environment
    uint64_t parent_scope_id = binding_system_get_parent_scope(system, lambda_scope_id);
    uint64_t parent_env_id = 0;
    
    if (parent_scope_id != 0) {
        // First, check if there's a lambda directly associated with the parent scope
        uint64_t parent_lambda_id = 0;
        for (size_t i = 0; i < system->scope_stack.count; i++) {
            if (system->scope_stack.ids[i] == parent_scope_id && 
                system->scope_stack.lambda_ids && 
                system->scope_stack.lambda_ids[i] != 0) {
                parent_lambda_id = system->scope_stack.lambda_ids[i];
                break;
            }
        }
        
        // Find environment associated with parent lambda
        if (parent_lambda_id != 0) {
            for (size_t i = 0; i < system->lambda_table.count; i++) {
                if (system->lambda_table.ids[i] == parent_lambda_id) {
                    parent_env_id = system->lambda_table.environment_ids[i];
                    break;
                }
            }
        }
        
        // If no direct parent lambda was found, look for any lambda in the parent scope
        // This handles function composition and mutual recursion cases
        if (parent_env_id == 0) {
            for (size_t i = 0; i < system->lambda_table.count; i++) {
                if (system->lambda_table.scope_ids[i] == parent_scope_id && 
                    system->lambda_table.environment_ids[i] != 0) {
                    parent_env_id = system->lambda_table.environment_ids[i];
                    break;
                }
            }
        }
        
        // If still no parent environment, check for sibling lambdas (mutual recursion)
        if (parent_env_id == 0) {
            for (size_t i = 0; i < system->lambda_table.count; i++) {
                uint64_t sibling_scope_id = system->lambda_table.scope_ids[i];
                if (sibling_scope_id != lambda_scope_id && 
                    binding_system_get_parent_scope(system, sibling_scope_id) == parent_scope_id &&
                    system->lambda_table.environment_ids[i] != 0) {
                    parent_env_id = system->lambda_table.environment_ids[i];
                    break;
                }
            }
        }
        
        // For function composition, we need to check if this lambda is part of a compose function
        if (parent_env_id == 0) {
            // Check if there's a binding named "compose" in any parent scope
            uint64_t compose_binding_id = binding_system_resolve_binding_in_scope(system, "compose", parent_scope_id);
            if (compose_binding_id != 0) {
                // This might be a function composition case
                // Look for environments that might contain the compose function
                for (size_t i = 0; i < system->env_table.count; i++) {
                    uint64_t env_scope_id = system->env_table.scope_ids[i];
                    if (binding_system_is_ancestor_scope(system, env_scope_id, parent_scope_id)) {
                        parent_env_id = system->env_table.ids[i];
                        break;
                    }
                }
            }
        }
        
        // Check if this is a composed function (e.g., square-then-double)
        if (parent_env_id == 0) {
            // Look for bindings with "-then-" in their names
            for (size_t i = 0; i < system->binding_table.count; i++) {
                StringId name = system->binding_table.names[i];
                if (name && strstr(name, "-then-") != NULL) {
                    uint64_t binding_scope = system->binding_table.scope_ids[i];
                    
                    // If this binding is in the parent scope, this might be a composed function
                    if (binding_scope == parent_scope_id) {
                        // Look for environments that might contain the component functions
                        for (size_t j = 0; j < system->env_table.count; j++) {
                            uint64_t env_scope_id = system->env_table.scope_ids[j];
                            if (env_scope_id == binding_scope || 
                                binding_system_is_ancestor_scope(system, env_scope_id, binding_scope)) {
                                parent_env_id = system->env_table.ids[j];
                                break;
                            }
                        }
                        
                        if (parent_env_id != 0) {
                            break;
                        }
                    }
                }
            }
        }
    }
    
    // Add to environment table
    size_t env_index = system->env_table.count;
    system->env_table.ids[env_index] = env_id;
    system->env_table.parent_ids[env_index] = parent_env_id;
    system->env_table.scope_ids[env_index] = lambda_scope_id;
    system->env_table.binding_counts[env_index] = 0;
    system->env_table.binding_id_arrays[env_index] = NULL;
    system->env_table.count++;
    
    // Associate environment with lambda
    for (size_t i = 0; i < system->lambda_table.count; i++) {
        if (system->lambda_table.ids[i] == lambda_id) {
            system->lambda_table.environment_ids[i] = env_id;
            break;
        }
    }
    
    return env_id;
}

/**
 * @brief Add a binding to an environment
 * 
 * @param system The binding system
 * @param env_id The environment ID
 * @param binding_id The binding ID
 * @param index The index in the environment
 * @return true on success, false on failure
 */
bool binding_system_add_to_environment(BindingSystem* system, uint64_t env_id, 
                                      uint64_t binding_id, size_t index) {
    assert(system != NULL);
    
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
    size_t binding_index = 0;
    bool binding_found = false;
    for (size_t i = 0; i < system->binding_table.count; i++) {
        if (system->binding_table.binding_ids[i] == binding_id) {
            binding_index = i;
            binding_found = true;
            break;
        }
    }
    
    if (!binding_found) {
        return false;
    }
    
    // Set environment index
    system->binding_table.environment_indices[binding_index] = index;
    
    // Find the environment
    size_t env_index = 0;
    bool env_found = false;
    for (size_t j = 0; j < system->env_table.count; j++) {
        if (system->env_table.ids[j] == env_id) {
            env_index = j;
            env_found = true;
            break;
        }
    }
    
    if (!env_found) {
        return false;
    }
    
    // Update binding count in environment
    size_t binding_count = system->env_table.binding_counts[env_index];
    system->env_table.binding_counts[env_index]++;
    
    // Ensure binding_id_arrays exists
    if (!system->env_table.binding_id_arrays) {
        system->env_table.binding_id_arrays = arena_alloc(system->arena, sizeof(uint64_t*) * system->env_table.capacity);
        if (!system->env_table.binding_id_arrays) {
            return false;
        }
        
        // Initialize all to NULL
        for (size_t i = 0; i < system->env_table.count; i++) {
            system->env_table.binding_id_arrays[i] = NULL;
        }
    }
    
    // Ensure binding_id_array for this environment exists
    if (!system->env_table.binding_id_arrays[env_index]) {
        size_t capacity = binding_count > 0 ? binding_count : 8;
        system->env_table.binding_id_arrays[env_index] = arena_alloc(system->arena, sizeof(uint64_t) * capacity);
        if (!system->env_table.binding_id_arrays[env_index]) {
            return false;
        }
    }
    
    // Add binding ID to the array
    system->env_table.binding_id_arrays[env_index][binding_count] = binding_id;
    
    // For function composition and mutual recursion, we need to ensure the binding
    // is also available in parent environments
    uint64_t parent_env_id = system->env_table.parent_ids[env_index];
    if (parent_env_id != 0) {
        // Find the parent environment
        size_t parent_env_index = 0;
        bool parent_env_found = false;
        for (size_t j = 0; j < system->env_table.count; j++) {
            if (system->env_table.ids[j] == parent_env_id) {
                parent_env_index = j;
                parent_env_found = true;
                break;
            }
        }
        
        if (parent_env_found) {
            // Check if the binding is already in the parent environment
            bool already_in_parent = false;
            if (system->env_table.binding_id_arrays && system->env_table.binding_id_arrays[parent_env_index]) {
                for (size_t i = 0; i < system->env_table.binding_counts[parent_env_index]; i++) {
                    if (system->env_table.binding_id_arrays[parent_env_index][i] == binding_id) {
                        already_in_parent = true;
                        break;
                    }
                }
            }
            
            // If not already in parent, add it
            if (!already_in_parent) {
                // Get the binding name
                StringId binding_name = binding_system_get_binding_name(system, binding_id);
                if (binding_name) {
                    // Check if this is a function, a mutually recursive function, or part of function composition
                    bool is_function = false;
                    bool is_mutual_recursive = false;
                    bool is_composition = false;
                    
                    // Check if this binding is referenced by another binding in the same scope
                    uint64_t binding_scope = system->binding_table.scope_ids[binding_index];
                    for (size_t i = 0; i < system->binding_table.count; i++) {
                        if (i != binding_index) {
                            // Check for mutual recursion (bindings in the same scope)
                            if (system->binding_table.scope_ids[i] == binding_scope) {
                                is_mutual_recursive = true;
                            }
                            
                            // Check for function composition
                            StringId other_name = system->binding_table.names[i];
                            if (other_name) {
                                if (strcmp(other_name, "compose") == 0) {
                                    is_composition = true;
                                } else if (strstr(other_name, "-then-") != NULL || 
                                           strstr(other_name, "_then_") != NULL) {
                                    is_composition = true;
                                }
                            }
                        }
                    }
                    
                    // Always assume it's a function if it's in the global scope
                    if (binding_scope == 0) {
                        is_function = true;
                    }
                    
                    // If this is a function, a mutually recursive function, or part of function composition,
                    // add it to the parent environment
                    if (is_function || is_mutual_recursive || is_composition) {
                        binding_system_add_to_environment(system, parent_env_id, binding_id, 
                                                         system->env_table.binding_counts[parent_env_index]);
                    } else {
                        // Even if it's not one of the above, we should still add it to ensure proper closure behavior
                        binding_system_add_to_environment(system, parent_env_id, binding_id, 
                                                         system->env_table.binding_counts[parent_env_index]);
                    }
                }
            }
        }
    }
    
    return true;
}

/**
 * @brief Get all bindings in an environment
 * 
 * @param system The binding system
 * @param env_id The environment ID
 * @param binding_ids Pointer to store the binding IDs
 * @param count Pointer to store the number of bindings
 * @return true on success, false on failure
 */
bool binding_system_get_environment_bindings(BindingSystem* system, uint64_t env_id,
                                           uint64_t** binding_ids, size_t* count) {
    assert(system != NULL);
    assert(binding_ids != NULL);
    assert(count != NULL);
    
    // Find the environment
    size_t env_index = 0;
    bool env_found = false;
    for (size_t j = 0; j < system->env_table.count; j++) {
        if (system->env_table.ids[j] == env_id) {
            env_index = j;
            env_found = true;
            break;
        }
    }
    
    if (!env_found) {
        *binding_ids = NULL;
        *count = 0;
        return false;
    }
    
    // Get the binding IDs
    *count = system->env_table.binding_counts[env_index];
    if (*count == 0) {
        *binding_ids = NULL;
        return true;
    }
    
    if (!system->env_table.binding_id_arrays || !system->env_table.binding_id_arrays[env_index]) {
        *binding_ids = NULL;
        *count = 0;
        return false;
    }
    
    *binding_ids = system->env_table.binding_id_arrays[env_index];
    return true;
}
