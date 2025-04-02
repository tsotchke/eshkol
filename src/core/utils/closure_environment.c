/**
 * @file closure_environment.c
 * @brief Implementation of the environment system for Eshkol closures
 */

#include "core/closure.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Create a new environment
 * 
 * @param parent Parent environment, or NULL for a global environment
 * @param capacity Initial capacity for captured values
 * @param scope_id Associated scope ID
 * @return A new environment, or NULL on failure
 */
EshkolEnvironment* eshkol_environment_create(EshkolEnvironment* parent, size_t capacity, uint64_t scope_id) {
    static uint64_t next_env_id = 1;  // Static counter for unique IDs
        
    // Allocate environment
    EshkolEnvironment* env = malloc(sizeof(EshkolEnvironment));
    if (!env) {
        return NULL;
    }
        
    // Initialize environment with unique ID and validation flag
    env->env_id = next_env_id++;
    env->parent = parent;
    env->value_count = 0;
    env->capacity = capacity;
    env->ref_count = 1;
    env->in_validation = false;
        
    // Initialize new fields for lexical scoping
    env->scope_id = scope_id;
    env->binding_count = 0;
        
    // Allocate arrays
    if (capacity > 0) {
        env->values = malloc(sizeof(void*) * capacity);
        env->types = malloc(sizeof(Type*) * capacity);
        env->binding_names = malloc(sizeof(char*) * capacity);
                
        if (!env->values || !env->types || !env->binding_names) {
            free(env->values);
            free(env->types);
            free(env->binding_names);
            free(env);
            return NULL;
        }
                
        // Initialize values to NULL
        for (size_t i = 0; i < capacity; i++) {
            env->values[i] = NULL;
            env->types[i] = NULL;
            env->binding_names[i] = NULL;
        }
    } else {
        env->values = NULL;
        env->types = NULL;
        env->binding_names = NULL;
    }
        
    // Retain parent environment
    if (parent) {
        eshkol_environment_retain(parent);
    }
        
    return env;
}

/**
 * @brief Add a value to an environment
 * 
 * @param env The environment
 * @param value The value to add
 * @param type The type of the value
 * @param binding_name The name of the binding (can be NULL)
 * @return The index of the added value, or -1 on failure
 */
int eshkol_environment_add(EshkolEnvironment* env, void* value, Type* type, const char* binding_name) {
    assert(env != NULL);
        
    // Check if we need to resize
    if (env->value_count >= env->capacity) {
        size_t new_capacity = env->capacity == 0 ? 8 : env->capacity * 2;
                
        // Reallocate arrays
        void** new_values = realloc(env->values, sizeof(void*) * new_capacity);
        Type** new_types = realloc(env->types, sizeof(Type*) * new_capacity);
        char** new_binding_names = realloc(env->binding_names, sizeof(char*) * new_capacity);
                
        if (!new_values || !new_types || !new_binding_names) {
            free(new_values);
            free(new_types);
            free(new_binding_names);
            return -1;
        }
                
        // Initialize new values to NULL
        for (size_t i = env->capacity; i < new_capacity; i++) {
            new_values[i] = NULL;
            new_types[i] = NULL;
            new_binding_names[i] = NULL;
        }
                
        env->values = new_values;
        env->types = new_types;
        env->binding_names = new_binding_names;
        env->capacity = new_capacity;
    }
        
    // Add value
    int index = (int)env->value_count;
    env->values[index] = value;
    env->types[index] = type;
        
    // Add binding name if provided
    if (binding_name != NULL) {
        env->binding_names[index] = strdup(binding_name);
        env->binding_count++;
    } else {
        env->binding_names[index] = NULL;
    }
        
    env->value_count++;
        
    return index;
}

/**
 * @brief Validate and repair an environment
 * 
 * This function checks if all values in the environment are valid,
 * and tries to repair any NULL values by looking for them in parent environments.
 * 
 * @param env The environment to validate
 */
void eshkol_environment_validate(EshkolEnvironment* env) {
    if (env == NULL || env->in_validation) {
        return;  // Avoid cycles
    }
    
    env->in_validation = true;
    
    // Check all values in the environment
    for (size_t i = 0; i < env->value_count; i++) {
        if (env->values[i] == NULL && env->parent != NULL) {
            // Try to find the value in the parent environment
            const char* binding_name = (i < env->binding_count) ? env->binding_names[i] : NULL;
            
            if (binding_name != NULL) {
                // Try to find by name
                for (size_t j = 0; j < env->parent->binding_count; j++) {
                    if (env->parent->binding_names[j] != NULL && 
                        strcmp(binding_name, env->parent->binding_names[j]) == 0) {
                        if (j < env->parent->value_count && env->parent->values[j] != NULL) {
                            env->values[i] = env->parent->values[j];
                            break;
                        }
                    }
                }
            }
            
            // If we couldn't find by name, try by index
            if (env->values[i] == NULL && i < env->parent->value_count) {
                env->values[i] = env->parent->values[i];
            }
        }
    }
    
    // Validate parent environment
    if (env->parent != NULL) {
        eshkol_environment_validate(env->parent);
    }
    
    env->in_validation = false;
}

/**
 * @brief Get a value from an environment
 * 
 * @param env The environment
 * @param index The index of the value
 * @param depth The number of parent environments to traverse (0 for current)
 * @return The value, or NULL if not found
 */
void* eshkol_environment_get(EshkolEnvironment* env, size_t index, size_t depth) {
    if (env == NULL) {
        return NULL;
    }
    
    // Traverse to the correct environment
    EshkolEnvironment* current = env;
    for (size_t i = 0; i < depth && current->parent; i++) {
        current = current->parent;
    }
    
    // Check if the index is valid
    if (index >= current->value_count) {
        // For mutual recursion and function composition, we need to search in parent environments
        if (current->parent != NULL) {
            // Check if this binding has a name
            const char* binding_name = (index < current->binding_count) ? current->binding_names[index] : NULL;
            
            // Try to find the value in any parent environment
            EshkolEnvironment* ancestor = current->parent;
            while (ancestor != NULL) {
                // First try to find by binding name if available
                if (binding_name != NULL) {
                    for (size_t j = 0; j < ancestor->binding_count; j++) {
                        if (ancestor->binding_names[j] != NULL && 
                            strcmp(binding_name, ancestor->binding_names[j]) == 0) {
                            if (j < ancestor->value_count && ancestor->values[j] != NULL) {
                                return ancestor->values[j];
                            }
                        }
                    }
                }
                
                // If we couldn't find by name or name is not available, try by index
                if (index < ancestor->value_count && ancestor->values[index] != NULL) {
                    return ancestor->values[index];
                }
                
                ancestor = ancestor->parent;
            }
        }
        return NULL;
    }
    
    // If the value is NULL, try to find it in parent environments
    if (current->values[index] == NULL) {
        // First validate the environment to ensure all values are properly initialized
        eshkol_environment_validate(current);
        
        // If the value is still NULL after validation, search in parent environments
        if (current->values[index] == NULL && current->parent != NULL) {
            // Check if this binding has a name
            const char* binding_name = (index < current->binding_count) ? current->binding_names[index] : NULL;
            
            // Search in all parent environments
            EshkolEnvironment* ancestor = current->parent;
            while (ancestor != NULL) {
                // First try to find by binding name if available
                if (binding_name != NULL) {
                    for (size_t j = 0; j < ancestor->binding_count; j++) {
                        if (ancestor->binding_names[j] != NULL && 
                            strcmp(binding_name, ancestor->binding_names[j]) == 0) {
                            if (j < ancestor->value_count && ancestor->values[j] != NULL) {
                                // Cache the value for future lookups
                                current->values[index] = ancestor->values[j];
                                return ancestor->values[j];
                            }
                        }
                    }
                }
                
                // If we couldn't find by name or name is not available, try by index
                if (index < ancestor->value_count && ancestor->values[index] != NULL) {
                    // Cache the value for future lookups
                    current->values[index] = ancestor->values[index];
                    return ancestor->values[index];
                }
                
                ancestor = ancestor->parent;
            }
        }
    }
    
    return current->values[index];
}

/**
 * @brief Increment the reference count of an environment
 * 
 * @param env The environment
 */
void eshkol_environment_retain(EshkolEnvironment* env) {
    if (env) {
        env->ref_count++;
    }
}

/**
 * @brief Decrement the reference count of an environment and free if zero
 * 
 * @param env The environment
 */
void eshkol_environment_release(EshkolEnvironment* env) {
    if (env && --env->ref_count == 0) {
        // Release parent environment
        if (env->parent) {
            eshkol_environment_release(env->parent);
        }
        
        // Free binding names
        for (size_t i = 0; i < env->binding_count; i++) {
            free(env->binding_names[i]);
        }
        
        // Free arrays
        free(env->values);
        free(env->types);
        free(env->binding_names);
        
        // Free environment
        free(env);
    }
}
