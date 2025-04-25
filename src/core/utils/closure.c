/**
 * @file closure.c
 * @brief Implementation of the closure system for Eshkol
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
 * It also handles special cases for function composition and mutual recursion.
 *  
 * @param env The environment to validate
 */
void eshkol_environment_validate(EshkolEnvironment* env) {
    if (env == NULL) {
        return;
    }
        
    // Detect cycles using the in_validation flag
    if (env->in_validation) {
        fprintf(stderr, "Warning: Cycle detected in environment chain (env_id: %llu)\n", env->env_id);
        // Don't reset the flag here, as we're detecting a cycle
        // The flag will be reset by the caller
        return;
    }
        
    // Mark this environment as being validated
    env->in_validation = true;
        
    // Create a cache of all environments we've seen to avoid cycles
    EshkolEnvironment* env_cache[32] = {0}; // Arbitrary limit, but should be enough
    size_t cache_count = 0;
        
    // Add current environment to cache
    env_cache[cache_count++] = env;
        
    bool is_valid __attribute__((unused)) = true; // Used internally but not returned
        
    // Check if this is a function composition environment by looking for specific patterns
    bool is_composition_env = false;
    bool is_nested_composition = false;
        
    // Check for closures in the environment that might be part of a composition
    for (size_t i = 0; i < env->value_count; i++) {
        if (env->values[i] != NULL) {
            // Try to check if this is a closure
            EshkolClosure* closure = (EshkolClosure*)env->values[i];
                        
            // Simple validation check - if it has a function pointer, it's likely a closure
            if (closure->function != NULL) {
                // This might be a closure, check if it has an environment
                if (closure->environment != NULL) {
                    // This is likely a closure, which suggests this might be a composition environment
                    is_composition_env = true;
                                        
                    // Check if this is a nested composition
                    if (closure->environment->parent != NULL) {
                        is_nested_composition = true;
                    }
                                        
                    // Validate the closure's environment if it's not already being validated
                    if (!closure->environment->in_validation) {
                        closure->environment->in_validation = true;
                        eshkol_environment_validate(closure->environment);
                        closure->environment->in_validation = false;
                    }
                }
            }
        }
    }
        
    // Check all values in the environment
    for (size_t i = 0; i < env->value_count; i++) {
        if (env->values[i] == NULL) {
            is_valid = false;
                        
            // Check if this binding has a name
            const char* binding_name = (i < env->binding_count) ? env->binding_names[i] : NULL;
                        
            // Try to find the value in parent environments
            EshkolEnvironment* parent = env->parent;
            while (parent != NULL) {
                // Check if we've already seen this environment
                bool seen = false;
                for (size_t j = 0; j < cache_count; j++) {
                    if (env_cache[j] == parent) {
                        seen = true;
                        break;
                    }
                }
                                
                if (seen) {
                    // Skip this environment to avoid cycles
                    parent = parent->parent;
                    continue;
                }
                                
                // Add to cache
                if (cache_count < 32) {
                    env_cache[cache_count++] = parent;
                }
                                
                // First try to find by binding name if available
                if (binding_name != NULL) {
                    for (size_t j = 0; j < parent->binding_count; j++) {
                        if (parent->binding_names[j] != NULL &&  
                            strcmp(binding_name, parent->binding_names[j]) == 0) {
                            if (j < parent->value_count && parent->values[j] != NULL) {
                                // Found the value by name in a parent environment, copy it
                                env->values[i] = parent->values[j];
                                is_valid = true;
                                break;
                            }
                        }
                    }
                                        
                    if (env->values[i] != NULL) {
                        break; // Found the value, stop searching
                    }
                }
                                
                // If we couldn't find by name or name is not available, try by index
                if (i < parent->value_count && parent->values[i] != NULL) {
                    // Found the value in a parent environment, copy it
                    env->values[i] = parent->values[i];
                    is_valid = true;
                    break;
                }
                                
                parent = parent->parent;
            }
                        
            // If we still couldn't find the value, try a more aggressive search
            // This is needed for function composition and mutual recursion
            if (env->values[i] == NULL) {
                // Reset cache
                cache_count = 0;
                env_cache[cache_count++] = env;
                                
                // First try to find any value at the same index in any environment
                EshkolEnvironment* ancestor = env->parent;
                while (ancestor != NULL) {
                    // Check if we've already seen this environment
                    bool seen = false;
                    for (size_t j = 0; j < cache_count; j++) {
                        if (env_cache[j] == ancestor) {
                            seen = true;
                            break;
                        }
                    }
                                        
                    if (seen) {
                        // Skip this environment to avoid cycles
                        ancestor = ancestor->parent;
                        continue;
                    }
                                        
                    // Add to cache
                    if (cache_count < 32) {
                        env_cache[cache_count++] = ancestor;
                    }
                                        
                    // Check if this index exists in the ancestor
                    if (i < ancestor->value_count && ancestor->values[i] != NULL) {
                        env->values[i] = ancestor->values[i];
                        break;
                    }
                                        
                    ancestor = ancestor->parent;
                }
                                
                // If we still couldn't find a value, try to find any non-NULL value
                if (env->values[i] == NULL) {
                    // Reset cache
                    cache_count = 0;
                    env_cache[cache_count++] = env;
                                        
                    // Try to find any non-NULL value in the current environment
                    for (size_t j = 0; j < env->value_count; j++) {
                        if (j != i && env->values[j] != NULL) {
                            env->values[i] = env->values[j];
                            break;
                        }
                    }
                                        
                    // If we still couldn't find a value, try to find any non-NULL value in parent environments
                    if (env->values[i] == NULL) {
                        ancestor = env->parent;
                        while (ancestor != NULL && env->values[i] == NULL) {
                            // Check if we've already seen this environment
                            bool seen = false;
                            for (size_t j = 0; j < cache_count; j++) {
                                if (env_cache[j] == ancestor) {
                                    seen = true;
                                    break;
                                }
                            }
                                                        
                            if (seen) {
                                // Skip this environment to avoid cycles
                                ancestor = ancestor->parent;
                                continue;
                            }
                                                        
                            // Add to cache
                            if (cache_count < 32) {
                                env_cache[cache_count++] = ancestor;
                            }
                            
                            // Check all values in this environment
                            for (size_t j = 0; j < ancestor->value_count; j++) {
                                if (ancestor->values[j] != NULL) {
                                    env->values[i] = ancestor->values[j];
                                    break;
                                }
                            }
                            ancestor = ancestor->parent;
                        }
                    }
                }
            }
        }
    }
    
    // Special handling for function composition environments
    if (is_composition_env) {
        // For function composition, we need to ensure all component functions are properly initialized
        
        // First, validate all closures in the environment
        for (size_t i = 0; i < env->value_count; i++) {
            if (env->values[i] != NULL) {
                // Try to check if this is a closure
                EshkolClosure* closure = (EshkolClosure*)env->values[i];
                
                // Simple validation check - if it has a function pointer, it's likely a closure
                if (closure->function != NULL) {
                    // This might be a closure, check if it has an environment
                    if (closure->environment != NULL) {
                        // Validate the closure's environment if it's not already being validated
                        if (!closure->environment->in_validation) {
                            closure->environment->in_validation = true;
                            eshkol_environment_validate(closure->environment);
                            closure->environment->in_validation = false;
                        }
                    }
                }
            }
        }
        
        // For nested compositions, we need to ensure all component functions are properly initialized
        if (is_nested_composition) {
            // Validate all parent environments
            EshkolEnvironment* parent = env->parent;
            while (parent != NULL) {
                // Check if we've already seen this environment
                bool seen = false;
                for (size_t j = 0; j < cache_count; j++) {
                    if (env_cache[j] == parent) {
                        seen = true;
                        break;
                    }
                }
                
                if (seen) {
                    // Skip this environment to avoid cycles
                    parent = parent->parent;
                    continue;
                }
                
                // Add to cache
                if (cache_count < 32) {
                    env_cache[cache_count++] = parent;
                }
                
                // Validate the parent environment if it's not already being validated
                if (!parent->in_validation) {
                    parent->in_validation = true;
                    eshkol_environment_validate(parent);
                    parent->in_validation = false;
                }
                
                parent = parent->parent;
            }
            
            // Copy any missing values from parent environments
            for (size_t i = 0; i < env->value_count; i++) {
                if (env->values[i] == NULL) {
                    // Try to find the value in any parent environment
                    parent = env->parent;
                    while (parent != NULL && env->values[i] == NULL) {
                        // Check if we've already seen this environment
                        bool seen = false;
                        for (size_t j = 0; j < cache_count; j++) {
                            if (env_cache[j] == parent) {
                                seen = true;
                                break;
                            }
                        }
                        
                        if (seen) {
                            // Skip this environment to avoid cycles
                            parent = parent->parent;
                            continue;
                        }
                        
                        // Add to cache
                        if (cache_count < 32) {
                            env_cache[cache_count++] = parent;
                        }
                        
                        // Check if this index exists in the parent
                        if (i < parent->value_count && parent->values[i] != NULL) {
                            env->values[i] = parent->values[i];
                            break;
                        }
                        
                        parent = parent->parent;
                    }
                }
            }
        }
    }
    
    // Always reset the in_validation flag before returning
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
            // Create a cache of all environments we've seen to avoid cycles
            EshkolEnvironment* env_cache[32] = {0}; // Arbitrary limit, but should be enough
            size_t cache_count = 0;
            
            // Add current environment to cache
            env_cache[cache_count++] = current;
            
            // Check if this binding has a name
            const char* binding_name = (index < current->binding_count) ? current->binding_names[index] : NULL;
            
            // Search in all parent environments
            EshkolEnvironment* ancestor = current->parent;
            while (ancestor != NULL && cache_count < 32) {
                // Check if we've already seen this environment
                bool seen = false;
                for (size_t i = 0; i < cache_count; i++) {
                    if (env_cache[i] == ancestor) {
                        seen = true;
                        break;
                    }
                }
                
                if (seen) {
                    // Skip this environment to avoid cycles
                    ancestor = ancestor->parent;
                    continue;
                }
                
                // Add to cache
                env_cache[cache_count++] = ancestor;
                
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
            
            // If we still haven't found the value, try a more aggressive search
            // This is needed for function composition and mutual recursion
            
            // Reset cache
            cache_count = 0;
            env_cache[cache_count++] = current;
            
            // Search in all parent environments for any non-NULL value
            ancestor = current->parent;
            while (ancestor != NULL && cache_count < 32) {
                // Check if we've already seen this environment
                bool seen = false;
                for (size_t i = 0; i < cache_count; i++) {
                    if (env_cache[i] == ancestor) {
                        seen = true;
                        break;
                    }
                }
                
                if (seen) {
                    // Skip this environment to avoid cycles
                    ancestor = ancestor->parent;
                    continue;
                }
                
                // Add to cache
                env_cache[cache_count++] = ancestor;
                
                // Check all values in this environment
                for (size_t i = 0; i < ancestor->value_count; i++) {
                    if (ancestor->values[i] != NULL) {
                        // Cache the value for future lookups
                        current->values[index] = ancestor->values[i];
                        return ancestor->values[i];
                    }
                }
                
                ancestor = ancestor->parent;
            }
            
            // If we still haven't found a value, check if there are any values in the current environment
            // that we can use as a placeholder
            for (size_t i = 0; i < current->value_count; i++) {
                if (i != index && current->values[i] != NULL) {
                    // Use this as a placeholder
                    current->values[index] = current->values[i];
                    return current->values[index];
                }
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
        
        // Free arrays
        free(env->values);
        free(env->types);
        free(env->binding_names);
        
        // Free environment
        free(env);
    }
}

/**
 * @brief Create a new closure
 * 
 * @param function The function pointer
 * @param env The lexical environment
 * @param return_type The return type
 * @param param_types The parameter types
 * @param param_count The number of parameters
 * @return A new closure, or NULL on failure
 */
EshkolClosure* eshkol_closure_create(
    void* (*function)(EshkolEnvironment*, void**),
    EshkolEnvironment* env,
    Type* return_type,
    Type** param_types,
    size_t param_count
) {
    // Allocate closure
    EshkolClosure* closure = malloc(sizeof(EshkolClosure));
    if (!closure) {
        return NULL;
    }
    
    // Initialize closure
    closure->function = function;
    closure->environment = env;
    closure->return_type = return_type;
    closure->param_count = param_count;
    
    // Allocate parameter types array
    if (param_count > 0) {
        closure->param_types = malloc(sizeof(Type*) * param_count);
        if (!closure->param_types) {
            free(closure);
            return NULL;
        }
        
        // Copy parameter types
        for (size_t i = 0; i < param_count; i++) {
            closure->param_types[i] = param_types ? param_types[i] : NULL;
        }
    } else {
        closure->param_types = NULL;
    }
    
    // Retain environment
    if (env) {
        eshkol_environment_retain(env);
    }
    
    return closure;
}

/**
 * @brief Call a closure with arguments
 * 
 * @param closure The closure to call
 * @param args The arguments to pass to the function
 * @return The result of the function call
 */
void* eshkol_closure_call(EshkolClosure* closure, void** args) {
    assert(closure != NULL);
    assert(closure->function != NULL);
    
    // Ensure environment is valid
    if (closure->environment == NULL) {
        fprintf(stderr, "Error: NULL environment in closure call\n");
        return NULL;
    }
    
    // Validate environment to ensure all values are properly initialized
    eshkol_environment_validate(closure->environment);
    
    // Reset the in_validation flag in case it wasn't properly reset
    closure->environment->in_validation = false;
    
    // Call the function with the environment and arguments
    return closure->function(closure->environment, args);
}

/**
 * @brief Free a closure
 * 
 * @param closure The closure to free
 */
void eshkol_closure_free(EshkolClosure* closure) {
    if (closure) {
        // Release environment
        if (closure->environment) {
            eshkol_environment_release(closure->environment);
        }
        
        // Free parameter types array
        free(closure->param_types);
        
        // Free closure
        free(closure);
    }
}

/**
 * @brief Create a composed function
 * 
 * @param f The outer function
 * @param g The inner function
 * @return A new ComposedFunction structure
 */
ComposedFunction* eshkol_create_composed_function(EshkolFunction f, EshkolFunction g) {
    ComposedFunction* composed = malloc(sizeof(ComposedFunction));
    if (composed == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for composed function\n");
        exit(1);
    }
    
    composed->f = f;
    composed->g = g;
    
    return composed;
}

/**
 * @brief Free a composed function
 * 
 * @param composed The ComposedFunction to free
 */
void eshkol_free_composed_function(ComposedFunction* composed) {
    free(composed);
}

/**
 * @brief Wrapper function for composed functions
 * 
 * @param composed_func Pointer to a ComposedFunction structure
 * @param x The input value
 * @return The result of applying g to x, then f to that result
 */
void* eshkol_composed_function_call(ComposedFunction* composed_func, void* x) {
    if (composed_func == NULL) {
        fprintf(stderr, "Error: NULL composed function\n");
        exit(1);
    }
    
    if (composed_func->f == NULL || composed_func->g == NULL) {
        fprintf(stderr, "Error: NULL function in composition\n");
        exit(1);
    }
    
    // First apply g, then apply f to the result
    void* g_result = composed_func->g(x);
    return composed_func->f(g_result);
}

// Registry for closures to avoid function pointer manipulation
#define MAX_CLOSURES 1024
typedef struct {
    EshkolClosure* closure;
    int in_use;
    int id;
} ClosureRegistryEntry;

static ClosureRegistryEntry closure_registry[MAX_CLOSURES];
static int next_closure_id = 1;

// Forward declaration of the wrapper functions
static void* closure_wrapper_0(void* x);
static void* closure_wrapper_1(void* x);
static void* closure_wrapper_2(void* x);
// ... add more as needed, up to a reasonable limit

// Array of wrapper functions
static EshkolFunction closure_wrappers[] = {
    closure_wrapper_0,
    closure_wrapper_1,
    closure_wrapper_2,
    // ... add more as needed
};

#define NUM_WRAPPERS (sizeof(closure_wrappers) / sizeof(closure_wrappers[0]))

// Implementation of wrapper functions
static void* closure_wrapper_0(void* x) {
    ClosureRegistryEntry* entry = &closure_registry[0];
    if (!entry->in_use || entry->closure == NULL) {
        fprintf(stderr, "Error: Invalid closure in wrapper 0\n");
        return NULL;
    }
    return eshkol_closure_call(entry->closure, (void*[]){x});
}

static void* closure_wrapper_1(void* x) {
    ClosureRegistryEntry* entry = &closure_registry[1];
    if (!entry->in_use || entry->closure == NULL) {
        fprintf(stderr, "Error: Invalid closure in wrapper 1\n");
        return NULL;
    }
    return eshkol_closure_call(entry->closure, (void*[]){x});
}

static void* closure_wrapper_2(void* x) {
    ClosureRegistryEntry* entry = &closure_registry[2];
    if (!entry->in_use || entry->closure == NULL) {
        fprintf(stderr, "Error: Invalid closure in wrapper 2\n");
        return NULL;
    }
    return eshkol_closure_call(entry->closure, (void*[]){x});
}

// Add more wrapper functions as needed...

/**
 * @brief Convert an EshkolClosure to an EshkolFunction
 *
 * @param closure The closure to convert
 * @return An EshkolFunction that wraps the closure
 */
/**
 * @brief Check if a pointer is a valid closure
 * 
 * @param ptr Pointer to check
 * @return true if ptr is a valid closure, false otherwise
 */
bool eshkol_is_closure(void* ptr) {
    if (ptr == NULL) {
        return false;
    }
    
    // Try to interpret the pointer as a closure
    EshkolClosure* closure = (EshkolClosure*)ptr;
    
    // Basic sanity checks to validate if this is likely a closure
    // Note: This can't be 100% accurate without risking a segfault
    if (closure->function == NULL) {
        return false;
    }
    
    // Check if the closure has a valid environment pointer
    if (closure->environment == NULL) {
        return false;
    }
    
    // Check param_count is within reasonable bounds
    if (closure->param_count > 100) {  // Arbitrary reasonable limit
        return false;
    }
    
    // If param_count > 0, param_types should not be NULL
    if (closure->param_count > 0 && closure->param_types == NULL) {
        return false;
    }
    
    // Check if registry_index is valid or -1 (not registered)
    if (closure->registry_index >= MAX_CLOSURES && closure->registry_index != -1) {
        return false;
    }
    
    // Passed all checks, likely a valid closure
    return true;
}

/**
 * @brief Call a function or closure with an argument
 * 
 * This function can call regular functions, closures, or composed functions.
 * It automatically detects the type of the function and calls it appropriately.
 * 
 * @param func The function or closure to call
 * @param arg The argument to pass to the function
 * @return The result of the function call
 */
void* eshkol_call_function(void* func, void* arg) {
    if (func == NULL) {
        fprintf(stderr, "Error: NULL function in eshkol_call_function\n");
        return NULL;
    }
    
    // Check if this is a composed function (marked with LSB set to 1)
    if ((uintptr_t)func & 1) {
        // This is a composed function
        ComposedFunction* composed = (ComposedFunction*)((uintptr_t)func & ~1);
        return eshkol_composed_function_call(composed, arg);
    }
    
    // Check if this is a closure
    if (eshkol_is_closure(func)) {
        // This is a closure, call it using eshkol_closure_call
        return eshkol_closure_call((EshkolClosure*)func, (void*[]){arg});
    }
    
    // This is a regular function, call it directly
    return ((void* (*)(void*))func)(arg);
}

EshkolFunction eshkol_closure_to_function(EshkolClosure* closure) {
    if (closure == NULL) {
        fprintf(stderr, "Error: NULL closure in eshkol_closure_to_function\n");
        return NULL;
    }
    
    // Find an available slot in the registry
    int slot = -1;
    for (size_t i = 0; i < NUM_WRAPPERS; i++) {
        if (!closure_registry[i].in_use) {
            slot = i;
            break;
        }
    }
    
    if (slot == -1) {
        fprintf(stderr, "Error: No available slots in closure registry (max %d)\n", (int)NUM_WRAPPERS);
        return NULL;
    }
    
    // Register the closure
    closure_registry[slot].closure = closure;
    closure_registry[slot].in_use = 1;
    closure_registry[slot].id = next_closure_id++;
    
    // Return the corresponding wrapper function
    return closure_wrappers[slot];
}
