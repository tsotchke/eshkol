/**
 * @file closure_management.c
 * @brief Implementation of closure creation and management for Eshkol
 */

#include "core/closure.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

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
    closure->registry_index = -1;  // Not registered yet
    
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
        // Remove from registry if registered
        if (closure->registry_index >= 0) {
            registry_remove_closure(closure->registry_index);
        }
        
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
