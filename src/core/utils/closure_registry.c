/**
 * @file closure_registry.c
 * @brief Implementation of the dynamic closure registry for Eshkol
 */

#include "core/closure.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Global registry
static ClosureRegistry global_registry = {NULL, 0, 0};

/**
 * @brief Initialize the closure registry
 * 
 * This function initializes the global closure registry.
 */
void init_closure_registry() {
    // Only initialize if not already initialized
    if (global_registry.closures == NULL) {
        global_registry.closures = malloc(sizeof(EshkolClosure*) * 16);
        if (global_registry.closures == NULL) {
            fprintf(stderr, "Error: Failed to allocate memory for closure registry\n");
            exit(1);
        }
        global_registry.capacity = 16;
        global_registry.count = 0;
        
        // Initialize all entries to NULL
        for (size_t i = 0; i < global_registry.capacity; i++) {
            global_registry.closures[i] = NULL;
        }
    }
}

/**
 * @brief Add a closure to the registry
 * 
 * @param closure The closure to add
 * @return The index of the added closure, or -1 on failure
 */
int registry_add_closure(EshkolClosure* closure) {
    assert(closure != NULL);
    
    // Initialize registry if needed
    if (global_registry.closures == NULL) {
        init_closure_registry();
    }
    
    // Check if we need to resize
    if (global_registry.count >= global_registry.capacity) {
        size_t new_capacity = global_registry.capacity * 2;
        EshkolClosure** new_closures = realloc(global_registry.closures, 
                                              sizeof(EshkolClosure*) * new_capacity);
        if (new_closures == NULL) {
            fprintf(stderr, "Error: Failed to resize closure registry\n");
            return -1;  // Failed to resize
        }
        
        // Initialize new entries to NULL
        for (size_t i = global_registry.capacity; i < new_capacity; i++) {
            new_closures[i] = NULL;
        }
        
        global_registry.closures = new_closures;
        global_registry.capacity = new_capacity;
    }
    
    // Add the closure to the registry
    int index = global_registry.count++;
    global_registry.closures[index] = closure;
    
    // Update the closure's registry index
    closure->registry_index = index;
    
    return index;
}

/**
 * @brief Remove a closure from the registry
 * 
 * @param index The index of the closure to remove
 */
void registry_remove_closure(int index) {
    // Check if the registry is initialized
    if (global_registry.closures == NULL) {
        return;  // Nothing to do
    }
    
    // Check if the index is valid
    if (index < 0 || index >= global_registry.count) {
        return;  // Invalid index
    }
    
    // Get the closure at the index
    EshkolClosure* closure = global_registry.closures[index];
    
    // Update the closure's registry index
    if (closure != NULL) {
        closure->registry_index = -1;
    }
    
    // Shift all closures after this one
    for (int i = index; i < global_registry.count - 1; i++) {
        global_registry.closures[i] = global_registry.closures[i + 1];
        
        // Update the registry index of the shifted closure
        if (global_registry.closures[i] != NULL) {
            global_registry.closures[i]->registry_index = i;
        }
    }
    
    // Clear the last entry
    global_registry.closures[global_registry.count - 1] = NULL;
    
    // Decrement the count
    global_registry.count--;
}

/**
 * @brief Get a closure from the registry
 * 
 * @param index The index of the closure to get
 * @return The closure, or NULL if not found
 */
EshkolClosure* registry_get_closure(int index) {
    // Check if the registry is initialized
    if (global_registry.closures == NULL) {
        return NULL;  // Registry not initialized
    }
    
    // Check if the index is valid
    if (index < 0 || index >= global_registry.count) {
        return NULL;  // Invalid index
    }
    
    return global_registry.closures[index];
}

/**
 * @brief Clean up the registry
 * 
 * This function frees all resources used by the registry.
 */
void cleanup_closure_registry() {
    // Check if the registry is initialized
    if (global_registry.closures == NULL) {
        return;  // Nothing to do
    }
    
    // Free all closures in the registry
    for (size_t i = 0; i < global_registry.count; i++) {
        if (global_registry.closures[i] != NULL) {
            // Set registry_index to -1 to avoid recursive calls to registry_remove_closure
            global_registry.closures[i]->registry_index = -1;
            
            // Free the closure
            eshkol_closure_free(global_registry.closures[i]);
            global_registry.closures[i] = NULL;
        }
    }
    
    // Free the registry array
    free(global_registry.closures);
    global_registry.closures = NULL;
    global_registry.capacity = 0;
    global_registry.count = 0;
}

/**
 * @brief Trampoline function for closure calls
 * 
 * This function is used as a bridge between the EshkolFunction interface
 * and the EshkolClosure interface. It looks up the closure in the registry
 * and calls it with the provided argument.
 * 
 * @param arg The argument to pass to the closure
 * @return The result of the closure call
 */
static void* closure_trampoline(void* arg) {
    // Get the closure index from thread-local storage
    static __thread int current_closure_index = -1;
    
    // If this is the first call in a chain, we need to set up the index
    if (current_closure_index == -1) {
        // Use the most recently added closure
        current_closure_index = global_registry.count - 1;
    }
    
    // Look up the closure in the registry
    EshkolClosure* closure = registry_get_closure(current_closure_index);
    if (closure == NULL) {
        fprintf(stderr, "Error: Failed to look up closure in trampoline (index: %d)\n", current_closure_index);
        current_closure_index = -1;  // Reset for next call
        return NULL;
    }
    
    // Create an array of arguments
    void* args[1] = { arg };
    
    // Reset the current closure index for the next call
    int index = current_closure_index;
    current_closure_index = -1;
    
    // Call the closure
    return closure->function(closure->environment, args);
}

/**
 * @brief Convert an EshkolClosure to an EshkolFunction
 * 
 * @param closure The closure to convert
 * @return An EshkolFunction that wraps the closure
 */
EshkolFunction eshkol_closure_to_function(EshkolClosure* closure) {
    if (closure == NULL) {
        fprintf(stderr, "Error: NULL closure in eshkol_closure_to_function\n");
        return NULL;
    }
    
    // Register the closure if not already registered
    if (closure->registry_index < 0) {
        int index = registry_add_closure(closure);
        if (index < 0) {
            fprintf(stderr, "Error: Failed to register closure\n");
            return NULL;
        }
    }
    
    // Return the trampoline function
    return closure_trampoline;
}
