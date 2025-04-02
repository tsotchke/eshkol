/**
 * @file closure.h
 * @brief Closure system for Eshkol
 * 
 * This file defines the closure system for the Eshkol language, which enables
 * first-class functions with proper lexical scoping.
 */

#ifndef ESHKOL_CLOSURE_H
#define ESHKOL_CLOSURE_H

#include "core/memory.h"
#include "core/type.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Function pointer type for Eshkol functions
 */
typedef void* (*EshkolFunction)(void*);

/**
 * @brief Structure for a composed function
 */
typedef struct {
    EshkolFunction f;  // First function to apply (outer)
    EshkolFunction g;  // Second function to apply (inner)
} ComposedFunction;

/**
 * @brief Environment structure for closures
 * 
 * This structure represents the lexical environment for a closure,
 * containing captured variables.
 */
typedef struct EshkolEnvironment {
    uint64_t env_id;                   /**< Unique ID for tracking and debugging */
    struct EshkolEnvironment* parent;  /**< Parent environment */
    void** values;                     /**< Array of captured values */
    Type** types;                      /**< Types of captured values */
    size_t value_count;                /**< Number of values in this environment */
    size_t capacity;                   /**< Capacity of the values array */
    int ref_count;                     /**< Reference count for memory management */
    bool in_validation;                /**< Flag to detect cycles during validation */
    
    // New fields for lexical scoping
    uint64_t scope_id;                 /**< Associated scope ID */
    char** binding_names;              /**< Array of binding names */
    size_t binding_count;              /**< Number of bindings */
} EshkolEnvironment;

/**
 * @brief Closure structure
 * 
 * This structure represents a closure, which is a function together with
 * its lexical environment.
 */
typedef struct EshkolClosure {
    void* (*function)(struct EshkolEnvironment*, void**); /**< Function pointer */
    struct EshkolEnvironment* environment;                /**< Lexical environment */
    Type* return_type;                                    /**< Return type */
    Type** param_types;                                   /**< Parameter types */
    size_t param_count;                                   /**< Number of parameters */
    int registry_index;                                   /**< Index in the registry, or -1 if not registered */
} EshkolClosure;

/**
 * @brief Dynamic registry for closures
 * 
 * This structure manages a dynamic array of closure pointers.
 */
typedef struct {
    EshkolClosure** closures;  /**< Array of closure pointers */
    size_t capacity;           /**< Current capacity of the array */
    size_t count;              /**< Number of closures in the registry */
} ClosureRegistry;

// Forward declarations for composed functions
void* composed_function_wrapper(EshkolEnvironment* env, void** args);

/**
 * @brief Initialize the closure registry
 * 
 * This function initializes the global closure registry.
 */
void init_closure_registry();

/**
 * @brief Add a closure to the registry
 * 
 * @param closure The closure to add
 * @return The index of the added closure, or -1 on failure
 */
int registry_add_closure(EshkolClosure* closure);

/**
 * @brief Remove a closure from the registry
 * 
 * @param index The index of the closure to remove
 */
void registry_remove_closure(int index);

/**
 * @brief Get a closure from the registry
 * 
 * @param index The index of the closure to get
 * @return The closure, or NULL if not found
 */
EshkolClosure* registry_get_closure(int index);

/**
 * @brief Clean up the registry
 * 
 * This function frees all resources used by the registry.
 */
void cleanup_closure_registry();

/**
 * @brief Create a new environment
 * 
 * @param parent Parent environment, or NULL for a global environment
 * @param capacity Initial capacity for captured values
 * @param scope_id Associated scope ID
 * @return A new environment, or NULL on failure
 */
EshkolEnvironment* eshkol_environment_create(EshkolEnvironment* parent, size_t capacity, uint64_t scope_id);

/**
 * @brief Add a value to an environment
 * 
 * @param env The environment
 * @param value The value to add
 * @param type The type of the value
 * @param binding_name The name of the binding (can be NULL)
 * @return The index of the added value, or -1 on failure
 */
int eshkol_environment_add(EshkolEnvironment* env, void* value, Type* type, const char* binding_name);

/**
 * @brief Get a value from an environment
 * 
 * @param env The environment
 * @param index The index of the value
 * @param depth The number of parent environments to traverse (0 for current)
 * @return The value, or NULL if not found
 */
void* eshkol_environment_get(EshkolEnvironment* env, size_t index, size_t depth);

/**
 * @brief Increment the reference count of an environment
 * 
 * @param env The environment
 */
void eshkol_environment_retain(EshkolEnvironment* env);

/**
 * @brief Decrement the reference count of an environment and free if zero
 * 
 * @param env The environment
 */
void eshkol_environment_release(EshkolEnvironment* env);

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
);

/**
 * @brief Call a closure with arguments
 * 
 * @param closure The closure to call
 * @param args The arguments to pass to the function
 * @return The result of the function call
 */
void* eshkol_closure_call(EshkolClosure* closure, void** args);

/**
 * @brief Free a closure
 * 
 * @param closure The closure to free
 */
void eshkol_closure_free(EshkolClosure* closure);

/**
 * @brief Validate an environment and ensure all values are properly initialized
 * 
 * This function checks the environment for NULL values and attempts to
 * initialize them from parent environments if possible.
 * 
 * @param env The environment to validate
 */
void eshkol_environment_validate(EshkolEnvironment* env);

/**
 * @brief Create a composed function
 * 
 * @param f The outer function
 * @param g The inner function
 * @return A new ComposedFunction structure
 */
ComposedFunction* eshkol_create_composed_function(EshkolFunction f, EshkolFunction g);

/**
 * @brief Free a composed function
 * 
 * @param composed The ComposedFunction to free
 */
void eshkol_free_composed_function(ComposedFunction* composed);

/**
 * @brief Call a composed function with an argument
 * 
 * @param composed_func Pointer to a ComposedFunction structure
 * @param x The input value
 * @return The result of applying g to x, then f to that result
 */
void* eshkol_composed_function_call(ComposedFunction* composed_func, void* x);

/**
 * @brief Convert an EshkolClosure to an EshkolFunction
 * 
 * @param closure The closure to convert
 * @return An EshkolFunction that wraps the closure
 */
EshkolFunction eshkol_closure_to_function(EshkolClosure* closure);

/**
 * @brief Create a composed function from two closures
 * 
 * @param f The outer closure
 * @param g The inner closure
 * @return A new closure that represents the composition of f and g
 */
EshkolClosure* eshkol_compose_functions(EshkolClosure* f, EshkolClosure* g);

/**
 * @brief Wrapper function for composed closures
 * 
 * @param env The environment containing the composed function
 * @param args The arguments to pass to the function
 * @return The result of applying g to args[0], then f to that result
 */
void* composed_function_wrapper(EshkolEnvironment* env, void** args);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CLOSURE_H */
