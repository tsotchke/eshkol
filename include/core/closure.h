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
} EshkolClosure;

/**
 * @brief Create a new environment
 * 
 * @param parent Parent environment, or NULL for a global environment
 * @param capacity Initial capacity for captured values
 * @return A new environment, or NULL on failure
 */
EshkolEnvironment* eshkol_environment_create(EshkolEnvironment* parent, size_t capacity);

/**
 * @brief Add a value to an environment
 * 
 * @param env The environment
 * @param value The value to add
 * @param type The type of the value
 * @return The index of the added value, or -1 on failure
 */
int eshkol_environment_add(EshkolEnvironment* env, void* value, Type* type);

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

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CLOSURE_H */
