/**
 * @file closure_composition.c
 * @brief Implementation of function composition for Eshkol
 */

#include "core/closure.h"
#include "core/jit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

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
 * @brief Call a composed function with an argument
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

/**
 * @brief Wrapper function for composed closures
 * 
 * @param env The environment containing the composed function
 * @param args The arguments to pass to the function
 * @return The result of applying g to args[0], then f to that result
 */
void* composed_function_wrapper(EshkolEnvironment* env, void** args) {
    // Get the composed function from the environment
    ComposedFunction* composed = (ComposedFunction*)eshkol_environment_get(env, 0, 0);
    if (composed == NULL) {
        fprintf(stderr, "Error: NULL composed function in wrapper\n");
        exit(1);
    }
    
    // Call the inner function
    void* g_result = composed->g(args[0]);
    
    // Call the outer function with the result
    return composed->f(g_result);
}

/**
 * @brief Create a composed function from two closures
 * 
 * @param f The outer closure
 * @param g The inner closure
 * @return A new closure that represents the composition of f and g
 */
EshkolClosure* eshkol_compose_functions(EshkolClosure* f, EshkolClosure* g) {
    if (f == NULL || g == NULL) {
        fprintf(stderr, "Error: NULL closure in eshkol_compose_functions\n");
        return NULL;
    }
    
    // Try JIT compilation first
    EshkolClosure* jit_closure = eshkol_compose_functions_jit(f, g);
    if (jit_closure) {
        return jit_closure;
    }
    
    // Fall back to the dynamic environment approach
    fprintf(stderr, "Warning: JIT compilation failed, falling back to dynamic environment approach\n");
    
    // Convert closures to functions
    EshkolFunction f_func = eshkol_closure_to_function(f);
    EshkolFunction g_func = eshkol_closure_to_function(g);
    
    if (f_func == NULL || g_func == NULL) {
        fprintf(stderr, "Error: Failed to convert closure to function\n");
        return NULL;
    }
    
    // Create a composed function structure
    ComposedFunction* composed = eshkol_create_composed_function(f_func, g_func);
    
    // Create a new environment for the closure
    EshkolEnvironment* env = eshkol_environment_create(NULL, 1, 0);
    
    // Add the composed function to the environment
    eshkol_environment_add(env, composed, NULL, "composed");
    
    // Create a closure with the wrapper function and environment
    EshkolClosure* closure = eshkol_closure_create(composed_function_wrapper, env, NULL, NULL, 1);
    
    return closure;
}
