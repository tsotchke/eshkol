/**
 * @file jit.h
 * @brief JIT compilation system for Eshkol
 * 
 * This file defines the JIT compilation system for the Eshkol language,
 * which enables efficient function composition and higher-order functions.
 */

#ifndef ESHKOL_JIT_H
#define ESHKOL_JIT_H

#include "core/closure.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Architecture types supported by the JIT compiler
 */
typedef enum {
    ARCH_UNKNOWN,
    ARCH_X86_64,
    ARCH_ARM64
} Architecture;

/**
 * @brief Code buffer for JIT compilation
 */
typedef struct {
    void* code;         // Pointer to the generated code
    size_t capacity;    // Total capacity of the code buffer (in bytes)
    size_t size;        // Current size of the generated code (in bytes)
    Architecture arch;  // Target architecture
} CodeBuffer;

/**
 * @brief JIT-compiled function structure
 */
typedef struct {
    void* (*function)(void*);  // The JIT-compiled function
    CodeBuffer* buffer;        // The code buffer (for cleanup)
    EshkolClosure* f;          // The outer function
    EshkolClosure* g;          // The inner function
    Architecture arch;         // The target architecture
} JitComposedFunction;

/**
 * @brief Detect the current architecture
 * 
 * @return The detected architecture
 */
Architecture jit_detect_architecture(void);

/**
 * @brief Allocate executable memory for JIT compilation
 * 
 * @param size The size of the memory to allocate
 * @return A pointer to the allocated memory, or NULL on failure
 */
void* jit_allocate_executable_memory(size_t size);

/**
 * @brief Free executable memory
 * 
 * @param mem The memory to free
 * @param size The size of the memory
 */
void jit_free_executable_memory(void* mem, size_t size);

/**
 * @brief Create a code buffer for JIT compilation
 * 
 * @param initial_capacity The initial capacity of the buffer
 * @param arch The target architecture
 * @return A new code buffer, or NULL on failure
 */
CodeBuffer* jit_create_code_buffer(size_t initial_capacity, Architecture arch);

/**
 * @brief Free a code buffer
 * 
 * @param buffer The buffer to free
 */
void jit_free_code_buffer(CodeBuffer* buffer);

/**
 * @brief Emit a byte to an x86-64 code buffer
 * 
 * @param buffer The code buffer
 * @param byte The byte to emit
 */
void jit_emit_byte_x86(CodeBuffer* buffer, uint8_t byte);

/**
 * @brief Emit a 32-bit instruction to an ARM64 code buffer
 * 
 * @param buffer The code buffer
 * @param instruction The instruction to emit
 */
void jit_emit_instruction_arm64(CodeBuffer* buffer, uint32_t instruction);

/**
 * @brief Generate a JIT-compiled function for function composition
 * 
 * @param f The outer function
 * @param g The inner function
 * @return A function pointer to the JIT-compiled function, or NULL on failure
 */
void* (*jit_generate_composed_function(EshkolClosure* f, EshkolClosure* g))(void*);

/**
 * @brief Generate a JIT-compiled function for x86-64
 * 
 * @param f The outer function
 * @param g The inner function
 * @return A function pointer to the JIT-compiled function, or NULL on failure
 */
void* (*jit_generate_composed_function_x86_64(EshkolClosure* f, EshkolClosure* g))(void*);

/**
 * @brief Generate a JIT-compiled function for ARM64
 * 
 * @param f The outer function
 * @param g The inner function
 * @return A function pointer to the JIT-compiled function, or NULL on failure
 */
void* (*jit_generate_composed_function_arm64(EshkolClosure* f, EshkolClosure* g))(void*);

/**
 * @brief Create a JIT-compiled composed function
 * 
 * @param f The outer function
 * @param g The inner function
 * @return A new JitComposedFunction structure, or NULL on failure
 */
JitComposedFunction* jit_create_composed_function(EshkolClosure* f, EshkolClosure* g);

/**
 * @brief Free a JIT-compiled composed function
 * 
 * @param jit The JitComposedFunction to free
 */
void jit_free_composed_function(JitComposedFunction* jit);

/**
 * @brief Look up a JIT-compiled function in the cache
 * 
 * @param f The outer function
 * @param g The inner function
 * @return The cached JitComposedFunction, or NULL if not found
 */
JitComposedFunction* jit_lookup_function(EshkolClosure* f, EshkolClosure* g);

/**
 * @brief Add a JIT-compiled function to the cache
 * 
 * @param f The outer function
 * @param g The inner function
 * @param jit The JitComposedFunction to cache
 */
void jit_cache_function(EshkolClosure* f, EshkolClosure* g, JitComposedFunction* jit);

/**
 * @brief Initialize the JIT system
 */
void jit_init(void);

/**
 * @brief Clean up the JIT system
 */
void jit_cleanup(void);

/**
 * @brief Direct wrapper for JIT-compiled functions
 * 
 * @param env The environment containing the JIT-compiled function
 * @param args The arguments to pass to the function
 * @return The result of the function call
 */
void* jit_direct_wrapper(EshkolEnvironment* env, void** args);

/**
 * @brief Create a composed function using JIT compilation
 * 
 * @param f The outer closure
 * @param g The inner closure
 * @return A new closure that represents the composition of f and g
 */
EshkolClosure* eshkol_compose_functions_jit(EshkolClosure* f, EshkolClosure* g);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_JIT_H */
