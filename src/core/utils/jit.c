/**
 * @file jit.c
 * @brief Implementation of the JIT compilation system for Eshkol
 */

#include "core/jit.h"
#include "core/simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

// Cache for JIT-compiled functions
#define MAX_CACHE_SIZE 128

typedef struct {
    EshkolClosure* f;
    EshkolClosure* g;
    JitComposedFunction* jit;
} CacheEntry;

static CacheEntry jit_cache[MAX_CACHE_SIZE];
static size_t jit_cache_size = 0;
static bool jit_initialized = false;

/**
 * @brief Detect the current architecture
 * 
 * @return The detected architecture
 */
Architecture jit_detect_architecture(void) {
    // Initialize SIMD detection if not already done
    if (!jit_initialized) {
        jit_init();
    }
    
    // Check architecture based on compiler defines
#if defined(__x86_64__) || defined(_M_X64)
    return ARCH_X86_64;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return ARCH_ARM64;
#else
    return ARCH_UNKNOWN;
#endif
}

/**
 * @brief Allocate executable memory for JIT compilation
 * 
 * @param size The size of the memory to allocate
 * @return A pointer to the allocated memory, or NULL on failure
 */
void* jit_allocate_executable_memory(size_t size) {
#ifdef _WIN32
    // Windows implementation
    return VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
    // POSIX implementation
    void* mem = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        return NULL;
    }
    return mem;
#endif
}

/**
 * @brief Free executable memory
 * 
 * @param mem The memory to free
 * @param size The size of the memory
 */
void jit_free_executable_memory(void* mem, size_t size) {
    if (mem == NULL) {
        return;
    }
    
#ifdef _WIN32
    // Windows implementation
    VirtualFree(mem, 0, MEM_RELEASE);
#else
    // POSIX implementation
    munmap(mem, size);
#endif
}

/**
 * @brief Create a code buffer for JIT compilation
 * 
 * @param initial_capacity The initial capacity of the buffer
 * @param arch The target architecture
 * @return A new code buffer, or NULL on failure
 */
CodeBuffer* jit_create_code_buffer(size_t initial_capacity, Architecture arch) {
    CodeBuffer* buffer = malloc(sizeof(CodeBuffer));
    if (!buffer) return NULL;
    
    buffer->code = jit_allocate_executable_memory(initial_capacity);
    if (!buffer->code) {
        free(buffer);
        return NULL;
    }
    
    buffer->capacity = initial_capacity;
    buffer->size = 0;
    buffer->arch = arch;
    
    return buffer;
}

/**
 * @brief Free a code buffer
 * 
 * @param buffer The buffer to free
 */
void jit_free_code_buffer(CodeBuffer* buffer) {
    if (!buffer) return;
    
    if (buffer->code) {
        jit_free_executable_memory(buffer->code, buffer->capacity);
    }
    
    free(buffer);
}

/**
 * @brief Emit a byte to an x86-64 code buffer
 * 
 * @param buffer The code buffer
 * @param byte The byte to emit
 */
void jit_emit_byte_x86(CodeBuffer* buffer, uint8_t byte) {
    if (buffer->size >= buffer->capacity) {
        // Resize the buffer if needed
        size_t new_capacity = buffer->capacity * 2;
        void* new_code = jit_allocate_executable_memory(new_capacity);
        if (!new_code) return;
        
        memcpy(new_code, buffer->code, buffer->size);
        jit_free_executable_memory(buffer->code, buffer->capacity);
        
        buffer->code = new_code;
        buffer->capacity = new_capacity;
    }
    
    ((uint8_t*)buffer->code)[buffer->size++] = byte;
}

/**
 * @brief Emit a 32-bit instruction to an ARM64 code buffer
 * 
 * @param buffer The code buffer
 * @param instruction The instruction to emit
 */
void jit_emit_instruction_arm64(CodeBuffer* buffer, uint32_t instruction) {
    if (buffer->size + 4 > buffer->capacity) {
        // Resize the buffer if needed
        size_t new_capacity = buffer->capacity * 2;
        void* new_code = jit_allocate_executable_memory(new_capacity);
        if (!new_code) return;
        
        memcpy(new_code, buffer->code, buffer->size);
        jit_free_executable_memory(buffer->code, buffer->capacity);
        
        buffer->code = new_code;
        buffer->capacity = new_capacity;
    }
    
    *((uint32_t*)((uint8_t*)buffer->code + buffer->size)) = instruction;
    buffer->size += 4;
}

/**
 * @brief Generate a JIT-compiled function for x86-64
 * 
 * @param f The outer function
 * @param g The inner function
 * @return A function pointer to the JIT-compiled function, or NULL on failure
 */
void* (*jit_generate_composed_function_x86_64(EshkolClosure* f, EshkolClosure* g))(void*) {
    CodeBuffer* buffer = jit_create_code_buffer(256, ARCH_X86_64);
    if (!buffer) return NULL;
    
    // Function prologue
    // push rbp
    jit_emit_byte_x86(buffer, 0x55);
    // mov rbp, rsp
    jit_emit_byte_x86(buffer, 0x48);
    jit_emit_byte_x86(buffer, 0x89);
    jit_emit_byte_x86(buffer, 0xe5);
    
    // Save RDI (first argument - arg_struct)
    // push rdi
    jit_emit_byte_x86(buffer, 0x57);
    
    // Extract the actual argument from the arg_struct
    // mov rax, [rdi+8]  ; Load the actual argument (arg_struct[1])
    jit_emit_byte_x86(buffer, 0x48);
    jit_emit_byte_x86(buffer, 0x8b);
    jit_emit_byte_x86(buffer, 0x47);
    jit_emit_byte_x86(buffer, 0x08);
    
    // Save the actual argument to RDI for the call to g
    // mov rdi, rax
    jit_emit_byte_x86(buffer, 0x48);
    jit_emit_byte_x86(buffer, 0x89);
    jit_emit_byte_x86(buffer, 0xc7);
    
    // Move g's function pointer to RAX
    // mov rax, g->function
    jit_emit_byte_x86(buffer, 0x48);
    jit_emit_byte_x86(buffer, 0xb8);
    // Embed g's function pointer (8 bytes)
    uint64_t g_func_ptr = (uint64_t)g->function;
    for (int i = 0; i < 8; i++) {
        jit_emit_byte_x86(buffer, (g_func_ptr >> (i * 8)) & 0xFF);
    }
    
    // Move g's environment pointer to RSI (second argument)
    // mov rsi, g->environment
    jit_emit_byte_x86(buffer, 0x48);
    jit_emit_byte_x86(buffer, 0xbe);
    // Embed g's environment pointer (8 bytes)
    uint64_t g_env_ptr = (uint64_t)g->environment;
    for (int i = 0; i < 8; i++) {
        jit_emit_byte_x86(buffer, (g_env_ptr >> (i * 8)) & 0xFF);
    }
    
    // Call g's function
    // call rax
    jit_emit_byte_x86(buffer, 0xff);
    jit_emit_byte_x86(buffer, 0xd0);
    
    // Result of g is now in RAX
    // Move it to RDI for the call to f
    // mov rdi, rax
    jit_emit_byte_x86(buffer, 0x48);
    jit_emit_byte_x86(buffer, 0x89);
    jit_emit_byte_x86(buffer, 0xc7);
    
    // Move f's function pointer to RAX
    // mov rax, f->function
    jit_emit_byte_x86(buffer, 0x48);
    jit_emit_byte_x86(buffer, 0xb8);
    // Embed f's function pointer (8 bytes)
    uint64_t f_func_ptr = (uint64_t)f->function;
    for (int i = 0; i < 8; i++) {
        jit_emit_byte_x86(buffer, (f_func_ptr >> (i * 8)) & 0xFF);
    }
    
    // Move f's environment pointer to RSI (second argument)
    // mov rsi, f->environment
    jit_emit_byte_x86(buffer, 0x48);
    jit_emit_byte_x86(buffer, 0xbe);
    // Embed f's environment pointer (8 bytes)
    uint64_t f_env_ptr = (uint64_t)f->environment;
    for (int i = 0; i < 8; i++) {
        jit_emit_byte_x86(buffer, (f_env_ptr >> (i * 8)) & 0xFF);
    }
    
    // Call f's function
    // call rax
    jit_emit_byte_x86(buffer, 0xff);
    jit_emit_byte_x86(buffer, 0xd0);
    
    // Result of f is now in RAX, which is the return value
    
    // Function epilogue
    // leave
    jit_emit_byte_x86(buffer, 0xc9);
    // ret
    jit_emit_byte_x86(buffer, 0xc3);
    
    // Return the function pointer
    return (void* (*)(void*))buffer->code;
}

/**
 * @brief Generate a JIT-compiled function for ARM64
 * 
 * @param f The outer function
 * @param g The inner function
 * @return A function pointer to the JIT-compiled function, or NULL on failure
 */
void* (*jit_generate_composed_function_arm64(EshkolClosure* f, EshkolClosure* g))(void*) {
    CodeBuffer* buffer = jit_create_code_buffer(256, ARCH_ARM64);
    if (!buffer) return NULL;
    
    // Function prologue
    // stp x29, x30, [sp, #-16]!  // Save frame pointer and link register
    jit_emit_instruction_arm64(buffer, 0xa9bf7bfd);
    
    // mov x29, sp                // Set frame pointer
    jit_emit_instruction_arm64(buffer, 0x910003fd);
    
    // stp x19, x20, [sp, #-16]!  // Save callee-saved registers
    jit_emit_instruction_arm64(buffer, 0xa9bf13f3);
    
    // Save x0 (input parameter - arg_struct) to a callee-saved register (x19)
    // mov x19, x0
    jit_emit_instruction_arm64(buffer, 0xaa0003f3);
    
    // Extract the actual argument from the arg_struct
    // ldr x0, [x19, #8]  ; Load the actual argument (arg_struct[1])
    jit_emit_instruction_arm64(buffer, 0xf9400660);
    
    // Load g's function pointer to x9
    // movz x9, #(g->function & 0xFFFF)
    uint64_t g_func_ptr = (uint64_t)g->function;
    jit_emit_instruction_arm64(buffer, 0xd2800009 | ((g_func_ptr & 0xFFFF) << 5));
    
    // movk x9, #((g->function >> 16) & 0xFFFF), lsl #16
    jit_emit_instruction_arm64(buffer, 0xf2a00009 | (((g_func_ptr >> 16) & 0xFFFF) << 5));
    
    // movk x9, #((g->function >> 32) & 0xFFFF), lsl #32
    jit_emit_instruction_arm64(buffer, 0xf2c00009 | (((g_func_ptr >> 32) & 0xFFFF) << 5));
    
    // movk x9, #((g->function >> 48) & 0xFFFF), lsl #48
    jit_emit_instruction_arm64(buffer, 0xf2e00009 | (((g_func_ptr >> 48) & 0xFFFF) << 5));
    
    // Load g's environment pointer to x1
    // movz x1, #(g->environment & 0xFFFF)
    uint64_t g_env_ptr = (uint64_t)g->environment;
    jit_emit_instruction_arm64(buffer, 0xd2800001 | ((g_env_ptr & 0xFFFF) << 5));
    
    // movk x1, #((g->environment >> 16) & 0xFFFF), lsl #16
    jit_emit_instruction_arm64(buffer, 0xf2a00001 | (((g_env_ptr >> 16) & 0xFFFF) << 5));
    
    // movk x1, #((g->environment >> 32) & 0xFFFF), lsl #32
    jit_emit_instruction_arm64(buffer, 0xf2c00001 | (((g_env_ptr >> 32) & 0xFFFF) << 5));
    
    // movk x1, #((g->environment >> 48) & 0xFFFF), lsl #48
    jit_emit_instruction_arm64(buffer, 0xf2e00001 | (((g_env_ptr >> 48) & 0xFFFF) << 5));
    
    // Call g's function
    // blr x9
    jit_emit_instruction_arm64(buffer, 0xd63f0120);
    
    // Result of g is now in x0
    // Save it to x20
    // mov x20, x0
    jit_emit_instruction_arm64(buffer, 0xaa0003f4);
    
    // Load f's function pointer to x9
    // movz x9, #(f->function & 0xFFFF)
    uint64_t f_func_ptr = (uint64_t)f->function;
    jit_emit_instruction_arm64(buffer, 0xd2800009 | ((f_func_ptr & 0xFFFF) << 5));
    
    // movk x9, #((f->function >> 16) & 0xFFFF), lsl #16
    jit_emit_instruction_arm64(buffer, 0xf2a00009 | (((f_func_ptr >> 16) & 0xFFFF) << 5));
    
    // movk x9, #((f->function >> 32) & 0xFFFF), lsl #32
    jit_emit_instruction_arm64(buffer, 0xf2c00009 | (((f_func_ptr >> 32) & 0xFFFF) << 5));
    
    // movk x9, #((f->function >> 48) & 0xFFFF), lsl #48
    jit_emit_instruction_arm64(buffer, 0xf2e00009 | (((f_func_ptr >> 48) & 0xFFFF) << 5));
    
    // Load f's environment pointer to x1
    // movz x1, #(f->environment & 0xFFFF)
    uint64_t f_env_ptr = (uint64_t)f->environment;
    jit_emit_instruction_arm64(buffer, 0xd2800001 | ((f_env_ptr & 0xFFFF) << 5));
    
    // movk x1, #((f->environment >> 16) & 0xFFFF), lsl #16
    jit_emit_instruction_arm64(buffer, 0xf2a00001 | (((f_env_ptr >> 16) & 0xFFFF) << 5));
    
    // movk x1, #((f->environment >> 32) & 0xFFFF), lsl #32
    jit_emit_instruction_arm64(buffer, 0xf2c00001 | (((f_env_ptr >> 32) & 0xFFFF) << 5));
    
    // movk x1, #((f->environment >> 48) & 0xFFFF), lsl #48
    jit_emit_instruction_arm64(buffer, 0xf2e00001 | (((f_env_ptr >> 48) & 0xFFFF) << 5));
    
    // Move g's result to x0 for the call to f
    // mov x0, x20
    jit_emit_instruction_arm64(buffer, 0xaa1403e0);
    
    // Call f's function
    // blr x9
    jit_emit_instruction_arm64(buffer, 0xd63f0120);
    
    // Result of f is now in x0, which is the return value
    
    // Function epilogue
    // ldp x19, x20, [sp], #16    // Restore callee-saved registers
    jit_emit_instruction_arm64(buffer, 0xa8c113f3);
    
    // ldp x29, x30, [sp], #16    // Restore frame pointer and link register
    jit_emit_instruction_arm64(buffer, 0xa8c17bfd);
    
    // ret
    jit_emit_instruction_arm64(buffer, 0xd65f03c0);
    
    // Flush instruction cache for ARM
#if defined(__aarch64__) || defined(_M_ARM64)
    __builtin___clear_cache((char*)buffer->code, (char*)((uint8_t*)buffer->code + buffer->size));
#endif
    
    // Return the function pointer
    return (void* (*)(void*))buffer->code;
}

/**
 * @brief Generate a JIT-compiled function for function composition
 * 
 * @param f The outer function
 * @param g The inner function
 * @return A function pointer to the JIT-compiled function, or NULL on failure
 */
void* (*jit_generate_composed_function(EshkolClosure* f, EshkolClosure* g))(void*) {
    Architecture arch = jit_detect_architecture();
    
    switch (arch) {
        case ARCH_X86_64:
            return jit_generate_composed_function_x86_64(f, g);
        case ARCH_ARM64:
            return jit_generate_composed_function_arm64(f, g);
        default:
            fprintf(stderr, "Error: Unsupported architecture for JIT compilation\n");
            return NULL;
    }
}

/**
 * @brief Create a JIT-compiled composed function
 * 
 * @param f The outer function
 * @param g The inner function
 * @return A new JitComposedFunction structure, or NULL on failure
 */
JitComposedFunction* jit_create_composed_function(EshkolClosure* f, EshkolClosure* g) {
    JitComposedFunction* jit = malloc(sizeof(JitComposedFunction));
    if (!jit) return NULL;
    
    jit->arch = jit_detect_architecture();
    jit->function = jit_generate_composed_function(f, g);
    if (!jit->function) {
        free(jit);
        return NULL;
    }
    
    jit->f = f;
    jit->g = g;
    
    // Retain the closures
    // TODO: Implement reference counting for closures
    
    return jit;
}

/**
 * @brief Free a JIT-compiled composed function
 * 
 * @param jit The JitComposedFunction to free
 */
void jit_free_composed_function(JitComposedFunction* jit) {
    if (!jit) return;
    
    // Free the code buffer
    if (jit->buffer) {
        jit_free_code_buffer(jit->buffer);
    }
    
    // Release the closures
    // TODO: Implement reference counting for closures
    
    free(jit);
}

/**
 * @brief Look up a JIT-compiled function in the cache
 * 
 * @param f The outer function
 * @param g The inner function
 * @return The cached JitComposedFunction, or NULL if not found
 */
JitComposedFunction* jit_lookup_function(EshkolClosure* f, EshkolClosure* g) {
    for (size_t i = 0; i < jit_cache_size; i++) {
        if (jit_cache[i].f == f && jit_cache[i].g == g) {
            return jit_cache[i].jit;
        }
    }
    return NULL;
}

/**
 * @brief Add a JIT-compiled function to the cache
 * 
 * @param f The outer function
 * @param g The inner function
 * @param jit The JitComposedFunction to cache
 */
void jit_cache_function(EshkolClosure* f, EshkolClosure* g, JitComposedFunction* jit) {
    if (jit_cache_size >= MAX_CACHE_SIZE) {
        // Cache is full, replace the oldest entry
        jit_free_composed_function(jit_cache[0].jit);
        
        // Shift all entries
        for (size_t i = 0; i < jit_cache_size - 1; i++) {
            jit_cache[i] = jit_cache[i + 1];
        }
        
        jit_cache_size--;
    }
    
    // Add the new entry
    jit_cache[jit_cache_size].f = f;
    jit_cache[jit_cache_size].g = g;
    jit_cache[jit_cache_size].jit = jit;
    jit_cache_size++;
}

/**
 * @brief Initialize the JIT system
 */
void jit_init(void) {
    if (jit_initialized) {
        return;
    }
    
    // Initialize SIMD detection
    simd_init();
    
    // Initialize the cache
    memset(jit_cache, 0, sizeof(jit_cache));
    jit_cache_size = 0;
    
    jit_initialized = true;
}

/**
 * @brief Clean up the JIT system
 */
void jit_cleanup(void) {
    if (!jit_initialized) {
        return;
    }
    
    // Free all cached functions
    for (size_t i = 0; i < jit_cache_size; i++) {
        jit_free_composed_function(jit_cache[i].jit);
    }
    
    // Reset the cache
    memset(jit_cache, 0, sizeof(jit_cache));
    jit_cache_size = 0;
    
    jit_initialized = false;
}

/**
 * @brief Direct wrapper for JIT-compiled functions
 * 
 * @param env The environment containing the JIT-compiled function
 * @param args The arguments to pass to the function
 * @return The result of the function call
 */
void* jit_direct_wrapper(EshkolEnvironment* env, void** args) {
    // Get the JIT-compiled function from the environment
    JitComposedFunction* jit = (JitComposedFunction*)eshkol_environment_get(env, 0, 0);
    if (!jit) {
        fprintf(stderr, "Error: NULL JIT function in wrapper\n");
        exit(1);
    }
    
    // Create a structure to pass both the closure index and the argument
    void** arg_struct = malloc(2 * sizeof(void*));
    if (!arg_struct) {
        fprintf(stderr, "Error: Failed to allocate memory for argument structure\n");
        return NULL;
    }
    
    // Store the closure index and the actual argument
    arg_struct[0] = (void*)(intptr_t)jit->f->registry_index;
    arg_struct[1] = args[0];
    
    // Call the JIT-compiled function with the argument structure
    void* result = jit->function(arg_struct);
    
    // Free the argument structure
    free(arg_struct);
    
    return result;
}

/**
 * @brief Create a composed function using JIT compilation
 * 
 * @param f The outer closure
 * @param g The inner closure
 * @return A new closure that represents the composition of f and g
 */
EshkolClosure* eshkol_compose_functions_jit(EshkolClosure* f, EshkolClosure* g) {
    // Check if we already have a JIT-compiled function for this composition
    JitComposedFunction* jit = jit_lookup_function(f, g);
    
    if (!jit) {
        // Create a new JIT-compiled function
        jit = jit_create_composed_function(f, g);
        if (!jit) {
            fprintf(stderr, "Error: Failed to create JIT-compiled function\n");
            return NULL;
        }
        
        // Add it to the cache
        jit_cache_function(f, g, jit);
    }
    
    // Create a new environment for the closure
    EshkolEnvironment* env = eshkol_environment_create(NULL, 1, 0);
    if (!env) {
        fprintf(stderr, "Error: Failed to create environment for JIT function\n");
        return NULL;
    }
    
    // Add the JIT-compiled function to the environment
    eshkol_environment_add(env, jit, NULL, "jit");
    
    // Create a closure with the wrapper function and environment
    EshkolClosure* closure = eshkol_closure_create(jit_direct_wrapper, env, NULL, NULL, 1);
    
    return closure;
}
