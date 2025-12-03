/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#ifndef ESHKOL_LLVM_BACKEND_H
#define ESHKOL_LLVM_BACKEND_H

#include <eshkol/eshkol.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

// Forward declarations for LLVM types to avoid including LLVM headers in C code
struct LLVMOpaqueModule;
typedef struct LLVMOpaqueModule *LLVMModuleRef;

/*
 * Generate LLVM IR from a vector of AST nodes
 * @param asts Array of AST nodes to compile
 * @param num_asts Number of AST nodes in the array
 * @param module_name Name for the LLVM module
 * @return LLVM module reference, or NULL on error
 */
LLVMModuleRef eshkol_generate_llvm_ir(const eshkol_ast_t* asts, size_t num_asts, const char* module_name);

/*
 * Generate LLVM IR in library mode (no main function, all functions exported)
 * Use this for compiling libraries like stdlib.esk that only contain definitions.
 * @param asts Array of AST nodes to compile
 * @param num_asts Number of AST nodes in the array
 * @param module_name Name for the LLVM module
 * @return LLVM module reference, or NULL on error
 */
LLVMModuleRef eshkol_generate_llvm_ir_library(const eshkol_ast_t* asts, size_t num_asts, const char* module_name);

/*
 * Register external function declarations from ASTs (for linking with pre-compiled libraries)
 * This extracts function signatures from the ASTs and registers them as external declarations
 * so the compiler knows about them, but doesn't generate code (code comes from linked library).
 * @param asts Array of AST nodes containing function definitions
 * @param num_asts Number of AST nodes in the array
 */
void eshkol_register_external_functions(const eshkol_ast_t* asts, size_t num_asts);

/*
 * Set flag indicating the program uses stdlib (linked with stdlib.o)
 * When set, the generated code will call __eshkol_lib_init__ for homoiconic display support.
 * @param uses_stdlib true if linking with stdlib.o
 */
void eshkol_set_uses_stdlib(int uses_stdlib);

/*
 * Write LLVM IR to a file in textual format
 * @param module LLVM module reference
 * @param filename Output filename (should end with .ll)
 * @return 0 on success, non-zero on error
 */
int eshkol_dump_llvm_ir_to_file(LLVMModuleRef module, const char* filename);

/*
 * Compile LLVM IR to object file
 * @param module LLVM module reference
 * @param filename Output object filename (should end with .o)
 * @return 0 on success, non-zero on error
 */
int eshkol_compile_llvm_ir_to_object(LLVMModuleRef module, const char* filename);

/*
 * Compile LLVM IR to executable
 * @param module LLVM module reference
 * @param filename Output executable filename
 * @param lib_paths Array of library search paths (can be NULL)
 * @param num_lib_paths Number of library search paths
 * @param linked_libs Array of libraries to link (can be NULL)
 * @param num_linked_libs Number of libraries to link
 * @return 0 on success, non-zero on error
 */
int eshkol_compile_llvm_ir_to_executable(LLVMModuleRef module, const char* filename, 
                                        const char* const* lib_paths, size_t num_lib_paths,
                                        const char* const* linked_libs, size_t num_linked_libs);

/*
 * Clean up LLVM module
 * @param module LLVM module reference to dispose
 */
void eshkol_dispose_llvm_module(LLVMModuleRef module);

/*
 * Release module ownership for JIT use
 * Removes module from internal storage so JIT can take ownership.
 * CRITICAL: After calling this, the caller MUST NOT call eshkol_dispose_llvm_module.
 * @param module LLVM module reference
 */
void eshkol_release_module_for_jit(LLVMModuleRef module);

/*
 * Print LLVM IR to stdout
 * @param module LLVM module reference
 */
void eshkol_print_llvm_ir(LLVMModuleRef module);

/*
 * REPL Mode: Enable symbol persistence across compiler invocations
 * When enabled, symbols registered via eshkol_repl_register_* will be available
 * in subsequent compilations, allowing cross-evaluation function calls.
 */
void eshkol_repl_enable(void);

/*
 * REPL Mode: Disable symbol persistence
 * Clears all registered REPL symbols.
 */
void eshkol_repl_disable(void);

/*
 * REPL Mode: Register a JIT symbol (variable) for cross-evaluation access
 * @param name Symbol name (e.g., "myvar")
 * @param address JIT address of the symbol
 */
void eshkol_repl_register_symbol(const char* name, uint64_t address);

/*
 * REPL Mode: Register a JIT function for cross-evaluation access
 * @param name Function name (e.g., "lambda_0" or "myfunc")
 * @param address JIT address of the function
 * @param arity Number of parameters the function takes
 */
void eshkol_repl_register_function(const char* name, uint64_t address, size_t arity);

/*
 * REPL Mode: Register variable -> lambda name mapping for s-expression lookup
 * @param var_name Variable name (e.g., "square")
 * @param lambda_name Lambda function name (e.g., "lambda_0")
 */
void eshkol_repl_register_lambda_name(const char* var_name, const char* lambda_name);

/*
 * REPL Mode: Register s-expression runtime value
 * @param sexpr_name S-expression global name (e.g., "lambda_0_sexpr")
 * @param sexpr_value Runtime pointer value of the s-expression
 */
void eshkol_repl_register_sexpr(const char* sexpr_name, uint64_t sexpr_value);

#else

// Stub implementations when LLVM backend is disabled
#define eshkol_generate_llvm_ir(asts, num_asts, module_name) (NULL)
#define eshkol_generate_llvm_ir_library(asts, num_asts, module_name) (NULL)
#define eshkol_register_external_functions(asts, num_asts) do {} while(0)
#define eshkol_dump_llvm_ir_to_file(module, filename) (-1)
#define eshkol_compile_llvm_ir_to_object(module, filename) (-1)
#define eshkol_compile_llvm_ir_to_executable(module, filename, lib_paths, num_lib_paths, linked_libs, num_linked_libs) (-1)
#define eshkol_dispose_llvm_module(module) do {} while(0)
#define eshkol_print_llvm_ir(module) do {} while(0)
#define eshkol_repl_enable() do {} while(0)
#define eshkol_repl_disable() do {} while(0)
#define eshkol_repl_register_symbol(name, address) do {} while(0)
#define eshkol_repl_register_function(name, address, arity) do {} while(0)
#define eshkol_repl_register_lambda_name(var_name, lambda_name) do {} while(0)
#define eshkol_repl_register_sexpr(sexpr_name, sexpr_value) do {} while(0)

#endif // ESHKOL_LLVM_BACKEND_ENABLED

#ifdef __cplusplus
}
#endif

#endif // ESHKOL_LLVM_BACKEND_H