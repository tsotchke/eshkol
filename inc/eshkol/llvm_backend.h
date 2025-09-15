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
 * Print LLVM IR to stdout
 * @param module LLVM module reference
 */
void eshkol_print_llvm_ir(LLVMModuleRef module);

#else

// Stub implementations when LLVM backend is disabled
#define eshkol_generate_llvm_ir(asts, num_asts, module_name) (NULL)
#define eshkol_dump_llvm_ir_to_file(module, filename) (-1)
#define eshkol_compile_llvm_ir_to_object(module, filename) (-1)
#define eshkol_compile_llvm_ir_to_executable(module, filename, lib_paths, num_lib_paths, linked_libs, num_linked_libs) (-1)
#define eshkol_dispose_llvm_module(module) do {} while(0)
#define eshkol_print_llvm_ir(module) do {} while(0)

#endif // ESHKOL_LLVM_BACKEND_ENABLED

#ifdef __cplusplus
}
#endif

#endif // ESHKOL_LLVM_BACKEND_H