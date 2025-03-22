/**
 * @file codegen.h
 * @brief Code generation interface for Eshkol
 * 
 * This file defines the code generation interface for Eshkol,
 * which translates the AST into C code.
 */

#ifndef ESHKOL_CODEGEN_H
#define ESHKOL_CODEGEN_H

#include "core/memory.h"
#include "core/diagnostics.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Code generator context
 * 
 * Manages the state for code generation.
 */
typedef struct CodegenContext CodegenContext;

/**
 * @brief Create a code generator context
 * 
 * @param arena Arena for allocations
 * @param diagnostics Diagnostic context for error reporting
 * @return A new code generator context, or NULL on failure
 */
CodegenContext* codegen_context_create(Arena* arena, DiagnosticContext* diagnostics);

/**
 * @brief Generate C code from an AST
 * 
 * @param context The code generator context
 * @param ast The AST to generate code from
 * @param output_file The output file path (can be NULL for stdout)
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate(CodegenContext* context, const AstNode* ast, const char* output_file);

/**
 * @brief Compile and execute the generated code
 * 
 * @param context The code generator context
 * @param c_file The C file to compile
 * @param args Command-line arguments for the program
 * @param argc Number of arguments
 * @return The exit code of the program, or -1 on failure
 */
int codegen_compile_and_execute(CodegenContext* context, const char* c_file, char** args, int argc);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_H */
