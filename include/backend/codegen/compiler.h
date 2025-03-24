/**
 * @file compiler.h
 * @brief Top-level compilation functions interface
 */

#ifndef ESHKOL_COMPILER_H
#define ESHKOL_COMPILER_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code from an AST
 * 
 * @param context The code generation context
 * @param ast The AST to generate code from
 * @param output_file The output file path, or NULL for stdout
 * @return true if successful, false otherwise
 */
bool codegen_generate(CodegenContext* context, const AstNode* ast, const char* output_file);

/**
 * @brief Compile and execute the generated code
 * 
 * @param context The code generation context
 * @param c_file The C file to compile
 * @param args The arguments to pass to the program
 * @param argc The number of arguments
 * @return The exit code of the program, or -1 on error
 */
int codegen_compile_and_execute(CodegenContext* context, const char* c_file, char** args, int argc);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_COMPILER_H */
