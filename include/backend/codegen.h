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
#include "frontend/type_inference/type_inference.h"
#include <stdio.h>
#include <stdbool.h>

/* Include all codegen system components */
#include "backend/codegen/blocks.h"
#include "backend/codegen/calls.h"
#include "backend/codegen/compiler.h"
#include "backend/codegen/conditionals.h"
#include "backend/codegen/context.h"
#include "backend/codegen/definitions.h"
#include "backend/codegen/expressions.h"
#include "backend/codegen/identifiers.h"
#include "backend/codegen/literals.h"
#include "backend/codegen/program.h"
#include "backend/codegen/statements.h"
#include "backend/codegen/type_conversion.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Code generator context structure
 */
typedef struct CodegenContext CodegenContext;

/**
 * @brief Create a code generator context
 * 
 * @param arena Arena for allocations
 * @param diagnostics Diagnostic context for error reporting
 * @param type_context Type inference context for type information
 * @return A new code generator context, or NULL on failure
 */
CodegenContext* codegen_context_create(Arena* arena, DiagnosticContext* diagnostics, TypeInferenceContext* type_context);

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
 * @param args Arguments to pass to the program
 * @param argc Number of arguments
 * @return The exit code of the program, or -1 on failure
 */
int codegen_compile_and_execute(CodegenContext* context, const char* c_file, char** args, int argc);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_H */
