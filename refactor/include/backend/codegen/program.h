/**
 * @file program.h
 * @brief Program code generation
 */

#ifndef ESHKOL_CODEGEN_PROGRAM_H
#define ESHKOL_CODEGEN_PROGRAM_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code for a program
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_program(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for program headers
 * 
 * @param context The code generator context
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_program_headers(CodegenContext* context);

/**
 * @brief Generate C code for forward declarations
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_forward_declarations(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a main function
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_main_function(CodegenContext* context, const AstNode* node);

/**
 * @brief Check if a program has a main function
 * 
 * @param node The AST node to check
 * @return true if the program has a main function, false otherwise
 */
bool codegen_program_has_main(const AstNode* node);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_PROGRAM_H */
