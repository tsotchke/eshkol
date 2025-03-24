/**
 * @file statements.h
 * @brief Statement code generation
 */

#ifndef ESHKOL_CODEGEN_STATEMENTS_H
#define ESHKOL_CODEGEN_STATEMENTS_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code for a statement
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_statement(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a function definition statement
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_function_def_statement(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a variable definition statement
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_variable_def_statement(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for an expression statement
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_expression_statement(CodegenContext* context, const AstNode* node);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_STATEMENTS_H */
