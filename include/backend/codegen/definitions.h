/**
 * @file definitions.h
 * @brief Definition code generation
 */

#ifndef ESHKOL_CODEGEN_DEFINITIONS_H
#define ESHKOL_CODEGEN_DEFINITIONS_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code for a define expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_define(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a function definition
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_function_def(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a variable definition
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_variable_def(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a lambda expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_lambda(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a set! expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_set(CodegenContext* context, const AstNode* node);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_DEFINITIONS_H */
