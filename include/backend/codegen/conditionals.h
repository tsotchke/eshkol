/**
 * @file conditionals.h
 * @brief Conditional code generation
 */

#ifndef ESHKOL_CODEGEN_CONDITIONALS_H
#define ESHKOL_CODEGEN_CONDITIONALS_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code for an if expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_if(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a cond expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_cond(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a case expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_case(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for an and expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_and(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for an or expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_or(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate code for an if expression that returns a string
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return The generated code as a string, or NULL on failure
 */
char* codegen_generate_if_expr(CodegenContext* context, const AstNode* node);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_CONDITIONALS_H */
