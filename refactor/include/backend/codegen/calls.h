/**
 * @file calls.h
 * @brief Function call code generation
 */

#ifndef ESHKOL_CODEGEN_CALLS_H
#define ESHKOL_CODEGEN_CALLS_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code for a function call
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_call(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for an arithmetic operator call
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @param op_name The operator name
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_arithmetic_op(CodegenContext* context, const AstNode* node, const char* op_name);

/**
 * @brief Generate C code for a comparison operator call
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @param op_name The operator name
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_comparison_op(CodegenContext* context, const AstNode* node, const char* op_name);

/**
 * @brief Generate C code for a vector operation call
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @param op_name The operator name
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_vector_op(CodegenContext* context, const AstNode* node, const char* op_name);

/**
 * @brief Generate C code for an autodiff operation call
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @param op_name The operator name
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_autodiff_op(CodegenContext* context, const AstNode* node, const char* op_name);

/**
 * @brief Generate C code for a standard library function call
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @param func_name The function name
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_stdlib_call(CodegenContext* context, const AstNode* node, const char* func_name);

/**
 * @brief Generate C code for a regular function call
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_regular_call(CodegenContext* context, const AstNode* node);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_CALLS_H */
