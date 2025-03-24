/**
 * @file blocks.h
 * @brief Block code generation
 */

#ifndef ESHKOL_CODEGEN_BLOCKS_H
#define ESHKOL_CODEGEN_BLOCKS_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code for a begin expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_begin(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a let expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_let(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a letrec expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_letrec(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a let* expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_letstar(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a sequence of expressions
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_sequence(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a do expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_do(CodegenContext* context, const AstNode* node);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_BLOCKS_H */
