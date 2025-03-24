/**
 * @file expressions.h
 * @brief Expression code generation
 */

#ifndef ESHKOL_CODEGEN_EXPRESSIONS_H
#define ESHKOL_CODEGEN_EXPRESSIONS_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code for an expression
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_expression(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a number literal
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_number_literal(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a boolean literal
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_boolean_literal(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a character literal
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_character_literal(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a string literal
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_string_literal(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a vector literal
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_vector_literal(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a nil literal
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_nil_literal(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for an identifier
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_identifier(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a begin expression
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_begin(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for an if expression
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_if(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a lambda expression
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_lambda(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a let expression
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_let(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a sequence of expressions
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_sequence(CodegenContext* context, const AstNode* node);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_EXPRESSIONS_H */
