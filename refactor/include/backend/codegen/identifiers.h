/**
 * @file identifiers.h
 * @brief Identifier code generation
 */

#ifndef ESHKOL_CODEGEN_IDENTIFIERS_H
#define ESHKOL_CODEGEN_IDENTIFIERS_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code for an identifier
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_identifier(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a variable reference
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_variable_ref(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a function reference
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_function_ref(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a symbol
 * 
 * @param context The code generator context
 * @param node The AST node to generate code for
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_symbol(CodegenContext* context, const AstNode* node);

/**
 * @brief Convert an Eshkol identifier to a valid C identifier
 * 
 * @param context The code generator context
 * @param identifier The Eshkol identifier to convert
 * @return The converted C identifier
 */
const char* codegen_convert_identifier(CodegenContext* context, const char* identifier);

/**
 * @brief Check if an identifier is a reserved C keyword
 * 
 * @param identifier The identifier to check
 * @return true if the identifier is a reserved C keyword, false otherwise
 */
bool codegen_is_reserved_keyword(const char* identifier);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_IDENTIFIERS_H */
