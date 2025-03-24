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
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_identifier(CodegenContext* context, const AstNode* node);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_IDENTIFIERS_H */
