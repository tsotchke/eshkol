/**
 * @file expressions.h
 * @brief Expression code generation
 */

#ifndef ESHKOL_CODEGEN_EXPRESSIONS_H
#define ESHKOL_CODEGEN_EXPRESSIONS_H

#include "backend/codegen/context.h"
#include "backend/codegen/literals.h"
#include "backend/codegen/identifiers.h"
#include "backend/codegen/calls.h"
#include "backend/codegen/conditionals.h"
#include "backend/codegen/blocks.h"
#include "backend/codegen/definitions.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code for an expression
 * 
 * @param context The code generator context
 * @param node The AST node
 * @return true if code generation succeeded, false otherwise
 */
bool codegen_generate_expression(CodegenContext* context, const AstNode* node);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_EXPRESSIONS_H */
