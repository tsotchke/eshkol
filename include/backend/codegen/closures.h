/**
 * @file closures.h
 * @brief Closure code generation interface
 */

#ifndef ESHKOL_CODEGEN_CLOSURES_H
#define ESHKOL_CODEGEN_CLOSURES_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code for a lambda expression (closure implementation)
 * 
 * @param context The code generation context
 * @param node The lambda AST node
 * @return true on success, false on failure
 */
bool codegen_generate_closure(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a closure call
 * 
 * @param context The code generation context
 * @param node The call AST node
 * @return true on success, false on failure
 */
bool codegen_generate_closure_call(CodegenContext* context, const AstNode* node);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_CLOSURES_H */
