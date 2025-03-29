/**
 * @file calls.h
 * @brief Function call code generation interface
 */

#ifndef ESHKOL_CALLS_H
#define ESHKOL_CALLS_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code for a function call
 * 
 * @param context The code generation context
 * @param node The AST node representing the function call
 * @return true if successful, false otherwise
 */
bool codegen_generate_call(CodegenContext* context, const AstNode* node);

/**
 * @brief Generate C code for a tail call
 * 
 * This function generates optimized code for tail calls, converting
 * recursive tail calls into loops to avoid stack overflow.
 * 
 * @param context The code generation context
 * @param node The AST node representing the tail call
 * @return true if successful, false otherwise
 */
bool codegen_generate_tail_call(CodegenContext* context, const AstNode* node);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CALLS_H */
