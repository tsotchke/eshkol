/**
 * @file closures.h
 * @brief Closure code generation interface
 */

#ifndef ESHKOL_CODEGEN_CLOSURES_H
#define ESHKOL_CODEGEN_CLOSURES_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>
#include <stdio.h>

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

/**
 * @brief Helper function to detect if a node is a function composition
 * 
 * @param node The AST node to check
 * @return true if the node is a function composition, false otherwise
 */
bool is_function_composition(const AstNode* node);

/**
 * @brief Helper function to handle direct function composition calls
 * 
 * @param context The code generation context
 * @param node The call AST node
 * @param output The output file
 * @return true on success, false on failure
 */
bool codegen_handle_direct_composition(CodegenContext* context, const AstNode* node, FILE* output);

/**
 * @brief Helper function to handle function composition calls
 * 
 * @param context The code generation context
 * @param node The call AST node
 * @param output The output file
 * @return true on success, false on failure
 */
bool codegen_handle_composition_call(CodegenContext* context, const AstNode* node, FILE* output);

/**
 * @brief Helper function to handle compose-n function calls
 * 
 * @param context The code generation context
 * @param node The call AST node
 * @param output The output file
 * @return true on success, false on failure
 */
bool codegen_handle_compose_n(CodegenContext* context, const AstNode* node, FILE* output);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_CLOSURES_H */
