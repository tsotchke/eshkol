#ifndef ESHKOL_CODEGEN_DEBUG_H
#define ESHKOL_CODEGEN_DEBUG_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"

/**
 * @brief Debug code generation process
 * 
 * @param context Code generation context
 * @param ast AST to debug
 * @param stage Stage to debug (ast, ir, c-code, all)
 * @return true on success, false on failure
 */
bool codegen_debug(CodegenContext* context, AstNode* ast, const char* stage);

/**
 * @brief Profile code generation process
 * 
 * @param context Code generation context
 * @param ast AST to profile
 * @return true on success, false on failure
 */
bool codegen_profile(CodegenContext* context, AstNode* ast);

#endif /* ESHKOL_CODEGEN_DEBUG_H */
