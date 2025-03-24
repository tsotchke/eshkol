/**
 * @file program.h
 * @brief Program code generation interface
 */

#ifndef ESHKOL_PROGRAM_H
#define ESHKOL_PROGRAM_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate C code for a program
 * 
 * @param context The code generation context
 * @param program The program AST node
 * @return true if successful, false otherwise
 */
bool codegen_generate_program(CodegenContext* context, const AstNode* program);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PROGRAM_H */
