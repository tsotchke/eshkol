/**
 * @file ast_parent.h
 * @brief Functions for setting parent pointers in the AST
 */

#ifndef ESHKOL_AST_PARENT_H
#define ESHKOL_AST_PARENT_H

#include "frontend/ast/core/ast_core.h"

/**
 * @brief Set the parent pointers in an AST
 * 
 * @param node The root node
 */
void ast_set_parent_pointers(AstNode* node);

#endif /* ESHKOL_AST_PARENT_H */
