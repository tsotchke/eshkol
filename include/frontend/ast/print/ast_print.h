/**
 * @file ast_print.h
 * @brief Functions for printing and visualizing AST nodes
 */

#ifndef ESHKOL_AST_PRINT_H
#define ESHKOL_AST_PRINT_H

#include "frontend/ast/core/ast_core.h"

/**
 * @brief Print an AST node
 * 
 * @param node The AST node
 * @param indent Indentation level
 */
void ast_print(const AstNode* node, int indent);

/**
 * @brief Visualize an AST node in a graph format
 * 
 * @param ast The AST node to visualize
 * @param format The output format ("dot" or "mermaid")
 */
void ast_visualize(AstNode* ast, const char* format);

#endif /* ESHKOL_AST_PRINT_H */
