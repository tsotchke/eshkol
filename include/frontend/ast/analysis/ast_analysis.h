/**
 * @file ast_analysis.h
 * @brief Functions for analyzing AST nodes
 */

#ifndef ESHKOL_AST_ANALYSIS_H
#define ESHKOL_AST_ANALYSIS_H

#include "frontend/ast/core/ast_core.h"

/**
 * @brief Mark tail positions in an AST
 * 
 * This function recursively traverses the AST and marks nodes that are in tail position.
 * A node is in tail position if its value is returned directly from the function.
 * 
 * @param node The AST node to mark
 * @param is_tail Whether the node is in tail position
 */
void ast_mark_tail_positions(AstNode* node, bool is_tail);

/**
 * @brief Mark self-recursive tail calls in an AST
 * 
 * This function identifies and marks self-recursive tail calls in the AST.
 * A self-recursive tail call is a call to the same function in tail position.
 * 
 * @param node The AST node to analyze
 * @param current_function The current function being analyzed (NULL for top-level)
 */
void ast_mark_self_tail_calls(AstNode* node, AstNode* current_function);

#endif /* ESHKOL_AST_ANALYSIS_H */
