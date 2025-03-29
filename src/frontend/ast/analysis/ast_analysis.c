/**
 * @file ast_analysis.c
 * @brief Functions for analyzing AST nodes
 */

#include "frontend/ast/analysis/ast_analysis.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Mark tail positions in an AST
 * 
 * This function recursively traverses the AST and marks nodes that are in tail position.
 * A node is in tail position if its value is returned directly from the function.
 * 
 * @param node The AST node to mark
 * @param is_tail Whether the node is in tail position
 */
void ast_mark_tail_positions(AstNode* node, bool is_tail) {
    if (!node) {
        return;
    }
    
    // Mark the node's tail position
    node->is_tail_position = is_tail;
    
    // Recursively mark children based on their context
    switch (node->type) {
        case AST_LAMBDA:
            // The body of a lambda is in tail position
            ast_mark_tail_positions(node->as.lambda.body, true);
            break;
            
        case AST_IF:
            // The condition is not in tail position
            ast_mark_tail_positions(node->as.if_expr.condition, false);
            // The branches are in tail position if the if is in tail position
            ast_mark_tail_positions(node->as.if_expr.then_branch, is_tail);
            ast_mark_tail_positions(node->as.if_expr.else_branch, is_tail);
            break;
            
        case AST_BEGIN:
            // Only the last expression in a begin is in tail position
            for (size_t i = 0; i < node->as.begin.expr_count; i++) {
                ast_mark_tail_positions(node->as.begin.exprs[i], 
                                       is_tail && (i == node->as.begin.expr_count - 1));
            }
            break;
            
        case AST_QUOTE:
        case AST_QUASIQUOTE:
        case AST_UNQUOTE:
        case AST_UNQUOTE_SPLICING:
        case AST_DELAY:
            // The quoted expression is not in tail position
            ast_mark_tail_positions(node->as.quote.expr, false);
            break;
            
        case AST_SET:
            // Neither the name nor the value are in tail position
            ast_mark_tail_positions(node->as.set.name, false);
            ast_mark_tail_positions(node->as.set.value, false);
            break;
            
        case AST_LET:
        case AST_LETREC:
        case AST_LETSTAR:
            // Bindings are not in tail position
            for (size_t i = 0; i < node->as.let.binding_count; i++) {
                ast_mark_tail_positions(node->as.let.bindings[i], false);
            }
            // The body is in tail position if the let is in tail position
            ast_mark_tail_positions(node->as.let.body, is_tail);
            break;
            
        case AST_AND:
        case AST_OR:
            // Only the last expression in a logical operation is in tail position
            for (size_t i = 0; i < node->as.logical.expr_count; i++) {
                ast_mark_tail_positions(node->as.logical.exprs[i], 
                                       is_tail && (i == node->as.logical.expr_count - 1));
            }
            break;
            
        case AST_COND:
            // All clauses are in tail position if the cond is in tail position
            for (size_t i = 0; i < node->as.cond.clause_count; i++) {
                ast_mark_tail_positions(node->as.cond.clauses[i], is_tail);
            }
            break;
            
        case AST_CASE:
            // The key is not in tail position
            ast_mark_tail_positions(node->as.case_expr.key, false);
            // All clauses are in tail position if the case is in tail position
            for (size_t i = 0; i < node->as.case_expr.clause_count; i++) {
                ast_mark_tail_positions(node->as.case_expr.clauses[i], is_tail);
            }
            break;
            
        case AST_DO:
            // Bindings and steps are not in tail position
            for (size_t i = 0; i < node->as.do_expr.binding_count; i++) {
                ast_mark_tail_positions(node->as.do_expr.bindings[i], false);
                ast_mark_tail_positions(node->as.do_expr.steps[i], false);
            }
            // The test is not in tail position
            ast_mark_tail_positions(node->as.do_expr.test, false);
            // The result expressions are in tail position if the do is in tail position
            for (size_t i = 0; i < node->as.do_expr.result_count; i++) {
                ast_mark_tail_positions(node->as.do_expr.result[i], 
                                       is_tail && (i == node->as.do_expr.result_count - 1));
            }
            // The body expressions are not in tail position
            for (size_t i = 0; i < node->as.do_expr.body_count; i++) {
                ast_mark_tail_positions(node->as.do_expr.body[i], false);
            }
            break;
            
        case AST_CALL:
            // The callee is not in tail position
            ast_mark_tail_positions(node->as.call.callee, false);
            // The arguments are not in tail position
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                ast_mark_tail_positions(node->as.call.args[i], false);
            }
            break;
            
        case AST_SEQUENCE:
            // Only the last expression in a sequence is in tail position
            for (size_t i = 0; i < node->as.sequence.expr_count; i++) {
                ast_mark_tail_positions(node->as.sequence.exprs[i], 
                                       is_tail && (i == node->as.sequence.expr_count - 1));
            }
            break;
            
        case AST_FUNCTION_DEF:
            // The name is not in tail position
            ast_mark_tail_positions(node->as.function_def.name, false);
            // The parameters are not in tail position
            for (size_t i = 0; i < node->as.function_def.param_count; i++) {
                if (node->as.function_def.param_nodes && node->as.function_def.param_nodes[i]) {
                    ast_mark_tail_positions(node->as.function_def.param_nodes[i], false);
                }
            }
            // The body is in tail position
            ast_mark_tail_positions(node->as.function_def.body, true);
            break;
            
        case AST_VARIABLE_DEF:
            // The name is not in tail position
            ast_mark_tail_positions(node->as.variable_def.name, false);
            // The value is not in tail position
            ast_mark_tail_positions(node->as.variable_def.value, false);
            break;
            
        case AST_PROGRAM:
            // Top-level expressions are not in tail position
            for (size_t i = 0; i < node->as.program.expr_count; i++) {
                ast_mark_tail_positions(node->as.program.exprs[i], false);
            }
            break;
            
        default:
            // Other node types don't have children to mark
            break;
    }
}

/**
 * @brief Mark self-recursive tail calls in an AST
 * 
 * This function identifies and marks self-recursive tail calls in the AST.
 * A self-recursive tail call is a call to the same function in tail position.
 * 
 * @param node The AST node to analyze
 * @param current_function The current function being analyzed (NULL for top-level)
 */
void ast_mark_self_tail_calls(AstNode* node, AstNode* current_function) {
    if (!node) {
        return;
    }
    
    // Check if this is a call in tail position
    if (node->type == AST_CALL && node->is_tail_position) {
        // Check if the callee is an identifier
        if (node->as.call.callee->type == AST_IDENTIFIER) {
            // If we're in a function definition, check if the callee is the same function
            if (current_function && current_function->type == AST_FUNCTION_DEF &&
                current_function->as.function_def.name->type == AST_IDENTIFIER) {
                const char* callee_name = node->as.call.callee->as.identifier.name;
                const char* function_name = current_function->as.function_def.name->as.identifier.name;
                
                if (strcmp(callee_name, function_name) == 0) {
                    node->is_self_tail_call = true;
                }
            }
        }
    }
    
    // Recursively process children
    switch (node->type) {
        case AST_LAMBDA:
            // Lambda creates a new function context
            ast_mark_self_tail_calls(node->as.lambda.body, NULL);
            break;
            
        case AST_FUNCTION_DEF:
            // Function definition creates a new function context
            ast_mark_self_tail_calls(node->as.function_def.body, node);
            break;
            
        case AST_IF:
            ast_mark_self_tail_calls(node->as.if_expr.condition, current_function);
            ast_mark_self_tail_calls(node->as.if_expr.then_branch, current_function);
            ast_mark_self_tail_calls(node->as.if_expr.else_branch, current_function);
            break;
            
        case AST_BEGIN:
            for (size_t i = 0; i < node->as.begin.expr_count; i++) {
                ast_mark_self_tail_calls(node->as.begin.exprs[i], current_function);
            }
            break;
            
        case AST_QUOTE:
        case AST_QUASIQUOTE:
        case AST_UNQUOTE:
        case AST_UNQUOTE_SPLICING:
        case AST_DELAY:
            ast_mark_self_tail_calls(node->as.quote.expr, current_function);
            break;
            
        case AST_SET:
            ast_mark_self_tail_calls(node->as.set.name, current_function);
            ast_mark_self_tail_calls(node->as.set.value, current_function);
            break;
            
        case AST_LET:
        case AST_LETREC:
        case AST_LETSTAR:
            for (size_t i = 0; i < node->as.let.binding_count; i++) {
                ast_mark_self_tail_calls(node->as.let.bindings[i], current_function);
            }
            ast_mark_self_tail_calls(node->as.let.body, current_function);
            break;
            
        case AST_AND:
        case AST_OR:
            for (size_t i = 0; i < node->as.logical.expr_count; i++) {
                ast_mark_self_tail_calls(node->as.logical.exprs[i], current_function);
            }
            break;
            
        case AST_COND:
            for (size_t i = 0; i < node->as.cond.clause_count; i++) {
                ast_mark_self_tail_calls(node->as.cond.clauses[i], current_function);
            }
            break;
            
        case AST_CASE:
            ast_mark_self_tail_calls(node->as.case_expr.key, current_function);
            for (size_t i = 0; i < node->as.case_expr.clause_count; i++) {
                ast_mark_self_tail_calls(node->as.case_expr.clauses[i], current_function);
            }
            break;
            
        case AST_DO:
            for (size_t i = 0; i < node->as.do_expr.binding_count; i++) {
                ast_mark_self_tail_calls(node->as.do_expr.bindings[i], current_function);
                ast_mark_self_tail_calls(node->as.do_expr.steps[i], current_function);
            }
            ast_mark_self_tail_calls(node->as.do_expr.test, current_function);
            for (size_t i = 0; i < node->as.do_expr.result_count; i++) {
                ast_mark_self_tail_calls(node->as.do_expr.result[i], current_function);
            }
            for (size_t i = 0; i < node->as.do_expr.body_count; i++) {
                ast_mark_self_tail_calls(node->as.do_expr.body[i], current_function);
            }
            break;
            
        case AST_CALL:
            ast_mark_self_tail_calls(node->as.call.callee, current_function);
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                ast_mark_self_tail_calls(node->as.call.args[i], current_function);
            }
            break;
            
        case AST_SEQUENCE:
            for (size_t i = 0; i < node->as.sequence.expr_count; i++) {
                ast_mark_self_tail_calls(node->as.sequence.exprs[i], current_function);
            }
            break;
            
        case AST_VARIABLE_DEF:
            ast_mark_self_tail_calls(node->as.variable_def.name, current_function);
            ast_mark_self_tail_calls(node->as.variable_def.value, current_function);
            break;
            
        case AST_PROGRAM:
            for (size_t i = 0; i < node->as.program.expr_count; i++) {
                ast_mark_self_tail_calls(node->as.program.exprs[i], NULL);
            }
            break;
            
        default:
            // Other node types don't have children to process
            break;
    }
}
