/**
 * @file ast_parent.c
 * @brief Functions for setting parent pointers in the AST
 */

#include "frontend/ast/core/ast_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Set the parent pointers in an AST
 * 
 * @param node The root node
 */
void ast_set_parent_pointers(AstNode* node) {
    if (!node) {
        return;
    }
    
    // Set parent pointers based on node type
    switch (node->type) {
        case AST_DEFINE:
            if (node->as.define.name) {
                node->as.define.name->parent = node;
                ast_set_parent_pointers(node->as.define.name);
            }
            if (node->as.define.value) {
                node->as.define.value->parent = node;
                ast_set_parent_pointers(node->as.define.value);
            }
            break;
            
        case AST_LAMBDA:
            if (node->as.lambda.body) {
                node->as.lambda.body->parent = node;
                ast_set_parent_pointers(node->as.lambda.body);
            }
            break;
            
        case AST_IF:
            if (node->as.if_expr.condition) {
                node->as.if_expr.condition->parent = node;
                ast_set_parent_pointers(node->as.if_expr.condition);
            }
            if (node->as.if_expr.then_branch) {
                node->as.if_expr.then_branch->parent = node;
                ast_set_parent_pointers(node->as.if_expr.then_branch);
            }
            if (node->as.if_expr.else_branch) {
                node->as.if_expr.else_branch->parent = node;
                ast_set_parent_pointers(node->as.if_expr.else_branch);
            }
            break;
            
        case AST_BEGIN:
            for (size_t i = 0; i < node->as.begin.expr_count; i++) {
                if (node->as.begin.exprs[i]) {
                    node->as.begin.exprs[i]->parent = node;
                    ast_set_parent_pointers(node->as.begin.exprs[i]);
                }
            }
            break;
            
        case AST_QUOTE:
            if (node->as.quote.expr) {
                node->as.quote.expr->parent = node;
                ast_set_parent_pointers(node->as.quote.expr);
            }
            break;
            
        case AST_SET:
            if (node->as.set.name) {
                node->as.set.name->parent = node;
                ast_set_parent_pointers(node->as.set.name);
            }
            if (node->as.set.value) {
                node->as.set.value->parent = node;
                ast_set_parent_pointers(node->as.set.value);
            }
            break;
            
        case AST_LET:
        case AST_LETREC:
        case AST_LETSTAR:
            for (size_t i = 0; i < node->as.let.binding_count; i++) {
                if (node->as.let.bindings[i]) {
                    node->as.let.bindings[i]->parent = node;
                    ast_set_parent_pointers(node->as.let.bindings[i]);
                }
            }
            if (node->as.let.body) {
                node->as.let.body->parent = node;
                ast_set_parent_pointers(node->as.let.body);
            }
            break;
            
        case AST_AND:
        case AST_OR:
            for (size_t i = 0; i < node->as.logical.expr_count; i++) {
                if (node->as.logical.exprs[i]) {
                    node->as.logical.exprs[i]->parent = node;
                    ast_set_parent_pointers(node->as.logical.exprs[i]);
                }
            }
            break;
            
        case AST_COND:
            for (size_t i = 0; i < node->as.cond.clause_count; i++) {
                if (node->as.cond.clauses[i]) {
                    node->as.cond.clauses[i]->parent = node;
                    ast_set_parent_pointers(node->as.cond.clauses[i]);
                }
            }
            break;
            
        case AST_CASE:
            if (node->as.case_expr.key) {
                node->as.case_expr.key->parent = node;
                ast_set_parent_pointers(node->as.case_expr.key);
            }
            for (size_t i = 0; i < node->as.case_expr.clause_count; i++) {
                if (node->as.case_expr.clauses[i]) {
                    node->as.case_expr.clauses[i]->parent = node;
                    ast_set_parent_pointers(node->as.case_expr.clauses[i]);
                }
            }
            break;
            
        case AST_DO:
            for (size_t i = 0; i < node->as.do_expr.binding_count; i++) {
                if (node->as.do_expr.bindings[i]) {
                    node->as.do_expr.bindings[i]->parent = node;
                    ast_set_parent_pointers(node->as.do_expr.bindings[i]);
                }
                if (node->as.do_expr.steps[i]) {
                    node->as.do_expr.steps[i]->parent = node;
                    ast_set_parent_pointers(node->as.do_expr.steps[i]);
                }
            }
            if (node->as.do_expr.test) {
                node->as.do_expr.test->parent = node;
                ast_set_parent_pointers(node->as.do_expr.test);
            }
            for (size_t i = 0; i < node->as.do_expr.result_count; i++) {
                if (node->as.do_expr.result[i]) {
                    node->as.do_expr.result[i]->parent = node;
                    ast_set_parent_pointers(node->as.do_expr.result[i]);
                }
            }
            for (size_t i = 0; i < node->as.do_expr.body_count; i++) {
                if (node->as.do_expr.body[i]) {
                    node->as.do_expr.body[i]->parent = node;
                    ast_set_parent_pointers(node->as.do_expr.body[i]);
                }
            }
            break;
            
        case AST_DELAY:
            if (node->as.delay.expr) {
                node->as.delay.expr->parent = node;
                ast_set_parent_pointers(node->as.delay.expr);
            }
            break;
            
        case AST_QUASIQUOTE:
            if (node->as.quasiquote.expr) {
                node->as.quasiquote.expr->parent = node;
                ast_set_parent_pointers(node->as.quasiquote.expr);
            }
            break;
            
        case AST_UNQUOTE:
            if (node->as.unquote.expr) {
                node->as.unquote.expr->parent = node;
                ast_set_parent_pointers(node->as.unquote.expr);
            }
            break;
            
        case AST_UNQUOTE_SPLICING:
            if (node->as.unquote_splicing.expr) {
                node->as.unquote_splicing.expr->parent = node;
                ast_set_parent_pointers(node->as.unquote_splicing.expr);
            }
            break;
            
        case AST_CALL:
            if (node->as.call.callee) {
                node->as.call.callee->parent = node;
                ast_set_parent_pointers(node->as.call.callee);
            }
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                if (node->as.call.args[i]) {
                    node->as.call.args[i]->parent = node;
                    ast_set_parent_pointers(node->as.call.args[i]);
                }
            }
            break;
            
        case AST_SEQUENCE:
            for (size_t i = 0; i < node->as.sequence.expr_count; i++) {
                if (node->as.sequence.exprs[i]) {
                    node->as.sequence.exprs[i]->parent = node;
                    ast_set_parent_pointers(node->as.sequence.exprs[i]);
                }
            }
            break;
            
        case AST_FUNCTION_DEF:
            if (node->as.function_def.name) {
                node->as.function_def.name->parent = node;
                ast_set_parent_pointers(node->as.function_def.name);
            }
            if (node->as.function_def.body) {
                node->as.function_def.body->parent = node;
                ast_set_parent_pointers(node->as.function_def.body);
            }
            break;
            
        case AST_VARIABLE_DEF:
            if (node->as.variable_def.name) {
                node->as.variable_def.name->parent = node;
                ast_set_parent_pointers(node->as.variable_def.name);
            }
            if (node->as.variable_def.value) {
                node->as.variable_def.value->parent = node;
                ast_set_parent_pointers(node->as.variable_def.value);
            }
            break;
            
        case AST_PROGRAM:
            for (size_t i = 0; i < node->as.program.expr_count; i++) {
                if (node->as.program.exprs[i]) {
                    node->as.program.exprs[i]->parent = node;
                    ast_set_parent_pointers(node->as.program.exprs[i]);
                }
            }
            break;
            
        case AST_BINDING_PAIR:
            if (node->as.binding_pair.name) {
                node->as.binding_pair.name->parent = node;
                ast_set_parent_pointers(node->as.binding_pair.name);
            }
            if (node->as.binding_pair.value) {
                node->as.binding_pair.value->parent = node;
                ast_set_parent_pointers(node->as.binding_pair.value);
            }
            break;
            
        case AST_COND_CLAUSE:
            if (node->as.cond_clause.test) {
                node->as.cond_clause.test->parent = node;
                ast_set_parent_pointers(node->as.cond_clause.test);
            }
            if (node->as.cond_clause.result) {
                node->as.cond_clause.result->parent = node;
                ast_set_parent_pointers(node->as.cond_clause.result);
            }
            break;
            
        case AST_CASE_CLAUSE:
            if (node->as.case_clause.datum) {
                node->as.case_clause.datum->parent = node;
                ast_set_parent_pointers(node->as.case_clause.datum);
            }
            if (node->as.case_clause.expr) {
                node->as.case_clause.expr->parent = node;
                ast_set_parent_pointers(node->as.case_clause.expr);
            }
            break;
            
        case AST_LITERAL_VECTOR:
            for (size_t i = 0; i < node->as.vector.count; i++) {
                if (node->as.vector.elements[i]) {
                    node->as.vector.elements[i]->parent = node;
                    ast_set_parent_pointers(node->as.vector.elements[i]);
                }
            }
            break;
            
        // Leaf nodes (no children)
        case AST_LITERAL_NUMBER:
        case AST_LITERAL_BOOLEAN:
        case AST_LITERAL_CHARACTER:
        case AST_LITERAL_STRING:
        case AST_LITERAL_NIL:
        case AST_IDENTIFIER:
        case AST_TYPE_DECLARATION:
        case AST_ERROR:
            // Nothing to do
            break;
    }
}
