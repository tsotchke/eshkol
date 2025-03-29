/**
 * @file ast_print.c
 * @brief Functions for printing and visualizing AST nodes
 */

#include "frontend/ast/print/ast_print.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Print indentation
 * 
 * @param indent Indentation level
 */
static void print_indent(int indent) {
    for (int i = 0; i < indent; i++) {
        printf("  ");
    }
}

/**
 * @brief Print an AST node
 * 
 * @param node The AST node
 * @param indent Indentation level
 */
void ast_print(const AstNode* node, int indent) {
    if (!node) {
        print_indent(indent);
        printf("NULL\n");
        return;
    }
    
    print_indent(indent);
    printf("%s", ast_node_type_to_string(node->type));
    
    switch (node->type) {
        case AST_LITERAL_NUMBER:
            printf(" %g", node->as.number.value);
            break;
        case AST_LITERAL_BOOLEAN:
            printf(" %s", node->as.boolean.value ? "#t" : "#f");
            break;
        case AST_LITERAL_CHARACTER:
            printf(" '%c'", node->as.character.value);
            break;
        case AST_LITERAL_STRING:
            printf(" \"%s\"", node->as.string.value);
            break;
        case AST_LITERAL_VECTOR:
            printf(" [%zu]", node->as.vector.count);
            break;
        case AST_IDENTIFIER:
            printf(" %s", node->as.identifier.name);
            break;
        case AST_TYPE_DECLARATION:
            printf(" %s", node->as.type_declaration.function_name);
            break;
        case AST_ERROR:
            printf(" \"%s\"", node->as.error.message);
            break;
        default:
            break;
    }
    
    // Print type information if available
    printf(" (line %zu, column %zu)\n", node->line, node->column);
    
    // Recursively print children
    switch (node->type) {
        case AST_LITERAL_VECTOR:
            for (size_t i = 0; i < node->as.vector.count; i++) {
                ast_print(node->as.vector.elements[i], indent + 1);
            }
            break;
        case AST_DEFINE:
            ast_print(node->as.define.name, indent + 1);
            ast_print(node->as.define.value, indent + 1);
            break;
        case AST_LAMBDA:
            print_indent(indent + 1);
            printf("Parameters:\n");
            for (size_t i = 0; i < node->as.lambda.param_count; i++) {
                ast_print((AstNode*)node->as.lambda.params[i], indent + 2);
            }
            print_indent(indent + 1);
            printf("Body:\n");
            ast_print(node->as.lambda.body, indent + 2);
            break;
        case AST_IF:
            print_indent(indent + 1);
            printf("Condition:\n");
            ast_print(node->as.if_expr.condition, indent + 2);
            print_indent(indent + 1);
            printf("Then:\n");
            ast_print(node->as.if_expr.then_branch, indent + 2);
            print_indent(indent + 1);
            printf("Else:\n");
            ast_print(node->as.if_expr.else_branch, indent + 2);
            break;
        case AST_BEGIN:
            for (size_t i = 0; i < node->as.begin.expr_count; i++) {
                ast_print(node->as.begin.exprs[i], indent + 1);
            }
            break;
        case AST_QUOTE:
        case AST_QUASIQUOTE:
        case AST_UNQUOTE:
        case AST_UNQUOTE_SPLICING:
        case AST_DELAY:
            ast_print(node->as.quote.expr, indent + 1);
            break;
        case AST_SET:
            ast_print(node->as.set.name, indent + 1);
            ast_print(node->as.set.value, indent + 1);
            break;
        case AST_LET:
        case AST_LETREC:
        case AST_LETSTAR:
            print_indent(indent + 1);
            printf("Bindings:\n");
            for (size_t i = 0; i < node->as.let.binding_count; i++) {
                ast_print(node->as.let.bindings[i], indent + 2);
            }
            print_indent(indent + 1);
            printf("Body:\n");
            ast_print(node->as.let.body, indent + 2);
            break;
        case AST_AND:
        case AST_OR:
            for (size_t i = 0; i < node->as.logical.expr_count; i++) {
                ast_print(node->as.logical.exprs[i], indent + 1);
            }
            break;
        case AST_COND:
            for (size_t i = 0; i < node->as.cond.clause_count; i++) {
                ast_print(node->as.cond.clauses[i], indent + 1);
            }
            break;
        case AST_CASE:
            print_indent(indent + 1);
            printf("Key:\n");
            ast_print(node->as.case_expr.key, indent + 2);
            print_indent(indent + 1);
            printf("Clauses:\n");
            for (size_t i = 0; i < node->as.case_expr.clause_count; i++) {
                ast_print(node->as.case_expr.clauses[i], indent + 2);
            }
            break;
        case AST_DO:
            print_indent(indent + 1);
            printf("Bindings:\n");
            for (size_t i = 0; i < node->as.do_expr.binding_count; i++) {
                ast_print(node->as.do_expr.bindings[i], indent + 2);
            }
            print_indent(indent + 1);
            printf("Test:\n");
            ast_print(node->as.do_expr.test, indent + 2);
            print_indent(indent + 1);
            printf("Result:\n");
            for (size_t i = 0; i < node->as.do_expr.result_count; i++) {
                ast_print(node->as.do_expr.result[i], indent + 2);
            }
            print_indent(indent + 1);
            printf("Body:\n");
            for (size_t i = 0; i < node->as.do_expr.body_count; i++) {
                ast_print(node->as.do_expr.body[i], indent + 2);
            }
            break;
        case AST_CALL:
            print_indent(indent + 1);
            printf("Callee:\n");
            ast_print(node->as.call.callee, indent + 2);
            print_indent(indent + 1);
            printf("Arguments:\n");
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                ast_print(node->as.call.args[i], indent + 2);
            }
            break;
        case AST_SEQUENCE:
            for (size_t i = 0; i < node->as.sequence.expr_count; i++) {
                ast_print(node->as.sequence.exprs[i], indent + 1);
            }
            break;
        case AST_FUNCTION_DEF:
            print_indent(indent + 1);
            printf("Name:\n");
            ast_print(node->as.function_def.name, indent + 2);
            print_indent(indent + 1);
            printf("Parameters:\n");
            for (size_t i = 0; i < node->as.function_def.param_count; i++) {
                ast_print((AstNode*)node->as.function_def.params[i], indent + 2);
            }
            print_indent(indent + 1);
            printf("Body:\n");
            ast_print(node->as.function_def.body, indent + 2);
            break;
        case AST_VARIABLE_DEF:
            print_indent(indent + 1);
            printf("Name:\n");
            ast_print(node->as.variable_def.name, indent + 2);
            print_indent(indent + 1);
            printf("Value:\n");
            ast_print(node->as.variable_def.value, indent + 2);
            break;
        case AST_PROGRAM:
            for (size_t i = 0; i < node->as.program.expr_count; i++) {
                ast_print(node->as.program.exprs[i], indent + 1);
            }
            break;
        case AST_BINDING_PAIR:
            print_indent(indent + 1);
            printf("Name:\n");
            ast_print(node->as.binding_pair.name, indent + 2);
            print_indent(indent + 1);
            printf("Value:\n");
            ast_print(node->as.binding_pair.value, indent + 2);
            break;
        case AST_COND_CLAUSE:
            print_indent(indent + 1);
            printf("Test:\n");
            ast_print(node->as.cond_clause.test, indent + 2);
            print_indent(indent + 1);
            printf("Result:\n");
            ast_print(node->as.cond_clause.result, indent + 2);
            break;
        case AST_CASE_CLAUSE:
            print_indent(indent + 1);
            printf("Datum:\n");
            ast_print(node->as.case_clause.datum, indent + 2);
            print_indent(indent + 1);
            printf("Expression:\n");
            ast_print(node->as.case_clause.expr, indent + 2);
            break;
        default:
            break;
    }
}

/**
 * @brief Visualize an AST node in a graph format
 * 
 * @param ast The AST node to visualize
 * @param format The output format ("dot" or "mermaid")
 */
void ast_visualize(AstNode* ast, const char* format) {
    // Delegate to the ast_visualize.c implementation
    extern void ast_visualize_impl(AstNode* ast, const char* format);
    ast_visualize_impl(ast, format);
}
