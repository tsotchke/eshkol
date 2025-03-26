#include "frontend/ast/ast.h"
#include <stdio.h>
#include <string.h>

static void print_mermaid_node(AstNode* node, int id) {
    printf("  node%d[\"", id);
    switch (node->type) {
        case AST_LITERAL_NUMBER:
            printf("Number\\n%g", node->as.number.value);
            break;
        case AST_LITERAL_BOOLEAN:
            printf("Boolean\\n%s", node->as.boolean.value ? "true" : "false");
            break;
        case AST_LITERAL_CHARACTER:
            printf("Character\\n'%c'", node->as.character.value);
            break;
        case AST_LITERAL_STRING:
            printf("String\\n\\\"%s\\\"", node->as.string.value);
            break;
        case AST_LITERAL_VECTOR:
            printf("Vector\\n[%zu]", node->as.vector.count);
            break;
        case AST_LITERAL_NIL:
            printf("Nil");
            break;
        case AST_IDENTIFIER:
            printf("Identifier\\n%s", node->as.identifier.name);
            break;
        case AST_DEFINE:
            printf("Define");
            break;
        case AST_LAMBDA:
            printf("Lambda\\n[%zu params]", node->as.lambda.param_count);
            break;
        case AST_IF:
            printf("If");
            break;
        case AST_BEGIN:
            printf("Begin\\n[%zu exprs]", node->as.begin.expr_count);
            break;
        case AST_QUOTE:
            printf("Quote");
            break;
        case AST_SET:
            printf("Set!");
            break;
        case AST_LET:
            printf("Let\\n[%zu bindings]", node->as.let.binding_count);
            break;
        case AST_LETREC:
            printf("Letrec\\n[%zu bindings]", node->as.let.binding_count);
            break;
        case AST_LETSTAR:
            printf("Let*\\n[%zu bindings]", node->as.let.binding_count);
            break;
        case AST_AND:
            printf("And\\n[%zu exprs]", node->as.logical.expr_count);
            break;
        case AST_OR:
            printf("Or\\n[%zu exprs]", node->as.logical.expr_count);
            break;
        case AST_COND:
            printf("Cond\\n[%zu clauses]", node->as.cond.clause_count);
            break;
        case AST_CASE:
            printf("Case\\n[%zu clauses]", node->as.case_expr.clause_count);
            break;
        case AST_DO:
            printf("Do\\n[%zu bindings]", node->as.do_expr.binding_count);
            break;
        case AST_DELAY:
            printf("Delay");
            break;
        case AST_QUASIQUOTE:
            printf("Quasiquote");
            break;
        case AST_UNQUOTE:
            printf("Unquote");
            break;
        case AST_UNQUOTE_SPLICING:
            printf("Unquote-splicing");
            break;
        case AST_CALL:
            printf("Call\\n[%zu args]", node->as.call.arg_count);
            break;
        case AST_SEQUENCE:
            printf("Sequence\\n[%zu exprs]", node->as.sequence.expr_count);
            break;
        case AST_FUNCTION_DEF:
            printf("Function\\n[%zu params]", node->as.function_def.param_count);
            break;
        case AST_VARIABLE_DEF:
            printf("Variable");
            break;
        case AST_TYPE_DECLARATION:
            printf("Type\\n%s", node->as.type_declaration.function_name);
            break;
        case AST_PROGRAM:
            printf("Program\\n[%zu exprs]", node->as.program.expr_count);
            break;
        case AST_ERROR:
            printf("Error\\n%s", node->as.error.message);
            break;
        default:
            printf("Unknown\\n%d", node->type);
    }
    printf("\"]\n");
}

static void print_mermaid_edges(AstNode* node, int parent_id, int* current_id) {
    int my_id = (*current_id)++;
    print_mermaid_node(node, my_id);
    
    if (parent_id >= 0) {
        printf("  node%d --> node%d\n", parent_id, my_id);
    }
    
    // Recursively process children
    switch (node->type) {
        case AST_LITERAL_VECTOR:
            for (size_t i = 0; i < node->as.vector.count; i++) {
                print_mermaid_edges(node->as.vector.elements[i], my_id, current_id);
            }
            break;
        case AST_DEFINE:
            print_mermaid_edges(node->as.define.name, my_id, current_id);
            print_mermaid_edges(node->as.define.value, my_id, current_id);
            break;
        case AST_LAMBDA:
            print_mermaid_edges(node->as.lambda.body, my_id, current_id);
            break;
        case AST_IF:
            print_mermaid_edges(node->as.if_expr.condition, my_id, current_id);
            print_mermaid_edges(node->as.if_expr.then_branch, my_id, current_id);
            if (node->as.if_expr.else_branch) {
                print_mermaid_edges(node->as.if_expr.else_branch, my_id, current_id);
            }
            break;
        case AST_BEGIN:
            for (size_t i = 0; i < node->as.begin.expr_count; i++) {
                print_mermaid_edges(node->as.begin.exprs[i], my_id, current_id);
            }
            break;
        case AST_QUOTE:
            print_mermaid_edges(node->as.quote.expr, my_id, current_id);
            break;
        case AST_SET:
            print_mermaid_edges(node->as.set.name, my_id, current_id);
            print_mermaid_edges(node->as.set.value, my_id, current_id);
            break;
        case AST_LET:
        case AST_LETREC:
        case AST_LETSTAR:
            for (size_t i = 0; i < node->as.let.binding_count; i++) {
                print_mermaid_edges(node->as.let.bindings[i], my_id, current_id);
            }
            print_mermaid_edges(node->as.let.body, my_id, current_id);
            break;
        case AST_AND:
        case AST_OR:
            for (size_t i = 0; i < node->as.logical.expr_count; i++) {
                print_mermaid_edges(node->as.logical.exprs[i], my_id, current_id);
            }
            break;
        case AST_COND:
            for (size_t i = 0; i < node->as.cond.clause_count; i++) {
                print_mermaid_edges(node->as.cond.clauses[i], my_id, current_id);
            }
            break;
        case AST_CASE:
            print_mermaid_edges(node->as.case_expr.key, my_id, current_id);
            for (size_t i = 0; i < node->as.case_expr.clause_count; i++) {
                print_mermaid_edges(node->as.case_expr.clauses[i], my_id, current_id);
            }
            break;
        case AST_DO:
            for (size_t i = 0; i < node->as.do_expr.binding_count; i++) {
                print_mermaid_edges(node->as.do_expr.bindings[i], my_id, current_id);
            }
            print_mermaid_edges(node->as.do_expr.test, my_id, current_id);
            for (size_t i = 0; i < node->as.do_expr.result_count; i++) {
                print_mermaid_edges(node->as.do_expr.result[i], my_id, current_id);
            }
            for (size_t i = 0; i < node->as.do_expr.body_count; i++) {
                print_mermaid_edges(node->as.do_expr.body[i], my_id, current_id);
            }
            break;
        case AST_DELAY:
            print_mermaid_edges(node->as.delay.expr, my_id, current_id);
            break;
        case AST_QUASIQUOTE:
            print_mermaid_edges(node->as.quasiquote.expr, my_id, current_id);
            break;
        case AST_UNQUOTE:
            print_mermaid_edges(node->as.unquote.expr, my_id, current_id);
            break;
        case AST_UNQUOTE_SPLICING:
            print_mermaid_edges(node->as.unquote_splicing.expr, my_id, current_id);
            break;
        case AST_CALL:
            print_mermaid_edges(node->as.call.callee, my_id, current_id);
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                print_mermaid_edges(node->as.call.args[i], my_id, current_id);
            }
            break;
        case AST_SEQUENCE:
            for (size_t i = 0; i < node->as.sequence.expr_count; i++) {
                print_mermaid_edges(node->as.sequence.exprs[i], my_id, current_id);
            }
            break;
        case AST_FUNCTION_DEF:
            print_mermaid_edges(node->as.function_def.name, my_id, current_id);
            print_mermaid_edges(node->as.function_def.body, my_id, current_id);
            break;
        case AST_VARIABLE_DEF:
            print_mermaid_edges(node->as.variable_def.name, my_id, current_id);
            print_mermaid_edges(node->as.variable_def.value, my_id, current_id);
            break;
        case AST_PROGRAM:
            for (size_t i = 0; i < node->as.program.expr_count; i++) {
                print_mermaid_edges(node->as.program.exprs[i], my_id, current_id);
            }
            break;
    }
}

static void print_dot_node(AstNode* node, int id) {
    printf("  node%d [label=\"", id);
    switch (node->type) {
        case AST_LITERAL_NUMBER:
            printf("Number\\n%g", node->as.number.value);
            break;
        case AST_LITERAL_BOOLEAN:
            printf("Boolean\\n%s", node->as.boolean.value ? "true" : "false");
            break;
        case AST_LITERAL_CHARACTER:
            printf("Character\\n'%c'", node->as.character.value);
            break;
        case AST_LITERAL_STRING:
            printf("String\\n\\\"%s\\\"", node->as.string.value);
            break;
        case AST_LITERAL_VECTOR:
            printf("Vector\\n[%zu]", node->as.vector.count);
            break;
        case AST_LITERAL_NIL:
            printf("Nil");
            break;
        case AST_IDENTIFIER:
            printf("Identifier\\n%s", node->as.identifier.name);
            break;
        case AST_DEFINE:
            printf("Define");
            break;
        case AST_LAMBDA:
            printf("Lambda\\n[%zu params]", node->as.lambda.param_count);
            break;
        case AST_IF:
            printf("If");
            break;
        case AST_BEGIN:
            printf("Begin\\n[%zu exprs]", node->as.begin.expr_count);
            break;
        case AST_QUOTE:
            printf("Quote");
            break;
        case AST_SET:
            printf("Set!");
            break;
        case AST_LET:
            printf("Let\\n[%zu bindings]", node->as.let.binding_count);
            break;
        case AST_LETREC:
            printf("Letrec\\n[%zu bindings]", node->as.let.binding_count);
            break;
        case AST_LETSTAR:
            printf("Let*\\n[%zu bindings]", node->as.let.binding_count);
            break;
        case AST_AND:
            printf("And\\n[%zu exprs]", node->as.logical.expr_count);
            break;
        case AST_OR:
            printf("Or\\n[%zu exprs]", node->as.logical.expr_count);
            break;
        case AST_COND:
            printf("Cond\\n[%zu clauses]", node->as.cond.clause_count);
            break;
        case AST_CASE:
            printf("Case\\n[%zu clauses]", node->as.case_expr.clause_count);
            break;
        case AST_DO:
            printf("Do\\n[%zu bindings]", node->as.do_expr.binding_count);
            break;
        case AST_DELAY:
            printf("Delay");
            break;
        case AST_QUASIQUOTE:
            printf("Quasiquote");
            break;
        case AST_UNQUOTE:
            printf("Unquote");
            break;
        case AST_UNQUOTE_SPLICING:
            printf("Unquote-splicing");
            break;
        case AST_CALL:
            printf("Call\\n[%zu args]", node->as.call.arg_count);
            break;
        case AST_SEQUENCE:
            printf("Sequence\\n[%zu exprs]", node->as.sequence.expr_count);
            break;
        case AST_FUNCTION_DEF:
            printf("Function\\n[%zu params]", node->as.function_def.param_count);
            break;
        case AST_VARIABLE_DEF:
            printf("Variable");
            break;
        case AST_TYPE_DECLARATION:
            printf("Type\\n%s", node->as.type_declaration.function_name);
            break;
        case AST_PROGRAM:
            printf("Program\\n[%zu exprs]", node->as.program.expr_count);
            break;
        case AST_ERROR:
            printf("Error\\n%s", node->as.error.message);
            break;
        default:
            printf("Unknown\\n%d", node->type);
    }
    printf("\"];\n");
}

static void print_dot_edges(AstNode* node, int parent_id, int* current_id) {
    int my_id = (*current_id)++;
    print_dot_node(node, my_id);
    
    if (parent_id >= 0) {
        printf("  node%d -> node%d;\n", parent_id, my_id);
    }
    
    // Recursively process children
    switch (node->type) {
        case AST_LITERAL_VECTOR:
            for (size_t i = 0; i < node->as.vector.count; i++) {
                print_dot_edges(node->as.vector.elements[i], my_id, current_id);
            }
            break;
        case AST_DEFINE:
            print_dot_edges(node->as.define.name, my_id, current_id);
            print_dot_edges(node->as.define.value, my_id, current_id);
            break;
        case AST_LAMBDA:
            print_dot_edges(node->as.lambda.body, my_id, current_id);
            break;
        case AST_IF:
            print_dot_edges(node->as.if_expr.condition, my_id, current_id);
            print_dot_edges(node->as.if_expr.then_branch, my_id, current_id);
            if (node->as.if_expr.else_branch) {
                print_dot_edges(node->as.if_expr.else_branch, my_id, current_id);
            }
            break;
        case AST_BEGIN:
            for (size_t i = 0; i < node->as.begin.expr_count; i++) {
                print_dot_edges(node->as.begin.exprs[i], my_id, current_id);
            }
            break;
        case AST_QUOTE:
            print_dot_edges(node->as.quote.expr, my_id, current_id);
            break;
        case AST_SET:
            print_dot_edges(node->as.set.name, my_id, current_id);
            print_dot_edges(node->as.set.value, my_id, current_id);
            break;
        case AST_LET:
        case AST_LETREC:
        case AST_LETSTAR:
            for (size_t i = 0; i < node->as.let.binding_count; i++) {
                print_dot_edges(node->as.let.bindings[i], my_id, current_id);
            }
            print_dot_edges(node->as.let.body, my_id, current_id);
            break;
        case AST_AND:
        case AST_OR:
            for (size_t i = 0; i < node->as.logical.expr_count; i++) {
                print_dot_edges(node->as.logical.exprs[i], my_id, current_id);
            }
            break;
        case AST_COND:
            for (size_t i = 0; i < node->as.cond.clause_count; i++) {
                print_dot_edges(node->as.cond.clauses[i], my_id, current_id);
            }
            break;
        case AST_CASE:
            print_dot_edges(node->as.case_expr.key, my_id, current_id);
            for (size_t i = 0; i < node->as.case_expr.clause_count; i++) {
                print_dot_edges(node->as.case_expr.clauses[i], my_id, current_id);
            }
            break;
        case AST_DO:
            for (size_t i = 0; i < node->as.do_expr.binding_count; i++) {
                print_dot_edges(node->as.do_expr.bindings[i], my_id, current_id);
            }
            print_dot_edges(node->as.do_expr.test, my_id, current_id);
            for (size_t i = 0; i < node->as.do_expr.result_count; i++) {
                print_dot_edges(node->as.do_expr.result[i], my_id, current_id);
            }
            for (size_t i = 0; i < node->as.do_expr.body_count; i++) {
                print_dot_edges(node->as.do_expr.body[i], my_id, current_id);
            }
            break;
        case AST_DELAY:
            print_dot_edges(node->as.delay.expr, my_id, current_id);
            break;
        case AST_QUASIQUOTE:
            print_dot_edges(node->as.quasiquote.expr, my_id, current_id);
            break;
        case AST_UNQUOTE:
            print_dot_edges(node->as.unquote.expr, my_id, current_id);
            break;
        case AST_UNQUOTE_SPLICING:
            print_dot_edges(node->as.unquote_splicing.expr, my_id, current_id);
            break;
        case AST_CALL:
            print_dot_edges(node->as.call.callee, my_id, current_id);
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                print_dot_edges(node->as.call.args[i], my_id, current_id);
            }
            break;
        case AST_SEQUENCE:
            for (size_t i = 0; i < node->as.sequence.expr_count; i++) {
                print_dot_edges(node->as.sequence.exprs[i], my_id, current_id);
            }
            break;
        case AST_FUNCTION_DEF:
            print_dot_edges(node->as.function_def.name, my_id, current_id);
            print_dot_edges(node->as.function_def.body, my_id, current_id);
            break;
        case AST_VARIABLE_DEF:
            print_dot_edges(node->as.variable_def.name, my_id, current_id);
            print_dot_edges(node->as.variable_def.value, my_id, current_id);
            break;
        case AST_PROGRAM:
            for (size_t i = 0; i < node->as.program.expr_count; i++) {
                print_dot_edges(node->as.program.exprs[i], my_id, current_id);
            }
            break;
    }
}

void ast_visualize(AstNode* ast, const char* format) {
    if (strcmp(format, "mermaid") == 0) {
        printf("graph TD\n");
        int current_id = 0;
        print_mermaid_edges(ast, -1, &current_id);
    } else if (strcmp(format, "dot") == 0) {
        printf("digraph AST {\n");
        printf("  node [shape=box];\n");
        int current_id = 0;
        print_dot_edges(ast, -1, &current_id);
        printf("}\n");
    }
}
