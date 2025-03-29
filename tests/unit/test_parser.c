/**
 * @file test_parser.c
 * @brief Unit tests for the parser
 */

#include "frontend/parser/parser.h"
#include "frontend/lexer/lexer.h"
#include "frontend/ast/ast.h"
#include "core/memory.h"
#include "core/string_table.h"
#include "core/diagnostics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Helper function to create parser components
static void setup_parser(Arena** arena, StringTable** strings, DiagnosticContext** diag, 
                         const char* source, Lexer** lexer, Parser** parser) {
    // Create an arena
    *arena = arena_create(1024);
    assert(*arena != NULL);
    
    // Create a string table
    *strings = string_table_create(*arena, 16);
    assert(*strings != NULL);
    
    // Create a diagnostic context
    *diag = diagnostic_context_create(*arena);
    assert(*diag != NULL);
    
    // Create a lexer
    *lexer = lexer_create(*arena, *strings, *diag, source);
    assert(*lexer != NULL);
    
    // Create a parser
    *parser = parser_create(*arena, *strings, *diag, *lexer);
    assert(*parser != NULL);
}

// Helper function to clean up parser components
static void teardown_parser(Arena* arena) {
    arena_destroy(arena);
}

/**
 * @brief Test parser creation
 */
static void test_parser_create(void) {
    printf("Testing parser creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a string table
    StringTable* strings = string_table_create(arena, 16);
    assert(strings != NULL);
    
    // Create a diagnostic context
    DiagnosticContext* diag = diagnostic_context_create(arena);
    assert(diag != NULL);
    
    // Create a lexer
    const char* source = "(define x 42)";
    Lexer* lexer = lexer_create(arena, strings, diag, source);
    assert(lexer != NULL);
    
    // Create a parser
    Parser* parser = parser_create(arena, strings, diag, lexer);
    assert(parser != NULL);
    assert(parser->arena == arena);
    assert(parser->strings == strings);
    assert(parser->diag == diag);
    assert(parser->lexer == lexer);
    assert(parser->had_error == false);
    assert(parser->panic_mode == false);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: parser_create\n");
}

/**
 * @brief Test parsing a simple program
 */
static void test_parser_parse_program(void) {
    printf("Testing parsing a simple program...\n");
    
    Arena* arena;
    StringTable* strings;
    DiagnosticContext* diag;
    Lexer* lexer;
    Parser* parser;
    
    // Setup parser
    setup_parser(&arena, &strings, &diag, "(define x 42)", &lexer, &parser);
    
    // Parse a program
    AstNode* program = parser_parse_program(parser);
    assert(program != NULL);
    assert(program->type == AST_PROGRAM);
    
    // Print the AST
    printf("\nPrinting program AST:\n");
    ast_print(program, 0);
    
    // Verify the program has one expression
    assert(program->as.program.expr_count == 1);
    
    // Verify the expression is a variable definition
    AstNode* var_def = program->as.program.exprs[0];
    assert(var_def->type == AST_VARIABLE_DEF);
    
    // Verify the variable name is "x"
    AstNode* name = var_def->as.variable_def.name;
    assert(name->type == AST_IDENTIFIER);
    assert(strcmp(name->as.identifier.name, "x") == 0);
    
    // Verify the value is 42
    AstNode* value = var_def->as.variable_def.value;
    assert(value->type == AST_LITERAL_NUMBER);
    assert(value->as.number.value == 42);
    
    // Teardown parser
    teardown_parser(arena);
    
    printf("PASS: parser_parse_program\n");
}

/**
 * @brief Test parsing an empty program
 */
static void test_parser_empty_program(void) {
    printf("Testing parsing an empty program...\n");
    
    Arena* arena;
    StringTable* strings;
    DiagnosticContext* diag;
    Lexer* lexer;
    Parser* parser;
    
    // Setup parser
    setup_parser(&arena, &strings, &diag, "", &lexer, &parser);
    
    // Parse a program
    AstNode* program = parser_parse_program(parser);
    assert(program != NULL);
    assert(program->type == AST_PROGRAM);
    
    // Verify the program has no expressions
    assert(program->as.program.expr_count == 0);
    
    // Teardown parser
    teardown_parser(arena);
    
    printf("PASS: parser_empty_program\n");
}

/**
 * @brief Test parsing basic literals
 */
static void test_parser_literals(void) {
    printf("Testing parsing literals...\n");
    
    // Test number literal
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, "42", &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        AstNode* number = program->as.program.exprs[0];
        assert(number->type == AST_LITERAL_NUMBER);
        assert(number->as.number.value == 42);
        
        teardown_parser(arena);
    }
    
    // Test boolean literal
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, "#t", &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        AstNode* boolean = program->as.program.exprs[0];
        assert(boolean->type == AST_LITERAL_BOOLEAN);
        assert(boolean->as.boolean.value == true);
        
        teardown_parser(arena);
    }
    
    // Test string literal
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, "\"hello\"", &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        AstNode* string = program->as.program.exprs[0];
        assert(string->type == AST_LITERAL_STRING);
        assert(strcmp(string->as.string.value, "hello") == 0);
        
        teardown_parser(arena);
    }
    
    printf("PASS: parser_literals\n");
}

/**
 * @brief Test parsing if expressions
 */
static void test_parser_if_expression(void) {
    printf("Testing parsing if expressions...\n");
    
    // Test if with both then and else branches
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, "(if #t 1 2)", &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        AstNode* if_expr = program->as.program.exprs[0];
        assert(if_expr->type == AST_IF);
        
        // Verify condition is #t
        AstNode* condition = if_expr->as.if_expr.condition;
        assert(condition->type == AST_LITERAL_BOOLEAN);
        assert(condition->as.boolean.value == true);
        
        // Verify then branch is 1
        AstNode* then_branch = if_expr->as.if_expr.then_branch;
        assert(then_branch->type == AST_LITERAL_NUMBER);
        assert(then_branch->as.number.value == 1);
        
        // Verify else branch is 2
        AstNode* else_branch = if_expr->as.if_expr.else_branch;
        assert(else_branch->type == AST_LITERAL_NUMBER);
        assert(else_branch->as.number.value == 2);
        
        teardown_parser(arena);
    }
    
    // Test if without else branch
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, "(if #t 1)", &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        AstNode* if_expr = program->as.program.exprs[0];
        assert(if_expr->type == AST_IF);
        
        // Verify condition is #t
        AstNode* condition = if_expr->as.if_expr.condition;
        assert(condition->type == AST_LITERAL_BOOLEAN);
        assert(condition->as.boolean.value == true);
        
        // Verify then branch is 1
        AstNode* then_branch = if_expr->as.if_expr.then_branch;
        assert(then_branch->type == AST_LITERAL_NUMBER);
        assert(then_branch->as.number.value == 1);
        
        // Verify else branch is NULL
        AstNode* else_branch = if_expr->as.if_expr.else_branch;
        assert(else_branch == NULL);
        
        teardown_parser(arena);
    }
    
    printf("PASS: parser_if_expression\n");
}

/**
 * @brief Test parsing lambda expressions
 */
static void test_parser_lambda_expression(void) {
    printf("Testing parsing lambda expressions...\n");
    
    // Test lambda with no parameters
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, "(lambda () 42)", &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        AstNode* lambda = program->as.program.exprs[0];
        assert(lambda->type == AST_LAMBDA);
        
        // Verify no parameters
        assert(lambda->as.lambda.param_count == 0);
        
        // Verify body is 42
        AstNode* body = lambda->as.lambda.body;
        assert(body->type == AST_LITERAL_NUMBER);
        assert(body->as.number.value == 42);
        
        teardown_parser(arena);
    }
    
    // Test lambda with parameters
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, "(lambda (x y) (+ x y))", &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        AstNode* lambda = program->as.program.exprs[0];
        assert(lambda->type == AST_LAMBDA);
        
        // Verify parameters
        assert(lambda->as.lambda.param_count == 2);
        assert(strcmp(lambda->as.lambda.params[0]->name, "x") == 0);
        assert(strcmp(lambda->as.lambda.params[1]->name, "y") == 0);
        
        // Verify body is a call to +
        AstNode* body = lambda->as.lambda.body;
        assert(body->type == AST_CALL);
        
        AstNode* callee = body->as.call.callee;
        assert(callee->type == AST_IDENTIFIER);
        assert(strcmp(callee->as.identifier.name, "+") == 0);
        
        // Verify arguments
        assert(body->as.call.arg_count == 2);
        
        AstNode* arg1 = body->as.call.args[0];
        assert(arg1->type == AST_IDENTIFIER);
        assert(strcmp(arg1->as.identifier.name, "x") == 0);
        
        AstNode* arg2 = body->as.call.args[1];
        assert(arg2->type == AST_IDENTIFIER);
        assert(strcmp(arg2->as.identifier.name, "y") == 0);
        
        teardown_parser(arena);
    }
    
    printf("PASS: parser_lambda_expression\n");
}

/**
 * @brief Test parsing begin expressions
 */
static void test_parser_begin_expression(void) {
    printf("Testing parsing begin expressions...\n");
    
    // Test begin with multiple expressions
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, "(begin 1 2 3)", &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        AstNode* begin = program->as.program.exprs[0];
        assert(begin->type == AST_BEGIN);
        
        // Verify expressions
        assert(begin->as.begin.expr_count == 3);
        
        AstNode* expr1 = begin->as.begin.exprs[0];
        assert(expr1->type == AST_LITERAL_NUMBER);
        assert(expr1->as.number.value == 1);
        
        AstNode* expr2 = begin->as.begin.exprs[1];
        assert(expr2->type == AST_LITERAL_NUMBER);
        assert(expr2->as.number.value == 2);
        
        AstNode* expr3 = begin->as.begin.exprs[2];
        assert(expr3->type == AST_LITERAL_NUMBER);
        assert(expr3->as.number.value == 3);
        
        teardown_parser(arena);
    }
    
    // Test begin with no expressions
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, "(begin)", &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        AstNode* begin = program->as.program.exprs[0];
        assert(begin->type == AST_BEGIN);
        
        // Verify no expressions
        assert(begin->as.begin.expr_count == 0);
        
        teardown_parser(arena);
    }
    
    printf("PASS: parser_begin_expression\n");
}

/**
 * @brief Test parsing and/or expressions
 */
static void test_parser_and_or_expressions(void) {
    printf("Testing parsing and/or expressions...\n");
    
    // Test and
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, "(and #t #f #t)", &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        AstNode* and_expr = program->as.program.exprs[0];
        assert(and_expr->type == AST_AND);
        
        // Verify expressions
        assert(and_expr->as.logical.expr_count == 3);
        
        AstNode* expr1 = and_expr->as.logical.exprs[0];
        assert(expr1->type == AST_LITERAL_BOOLEAN);
        assert(expr1->as.boolean.value == true);
        
        AstNode* expr2 = and_expr->as.logical.exprs[1];
        assert(expr2->type == AST_LITERAL_BOOLEAN);
        assert(expr2->as.boolean.value == false);
        
        AstNode* expr3 = and_expr->as.logical.exprs[2];
        assert(expr3->type == AST_LITERAL_BOOLEAN);
        assert(expr3->as.boolean.value == true);
        
        teardown_parser(arena);
    }
    
    // Test or
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, "(or #t #f #t)", &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        AstNode* or_expr = program->as.program.exprs[0];
        assert(or_expr->type == AST_OR);
        
        // Verify expressions
        assert(or_expr->as.logical.expr_count == 3);
        
        AstNode* expr1 = or_expr->as.logical.exprs[0];
        assert(expr1->type == AST_LITERAL_BOOLEAN);
        assert(expr1->as.boolean.value == true);
        
        AstNode* expr2 = or_expr->as.logical.exprs[1];
        assert(expr2->type == AST_LITERAL_BOOLEAN);
        assert(expr2->as.boolean.value == false);
        
        AstNode* expr3 = or_expr->as.logical.exprs[2];
        assert(expr3->type == AST_LITERAL_BOOLEAN);
        assert(expr3->as.boolean.value == true);
        
        teardown_parser(arena);
    }
    
    printf("PASS: parser_and_or_expressions\n");
}

/**
 * @brief Test parsing nested expressions
 */
static void test_parser_nested_expressions(void) {
    printf("Testing parsing nested expressions...\n");
    
    Arena* arena;
    StringTable* strings;
    DiagnosticContext* diag;
    Lexer* lexer;
    Parser* parser;
    
    setup_parser(&arena, &strings, &diag, 
                 "(if (and #t (> x 10)) (begin (set! y 20) (+ y 5)) (lambda () 42))", 
                 &lexer, &parser);
    
    AstNode* program = parser_parse_program(parser);
    assert(program != NULL);
    assert(program->type == AST_PROGRAM);
    assert(program->as.program.expr_count == 1);
    
    AstNode* if_expr = program->as.program.exprs[0];
    assert(if_expr->type == AST_IF);
    
    // Verify condition is (and #t (> x 10))
    AstNode* condition = if_expr->as.if_expr.condition;
    assert(condition->type == AST_AND);
    assert(condition->as.logical.expr_count == 2);
    
    AstNode* and_expr1 = condition->as.logical.exprs[0];
    assert(and_expr1->type == AST_LITERAL_BOOLEAN);
    assert(and_expr1->as.boolean.value == true);
    
    AstNode* and_expr2 = condition->as.logical.exprs[1];
    assert(and_expr2->type == AST_CALL);
    assert(and_expr2->as.call.arg_count == 2);
    
    // Verify then branch is (begin (set! y 20) (+ y 5))
    AstNode* then_branch = if_expr->as.if_expr.then_branch;
    assert(then_branch->type == AST_BEGIN);
    assert(then_branch->as.begin.expr_count == 2);
    
    AstNode* begin_expr1 = then_branch->as.begin.exprs[0];
    assert(begin_expr1->type == AST_SET);
    
    AstNode* begin_expr2 = then_branch->as.begin.exprs[1];
    assert(begin_expr2->type == AST_CALL);
    
    // Verify else branch is (lambda () 42)
    AstNode* else_branch = if_expr->as.if_expr.else_branch;
    assert(else_branch->type == AST_LAMBDA);
    assert(else_branch->as.lambda.param_count == 0);
    
    AstNode* lambda_body = else_branch->as.lambda.body;
    assert(lambda_body->type == AST_LITERAL_NUMBER);
    assert(lambda_body->as.number.value == 42);
    
    teardown_parser(arena);
    
    printf("PASS: parser_nested_expressions\n");
}

/**
 * @brief Test parsing edge cases
 */
static void test_parser_edge_cases(void) {
    printf("Testing parsing edge cases...\n");
    
    // Test empty list
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, "()", &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        AstNode* call = program->as.program.exprs[0];
        assert(call->type == AST_CALL);
        assert(call->as.call.arg_count == 0);
        
        teardown_parser(arena);
    }
    
    // Test deeply nested expressions
    {
        Arena* arena;
        StringTable* strings;
        DiagnosticContext* diag;
        Lexer* lexer;
        Parser* parser;
        
        setup_parser(&arena, &strings, &diag, 
                     "(((((42)))))", 
                     &lexer, &parser);
        
        AstNode* program = parser_parse_program(parser);
        assert(program != NULL);
        assert(program->type == AST_PROGRAM);
        assert(program->as.program.expr_count == 1);
        
        // Navigate through the nested calls
        AstNode* call1 = program->as.program.exprs[0];
        assert(call1->type == AST_CALL);
        assert(call1->as.call.arg_count == 1);
        
        AstNode* call2 = call1->as.call.args[0];
        assert(call2->type == AST_CALL);
        assert(call2->as.call.arg_count == 1);
        
        AstNode* call3 = call2->as.call.args[0];
        assert(call3->type == AST_CALL);
        assert(call3->as.call.arg_count == 1);
        
        AstNode* call4 = call3->as.call.args[0];
        assert(call4->type == AST_CALL);
        assert(call4->as.call.arg_count == 1);
        
        AstNode* number = call4->as.call.args[0];
        assert(number->type == AST_LITERAL_NUMBER);
        assert(number->as.number.value == 42);
        
        teardown_parser(arena);
    }
    
    printf("PASS: parser_edge_cases\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running parser tests...\n");
    
    test_parser_create();
    test_parser_parse_program();
    test_parser_empty_program();
    test_parser_literals();
    test_parser_if_expression();
    test_parser_lambda_expression();
    test_parser_begin_expression();
    test_parser_and_or_expressions();
    test_parser_nested_expressions();
    test_parser_edge_cases();
    
    printf("All parser tests passed!\n");
    return 0;
}
