/**
 * @file test_ast.c
 * @brief Unit tests for the AST
 */

#include "frontend/ast/ast.h"
#include "core/memory.h"
#include "core/string_table.h"
#include "core/diagnostics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Test AST node creation
 */
static void test_ast_create_node(void) {
    printf("Testing AST node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a node
    AstNode* node = ast_create_node(arena, AST_LITERAL_NUMBER, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_LITERAL_NUMBER);
    assert(node->line == 1);
    assert(node->column == 1);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_node\n");
}

/**
 * @brief Test number literal node creation
 */
static void test_ast_create_number(void) {
    printf("Testing number literal node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a number node
    AstNode* node = ast_create_number(arena, 42.0, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_LITERAL_NUMBER);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.number.value == 42.0);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_number\n");
}

/**
 * @brief Test boolean literal node creation
 */
static void test_ast_create_boolean(void) {
    printf("Testing boolean literal node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a boolean node (true)
    AstNode* node1 = ast_create_boolean(arena, true, 1, 1);
    assert(node1 != NULL);
    assert(node1->type == AST_LITERAL_BOOLEAN);
    assert(node1->line == 1);
    assert(node1->column == 1);
    assert(node1->as.boolean.value == true);
    
    // Create a boolean node (false)
    AstNode* node2 = ast_create_boolean(arena, false, 2, 2);
    assert(node2 != NULL);
    assert(node2->type == AST_LITERAL_BOOLEAN);
    assert(node2->line == 2);
    assert(node2->column == 2);
    assert(node2->as.boolean.value == false);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_boolean\n");
}

/**
 * @brief Test character literal node creation
 */
static void test_ast_create_character(void) {
    printf("Testing character literal node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a character node
    AstNode* node = ast_create_character(arena, 'a', 1, 1);
    assert(node != NULL);
    assert(node->type == AST_LITERAL_CHARACTER);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.character.value == 'a');
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_character\n");
}

/**
 * @brief Test string literal node creation
 */
static void test_ast_create_string(void) {
    printf("Testing string literal node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a string table
    StringTable* strings = string_table_create(arena, 16);
    assert(strings != NULL);
    
    // Intern a string
    const char* str = "Hello, world!";
    StringId id = string_table_intern(strings, str);
    assert(id != NULL);
    
    // Create a string node
    AstNode* node = ast_create_string(arena, id, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_LITERAL_STRING);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.string.value == id);
    assert(strcmp(node->as.string.value, str) == 0);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_string\n");
}

/**
 * @brief Test vector literal node creation
 */
static void test_ast_create_vector(void) {
    printf("Testing vector literal node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create some elements
    AstNode* elements[3];
    elements[0] = ast_create_number(arena, 1.0, 1, 1);
    elements[1] = ast_create_number(arena, 2.0, 1, 3);
    elements[2] = ast_create_number(arena, 3.0, 1, 5);
    
    // Create a vector node
    AstNode* node = ast_create_vector(arena, elements, 3, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_LITERAL_VECTOR);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.vector.count == 3);
    assert(node->as.vector.elements[0] == elements[0]);
    assert(node->as.vector.elements[1] == elements[1]);
    assert(node->as.vector.elements[2] == elements[2]);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_vector\n");
}

/**
 * @brief Test nil literal node creation
 */
static void test_ast_create_nil(void) {
    printf("Testing nil literal node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a nil node
    AstNode* node = ast_create_nil(arena, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_LITERAL_NIL);
    assert(node->line == 1);
    assert(node->column == 1);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_nil\n");
}

/**
 * @brief Test identifier node creation
 */
static void test_ast_create_identifier(void) {
    printf("Testing identifier node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a string table
    StringTable* strings = string_table_create(arena, 16);
    assert(strings != NULL);
    
    // Intern a string
    const char* str = "x";
    StringId id = string_table_intern(strings, str);
    assert(id != NULL);
    
    // Create an identifier node
    AstNode* node = ast_create_identifier(arena, id, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_IDENTIFIER);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.identifier.name == id);
    assert(strcmp(node->as.identifier.name, str) == 0);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_identifier\n");
}

/**
 * @brief Test define node creation
 */
static void test_ast_create_define(void) {
    printf("Testing define node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a string table
    StringTable* strings = string_table_create(arena, 16);
    assert(strings != NULL);
    
    // Intern a string
    const char* str = "x";
    StringId id = string_table_intern(strings, str);
    assert(id != NULL);
    
    // Create an identifier node
    AstNode* name = ast_create_identifier(arena, id, 1, 8);
    assert(name != NULL);
    
    // Create a number node
    AstNode* value = ast_create_number(arena, 42.0, 1, 10);
    assert(value != NULL);
    
    // Create a define node
    AstNode* node = ast_create_define(arena, name, value, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_DEFINE);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.define.name == name);
    assert(node->as.define.value == value);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_define\n");
}

/**
 * @brief Test lambda node creation
 */
static void test_ast_create_lambda(void) {
    printf("Testing lambda node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a string table
    StringTable* strings = string_table_create(arena, 16);
    assert(strings != NULL);
    
    // Intern a string
    const char* str = "x";
    StringId id = string_table_intern(strings, str);
    assert(id != NULL);
    
    // Create a parameter node
    AstNode* param = ast_create_identifier(arena, id, 1, 10);
    assert(param != NULL);
    
    // Create a parameter list
    AstNode* params[1] = { param };
    
    // Create a body node
    AstNode* body = ast_create_identifier(arena, id, 1, 14);
    assert(body != NULL);
    
    // Create a lambda node
    AstNode* node = ast_create_lambda(arena, params, 1, body, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_LAMBDA);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.lambda.param_count == 1);
    assert(node->as.lambda.params[0] == param);
    assert(node->as.lambda.body == body);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_lambda\n");
}

/**
 * @brief Test if node creation
 */
static void test_ast_create_if(void) {
    printf("Testing if node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a condition node
    AstNode* condition = ast_create_boolean(arena, true, 1, 4);
    assert(condition != NULL);
    
    // Create a then node
    AstNode* then_branch = ast_create_number(arena, 1.0, 1, 9);
    assert(then_branch != NULL);
    
    // Create an else node
    AstNode* else_branch = ast_create_number(arena, 2.0, 1, 11);
    assert(else_branch != NULL);
    
    // Create an if node
    AstNode* node = ast_create_if(arena, condition, then_branch, else_branch, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_IF);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.if_expr.condition == condition);
    assert(node->as.if_expr.then_branch == then_branch);
    assert(node->as.if_expr.else_branch == else_branch);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_if\n");
}

/**
 * @brief Test begin node creation
 */
static void test_ast_create_begin(void) {
    printf("Testing begin node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create some expressions
    AstNode* exprs[3];
    exprs[0] = ast_create_number(arena, 1.0, 1, 8);
    exprs[1] = ast_create_number(arena, 2.0, 1, 10);
    exprs[2] = ast_create_number(arena, 3.0, 1, 12);
    
    // Create a begin node
    AstNode* node = ast_create_begin(arena, exprs, 3, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_BEGIN);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.begin.expr_count == 3);
    assert(node->as.begin.exprs[0] == exprs[0]);
    assert(node->as.begin.exprs[1] == exprs[1]);
    assert(node->as.begin.exprs[2] == exprs[2]);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_begin\n");
}

/**
 * @brief Test quote node creation
 */
static void test_ast_create_quote(void) {
    printf("Testing quote node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a string table
    StringTable* strings = string_table_create(arena, 16);
    assert(strings != NULL);
    
    // Intern a string
    const char* str = "x";
    StringId id = string_table_intern(strings, str);
    assert(id != NULL);
    
    // Create an expression node
    AstNode* expr = ast_create_identifier(arena, id, 1, 2);
    assert(expr != NULL);
    
    // Create a quote node
    AstNode* node = ast_create_quote(arena, expr, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_QUOTE);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.quote.expr == expr);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_quote\n");
}

/**
 * @brief Test set! node creation
 */
static void test_ast_create_set(void) {
    printf("Testing set! node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a string table
    StringTable* strings = string_table_create(arena, 16);
    assert(strings != NULL);
    
    // Intern a string
    const char* str = "x";
    StringId id = string_table_intern(strings, str);
    assert(id != NULL);
    
    // Create a name node
    AstNode* name = ast_create_identifier(arena, id, 1, 6);
    assert(name != NULL);
    
    // Create a value node
    AstNode* value = ast_create_number(arena, 42.0, 1, 8);
    assert(value != NULL);
    
    // Create a set! node
    AstNode* node = ast_create_set(arena, name, value, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_SET);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.set.name == name);
    assert(node->as.set.value == value);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_set\n");
}

/**
 * @brief Test call node creation
 */
static void test_ast_create_call(void) {
    printf("Testing call node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a string table
    StringTable* strings = string_table_create(arena, 16);
    assert(strings != NULL);
    
    // Intern a string
    const char* str = "f";
    StringId id = string_table_intern(strings, str);
    assert(id != NULL);
    
    // Create a callee node
    AstNode* callee = ast_create_identifier(arena, id, 1, 1);
    assert(callee != NULL);
    
    // Create some argument nodes
    AstNode* args[2];
    args[0] = ast_create_number(arena, 1.0, 1, 3);
    args[1] = ast_create_number(arena, 2.0, 1, 5);
    
    // Create a call node
    AstNode* node = ast_create_call(arena, callee, args, 2, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_CALL);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.call.callee == callee);
    assert(node->as.call.arg_count == 2);
    assert(node->as.call.args[0] == args[0]);
    assert(node->as.call.args[1] == args[1]);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_call\n");
}

/**
 * @brief Test program node creation
 */
static void test_ast_create_program(void) {
    printf("Testing program node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a string table
    StringTable* strings = string_table_create(arena, 16);
    assert(strings != NULL);
    
    // Intern a string
    const char* str = "x";
    StringId id = string_table_intern(strings, str);
    assert(id != NULL);
    
    // Create some expressions
    AstNode* exprs[2];
    
    // (define x 42)
    AstNode* name = ast_create_identifier(arena, id, 1, 8);
    AstNode* value = ast_create_number(arena, 42.0, 1, 10);
    exprs[0] = ast_create_define(arena, name, value, 1, 1);
    
    // x
    exprs[1] = ast_create_identifier(arena, id, 2, 1);
    
    // Create a program node
    AstNode* node = ast_create_program(arena, exprs, 2, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_PROGRAM);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.program.expr_count == 2);
    assert(node->as.program.exprs[0] == exprs[0]);
    assert(node->as.program.exprs[1] == exprs[1]);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_program\n");
}

/**
 * @brief Test error node creation
 */
static void test_ast_create_error(void) {
    printf("Testing error node creation...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a string table
    StringTable* strings = string_table_create(arena, 16);
    assert(strings != NULL);
    
    // Intern a string
    const char* str = "Syntax error";
    StringId id = string_table_intern(strings, str);
    assert(id != NULL);
    
    // Create an error node
    AstNode* node = ast_create_error(arena, id, 1, 1);
    assert(node != NULL);
    assert(node->type == AST_ERROR);
    assert(node->line == 1);
    assert(node->column == 1);
    assert(node->as.error.message == id);
    assert(strcmp(node->as.error.message, str) == 0);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_create_error\n");
}

/**
 * @brief Test ast_node_type_to_string
 */
static void test_ast_node_type_to_string(void) {
    printf("Testing ast_node_type_to_string...\n");
    
    assert(strcmp(ast_node_type_to_string(AST_LITERAL_NUMBER), "NUMBER") == 0);
    assert(strcmp(ast_node_type_to_string(AST_LITERAL_BOOLEAN), "BOOLEAN") == 0);
    assert(strcmp(ast_node_type_to_string(AST_LITERAL_CHARACTER), "CHARACTER") == 0);
    assert(strcmp(ast_node_type_to_string(AST_LITERAL_STRING), "STRING") == 0);
    assert(strcmp(ast_node_type_to_string(AST_LITERAL_VECTOR), "VECTOR") == 0);
    assert(strcmp(ast_node_type_to_string(AST_LITERAL_NIL), "NIL") == 0);
    assert(strcmp(ast_node_type_to_string(AST_IDENTIFIER), "IDENTIFIER") == 0);
    assert(strcmp(ast_node_type_to_string(AST_DEFINE), "DEFINE") == 0);
    assert(strcmp(ast_node_type_to_string(AST_LAMBDA), "LAMBDA") == 0);
    assert(strcmp(ast_node_type_to_string(AST_IF), "IF") == 0);
    assert(strcmp(ast_node_type_to_string(AST_BEGIN), "BEGIN") == 0);
    assert(strcmp(ast_node_type_to_string(AST_QUOTE), "QUOTE") == 0);
    assert(strcmp(ast_node_type_to_string(AST_SET), "SET!") == 0);
    assert(strcmp(ast_node_type_to_string(AST_LET), "LET") == 0);
    assert(strcmp(ast_node_type_to_string(AST_LETREC), "LETREC") == 0);
    assert(strcmp(ast_node_type_to_string(AST_LETSTAR), "LET*") == 0);
    assert(strcmp(ast_node_type_to_string(AST_AND), "AND") == 0);
    assert(strcmp(ast_node_type_to_string(AST_OR), "OR") == 0);
    assert(strcmp(ast_node_type_to_string(AST_COND), "COND") == 0);
    assert(strcmp(ast_node_type_to_string(AST_CASE), "CASE") == 0);
    assert(strcmp(ast_node_type_to_string(AST_DO), "DO") == 0);
    assert(strcmp(ast_node_type_to_string(AST_DELAY), "DELAY") == 0);
    assert(strcmp(ast_node_type_to_string(AST_QUASIQUOTE), "QUASIQUOTE") == 0);
    assert(strcmp(ast_node_type_to_string(AST_UNQUOTE), "UNQUOTE") == 0);
    assert(strcmp(ast_node_type_to_string(AST_UNQUOTE_SPLICING), "UNQUOTE-SPLICING") == 0);
    assert(strcmp(ast_node_type_to_string(AST_CALL), "CALL") == 0);
    assert(strcmp(ast_node_type_to_string(AST_SEQUENCE), "SEQUENCE") == 0);
    assert(strcmp(ast_node_type_to_string(AST_FUNCTION_DEF), "FUNCTION-DEF") == 0);
    assert(strcmp(ast_node_type_to_string(AST_VARIABLE_DEF), "VARIABLE-DEF") == 0);
    assert(strcmp(ast_node_type_to_string(AST_PROGRAM), "PROGRAM") == 0);
    assert(strcmp(ast_node_type_to_string(AST_ERROR), "ERROR") == 0);
    
    printf("PASS: ast_node_type_to_string\n");
}

/**
 * @brief Test ast_print
 */
static void test_ast_print(void) {
    printf("Testing ast_print...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a string table
    StringTable* strings = string_table_create(arena, 16);
    assert(strings != NULL);
    
    // Create a factorial program
    // (define (factorial n)
    //   (if (< n 2)
    //       1
    //       (* n (factorial (- n 1)))))
    
    // Intern strings
    StringId factorial_id = string_table_intern(strings, "factorial");
    StringId n_id = string_table_intern(strings, "n");
    StringId lt_id = string_table_intern(strings, "<");
    StringId mul_id = string_table_intern(strings, "*");
    StringId sub_id = string_table_intern(strings, "-");
    
    // Create parameter list
    AstNode* param = ast_create_identifier(arena, n_id, 1, 20);
    AstNode* params[1] = { param };
    
    // Create function name
    AstNode* name = ast_create_identifier(arena, factorial_id, 1, 9);
    
    // Create condition: (< n 2)
    AstNode* lt = ast_create_identifier(arena, lt_id, 2, 7);
    AstNode* n1 = ast_create_identifier(arena, n_id, 2, 9);
    AstNode* two = ast_create_number(arena, 2.0, 2, 11);
    AstNode* lt_args[2] = { n1, two };
    AstNode* condition = ast_create_call(arena, lt, lt_args, 2, 2, 6);
    
    // Create then branch: 1
    AstNode* one = ast_create_number(arena, 1.0, 3, 7);
    
    // Create else branch: (* n (factorial (- n 1)))
    AstNode* mul = ast_create_identifier(arena, mul_id, 4, 7);
    AstNode* n2 = ast_create_identifier(arena, n_id, 4, 9);
    
    // Create (- n 1)
    AstNode* sub = ast_create_identifier(arena, sub_id, 4, 22);
    AstNode* n3 = ast_create_identifier(arena, n_id, 4, 24);
    AstNode* one2 = ast_create_number(arena, 1.0, 4, 26);
    AstNode* sub_args[2] = { n3, one2 };
    AstNode* sub_call = ast_create_call(arena, sub, sub_args, 2, 4, 21);
    
    // Create (factorial (- n 1))
    AstNode* factorial = ast_create_identifier(arena, factorial_id, 4, 11);
    AstNode* factorial_args[1] = { sub_call };
    AstNode* factorial_call = ast_create_call(arena, factorial, factorial_args, 1, 4, 10);
    
    // Create (* n (factorial (- n 1)))
    AstNode* mul_args[2] = { n2, factorial_call };
    AstNode* mul_call = ast_create_call(arena, mul, mul_args, 2, 4, 6);
    
    // Create if expression
    AstNode* if_expr = ast_create_if(arena, condition, one, mul_call, 2, 3);
    
    // Create function definition
    AstNode* func_def = ast_create_function_def(arena, name, params, 1, if_expr, 1, 1);
    
    // Print the AST
    printf("\nPrinting factorial function AST:\n");
    ast_print(func_def, 0);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: ast_print\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running AST tests...\n");
    
    test_ast_create_node();
    test_ast_create_number();
    test_ast_create_boolean();
    test_ast_create_character();
    test_ast_create_string();
    test_ast_create_vector();
    test_ast_create_nil();
    test_ast_create_identifier();
    test_ast_create_define();
    test_ast_create_lambda();
    test_ast_create_if();
    test_ast_create_begin();
    test_ast_create_quote();
    test_ast_create_set();
    test_ast_create_call();
    test_ast_create_program();
    test_ast_create_error();
    test_ast_node_type_to_string();
    test_ast_print();
    
    printf("All AST tests passed!\n");
    return 0;
}
