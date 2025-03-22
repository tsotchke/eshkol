/**
 * @file test_lexer.c
 * @brief Unit tests for the lexer
 */

#include "frontend/lexer/lexer.h"
#include "core/memory.h"
#include "core/string_table.h"
#include "core/diagnostics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Test lexer creation
 */
static void test_lexer_create(void) {
    printf("Testing lexer creation...\n");
    
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
    const char* source = "(define (factorial n) (if (< n 2) 1 (* n (factorial (- n 1)))))";
    Lexer* lexer = lexer_create(arena, strings, diag, source);
    assert(lexer != NULL);
    assert(lexer->arena == arena);
    assert(lexer->strings == strings);
    assert(lexer->diag == diag);
    assert(lexer->source == source);
    assert(lexer->start == source);
    assert(lexer->current == source);
    assert(lexer->line == 1);
    assert(lexer->column == 1);
    assert(lexer->had_error == false);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: lexer_create\n");
}

/**
 * @brief Test token scanning
 */
static void test_lexer_scan_token(void) {
    printf("Testing token scanning...\n");
    
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
    const char* source = "(define (factorial n) (if (< n 2) 1 (* n (factorial (- n 1)))))";
    Lexer* lexer = lexer_create(arena, strings, diag, source);
    assert(lexer != NULL);
    
    // Scan tokens
    Token token;
    
    // (
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_LPAREN));
    
    // define
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strcmp(token.value.string_id, "define") == 0);
    
    // (
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_LPAREN));
    
    // factorial
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strcmp(token.value.string_id, "factorial") == 0);
    
    // n
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strcmp(token.value.string_id, "n") == 0);
    
    // )
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_RPAREN));
    
    // (
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_LPAREN));
    
    // if
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strcmp(token.value.string_id, "if") == 0);
    
    // (
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_LPAREN));
    
    // <
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strcmp(token.value.string_id, "<") == 0);
    
    // n
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strcmp(token.value.string_id, "n") == 0);
    
    // 2
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_NUMBER));
    assert(token.value.number == 2.0);
    
    // )
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_RPAREN));
    
    // 1
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_NUMBER));
    assert(token.value.number == 1.0);
    
    // (
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_LPAREN));
    
    // *
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strcmp(token.value.string_id, "*") == 0);
    
    // n
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strcmp(token.value.string_id, "n") == 0);
    
    // (
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_LPAREN));
    
    // factorial
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strcmp(token.value.string_id, "factorial") == 0);
    
    // (
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_LPAREN));
    
    // -
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strcmp(token.value.string_id, "-") == 0);
    
    // n
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strcmp(token.value.string_id, "n") == 0);
    
    // 1
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_NUMBER));
    assert(token.value.number == 1.0);
    
    // )
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_RPAREN));
    
    // )
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_RPAREN));
    
    // )
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_RPAREN));
    
    // )
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_RPAREN));
    
    // )
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_RPAREN));
    
    // EOF
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_EOF));
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: lexer_scan_token\n");
}

/**
 * @brief Test string literals
 */
static void test_string_literals(void) {
    printf("Testing string literals...\n");
    
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
    const char* source = "\"Hello, world!\" \"Line 1\\nLine 2\"";
    Lexer* lexer = lexer_create(arena, strings, diag, source);
    assert(lexer != NULL);
    
    // Scan tokens
    Token token;
    
    // "Hello, world!"
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_STRING));
    assert(strcmp(token.value.string_id, "Hello, world!") == 0);
    
    // "Line 1\nLine 2"
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_STRING));
    assert(strcmp(token.value.string_id, "Line 1\\nLine 2") == 0);
    
    // EOF
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_EOF));
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: string_literals\n");
}

/**
 * @brief Test boolean literals
 */
static void test_boolean_literals(void) {
    printf("Testing boolean literals...\n");
    
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
    const char* source = "#t #f";
    Lexer* lexer = lexer_create(arena, strings, diag, source);
    assert(lexer != NULL);
    
    // Scan tokens
    Token token;
    
    // #t
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_BOOLEAN));
    assert(token.value.boolean == true);
    
    // #f
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_BOOLEAN));
    assert(token.value.boolean == false);
    
    // EOF
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_EOF));
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: boolean_literals\n");
}

/**
 * @brief Test character literals
 */
static void test_character_literals(void) {
    printf("Testing character literals...\n");
    
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
    const char* source = "#\\a #\\b #\\c";
    Lexer* lexer = lexer_create(arena, strings, diag, source);
    assert(lexer != NULL);
    
    // Scan tokens
    Token token;
    
    // #\a
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_CHARACTER));
    assert(token.value.character == 'a');
    
    // #\b
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_CHARACTER));
    assert(token.value.character == 'b');
    
    // #\c
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_CHARACTER));
    assert(token.value.character == 'c');
    
    // EOF
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_EOF));
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: character_literals\n");
}

/**
 * @brief Test number literals
 */
static void test_number_literals(void) {
    printf("Testing number literals...\n");
    
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
    const char* source = "123 456.789";
    Lexer* lexer = lexer_create(arena, strings, diag, source);
    assert(lexer != NULL);
    
    // Scan tokens
    Token token;
    
    // 123
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_NUMBER));
    assert(token.value.number == 123.0);
    
    // 456.789
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_NUMBER));
    assert(token.value.number == 456.789);
    
    // EOF
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_EOF));
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: number_literals\n");
}

/**
 * @brief Test vector literals
 */
static void test_vector_literals(void) {
    printf("Testing vector literals...\n");
    
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
    const char* source = "#(1 2 3)";
    Lexer* lexer = lexer_create(arena, strings, diag, source);
    assert(lexer != NULL);
    
    // Scan tokens
    Token token;
    
    // #(
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_VECTOR_START));
    
    // 1
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_NUMBER));
    assert(token.value.number == 1.0);
    
    // 2
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_NUMBER));
    assert(token.value.number == 2.0);
    
    // 3
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_NUMBER));
    assert(token.value.number == 3.0);
    
    // )
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_RPAREN));
    
    // EOF
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_EOF));
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: vector_literals\n");
}

/**
 * @brief Test comments
 */
static void test_comments(void) {
    printf("Testing comments...\n");
    
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
    const char* source = "; This is a comment\n123";
    Lexer* lexer = lexer_create(arena, strings, diag, source);
    assert(lexer != NULL);
    
    // Scan tokens
    Token token;
    
    // 123
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_NUMBER));
    assert(token.value.number == 123.0);
    
    // EOF
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_EOF));
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: comments\n");
}

/**
 * @brief Test token_to_string
 */
static void test_token_to_string(void) {
    printf("Testing token_to_string...\n");
    
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
    const char* source = "(define x 123)";
    Lexer* lexer = lexer_create(arena, strings, diag, source);
    assert(lexer != NULL);
    
    // Scan tokens
    Token token;
    
    // (
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_LPAREN));
    assert(strcmp(token_to_string(&token), "(") == 0);
    
    // define
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strstr(token_to_string(&token), "IDENTIFIER") != NULL);
    assert(strstr(token_to_string(&token), "define") != NULL);
    
    // x
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_IDENTIFIER));
    assert(strstr(token_to_string(&token), "IDENTIFIER") != NULL);
    assert(strstr(token_to_string(&token), "x") != NULL);
    
    // 123
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_NUMBER));
    assert(strstr(token_to_string(&token), "NUMBER") != NULL);
    assert(strstr(token_to_string(&token), "123") != NULL);
    
    // )
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_RPAREN));
    assert(strcmp(token_to_string(&token), ")") == 0);
    
    // EOF
    token = lexer_scan_token(lexer);
    assert(token_is(&token, TOKEN_EOF));
    assert(strcmp(token_to_string(&token), "EOF") == 0);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: token_to_string\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running lexer tests...\n");
    
    test_lexer_create();
    test_lexer_scan_token();
    test_string_literals();
    test_boolean_literals();
    test_character_literals();
    test_number_literals();
    test_vector_literals();
    test_comments();
    test_token_to_string();
    
    printf("All lexer tests passed!\n");
    return 0;
}
