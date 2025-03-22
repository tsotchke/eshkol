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
 * @brief Test parsing a program
 */
static void test_parser_parse_program(void) {
    printf("Testing parsing a program...\n");
    
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
    
    // Parse a program
    AstNode* program = parser_parse_program(parser);
    assert(program != NULL);
    assert(program->type == AST_PROGRAM);
    
    // Print the AST
    printf("\nPrinting program AST:\n");
    ast_print(program, 0);
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: parser_parse_program\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running parser tests...\n");
    
    test_parser_create();
    test_parser_parse_program();
    
    printf("All parser tests passed!\n");
    return 0;
}
