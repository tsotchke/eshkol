/**
 * @file parser.c
 * @brief Implementation of the parser for Eshkol
 */

#include "frontend/parser/parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Create a new parser
 * 
 * @param arena Arena allocator
 * @param strings String table
 * @param diag Diagnostic context
 * @param lexer Lexer
 * @return A new parser, or NULL on failure
 */
Parser* parser_create(Arena* arena, StringTable* strings, DiagnosticContext* diag, Lexer* lexer) {
    assert(arena != NULL);
    assert(strings != NULL);
    assert(diag != NULL);
    assert(lexer != NULL);
    
    Parser* parser = arena_alloc(arena, sizeof(Parser));
    if (!parser) {
        return NULL;
    }
    
    parser->arena = arena;
    parser->strings = strings;
    parser->diag = diag;
    parser->lexer = lexer;
    parser->had_error = false;
    parser->panic_mode = false;
    
    // Initialize current and previous tokens
    parser->current = lexer_scan_token(lexer);
    parser->previous = parser->current;
    
    return parser;
}

/**
 * @brief Parse a program
 * 
 * @param parser The parser
 * @return The program AST node, or NULL on failure
 */
AstNode* parser_parse_program(Parser* parser) {
    // TODO: Implement the parser
    
    // For now, just return a placeholder program node
    AstNode** exprs = arena_alloc(parser->arena, sizeof(AstNode*));
    if (!exprs) {
        return NULL;
    }
    
    // Create a placeholder expression (a number literal)
    exprs[0] = ast_create_number(parser->arena, 42.0, 1, 1);
    if (!exprs[0]) {
        return NULL;
    }
    
    // Create a program node
    AstNode* program = ast_create_program(parser->arena, exprs, 1, 1, 1);
    if (!program) {
        return NULL;
    }
    
    return program;
}
