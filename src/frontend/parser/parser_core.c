/**
 * @file parser_core.c
 * @brief Core functionality for the Eshkol parser
 */

#include "frontend/parser/parser_core.h"
#include "frontend/parser/parser_helpers.h"
#include "frontend/parser/parser_error.h"
#include "frontend/parser/parser_expressions.h"
#include "frontend/ast/analysis/ast_parent.h"
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
    
    // Initialize binding system
    parser->bindings = binding_system_create(arena, diag);
    if (!parser->bindings) {
        return NULL;
    }
    
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
    size_t expr_count = 0;
    AstNode** exprs = NULL;
    
    // Parse top-level expressions
    exprs = parser_parse_expressions(parser, &expr_count);
    if (!exprs && expr_count > 0) {
        return NULL;
    }
    
    // Create a program node
    AstNode* program = ast_create_program(
        parser->arena, 
        exprs, 
        expr_count, 
        parser->previous.line, 
        parser->previous.column
    );
    
    if (!program) {
        return NULL;
    }
    
    // Set parent pointers in the AST
    ast_set_parent_pointers(program);
    
    return program;
}

/**
 * @brief Parse a list of expressions
 * 
 * @param parser The parser
 * @param count Pointer to store the number of expressions
 * @return Array of expression nodes, or NULL on failure
 */
AstNode** parser_parse_expressions(Parser* parser, size_t* count) {
    // Allocate initial array
    size_t capacity = 8;
    AstNode** exprs = arena_alloc(parser->arena, sizeof(AstNode*) * capacity);
    if (!exprs) {
        *count = 0;
        return NULL;
    }
    
    size_t expr_count = 0;
    
    // Parse expressions until end of file
    while (!parser_is_at_end(parser)) {
        // Skip comments
        while (parser_match(parser, TOKEN_COMMENT)) {
            // Do nothing, just advance
        }
        
        // Check if we're at the end
        if (parser_is_at_end(parser)) {
            break;
        }
        
        // Parse an expression
        AstNode* expr = parser_parse_expression(parser);
        if (!expr) {
            // Error occurred, but we'll try to continue
            parser_synchronize(parser);
            continue;
        }
        
        // Add the expression to the array
        if (expr_count >= capacity) {
            // Resize the array
            size_t new_capacity = capacity * 2;
            AstNode** new_exprs = arena_alloc(parser->arena, sizeof(AstNode*) * new_capacity);
            if (!new_exprs) {
                *count = expr_count;
                return exprs;
            }
            
            // Copy the old array
            for (size_t i = 0; i < expr_count; i++) {
                new_exprs[i] = exprs[i];
            }
            
            exprs = new_exprs;
            capacity = new_capacity;
        }
        
        exprs[expr_count++] = expr;
    }
    
    *count = expr_count;
    return exprs;
}
