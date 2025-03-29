/**
 * @file parser_conditionals.c
 * @brief Conditional special form parsing for the Eshkol parser
 */

#include "frontend/parser/parser_conditionals.h"
#include "frontend/parser/parser_helpers.h"
#include "frontend/parser/parser_error.h"
#include "frontend/parser/parser_expressions.h"
#include <assert.h>
#include <string.h>

/**
 * @brief Parse an if special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return If node, or NULL on failure
 */
AstNode* parser_parse_if(Parser* parser, size_t line, size_t column) {
    // Parse the condition
    AstNode* condition = parser_parse_expression(parser);
    if (!condition) {
        parser_error(parser, "Expected condition expression");
        return NULL;
    }
    
    // Parse the then branch
    AstNode* then_branch = parser_parse_expression(parser);
    if (!then_branch) {
        parser_error(parser, "Expected then expression");
        return NULL;
    }
    
    // Parse the else branch (optional)
    AstNode* else_branch = NULL;
    if (!parser_check(parser, TOKEN_RPAREN)) {
        else_branch = parser_parse_expression(parser);
        if (!else_branch) {
            parser_error(parser, "Expected else expression");
            return NULL;
        }
    }
    
    // Consume the closing parenthesis
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after if");
    
    // Create an if node
    return ast_create_if(parser->arena, condition, then_branch, else_branch, line, column);
}

/**
 * @brief Parse an and or or special form
 * 
 * @param parser The parser
 * @param type AST_AND or AST_OR
 * @param line Line number
 * @param column Column number
 * @return And or or node, or NULL on failure
 */
AstNode* parser_parse_and_or(Parser* parser, AstNodeType type, size_t line, size_t column) {
    // Parse the expressions
    size_t expr_count = 0;
    AstNode** exprs = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
    if (!exprs) {
        parser_error(parser, "Failed to allocate memory for expressions");
        return NULL;
    }
    
    while (!parser_check(parser, TOKEN_RPAREN) && !parser_is_at_end(parser)) {
        if (expr_count >= 16) {
            parser_error(parser, "Too many expressions in and/or");
            return NULL;
        }
        
        AstNode* expr = parser_parse_expression(parser);
        if (!expr) {
            parser_error(parser, "Expected expression");
            return NULL;
        }
        
        exprs[expr_count++] = expr;
    }
    
    // Consume the closing parenthesis
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after and/or");
    
    // Create an and or or node
    if (type == AST_AND) {
        return ast_create_and(parser->arena, exprs, expr_count, line, column);
    } else {
        return ast_create_or(parser->arena, exprs, expr_count, line, column);
    }
}
