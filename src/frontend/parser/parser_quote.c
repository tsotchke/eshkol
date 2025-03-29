/**
 * @file parser_quote.c
 * @brief Quote special form parsing for the Eshkol parser
 */

#include "frontend/parser/parser_quote.h"
#include "frontend/parser/parser_helpers.h"
#include "frontend/parser/parser_error.h"
#include "frontend/parser/parser_expressions.h"
#include <assert.h>
#include <string.h>

/**
 * @brief Parse a quote special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Quote node, or NULL on failure
 */
AstNode* parser_parse_quote(Parser* parser, size_t line, size_t column) {
    // Parse the quoted expression
    AstNode* expr = parser_parse_expression(parser);
    if (!expr) {
        parser_error(parser, "Expected expression after quote");
        return NULL;
    }
    
    // Consume the closing parenthesis
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after quote");
    
    // Create a quote node
    return ast_create_quote(parser->arena, expr, line, column);
}
