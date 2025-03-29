/**
 * @file parser_helpers.c
 * @brief Helper functions for the Eshkol parser
 */

#include "frontend/parser/parser_helpers.h"
#include "frontend/parser/parser_error.h"
#include <string.h>
#include <assert.h>

/**
 * @brief Advance to the next token
 * 
 * @param parser The parser
 */
void parser_advance(Parser* parser) {
    parser->previous = parser->current;
    parser->current = lexer_scan_token(parser->lexer);
}

/**
 * @brief Check if the current token is of the specified type
 * 
 * @param parser The parser
 * @param type Token type to check
 * @return true if the current token is of the specified type, false otherwise
 */
bool parser_check(Parser* parser, TokenType type) {
    if (parser_is_at_end(parser)) {
        return false;
    }
    
    return parser->current.type == type;
}

/**
 * @brief Match and consume the current token if it's of the specified type
 * 
 * @param parser The parser
 * @param type Token type to match
 * @return true if the current token was matched and consumed, false otherwise
 */
bool parser_match(Parser* parser, TokenType type) {
    if (!parser_check(parser, type)) {
        return false;
    }
    
    parser_advance(parser);
    return true;
}

/**
 * @brief Consume the current token if it's of the specified type, or report an error
 * 
 * @param parser The parser
 * @param type Token type to consume
 * @param message Error message to report if the token doesn't match
 */
void parser_consume(Parser* parser, TokenType type, const char* message) {
    if (parser_check(parser, type)) {
        parser_advance(parser);
        return;
    }
    
    parser_error(parser, message);
}

/**
 * @brief Check if we've reached the end of the file
 * 
 * @param parser The parser
 * @return true if we've reached the end of the file, false otherwise
 */
bool parser_is_at_end(Parser* parser) {
    return parser->current.type == TOKEN_EOF;
}

/**
 * @brief Check if a name is a special form
 * 
 * @param name Name to check
 * @return true if the name is a special form, false otherwise
 */
bool parser_is_special_form(StringId name) {
    return strcmp(name, "define") == 0 ||
           strcmp(name, "if") == 0 ||
           strcmp(name, "lambda") == 0 ||
           strcmp(name, "begin") == 0 ||
           strcmp(name, "set!") == 0 ||
           strcmp(name, "let") == 0 ||
           strcmp(name, "do") == 0 ||
           strcmp(name, "and") == 0 ||
           strcmp(name, "or") == 0 ||
           strcmp(name, "vector") == 0 ||
           strcmp(name, "v+") == 0 ||
           strcmp(name, "v-") == 0 ||
           strcmp(name, "v*") == 0 ||
           strcmp(name, "dot") == 0 ||
           strcmp(name, "cross") == 0 ||
           strcmp(name, "norm") == 0 ||
           strcmp(name, "gradient") == 0 ||
           strcmp(name, "divergence") == 0 ||
           strcmp(name, "curl") == 0 ||
           strcmp(name, "laplacian") == 0;
}
