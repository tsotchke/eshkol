/**
 * @file parser_helpers.h
 * @brief Helper functions for the Eshkol parser
 * 
 * This file defines helper functions for token handling in the Eshkol parser.
 */

#ifndef ESHKOL_PARSER_HELPERS_H
#define ESHKOL_PARSER_HELPERS_H

#include "frontend/parser/parser_core.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Advance to the next token
 * 
 * @param parser The parser
 */
void parser_advance(Parser* parser);

/**
 * @brief Check if the current token is of the specified type
 * 
 * @param parser The parser
 * @param type Token type to check
 * @return true if the current token is of the specified type, false otherwise
 */
bool parser_check(Parser* parser, TokenType type);

/**
 * @brief Match and consume the current token if it's of the specified type
 * 
 * @param parser The parser
 * @param type Token type to match
 * @return true if the current token was matched and consumed, false otherwise
 */
bool parser_match(Parser* parser, TokenType type);

/**
 * @brief Consume the current token if it's of the specified type, or report an error
 * 
 * @param parser The parser
 * @param type Token type to consume
 * @param message Error message to report if the token doesn't match
 */
void parser_consume(Parser* parser, TokenType type, const char* message);

/**
 * @brief Check if we've reached the end of the file
 * 
 * @param parser The parser
 * @return true if we've reached the end of the file, false otherwise
 */
bool parser_is_at_end(Parser* parser);

/**
 * @brief Check if a name is a special form
 * 
 * @param name Name to check
 * @return true if the name is a special form, false otherwise
 */
bool parser_is_special_form(StringId name);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PARSER_HELPERS_H */
