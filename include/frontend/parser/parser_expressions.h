/**
 * @file parser_expressions.h
 * @brief Expression parsing for the Eshkol parser
 * 
 * This file defines functions for parsing expressions in the Eshkol parser.
 */

#ifndef ESHKOL_PARSER_EXPRESSIONS_H
#define ESHKOL_PARSER_EXPRESSIONS_H

#include "frontend/parser/parser_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parse an expression
 * 
 * @param parser The parser
 * @return Expression node, or NULL on failure
 */
AstNode* parser_parse_expression(Parser* parser);

/**
 * @brief Parse a list expression (either a special form or a function call)
 * 
 * @param parser The parser
 * @return List node, or NULL on failure
 */
AstNode* parser_parse_list(Parser* parser);

/**
 * @brief Parse an atom (literal or identifier)
 * 
 * @param parser The parser
 * @return Atom node, or NULL on failure
 */
AstNode* parser_parse_atom(Parser* parser);

/**
 * @brief Parse a function call
 * 
 * @param parser The parser
 * @param callee Function being called
 * @param line Line number
 * @param column Column number
 * @return Call node, or NULL on failure
 */
AstNode* parser_parse_call(Parser* parser, AstNode* callee, size_t line, size_t column);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PARSER_EXPRESSIONS_H */
