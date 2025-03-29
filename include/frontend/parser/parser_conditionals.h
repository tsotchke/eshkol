/**
 * @file parser_conditionals.h
 * @brief Conditional special form parsing for the Eshkol parser
 * 
 * This file defines functions for parsing conditional special forms in the Eshkol parser.
 */

#ifndef ESHKOL_PARSER_CONDITIONALS_H
#define ESHKOL_PARSER_CONDITIONALS_H

#include "frontend/parser/parser_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parse an if special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return If node, or NULL on failure
 */
AstNode* parser_parse_if(Parser* parser, size_t line, size_t column);

/**
 * @brief Parse an and or or special form
 * 
 * @param parser The parser
 * @param type AST_AND or AST_OR
 * @param line Line number
 * @param column Column number
 * @return And or or node, or NULL on failure
 */
AstNode* parser_parse_and_or(Parser* parser, AstNodeType type, size_t line, size_t column);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PARSER_CONDITIONALS_H */
