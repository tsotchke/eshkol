/**
 * @file parser_lambda.h
 * @brief Lambda special form parsing for the Eshkol parser
 * 
 * This file defines functions for parsing lambda special forms in the Eshkol parser.
 */

#ifndef ESHKOL_PARSER_LAMBDA_H
#define ESHKOL_PARSER_LAMBDA_H

#include "frontend/parser/parser_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parse a lambda special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Lambda node, or NULL on failure
 */
AstNode* parser_parse_lambda(Parser* parser, size_t line, size_t column);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PARSER_LAMBDA_H */
