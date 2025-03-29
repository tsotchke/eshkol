/**
 * @file parser_quote.h
 * @brief Quote special form parsing for the Eshkol parser
 * 
 * This file defines functions for parsing quote special forms in the Eshkol parser.
 */

#ifndef ESHKOL_PARSER_QUOTE_H
#define ESHKOL_PARSER_QUOTE_H

#include "frontend/parser/parser_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parse a quote special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Quote node, or NULL on failure
 */
AstNode* parser_parse_quote(Parser* parser, size_t line, size_t column);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PARSER_QUOTE_H */
