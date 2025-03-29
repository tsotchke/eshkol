/**
 * @file parser_define.h
 * @brief Define special form parsing for the Eshkol parser
 * 
 * This file defines functions for parsing define special forms in the Eshkol parser.
 */

#ifndef ESHKOL_PARSER_DEFINE_H
#define ESHKOL_PARSER_DEFINE_H

#include "frontend/parser/parser_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parse a define special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Define node, or NULL on failure
 */
AstNode* parser_parse_define(Parser* parser, size_t line, size_t column);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PARSER_DEFINE_H */
