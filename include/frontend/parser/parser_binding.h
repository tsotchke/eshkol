/**
 * @file parser_binding.h
 * @brief Binding special form parsing for the Eshkol parser
 * 
 * This file defines functions for parsing binding special forms in the Eshkol parser.
 */

#ifndef ESHKOL_PARSER_BINDING_H
#define ESHKOL_PARSER_BINDING_H

#include "frontend/parser/parser_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parse a let special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Let node, or NULL on failure
 */
AstNode* parser_parse_let(Parser* parser, size_t line, size_t column);

/**
 * @brief Parse a set! special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Set node, or NULL on failure
 */
AstNode* parser_parse_set(Parser* parser, size_t line, size_t column);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PARSER_BINDING_H */
