/**
 * @file parser_special_forms.h
 * @brief Special form parsing for the Eshkol parser
 * 
 * This file defines functions for parsing special forms in the Eshkol parser.
 */

#ifndef ESHKOL_PARSER_SPECIAL_FORMS_H
#define ESHKOL_PARSER_SPECIAL_FORMS_H

#include "frontend/parser/parser_core.h"
#include "frontend/parser/parser_define.h"
#include "frontend/parser/parser_lambda.h"
#include "frontend/parser/parser_conditionals.h"
#include "frontend/parser/parser_binding.h"
#include "frontend/parser/parser_sequence.h"
#include "frontend/parser/parser_quote.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parse a special form
 * 
 * @param parser The parser
 * @param name Name of the special form
 * @param line Line number
 * @param column Column number
 * @return Special form node, or NULL on failure
 */
AstNode* parser_parse_special_form(Parser* parser, StringId name, size_t line, size_t column);

/**
 * @brief Parse a define special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Define node, or NULL on failure
 */
AstNode* parser_parse_define(Parser* parser, size_t line, size_t column);

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
 * @brief Parse a lambda special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Lambda node, or NULL on failure
 */
AstNode* parser_parse_lambda(Parser* parser, size_t line, size_t column);

/**
 * @brief Parse a begin special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Begin node, or NULL on failure
 */
AstNode* parser_parse_begin(Parser* parser, size_t line, size_t column);

/**
 * @brief Parse a quote special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Quote node, or NULL on failure
 */
AstNode* parser_parse_quote(Parser* parser, size_t line, size_t column);

/**
 * @brief Parse a set! special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Set node, or NULL on failure
 */
AstNode* parser_parse_set(Parser* parser, size_t line, size_t column);

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
 * @brief Parse a do special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Do node, or NULL on failure
 */
AstNode* parser_parse_do(Parser* parser, size_t line, size_t column);

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

#endif /* ESHKOL_PARSER_SPECIAL_FORMS_H */
