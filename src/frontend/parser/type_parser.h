/**
 * @file type_parser.h
 * @brief Type parser for Eshkol
 * 
 * This file defines functions for parsing type annotations in the Eshkol language.
 */

#ifndef ESHKOL_TYPE_PARSER_H
#define ESHKOL_TYPE_PARSER_H

#include "frontend/parser/parser.h"
#include "core/type.h"
#include "frontend/ast/parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parse a type annotation
 * 
 * @param parser The parser
 * @return The parsed type, or NULL on failure
 */
Type* parse_type(Parser* parser);

/**
 * @brief Parse a parameter with optional type annotation
 * 
 * @param parser The parser
 * @param arena The arena allocator
 * @return The parsed parameter, or NULL on failure
 */
Parameter* parse_parameter(Parser* parser, Arena* arena);

/**
 * @brief Skip a type annotation
 * 
 * @param parser The parser
 */
void skip_type_annotation(Parser* parser);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TYPE_PARSER_H */
