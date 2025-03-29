/**
 * @file parser_error.h
 * @brief Error handling for the Eshkol parser
 * 
 * This file defines error handling functions for the Eshkol parser.
 */

#ifndef ESHKOL_PARSER_ERROR_H
#define ESHKOL_PARSER_ERROR_H

#include "frontend/parser/parser_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Report an error at the current token
 * 
 * @param parser The parser
 * @param message Error message
 */
void parser_error(Parser* parser, const char* message);

/**
 * @brief Synchronize after an error
 * 
 * This function skips tokens until it finds a synchronization point,
 * which is typically the start of a new statement or expression.
 * 
 * @param parser The parser
 */
void parser_synchronize(Parser* parser);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PARSER_ERROR_H */
