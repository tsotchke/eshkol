/**
 * @file parser_error.c
 * @brief Error handling for the Eshkol parser
 */

#include "frontend/parser/parser_error.h"
#include "frontend/parser/parser_helpers.h"
#include <assert.h>

/**
 * @brief Report an error at the current token
 * 
 * @param parser The parser
 * @param message Error message
 */
void parser_error(Parser* parser, const char* message) {
    if (parser->panic_mode) {
        return;
    }
    
    parser->had_error = true;
    parser->panic_mode = true;
    
    // Report the error to the diagnostic context
    diagnostic_error(parser->diag, parser->current.line, parser->current.column, message);
}

/**
 * @brief Synchronize after an error
 * 
 * @param parser The parser
 */
void parser_synchronize(Parser* parser) {
    parser->panic_mode = false;
    
    while (!parser_is_at_end(parser)) {
        if (parser->previous.type == TOKEN_RPAREN) {
            return;
        }
        
        switch (parser->current.type) {
            case TOKEN_LPAREN:
                return;
            default:
                // Do nothing
                break;
        }
        
        parser_advance(parser);
    }
}
