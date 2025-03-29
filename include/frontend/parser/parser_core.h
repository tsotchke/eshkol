/**
 * @file parser_core.h
 * @brief Core functionality for the Eshkol parser
 * 
 * This file defines the core functionality for the Eshkol parser,
 * including parser creation and initialization.
 */

#ifndef ESHKOL_PARSER_CORE_H
#define ESHKOL_PARSER_CORE_H

#include "frontend/lexer/lexer.h"
#include "frontend/ast/ast.h"
#include "frontend/binding/binding.h"
#include "core/memory.h"
#include "core/string_table.h"
#include "core/diagnostics.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parser structure
 */
typedef struct {
    Arena* arena;           // Arena allocator
    StringTable* strings;   // String table
    DiagnosticContext* diag; // Diagnostic context
    Lexer* lexer;           // Lexer
    Token current;          // Current token
    Token previous;         // Previous token
    bool had_error;         // Whether an error occurred
    bool panic_mode;        // Whether we're in panic mode
    BindingSystem* bindings; // Binding system
} Parser;

/**
 * @brief Create a new parser
 * 
 * @param arena Arena allocator
 * @param strings String table
 * @param diag Diagnostic context
 * @param lexer Lexer
 * @return A new parser, or NULL on failure
 */
Parser* parser_create(Arena* arena, StringTable* strings, DiagnosticContext* diag, Lexer* lexer);

/**
 * @brief Parse a program
 * 
 * @param parser The parser
 * @return The program AST node, or NULL on failure
 */
AstNode* parser_parse_program(Parser* parser);

/**
 * @brief Parse a list of expressions
 * 
 * @param parser The parser
 * @param count Pointer to store the number of expressions
 * @return Array of expression nodes, or NULL on failure
 */
AstNode** parser_parse_expressions(Parser* parser, size_t* count);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PARSER_CORE_H */
