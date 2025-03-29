/**
 * @file parser_expressions.c
 * @brief Expression parsing for the Eshkol parser
 */

#include "frontend/parser/parser_expressions.h"
#include "frontend/parser/parser_helpers.h"
#include "frontend/parser/parser_error.h"
#include "frontend/parser/parser_special_forms.h"
#include "frontend/parser/parser_define.h"
#include "frontend/parser/parser_lambda.h"
#include "frontend/parser/parser_conditionals.h"
#include "frontend/parser/parser_binding.h"
#include "frontend/parser/parser_sequence.h"
#include "frontend/parser/parser_quote.h"
#include <assert.h>
#include <string.h>

/**
 * @brief Parse an expression
 * 
 * @param parser The parser
 * @return Expression node, or NULL on failure
 */
AstNode* parser_parse_expression(Parser* parser) {
    // Skip comments
    while (parser_match(parser, TOKEN_COMMENT)) {
        // Do nothing, just advance
    }
    
    // Check for different expression types
    if (parser_match(parser, TOKEN_LPAREN)) {
        return parser_parse_list(parser);
    } else if (parser_match(parser, TOKEN_QUOTE)) {
        return parser_parse_quote(parser, parser->previous.line, parser->previous.column);
    } else {
        return parser_parse_atom(parser);
    }
}

/**
 * @brief Parse a list expression (either a special form or a function call)
 * 
 * @param parser The parser
 * @return List node, or NULL on failure
 */
AstNode* parser_parse_list(Parser* parser) {
    size_t line = parser->previous.line;
    size_t column = parser->previous.column;
    
    // Empty list
    if (parser_match(parser, TOKEN_RPAREN)) {
        // Create a nil literal
        return ast_create_nil(parser->arena, line, column);
    }
    
    // Check if it's a special form
    if (parser_match(parser, TOKEN_IDENTIFIER)) {
        StringId name = parser->previous.value.string_id;
        
        // Check if it's a special form
        if (parser_is_special_form(name)) {
            return parser_parse_special_form(parser, name, line, column);
        }
        
        // It's a function call
        AstNode* callee = ast_create_identifier(parser->arena, name, line, column);
        if (!callee) {
            parser_error(parser, "Failed to create identifier node");
            return NULL;
        }
        
        return parser_parse_call(parser, callee, line, column);
    }
    
    // It's a function call with a complex expression as the callee
    AstNode* callee = parser_parse_expression(parser);
    if (!callee) {
        parser_error(parser, "Expected expression");
        return NULL;
    }
    
    return parser_parse_call(parser, callee, line, column);
}

/**
 * @brief Parse a special form
 * 
 * @param parser The parser
 * @param name Name of the special form
 * @param line Line number
 * @param column Column number
 * @return Special form node, or NULL on failure
 */
AstNode* parser_parse_special_form(Parser* parser, StringId name, size_t line, size_t column) {
    // Check which special form it is
    if (strcmp(name, "define") == 0) {
        return parser_parse_define(parser, line, column);
    } else if (strcmp(name, "if") == 0) {
        return parser_parse_if(parser, line, column);
    } else if (strcmp(name, "lambda") == 0) {
        return parser_parse_lambda(parser, line, column);
    } else if (strcmp(name, "begin") == 0) {
        return parser_parse_begin(parser, line, column);
    } else if (strcmp(name, "set!") == 0) {
        return parser_parse_set(parser, line, column);
    } else if (strcmp(name, "let") == 0) {
        return parser_parse_let(parser, line, column);
    } else if (strcmp(name, "do") == 0) {
        return parser_parse_do(parser, line, column);
    } else if (strcmp(name, "and") == 0) {
        return parser_parse_and_or(parser, AST_AND, line, column);
    } else if (strcmp(name, "or") == 0) {
        return parser_parse_and_or(parser, AST_OR, line, column);
    } else {
        // Unknown special form, treat as a function call
        AstNode* callee = ast_create_identifier(parser->arena, name, line, column);
        if (!callee) {
            parser_error(parser, "Failed to create identifier node");
            return NULL;
        }
        
        return parser_parse_call(parser, callee, line, column);
    }
}

/**
 * @brief Parse an atom (literal or identifier)
 * 
 * @param parser The parser
 * @return Atom node, or NULL on failure
 */
AstNode* parser_parse_atom(Parser* parser) {
    size_t line = parser->current.line;
    size_t column = parser->current.column;
    
    if (parser_match(parser, TOKEN_NUMBER)) {
        return ast_create_number(parser->arena, parser->previous.value.number, line, column);
    } else if (parser_match(parser, TOKEN_BOOLEAN)) {
        return ast_create_boolean(parser->arena, parser->previous.value.boolean, line, column);
    } else if (parser_match(parser, TOKEN_CHARACTER)) {
        return ast_create_character(parser->arena, parser->previous.value.character, line, column);
    } else if (parser_match(parser, TOKEN_STRING)) {
        return ast_create_string(parser->arena, parser->previous.value.string_id, line, column);
    } else if (parser_match(parser, TOKEN_IDENTIFIER)) {
        StringId name = parser->previous.value.string_id;
        AstNode* node = ast_create_identifier(parser->arena, name, line, column);
        if (!node) {
            parser_error(parser, "Failed to create identifier node");
            return NULL;
        }
        
        // Resolve the binding
        uint64_t binding_id = binding_system_resolve_binding(parser->bindings, name);
        if (binding_id != 0) {
            // Set the binding ID in the node
            node->binding_id = binding_id;
            
            // Set the scope ID in the node
            node->scope_id = binding_system_get_binding_scope(parser->bindings, binding_id);
        }
        
        return node;
    } else {
        parser_error(parser, "Expected expression");
        return NULL;
    }
}

/**
 * @brief Parse a function call
 * 
 * @param parser The parser
 * @param callee Function being called
 * @param line Line number
 * @param column Column number
 * @return Call node, or NULL on failure
 */
AstNode* parser_parse_call(Parser* parser, AstNode* callee, size_t line, size_t column) {
    // Parse the arguments
    size_t arg_count = 0;
    AstNode** args = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
    if (!args) {
        parser_error(parser, "Failed to allocate memory for arguments");
        return NULL;
    }
    
    while (!parser_check(parser, TOKEN_RPAREN) && !parser_is_at_end(parser)) {
        if (arg_count >= 16) {
            parser_error(parser, "Too many arguments");
            return NULL;
        }
        
        AstNode* arg = parser_parse_expression(parser);
        if (!arg) {
            parser_error(parser, "Expected expression");
            return NULL;
        }
        
        args[arg_count++] = arg;
    }
    
    // Consume the closing parenthesis
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after arguments");
    
    // Create a call node
    return ast_create_call(parser->arena, callee, args, arg_count, line, column);
}
