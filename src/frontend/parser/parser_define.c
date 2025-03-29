/**
 * @file parser_define.c
 * @brief Define special form parsing for the Eshkol parser
 */

#include "frontend/parser/parser_define.h"
#include "frontend/parser/parser_helpers.h"
#include "frontend/parser/parser_error.h"
#include "frontend/parser/parser_expressions.h"
#include <assert.h>
#include <string.h>

/**
 * @brief Parse a define special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Define node, or NULL on failure
 */
AstNode* parser_parse_define(Parser* parser, size_t line, size_t column) {
    // Check if it's a variable definition or a function definition
    if (parser_match(parser, TOKEN_IDENTIFIER)) {
        // Variable definition
        StringId name_str = parser->previous.value.string_id;
        AstNode* name = ast_create_identifier(parser->arena, name_str, line, column);
        if (!name) {
            parser_error(parser, "Failed to create identifier node");
            return NULL;
        }
        
        // Parse the value
        AstNode* value = parser_parse_expression(parser);
        if (!value) {
            parser_error(parser, "Expected expression");
            return NULL;
        }
        
        // Consume the closing parenthesis
        parser_consume(parser, TOKEN_RPAREN, "Expected ')' after define");
        
        // Add the binding to the binding system
        uint64_t binding_id = binding_system_add_binding(parser->bindings, name_str, true);
        if (binding_id == 0) {
            parser_error(parser, "Failed to add binding");
            return NULL;
        }
        
        // Set the binding ID in the name node
        name->binding_id = binding_id;
        
        // Create a variable definition node
        AstNode* var_def = ast_create_variable_def(parser->arena, name, value, line, column);
        if (!var_def) {
            parser_error(parser, "Failed to create variable definition node");
            return NULL;
        }
        
        return var_def;
    } else if (parser_match(parser, TOKEN_LPAREN)) {
        // Function definition
        
        // Parse the function name
        if (!parser_match(parser, TOKEN_IDENTIFIER)) {
            parser_error(parser, "Expected function name");
            return NULL;
        }
        
        StringId name_str = parser->previous.value.string_id;
        AstNode* name = ast_create_identifier(parser->arena, name_str, line, column);
        if (!name) {
            parser_error(parser, "Failed to create identifier node");
            return NULL;
        }
        
        // Parse the parameter list
        size_t param_count = 0;
        Parameter** params = arena_alloc(parser->arena, sizeof(Parameter*) * 16); // Arbitrary initial capacity
        if (!params) {
            parser_error(parser, "Failed to allocate memory for parameters");
            return NULL;
        }
        
        while (!parser_check(parser, TOKEN_RPAREN) && !parser_is_at_end(parser)) {
            if (param_count >= 16) {
                parser_error(parser, "Too many parameters");
                return NULL;
            }
            
            if (!parser_match(parser, TOKEN_IDENTIFIER)) {
                parser_error(parser, "Expected parameter name");
                return NULL;
            }
            
            StringId param_name = parser->previous.value.string_id;
            Parameter* param = parameter_create(parser->arena, param_name, NULL, parser->previous.line, parser->previous.column);
            if (!param) {
                parser_error(parser, "Failed to create parameter");
                return NULL;
            }
            
            params[param_count++] = param;
        }
        
        // Consume the closing parenthesis of the parameter list
        parser_consume(parser, TOKEN_RPAREN, "Expected ')' after parameter list");
        
        // Parse the function body
        AstNode* body = NULL;
        
        // Parse the body expressions
        size_t body_expr_count = 0;
        AstNode** body_exprs = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
        if (!body_exprs) {
            parser_error(parser, "Failed to allocate memory for body expressions");
            return NULL;
        }
        
        while (!parser_check(parser, TOKEN_RPAREN) && !parser_is_at_end(parser)) {
            if (body_expr_count >= 16) {
                parser_error(parser, "Too many expressions in function body");
                return NULL;
            }
            
            AstNode* expr = parser_parse_expression(parser);
            if (!expr) {
                parser_error(parser, "Expected expression");
                return NULL;
            }
            
            body_exprs[body_expr_count++] = expr;
        }
        
        // Create a begin node for the body if there are multiple expressions
        if (body_expr_count > 1) {
            body = ast_create_begin(parser->arena, body_exprs, body_expr_count, line, column);
        } else if (body_expr_count == 1) {
            body = body_exprs[0];
        } else {
            parser_error(parser, "Expected at least one expression in function body");
            return NULL;
        }
        
        if (!body) {
            parser_error(parser, "Failed to create function body");
            return NULL;
        }
        
        // Consume the closing parenthesis of the define form
        parser_consume(parser, TOKEN_RPAREN, "Expected ')' after define");
        
        // Create a function definition node
        // Create param_nodes array (NULL for now since we don't have parameter nodes)
        AstNode** param_nodes = NULL;
        if (param_count > 0) {
            param_nodes = arena_alloc(parser->arena, sizeof(AstNode*) * param_count);
            if (!param_nodes) {
                parser_error(parser, "Failed to allocate memory for parameter nodes");
                return NULL;
            }
            for (size_t i = 0; i < param_count; i++) {
                param_nodes[i] = NULL; // We don't have parameter nodes yet
            }
        }
        
        return ast_create_function_def(parser->arena, name, params, param_nodes, param_count, NULL, body, line, column);
    } else {
        parser_error(parser, "Expected variable name or function definition");
        return NULL;
    }
}
