/**
 * @file parser_lambda.c
 * @brief Lambda special form parsing for the Eshkol parser
 */

#include "frontend/parser/parser_lambda.h"
#include "frontend/parser/parser_helpers.h"
#include "frontend/parser/parser_error.h"
#include "frontend/parser/parser_expressions.h"
#include "frontend/parser/parser_sequence.h"
#include <assert.h>
#include <string.h>

/**
 * @brief Parse a lambda special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Lambda node, or NULL on failure
 */
AstNode* parser_parse_lambda(Parser* parser, size_t line, size_t column) {
    // Parse the parameter list
    if (!parser_match(parser, TOKEN_LPAREN)) {
        parser_error(parser, "Expected '(' before parameter list");
        return NULL;
    }
    
    // Create a new scope for the lambda
    uint64_t lambda_scope_id = binding_system_enter_scope(parser->bindings);
    if (lambda_scope_id == 0) {
        parser_error(parser, "Failed to create scope for lambda");
        return NULL;
    }
    
    size_t param_count = 0;
    Parameter** params = arena_alloc(parser->arena, sizeof(Parameter*) * 16); // Arbitrary initial capacity
    if (!params) {
        parser_error(parser, "Failed to allocate memory for parameters");
        binding_system_exit_scope(parser->bindings);
        return NULL;
    }
    
    while (!parser_check(parser, TOKEN_RPAREN) && !parser_is_at_end(parser)) {
        if (param_count >= 16) {
            parser_error(parser, "Too many parameters");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        if (!parser_match(parser, TOKEN_IDENTIFIER)) {
            parser_error(parser, "Expected parameter name");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        StringId param_name = parser->previous.value.string_id;
        
        // Add the parameter to the binding system
        uint64_t binding_id = binding_system_add_binding(parser->bindings, param_name, false);
        if (binding_id == 0) {
            parser_error(parser, "Failed to add binding for parameter");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        Parameter* param = parameter_create(parser->arena, param_name, NULL, parser->previous.line, parser->previous.column);
        if (!param) {
            parser_error(parser, "Failed to create parameter");
            binding_system_exit_scope(parser->bindings);
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
        binding_system_exit_scope(parser->bindings);
        return NULL;
    }
    
    while (!parser_check(parser, TOKEN_RPAREN) && !parser_is_at_end(parser)) {
        if (body_expr_count >= 16) {
            parser_error(parser, "Too many expressions in function body");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        AstNode* expr = parser_parse_expression(parser);
        if (!expr) {
            parser_error(parser, "Expected expression");
            binding_system_exit_scope(parser->bindings);
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
        binding_system_exit_scope(parser->bindings);
        return NULL;
    }
    
    if (!body) {
        parser_error(parser, "Failed to create function body");
        binding_system_exit_scope(parser->bindings);
        return NULL;
    }
    
    // Consume the closing parenthesis of the lambda form
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after lambda");
    
    // Exit the lambda scope
    binding_system_exit_scope(parser->bindings);
    
    // Create a lambda node
    AstNode* lambda = ast_create_lambda(parser->arena, params, param_count, NULL, body, line, column);
    if (!lambda) {
        parser_error(parser, "Failed to create lambda node");
        return NULL;
    }
    
    // Set the scope ID in the lambda node
    lambda->scope_id = lambda_scope_id;
    
    return lambda;
}
