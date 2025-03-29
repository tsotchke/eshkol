/**
 * @file parser_binding.c
 * @brief Binding special form parsing for the Eshkol parser
 */

#include "frontend/parser/parser_binding.h"
#include "frontend/parser/parser_helpers.h"
#include "frontend/parser/parser_error.h"
#include "frontend/parser/parser_expressions.h"
#include "frontend/parser/parser_sequence.h"
#include <assert.h>
#include <string.h>

/**
 * @brief Parse a set! special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Set node, or NULL on failure
 */
AstNode* parser_parse_set(Parser* parser, size_t line, size_t column) {
    // Parse the variable name
    if (!parser_match(parser, TOKEN_IDENTIFIER)) {
        parser_error(parser, "Expected variable name");
        return NULL;
    }
    
    StringId name_str = parser->previous.value.string_id;
    AstNode* name = ast_create_identifier(parser->arena, name_str, parser->previous.line, parser->previous.column);
    if (!name) {
        parser_error(parser, "Failed to create identifier node");
        return NULL;
    }
    
    // Resolve the binding
    uint64_t binding_id = binding_system_resolve_binding(parser->bindings, name_str);
    if (binding_id == 0) {
        // Variable not found, report an error
        parser_error(parser, "Undefined variable");
        return NULL;
    }
    
    // Set the binding ID in the name node
    name->binding_id = binding_id;
    
    // Set the scope ID in the name node
    name->scope_id = binding_system_get_binding_scope(parser->bindings, binding_id);
    
    // Parse the value
    AstNode* value = parser_parse_expression(parser);
    if (!value) {
        parser_error(parser, "Expected expression");
        return NULL;
    }
    
    // Consume the closing parenthesis
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after set!");
    
    // Create a set node
    return ast_create_set(parser->arena, name, value, line, column);
}

/**
 * @brief Parse a let special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Let node, or NULL on failure
 */
AstNode* parser_parse_let(Parser* parser, size_t line, size_t column) {
    // Parse the bindings
    if (!parser_match(parser, TOKEN_LPAREN)) {
        parser_error(parser, "Expected '(' before bindings");
        return NULL;
    }
    
    // Create a new scope for the let
    uint64_t let_scope_id = binding_system_enter_scope(parser->bindings);
    if (let_scope_id == 0) {
        parser_error(parser, "Failed to create scope for let");
        return NULL;
    }
    
    size_t binding_count = 0;
    AstNode** bindings = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
    if (!bindings) {
        parser_error(parser, "Failed to allocate memory for bindings");
        binding_system_exit_scope(parser->bindings);
        return NULL;
    }
    
    while (!parser_check(parser, TOKEN_RPAREN) && !parser_is_at_end(parser)) {
        if (binding_count >= 16) {
            parser_error(parser, "Too many bindings");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        // Parse a binding
        if (!parser_match(parser, TOKEN_LPAREN)) {
            parser_error(parser, "Expected '(' before binding");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        // Parse the variable name
        if (!parser_match(parser, TOKEN_IDENTIFIER)) {
            parser_error(parser, "Expected variable name");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        StringId name_str = parser->previous.value.string_id;
        AstNode* name = ast_create_identifier(parser->arena, name_str, parser->previous.line, parser->previous.column);
        if (!name) {
            parser_error(parser, "Failed to create identifier node");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        // Parse the value
        AstNode* value = parser_parse_expression(parser);
        if (!value) {
            parser_error(parser, "Expected expression");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        // Consume the closing parenthesis of the binding
        parser_consume(parser, TOKEN_RPAREN, "Expected ')' after binding");
        
        // Add the binding to the binding system
        uint64_t binding_id = binding_system_add_binding(parser->bindings, name_str, false);
        if (binding_id == 0) {
            parser_error(parser, "Failed to add binding");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        // Set the binding ID in the name node
        name->binding_id = binding_id;
        
        // Create a variable definition node for the binding
        AstNode* binding = ast_create_variable_def(parser->arena, name, value, parser->previous.line, parser->previous.column);
        if (!binding) {
            parser_error(parser, "Failed to create binding node");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        bindings[binding_count++] = binding;
    }
    
    // Consume the closing parenthesis of the bindings list
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after bindings");
    
    // Parse the body
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
            parser_error(parser, "Too many expressions in let body");
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
        parser_error(parser, "Expected at least one expression in let body");
        binding_system_exit_scope(parser->bindings);
        return NULL;
    }
    
    if (!body) {
        parser_error(parser, "Failed to create let body");
        binding_system_exit_scope(parser->bindings);
        return NULL;
    }
    
    // Consume the closing parenthesis of the let form
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after let");
    
    // Exit the let scope
    binding_system_exit_scope(parser->bindings);
    
    // Create a let node
    // Create binding_nodes array (NULL for now since we don't have binding nodes)
    AstNode** binding_nodes = NULL;
    if (binding_count > 0) {
        binding_nodes = arena_alloc(parser->arena, sizeof(AstNode*) * binding_count);
        if (!binding_nodes) {
            parser_error(parser, "Failed to allocate memory for binding nodes");
            return NULL;
        }
        for (size_t i = 0; i < binding_count; i++) {
            binding_nodes[i] = NULL; // We don't have binding nodes yet
        }
    }
    
    AstNode* let = ast_create_let(parser->arena, bindings, binding_nodes, binding_count, body, line, column);
    if (!let) {
        parser_error(parser, "Failed to create let node");
        return NULL;
    }
    
    // Set the scope ID in the let node
    let->scope_id = let_scope_id;
    
    return let;
}
