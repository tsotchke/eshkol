/**
 * @file parser_sequence.c
 * @brief Sequence special form parsing for the Eshkol parser
 */

#include "frontend/parser/parser_sequence.h"
#include "frontend/parser/parser_helpers.h"
#include "frontend/parser/parser_error.h"
#include "frontend/parser/parser_expressions.h"
#include <assert.h>
#include <string.h>

/**
 * @brief Parse a begin special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Begin node, or NULL on failure
 */
AstNode* parser_parse_begin(Parser* parser, size_t line, size_t column) {
    // Parse the expressions
    size_t expr_count = 0;
    AstNode** exprs = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
    if (!exprs) {
        parser_error(parser, "Failed to allocate memory for expressions");
        return NULL;
    }
    
    while (!parser_check(parser, TOKEN_RPAREN) && !parser_is_at_end(parser)) {
        if (expr_count >= 16) {
            parser_error(parser, "Too many expressions in begin");
            return NULL;
        }
        
        AstNode* expr = parser_parse_expression(parser);
        if (!expr) {
            parser_error(parser, "Expected expression");
            return NULL;
        }
        
        exprs[expr_count++] = expr;
    }
    
    // Consume the closing parenthesis
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after begin");
    
    // Create a begin node
    return ast_create_begin(parser->arena, exprs, expr_count, line, column);
}

/**
 * @brief Parse a do special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Do node, or NULL on failure
 */
AstNode* parser_parse_do(Parser* parser, size_t line, size_t column) {
    // Parse the variable specifications
    if (!parser_match(parser, TOKEN_LPAREN)) {
        parser_error(parser, "Expected '(' before variable specifications");
        return NULL;
    }
    
    // Create a new scope for the do
    uint64_t do_scope_id = binding_system_enter_scope(parser->bindings);
    if (do_scope_id == 0) {
        parser_error(parser, "Failed to create scope for do");
        return NULL;
    }
    
    // Parse the variable specifications
    size_t var_count = 0;
    AstNode** vars = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
    AstNode** steps = arena_alloc(parser->arena, sizeof(AstNode*) * 16);
    if (!vars || !steps) {
        parser_error(parser, "Failed to allocate memory for variables");
        binding_system_exit_scope(parser->bindings);
        return NULL;
    }
    
    while (!parser_check(parser, TOKEN_RPAREN) && !parser_is_at_end(parser)) {
        if (var_count >= 16) {
            parser_error(parser, "Too many variables in do");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        // Parse a variable specification
        if (!parser_match(parser, TOKEN_LPAREN)) {
            parser_error(parser, "Expected '(' before variable specification");
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
        
        // Add the binding to the binding system
        uint64_t binding_id = binding_system_add_binding(parser->bindings, name_str, false);
        if (binding_id == 0) {
            parser_error(parser, "Failed to add binding");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        // Set the binding ID in the name node
        name->binding_id = binding_id;
        
        // Parse the initial value
        AstNode* init = parser_parse_expression(parser);
        if (!init) {
            parser_error(parser, "Expected initial value");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        // Parse the step expression
        AstNode* step = parser_parse_expression(parser);
        if (!step) {
            parser_error(parser, "Expected step expression");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        // Consume the closing parenthesis of the variable specification
        parser_consume(parser, TOKEN_RPAREN, "Expected ')' after variable specification");
        
        // Create a define node
        AstNode* var = ast_create_define(parser->arena, name, init, line, column);
        if (!var) {
            parser_error(parser, "Failed to create variable node");
            binding_system_exit_scope(parser->bindings);
            return NULL;
        }
        
        vars[var_count] = var;
        steps[var_count] = step;
        var_count++;
    }
    
    // Consume the closing parenthesis of the variable specifications
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after variable specifications");
    
    // Parse the test clause
    if (!parser_match(parser, TOKEN_LPAREN)) {
        parser_error(parser, "Expected '(' before test clause");
        binding_system_exit_scope(parser->bindings);
        return NULL;
    }
    
    // Parse the test expression
    AstNode* test = parser_parse_expression(parser);
    if (!test) {
        parser_error(parser, "Expected test expression");
        binding_system_exit_scope(parser->bindings);
        return NULL;
    }
    
    // Parse the result expression
    AstNode* result = parser_parse_expression(parser);
    if (!result) {
        parser_error(parser, "Expected result expression");
        binding_system_exit_scope(parser->bindings);
        return NULL;
    }
    
    // Consume the closing parenthesis of the test clause
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after test clause");
    
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
            parser_error(parser, "Too many expressions in do body");
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
    AstNode* body = NULL;
    if (body_expr_count > 1) {
        body = ast_create_begin(parser->arena, body_exprs, body_expr_count, line, column);
    } else if (body_expr_count == 1) {
        body = body_exprs[0];
    } else {
        body = ast_create_nil(parser->arena, line, column); // Empty body is allowed
    }
    
    if (!body) {
        parser_error(parser, "Failed to create do body");
        binding_system_exit_scope(parser->bindings);
        return NULL;
    }
    
    // Consume the closing parenthesis of the do form
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after do");
    
    // Exit the do scope
    binding_system_exit_scope(parser->bindings);
    
    // Create result array with just the one result expression
    AstNode** result_exprs = arena_alloc(parser->arena, sizeof(AstNode*));
    if (!result_exprs) {
        parser_error(parser, "Failed to allocate memory for result expressions");
        return NULL;
    }
    result_exprs[0] = result;

    // Create a do node
    AstNode* do_node = ast_create_do(parser->arena, vars, steps, var_count, test, result_exprs, 1, body_exprs, body_expr_count, line, column);
    if (!do_node) {
        parser_error(parser, "Failed to create do node");
        return NULL;
    }
    
    // Set the scope ID in the do node
    do_node->scope_id = do_scope_id;
    
    return do_node;
}
