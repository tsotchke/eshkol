/**
 * @file parser_special_forms.c
 * @brief Special form parsing for the Eshkol parser
 */

#include "frontend/parser/parser_special_forms.h"
#include "frontend/parser/parser_helpers.h"
#include "frontend/parser/parser_error.h"
#include "frontend/parser/parser_expressions.h"
#include <assert.h>
#include <string.h>

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

/**
 * @brief Parse an if special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return If node, or NULL on failure
 */
AstNode* parser_parse_if(Parser* parser, size_t line, size_t column) {
    // Parse the condition
    AstNode* condition = parser_parse_expression(parser);
    if (!condition) {
        parser_error(parser, "Expected condition expression");
        return NULL;
    }
    
    // Parse the then branch
    AstNode* then_branch = parser_parse_expression(parser);
    if (!then_branch) {
        parser_error(parser, "Expected then expression");
        return NULL;
    }
    
    // Parse the else branch (optional)
    AstNode* else_branch = NULL;
    if (!parser_check(parser, TOKEN_RPAREN)) {
        else_branch = parser_parse_expression(parser);
        if (!else_branch) {
            parser_error(parser, "Expected else expression");
            return NULL;
        }
    }
    
    // Consume the closing parenthesis
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after if");
    
    // Create an if node
    return ast_create_if(parser->arena, condition, then_branch, else_branch, line, column);
}

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
 * @brief Parse a quote special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Quote node, or NULL on failure
 */
AstNode* parser_parse_quote(Parser* parser, size_t line, size_t column) {
    // Parse the quoted expression
    AstNode* expr = parser_parse_expression(parser);
    if (!expr) {
        parser_error(parser, "Expected expression after quote");
        return NULL;
    }
    
    // Create a quote node
    return ast_create_quote(parser->arena, expr, line, column);
}

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

/**
 * @brief Parse an and or or special form
 * 
 * @param parser The parser
 * @param type AST_AND or AST_OR
 * @param line Line number
 * @param column Column number
 * @return And or or node, or NULL on failure
 */
AstNode* parser_parse_and_or(Parser* parser, AstNodeType type, size_t line, size_t column) {
    // Parse the expressions
    size_t expr_count = 0;
    AstNode** exprs = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
    if (!exprs) {
        parser_error(parser, "Failed to allocate memory for expressions");
        return NULL;
    }
    
    while (!parser_check(parser, TOKEN_RPAREN) && !parser_is_at_end(parser)) {
        if (expr_count >= 16) {
            parser_error(parser, "Too many expressions in and/or");
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
    parser_consume(parser, TOKEN_RPAREN, "Expected ')' after and/or");
    
    // Create an and or or node
    if (type == AST_AND) {
        return ast_create_and(parser->arena, exprs, expr_count, line, column);
    } else {
        return ast_create_or(parser->arena, exprs, expr_count, line, column);
    }
}
