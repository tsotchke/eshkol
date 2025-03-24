/**
 * @file parser.c
 * @brief Implementation of the parser for Eshkol
 */

#include "frontend/parser/parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>

// Forward declarations
static AstNode* parse_expression(Parser* parser);
static AstNode* parse_list(Parser* parser);
static AstNode* parse_atom(Parser* parser);
static AstNode* parse_special_form(Parser* parser, StringId name, size_t line, size_t column);
static AstNode* parse_define(Parser* parser, size_t line, size_t column);
static AstNode* parse_if(Parser* parser, size_t line, size_t column);
static AstNode* parse_lambda(Parser* parser, size_t line, size_t column);
static AstNode* parse_begin(Parser* parser, size_t line, size_t column);
static AstNode* parse_quote(Parser* parser, size_t line, size_t column);
static AstNode* parse_set(Parser* parser, size_t line, size_t column);
static AstNode* parse_let(Parser* parser, size_t line, size_t column);
static AstNode* parse_and_or(Parser* parser, AstNodeType type, size_t line, size_t column);
static AstNode* parse_call(Parser* parser, AstNode* callee, size_t line, size_t column);
static AstNode** parse_expressions(Parser* parser, size_t* count);

// Helper functions
static void advance(Parser* parser);
static bool check(Parser* parser, TokenType type);
static bool match(Parser* parser, TokenType type);
static void consume(Parser* parser, TokenType type, const char* message);
static void error(Parser* parser, const char* message);
static void synchronize(Parser* parser);
static bool is_at_end(Parser* parser);
static bool is_special_form(StringId name);

/**
 * @brief Create a new parser
 * 
 * @param arena Arena allocator
 * @param strings String table
 * @param diag Diagnostic context
 * @param lexer Lexer
 * @return A new parser, or NULL on failure
 */
Parser* parser_create(Arena* arena, StringTable* strings, DiagnosticContext* diag, Lexer* lexer) {
    assert(arena != NULL);
    assert(strings != NULL);
    assert(diag != NULL);
    assert(lexer != NULL);
    
    Parser* parser = arena_alloc(arena, sizeof(Parser));
    if (!parser) {
        return NULL;
    }
    
    parser->arena = arena;
    parser->strings = strings;
    parser->diag = diag;
    parser->lexer = lexer;
    parser->had_error = false;
    parser->panic_mode = false;
    
    // Initialize current and previous tokens
    parser->current = lexer_scan_token(lexer);
    parser->previous = parser->current;
    
    return parser;
}

/**
 * @brief Parse a program
 * 
 * @param parser The parser
 * @return The program AST node, or NULL on failure
 */
AstNode* parser_parse_program(Parser* parser) {
    size_t expr_count = 0;
    AstNode** exprs = NULL;
    
    // Parse top-level expressions
    exprs = parse_expressions(parser, &expr_count);
    if (!exprs && expr_count > 0) {
        return NULL;
    }
    
    // Create a program node
    AstNode* program = ast_create_program(
        parser->arena, 
        exprs, 
        expr_count, 
        parser->previous.line, 
        parser->previous.column
    );
    
    if (!program) {
        return NULL;
    }
    
    return program;
}

/**
 * @brief Parse a list of expressions
 * 
 * @param parser The parser
 * @param count Pointer to store the number of expressions
 * @return Array of expression nodes, or NULL on failure
 */
static AstNode** parse_expressions(Parser* parser, size_t* count) {
    // Allocate initial array
    size_t capacity = 8;
    AstNode** exprs = arena_alloc(parser->arena, sizeof(AstNode*) * capacity);
    if (!exprs) {
        *count = 0;
        return NULL;
    }
    
    size_t expr_count = 0;
    
    // Parse expressions until end of file
    while (!is_at_end(parser)) {
        // Skip comments
        while (match(parser, TOKEN_COMMENT)) {
            // Do nothing, just advance
        }
        
        // Check if we're at the end
        if (is_at_end(parser)) {
            break;
        }
        
        // Parse an expression
        AstNode* expr = parse_expression(parser);
        if (!expr) {
            // Error occurred, but we'll try to continue
            synchronize(parser);
            continue;
        }
        
        // Add the expression to the array
        if (expr_count >= capacity) {
            // Resize the array
            size_t new_capacity = capacity * 2;
            AstNode** new_exprs = arena_alloc(parser->arena, sizeof(AstNode*) * new_capacity);
            if (!new_exprs) {
                *count = expr_count;
                return exprs;
            }
            
            // Copy the old array
            for (size_t i = 0; i < expr_count; i++) {
                new_exprs[i] = exprs[i];
            }
            
            exprs = new_exprs;
            capacity = new_capacity;
        }
        
        exprs[expr_count++] = expr;
    }
    
    *count = expr_count;
    return exprs;
}

/**
 * @brief Parse an expression
 * 
 * @param parser The parser
 * @return Expression node, or NULL on failure
 */
static AstNode* parse_expression(Parser* parser) {
    // Skip comments
    while (match(parser, TOKEN_COMMENT)) {
        // Do nothing, just advance
    }
    
    // Check for different expression types
    if (match(parser, TOKEN_LPAREN)) {
        return parse_list(parser);
    } else if (match(parser, TOKEN_QUOTE)) {
        return parse_quote(parser, parser->previous.line, parser->previous.column);
    } else {
        return parse_atom(parser);
    }
}

/**
 * @brief Parse a list expression (either a special form or a function call)
 * 
 * @param parser The parser
 * @return List node, or NULL on failure
 */
static AstNode* parse_list(Parser* parser) {
    size_t line = parser->previous.line;
    size_t column = parser->previous.column;
    
    // Empty list
    if (match(parser, TOKEN_RPAREN)) {
        // Create a nil literal
        return ast_create_nil(parser->arena, line, column);
    }
    
    // Check if it's a special form
    if (match(parser, TOKEN_IDENTIFIER)) {
        StringId name = parser->previous.value.string_id;
        
        // Check if it's a special form
        if (is_special_form(name)) {
            return parse_special_form(parser, name, line, column);
        }
        
        // It's a function call
        AstNode* callee = ast_create_identifier(parser->arena, name, line, column);
        if (!callee) {
            error(parser, "Failed to create identifier node");
            return NULL;
        }
        
        return parse_call(parser, callee, line, column);
    }
    
    // It's a function call with a complex expression as the callee
    AstNode* callee = parse_expression(parser);
    if (!callee) {
        error(parser, "Expected expression");
        return NULL;
    }
    
    return parse_call(parser, callee, line, column);
}

/**
 * @brief Parse an atom (literal or identifier)
 * 
 * @param parser The parser
 * @return Atom node, or NULL on failure
 */
static AstNode* parse_atom(Parser* parser) {
    size_t line = parser->current.line;
    size_t column = parser->current.column;
    
    if (match(parser, TOKEN_NUMBER)) {
        return ast_create_number(parser->arena, parser->previous.value.number, line, column);
    } else if (match(parser, TOKEN_BOOLEAN)) {
        return ast_create_boolean(parser->arena, parser->previous.value.boolean, line, column);
    } else if (match(parser, TOKEN_CHARACTER)) {
        return ast_create_character(parser->arena, parser->previous.value.character, line, column);
    } else if (match(parser, TOKEN_STRING)) {
        return ast_create_string(parser->arena, parser->previous.value.string_id, line, column);
    } else if (match(parser, TOKEN_IDENTIFIER)) {
        return ast_create_identifier(parser->arena, parser->previous.value.string_id, line, column);
    } else {
        error(parser, "Expected expression");
        return NULL;
    }
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
static AstNode* parse_special_form(Parser* parser, StringId name, size_t line, size_t column) {
    // Check which special form it is
    if (strcmp(name, "define") == 0) {
        return parse_define(parser, line, column);
    } else if (strcmp(name, "if") == 0) {
        return parse_if(parser, line, column);
    } else if (strcmp(name, "lambda") == 0) {
        return parse_lambda(parser, line, column);
    } else if (strcmp(name, "begin") == 0) {
        return parse_begin(parser, line, column);
    } else if (strcmp(name, "set!") == 0) {
        return parse_set(parser, line, column);
    } else if (strcmp(name, "let") == 0) {
        return parse_let(parser, line, column);
    } else if (strcmp(name, "and") == 0) {
        return parse_and_or(parser, AST_AND, line, column);
    } else if (strcmp(name, "or") == 0) {
        return parse_and_or(parser, AST_OR, line, column);
    } else {
        // Unknown special form, treat as a function call
        AstNode* callee = ast_create_identifier(parser->arena, name, line, column);
        if (!callee) {
            error(parser, "Failed to create identifier node");
            return NULL;
        }
        
        return parse_call(parser, callee, line, column);
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
static AstNode* parse_define(Parser* parser, size_t line, size_t column) {
    // Check if it's a variable definition or a function definition
    if (match(parser, TOKEN_IDENTIFIER)) {
        // Variable definition
        StringId name_str = parser->previous.value.string_id;
        AstNode* name = ast_create_identifier(parser->arena, name_str, line, column);
        if (!name) {
            error(parser, "Failed to create identifier node");
            return NULL;
        }
        
        // Parse the value
        AstNode* value = parse_expression(parser);
        if (!value) {
            error(parser, "Expected expression");
            return NULL;
        }
        
        // Consume the closing parenthesis
        consume(parser, TOKEN_RPAREN, "Expected ')' after define");
        
        // Create a variable definition node
        return ast_create_variable_def(parser->arena, name, value, line, column);
    } else if (match(parser, TOKEN_LPAREN)) {
        // Function definition
        
        // Parse the function name
        if (!match(parser, TOKEN_IDENTIFIER)) {
            error(parser, "Expected function name");
            return NULL;
        }
        
        StringId name_str = parser->previous.value.string_id;
        AstNode* name = ast_create_identifier(parser->arena, name_str, line, column);
        if (!name) {
            error(parser, "Failed to create identifier node");
            return NULL;
        }
        
        // Parse the parameter list
        size_t param_count = 0;
        Parameter** params = arena_alloc(parser->arena, sizeof(Parameter*) * 16); // Arbitrary initial capacity
        if (!params) {
            error(parser, "Failed to allocate memory for parameters");
            return NULL;
        }
        
        while (!check(parser, TOKEN_RPAREN) && !is_at_end(parser)) {
            if (param_count >= 16) {
                error(parser, "Too many parameters");
                return NULL;
            }
            
            if (!match(parser, TOKEN_IDENTIFIER)) {
                error(parser, "Expected parameter name");
                return NULL;
            }
            
            StringId param_name = parser->previous.value.string_id;
            Parameter* param = parameter_create(parser->arena, param_name, NULL, parser->previous.line, parser->previous.column);
            if (!param) {
                error(parser, "Failed to create parameter");
                return NULL;
            }
            
            params[param_count++] = param;
        }
        
        // Consume the closing parenthesis of the parameter list
        consume(parser, TOKEN_RPAREN, "Expected ')' after parameter list");
        
        // Parse the function body
        AstNode* body = NULL;
        
        // Parse the body expressions
        size_t body_expr_count = 0;
        AstNode** body_exprs = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
        if (!body_exprs) {
            error(parser, "Failed to allocate memory for body expressions");
            return NULL;
        }
        
        while (!check(parser, TOKEN_RPAREN) && !is_at_end(parser)) {
            if (body_expr_count >= 16) {
                error(parser, "Too many expressions in function body");
                return NULL;
            }
            
            AstNode* expr = parse_expression(parser);
            if (!expr) {
                error(parser, "Expected expression");
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
            error(parser, "Expected at least one expression in function body");
            return NULL;
        }
        
        if (!body) {
            error(parser, "Failed to create function body");
            return NULL;
        }
        
        // Consume the closing parenthesis of the define form
        consume(parser, TOKEN_RPAREN, "Expected ')' after define");
        
        // Create a function definition node
        // Create param_nodes array (NULL for now since we don't have parameter nodes)
        AstNode** param_nodes = NULL;
        if (param_count > 0) {
            param_nodes = arena_alloc(parser->arena, sizeof(AstNode*) * param_count);
            if (!param_nodes) {
                error(parser, "Failed to allocate memory for parameter nodes");
                return NULL;
            }
            for (size_t i = 0; i < param_count; i++) {
                param_nodes[i] = NULL; // We don't have parameter nodes yet
            }
        }
        
        return ast_create_function_def(parser->arena, name, params, param_nodes, param_count, NULL, body, line, column);
    } else {
        error(parser, "Expected variable name or function definition");
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
static AstNode* parse_if(Parser* parser, size_t line, size_t column) {
    // Parse the condition
    AstNode* condition = parse_expression(parser);
    if (!condition) {
        error(parser, "Expected condition expression");
        return NULL;
    }
    
    // Parse the then branch
    AstNode* then_branch = parse_expression(parser);
    if (!then_branch) {
        error(parser, "Expected then expression");
        return NULL;
    }
    
    // Parse the else branch (optional)
    AstNode* else_branch = NULL;
    if (!check(parser, TOKEN_RPAREN)) {
        else_branch = parse_expression(parser);
        if (!else_branch) {
            error(parser, "Expected else expression");
            return NULL;
        }
    }
    
    // Consume the closing parenthesis
    consume(parser, TOKEN_RPAREN, "Expected ')' after if");
    
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
static AstNode* parse_lambda(Parser* parser, size_t line, size_t column) {
    // Parse the parameter list
    if (!match(parser, TOKEN_LPAREN)) {
        error(parser, "Expected '(' before parameter list");
        return NULL;
    }
    
    size_t param_count = 0;
    Parameter** params = arena_alloc(parser->arena, sizeof(Parameter*) * 16); // Arbitrary initial capacity
    if (!params) {
        error(parser, "Failed to allocate memory for parameters");
        return NULL;
    }
    
    while (!check(parser, TOKEN_RPAREN) && !is_at_end(parser)) {
        if (param_count >= 16) {
            error(parser, "Too many parameters");
            return NULL;
        }
        
        if (!match(parser, TOKEN_IDENTIFIER)) {
            error(parser, "Expected parameter name");
            return NULL;
        }
        
        StringId param_name = parser->previous.value.string_id;
        Parameter* param = parameter_create(parser->arena, param_name, NULL, parser->previous.line, parser->previous.column);
        if (!param) {
            error(parser, "Failed to create parameter");
            return NULL;
        }
        
        params[param_count++] = param;
    }
    
    // Consume the closing parenthesis of the parameter list
    consume(parser, TOKEN_RPAREN, "Expected ')' after parameter list");
    
    // Parse the function body
    AstNode* body = NULL;
    
    // Parse the body expressions
    size_t body_expr_count = 0;
    AstNode** body_exprs = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
    if (!body_exprs) {
        error(parser, "Failed to allocate memory for body expressions");
        return NULL;
    }
    
    while (!check(parser, TOKEN_RPAREN) && !is_at_end(parser)) {
        if (body_expr_count >= 16) {
            error(parser, "Too many expressions in function body");
            return NULL;
        }
        
        AstNode* expr = parse_expression(parser);
        if (!expr) {
            error(parser, "Expected expression");
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
        error(parser, "Expected at least one expression in function body");
        return NULL;
    }
    
    if (!body) {
        error(parser, "Failed to create function body");
        return NULL;
    }
    
    // Consume the closing parenthesis of the lambda form
    consume(parser, TOKEN_RPAREN, "Expected ')' after lambda");
    
    // Create a lambda node
    return ast_create_lambda(parser->arena, params, param_count, NULL, body, line, column);
}

/**
 * @brief Parse a begin special form
 * 
 * @param parser The parser
 * @param line Line number
 * @param column Column number
 * @return Begin node, or NULL on failure
 */
static AstNode* parse_begin(Parser* parser, size_t line, size_t column) {
    // Parse the expressions
    size_t expr_count = 0;
    AstNode** exprs = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
    if (!exprs) {
        error(parser, "Failed to allocate memory for expressions");
        return NULL;
    }
    
    while (!check(parser, TOKEN_RPAREN) && !is_at_end(parser)) {
        if (expr_count >= 16) {
            error(parser, "Too many expressions in begin");
            return NULL;
        }
        
        AstNode* expr = parse_expression(parser);
        if (!expr) {
            error(parser, "Expected expression");
            return NULL;
        }
        
        exprs[expr_count++] = expr;
    }
    
    // Consume the closing parenthesis
    consume(parser, TOKEN_RPAREN, "Expected ')' after begin");
    
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
static AstNode* parse_quote(Parser* parser, size_t line, size_t column) {
    // Parse the quoted expression
    AstNode* expr = parse_expression(parser);
    if (!expr) {
        error(parser, "Expected expression after quote");
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
static AstNode* parse_set(Parser* parser, size_t line, size_t column) {
    // Parse the variable name
    if (!match(parser, TOKEN_IDENTIFIER)) {
        error(parser, "Expected variable name");
        return NULL;
    }
    
    StringId name_str = parser->previous.value.string_id;
    AstNode* name = ast_create_identifier(parser->arena, name_str, parser->previous.line, parser->previous.column);
    if (!name) {
        error(parser, "Failed to create identifier node");
        return NULL;
    }
    
    // Parse the value
    AstNode* value = parse_expression(parser);
    if (!value) {
        error(parser, "Expected expression");
        return NULL;
    }
    
    // Consume the closing parenthesis
    consume(parser, TOKEN_RPAREN, "Expected ')' after set!");
    
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
static AstNode* parse_let(Parser* parser, size_t line, size_t column) {
    // Parse the bindings
    if (!match(parser, TOKEN_LPAREN)) {
        error(parser, "Expected '(' before bindings");
        return NULL;
    }
    
    size_t binding_count = 0;
    AstNode** bindings = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
    if (!bindings) {
        error(parser, "Failed to allocate memory for bindings");
        return NULL;
    }
    
    while (!check(parser, TOKEN_RPAREN) && !is_at_end(parser)) {
        if (binding_count >= 16) {
            error(parser, "Too many bindings");
            return NULL;
        }
        
        // Parse a binding
        if (!match(parser, TOKEN_LPAREN)) {
            error(parser, "Expected '(' before binding");
            return NULL;
        }
        
        // Parse the variable name
        if (!match(parser, TOKEN_IDENTIFIER)) {
            error(parser, "Expected variable name");
            return NULL;
        }
        
        StringId name_str = parser->previous.value.string_id;
        AstNode* name = ast_create_identifier(parser->arena, name_str, parser->previous.line, parser->previous.column);
        if (!name) {
            error(parser, "Failed to create identifier node");
            return NULL;
        }
        
        // Parse the value
        AstNode* value = parse_expression(parser);
        if (!value) {
            error(parser, "Expected expression");
            return NULL;
        }
        
        // Consume the closing parenthesis of the binding
        consume(parser, TOKEN_RPAREN, "Expected ')' after binding");
        
        // Create a variable definition node for the binding
        AstNode* binding = ast_create_variable_def(parser->arena, name, value, parser->previous.line, parser->previous.column);
        if (!binding) {
            error(parser, "Failed to create binding node");
            return NULL;
        }
        
        bindings[binding_count++] = binding;
    }
    
    // Consume the closing parenthesis of the bindings list
    consume(parser, TOKEN_RPAREN, "Expected ')' after bindings");
    
    // Parse the body
    AstNode* body = NULL;
    
    // Parse the body expressions
    size_t body_expr_count = 0;
    AstNode** body_exprs = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
    if (!body_exprs) {
        error(parser, "Failed to allocate memory for body expressions");
        return NULL;
    }
    
    while (!check(parser, TOKEN_RPAREN) && !is_at_end(parser)) {
        if (body_expr_count >= 16) {
            error(parser, "Too many expressions in let body");
            return NULL;
        }
        
        AstNode* expr = parse_expression(parser);
        if (!expr) {
            error(parser, "Expected expression");
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
        error(parser, "Expected at least one expression in let body");
        return NULL;
    }
    
    if (!body) {
        error(parser, "Failed to create let body");
        return NULL;
    }
    
    // Consume the closing parenthesis of the let form
    consume(parser, TOKEN_RPAREN, "Expected ')' after let");
    
    // Create a let node
    // Create binding_nodes array (NULL for now since we don't have binding nodes)
    AstNode** binding_nodes = NULL;
    if (binding_count > 0) {
        binding_nodes = arena_alloc(parser->arena, sizeof(AstNode*) * binding_count);
        if (!binding_nodes) {
            error(parser, "Failed to allocate memory for binding nodes");
            return NULL;
        }
        for (size_t i = 0; i < binding_count; i++) {
            binding_nodes[i] = NULL; // We don't have binding nodes yet
        }
    }
    
    return ast_create_let(parser->arena, bindings, binding_nodes, binding_count, body, line, column);
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
static AstNode* parse_and_or(Parser* parser, AstNodeType type, size_t line, size_t column) {
    // Parse the expressions
    size_t expr_count = 0;
    AstNode** exprs = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
    if (!exprs) {
        error(parser, "Failed to allocate memory for expressions");
        return NULL;
    }
    
    while (!check(parser, TOKEN_RPAREN) && !is_at_end(parser)) {
        if (expr_count >= 16) {
            error(parser, "Too many expressions in and/or");
            return NULL;
        }
        
        AstNode* expr = parse_expression(parser);
        if (!expr) {
            error(parser, "Expected expression");
            return NULL;
        }
        
        exprs[expr_count++] = expr;
    }
    
    // Consume the closing parenthesis
    consume(parser, TOKEN_RPAREN, "Expected ')' after and/or");
    
    // Create an and or or node
    if (type == AST_AND) {
        return ast_create_and(parser->arena, exprs, expr_count, line, column);
    } else {
        return ast_create_or(parser->arena, exprs, expr_count, line, column);
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
static AstNode* parse_call(Parser* parser, AstNode* callee, size_t line, size_t column) {
    // Parse the arguments
    size_t arg_count = 0;
    AstNode** args = arena_alloc(parser->arena, sizeof(AstNode*) * 16); // Arbitrary initial capacity
    if (!args) {
        error(parser, "Failed to allocate memory for arguments");
        return NULL;
    }
    
    while (!check(parser, TOKEN_RPAREN) && !is_at_end(parser)) {
        if (arg_count >= 16) {
            error(parser, "Too many arguments");
            return NULL;
        }
        
        AstNode* arg = parse_expression(parser);
        if (!arg) {
            error(parser, "Expected expression");
            return NULL;
        }
        
        args[arg_count++] = arg;
    }
    
    // Consume the closing parenthesis
    consume(parser, TOKEN_RPAREN, "Expected ')' after arguments");
    
    // Create a call node
    return ast_create_call(parser->arena, callee, args, arg_count, line, column);
}

/**
 * @brief Advance to the next token
 * 
 * @param parser The parser
 */
static void advance(Parser* parser) {
    parser->previous = parser->current;
    parser->current = lexer_scan_token(parser->lexer);
}

/**
 * @brief Check if the current token is of the specified type
 * 
 * @param parser The parser
 * @param type Token type to check
 * @return true if the current token is of the specified type, false otherwise
 */
static bool check(Parser* parser, TokenType type) {
    if (is_at_end(parser)) {
        return false;
    }
    
    return parser->current.type == type;
}

/**
 * @brief Match and consume the current token if it's of the specified type
 * 
 * @param parser The parser
 * @param type Token type to match
 * @return true if the current token was matched and consumed, false otherwise
 */
static bool match(Parser* parser, TokenType type) {
    if (!check(parser, type)) {
        return false;
    }
    
    advance(parser);
    return true;
}

/**
 * @brief Consume the current token if it's of the specified type, or report an error
 * 
 * @param parser The parser
 * @param type Token type to consume
 * @param message Error message to report if the token doesn't match
 */
static void consume(Parser* parser, TokenType type, const char* message) {
    if (check(parser, type)) {
        advance(parser);
        return;
    }
    
    error(parser, message);
}

/**
 * @brief Report an error at the current token
 * 
 * @param parser The parser
 * @param message Error message
 */
static void error(Parser* parser, const char* message) {
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
static void synchronize(Parser* parser) {
    parser->panic_mode = false;
    
    while (!is_at_end(parser)) {
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
        
        advance(parser);
    }
}

/**
 * @brief Check if we've reached the end of the file
 * 
 * @param parser The parser
 * @return true if we've reached the end of the file, false otherwise
 */
static bool is_at_end(Parser* parser) {
    return parser->current.type == TOKEN_EOF;
}

/**
 * @brief Check if a name is a special form
 * 
 * @param name Name to check
 * @return true if the name is a special form, false otherwise
 */
static bool is_special_form(StringId name) {
    return strcmp(name, "define") == 0 ||
           strcmp(name, "if") == 0 ||
           strcmp(name, "lambda") == 0 ||
           strcmp(name, "begin") == 0 ||
           strcmp(name, "set!") == 0 ||
           strcmp(name, "let") == 0 ||
           strcmp(name, "and") == 0 ||
           strcmp(name, "or") == 0 ||
           strcmp(name, "vector") == 0 ||
           strcmp(name, "v+") == 0 ||
           strcmp(name, "v-") == 0 ||
           strcmp(name, "v*") == 0 ||
           strcmp(name, "dot") == 0 ||
           strcmp(name, "cross") == 0 ||
           strcmp(name, "norm") == 0 ||
           strcmp(name, "gradient") == 0 ||
           strcmp(name, "divergence") == 0 ||
           strcmp(name, "curl") == 0 ||
           strcmp(name, "laplacian") == 0;
}
