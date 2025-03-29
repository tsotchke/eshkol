/**
 * @file ast_create.c
 * @brief Functions for creating AST nodes
 */

#include "frontend/ast/create/ast_create.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Create a number literal node
 * 
 * @param arena Arena allocator
 * @param value Number value
 * @param line Line number
 * @param column Column number
 * @return A new number literal node, or NULL on failure
 */
AstNode* ast_create_number(Arena* arena, double value, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_LITERAL_NUMBER, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.number.value = value;
    
    return node;
}

/**
 * @brief Create a boolean literal node
 * 
 * @param arena Arena allocator
 * @param value Boolean value
 * @param line Line number
 * @param column Column number
 * @return A new boolean literal node, or NULL on failure
 */
AstNode* ast_create_boolean(Arena* arena, bool value, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_LITERAL_BOOLEAN, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.boolean.value = value;
    
    return node;
}

/**
 * @brief Create a character literal node
 * 
 * @param arena Arena allocator
 * @param value Character value
 * @param line Line number
 * @param column Column number
 * @return A new character literal node, or NULL on failure
 */
AstNode* ast_create_character(Arena* arena, char value, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_LITERAL_CHARACTER, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.character.value = value;
    
    return node;
}

/**
 * @brief Create a string literal node
 * 
 * @param arena Arena allocator
 * @param value String value
 * @param line Line number
 * @param column Column number
 * @return A new string literal node, or NULL on failure
 */
AstNode* ast_create_string(Arena* arena, StringId value, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_LITERAL_STRING, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.string.value = value;
    
    return node;
}

/**
 * @brief Create a vector literal node
 * 
 * @param arena Arena allocator
 * @param elements Vector elements
 * @param count Number of elements
 * @param line Line number
 * @param column Column number
 * @return A new vector literal node, or NULL on failure
 */
AstNode* ast_create_vector(Arena* arena, AstNode** elements, size_t count, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_LITERAL_VECTOR, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.vector.elements = elements;
    node->as.vector.count = count;
    
    return node;
}

/**
 * @brief Create a nil literal node
 * 
 * @param arena Arena allocator
 * @param line Line number
 * @param column Column number
 * @return A new nil literal node, or NULL on failure
 */
AstNode* ast_create_nil(Arena* arena, size_t line, size_t column) {
    return ast_create_node(arena, AST_LITERAL_NIL, line, column);
}

/**
 * @brief Create an identifier node
 * 
 * @param arena Arena allocator
 * @param name Identifier name
 * @param line Line number
 * @param column Column number
 * @return A new identifier node, or NULL on failure
 */
AstNode* ast_create_identifier(Arena* arena, StringId name, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_IDENTIFIER, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.identifier.name = name;
    
    return node;
}

/**
 * @brief Create a define node
 * 
 * @param arena Arena allocator
 * @param name Name being defined
 * @param value Value being bound
 * @param line Line number
 * @param column Column number
 * @return A new define node, or NULL on failure
 */
AstNode* ast_create_define(Arena* arena, AstNode* name, AstNode* value, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_DEFINE, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.define.name = name;
    node->as.define.value = value;
    
    return node;
}

/**
 * @brief Create a lambda node
 * 
 * @param arena Arena allocator
 * @param params Parameter list
 * @param param_count Number of parameters
 * @param return_type Return type (can be NULL for untyped)
 * @param body Function body
 * @param line Line number
 * @param column Column number
 * @return A new lambda node, or NULL on failure
 */
AstNode* ast_create_lambda(Arena* arena, Parameter** params, size_t param_count, Type* return_type, AstNode* body, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_LAMBDA, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.lambda.params = params;
    node->as.lambda.param_count = param_count;
    node->as.lambda.return_type = return_type;
    node->as.lambda.body = body;
    
    return node;
}

/**
 * @brief Create an if node
 * 
 * @param arena Arena allocator
 * @param condition Condition expression
 * @param then_branch Then branch
 * @param else_branch Else branch
 * @param line Line number
 * @param column Column number
 * @return A new if node, or NULL on failure
 */
AstNode* ast_create_if(Arena* arena, AstNode* condition, AstNode* then_branch, AstNode* else_branch, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_IF, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.if_expr.condition = condition;
    node->as.if_expr.then_branch = then_branch;
    node->as.if_expr.else_branch = else_branch;
    
    return node;
}

/**
 * @brief Create a begin node
 * 
 * @param arena Arena allocator
 * @param exprs Expressions
 * @param expr_count Number of expressions
 * @param line Line number
 * @param column Column number
 * @return A new begin node, or NULL on failure
 */
AstNode* ast_create_begin(Arena* arena, AstNode** exprs, size_t expr_count, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_BEGIN, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.begin.exprs = exprs;
    node->as.begin.expr_count = expr_count;
    
    return node;
}

/**
 * @brief Create a quote node
 * 
 * @param arena Arena allocator
 * @param expr Quoted expression
 * @param line Line number
 * @param column Column number
 * @return A new quote node, or NULL on failure
 */
AstNode* ast_create_quote(Arena* arena, AstNode* expr, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_QUOTE, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.quote.expr = expr;
    
    return node;
}

/**
 * @brief Create a set! node
 * 
 * @param arena Arena allocator
 * @param name Name being set
 * @param value Value being assigned
 * @param line Line number
 * @param column Column number
 * @return A new set! node, or NULL on failure
 */
AstNode* ast_create_set(Arena* arena, AstNode* name, AstNode* value, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_SET, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.set.name = name;
    node->as.set.value = value;
    
    return node;
}

/**
 * @brief Create a let node
 * 
 * @param arena Arena allocator
 * @param bindings Bindings
 * @param binding_nodes Binding nodes (for type information)
 * @param binding_count Number of bindings
 * @param body Body expression
 * @param line Line number
 * @param column Column number
 * @return A new let node, or NULL on failure
 */
AstNode* ast_create_let(Arena* arena, AstNode** bindings, AstNode** binding_nodes, size_t binding_count, AstNode* body, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_LET, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.let.bindings = bindings;
    node->as.let.binding_nodes = binding_nodes;
    node->as.let.binding_count = binding_count;
    node->as.let.body = body;
    
    return node;
}

/**
 * @brief Create a letrec node
 * 
 * @param arena Arena allocator
 * @param bindings Bindings
 * @param binding_nodes Binding nodes (for type information)
 * @param binding_count Number of bindings
 * @param body Body expression
 * @param line Line number
 * @param column Column number
 * @return A new letrec node, or NULL on failure
 */
AstNode* ast_create_letrec(Arena* arena, AstNode** bindings, AstNode** binding_nodes, size_t binding_count, AstNode* body, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_LETREC, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.let.bindings = bindings;
    node->as.let.binding_nodes = binding_nodes;
    node->as.let.binding_count = binding_count;
    node->as.let.body = body;
    
    return node;
}

/**
 * @brief Create a let* node
 * 
 * @param arena Arena allocator
 * @param bindings Bindings
 * @param binding_nodes Binding nodes (for type information)
 * @param binding_count Number of bindings
 * @param body Body expression
 * @param line Line number
 * @param column Column number
 * @return A new let* node, or NULL on failure
 */
AstNode* ast_create_letstar(Arena* arena, AstNode** bindings, AstNode** binding_nodes, size_t binding_count, AstNode* body, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_LETSTAR, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.let.bindings = bindings;
    node->as.let.binding_nodes = binding_nodes;
    node->as.let.binding_count = binding_count;
    node->as.let.body = body;
    
    return node;
}

/**
 * @brief Create an and node
 * 
 * @param arena Arena allocator
 * @param exprs Expressions
 * @param expr_count Number of expressions
 * @param line Line number
 * @param column Column number
 * @return A new and node, or NULL on failure
 */
AstNode* ast_create_and(Arena* arena, AstNode** exprs, size_t expr_count, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_AND, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.logical.exprs = exprs;
    node->as.logical.expr_count = expr_count;
    
    return node;
}

/**
 * @brief Create an or node
 * 
 * @param arena Arena allocator
 * @param exprs Expressions
 * @param expr_count Number of expressions
 * @param line Line number
 * @param column Column number
 * @return A new or node, or NULL on failure
 */
AstNode* ast_create_or(Arena* arena, AstNode** exprs, size_t expr_count, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_OR, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.logical.exprs = exprs;
    node->as.logical.expr_count = expr_count;
    
    return node;
}

/**
 * @brief Create a cond node
 * 
 * @param arena Arena allocator
 * @param clauses Clauses
 * @param clause_count Number of clauses
 * @param line Line number
 * @param column Column number
 * @return A new cond node, or NULL on failure
 */
AstNode* ast_create_cond(Arena* arena, AstNode** clauses, size_t clause_count, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_COND, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.cond.clauses = clauses;
    node->as.cond.clause_count = clause_count;
    
    return node;
}

/**
 * @brief Create a case node
 * 
 * @param arena Arena allocator
 * @param key Key expression
 * @param clauses Clauses
 * @param clause_count Number of clauses
 * @param line Line number
 * @param column Column number
 * @return A new case node, or NULL on failure
 */
AstNode* ast_create_case(Arena* arena, AstNode* key, AstNode** clauses, size_t clause_count, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_CASE, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.case_expr.key = key;
    node->as.case_expr.clauses = clauses;
    node->as.case_expr.clause_count = clause_count;
    
    return node;
}

/**
 * @brief Create a do node
 * 
 * @param arena Arena allocator
 * @param bindings Bindings
 * @param steps Step expressions
 * @param binding_count Number of bindings
 * @param test Test expression
 * @param result Result expressions
 * @param result_count Number of result expressions
 * @param body Body expressions
 * @param body_count Number of body expressions
 * @param line Line number
 * @param column Column number
 * @return A new do node, or NULL on failure
 */
AstNode* ast_create_do(Arena* arena, AstNode** bindings, AstNode** steps, size_t binding_count, AstNode* test, AstNode** result, size_t result_count, AstNode** body, size_t body_count, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_DO, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.do_expr.bindings = bindings;
    node->as.do_expr.steps = steps;
    node->as.do_expr.binding_count = binding_count;
    node->as.do_expr.test = test;
    node->as.do_expr.result = result;
    node->as.do_expr.result_count = result_count;
    node->as.do_expr.body = body;
    node->as.do_expr.body_count = body_count;
    
    return node;
}

/**
 * @brief Create a delay node
 * 
 * @param arena Arena allocator
 * @param expr Delayed expression
 * @param line Line number
 * @param column Column number
 * @return A new delay node, or NULL on failure
 */
AstNode* ast_create_delay(Arena* arena, AstNode* expr, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_DELAY, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.delay.expr = expr;
    
    return node;
}

/**
 * @brief Create a quasiquote node
 * 
 * @param arena Arena allocator
 * @param expr Quasiquoted expression
 * @param line Line number
 * @param column Column number
 * @return A new quasiquote node, or NULL on failure
 */
AstNode* ast_create_quasiquote(Arena* arena, AstNode* expr, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_QUASIQUOTE, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.quasiquote.expr = expr;
    
    return node;
}

/**
 * @brief Create an unquote node
 * 
 * @param arena Arena allocator
 * @param expr Unquoted expression
 * @param line Line number
 * @param column Column number
 * @return A new unquote node, or NULL on failure
 */
AstNode* ast_create_unquote(Arena* arena, AstNode* expr, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_UNQUOTE, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.unquote.expr = expr;
    
    return node;
}

/**
 * @brief Create an unquote-splicing node
 * 
 * @param arena Arena allocator
 * @param expr Unquote-spliced expression
 * @param line Line number
 * @param column Column number
 * @return A new unquote-splicing node, or NULL on failure
 */
AstNode* ast_create_unquote_splicing(Arena* arena, AstNode* expr, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_UNQUOTE_SPLICING, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.unquote_splicing.expr = expr;
    
    return node;
}

/**
 * @brief Create a call node
 * 
 * @param arena Arena allocator
 * @param callee Function being called
 * @param args Arguments
 * @param arg_count Number of arguments
 * @param line Line number
 * @param column Column number
 * @return A new call node, or NULL on failure
 */
AstNode* ast_create_call(Arena* arena, AstNode* callee, AstNode** args, size_t arg_count, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_CALL, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.call.callee = callee;
    node->as.call.args = args;
    node->as.call.arg_count = arg_count;
    
    return node;
}

/**
 * @brief Create a sequence node
 * 
 * @param arena Arena allocator
 * @param exprs Expressions
 * @param expr_count Number of expressions
 * @param line Line number
 * @param column Column number
 * @return A new sequence node, or NULL on failure
 */
AstNode* ast_create_sequence(Arena* arena, AstNode** exprs, size_t expr_count, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_SEQUENCE, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.sequence.exprs = exprs;
    node->as.sequence.expr_count = expr_count;
    
    return node;
}

/**
 * @brief Create a function definition node
 * 
 * @param arena Arena allocator
 * @param name Function name
 * @param params Parameter list
 * @param param_nodes Parameter nodes (for type information)
 * @param param_count Number of parameters
 * @param return_type Return type (can be NULL for untyped)
 * @param body Function body
 * @param line Line number
 * @param column Column number
 * @return A new function definition node, or NULL on failure
 */
AstNode* ast_create_function_def(Arena* arena, AstNode* name, Parameter** params, AstNode** param_nodes, size_t param_count, Type* return_type, AstNode* body, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_FUNCTION_DEF, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.function_def.name = name;
    node->as.function_def.params = params;
    node->as.function_def.param_nodes = param_nodes;
    node->as.function_def.param_count = param_count;
    node->as.function_def.return_type = return_type;
    node->as.function_def.body = body;
    
    return node;
}

/**
 * @brief Create a variable definition node
 * 
 * @param arena Arena allocator
 * @param name Variable name
 * @param value Initial value
 * @param line Line number
 * @param column Column number
 * @return A new variable definition node, or NULL on failure
 */
AstNode* ast_create_variable_def(Arena* arena, AstNode* name, AstNode* value, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_VARIABLE_DEF, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.variable_def.name = name;
    node->as.variable_def.value = value;
    
    return node;
}

/**
 * @brief Create a type declaration node
 * 
 * @param arena Arena allocator
 * @param function_name Name of the function being typed
 * @param type Function type (including params and return)
 * @param line Line number
 * @param column Column number
 * @return A new type declaration node, or NULL on failure
 */
AstNode* ast_create_type_declaration(Arena* arena, StringId function_name, Type* type, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_TYPE_DECLARATION, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.type_declaration.function_name = function_name;
    node->as.type_declaration.type = type;
    node->type_info = type;  // Store the type in the node itself
    
    return node;
}

/**
 * @brief Create a program node
 * 
 * @param arena Arena allocator
 * @param exprs Top-level expressions
 * @param expr_count Number of expressions
 * @param line Line number
 * @param column Column number
 * @return A new program node, or NULL on failure
 */
AstNode* ast_create_program(Arena* arena, AstNode** exprs, size_t expr_count, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_PROGRAM, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.program.exprs = exprs;
    node->as.program.expr_count = expr_count;
    
    return node;
}

/**
 * @brief Create an error node
 * 
 * @param arena Arena allocator
 * @param message Error message
 * @param line Line number
 * @param column Column number
 * @return A new error node, or NULL on failure
 */
AstNode* ast_create_error(Arena* arena, StringId message, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_ERROR, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.error.message = message;
    
    return node;
}

/**
 * @brief Create a binding pair node
 * 
 * @param arena Arena allocator
 * @param name Name being bound
 * @param value Value being bound
 * @param line Line number
 * @param column Column number
 * @return A new binding pair node, or NULL on failure
 */
AstNode* ast_create_binding_pair(Arena* arena, AstNode* name, AstNode* value, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_BINDING_PAIR, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.binding_pair.name = name;
    node->as.binding_pair.value = value;
    
    return node;
}

/**
 * @brief Create a cond clause node
 * 
 * @param arena Arena allocator
 * @param test Test expression
 * @param result Result expression
 * @param line Line number
 * @param column Column number
 * @return A new cond clause node, or NULL on failure
 */
AstNode* ast_create_cond_clause(Arena* arena, AstNode* test, AstNode* result, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_COND_CLAUSE, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.cond_clause.test = test;
    node->as.cond_clause.result = result;
    
    return node;
}

/**
 * @brief Create a case clause node
 * 
 * @param arena Arena allocator
 * @param datum Datum expression
 * @param expr Expression to evaluate
 * @param line Line number
 * @param column Column number
 * @return A new case clause node, or NULL on failure
 */
AstNode* ast_create_case_clause(Arena* arena, AstNode* datum, AstNode* expr, size_t line, size_t column) {
    AstNode* node = ast_create_node(arena, AST_CASE_CLAUSE, line, column);
    if (!node) {
        return NULL;
    }
    
    node->as.case_clause.datum = datum;
    node->as.case_clause.expr = expr;
    
    return node;
}
