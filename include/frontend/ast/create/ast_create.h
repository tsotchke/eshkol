/**
 * @file ast_create.h
 * @brief Functions for creating AST nodes
 */

#ifndef ESHKOL_AST_CREATE_H
#define ESHKOL_AST_CREATE_H

#include "frontend/ast/core/ast_core.h"

/**
 * @brief Create a number literal node
 * 
 * @param arena Arena allocator
 * @param value Number value
 * @param line Line number
 * @param column Column number
 * @return A new number literal node, or NULL on failure
 */
AstNode* ast_create_number(Arena* arena, double value, size_t line, size_t column);

/**
 * @brief Create a boolean literal node
 * 
 * @param arena Arena allocator
 * @param value Boolean value
 * @param line Line number
 * @param column Column number
 * @return A new boolean literal node, or NULL on failure
 */
AstNode* ast_create_boolean(Arena* arena, bool value, size_t line, size_t column);

/**
 * @brief Create a character literal node
 * 
 * @param arena Arena allocator
 * @param value Character value
 * @param line Line number
 * @param column Column number
 * @return A new character literal node, or NULL on failure
 */
AstNode* ast_create_character(Arena* arena, char value, size_t line, size_t column);

/**
 * @brief Create a string literal node
 * 
 * @param arena Arena allocator
 * @param value String value
 * @param line Line number
 * @param column Column number
 * @return A new string literal node, or NULL on failure
 */
AstNode* ast_create_string(Arena* arena, StringId value, size_t line, size_t column);

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
AstNode* ast_create_vector(Arena* arena, AstNode** elements, size_t count, size_t line, size_t column);

/**
 * @brief Create a nil literal node
 * 
 * @param arena Arena allocator
 * @param line Line number
 * @param column Column number
 * @return A new nil literal node, or NULL on failure
 */
AstNode* ast_create_nil(Arena* arena, size_t line, size_t column);

/**
 * @brief Create an identifier node
 * 
 * @param arena Arena allocator
 * @param name Identifier name
 * @param line Line number
 * @param column Column number
 * @return A new identifier node, or NULL on failure
 */
AstNode* ast_create_identifier(Arena* arena, StringId name, size_t line, size_t column);

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
AstNode* ast_create_define(Arena* arena, AstNode* name, AstNode* value, size_t line, size_t column);

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
AstNode* ast_create_lambda(Arena* arena, Parameter** params, size_t param_count, Type* return_type, AstNode* body, size_t line, size_t column);

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
AstNode* ast_create_if(Arena* arena, AstNode* condition, AstNode* then_branch, AstNode* else_branch, size_t line, size_t column);

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
AstNode* ast_create_begin(Arena* arena, AstNode** exprs, size_t expr_count, size_t line, size_t column);

/**
 * @brief Create a quote node
 * 
 * @param arena Arena allocator
 * @param expr Quoted expression
 * @param line Line number
 * @param column Column number
 * @return A new quote node, or NULL on failure
 */
AstNode* ast_create_quote(Arena* arena, AstNode* expr, size_t line, size_t column);

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
AstNode* ast_create_set(Arena* arena, AstNode* name, AstNode* value, size_t line, size_t column);

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
AstNode* ast_create_let(Arena* arena, AstNode** bindings, AstNode** binding_nodes, size_t binding_count, AstNode* body, size_t line, size_t column);

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
AstNode* ast_create_letrec(Arena* arena, AstNode** bindings, AstNode** binding_nodes, size_t binding_count, AstNode* body, size_t line, size_t column);

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
AstNode* ast_create_letstar(Arena* arena, AstNode** bindings, AstNode** binding_nodes, size_t binding_count, AstNode* body, size_t line, size_t column);

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
AstNode* ast_create_and(Arena* arena, AstNode** exprs, size_t expr_count, size_t line, size_t column);

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
AstNode* ast_create_or(Arena* arena, AstNode** exprs, size_t expr_count, size_t line, size_t column);

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
AstNode* ast_create_cond(Arena* arena, AstNode** clauses, size_t clause_count, size_t line, size_t column);

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
AstNode* ast_create_case(Arena* arena, AstNode* key, AstNode** clauses, size_t clause_count, size_t line, size_t column);

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
AstNode* ast_create_do(Arena* arena, AstNode** bindings, AstNode** steps, size_t binding_count, AstNode* test, AstNode** result, size_t result_count, AstNode** body, size_t body_count, size_t line, size_t column);

/**
 * @brief Create a delay node
 * 
 * @param arena Arena allocator
 * @param expr Delayed expression
 * @param line Line number
 * @param column Column number
 * @return A new delay node, or NULL on failure
 */
AstNode* ast_create_delay(Arena* arena, AstNode* expr, size_t line, size_t column);

/**
 * @brief Create a quasiquote node
 * 
 * @param arena Arena allocator
 * @param expr Quasiquoted expression
 * @param line Line number
 * @param column Column number
 * @return A new quasiquote node, or NULL on failure
 */
AstNode* ast_create_quasiquote(Arena* arena, AstNode* expr, size_t line, size_t column);

/**
 * @brief Create an unquote node
 * 
 * @param arena Arena allocator
 * @param expr Unquoted expression
 * @param line Line number
 * @param column Column number
 * @return A new unquote node, or NULL on failure
 */
AstNode* ast_create_unquote(Arena* arena, AstNode* expr, size_t line, size_t column);

/**
 * @brief Create an unquote-splicing node
 * 
 * @param arena Arena allocator
 * @param expr Unquote-spliced expression
 * @param line Line number
 * @param column Column number
 * @return A new unquote-splicing node, or NULL on failure
 */
AstNode* ast_create_unquote_splicing(Arena* arena, AstNode* expr, size_t line, size_t column);

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
AstNode* ast_create_call(Arena* arena, AstNode* callee, AstNode** args, size_t arg_count, size_t line, size_t column);

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
AstNode* ast_create_sequence(Arena* arena, AstNode** exprs, size_t expr_count, size_t line, size_t column);

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
AstNode* ast_create_function_def(Arena* arena, AstNode* name, Parameter** params, AstNode** param_nodes, size_t param_count, Type* return_type, AstNode* body, size_t line, size_t column);

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
AstNode* ast_create_variable_def(Arena* arena, AstNode* name, AstNode* value, size_t line, size_t column);

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
AstNode* ast_create_type_declaration(Arena* arena, StringId function_name, Type* type, size_t line, size_t column);

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
AstNode* ast_create_program(Arena* arena, AstNode** exprs, size_t expr_count, size_t line, size_t column);

/**
 * @brief Create an error node
 * 
 * @param arena Arena allocator
 * @param message Error message
 * @param line Line number
 * @param column Column number
 * @return A new error node, or NULL on failure
 */
AstNode* ast_create_error(Arena* arena, StringId message, size_t line, size_t column);

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
AstNode* ast_create_binding_pair(Arena* arena, AstNode* name, AstNode* value, size_t line, size_t column);

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
AstNode* ast_create_cond_clause(Arena* arena, AstNode* test, AstNode* result, size_t line, size_t column);

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
AstNode* ast_create_case_clause(Arena* arena, AstNode* datum, AstNode* expr, size_t line, size_t column);

#endif /* ESHKOL_AST_CREATE_H */
