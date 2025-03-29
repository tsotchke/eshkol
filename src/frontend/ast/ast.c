/**
 * @file ast.c
 * @brief Implementation of the Abstract Syntax Tree for Eshkol
 */

#include "frontend/ast/ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Create a new AST node
 * 
 * @param arena Arena allocator
 * @param type Node type
 * @param line Line number
 * @param column Column number
 * @return A new AST node, or NULL on failure
 */
AstNode* ast_create_node(Arena* arena, AstNodeType type, size_t line, size_t column) {
    assert(arena != NULL);
    
    AstNode* node = arena_alloc(arena, sizeof(AstNode));
    if (!node) {
        return NULL;
    }
    
    node->type = type;
    node->line = line;
    node->column = column;
    node->type_info = NULL;     // Initialize explicit type to NULL
    node->inferred_type = NULL; // Initialize inferred type to NULL
    node->binding_id = 0;       // Initialize binding ID to 0 (not bound)
    node->scope_id = 0;         // Initialize scope ID to 0 (global scope)
    node->is_tail_position = false; // Initialize tail position flag to false
    node->is_self_tail_call = false; // Initialize self tail call flag to false
    
    return node;
}

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
 * @brief Get the string representation of an AST node type
 * 
 * @param type The AST node type
 * @return The string representation
 */
const char* ast_node_type_to_string(AstNodeType type) {
    switch (type) {
        case AST_LITERAL_NUMBER: return "NUMBER";
        case AST_LITERAL_BOOLEAN: return "BOOLEAN";
        case AST_LITERAL_CHARACTER: return "CHARACTER";
        case AST_LITERAL_STRING: return "STRING";
        case AST_LITERAL_VECTOR: return "VECTOR";
        case AST_LITERAL_NIL: return "NIL";
        case AST_IDENTIFIER: return "IDENTIFIER";
        case AST_DEFINE: return "DEFINE";
        case AST_LAMBDA: return "LAMBDA";
        case AST_IF: return "IF";
        case AST_BEGIN: return "BEGIN";
        case AST_QUOTE: return "QUOTE";
        case AST_SET: return "SET!";
        case AST_LET: return "LET";
        case AST_LETREC: return "LETREC";
        case AST_LETSTAR: return "LET*";
        case AST_AND: return "AND";
        case AST_OR: return "OR";
        case AST_COND: return "COND";
        case AST_CASE: return "CASE";
        case AST_DO: return "DO";
        case AST_DELAY: return "DELAY";
        case AST_QUASIQUOTE: return "QUASIQUOTE";
        case AST_UNQUOTE: return "UNQUOTE";
        case AST_UNQUOTE_SPLICING: return "UNQUOTE-SPLICING";
        case AST_CALL: return "CALL";
        case AST_SEQUENCE: return "SEQUENCE";
        case AST_FUNCTION_DEF: return "FUNCTION-DEF";
        case AST_VARIABLE_DEF: return "VARIABLE-DEF";
        case AST_TYPE_DECLARATION: return "TYPE-DECLARATION";
        case AST_PROGRAM: return "PROGRAM";
        case AST_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Print indentation
 * 
 * @param indent Indentation level
 */
static void print_indent(int indent) {
    for (int i = 0; i < indent; i++) {
        printf("  ");
    }
}

/**
 * @brief Print an AST node
 * 
 * @param node The AST node
 * @param indent Indentation level
 */
/**
 * @brief Mark tail positions in an AST
 * 
 * This function recursively traverses the AST and marks nodes that are in tail position.
 * A node is in tail position if its value is returned directly from the function.
 * 
 * @param node The AST node to mark
 * @param is_tail Whether the node is in tail position
 */
void ast_mark_tail_positions(AstNode* node, bool is_tail) {
    if (!node) {
        return;
    }
    
    // Mark the node's tail position
    node->is_tail_position = is_tail;
    
    // Recursively mark children based on their context
    switch (node->type) {
        case AST_LAMBDA:
            // The body of a lambda is in tail position
            ast_mark_tail_positions(node->as.lambda.body, true);
            break;
            
        case AST_IF:
            // The condition is not in tail position
            ast_mark_tail_positions(node->as.if_expr.condition, false);
            // The branches are in tail position if the if is in tail position
            ast_mark_tail_positions(node->as.if_expr.then_branch, is_tail);
            ast_mark_tail_positions(node->as.if_expr.else_branch, is_tail);
            break;
            
        case AST_BEGIN:
            // Only the last expression in a begin is in tail position
            for (size_t i = 0; i < node->as.begin.expr_count; i++) {
                ast_mark_tail_positions(node->as.begin.exprs[i], 
                                       is_tail && (i == node->as.begin.expr_count - 1));
            }
            break;
            
        case AST_QUOTE:
        case AST_QUASIQUOTE:
        case AST_UNQUOTE:
        case AST_UNQUOTE_SPLICING:
        case AST_DELAY:
            // The quoted expression is not in tail position
            ast_mark_tail_positions(node->as.quote.expr, false);
            break;
            
        case AST_SET:
            // Neither the name nor the value are in tail position
            ast_mark_tail_positions(node->as.set.name, false);
            ast_mark_tail_positions(node->as.set.value, false);
            break;
            
        case AST_LET:
        case AST_LETREC:
        case AST_LETSTAR:
            // Bindings are not in tail position
            for (size_t i = 0; i < node->as.let.binding_count; i++) {
                ast_mark_tail_positions(node->as.let.bindings[i], false);
            }
            // The body is in tail position if the let is in tail position
            ast_mark_tail_positions(node->as.let.body, is_tail);
            break;
            
        case AST_AND:
        case AST_OR:
            // Only the last expression in a logical operation is in tail position
            for (size_t i = 0; i < node->as.logical.expr_count; i++) {
                ast_mark_tail_positions(node->as.logical.exprs[i], 
                                       is_tail && (i == node->as.logical.expr_count - 1));
            }
            break;
            
        case AST_COND:
            // All clauses are in tail position if the cond is in tail position
            for (size_t i = 0; i < node->as.cond.clause_count; i++) {
                ast_mark_tail_positions(node->as.cond.clauses[i], is_tail);
            }
            break;
            
        case AST_CASE:
            // The key is not in tail position
            ast_mark_tail_positions(node->as.case_expr.key, false);
            // All clauses are in tail position if the case is in tail position
            for (size_t i = 0; i < node->as.case_expr.clause_count; i++) {
                ast_mark_tail_positions(node->as.case_expr.clauses[i], is_tail);
            }
            break;
            
        case AST_DO:
            // Bindings and steps are not in tail position
            for (size_t i = 0; i < node->as.do_expr.binding_count; i++) {
                ast_mark_tail_positions(node->as.do_expr.bindings[i], false);
                ast_mark_tail_positions(node->as.do_expr.steps[i], false);
            }
            // The test is not in tail position
            ast_mark_tail_positions(node->as.do_expr.test, false);
            // The result expressions are in tail position if the do is in tail position
            for (size_t i = 0; i < node->as.do_expr.result_count; i++) {
                ast_mark_tail_positions(node->as.do_expr.result[i], 
                                       is_tail && (i == node->as.do_expr.result_count - 1));
            }
            // The body expressions are not in tail position
            for (size_t i = 0; i < node->as.do_expr.body_count; i++) {
                ast_mark_tail_positions(node->as.do_expr.body[i], false);
            }
            break;
            
        case AST_CALL:
            // The callee is not in tail position
            ast_mark_tail_positions(node->as.call.callee, false);
            // The arguments are not in tail position
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                ast_mark_tail_positions(node->as.call.args[i], false);
            }
            
            // Check if this is a self-recursive tail call
            if (is_tail && node->as.call.callee->type == AST_IDENTIFIER) {
                // If we're in a function definition, check if the callee is the same function
                AstNode* current_function = NULL;
                // TODO: Get the current function from the context
                
                if (current_function && current_function->type == AST_FUNCTION_DEF &&
                    current_function->as.function_def.name->type == AST_IDENTIFIER) {
                    const char* callee_name = node->as.call.callee->as.identifier.name;
                    const char* function_name = current_function->as.function_def.name->as.identifier.name;
                    
                    if (strcmp(callee_name, function_name) == 0) {
                        node->is_self_tail_call = true;
                    }
                }
            }
            break;
            
        case AST_SEQUENCE:
            // Only the last expression in a sequence is in tail position
            for (size_t i = 0; i < node->as.sequence.expr_count; i++) {
                ast_mark_tail_positions(node->as.sequence.exprs[i], 
                                       is_tail && (i == node->as.sequence.expr_count - 1));
            }
            break;
            
        case AST_FUNCTION_DEF:
            // The name is not in tail position
            ast_mark_tail_positions(node->as.function_def.name, false);
            // The parameters are not in tail position
            for (size_t i = 0; i < node->as.function_def.param_count; i++) {
                ast_mark_tail_positions(node->as.function_def.param_nodes[i], false);
            }
            // The body is in tail position
            ast_mark_tail_positions(node->as.function_def.body, true);
            break;
            
        case AST_VARIABLE_DEF:
            // The name is not in tail position
            ast_mark_tail_positions(node->as.variable_def.name, false);
            // The value is not in tail position
            ast_mark_tail_positions(node->as.variable_def.value, false);
            break;
            
        case AST_PROGRAM:
            // Top-level expressions are not in tail position
            for (size_t i = 0; i < node->as.program.expr_count; i++) {
                ast_mark_tail_positions(node->as.program.exprs[i], false);
            }
            break;
            
        default:
            // Other node types don't have children to mark
            break;
    }
}

void ast_print(const AstNode* node, int indent) {
    if (!node) {
        print_indent(indent);
        printf("NULL\n");
        return;
    }
    
    print_indent(indent);
    printf("%s", ast_node_type_to_string(node->type));
    
    switch (node->type) {
        case AST_LITERAL_NUMBER:
            printf(" %g", node->as.number.value);
            break;
        case AST_LITERAL_BOOLEAN:
            printf(" %s", node->as.boolean.value ? "#t" : "#f");
            break;
        case AST_LITERAL_CHARACTER:
            printf(" '%c'", node->as.character.value);
            break;
        case AST_LITERAL_STRING:
            printf(" \"%s\"", node->as.string.value);
            break;
        case AST_LITERAL_VECTOR:
            printf(" [%zu]", node->as.vector.count);
            break;
        case AST_IDENTIFIER:
            printf(" %s", node->as.identifier.name);
            break;
        case AST_TYPE_DECLARATION:
            printf(" %s", node->as.type_declaration.function_name);
            break;
        case AST_ERROR:
            printf(" \"%s\"", node->as.error.message);
            break;
        default:
            break;
    }
    
    // Print type information if available
    printf(" (line %zu, column %zu)\n", node->line, node->column);
    
    // Recursively print children
    switch (node->type) {
        case AST_LITERAL_VECTOR:
            for (size_t i = 0; i < node->as.vector.count; i++) {
                ast_print(node->as.vector.elements[i], indent + 1);
            }
            break;
        case AST_DEFINE:
            ast_print(node->as.define.name, indent + 1);
            ast_print(node->as.define.value, indent + 1);
            break;
        case AST_LAMBDA:
            print_indent(indent + 1);
            printf("Parameters:\n");
            for (size_t i = 0; i < node->as.lambda.param_count; i++) {
                ast_print(node->as.lambda.params[i], indent + 2);
            }
            print_indent(indent + 1);
            printf("Body:\n");
            ast_print(node->as.lambda.body, indent + 2);
            break;
        case AST_IF:
            print_indent(indent + 1);
            printf("Condition:\n");
            ast_print(node->as.if_expr.condition, indent + 2);
            print_indent(indent + 1);
            printf("Then:\n");
            ast_print(node->as.if_expr.then_branch, indent + 2);
            print_indent(indent + 1);
            printf("Else:\n");
            ast_print(node->as.if_expr.else_branch, indent + 2);
            break;
        case AST_BEGIN:
            for (size_t i = 0; i < node->as.begin.expr_count; i++) {
                ast_print(node->as.begin.exprs[i], indent + 1);
            }
            break;
        case AST_QUOTE:
        case AST_QUASIQUOTE:
        case AST_UNQUOTE:
        case AST_UNQUOTE_SPLICING:
        case AST_DELAY:
            ast_print(node->as.quote.expr, indent + 1);
            break;
        case AST_SET:
            ast_print(node->as.set.name, indent + 1);
            ast_print(node->as.set.value, indent + 1);
            break;
        case AST_LET:
        case AST_LETREC:
        case AST_LETSTAR:
            print_indent(indent + 1);
            printf("Bindings:\n");
            for (size_t i = 0; i < node->as.let.binding_count; i++) {
                ast_print(node->as.let.bindings[i], indent + 2);
            }
            print_indent(indent + 1);
            printf("Body:\n");
            ast_print(node->as.let.body, indent + 2);
            break;
        case AST_AND:
        case AST_OR:
            for (size_t i = 0; i < node->as.logical.expr_count; i++) {
                ast_print(node->as.logical.exprs[i], indent + 1);
            }
            break;
        case AST_COND:
            for (size_t i = 0; i < node->as.cond.clause_count; i++) {
                ast_print(node->as.cond.clauses[i], indent + 1);
            }
            break;
        case AST_CASE:
            print_indent(indent + 1);
            printf("Key:\n");
            ast_print(node->as.case_expr.key, indent + 2);
            print_indent(indent + 1);
            printf("Clauses:\n");
            for (size_t i = 0; i < node->as.case_expr.clause_count; i++) {
                ast_print(node->as.case_expr.clauses[i], indent + 2);
            }
            break;
        case AST_DO:
            print_indent(indent + 1);
            printf("Bindings:\n");
            for (size_t i = 0; i < node->as.do_expr.binding_count; i++) {
                ast_print(node->as.do_expr.bindings[i], indent + 2);
            }
            print_indent(indent + 1);
            printf("Test:\n");
            ast_print(node->as.do_expr.test, indent + 2);
            print_indent(indent + 1);
            printf("Result:\n");
            for (size_t i = 0; i < node->as.do_expr.result_count; i++) {
                ast_print(node->as.do_expr.result[i], indent + 2);
            }
            print_indent(indent + 1);
            printf("Body:\n");
            for (size_t i = 0; i < node->as.do_expr.body_count; i++) {
                ast_print(node->as.do_expr.body[i], indent + 2);
            }
            break;
        case AST_CALL:
            print_indent(indent + 1);
            printf("Callee:\n");
            ast_print(node->as.call.callee, indent + 2);
            print_indent(indent + 1);
            printf("Arguments:\n");
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                ast_print(node->as.call.args[i], indent + 2);
            }
            break;
        case AST_SEQUENCE:
            for (size_t i = 0; i < node->as.sequence.expr_count; i++) {
                ast_print(node->as.sequence.exprs[i], indent + 1);
            }
            break;
        case AST_FUNCTION_DEF:
            print_indent(indent + 1);
            printf("Name:\n");
            ast_print(node->as.function_def.name, indent + 2);
            print_indent(indent + 1);
            printf("Parameters:\n");
            for (size_t i = 0; i < node->as.function_def.param_count; i++) {
                ast_print(node->as.function_def.params[i], indent + 2);
            }
            print_indent(indent + 1);
            printf("Body:\n");
            ast_print(node->as.function_def.body, indent + 2);
            break;
        case AST_VARIABLE_DEF:
            print_indent(indent + 1);
            printf("Name:\n");
            ast_print(node->as.variable_def.name, indent + 2);
            print_indent(indent + 1);
            printf("Value:\n");
            ast_print(node->as.variable_def.value, indent + 2);
            break;
        case AST_PROGRAM:
            for (size_t i = 0; i < node->as.program.expr_count; i++) {
                ast_print(node->as.program.exprs[i], indent + 1);
            }
            break;
        default:
            break;
    }
}
