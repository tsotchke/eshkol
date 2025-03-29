/**
 * @file ast_core.c
 * @brief Core functionality for the Abstract Syntax Tree
 */

#include "frontend/ast/core/ast_core.h"
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
    node->parent = NULL;        // Initialize parent node to NULL
    
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
        case AST_BINDING_PAIR: return "BINDING-PAIR";
        case AST_COND_CLAUSE: return "COND-CLAUSE";
        case AST_CASE_CLAUSE: return "CASE-CLAUSE";
        default: return "UNKNOWN";
    }
}
