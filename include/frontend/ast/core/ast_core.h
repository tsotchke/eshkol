/**
 * @file ast_core.h
 * @brief Core definitions for the Abstract Syntax Tree
 */

#ifndef ESHKOL_AST_CORE_H
#define ESHKOL_AST_CORE_H

#include <stdbool.h>
#include <stddef.h>
#include "core/memory.h"
#include "core/string_table.h"
#include "frontend/ast/parameter.h"

/**
 * @brief AST node types
 */
typedef enum {
    AST_LITERAL_NUMBER,
    AST_LITERAL_BOOLEAN,
    AST_LITERAL_CHARACTER,
    AST_LITERAL_STRING,
    AST_LITERAL_VECTOR,
    AST_LITERAL_NIL,
    AST_IDENTIFIER,
    AST_DEFINE,
    AST_LAMBDA,
    AST_IF,
    AST_BEGIN,
    AST_QUOTE,
    AST_SET,
    AST_LET,
    AST_LETREC,
    AST_LETSTAR,
    AST_AND,
    AST_OR,
    AST_COND,
    AST_CASE,
    AST_DO,
    AST_DELAY,
    AST_QUASIQUOTE,
    AST_UNQUOTE,
    AST_UNQUOTE_SPLICING,
    AST_CALL,
    AST_SEQUENCE,
    AST_FUNCTION_DEF,
    AST_VARIABLE_DEF,
    AST_TYPE_DECLARATION,
    AST_PROGRAM,
    AST_ERROR,
    AST_BINDING_PAIR,
    AST_COND_CLAUSE,
    AST_CASE_CLAUSE
} AstNodeType;

/**
 * @brief Forward declaration of AST node
 */
typedef struct AstNode AstNode;

/**
 * @brief Forward declaration of Type
 */
typedef struct Type Type;

/**
 * @brief AST node structure
 */
struct AstNode {
    AstNodeType type;
    size_t line;
    size_t column;
    Type* type_info;      // Explicit type annotation
    Type* inferred_type;  // Type inferred by the type system
    size_t binding_id;    // ID for binding analysis
    size_t scope_id;      // ID for scope analysis
    bool is_tail_position;  // Whether this node is in tail position
    bool is_self_tail_call; // Whether this is a self-recursive tail call
    struct AstNode* parent; // Parent node in the AST
    
    union {
        struct {
            double value;
        } number;
        
        struct {
            bool value;
        } boolean;
        
        struct {
            char value;
        } character;
        
        struct {
            StringId value;
        } string;
        
        struct {
            AstNode** elements;
            size_t count;
        } vector;
        
        struct {
            StringId name;
        } identifier;
        
        struct {
            AstNode* name;
            AstNode* value;
        } define;
        
        struct {
            Parameter** params;
            size_t param_count;
            Type* return_type;
            AstNode* body;
        } lambda;
        
        struct {
            AstNode* condition;
            AstNode* then_branch;
            AstNode* else_branch;
        } if_expr;
        
        struct {
            AstNode** exprs;
            size_t expr_count;
        } begin;
        
        struct {
            AstNode* expr;
        } quote;
        
        struct {
            AstNode* name;
            AstNode* value;
        } set;
        
        struct {
            AstNode** bindings;
            AstNode** binding_nodes;
            size_t binding_count;
            AstNode* body;
        } let;
        
        struct {
            AstNode** exprs;
            size_t expr_count;
        } logical;
        
        struct {
            AstNode** clauses;
            size_t clause_count;
        } cond;
        
        struct {
            AstNode* key;
            AstNode** clauses;
            size_t clause_count;
        } case_expr;
        
        struct {
            AstNode** bindings;
            AstNode** steps;
            size_t binding_count;
            AstNode* test;
            AstNode** result;
            size_t result_count;
            AstNode** body;
            size_t body_count;
        } do_expr;
        
        struct {
            AstNode* expr;
        } delay;
        
        struct {
            AstNode* expr;
        } quasiquote;
        
        struct {
            AstNode* expr;
        } unquote;
        
        struct {
            AstNode* expr;
        } unquote_splicing;
        
        struct {
            AstNode* callee;
            AstNode** args;
            size_t arg_count;
        } call;
        
        struct {
            AstNode** exprs;
            size_t expr_count;
        } sequence;
        
        struct {
            AstNode* name;
            Parameter** params;
            AstNode** param_nodes;
            size_t param_count;
            Type* return_type;
            AstNode* body;
        } function_def;
        
        struct {
            AstNode* name;
            AstNode* value;
        } variable_def;
        
        struct {
            StringId function_name;
            Type* type;
        } type_declaration;
        
        struct {
            AstNode** exprs;
            size_t expr_count;
        } program;
        
        struct {
            StringId message;
        } error;
        
        struct {
            AstNode* name;
            AstNode* value;
        } binding_pair;
        
        struct {
            AstNode* test;
            AstNode* result;
        } cond_clause;
        
        struct {
            AstNode* datum;
            AstNode* expr;
        } case_clause;
    } as;
};

/**
 * @brief Create a new AST node
 * 
 * @param arena Arena allocator
 * @param type Node type
 * @param line Line number
 * @param column Column number
 * @return A new AST node, or NULL on failure
 */
AstNode* ast_create_node(Arena* arena, AstNodeType type, size_t line, size_t column);

/**
 * @brief Get the string representation of an AST node type
 * 
 * @param type The AST node type
 * @return The string representation
 */
const char* ast_node_type_to_string(AstNodeType type);

#endif /* ESHKOL_AST_CORE_H */
