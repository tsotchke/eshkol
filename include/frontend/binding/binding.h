/**
 * @file binding.h
 * @brief Binding system for Eshkol
 * 
 * This file defines the binding system for the Eshkol language, which tracks
 * variable bindings and scopes.
 */

#ifndef ESHKOL_BINDING_H
#define ESHKOL_BINDING_H

#include "core/memory.h"
#include "core/string_table.h"
#include "core/diagnostics.h"
#include "frontend/ast/core/ast_core.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief String identifier type
 * 
 * This is a pointer to an interned string in the string table.
 */
typedef const char* StringId;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Binding table structure
 */
typedef struct {
    StringId* names;           /**< Binding names */
    uint64_t* binding_ids;     /**< Binding IDs */
    uint64_t* scope_ids;       /**< Scope IDs where bindings are defined */
    bool* is_mutable;          /**< Whether the binding is mutable */
    bool* is_captured;         /**< Whether the binding is captured by a closure */
    uint64_t* capture_depth;   /**< How many scope levels up the binding is captured from */
    uint64_t* environment_indices; /**< Indices in the environment structure */
    size_t count;              /**< Number of bindings */
    size_t capacity;           /**< Capacity of the binding table */
} BindingTable;

/**
 * @brief Scope type enumeration
 */
typedef enum {
    SCOPE_GLOBAL,              /**< Global scope */
    SCOPE_LOCAL,               /**< Local scope */
    SCOPE_LAMBDA,              /**< Lambda scope */
    SCOPE_LET,                 /**< Let scope */
    SCOPE_LETREC               /**< Letrec scope */
} ScopeType;

/**
 * @brief Scope stack structure
 */
typedef struct {
    uint64_t* ids;             /**< Scope IDs */
    uint64_t* parent_ids;      /**< Parent scope IDs */
    ScopeType* types;          /**< Scope types */
    uint64_t* lambda_ids;      /**< Associated lambda IDs for lambda scopes */
    size_t count;              /**< Number of scopes */
    size_t capacity;           /**< Capacity of the scope stack */
} ScopeStack;

/**
 * @brief Lambda table structure
 */
typedef struct {
    uint64_t* ids;             /**< Lambda IDs */
    uint64_t* scope_ids;       /**< Scope IDs where lambdas are defined */
    uint64_t* environment_ids; /**< Environment IDs for lambdas */
    size_t count;              /**< Number of lambdas */
    size_t capacity;           /**< Capacity of the lambda table */
} LambdaTable;

/**
 * @brief Environment table structure
 */
typedef struct {
    uint64_t* ids;             /**< Environment IDs */
    uint64_t* parent_ids;      /**< Parent environment IDs */
    uint64_t* scope_ids;       /**< Associated scope IDs */
    size_t* binding_counts;    /**< Number of bindings in each environment */
    uint64_t** binding_id_arrays; /**< Arrays of binding IDs for each environment */
    size_t count;              /**< Number of environments */
    size_t capacity;           /**< Capacity of the environment table */
} EnvironmentTable;

/**
  * @brief Definitions table structure
*/
typedef struct {
    uint64_t* ids;             /**< Definition IDs */
    uint64_t* binding_ids;     /**< Binding IDs */
    AstNode** nodes;           /**< Node of the definition*/
    size_t count;              /**< Number of definitions */
    size_t capacity;           /**< Capacity of the definitions table */
} DefinitionsTable;

/** @brief Binding system structure
 */
typedef struct {
    Arena* arena;              /**< Arena allocator */
    DiagnosticContext* diag;   /**< Diagnostic context */
    BindingTable binding_table; /**< Binding table */
    DefinitionsTable def_table; /**< Definitions table */
    ScopeStack scope_stack;    /**< Scope stack */
    LambdaTable lambda_table;  /**< Lambda table */
    EnvironmentTable env_table; /**< Environment table */
    uint64_t current_scope_id; /**< Current scope ID */
    uint64_t next_binding_id;  /**< Next binding ID to assign */
    uint64_t next_scope_id;    /**< Next scope ID to assign */
    uint64_t next_lambda_id;   /**< Next lambda ID to assign */
    uint64_t next_env_id;      /**< Next environment ID to assign */
    uint64_t next_def_id;      /**< Next definition ID to assign */
} BindingSystem;

/**
 * @brief Create a new binding system
 * 
 * @param arena Arena allocator
 * @param diag Diagnostic context
 * @return A new binding system, or NULL on failure
 */
BindingSystem* binding_system_create(Arena* arena, DiagnosticContext* diag);

/**
 * @brief Enter a new scope
 * 
 * @param system The binding system
 * @return The ID of the new scope, or 0 on failure
 */
uint64_t binding_system_enter_scope(BindingSystem* system);

/**
 * @brief Enter a new lambda scope
 * 
 * @param system The binding system
 * @return The ID of the new scope, or 0 on failure
 */
uint64_t binding_system_enter_lambda_scope(BindingSystem* system);

/**
 * @brief Exit the current scope
 * 
 * @param system The binding system
 * @return true on success, false on failure
 */
bool binding_system_exit_scope(BindingSystem* system);

/**
 * @brief Add a binding to the current scope
 * 
 * @param system The binding system
 * @param name The name of the binding
 * @param is_mutable Whether the binding is mutable
 * @return The ID of the new binding, or 0 on failure
 */
uint64_t binding_system_add_binding(BindingSystem* system, StringId name, bool is_mutable);

/**
 * @brief Resolve a binding in the current scope or any parent scope
 * 
 * @param system The binding system
 * @param name The name to resolve
 * @return The ID of the binding, or 0 if not found
 */
uint64_t binding_system_resolve_binding(BindingSystem* system, StringId name);

/**
 * @brief Resolve a binding in a specific scope or any parent scope
 * 
 * @param system The binding system
 * @param name The name to resolve
 * @param scope_id The scope ID to start the search from
 * @return The ID of the binding, or 0 if not found
 */
uint64_t binding_system_resolve_binding_in_scope(BindingSystem* system, StringId name, uint64_t scope_id);

/**
 * @brief Get the name of a binding by ID
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return The name of the binding, or NULL if not found
 */
StringId binding_system_get_binding_name(BindingSystem* system, uint64_t binding_id);

/**
 * @brief Get the scope of a binding by ID
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return The scope ID of the binding, or 0 if not found
 */
uint64_t binding_system_get_binding_scope(BindingSystem* system, uint64_t binding_id);

/**
 * @brief Check if a binding is mutable
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return true if the binding is mutable, false otherwise
 */
bool binding_system_is_binding_mutable(BindingSystem* system, uint64_t binding_id);

/**
 * @brief Check if a binding is in the current scope
 * 
 * @param system The binding system
 * @param name The name to check
 * @return true if the binding is in the current scope, false otherwise
 */
bool binding_system_binding_in_current_scope(BindingSystem* system, StringId name);

/**
 * @brief Check if a scope is a descendant of another scope
 * 
 * @param system The binding system
 * @param scope_id The scope ID to check
 * @param ancestor_id The potential ancestor scope ID
 * @return true if scope_id is a descendant of ancestor_id, false otherwise
 */
bool binding_system_is_descendant_scope(BindingSystem* system, uint64_t scope_id, uint64_t ancestor_id);

/**
 * @brief Check if a scope is an ancestor of another scope
 * 
 * @param system The binding system
 * @param scope_id The scope ID to check
 * @param descendant_id The potential descendant scope ID
 * @return true if scope_id is an ancestor of descendant_id, false otherwise
 */
bool binding_system_is_ancestor_scope(BindingSystem* system, uint64_t scope_id, uint64_t descendant_id);

/**
 * @brief Get the parent scope of a scope
 * 
 * @param system The binding system
 * @param scope_id The scope ID
 * @return The parent scope ID, or 0 if not found or if the scope is the global scope
 */
uint64_t binding_system_get_parent_scope(BindingSystem* system, uint64_t scope_id);

/**
 * @brief Get the current scope ID
 * 
 * @param system The binding system
 * @return The current scope ID
 */
uint64_t binding_system_get_current_scope(BindingSystem* system);

/**
 * @brief Register a lambda expression
 * 
 * @param system The binding system
 * @param scope_id The scope ID where the lambda is defined
 * @return The ID of the new lambda, or 0 on failure
 */
uint64_t binding_system_register_lambda(BindingSystem* system, uint64_t scope_id);

/**
 * @brief Mark a binding as captured
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return true on success, false on failure
 */
bool binding_system_mark_binding_captured(BindingSystem* system, uint64_t binding_id);

/**
 * @brief Get all captured bindings for a lambda
 * 
 * @param system The binding system
 * @param lambda_id The lambda ID
 * @param binding_ids Pointer to an array to store the binding IDs
 * @param count Pointer to store the number of bindings
 * @return true on success, false on failure
 */
bool binding_system_get_lambda_captures(BindingSystem* system, uint64_t lambda_id, 
                                       uint64_t** binding_ids, size_t* count);

/**
 * @brief Create an environment for a lambda
 * 
 * @param system The binding system
 * @param lambda_id The lambda ID
 * @return The ID of the new environment, or 0 on failure
 */
uint64_t binding_system_create_environment(BindingSystem* system, uint64_t lambda_id);

/**
 * @brief Add a binding to an environment
 * 
 * @param system The binding system
 * @param env_id The environment ID
 * @param binding_id The binding ID
 * @param index The index in the environment
 * @return true on success, false on failure
 */
bool binding_system_add_to_environment(BindingSystem* system, uint64_t env_id, 
                                      uint64_t binding_id, size_t index);

/**
 * @brief Get all bindings in an environment
 * 
 * @param system The binding system
 * @param env_id The environment ID
 * @param binding_ids Pointer to store the binding IDs
 * @param count Pointer to store the number of bindings
 * @return true on success, false on failure
 */
bool binding_system_get_environment_bindings(BindingSystem* system, uint64_t env_id,
                                           uint64_t** binding_ids, size_t* count);

/**
 * @brief Check if a binding is captured
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return true if the binding is captured, false otherwise
 */
bool binding_system_is_binding_captured(BindingSystem* system, uint64_t binding_id);

/**
 * @brief Get the environment index of a binding
 * 
 * @param system The binding system
 * @param binding_id The binding ID
 * @return The environment index, or -1 if not found
 */
int binding_system_get_binding_env_index(BindingSystem* system, uint64_t binding_id);

// Forward declaration of AstNode
struct AstNode;

/**
 * @brief Analyze an AST node for captures
 * 
 * @param system The binding system
 * @param node The AST node
 * @return true on success, false on failure
 */
bool binding_system_analyze_captures(BindingSystem* system, const struct AstNode* node);

/**
 * @brief Analyze a lambda expression for captures
 * 
 * @param system The binding system
 * @param lambda The lambda expression
 * @param lambda_id The lambda ID
 * @return true on success, false on failure
 */
bool binding_system_analyze_lambda_captures(BindingSystem* system, 
                                          const struct AstNode* lambda, 
                                          uint64_t lambda_id);

/**
 * @brief Bind a node to the given binding
 * 
 * @param system The binding system
 * @param bind_id The id of the binding
 * @param node The node that the bind defines
 * @return The ID of the new definition, or 0 on failure
 */ 
uint64_t binding_system_register_define(BindingSystem* system, uint64_t bind_id, AstNode* node);


/**
 * @brief Retrieve a node from the given binding
 * 
 * @param system The binding system
 * @param bind_id The id of the binding
 * @return The node of the definition 
 */ 
AstNode* binding_system_get_definition(BindingSystem* system, uint64_t bind_id);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_BINDING_H */
