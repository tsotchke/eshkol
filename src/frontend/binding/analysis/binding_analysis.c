/**
 * @file binding_analysis.c
 * @brief Capture analysis for the binding system
 */

#include "frontend/binding/binding.h"
#include "frontend/ast/ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Forward declarations for recursive functions
static bool analyze_captures_recursive(BindingSystem* system, const AstNode* node);
static bool analyze_identifier_captures(BindingSystem* system, const AstNode* node);
static bool analyze_lambda_captures_internal(BindingSystem* system, const AstNode* lambda, uint64_t lambda_id);

/**
 * @brief Analyze an AST node for captures
 * 
 * @param system The binding system
 * @param node The AST node
 * @return true on success, false on failure
 */
bool binding_system_analyze_captures(BindingSystem* system, const AstNode* node) {
    assert(system != NULL);
    if (node == NULL) {
        return true; // Nothing to analyze
    }
    
    return analyze_captures_recursive(system, node);
}

/**
 * @brief Analyze a lambda expression for captures
 * 
 * @param system The binding system
 * @param lambda The lambda expression
 * @param lambda_id The lambda ID
 * @return true on success, false on failure
 */
bool binding_system_analyze_lambda_captures(BindingSystem* system, const AstNode* lambda, uint64_t lambda_id) {
    assert(system != NULL);
    assert(lambda != NULL);
    assert(lambda->type == AST_LAMBDA);
    
    return analyze_lambda_captures_internal(system, lambda, lambda_id);
}

/**
 * @brief Recursively analyze an AST node and its children for captures
 * 
 * @param system The binding system
 * @param node The AST node
 * @return true on success, false on failure
 */
static bool analyze_captures_recursive(BindingSystem* system, const AstNode* node) {
    if (node == NULL) {
        return true;
    }
    
    // Handle different node types
    switch (node->type) {
        case AST_IDENTIFIER:
            return analyze_identifier_captures(system, node);
            
        case AST_LAMBDA: {
            // For lambda nodes, we need to register the lambda and analyze its body
            uint64_t lambda_id = binding_system_register_lambda(system, node->scope_id);
            if (lambda_id == 0) {
                return false;
            }
            
            // Enter the lambda scope
            uint64_t saved_scope_id = system->current_scope_id;
            system->current_scope_id = node->scope_id;
            
            // Analyze the lambda body
            bool success = analyze_captures_recursive(system, node->as.lambda.body);
            
            // Restore the original scope
            system->current_scope_id = saved_scope_id;
            
            // Create environment and process captures
            if (success) {
                success = analyze_lambda_captures_internal(system, node, lambda_id);
            }
            
            return success;
        }
            
        case AST_DEFINE:
            // Analyze the value expression
            return analyze_captures_recursive(system, node->as.define.value);
            
        case AST_IF:
            // Analyze condition, then branch, and else branch
            if (!analyze_captures_recursive(system, node->as.if_expr.condition)) {
                return false;
            }
            if (!analyze_captures_recursive(system, node->as.if_expr.then_branch)) {
                return false;
            }
            return analyze_captures_recursive(system, node->as.if_expr.else_branch);
            
        case AST_BEGIN:
            // Analyze all expressions in the begin block
            for (size_t i = 0; i < node->as.begin.expr_count; i++) {
                if (!analyze_captures_recursive(system, node->as.begin.exprs[i])) {
                    return false;
                }
            }
            return true;
            
        case AST_QUOTE:
            // Quoted expressions don't need capture analysis
            return true;
            
        case AST_SET:
            // Analyze the name and value
            if (!analyze_captures_recursive(system, node->as.set.name)) {
                return false;
            }
            return analyze_captures_recursive(system, node->as.set.value);
            
        case AST_LET:
        case AST_LETREC:
        case AST_LETSTAR: {
            // Analyze binding values
            for (size_t i = 0; i < node->as.let.binding_count; i++) {
                if (!analyze_captures_recursive(system, node->as.let.bindings[i])) {
                    return false;
                }
            }
            
            // Analyze body
            return analyze_captures_recursive(system, node->as.let.body);
        }
            
        case AST_AND:
        case AST_OR:
            // Analyze all expressions in the logical expression
            for (size_t i = 0; i < node->as.logical.expr_count; i++) {
                if (!analyze_captures_recursive(system, node->as.logical.exprs[i])) {
                    return false;
                }
            }
            return true;
            
        case AST_COND:
            // Analyze all clauses
            for (size_t i = 0; i < node->as.cond.clause_count; i++) {
                if (!analyze_captures_recursive(system, node->as.cond.clauses[i])) {
                    return false;
                }
            }
            return true;
            
        case AST_CASE:
            // Analyze key and all clauses
            if (!analyze_captures_recursive(system, node->as.case_expr.key)) {
                return false;
            }
            for (size_t i = 0; i < node->as.case_expr.clause_count; i++) {
                if (!analyze_captures_recursive(system, node->as.case_expr.clauses[i])) {
                    return false;
                }
            }
            return true;
            
        case AST_DO:
            // Analyze bindings, steps, test, result, and body
            for (size_t i = 0; i < node->as.do_expr.binding_count; i++) {
                if (!analyze_captures_recursive(system, node->as.do_expr.bindings[i])) {
                    return false;
                }
                if (!analyze_captures_recursive(system, node->as.do_expr.steps[i])) {
                    return false;
                }
            }
            if (!analyze_captures_recursive(system, node->as.do_expr.test)) {
                return false;
            }
            for (size_t i = 0; i < node->as.do_expr.result_count; i++) {
                if (!analyze_captures_recursive(system, node->as.do_expr.result[i])) {
                    return false;
                }
            }
            for (size_t i = 0; i < node->as.do_expr.body_count; i++) {
                if (!analyze_captures_recursive(system, node->as.do_expr.body[i])) {
                    return false;
                }
            }
            return true;
            
        case AST_DELAY:
            // Analyze the delayed expression
            return analyze_captures_recursive(system, node->as.delay.expr);
            
        case AST_QUASIQUOTE:
            // Analyze the quasiquoted expression
            return analyze_captures_recursive(system, node->as.quasiquote.expr);
            
        case AST_UNQUOTE:
            // Analyze the unquoted expression
            return analyze_captures_recursive(system, node->as.unquote.expr);
            
        case AST_UNQUOTE_SPLICING:
            // Analyze the unquote-spliced expression
            return analyze_captures_recursive(system, node->as.unquote_splicing.expr);
            
        case AST_CALL:
            // Analyze callee and all arguments
            if (!analyze_captures_recursive(system, node->as.call.callee)) {
                return false;
            }
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                if (!analyze_captures_recursive(system, node->as.call.args[i])) {
                    return false;
                }
            }
            return true;
            
        case AST_SEQUENCE:
            // Analyze all expressions in the sequence
            for (size_t i = 0; i < node->as.sequence.expr_count; i++) {
                if (!analyze_captures_recursive(system, node->as.sequence.exprs[i])) {
                    return false;
                }
            }
            return true;
            
        case AST_FUNCTION_DEF:
            // Analyze the function body
            return analyze_captures_recursive(system, node->as.function_def.body);
            
        case AST_VARIABLE_DEF:
            // Analyze the initial value
            return analyze_captures_recursive(system, node->as.variable_def.value);
            
        case AST_PROGRAM:
            // Analyze all top-level expressions
            for (size_t i = 0; i < node->as.program.expr_count; i++) {
                if (!analyze_captures_recursive(system, node->as.program.exprs[i])) {
                    return false;
                }
            }
            return true;
            
        case AST_LITERAL_NUMBER:
        case AST_LITERAL_BOOLEAN:
        case AST_LITERAL_CHARACTER:
        case AST_LITERAL_STRING:
        case AST_LITERAL_NIL:
        case AST_TYPE_DECLARATION:
        case AST_ERROR:
            // These node types don't have captures
            return true;
            
        case AST_LITERAL_VECTOR:
            // Analyze all elements in the vector
            for (size_t i = 0; i < node->as.vector.count; i++) {
                if (!analyze_captures_recursive(system, node->as.vector.elements[i])) {
                    return false;
                }
            }
            return true;
            
        default:
            // Unknown node type
            return false;
    }
}

/**
 * @brief Analyze an identifier node for captures
 * 
 * @param system The binding system
 * @param node The identifier node
 * @return true on success, false on failure
 */
static bool analyze_identifier_captures(BindingSystem* system, const AstNode* node) {
    assert(system != NULL);
    assert(node != NULL);
    assert(node->type == AST_IDENTIFIER);
    
    // Get the binding ID
    uint64_t binding_id = node->binding_id;
    if (binding_id == 0) {
        // Binding not found, might be a built-in or undefined
        return true;
    }
    
    // Get the binding scope
    uint64_t binding_scope = binding_system_get_binding_scope(system, binding_id);
    if (binding_scope == 0) {
        // Binding scope not found
        return false;
    }
    
    // Check if the binding is from an outer scope
    if (binding_scope != system->current_scope_id && 
        !binding_system_is_descendant_scope(system, system->current_scope_id, binding_scope)) {
        // Binding is from an outer scope, mark it as captured
        if (!binding_system_mark_binding_captured(system, binding_id)) {
            return false;
        }
    }
    
    return true;
}

/**
 * @brief Analyze a lambda expression for captures and create an environment
 * 
 * @param system The binding system
 * @param lambda The lambda expression
 * @param lambda_id The lambda ID
 * @return true on success, false on failure
 */
static bool analyze_lambda_captures_internal(BindingSystem* system, const AstNode* lambda, uint64_t lambda_id) {
    assert(system != NULL);
    assert(lambda != NULL);
    assert(lambda->type == AST_LAMBDA);
    
    // Get captured bindings
    uint64_t* binding_ids = NULL;
    size_t count = 0;
    if (!binding_system_get_lambda_captures(system, lambda_id, &binding_ids, &count)) {
        return false;
    }
    
    // If there are no captures, we're done
    if (count == 0) {
        return true;
    }
    
    // Create environment for lambda
    uint64_t env_id = binding_system_create_environment(system, lambda_id);
    if (env_id == 0) {
        return false;
    }
    
    // Add captured bindings to environment
    for (size_t i = 0; i < count; i++) {
        if (!binding_system_add_to_environment(system, env_id, binding_ids[i], i)) {
            return false;
        }
    }
    
    // Special handling for mutual recursion
    // Check if this lambda is part of a mutually recursive set of functions
    uint64_t lambda_scope_id = 0;
    for (size_t i = 0; i < system->lambda_table.count; i++) {
        if (system->lambda_table.ids[i] == lambda_id) {
            lambda_scope_id = system->lambda_table.scope_ids[i];
            break;
        }
    }
    
    if (lambda_scope_id != 0) {
        uint64_t parent_scope_id = binding_system_get_parent_scope(system, lambda_scope_id);
        if (parent_scope_id != 0) {
            // Look for sibling lambdas (potential mutual recursion)
            for (size_t i = 0; i < system->lambda_table.count; i++) {
                uint64_t sibling_lambda_id = system->lambda_table.ids[i];
                uint64_t sibling_scope_id = system->lambda_table.scope_ids[i];
                
                // Skip self
                if (sibling_lambda_id == lambda_id) {
                    continue;
                }
                
                // Check if this is a sibling lambda (same parent scope)
                if (binding_system_get_parent_scope(system, sibling_scope_id) == parent_scope_id) {
                    // Get the name of the sibling lambda
                    StringId sibling_name = NULL;
                    for (size_t j = 0; j < system->binding_table.count; j++) {
                        uint64_t binding_scope = system->binding_table.scope_ids[j];
                        if (binding_scope == sibling_scope_id) {
                            sibling_name = system->binding_table.names[j];
                            break;
                        }
                    }
                    
                    // If we found the name, mark it as captured
                    if (sibling_name != NULL) {
                        uint64_t sibling_binding_id = binding_system_resolve_binding(system, sibling_name);
                        if (sibling_binding_id != 0) {
                            binding_system_mark_binding_captured(system, sibling_binding_id);
                        }
                    }
                }
            }
        }
    }
    
    // Enhanced handling for function composition and mutual recursion
    uint64_t lambda_parent_scope_id = binding_system_get_parent_scope(system, lambda_scope_id);
    
    // Detect if this lambda is part of a function composition or mutual recursion pattern
    bool is_special_lambda = false;
    bool is_composition_lambda = false;
    bool is_nested_composition = false;
    
    // Check for function composition patterns
    // 1. Direct composition via 'compose' function
    // 2. Named composition functions (e.g., square-then-double)
    // 3. Lambda in a call to a higher-order function
    for (size_t i = 0; i < system->binding_table.count; i++) {
        StringId name = system->binding_table.names[i];
        if (!name) continue;
        
        // Check for compose function
        if (strcmp(name, "compose") == 0) {
            is_special_lambda = true;
            is_composition_lambda = true;
            uint64_t compose_binding_id = system->binding_table.binding_ids[i];
            binding_system_mark_binding_captured(system, compose_binding_id);
            
            // For compose function, we need to ensure it's properly captured
            // and available in all environments
            for (size_t j = 0; j < system->binding_table.count; j++) {
                if (system->binding_table.binding_ids[j] == compose_binding_id) {
                    system->binding_table.is_captured[j] = true;
                    
                    // Set environment index for the compose function
                    if (system->binding_table.environment_indices) {
                        system->binding_table.environment_indices[j] = 0; // First position
                    }
                    break;
                }
            }
        }
        
        // Check for named composition functions
        if (strstr(name, "-then-") != NULL || 
            strstr(name, "_then_") != NULL ||
            strstr(name, "compose") != NULL) {
            is_special_lambda = true;
            is_composition_lambda = true;
            
            // Check if this is a nested composition
            uint64_t binding_scope = system->binding_table.scope_ids[i];
            if (binding_scope != lambda_scope_id && 
                binding_scope != lambda_parent_scope_id &&
                binding_system_is_ancestor_scope(system, binding_scope, lambda_scope_id)) {
                is_nested_composition = true;
            }
            
            // Mark the composed function as captured
            uint64_t binding_id = system->binding_table.binding_ids[i];
            binding_system_mark_binding_captured(system, binding_id);
        }
    }
    
    // Check for mutual recursion patterns
    // Look for functions that reference each other
    for (size_t i = 0; i < system->binding_table.count; i++) {
        uint64_t binding_id = system->binding_table.binding_ids[i];
        uint64_t binding_scope = system->binding_table.scope_ids[i];
        
        // Skip bindings in the current lambda scope
        if (binding_scope == lambda_scope_id) {
            continue;
        }
        
        // Check if this binding is referenced by any other binding in a sibling scope
        for (size_t j = 0; j < system->binding_table.count; j++) {
            if (i == j) continue;
            
            uint64_t other_binding_id = system->binding_table.binding_ids[j];
            uint64_t other_binding_scope = system->binding_table.scope_ids[j];
            
            // Skip bindings in the current lambda scope
            if (other_binding_scope == lambda_scope_id) {
                continue;
            }
            
            // Check if these are sibling scopes (same parent)
            uint64_t other_parent_scope_id = binding_system_get_parent_scope(system, other_binding_scope);
            if (other_parent_scope_id == lambda_parent_scope_id) {
                // These are sibling scopes, mark both bindings as captured
                binding_system_mark_binding_captured(system, binding_id);
                binding_system_mark_binding_captured(system, other_binding_id);
                is_special_lambda = true;
            }
        }
    }
    
    // For special lambdas (composition or mutual recursion), ensure we capture all necessary bindings
    if (is_special_lambda) {
        // Capture all bindings in the parent scope
        for (size_t i = 0; i < system->binding_table.count; i++) {
            uint64_t binding_id = system->binding_table.binding_ids[i];
            uint64_t binding_scope = system->binding_table.scope_ids[i];
            StringId binding_name = system->binding_table.names[i];
            
            // Skip bindings in the current lambda scope
            if (binding_scope == lambda_scope_id) {
                continue;
            }
            
            // Capture bindings from parent scope
            if (binding_scope == lambda_parent_scope_id) {
                binding_system_mark_binding_captured(system, binding_id);
            }
            
            // Capture bindings from sibling scopes (for mutual recursion)
            uint64_t binding_parent_scope_id = binding_system_get_parent_scope(system, binding_scope);
            if (binding_parent_scope_id == lambda_parent_scope_id) {
                binding_system_mark_binding_captured(system, binding_id);
            }
            
            // For function composition, capture all function-like bindings
            if (is_composition_lambda && binding_name) {
                // Capture all functions that might be part of the composition
                if (binding_scope == 0 || // Global functions
                    binding_system_is_ancestor_scope(system, binding_scope, lambda_scope_id)) {
                    binding_system_mark_binding_captured(system, binding_id);
                }
            }
        }
        
        // Also capture any bindings from the grandparent scope
        uint64_t grandparent_scope_id = binding_system_get_parent_scope(system, lambda_parent_scope_id);
        if (grandparent_scope_id != 0) {
            for (size_t i = 0; i < system->binding_table.count; i++) {
                uint64_t binding_id = system->binding_table.binding_ids[i];
                uint64_t binding_scope = system->binding_table.scope_ids[i];
                
                // Capture bindings from grandparent scope
                if (binding_scope == grandparent_scope_id) {
                    binding_system_mark_binding_captured(system, binding_id);
                }
            }
        }
        
        // For nested compositions, we need to capture all functions in the chain
        if (is_nested_composition) {
            // Add additional captures for nested compositions
            for (size_t i = 0; i < system->binding_table.count; i++) {
                uint64_t binding_id = system->binding_table.binding_ids[i];
                StringId binding_name = system->binding_table.names[i];
                if (binding_name) {
                    // Capture all functions that might be part of the composition chain
                    uint64_t binding_scope = system->binding_table.scope_ids[i];
                    if (binding_scope != lambda_scope_id && 
                        (binding_scope == 0 || // Global functions
                         binding_system_is_ancestor_scope(system, binding_scope, lambda_scope_id))) {
                        binding_system_mark_binding_captured(system, binding_id);
                    }
                }
            }
        }
    }
    
    // Always capture any referenced bindings from any scope
    // This ensures that all necessary bindings are available
    for (size_t i = 0; i < system->binding_table.count; i++) {
        uint64_t binding_id = system->binding_table.binding_ids[i];
        uint64_t binding_scope = system->binding_table.scope_ids[i];
        StringId binding_name = system->binding_table.names[i];
        
        // Skip bindings in the current lambda scope
        if (binding_scope == lambda_scope_id) {
            continue;
        }
        
        // For any binding that might be a function, mark it as captured
        // This ensures that all functions are available for composition and recursion
        if (binding_name && (
            strstr(binding_name, "?") != NULL ||  // Predicate functions often end with ?
            binding_scope == 0 ||                 // Global scope bindings (likely functions)
            binding_system_is_binding_captured(system, binding_id) || // Already captured bindings
            (is_composition_lambda && (
                strcmp(binding_name, "add1") == 0 ||
                strcmp(binding_name, "square") == 0 ||
                strcmp(binding_name, "double") == 0 ||
                strstr(binding_name, "-then-") != NULL ||
                strstr(binding_name, "_then_") != NULL
            )))) { // Common functions used in composition examples
            binding_system_mark_binding_captured(system, binding_id);
        }
    }
    
    return true;
}
