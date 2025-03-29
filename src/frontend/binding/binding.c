/**
 * @file binding.c
 * @brief Implementation of the binding system for Eshkol
 */

#include "frontend/binding/binding.h"
#include "frontend/ast/ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Create a new binding system
 * 
 * @param arena Arena allocator
 * @param diag Diagnostic context
 * @return A new binding system, or NULL on failure
 */
BindingSystem* binding_system_create(Arena* arena, DiagnosticContext* diag) {
    BindingSystem* system = arena_alloc(arena, sizeof(BindingSystem));
    if (!system) {
        return NULL;
    }
    
    // Initialize the binding system
    system->arena = arena;
    system->diag = diag;
    system->binding_table.names = NULL;
    system->binding_table.binding_ids = NULL;
    system->binding_table.scope_ids = NULL;
    system->binding_table.is_mutable = NULL;
    system->binding_table.is_captured = NULL;
    system->binding_table.capture_depth = NULL;
    system->binding_table.environment_indices = NULL;
    system->binding_table.count = 0;
    system->binding_table.capacity = 0;
    system->scope_stack.ids = NULL;
    system->scope_stack.parent_ids = NULL;
    system->scope_stack.types = NULL;
    system->scope_stack.lambda_ids = NULL;
    system->scope_stack.count = 0;
    system->scope_stack.capacity = 0;
    system->lambda_table.ids = NULL;
    system->lambda_table.scope_ids = NULL;
    system->lambda_table.environment_ids = NULL;
    system->lambda_table.count = 0;
    system->lambda_table.capacity = 0;
    system->env_table.ids = NULL;
    system->env_table.parent_ids = NULL;
    system->env_table.scope_ids = NULL;
    system->env_table.binding_counts = NULL;
    system->env_table.count = 0;
    system->env_table.capacity = 0;
    system->current_scope_id = 0;
    system->next_binding_id = 1; // Start at 1, 0 is reserved for "not found"
    system->next_scope_id = 1;   // Start at 1, 0 is reserved for global scope
    system->next_lambda_id = 1;  // Start at 1, 0 is reserved for "not found"
    system->next_env_id = 1;     // Start at 1, 0 is reserved for "not found"
    
    // Create global scope
    uint64_t global_scope_id = binding_system_enter_scope(system);
    if (global_scope_id == 0) {
        return NULL;
    }
    
    // Set the global scope type
    if (system->scope_stack.types) {
        system->scope_stack.types[0] = SCOPE_GLOBAL;
    }
    
    return system;
}
