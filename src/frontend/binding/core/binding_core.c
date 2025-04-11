/**
 * @file binding_core.c
 * @brief Core implementation of the binding system for Eshkol
 */

#include "frontend/binding/binding.h"
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
    assert(arena != NULL);
    assert(diag != NULL);
    
    // Allocate binding system
    BindingSystem* system = arena_alloc(arena, sizeof(BindingSystem));
    if (!system) {
        return NULL;
    }
    
    // Initialize binding system
    system->arena = arena;
    system->diag = diag;
    
    // Initialize binding table
    system->binding_table.names = NULL;
    system->binding_table.binding_ids = NULL;
    system->binding_table.scope_ids = NULL;
    system->binding_table.is_mutable = NULL;
    system->binding_table.is_captured = NULL;
    system->binding_table.capture_depth = NULL;
    system->binding_table.environment_indices = NULL;
    system->binding_table.count = 0;
    system->binding_table.capacity = 0;
    
    // Initialize scope stack
    system->scope_stack.ids = NULL;
    system->scope_stack.parent_ids = NULL;
    system->scope_stack.types = NULL;
    system->scope_stack.lambda_ids = NULL;
    system->scope_stack.count = 0;
    system->scope_stack.capacity = 0;
    
    // Initialize lambda table
    system->lambda_table.ids = NULL;
    system->lambda_table.scope_ids = NULL;
    system->lambda_table.environment_ids = NULL;
    system->lambda_table.count = 0;
    system->lambda_table.capacity = 0;
    
    // Initialize environment table
    system->env_table.ids = NULL;
    system->env_table.parent_ids = NULL;
    system->env_table.scope_ids = NULL;
    system->env_table.binding_counts = NULL;
    system->env_table.count = 0;
    system->env_table.capacity = 0;
    
    // Initialize definitions tables
    system->def_table.ids = NULL;
    system->def_table.binding_ids = NULL;
    system->def_table.nodes = NULL;
    system->def_table.count = 0;
    system->def_table.capacity = 0;

    // Initialize IDs
    system->current_scope_id = 0;
    system->next_binding_id = 1; // Start at 1, 0 is reserved for "not found"
    system->next_scope_id = 1;   // Start at 1, 0 is reserved for "global scope"
    system->next_lambda_id = 1;  // Start at 1, 0 is reserved for "not found"
    system->next_env_id = 1;     // Start at 1, 0 is reserved for "global environment"
    system->next_def_id = 1;     // Start at 1, 0 is reserved for "not found" 

    // Create global scope
    if (!binding_system_enter_scope(system)) {
        return NULL;
    }
    
    return system;
}

/**
 * @brief Get the current scope ID
 * 
 * @param system The binding system
 * @return The current scope ID
 */
uint64_t binding_system_get_current_scope(BindingSystem* system) {
    assert(system != NULL);
    
    return system->current_scope_id;
}
