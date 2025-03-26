/**
 * @file context.c
 * @brief Implementation of the type inference context
 */

#include "frontend/type_inference/context.h"
#include "core/memory.h"
#include "core/type.h"
#include "core/type_creation.h"
#include "core/type_comparison.h"
#include "core/type_conversion.h"
#include "frontend/ast/ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

/**
 * @brief Type inference context structure
 */
struct TypeInferenceContext {
    Arena* arena;                // Arena for allocations 
    DiagnosticContext* diagnostics; // Diagnostic context for error reporting 
    AstNode** nodes;             // Array of AST nodes 
    Type** inferred_types;       // Array of inferred types 
    size_t capacity;             // Capacity of the arrays 
    size_t count;                // Number of nodes 
    
    // Enhanced type system support
    AstNode** explicit_nodes;    // Array of explicitly typed AST nodes 
    Type** explicit_types;       // Array of explicit types 
    size_t explicit_capacity;    // Capacity of the explicit arrays 
    size_t explicit_count;       // Number of explicitly typed nodes 
    
    StringId* function_names;    // Array of function names 
    Type** function_signatures;  // Array of function signatures 
    size_t signature_capacity;   // Capacity of the signature arrays 
    size_t signature_count;      // Number of function signatures 
};

/**
 * @brief Create a type inference context
 */
TypeInferenceContext* type_inference_context_create(Arena* arena, DiagnosticContext* diagnostics) {
    assert(arena != NULL);
    
    // Allocate context
    TypeInferenceContext* context = arena_alloc(arena, sizeof(TypeInferenceContext));
    if (!context) return NULL;
    
    // Initialize context
    context->arena = arena;
    context->diagnostics = diagnostics;
    context->capacity = 1024;
    context->count = 0;
    context->explicit_capacity = 256;
    context->explicit_count = 0;
    context->signature_capacity = 128;
    context->signature_count = 0;
    
    // Allocate arrays
    context->nodes = arena_alloc(arena, context->capacity * sizeof(AstNode*));
    if (!context->nodes) return NULL;
    
    context->inferred_types = arena_alloc(arena, context->capacity * sizeof(Type*));
    if (!context->inferred_types) return NULL;
    
    context->explicit_nodes = arena_alloc(arena, context->explicit_capacity * sizeof(AstNode*));
    if (!context->explicit_nodes) return NULL;
    
    context->explicit_types = arena_alloc(arena, context->explicit_capacity * sizeof(Type*));
    if (!context->explicit_types) return NULL;
    
    context->function_names = arena_alloc(arena, context->signature_capacity * sizeof(StringId));
    if (!context->function_names) return NULL;
    
    context->function_signatures = arena_alloc(arena, context->signature_capacity * sizeof(Type*));
    if (!context->function_signatures) return NULL;
    
    return context;
}

/**
 * @brief Add a node to the context with its inferred type
 */
bool type_inference_add_node(TypeInferenceContext* context, AstNode* node, Type* type) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Check if the node is already in the context
    for (size_t i = 0; i < context->count; i++) {
        if (context->nodes[i] == node) {
            // Update the type
            context->inferred_types[i] = type;
            node->inferred_type = type;  // Also store directly in the node
            return true;
        }
    }
    
    // Check if we need to resize the arrays
    if (context->count >= context->capacity) {
        size_t new_capacity = context->capacity * 2;
        
        // Allocate new arrays
        AstNode** new_nodes = arena_alloc(context->arena, new_capacity * sizeof(AstNode*));
        if (!new_nodes) return false;
        
        Type** new_types = arena_alloc(context->arena, new_capacity * sizeof(Type*));
        if (!new_types) return false;
        
        // Copy old arrays to new arrays
        for (size_t i = 0; i < context->count; i++) {
            new_nodes[i] = context->nodes[i];
            new_types[i] = context->inferred_types[i];
        }
        
        // Update context
        context->nodes = new_nodes;
        context->inferred_types = new_types;
        context->capacity = new_capacity;
    }
    
    // Add the node
    context->nodes[context->count] = node;
    context->inferred_types[context->count] = type;
    node->inferred_type = type;  // Also store directly in the node
    context->count++;
    
    return true;
}

/**
 * @brief Set an explicit type for an AST node
 */
void type_inference_set_explicit_type(TypeInferenceContext* context, AstNode* node, Type* type) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Check if the node is already in the explicit context
    for (size_t i = 0; i < context->explicit_count; i++) {
        if (context->explicit_nodes[i] == node) {
            // Update the type
            context->explicit_types[i] = type;
            node->type_info = type;  // Also store directly in the node
            return;
        }
    }
    
    // Check if we need to resize the arrays
    if (context->explicit_count >= context->explicit_capacity) {
        size_t new_capacity = context->explicit_capacity * 2;
        
        // Allocate new arrays
        AstNode** new_nodes = arena_alloc(context->arena, new_capacity * sizeof(AstNode*));
        if (!new_nodes) return;
        
        Type** new_types = arena_alloc(context->arena, new_capacity * sizeof(Type*));
        if (!new_types) return;
        
        // Copy old arrays to new arrays
        for (size_t i = 0; i < context->explicit_count; i++) {
            new_nodes[i] = context->explicit_nodes[i];
            new_types[i] = context->explicit_types[i];
        }
        
        // Update context
        context->explicit_nodes = new_nodes;
        context->explicit_types = new_types;
        context->explicit_capacity = new_capacity;
    }
    
    // Add the node
    context->explicit_nodes[context->explicit_count] = node;
    context->explicit_types[context->explicit_count] = type;
    node->type_info = type;  // Also store directly in the node
    context->explicit_count++;
}

/**
 * @brief Set a function signature
 */
void type_inference_set_function_signature(TypeInferenceContext* context, StringId function_name, Type* type) {
    assert(context != NULL);
    assert(function_name != NULL);
    
    // Check if the function is already in the context
    for (size_t i = 0; i < context->signature_count; i++) {
        if (context->function_names[i] == function_name) {
            // Update the signature
            context->function_signatures[i] = type;
            return;
        }
    }
    
    // Check if we need to resize the arrays
    if (context->signature_count >= context->signature_capacity) {
        size_t new_capacity = context->signature_capacity * 2;
        
        // Allocate new arrays
        StringId* new_names = arena_alloc(context->arena, new_capacity * sizeof(StringId));
        if (!new_names) return;
        
        Type** new_signatures = arena_alloc(context->arena, new_capacity * sizeof(Type*));
        if (!new_signatures) return;
        
        // Copy old arrays to new arrays
        for (size_t i = 0; i < context->signature_count; i++) {
            new_names[i] = context->function_names[i];
            new_signatures[i] = context->function_signatures[i];
        }
        
        // Update context
        context->function_names = new_names;
        context->function_signatures = new_signatures;
        context->signature_capacity = new_capacity;
    }
    
    // Add the function
    context->function_names[context->signature_count] = function_name;
    context->function_signatures[context->signature_count] = type;
    context->signature_count++;
}

/**
 * @brief Get the inferred type for an AST node
 */
Type* type_inference_get_type(TypeInferenceContext* context, const AstNode* node) {
    assert(context != NULL);
    
    // Check if node is NULL
    if (node == NULL) {
        return NULL;
    }
    
    // Check if the node is in the context
    for (size_t i = 0; i < context->count; i++) {
        if (context->nodes[i] == node) {
            return context->inferred_types[i];
        }
    }
    
    // Check if the node has an inferred type stored directly
    if (node->inferred_type) {
        return node->inferred_type;
    }
    
    return NULL;
}

/**
 * @brief Get the explicit type for an AST node
 */
Type* type_inference_get_explicit_type(TypeInferenceContext* context, const AstNode* node) {
    assert(context != NULL);
    
    // Check if node is NULL
    if (node == NULL) {
        return NULL;
    }
    
    // Check if the node is in the explicit context
    for (size_t i = 0; i < context->explicit_count; i++) {
        if (context->explicit_nodes[i] == node) {
            return context->explicit_types[i];
        }
    }
    
    // Check if the node has an explicit type stored directly
    if (node->type_info) {
        return node->type_info;
    }
    
    return NULL;
}

/**
 * @brief Get the function signature for a function name
 */
Type* type_inference_get_function_signature(TypeInferenceContext* context, StringId function_name) {
    assert(context != NULL);
    
    // Check if function_name is NULL
    if (function_name == NULL) {
        return NULL;
    }
    
    // Check if the function is in the context
    for (size_t i = 0; i < context->signature_count; i++) {
        if (context->function_names[i] == function_name) {
            return context->function_signatures[i];
        }
    }
    
    return NULL;
}

/**
 * @brief Get the arena from the context
 */
Arena* type_inference_get_arena(TypeInferenceContext* context) {
    assert(context != NULL);
    return context->arena;
}

/**
 * @brief Get the diagnostics from the context
 */
DiagnosticContext* type_inference_get_diagnostics(TypeInferenceContext* context) {
    assert(context != NULL);
    return context->diagnostics;
}
