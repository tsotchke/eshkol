/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Arena Memory Management Implementation
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef __cplusplus
#include <new>      // for std::bad_alloc
#include <stdexcept>
#endif

// Default alignment for memory allocations
#define DEFAULT_ALIGNMENT 8

// Global tape pointer for AD operations (shared across JIT modules in REPL)
ad_tape_t* __current_ad_tape = nullptr;

// Global AD mode flag (shared across JIT modules in REPL)
// CRITICAL: This must be shared so lambdas from one module can see AD mode set by another
bool __ad_mode_active = false;

// NESTED GRADIENT FIX: Tape stack for arbitrary-depth nested gradients
// MAX_TAPE_DEPTH must match the value in llvm_codegen.cpp
static const size_t MAX_TAPE_DEPTH = 32;
ad_tape_t* __ad_tape_stack[MAX_TAPE_DEPTH] = {nullptr};
uint64_t __ad_tape_depth = 0;

// DOUBLE BACKWARD: Storage for outer AD node when in nested gradient
void* __outer_ad_node_storage = nullptr;
void* __outer_ad_node_to_inner = nullptr;
void* __outer_grad_accumulator = nullptr;
void* __inner_var_node_ptr = nullptr;
uint64_t __gradient_x_degree = 0;

// N-DIMENSIONAL DERIVATIVES: Stack of outer AD nodes for arbitrary depth nesting
#define MAX_TAPE_DEPTH 16
void* __outer_ad_node_stack[MAX_TAPE_DEPTH] = {nullptr};
uint64_t __outer_ad_node_depth = 0;

// Global shared arena for REPL mode (persistent across evaluations)
arena_t* __repl_shared_arena = nullptr;

// Utility function to align size to boundary
static size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Create a new arena block
static arena_block_t* create_arena_block(size_t size) {
    arena_block_t* block = (arena_block_t*)malloc(sizeof(arena_block_t));
    if (!block) {
        eshkol_error("Failed to allocate arena block structure");
        return nullptr;
    }
    
    block->memory = (uint8_t*)malloc(size);
    if (!block->memory) {
        eshkol_error("Failed to allocate arena block memory of size %zu", size);
        free(block);
        return nullptr;
    }
    
    block->size = size;
    block->used = 0;
    block->next = nullptr;
    
    return block;
}

// Free an arena block
static void free_arena_block(arena_block_t* block) {
    if (block) {
        free(block->memory);
        free(block);
    }
}

// Arena creation and destruction
arena_t* arena_create(size_t default_block_size) {
    if (default_block_size < 1024) {
        default_block_size = 1024; // Minimum block size
    }
    
    arena_t* arena = (arena_t*)malloc(sizeof(arena_t));
    if (!arena) {
        eshkol_error("Failed to allocate arena structure");
        return nullptr;
    }
    
    arena->current_block = create_arena_block(default_block_size);
    if (!arena->current_block) {
        free(arena);
        return nullptr;
    }
    
    arena->current_scope = nullptr;
    arena->default_block_size = default_block_size;
    arena->total_allocated = default_block_size;
    arena->alignment = DEFAULT_ALIGNMENT;
    
    eshkol_debug("Created arena with default block size %zu", default_block_size);
    return arena;
}

void arena_destroy(arena_t* arena) {
    if (!arena) return;
    
    // Free all blocks
    arena_block_t* block = arena->current_block;
    while (block) {
        arena_block_t* next = block->next;
        free_arena_block(block);
        block = next;
    }
    
    // Free all scopes
    arena_scope_t* scope = arena->current_scope;
    while (scope) {
        arena_scope_t* parent = scope->parent;
        free(scope);
        scope = parent;
    }
    
    eshkol_debug("Destroyed arena, freed %zu bytes", arena->total_allocated);
    free(arena);
}

// Core allocation function
void* arena_allocate_aligned(arena_t* arena, size_t size, size_t alignment) {
    if (!arena || size == 0) return nullptr;
    
    if (alignment == 0) alignment = DEFAULT_ALIGNMENT;
    
    // Align the size
    size_t aligned_size = align_size(size, alignment);
    
    // Check if current block has enough space
    arena_block_t* block = arena->current_block;
    size_t current_used = align_size(block->used, alignment);
    
    if (current_used + aligned_size > block->size) {
        // Need a new block
        size_t new_block_size = (aligned_size > arena->default_block_size) ? 
                               aligned_size : arena->default_block_size;
        
        arena_block_t* new_block = create_arena_block(new_block_size);
        if (!new_block) {
            eshkol_error("Failed to allocate new arena block of size %zu", new_block_size);
            return nullptr;
        }
        
        // Link the new block to the front
        new_block->next = arena->current_block;
        arena->current_block = new_block;
        arena->total_allocated += new_block_size;
        
        block = new_block;
        current_used = 0;
    }
    
    // Allocate from current block
    void* ptr = block->memory + current_used;
    block->used = current_used + aligned_size;
    
    return ptr;
}

void* arena_allocate(arena_t* arena, size_t size) {
    return arena_allocate_aligned(arena, size, DEFAULT_ALIGNMENT);
}

void* arena_allocate_zeroed(arena_t* arena, size_t size) {
    void* ptr = arena_allocate(arena, size);
    if (ptr) {
        memset(ptr, 0, size);
    }
    return ptr;
}

// Scope management
void arena_push_scope(arena_t* arena) {
    if (!arena) return;
    
    arena_scope_t* scope = (arena_scope_t*)malloc(sizeof(arena_scope_t));
    if (!scope) {
        eshkol_error("Failed to allocate arena scope");
        return;
    }
    
    scope->block = arena->current_block;
    scope->used = arena->current_block->used;
    scope->parent = arena->current_scope;
    arena->current_scope = scope;
    
    eshkol_debug("Pushed arena scope");
}

void arena_pop_scope(arena_t* arena) {
    if (!arena || !arena->current_scope) {
        eshkol_warn("Attempted to pop arena scope with no active scope");
        return;
    }
    
    arena_scope_t* scope = arena->current_scope;
    
    // Restore arena state to scope start
    // Free any blocks allocated after this scope
    arena_block_t* block = arena->current_block;
    while (block && block != scope->block) {
        arena_block_t* next = block->next;
        arena->total_allocated -= block->size;
        free_arena_block(block);
        block = next;
    }
    
    arena->current_block = scope->block;
    if (arena->current_block) {
        arena->current_block->used = scope->used;
    }
    
    arena->current_scope = scope->parent;
    free(scope);
    
    eshkol_debug("Popped arena scope");
}

void arena_reset(arena_t* arena) {
    if (!arena) return;
    
    // Reset all blocks except the first one
    arena_block_t* first_block = nullptr;
    arena_block_t* block = arena->current_block;
    
    // Find the last block (first allocated)
    while (block) {
        if (block->next == nullptr) {
            first_block = block;
            break;
        }
        block = block->next;
    }
    
    // Free all blocks except the first
    block = arena->current_block;
    while (block && block != first_block) {
        arena_block_t* next = block->next;
        arena->total_allocated -= block->size;
        free_arena_block(block);
        block = next;
    }
    
    // Reset first block
    if (first_block) {
        first_block->used = 0;
        first_block->next = nullptr;
        arena->current_block = first_block;
    }
    
    // Clear all scopes
    arena_scope_t* scope = arena->current_scope;
    while (scope) {
        arena_scope_t* parent = scope->parent;
        free(scope);
        scope = parent;
    }
    arena->current_scope = nullptr;
    
    eshkol_debug("Reset arena");
}

// Statistics
size_t arena_get_used_memory(const arena_t* arena) {
    if (!arena) return 0;
    
    size_t used = 0;
    arena_block_t* block = arena->current_block;
    while (block) {
        used += block->used;
        block = block->next;
    }
    return used;
}

size_t arena_get_total_memory(const arena_t* arena) {
    return arena ? arena->total_allocated : 0;
}

size_t arena_get_block_count(const arena_t* arena) {
    if (!arena) return 0;
    
    size_t count = 0;
    arena_block_t* block = arena->current_block;
    while (block) {
        count++;
        block = block->next;
    }
    return count;
}

// List-specific allocation functions
arena_cons_cell_t* arena_allocate_cons_cell(arena_t* arena) {
    return (arena_cons_cell_t*)arena_allocate_aligned(arena, sizeof(arena_cons_cell_t), 
                                                     alignof(arena_cons_cell_t));
}

void* arena_allocate_list_node(arena_t* arena, size_t element_size, size_t count) {
    return arena_allocate(arena, element_size * count);
}

// Tagged cons cell allocation implementation
arena_tagged_cons_cell_t* arena_allocate_tagged_cons_cell(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate tagged cons cell: null arena");
        return nullptr;
    }
    
    arena_tagged_cons_cell_t* cell = (arena_tagged_cons_cell_t*)
        arena_allocate_aligned(arena, sizeof(arena_tagged_cons_cell_t), 16);
    
    if (!cell) {
        eshkol_error("Failed to allocate tagged cons cell");
        return nullptr;
    }
    
    // Initialize with null types and zero values (Phase 3B: new structure)
    cell->car.type = ESHKOL_VALUE_NULL;
    cell->car.flags = 0;
    cell->car.reserved = 0;
    cell->car.data.raw_val = 0;
    
    cell->cdr.type = ESHKOL_VALUE_NULL;
    cell->cdr.flags = 0;
    cell->cdr.reserved = 0;
    cell->cdr.data.raw_val = 0;
    
    return cell;
}

// Batch allocation for efficiency
arena_tagged_cons_cell_t* arena_allocate_tagged_cons_batch(arena_t* arena, size_t count) {
    if (!arena || count == 0) {
        eshkol_error("Invalid parameters for batch tagged cons allocation");
        return nullptr;
    }
    
    size_t total_size = sizeof(arena_tagged_cons_cell_t) * count;
    arena_tagged_cons_cell_t* cells = (arena_tagged_cons_cell_t*)
        arena_allocate_aligned(arena, total_size, 16);
    
    if (!cells) {
        eshkol_error("Failed to allocate %zu tagged cons cells", count);
        return nullptr;
    }
    
    // Initialize all cells (Phase 3B: new structure)
    for (size_t i = 0; i < count; i++) {
        cells[i].car.type = ESHKOL_VALUE_NULL;
        cells[i].car.flags = 0;
        cells[i].car.reserved = 0;
        cells[i].car.data.raw_val = 0;
        
        cells[i].cdr.type = ESHKOL_VALUE_NULL;
        cells[i].cdr.flags = 0;
        cells[i].cdr.reserved = 0;
        cells[i].cdr.data.raw_val = 0;
    }
    
    return cells;
}

// Convenience constructor for int64 values
arena_tagged_cons_cell_t* arena_create_int64_cons(arena_t* arena,
                                                   int64_t car, uint8_t car_type,
                                                   int64_t cdr, uint8_t cdr_type) {
    arena_tagged_cons_cell_t* cell = arena_allocate_tagged_cons_cell(arena);
    if (!cell) return nullptr;
    
    // Phase 3B: Access nested tagged_value structure
    cell->car.type = car_type;
    cell->car.data.int_val = car;
    cell->cdr.type = cdr_type;
    cell->cdr.data.int_val = cdr;
    
    return cell;
}

// Convenience constructor for mixed values
arena_tagged_cons_cell_t* arena_create_mixed_cons(arena_t* arena,
                                                   eshkol_tagged_data_t car, uint8_t car_type,
                                                   eshkol_tagged_data_t cdr, uint8_t cdr_type) {
    arena_tagged_cons_cell_t* cell = arena_allocate_tagged_cons_cell(arena);
    if (!cell) return nullptr;
    
    // Phase 3B: Access nested tagged_value structure
    // Use raw_val for efficient union copy (all union members share the same memory)
    cell->car.type = car_type;
    cell->car.data.raw_val = car.raw_val;
    
    cell->cdr.type = cdr_type;
    cell->cdr.data.raw_val = cdr.raw_val;
    
    return cell;
}

// Type-safe data retrieval functions
int64_t arena_tagged_cons_get_int64(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get int64 from null tagged cons cell");
        return 0;
    }

    // Phase 3B: Access nested tagged_value structure
    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    uint8_t type = tv->type;
    if (!ESHKOL_IS_INT64_TYPE(type)) {
        eshkol_error("Attempted to get int64 from non-int64 cell (type=%d)", type);
        return 0;
    }

    return tv->data.int_val;
}

double arena_tagged_cons_get_double(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get double from null tagged cons cell");
        return 0.0;
    }
    
    // Phase 3B: Access nested tagged_value structure
    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    uint8_t type = tv->type;
    if (!ESHKOL_IS_DOUBLE_TYPE(type)) {
        eshkol_error("Attempted to get double from non-double cell (type=%d)", type);
        return 0.0;
    }
    
    return tv->data.double_val;
}

uint64_t arena_tagged_cons_get_ptr(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get pointer from null tagged cons cell");
        return 0;
    }

    // Phase 3B: Access nested tagged_value structure
    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    uint8_t type = tv->type;
    uint8_t base_type = ESHKOL_GET_BASE_TYPE(type);

    if (base_type == ESHKOL_VALUE_NULL) {
        return 0;
    }

    // HOMOICONIC FIX: Accept any pointer type (CONS_PTR, STRING_PTR, VECTOR_PTR, etc.)
    if (!ESHKOL_IS_ANY_PTR_TYPE(type)) {
        eshkol_error("Attempted to get pointer from non-pointer cell (type=%d)", type);
        return 0;
    }

    return tv->data.ptr_val;
}

// Type-safe data setting functions
void arena_tagged_cons_set_int64(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                  int64_t value, uint8_t type) {
    if (!cell) {
        eshkol_error("Cannot set int64 on null tagged cons cell");
        return;
    }
    
    if (!ESHKOL_IS_INT64_TYPE(type)) {
        eshkol_error("Invalid type for int64 value: %d", type);
        return;
    }
    
    // Phase 3B: Access nested tagged_value structure
    eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    tv->type = type;
    tv->data.int_val = value;
}

void arena_tagged_cons_set_double(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                   double value, uint8_t type) {
    if (!cell) {
        eshkol_error("Cannot set double on null tagged cons cell");
        return;
    }
    
    if (!ESHKOL_IS_DOUBLE_TYPE(type)) {
        eshkol_error("Invalid type for double value: %d", type);
        return;
    }
    
    // Phase 3B: Access nested tagged_value structure
    eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    tv->type = type;
    tv->data.double_val = value;
}

void arena_tagged_cons_set_ptr(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                uint64_t value, uint8_t type) {
    if (!cell) {
        eshkol_error("Cannot set pointer on null tagged cons cell");
        return;
    }

    // Allow any pointer type (CONS_PTR, STRING_PTR, VECTOR_PTR, TENSOR_PTR, AD_NODE_PTR, LAMBDA_SEXPR)
    if (!ESHKOL_IS_ANY_PTR_TYPE(type)) {
        eshkol_error("Invalid type for pointer value: %d", type);
        return;
    }
    
    // Phase 3B: Access nested tagged_value structure
    eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    tv->type = type;
    tv->data.ptr_val = value;
}

void arena_tagged_cons_set_null(arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot set null on null tagged cons cell");
        return;
    }
    
    // Phase 3B: Access nested tagged_value structure
    eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    tv->type = ESHKOL_VALUE_NULL;
    tv->data.raw_val = 0;
}

// Type query functions
uint8_t arena_tagged_cons_get_type(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get type from null tagged cons cell");
        return ESHKOL_VALUE_NULL;
    }

    // Phase 3B: Access nested tagged_value structure
    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    return tv->type;
}

// Get flags from tagged cons cell
uint8_t arena_tagged_cons_get_flags(const arena_tagged_cons_cell_t* cell, bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get flags from null tagged cons cell");
        return 0;
    }

    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    return tv->flags;
}

bool arena_tagged_cons_is_type(const arena_tagged_cons_cell_t* cell, bool is_cdr, uint8_t type) {
    if (!cell) return false;
    
    // Phase 3B: Access nested tagged_value structure
    const eshkol_tagged_value_t* tv = is_cdr ? &cell->cdr : &cell->car;
    uint8_t actual_type = tv->type;
    return ESHKOL_GET_BASE_TYPE(actual_type) == ESHKOL_GET_BASE_TYPE(type);
}

// Direct tagged value access functions (NEW in Phase 3B)
// These enable direct storage and retrieval of complete tagged_value structs
void arena_tagged_cons_set_tagged_value(arena_tagged_cons_cell_t* cell,
                                         bool is_cdr,
                                         const eshkol_tagged_value_t* value) {
    if (!cell || !value) {
        eshkol_error("Cannot set tagged value: null parameter");
        return;
    }

    // Direct struct copy - this is the key optimization of Phase 3B!
    if (is_cdr) {
        cell->cdr = *value;
    } else {
        cell->car = *value;
    }
}

eshkol_tagged_value_t arena_tagged_cons_get_tagged_value(const arena_tagged_cons_cell_t* cell,
                                                          bool is_cdr) {
    if (!cell) {
        eshkol_error("Cannot get tagged value from null cell");
        // Return null value
        eshkol_tagged_value_t null_val;
        null_val.type = ESHKOL_VALUE_NULL;
        null_val.flags = 0;
        null_val.reserved = 0;
        null_val.data.int_val = 0;
        return null_val;
    }
    
    // Direct struct copy - this is the key optimization of Phase 3B!
    return is_cdr ? cell->cdr : cell->car;
}

// ===== DEEP EQUALITY COMPARISON =====
// Runtime helper for deep structural equality of tagged values
// Takes pointers to avoid struct-by-value ABI issues

bool eshkol_deep_equal(const eshkol_tagged_value_t* val1, const eshkol_tagged_value_t* val2) {
    if (!val1 || !val2) {
        return val1 == val2;  // Both null -> equal, one null -> not equal
    }

    uint8_t type1 = val1->type & 0x0F;
    uint8_t type2 = val2->type & 0x0F;

    // Helper to check if a value represents empty list
    auto is_empty_list = [](uint8_t type, const eshkol_tagged_value_t* val) -> bool {
        if (type == ESHKOL_VALUE_NULL) return true;
        if (type == ESHKOL_VALUE_CONS_PTR && val->data.ptr_val == 0) return true;
        return false;
    };

    bool empty1 = is_empty_list(type1, val1);
    bool empty2 = is_empty_list(type2, val2);

    // Both empty lists: equal (handles NULL vs CONS_PTR with null pointer)
    if (empty1 && empty2) {
        return true;
    }

    // One empty, one not: not equal
    if (empty1 || empty2) {
        return false;
    }

    // Both cons: recursively compare cars and cdrs
    if (type1 == ESHKOL_VALUE_CONS_PTR && type2 == ESHKOL_VALUE_CONS_PTR) {
        arena_tagged_cons_cell_t* cons1 = (arena_tagged_cons_cell_t*)val1->data.ptr_val;
        arena_tagged_cons_cell_t* cons2 = (arena_tagged_cons_cell_t*)val2->data.ptr_val;

        if (!cons1 || !cons2) {
            // If both null pointers, equal; if one null, not equal
            return cons1 == cons2;
        }

        // Compare cars recursively
        eshkol_tagged_value_t car1 = arena_tagged_cons_get_tagged_value(cons1, false);
        eshkol_tagged_value_t car2 = arena_tagged_cons_get_tagged_value(cons2, false);
        if (!eshkol_deep_equal(&car1, &car2)) {
            return false;
        }

        // Compare cdrs recursively
        eshkol_tagged_value_t cdr1 = arena_tagged_cons_get_tagged_value(cons1, true);
        eshkol_tagged_value_t cdr2 = arena_tagged_cons_get_tagged_value(cons2, true);
        return eshkol_deep_equal(&cdr1, &cdr2);
    }

    // Different types: not equal
    if (type1 != type2) {
        return false;
    }

    // Same types, compare values based on type
    switch (type1) {
        case ESHKOL_VALUE_INT64:
        case ESHKOL_VALUE_BOOL:
            return val1->data.int_val == val2->data.int_val;

        case ESHKOL_VALUE_DOUBLE:
            return val1->data.double_val == val2->data.double_val;

        case ESHKOL_VALUE_STRING_PTR:
            // For symbols/strings, compare pointer addresses (interned symbols)
            return val1->data.ptr_val == val2->data.ptr_val;

        case ESHKOL_VALUE_CLOSURE_PTR:
        case ESHKOL_VALUE_LAMBDA_SEXPR:
            // For closures and lambdas, compare pointer addresses
            return val1->data.ptr_val == val2->data.ptr_val;

        default:
            // Unknown types: compare raw int data
            return val1->data.int_val == val2->data.int_val;
    }
}

// ===== AD MEMORY MANAGEMENT IMPLEMENTATION =====

// Dual number allocation
eshkol_dual_number_t* arena_allocate_dual_number(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate dual number: null arena");
        return nullptr;
    }
    
    eshkol_dual_number_t* dual = (eshkol_dual_number_t*)
        arena_allocate_aligned(arena, sizeof(eshkol_dual_number_t), 8);
    
    if (dual) {
        dual->value = 0.0;
        dual->derivative = 0.0;
    }
    
    return dual;
}

eshkol_dual_number_t* arena_allocate_dual_batch(arena_t* arena, size_t count) {
    if (!arena || count == 0) {
        eshkol_error("Invalid parameters for batch dual number allocation");
        return nullptr;
    }
    
    size_t total_size = sizeof(eshkol_dual_number_t) * count;
    eshkol_dual_number_t* duals = (eshkol_dual_number_t*)
        arena_allocate_aligned(arena, total_size, 8);
    
    if (duals) {
        for (size_t i = 0; i < count; i++) {
            duals[i].value = 0.0;
            duals[i].derivative = 0.0;
        }
    }
    
    return duals;
}

// AD node allocation for computational graphs
ad_node_t* arena_allocate_ad_node(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate AD node: null arena");
        return nullptr;
    }
    
    ad_node_t* node = (ad_node_t*)
        arena_allocate_aligned(arena, sizeof(ad_node_t), 8);
    
    if (node) {
        node->type = AD_NODE_CONSTANT;
        node->value = 0.0;
        node->gradient = 0.0;
        node->input1 = nullptr;
        node->input2 = nullptr;
        node->id = 0;
    }
    
    return node;
}

ad_node_t* arena_allocate_ad_batch(arena_t* arena, size_t count) {
    if (!arena || count == 0) {
        eshkol_error("Invalid parameters for batch AD node allocation");
        return nullptr;
    }
    
    size_t total_size = sizeof(ad_node_t) * count;
    ad_node_t* nodes = (ad_node_t*)
        arena_allocate_aligned(arena, total_size, 8);
    
    if (nodes) {
        for (size_t i = 0; i < count; i++) {
            nodes[i].type = AD_NODE_CONSTANT;
            nodes[i].value = 0.0;
            nodes[i].gradient = 0.0;
            nodes[i].input1 = nullptr;
            nodes[i].input2 = nullptr;
            nodes[i].id = 0;
        }
    }
    
    return nodes;
}

// Tape allocation and management
ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity) {
    if (!arena) {
        eshkol_error("Cannot allocate tape: null arena");
        return nullptr;
    }
    
    if (initial_capacity == 0) {
        initial_capacity = 64; // Default capacity
    }
    
    ad_tape_t* tape = (ad_tape_t*)
        arena_allocate_aligned(arena, sizeof(ad_tape_t), 8);
    
    if (!tape) {
        eshkol_error("Failed to allocate tape structure");
        return nullptr;
    }
    
    // Allocate nodes array
    size_t nodes_size = sizeof(ad_node_t*) * initial_capacity;
    tape->nodes = (ad_node_t**)arena_allocate_aligned(arena, nodes_size, 8);
    
    if (!tape->nodes) {
        eshkol_error("Failed to allocate tape nodes array");
        return nullptr;
    }
    
    tape->num_nodes = 0;
    tape->capacity = initial_capacity;
    tape->variables = nullptr;
    tape->num_variables = 0;
    
    return tape;
}

void arena_tape_add_node(ad_tape_t* tape, ad_node_t* node) {
    if (!tape || !node) {
        eshkol_error("Cannot add node to tape: null parameter");
        return;
    }
    
    if (tape->num_nodes >= tape->capacity) {
        eshkol_error("Tape capacity exceeded: %zu/%zu", tape->num_nodes, tape->capacity);
        return;
    }
    
    tape->nodes[tape->num_nodes++] = node;
}

void arena_tape_reset(ad_tape_t* tape) {
    if (!tape) {
        eshkol_error("Cannot reset tape: null parameter");
        return;
    }
    
    // CRITICAL FIX: Reset gradients BEFORE resetting node count!
    // The loop was never executing because num_nodes was already 0
    size_t node_count = tape->num_nodes;
    
    // Reset all node gradients to 0
    for (size_t i = 0; i < node_count; i++) {
        if (tape->nodes[i]) {
            tape->nodes[i]->gradient = 0.0;
        }
    }
    
    // Now reset node count (for next forward pass)
    tape->num_nodes = 0;
}

// Additional tape query functions
ad_node_t* arena_tape_get_node(const ad_tape_t* tape, size_t index) {
    if (!tape) {
        eshkol_error("Cannot get node from null tape");
        return nullptr;
    }
    
    if (index >= tape->num_nodes) {
        eshkol_error("Tape index out of bounds: %zu >= %zu", index, tape->num_nodes);
        return nullptr;
    }
    
    return tape->nodes[index];
}

size_t arena_tape_get_node_count(const ad_tape_t* tape) {
    if (!tape) {
        eshkol_error("Cannot get node count from null tape");
        return 0;
    }
    
    return tape->num_nodes;
}

// ===== END AD MEMORY MANAGEMENT IMPLEMENTATION =====

// ===== OALR (Ownership-Aware Lexical Regions) IMPLEMENTATION =====

// Global region stack
eshkol_region_t* __region_stack[MAX_REGION_DEPTH] = {nullptr};
uint64_t __region_stack_depth = 0;

// Default global arena for allocations outside of any region
static arena_t* __global_arena = nullptr;

// Get or create the global arena
static arena_t* get_global_arena() {
    if (!__global_arena) {
        __global_arena = arena_create(65536);  // 64KB default
    }
    return __global_arena;
}

// Region creation
eshkol_region_t* region_create(const char* name, size_t size_hint) {
    // Use global arena to allocate the region structure itself
    arena_t* global = get_global_arena();

    eshkol_region_t* region = (eshkol_region_t*)
        arena_allocate_aligned(global, sizeof(eshkol_region_t), 8);

    if (!region) {
        eshkol_error("Failed to allocate region structure");
        return nullptr;
    }

    // Determine arena size
    size_t arena_size = (size_hint > 0) ? size_hint : 8192;
    if (arena_size < 1024) arena_size = 1024;

    // Create the region's arena
    region->arena = arena_create(arena_size);
    if (!region->arena) {
        eshkol_error("Failed to create region arena");
        return nullptr;
    }

    // Copy name if provided
    if (name) {
        size_t name_len = strlen(name) + 1;
        char* name_copy = (char*)arena_allocate(region->arena, name_len);
        if (name_copy) {
            strcpy(name_copy, name);
            region->name = name_copy;
        } else {
            region->name = nullptr;
        }
    } else {
        region->name = nullptr;
    }

    region->parent = nullptr;
    region->size_hint = size_hint;
    region->escape_count = 0;
    region->is_active = 0;

    eshkol_debug("Created region '%s' with size hint %zu",
                name ? name : "(anonymous)", size_hint);

    return region;
}

// Region destruction
void region_destroy(eshkol_region_t* region) {
    if (!region) return;

    if (region->is_active) {
        eshkol_warn("Destroying active region '%s' - popping from stack first",
                   region->name ? region->name : "(anonymous)");
        region_pop();
    }

    const char* name = region->name ? region->name : "(anonymous)";
    size_t used = region->arena ? arena_get_used_memory(region->arena) : 0;

    // Destroy the region's arena
    if (region->arena) {
        arena_destroy(region->arena);
        region->arena = nullptr;
    }

    eshkol_debug("Destroyed region '%s', freed %zu bytes", name, used);
}

// Push a region onto the stack
void region_push(eshkol_region_t* region) {
    if (!region) {
        eshkol_error("Cannot push null region");
        return;
    }

    if (__region_stack_depth >= MAX_REGION_DEPTH) {
        eshkol_error("Region stack overflow (max depth: %d)", MAX_REGION_DEPTH);
        return;
    }

    // Set parent to current top
    if (__region_stack_depth > 0) {
        region->parent = __region_stack[__region_stack_depth - 1];
    } else {
        region->parent = nullptr;
    }

    region->is_active = 1;
    __region_stack[__region_stack_depth++] = region;

    eshkol_debug("Pushed region '%s' (depth: %llu)",
                region->name ? region->name : "(anonymous)",
                __region_stack_depth);
}

// Pop the current region from the stack
void region_pop(void) {
    if (__region_stack_depth == 0) {
        eshkol_warn("Attempted to pop from empty region stack");
        return;
    }

    eshkol_region_t* region = __region_stack[--__region_stack_depth];
    region->is_active = 0;

    eshkol_debug("Popped region '%s' (depth: %llu)",
                region->name ? region->name : "(anonymous)",
                __region_stack_depth);

    // Destroy the region (this frees all its memory)
    region_destroy(region);
}

// Get the current active region
eshkol_region_t* region_current(void) {
    if (__region_stack_depth == 0) {
        return nullptr;
    }
    return __region_stack[__region_stack_depth - 1];
}

// Allocate in the current region (or global arena if no region)
void* region_allocate(size_t size) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate(region->arena, size);
    }
    // Fallback to global arena
    return arena_allocate(get_global_arena(), size);
}

void* region_allocate_aligned(size_t size, size_t alignment) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate_aligned(region->arena, size, alignment);
    }
    return arena_allocate_aligned(get_global_arena(), size, alignment);
}

void* region_allocate_zeroed(size_t size) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate_zeroed(region->arena, size);
    }
    return arena_allocate_zeroed(get_global_arena(), size);
}

// Region-aware tagged cons cell allocation
arena_tagged_cons_cell_t* region_allocate_tagged_cons_cell(void) {
    eshkol_region_t* region = region_current();
    if (region && region->arena) {
        return arena_allocate_tagged_cons_cell(region->arena);
    }
    return arena_allocate_tagged_cons_cell(get_global_arena());
}

// Region statistics
size_t region_get_used_memory(const eshkol_region_t* region) {
    if (!region || !region->arena) return 0;
    return arena_get_used_memory(region->arena);
}

size_t region_get_total_memory(const eshkol_region_t* region) {
    if (!region || !region->arena) return 0;
    return arena_get_total_memory(region->arena);
}

const char* region_get_name(const eshkol_region_t* region) {
    if (!region) return nullptr;
    return region->name;
}

uint64_t region_get_depth(void) {
    return __region_stack_depth;
}

// ===== END OALR IMPLEMENTATION =====

// ===== SHARED (REFERENCE-COUNTED) MEMORY MANAGEMENT IMPLEMENTATION =====
// Reference-counted allocation for values with complex, dynamic lifetimes

// Get the shared header from a user pointer (header is before user data)
eshkol_shared_header_t* shared_get_header(void* ptr) {
    if (!ptr) return nullptr;
    // Header is stored before the user data
    return (eshkol_shared_header_t*)((uint8_t*)ptr - sizeof(eshkol_shared_header_t));
}

// Allocate a shared (reference-counted) value
void* shared_allocate(size_t size, void (*destructor)(void*)) {
    return shared_allocate_typed(size, ESHKOL_VALUE_NULL, destructor);
}

void* shared_allocate_typed(size_t size, uint8_t value_type, void (*destructor)(void*)) {
    // Allocate header + user data
    size_t total_size = sizeof(eshkol_shared_header_t) + size;

    // Use malloc for shared allocations (they outlive regions)
    uint8_t* memory = (uint8_t*)malloc(total_size);
    if (!memory) {
        eshkol_error("Failed to allocate shared memory of size %zu", size);
        return nullptr;
    }

    // Initialize header
    eshkol_shared_header_t* header = (eshkol_shared_header_t*)memory;
    header->destructor = destructor;
    header->ref_count = 1;      // Starts with ref count of 1
    header->weak_count = 0;
    header->flags = 0;
    header->value_type = value_type;
    header->reserved = 0;
    header->reserved2 = 0;

    // Return pointer to user data (after header)
    void* user_data = memory + sizeof(eshkol_shared_header_t);

    eshkol_debug("Allocated shared memory at %p (header at %p), size=%zu, type=%d",
                user_data, (void*)header, size, value_type);

    return user_data;
}

// Increment reference count
void shared_retain(void* ptr) {
    if (!ptr) return;

    eshkol_shared_header_t* header = shared_get_header(ptr);
    if (!header) return;

    header->ref_count++;

    eshkol_debug("Retained shared at %p, ref_count now %u", ptr, header->ref_count);
}

// Decrement reference count, deallocate if zero
void shared_release(void* ptr) {
    if (!ptr) return;

    eshkol_shared_header_t* header = shared_get_header(ptr);
    if (!header) return;

    if (header->ref_count == 0) {
        eshkol_warn("Releasing shared with zero ref count at %p", ptr);
        return;
    }

    header->ref_count--;

    eshkol_debug("Released shared at %p, ref_count now %u", ptr, header->ref_count);

    if (header->ref_count == 0) {
        // Call destructor if provided
        if (header->destructor) {
            eshkol_debug("Calling destructor for shared at %p", ptr);
            header->destructor(ptr);
        }

        // If there are no weak references, free immediately
        if (header->weak_count == 0) {
            eshkol_debug("Freeing shared memory at %p", ptr);
            free(header);
        } else {
            // Mark as deallocated but keep header for weak refs
            header->flags |= 0x01;  // DEALLOCATED flag
            eshkol_debug("Shared at %p deallocated but %u weak refs remain",
                        ptr, header->weak_count);
        }
    }
}

// Get current reference count (for debugging)
uint32_t shared_ref_count(void* ptr) {
    if (!ptr) return 0;

    eshkol_shared_header_t* header = shared_get_header(ptr);
    if (!header) return 0;

    return header->ref_count;
}

// Create a weak reference to a shared value
eshkol_weak_ref_t* weak_ref_create(void* shared_ptr) {
    if (!shared_ptr) return nullptr;

    eshkol_shared_header_t* header = shared_get_header(shared_ptr);
    if (!header) return nullptr;

    // Allocate weak ref structure
    eshkol_weak_ref_t* weak = (eshkol_weak_ref_t*)malloc(sizeof(eshkol_weak_ref_t));
    if (!weak) {
        eshkol_error("Failed to allocate weak reference");
        return nullptr;
    }

    weak->header = header;
    weak->data = shared_ptr;

    // Increment weak count
    header->weak_count++;

    eshkol_debug("Created weak ref at %p to shared %p, weak_count now %u",
                (void*)weak, shared_ptr, header->weak_count);

    return weak;
}

// Upgrade weak reference to strong (returns NULL if value was freed)
void* weak_ref_upgrade(eshkol_weak_ref_t* weak) {
    if (!weak || !weak->header) return nullptr;

    // Check if the shared value has been deallocated
    if (weak->header->flags & 0x01) {  // DEALLOCATED flag
        eshkol_debug("Cannot upgrade weak ref - target deallocated");
        return nullptr;
    }

    // Check if ref count is zero (shouldn't happen if not deallocated)
    if (weak->header->ref_count == 0) {
        return nullptr;
    }

    // Increment strong ref count
    weak->header->ref_count++;

    eshkol_debug("Upgraded weak ref at %p to strong, ref_count now %u",
                (void*)weak, weak->header->ref_count);

    return weak->data;
}

// Release a weak reference
void weak_ref_release(eshkol_weak_ref_t* weak) {
    if (!weak) return;

    if (weak->header) {
        weak->header->weak_count--;

        eshkol_debug("Released weak ref at %p, weak_count now %u",
                    (void*)weak, weak->header->weak_count);

        // If shared value was deallocated and this was last weak ref, free header
        if ((weak->header->flags & 0x01) && weak->header->weak_count == 0) {
            eshkol_debug("Freeing shared header at %p (all refs gone)", (void*)weak->header);
            free(weak->header);
        }
    }

    free(weak);
}

// Check if weak reference target still exists
bool weak_ref_is_alive(eshkol_weak_ref_t* weak) {
    if (!weak || !weak->header) return false;

    // Check deallocated flag
    if (weak->header->flags & 0x01) return false;

    // Check ref count
    return weak->header->ref_count > 0;
}

// ===== END SHARED MEMORY MANAGEMENT IMPLEMENTATION =====

#ifdef __cplusplus

// C++ Arena wrapper implementation
Arena::Arena(size_t default_block_size) : arena_(arena_create(default_block_size)) {
    if (!arena_) {
        throw std::bad_alloc();
    }
}

Arena::~Arena() {
    arena_destroy(arena_);
}

Arena::Arena(Arena&& other) noexcept : arena_(other.arena_) {
    other.arena_ = nullptr;
}

Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        arena_destroy(arena_);
        arena_ = other.arena_;
        other.arena_ = nullptr;
    }
    return *this;
}

void* Arena::allocate(size_t size) {
    return arena_allocate(arena_, size);
}

void* Arena::allocate_aligned(size_t size, size_t alignment) {
    return arena_allocate_aligned(arena_, size, alignment);
}

void* Arena::allocate_zeroed(size_t size) {
    return arena_allocate_zeroed(arena_, size);
}

size_t Arena::get_used_memory() const {
    return arena_get_used_memory(arena_);
}

size_t Arena::get_total_memory() const {
    return arena_get_total_memory(arena_);
}

size_t Arena::get_block_count() const {
    return arena_get_block_count(arena_);
}

void Arena::reset() {
    arena_reset(arena_);
}

// Arena::Scope implementation
Arena::Scope::Scope(Arena& arena) : arena_(&arena), active_(true) {
    arena_push_scope(arena.arena_);
}

Arena::Scope::~Scope() {
    if (active_) {
        arena_pop_scope(arena_->arena_);
    }
}


// ===== CLOSURE ENVIRONMENT MEMORY MANAGEMENT IMPLEMENTATION =====

eshkol_closure_env_t* arena_allocate_closure_env(arena_t* arena, size_t num_captures) {
    if (!arena) {
        eshkol_error("Cannot allocate closure environment: null arena");
        return nullptr;
    }
    
    if (num_captures == 0) {
        eshkol_warn("Allocating closure environment with zero captures");
    }
    
    // Calculate total size: header + captures array
    size_t size = sizeof(eshkol_closure_env_t) + 
                  (num_captures * sizeof(eshkol_tagged_value_t));
    
    eshkol_closure_env_t* env = (eshkol_closure_env_t*)
        arena_allocate_aligned(arena, size, 16);
    
    if (!env) {
        eshkol_error("Failed to allocate closure environment for %zu captures", num_captures);
        return nullptr;
    }
    
    // Initialize environment header
    env->num_captures = num_captures;
    
    // Initialize all captures to null
    for (size_t i = 0; i < num_captures; i++) {
        env->captures[i].type = ESHKOL_VALUE_NULL;
        env->captures[i].flags = 0;
        env->captures[i].reserved = 0;
        env->captures[i].data.raw_val = 0;
    }
    
    eshkol_debug("Allocated closure environment for %zu captures at %p",
                num_captures, (void*)env);

    return env;
}

eshkol_closure_t* arena_allocate_closure(arena_t* arena, uint64_t func_ptr, size_t num_captures, uint64_t sexpr_ptr) {
    if (!arena) {
        eshkol_error("Cannot allocate closure: null arena");
        return nullptr;
    }

    // Allocate the closure structure
    eshkol_closure_t* closure = (eshkol_closure_t*)
        arena_allocate_aligned(arena, sizeof(eshkol_closure_t), 16);

    if (!closure) {
        eshkol_error("Failed to allocate closure structure");
        return nullptr;
    }

    closure->func_ptr = func_ptr;
    closure->sexpr_ptr = sexpr_ptr;  // Store S-expression for homoiconicity

    // Allocate environment if there are captures
    if (num_captures > 0) {
        closure->env = arena_allocate_closure_env(arena, num_captures);
        if (!closure->env) {
            eshkol_error("Failed to allocate closure environment");
            return nullptr;
        }
    } else {
        closure->env = nullptr;
    }

    eshkol_debug("Allocated closure at %p with func_ptr=%p, env=%p (%zu captures), sexpr=%p",
                (void*)closure, (void*)func_ptr, (void*)closure->env, num_captures, (void*)sexpr_ptr);

    return closure;
}

// ===== END CLOSURE ENVIRONMENT MEMORY MANAGEMENT IMPLEMENTATION =====

// ===== LAMBDA REGISTRY IMPLEMENTATION =====
// Runtime table for mapping function pointers to S-expressions (homoiconicity)

eshkol_lambda_registry_t* g_lambda_registry = nullptr;

void eshkol_lambda_registry_init(void) {
    if (g_lambda_registry) {
        return;  // Already initialized
    }

    g_lambda_registry = (eshkol_lambda_registry_t*)malloc(sizeof(eshkol_lambda_registry_t));
    if (!g_lambda_registry) {
        eshkol_error("Failed to allocate lambda registry");
        return;
    }

    g_lambda_registry->capacity = 64;  // Initial capacity
    g_lambda_registry->count = 0;
    g_lambda_registry->entries = (eshkol_lambda_entry_t*)malloc(
        sizeof(eshkol_lambda_entry_t) * g_lambda_registry->capacity);

    if (!g_lambda_registry->entries) {
        eshkol_error("Failed to allocate lambda registry entries");
        free(g_lambda_registry);
        g_lambda_registry = nullptr;
        return;
    }

    eshkol_debug("Lambda registry initialized with capacity %zu", g_lambda_registry->capacity);
}

void eshkol_lambda_registry_destroy(void) {
    if (!g_lambda_registry) {
        return;
    }

    if (g_lambda_registry->entries) {
        free(g_lambda_registry->entries);
    }
    free(g_lambda_registry);
    g_lambda_registry = nullptr;
}

void eshkol_lambda_registry_add(uint64_t func_ptr, uint64_t sexpr_ptr, const char* name) {
    if (!g_lambda_registry) {
        eshkol_lambda_registry_init();
    }

    if (!g_lambda_registry) {
        return;  // Init failed
    }

    // Check if already registered (update if so)
    for (size_t i = 0; i < g_lambda_registry->count; i++) {
        if (g_lambda_registry->entries[i].func_ptr == func_ptr) {
            g_lambda_registry->entries[i].sexpr_ptr = sexpr_ptr;
            g_lambda_registry->entries[i].name = name;
            eshkol_debug("Updated lambda registry entry for %s at %p -> sexpr %p",
                        name ? name : "(anonymous)", (void*)func_ptr, (void*)sexpr_ptr);
            return;
        }
    }

    // Grow if needed
    if (g_lambda_registry->count >= g_lambda_registry->capacity) {
        size_t new_capacity = g_lambda_registry->capacity * 2;
        eshkol_lambda_entry_t* new_entries = (eshkol_lambda_entry_t*)realloc(
            g_lambda_registry->entries,
            sizeof(eshkol_lambda_entry_t) * new_capacity);

        if (!new_entries) {
            eshkol_error("Failed to grow lambda registry");
            return;
        }

        g_lambda_registry->entries = new_entries;
        g_lambda_registry->capacity = new_capacity;
    }

    // Add new entry
    size_t idx = g_lambda_registry->count++;
    g_lambda_registry->entries[idx].func_ptr = func_ptr;
    g_lambda_registry->entries[idx].sexpr_ptr = sexpr_ptr;
    g_lambda_registry->entries[idx].name = name;

    eshkol_debug("Lambda registry: added %s func=%p sexpr=%p",
                name ? name : "(anon)", (void*)func_ptr, (void*)sexpr_ptr);
}

uint64_t eshkol_lambda_registry_lookup(uint64_t func_ptr) {
    if (!g_lambda_registry) {
        return 0;
    }

    for (size_t i = 0; i < g_lambda_registry->count; i++) {
        if (g_lambda_registry->entries[i].func_ptr == func_ptr) {
            return g_lambda_registry->entries[i].sexpr_ptr;
        }
    }

    return 0;  // Not found
}

// ===== END LAMBDA REGISTRY IMPLEMENTATION =====

// ===== UNIFIED DISPLAY IMPLEMENTATION =====
// Single source of truth for displaying all Eshkol values

// Forward declarations for internal helpers
static void display_atom(const eshkol_tagged_value_t* value, eshkol_display_opts_t* opts);
static void display_tensor(uint64_t tensor_ptr, eshkol_display_opts_t* opts);
static void display_vector(uint64_t vector_ptr, eshkol_display_opts_t* opts);
static void display_char(uint32_t codepoint, eshkol_display_opts_t* opts);

// Get output stream (defaults to stdout)
static FILE* get_output(eshkol_display_opts_t* opts) {
    if (opts && opts->output) {
        return (FILE*)opts->output;
    }
    return stdout;
}

// Display a single tagged value
void eshkol_display_value(const eshkol_tagged_value_t* value) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    eshkol_display_value_opts(value, &opts);
}

void eshkol_display_value_opts(const eshkol_tagged_value_t* value, eshkol_display_opts_t* opts) {
    if (!value) {
        fprintf(get_output(opts), "()");
        return;
    }

    // Check depth limit
    if (opts->current_depth > opts->max_depth) {
        fprintf(get_output(opts), "...");
        return;
    }

    uint8_t base_type = value->type & 0x0F;

    switch (base_type) {
        case ESHKOL_VALUE_NULL:
            fprintf(get_output(opts), "()");
            break;

        case ESHKOL_VALUE_INT64:
            fprintf(get_output(opts), "%lld", (long long)value->data.int_val);
            break;

        case ESHKOL_VALUE_DOUBLE:
            fprintf(get_output(opts), "%g", value->data.double_val);
            break;

        case ESHKOL_VALUE_BOOL:
            fprintf(get_output(opts), "%s", value->data.int_val ? "#t" : "#f");
            break;

        case ESHKOL_VALUE_CHAR:
            display_char((uint32_t)value->data.int_val, opts);
            break;

        case ESHKOL_VALUE_STRING_PTR:
            if (opts->quote_strings) {
                fprintf(get_output(opts), "\"%s\"", (const char*)value->data.ptr_val);
            } else {
                fprintf(get_output(opts), "%s", (const char*)value->data.ptr_val);
            }
            break;

        case ESHKOL_VALUE_SYMBOL:
            fprintf(get_output(opts), "%s", (const char*)value->data.ptr_val);
            break;

        case ESHKOL_VALUE_CONS_PTR:
            eshkol_display_list(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_LAMBDA_SEXPR:
            eshkol_display_lambda(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_CLOSURE_PTR:
            eshkol_display_closure(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_TENSOR_PTR:
            display_tensor(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_VECTOR_PTR:
            display_vector(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_DUAL_NUMBER: {
            eshkol_dual_number_t* dual = (eshkol_dual_number_t*)value->data.ptr_val;
            if (dual) {
                fprintf(get_output(opts), "(dual %g %g)", dual->value, dual->derivative);
            } else {
                fprintf(get_output(opts), "(dual 0 0)");
            }
            break;
        }

        case ESHKOL_VALUE_AD_NODE_PTR:
            fprintf(get_output(opts), "#<ad-node>");
            break;

        default:
            if (opts->show_types) {
                fprintf(get_output(opts), "#<unknown-type-%d:0x%llx>",
                       base_type, (unsigned long long)value->data.ptr_val);
            } else {
                fprintf(get_output(opts), "#<unknown>");
            }
            break;
    }
}

// Scheme 'write' semantics - quotes strings
void eshkol_write_value(const eshkol_tagged_value_t* value) {
    eshkol_display_opts_t opts = eshkol_display_default_opts();
    opts.quote_strings = 1;
    eshkol_display_value_opts(value, &opts);
}

// Display a list (cons cell chain)
void eshkol_display_list(uint64_t cons_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (cons_ptr == 0) {
        fprintf(out, "()");
        return;
    }

    // Check depth limit
    if (opts->current_depth > opts->max_depth) {
        fprintf(out, "(...)");
        return;
    }

    fprintf(out, "(");
    opts->current_depth++;

    uint64_t current = cons_ptr;
    bool first = true;

    while (current != 0) {
        arena_tagged_cons_cell_t* cell = (arena_tagged_cons_cell_t*)current;

        if (!first) {
            fprintf(out, " ");
        }
        first = false;

        // Display car
        eshkol_display_value_opts(&cell->car, opts);

        // Check cdr type
        uint8_t cdr_type = cell->cdr.type & 0x0F;

        if (cdr_type == ESHKOL_VALUE_NULL) {
            // Proper list end
            break;
        } else if (cdr_type == ESHKOL_VALUE_CONS_PTR) {
            // Continue to next cell
            current = cell->cdr.data.ptr_val;
        } else {
            // Dotted pair - display cdr and break
            fprintf(out, " . ");
            eshkol_display_value_opts(&cell->cdr, opts);
            break;
        }
    }

    opts->current_depth--;
    fprintf(out, ")");
}

// Display a lambda by looking up its S-expression in the registry
void eshkol_display_lambda(uint64_t func_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    // Look up S-expression in registry
    uint64_t sexpr_ptr = eshkol_lambda_registry_lookup(func_ptr);

    if (sexpr_ptr != 0) {
        // Found S-expression - display it
        eshkol_display_list(sexpr_ptr, opts);
    } else {
        // No S-expression found - display placeholder
        fprintf(out, "#<procedure>");
    }
}

// Display a closure by extracting its embedded S-expression
void eshkol_display_closure(uint64_t closure_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (closure_ptr == 0) {
        fprintf(out, "#<closure>");
        return;
    }

    // Closure struct: { func_ptr (8), env (8), sexpr_ptr (8) }
    eshkol_closure_t* closure = (eshkol_closure_t*)closure_ptr;
    uint64_t sexpr = closure->sexpr_ptr;

    if (sexpr != 0) {
        // Display the embedded S-expression
        eshkol_display_list(sexpr, opts);
    } else {
        // No S-expression - try the registry as fallback
        uint64_t registry_sexpr = eshkol_lambda_registry_lookup(closure->func_ptr);
        if (registry_sexpr != 0) {
            eshkol_display_list(registry_sexpr, opts);
        } else {
            fprintf(out, "#<closure>");
        }
    }
}

// Display a character
static void display_char(uint32_t codepoint, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    // Special character names
    switch (codepoint) {
        case ' ':  fprintf(out, "#\\space"); break;
        case '\n': fprintf(out, "#\\newline"); break;
        case '\t': fprintf(out, "#\\tab"); break;
        case '\r': fprintf(out, "#\\return"); break;
        case 0:    fprintf(out, "#\\null"); break;
        default:
            if (codepoint < 128 && codepoint >= 32) {
                // Printable ASCII
                fprintf(out, "#\\%c", (char)codepoint);
            } else {
                // Unicode or non-printable
                fprintf(out, "#\\x%X", codepoint);
            }
            break;
    }
}

// Tensor struct layout (must match LLVM TypeSystem tensor_type):
// struct Tensor {
//     uint64_t* dimensions;      // idx 0: array of dimension sizes
//     uint64_t  num_dimensions;  // idx 1: number of dimensions
//     int64_t*  elements;        // idx 2: element data (doubles stored as int64 bits)
//     uint64_t  total_elements;  // idx 3: total number of elements
// };
typedef struct {
    uint64_t* dimensions;
    uint64_t  num_dimensions;
    int64_t*  elements;
    uint64_t  total_elements;
} eshkol_tensor_t;

// Recursive helper for displaying N-dimensional tensors
static void display_tensor_recursive(FILE* out, const eshkol_tensor_t* tensor,
                                      uint64_t current_dim, uint64_t offset) {
    if (tensor->num_dimensions == 0) {
        fprintf(out, "#()");
        return;
    }

    uint64_t dim_size = tensor->dimensions[current_dim];

    // Base case: innermost dimension - print actual elements
    if (current_dim == tensor->num_dimensions - 1) {
        fprintf(out, "(");
        for (uint64_t i = 0; i < dim_size; i++) {
            if (i > 0) fprintf(out, " ");
            // Elements stored as int64 bit pattern of double
            int64_t bits = tensor->elements[offset + i];
            double value;
            memcpy(&value, &bits, sizeof(double));
            fprintf(out, "%g", value);
        }
        fprintf(out, ")");
        return;
    }

    // Recursive case: compute stride and iterate over slices
    uint64_t stride = 1;
    for (uint64_t k = current_dim + 1; k < tensor->num_dimensions; k++) {
        stride *= tensor->dimensions[k];
    }

    fprintf(out, "(");
    for (uint64_t i = 0; i < dim_size; i++) {
        if (i > 0) fprintf(out, " ");
        display_tensor_recursive(out, tensor, current_dim + 1, offset + i * stride);
    }
    fprintf(out, ")");
}

// Display a tensor with proper N-dimensional structure
static void display_tensor(uint64_t tensor_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (tensor_ptr == 0) {
        fprintf(out, "#()");
        return;
    }

    const eshkol_tensor_t* tensor = (const eshkol_tensor_t*)tensor_ptr;

    // Validate tensor structure
    if (tensor->num_dimensions == 0 || tensor->total_elements == 0) {
        fprintf(out, "#()");
        return;
    }

    if (tensor->dimensions == NULL || tensor->elements == NULL) {
        fprintf(out, "#<invalid-tensor>");
        return;
    }

    // Print tensor prefix then contents
    fprintf(out, "#");
    display_tensor_recursive(out, tensor, 0, 0);
}

// Scheme vector structure:
// [length: i64] at offset 0 (8 bytes)
// [elem0: tagged_value] at offset 8 (16 bytes each)
// [elem1: tagged_value] at offset 24
// etc.

// Display a Scheme vector with proper element formatting
static void display_vector(uint64_t vector_ptr, eshkol_display_opts_t* opts) {
    FILE* out = get_output(opts);

    if (vector_ptr == 0) {
        fprintf(out, "#()");
        return;
    }

    // Read length from start of vector
    uint64_t* len_ptr = (uint64_t*)vector_ptr;
    uint64_t length = *len_ptr;

    // Validate length (sanity check)
    if (length > 10000) {
        fprintf(out, "#<invalid-vector>");
        return;
    }

    fprintf(out, "#(");

    // Elements start after the 8-byte length field
    // Each element is a tagged_value (16 bytes)
    uint8_t* elem_base = (uint8_t*)vector_ptr + 8;

    for (uint64_t i = 0; i < length; i++) {
        if (i > 0) fprintf(out, " ");

        // Get pointer to i-th tagged_value element
        eshkol_tagged_value_t* elem = (eshkol_tagged_value_t*)(elem_base + i * sizeof(eshkol_tagged_value_t));

        // Recursively display the element
        eshkol_display_value_opts(elem, opts);
    }

    fprintf(out, ")");
}

// ===== END UNIFIED DISPLAY IMPLEMENTATION =====

#endif // __cplusplus
