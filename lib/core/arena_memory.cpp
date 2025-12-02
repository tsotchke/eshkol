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
#endif // __cplusplus
