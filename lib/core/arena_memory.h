/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Arena Memory Management System for Eshkol List Operations
 * 
 * This system provides stack-based memory management for Scheme list operations,
 * using LLVM's CreateAlloca for automatic cleanup and GC-free memory management.
 */

#ifndef ESHKOL_ARENA_MEMORY_H
#define ESHKOL_ARENA_MEMORY_H

#include <stdint.h>
#include <stddef.h>

// Include main Eshkol header for tagged data types
#include "../../inc/eshkol/eshkol.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct arena_block arena_block_t;
typedef struct arena arena_t;
typedef struct arena_scope arena_scope_t;

// Arena block structure for linked list of memory blocks
struct arena_block {
    uint8_t* memory;        // Block memory
    size_t size;           // Total block size
    size_t used;           // Used bytes in this block
    arena_block_t* next;   // Next block in chain
};

// Arena scope for nested allocation contexts
struct arena_scope {
    arena_block_t* block;  // Block at scope start
    size_t used;           // Used bytes at scope start
    arena_scope_t* parent; // Parent scope
};

// Main Arena structure
struct arena {
    arena_block_t* current_block;  // Current allocation block
    arena_scope_t* current_scope;  // Current scope
    size_t default_block_size;     // Default size for new blocks
    size_t total_allocated;        // Total memory allocated
    size_t alignment;              // Memory alignment requirement
};

// Arena management functions
arena_t* arena_create(size_t default_block_size);
void arena_destroy(arena_t* arena);

// Memory allocation
void* arena_allocate(arena_t* arena, size_t size);
void* arena_allocate_aligned(arena_t* arena, size_t size, size_t alignment);
void* arena_allocate_zeroed(arena_t* arena, size_t size);

// Scope management
void arena_push_scope(arena_t* arena);
void arena_pop_scope(arena_t* arena);
void arena_reset(arena_t* arena);

// Statistics and debugging
size_t arena_get_used_memory(const arena_t* arena);
size_t arena_get_total_memory(const arena_t* arena);
size_t arena_get_block_count(const arena_t* arena);

// Cons cell structure optimized for arena allocation
typedef struct arena_cons_cell {
    int64_t car;           // Car value (as int64 for compatibility)
    int64_t cdr;           // Cdr value (as int64 for compatibility)
} arena_cons_cell_t;

// Enhanced cons cell structure with type information
// Phase 3B: Now stores complete eshkol_tagged_value_t for direct tagged value storage
typedef struct arena_tagged_cons_cell {
    eshkol_tagged_value_t car;  // 16 bytes - Complete tagged value with type+flags+data
    eshkol_tagged_value_t cdr;  // 16 bytes - Complete tagged value with type+flags+data
} arena_tagged_cons_cell_t;     // Total: 32 bytes (perfect cache alignment!)

// Compile-time size validation
_Static_assert(sizeof(arena_tagged_cons_cell_t) == 32,
               "Tagged cons cell must be exactly 32 bytes for optimal cache alignment");
_Static_assert(sizeof(arena_cons_cell_t) == 16,
               "Legacy cons cell size changed unexpectedly");

// List-specific allocation functions
arena_cons_cell_t* arena_allocate_cons_cell(arena_t* arena);
void* arena_allocate_list_node(arena_t* arena, size_t element_size, size_t count);

// Tagged cons cell allocation functions
arena_tagged_cons_cell_t* arena_allocate_tagged_cons_cell(arena_t* arena);
arena_tagged_cons_cell_t* arena_allocate_tagged_cons_batch(arena_t* arena, size_t count);

// Convenience constructors
arena_tagged_cons_cell_t* arena_create_int64_cons(arena_t* arena,
                                                   int64_t car, uint8_t car_type,
                                                   int64_t cdr, uint8_t cdr_type);
arena_tagged_cons_cell_t* arena_create_mixed_cons(arena_t* arena,
                                                   eshkol_tagged_data_t car, uint8_t car_type,
                                                   eshkol_tagged_data_t cdr, uint8_t cdr_type);

// Type-safe data access functions
int64_t arena_tagged_cons_get_int64(const arena_tagged_cons_cell_t* cell, bool is_cdr);
double arena_tagged_cons_get_double(const arena_tagged_cons_cell_t* cell, bool is_cdr);
uint64_t arena_tagged_cons_get_ptr(const arena_tagged_cons_cell_t* cell, bool is_cdr);

// Type-safe data setting functions
void arena_tagged_cons_set_int64(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                  int64_t value, uint8_t type);
void arena_tagged_cons_set_double(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                   double value, uint8_t type);
void arena_tagged_cons_set_ptr(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                uint64_t value, uint8_t type);
void arena_tagged_cons_set_null(arena_tagged_cons_cell_t* cell, bool is_cdr);

// Type query functions
uint8_t arena_tagged_cons_get_type(const arena_tagged_cons_cell_t* cell, bool is_cdr);
bool arena_tagged_cons_is_type(const arena_tagged_cons_cell_t* cell, bool is_cdr, uint8_t type);

// Direct tagged value access functions (NEW in Phase 3B)
// These functions enable direct storage and retrieval of complete tagged_value structs
void arena_tagged_cons_set_tagged_value(arena_tagged_cons_cell_t* cell,
                                         bool is_cdr,
                                         const eshkol_tagged_value_t* value);
eshkol_tagged_value_t arena_tagged_cons_get_tagged_value(const arena_tagged_cons_cell_t* cell,
                                                          bool is_cdr);

// ===== AD MEMORY MANAGEMENT =====
// Allocation functions for automatic differentiation structures

// Dual number allocation
eshkol_dual_number_t* arena_allocate_dual_number(arena_t* arena);
eshkol_dual_number_t* arena_allocate_dual_batch(arena_t* arena, size_t count);

// AD node allocation for computational graphs
ad_node_t* arena_allocate_ad_node(arena_t* arena);
ad_node_t* arena_allocate_ad_batch(arena_t* arena, size_t count);

// Tape allocation and management
ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity);
void arena_tape_add_node(ad_tape_t* tape, ad_node_t* node);
void arena_tape_reset(ad_tape_t* tape);

// Tape query functions
ad_node_t* arena_tape_get_node(const ad_tape_t* tape, size_t index);
size_t arena_tape_get_node_count(const ad_tape_t* tape);

// ===== CLOSURE ENVIRONMENT MEMORY MANAGEMENT =====
// Allocation functions for lexical closure environments

// Allocate closure environment with space for captured variables
eshkol_closure_env_t* arena_allocate_closure_env(arena_t* arena, size_t num_captures);

// ===== END CLOSURE ENVIRONMENT MEMORY MANAGEMENT =====

// ===== END AD MEMORY MANAGEMENT =====

#ifdef __cplusplus
} // extern "C"

// C++ Arena wrapper class for RAII
class Arena {
private:
    arena_t* arena_;

public:
    explicit Arena(size_t default_block_size = 8192);
    ~Arena();
    
    // Non-copyable
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    
    // Movable
    Arena(Arena&& other) noexcept;
    Arena& operator=(Arena&& other) noexcept;
    
    // Allocation
    void* allocate(size_t size);
    void* allocate_aligned(size_t size, size_t alignment);
    void* allocate_zeroed(size_t size);
    
    template<typename T>
    T* allocate() {
        return static_cast<T*>(allocate_aligned(sizeof(T), alignof(T)));
    }
    
    template<typename T>
    T* allocate_array(size_t count) {
        return static_cast<T*>(allocate_aligned(sizeof(T) * count, alignof(T)));
    }
    
    // Scope management
    class Scope {
    private:
        Arena* arena_;
        bool active_;
        
    public:
        explicit Scope(Arena& arena);
        ~Scope();
        
        // Non-copyable, non-movable
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&&) = delete;
        Scope& operator=(Scope&&) = delete;
    };
    
    // Statistics
    size_t get_used_memory() const;
    size_t get_total_memory() const;
    size_t get_block_count() const;
    
    // Reset arena (clear all memory)
    void reset();
    
    // Get underlying C arena
    arena_t* get_arena() const { return arena_; }
};

#endif // __cplusplus

#endif // ESHKOL_ARENA_MEMORY_H
