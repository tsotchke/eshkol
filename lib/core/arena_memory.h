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

// List-specific allocation functions
arena_cons_cell_t* arena_allocate_cons_cell(arena_t* arena);
void* arena_allocate_list_node(arena_t* arena, size_t element_size, size_t count);

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
