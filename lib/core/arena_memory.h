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

// ─────────────────────────────────────────────────────────────────────────────
// Header-aware allocation functions
// These allocate objects with an eshkol_object_header_t prepended.
// The returned pointer points to the DATA, not the header.
// Use ESHKOL_GET_HEADER(ptr) to access the header from the data pointer.
// Memory layout: [header (8 bytes)][object data (variable)]
// ─────────────────────────────────────────────────────────────────────────────
void* arena_allocate_with_header(arena_t* arena, size_t data_size, uint8_t subtype, uint8_t flags);
void* arena_allocate_with_header_zeroed(arena_t* arena, size_t data_size, uint8_t subtype, uint8_t flags);

// Typed allocation helpers for new consolidated types
void* arena_allocate_multi_value(arena_t* arena, size_t count);

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

// Allocate cons cell with object header (for consolidated HEAP_PTR type)
arena_tagged_cons_cell_t* arena_allocate_cons_with_header(arena_t* arena);

// Allocate string with object header (for consolidated HEAP_PTR type)
// Returns pointer to string data (header is at offset -8)
char* arena_allocate_string_with_header(arena_t* arena, size_t length);

// Allocate vector with object header (for consolidated HEAP_PTR type)
// Returns pointer to vector data (header is at offset -8)
void* arena_allocate_vector_with_header(arena_t* arena, size_t capacity);

// Allocate closure with object header (for consolidated CALLABLE type)
// Returns pointer to closure data (header is at offset -8)
eshkol_closure_t* arena_allocate_closure_with_header(arena_t* arena, uint64_t func_ptr,
                                                      size_t num_captures, uint64_t sexpr_ptr,
                                                      uint64_t return_type_info);

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
uint8_t arena_tagged_cons_get_flags(const arena_tagged_cons_cell_t* cell, bool is_cdr);
bool arena_tagged_cons_is_type(const arena_tagged_cons_cell_t* cell, bool is_cdr, uint8_t type);

// Direct tagged value access functions (NEW in Phase 3B)
// These functions enable direct storage and retrieval of complete tagged_value structs
void arena_tagged_cons_set_tagged_value(arena_tagged_cons_cell_t* cell,
                                         bool is_cdr,
                                         const eshkol_tagged_value_t* value);
eshkol_tagged_value_t arena_tagged_cons_get_tagged_value(const arena_tagged_cons_cell_t* cell,
                                                          bool is_cdr);

// ===== DEEP EQUALITY COMPARISON =====
// Runtime helper for deep structural equality of tagged values
// Used by equal? to compare nested lists recursively
// Takes pointers to avoid struct-by-value ABI issues
bool eshkol_deep_equal(const eshkol_tagged_value_t* val1, const eshkol_tagged_value_t* val2);

// ===== AD MEMORY MANAGEMENT =====
// Allocation functions for automatic differentiation structures

// Dual number allocation
eshkol_dual_number_t* arena_allocate_dual_number(arena_t* arena);
eshkol_dual_number_t* arena_allocate_dual_batch(arena_t* arena, size_t count);

// AD node allocation for computational graphs
ad_node_t* arena_allocate_ad_node(arena_t* arena);
ad_node_t* arena_allocate_ad_node_with_header(arena_t* arena);  // For consolidated CALLABLE type
ad_node_t* arena_allocate_ad_batch(arena_t* arena, size_t count);

// Global tape pointer for AD operations (shared across JIT modules in REPL)
extern ad_tape_t* __current_ad_tape;

// Global AD mode flag (shared across JIT modules in REPL)
// CRITICAL: This must be shared so lambdas from one module can see AD mode set by another
extern bool __ad_mode_active;

// Debug helper to print AD mode state
void debug_print_ad_mode(const char* context);

// Debug helper to print pointer value
void debug_print_ptr(const char* context, void* ptr);

// Global shared arena for REPL mode (persistent across evaluations)
extern arena_t* __repl_shared_arena;

// Tape allocation and management
ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity);
void arena_tape_add_node(ad_tape_t* tape, ad_node_t* node);
void arena_tape_reset(ad_tape_t* tape);

// Tape query functions
ad_node_t* arena_tape_get_node(const ad_tape_t* tape, size_t index);
size_t arena_tape_get_node_count(const ad_tape_t* tape);

// ===== OALR (Ownership-Aware Lexical Regions) MEMORY MANAGEMENT =====
// Region-based memory management for predictable, GC-free allocation

// Forward declaration
typedef struct eshkol_region eshkol_region_t;

// Region structure - wraps an arena with lexical scoping
struct eshkol_region {
    arena_t* arena;                  // Region's dedicated arena
    const char* name;                // Optional name (NULL for anonymous)
    eshkol_region_t* parent;         // Parent region (for nesting)
    size_t size_hint;                // Size hint provided at creation
    size_t escape_count;             // Track escaping allocations
    uint8_t is_active;               // Whether this region is currently active
};

// Global region stack
#define MAX_REGION_DEPTH 64
extern eshkol_region_t* __region_stack[MAX_REGION_DEPTH];
extern uint64_t __region_stack_depth;

// Region lifecycle functions
eshkol_region_t* region_create(const char* name, size_t size_hint);
void region_destroy(eshkol_region_t* region);

// Region stack management
void region_push(eshkol_region_t* region);
void region_pop(void);
eshkol_region_t* region_current(void);

// Region allocation - allocates in the current region
void* region_allocate(size_t size);
void* region_allocate_aligned(size_t size, size_t alignment);
void* region_allocate_zeroed(size_t size);

// Region-aware cons cell allocation
arena_tagged_cons_cell_t* region_allocate_tagged_cons_cell(void);

// Region statistics
size_t region_get_used_memory(const eshkol_region_t* region);
size_t region_get_total_memory(const eshkol_region_t* region);
const char* region_get_name(const eshkol_region_t* region);
uint64_t region_get_depth(void);

// ===== CLOSURE ENVIRONMENT MEMORY MANAGEMENT =====
// Allocation functions for lexical closure environments

// Allocate closure environment with space for captured variables
eshkol_closure_env_t* arena_allocate_closure_env(arena_t* arena, size_t num_captures);

// Allocate full closure structure (func_ptr + environment + sexpr for homoiconicity)
// return_type_info: packed return type metadata (return_type | (input_arity << 8) | (hott_type_id << 16))
eshkol_closure_t* arena_allocate_closure(arena_t* arena, uint64_t func_ptr, size_t num_captures,
                                         uint64_t sexpr_ptr, uint64_t return_type_info);

// ===== END CLOSURE ENVIRONMENT MEMORY MANAGEMENT =====

// ===== SHARED (REFERENCE-COUNTED) MEMORY MANAGEMENT =====
// Reference-counted allocation for values with complex, dynamic lifetimes

// Shared header structure - prepended to all shared allocations
// Layout optimized for 64-bit: 24 bytes with natural alignment
typedef struct eshkol_shared_header {
    void (*destructor)(void*);      // Custom cleanup function (NULL if none) - 8 bytes
    uint32_t ref_count;             // Strong reference count - 4 bytes
    uint32_t weak_count;            // Weak reference count - 4 bytes
    uint8_t flags;                  // Flags (e.g., marked for collection)
    uint8_t value_type;             // Type of the shared value
    uint16_t reserved;              // Alignment padding
    uint32_t reserved2;             // Padding to 24 bytes
} eshkol_shared_header_t;

// Weak reference structure - points to shared data
typedef struct eshkol_weak_ref {
    eshkol_shared_header_t* header; // Pointer to shared header (NULL if deallocated)
    void* data;                     // Original data pointer (may be invalid)
} eshkol_weak_ref_t;

// Compile-time size validation
_Static_assert(sizeof(eshkol_shared_header_t) == 24,
               "Shared header must be 24 bytes for optimal alignment");

// Shared allocation functions
void* shared_allocate(size_t size, void (*destructor)(void*));
void* shared_allocate_typed(size_t size, uint8_t value_type, void (*destructor)(void*));

// Reference counting operations
void shared_retain(void* ptr);      // Increment ref count
void shared_release(void* ptr);     // Decrement ref count (deallocates at zero)
uint32_t shared_ref_count(void* ptr);  // Get current ref count (for debugging)

// Weak reference operations
eshkol_weak_ref_t* weak_ref_create(void* shared_ptr);   // Create weak ref to shared value
void* weak_ref_upgrade(eshkol_weak_ref_t* weak);        // Upgrade to strong ref (returns NULL if freed)
void weak_ref_release(eshkol_weak_ref_t* weak);         // Release the weak reference
bool weak_ref_is_alive(eshkol_weak_ref_t* weak);        // Check if target still exists

// Get the shared header from a shared pointer
eshkol_shared_header_t* shared_get_header(void* ptr);

// ===== END SHARED MEMORY MANAGEMENT =====

// ===== TENSOR MEMORY MANAGEMENT =====
// N-dimensional numeric tensor with arena allocation

// Tensor structure for multi-dimensional arrays
// Must match LLVM TypeSystem tensor_type layout:
// Fields are all 8 bytes for natural alignment (32 bytes total)
// NOTE: elements stored as int64_t bit patterns of doubles for compatibility
typedef struct eshkol_tensor {
    uint64_t* dimensions;     // idx 0: Pointer to dimension sizes array
    uint64_t  num_dimensions; // idx 1: Number of dimensions (rank)
    int64_t*  elements;       // idx 2: Element data (doubles stored as int64 bits)
    uint64_t  total_elements; // idx 3: Product of all dimensions
} eshkol_tensor_t;

// Compile-time size validation
_Static_assert(sizeof(eshkol_tensor_t) == 32,
               "Tensor struct must be 32 bytes for optimal alignment");

// Allocate tensor with object header (for consolidated HEAP_PTR type)
// Returns pointer to tensor data (header is at offset -8)
// Does NOT allocate dims or elements arrays - caller must allocate separately
eshkol_tensor_t* arena_allocate_tensor_with_header(arena_t* arena);

// Allocate tensor with dimensions and elements arrays in one call
// Returns fully initialized tensor with dims and elements arrays allocated
eshkol_tensor_t* arena_allocate_tensor_full(arena_t* arena, uint64_t num_dims, uint64_t total_elements);

// ===== END TENSOR MEMORY MANAGEMENT =====

// ===== HASH TABLE MEMORY MANAGEMENT =====
// Open-addressing hash table with linear probing for O(1) average lookup

// Hash table entry status
typedef enum {
    HASH_ENTRY_EMPTY = 0,     // Slot never used
    HASH_ENTRY_OCCUPIED = 1,  // Slot contains valid key-value pair
    HASH_ENTRY_DELETED = 2    // Slot was deleted (tombstone)
} hash_entry_status_t;

// Hash table structure
// Uses open addressing with linear probing for simplicity and cache efficiency
typedef struct eshkol_hash_table {
    size_t capacity;                      // Number of buckets
    size_t size;                          // Number of stored entries
    size_t tombstones;                    // Number of deleted entries (for rehashing decisions)
    eshkol_tagged_value_t* keys;          // Array of keys (tagged values)
    eshkol_tagged_value_t* values;        // Array of values (tagged values)
    uint8_t* status;                      // Entry status array (EMPTY/OCCUPIED/DELETED)
} eshkol_hash_table_t;

// Initial capacity for new hash tables
#define HASH_TABLE_INITIAL_CAPACITY 16

// Load factor threshold for rehashing (0.75 = 75%)
#define HASH_TABLE_LOAD_FACTOR 0.75

// Hash table allocation and creation
eshkol_hash_table_t* arena_allocate_hash_table(arena_t* arena, size_t initial_capacity);
eshkol_hash_table_t* arena_hash_table_create(arena_t* arena);
eshkol_hash_table_t* arena_hash_table_create_with_header(arena_t* arena);  // With object header for HEAP_PTR type

// Hash table operations
bool hash_table_set(arena_t* arena, eshkol_hash_table_t* table,
                    const eshkol_tagged_value_t* key, const eshkol_tagged_value_t* value);
bool hash_table_get(const eshkol_hash_table_t* table,
                    const eshkol_tagged_value_t* key, eshkol_tagged_value_t* out_value);
bool hash_table_has_key(const eshkol_hash_table_t* table, const eshkol_tagged_value_t* key);
bool hash_table_remove(eshkol_hash_table_t* table, const eshkol_tagged_value_t* key);
void hash_table_clear(eshkol_hash_table_t* table);
size_t hash_table_count(const eshkol_hash_table_t* table);

// Hash table iteration (returns arena-allocated list of keys/values)
arena_tagged_cons_cell_t* hash_table_keys(arena_t* arena, const eshkol_hash_table_t* table);
arena_tagged_cons_cell_t* hash_table_values(arena_t* arena, const eshkol_hash_table_t* table);

// Hash function for tagged values
uint64_t hash_tagged_value(const eshkol_tagged_value_t* value);

// Equality comparison for hash keys
bool hash_keys_equal(const eshkol_tagged_value_t* a, const eshkol_tagged_value_t* b);

// ===== END HASH TABLE MEMORY MANAGEMENT =====

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
