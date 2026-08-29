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

#if defined(_WIN32)
#define ESHKOL_RUNTIME_WEAK __declspec(selectany)
#elif defined(__GNUC__)
#define ESHKOL_RUNTIME_WEAK __attribute__((weak))
#else
#define ESHKOL_RUNTIME_WEAK
#endif

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
    // SW-74: blocks ADOPTED from a promoted region arena (arena_adopt_blocks).
    // They are owned by this arena — freed by arena_destroy()/arena_reset() —
    // but they are deliberately NOT part of the allocation chain: allocation
    // only ever bumps `current_block`, and scope pops only ever walk the
    // current_block chain, so no later allocation can land inside an adopted
    // block and no rewind can free one early. That is precisely the guarantee
    // the bytecode VM's vm_evac_promote_all_blocks() gives when it splices a
    // pinned region's blocks behind the parent's bump block.
    arena_block_t* adopted_blocks;
    arena_scope_t* current_scope;  // Current scope
    size_t default_block_size;     // Default size for new blocks
    size_t total_allocated;        // Total memory allocated
    size_t alignment;              // Memory alignment requirement
    void* mutex;                   // Optional mutex for thread-safe access (platform-specific)
    bool thread_safe;              // Whether this arena uses mutex locking
    bool bounded;                  // ESH-0039/v1.8: if set, allocation NEVER grows
                                   // the arena (no new-block malloc); requests that
                                   // overflow the fixed capacity return NULL instead.
};

// Arena management functions
arena_t* arena_create(size_t default_block_size);
arena_t* arena_create_threadsafe(size_t default_block_size);  // Thread-safe variant with mutex

/* ESH-0039 / v1.8 embedded seam: bounded, no-grow arena.
 *
 * Creates an arena with a single fixed-capacity block. Allocation never mallocs
 * a new block: once the capacity is exhausted, arena_allocate* returns NULL.
 * This is the signature the v1.8 embedded / bounded-memory target needs so that
 * a region (or a worker subarena) can be given a hard memory ceiling. The hosted
 * implementation still malloc()s the single backing block once; a freestanding
 * build can swap the backing buffer without changing this contract. */
arena_t* arena_create_bounded(size_t capacity);

void arena_destroy(arena_t* arena);

// Thread-safety control
void arena_lock(arena_t* arena);
void arena_unlock(arena_t* arena);

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

// ESH-0214b: automatic per-iteration loop scope reclamation.
// arena_commit_scope discards the innermost scope record WITHOUT rewinding
// (all allocations since the matching push are kept); arena_top_scope_contains
// tests whether ptr points into memory allocated after the innermost scope
// mark; eshkol_arena_iter_scope_end pops the scope when none of vals[0..n)
// can point into it (bounded-RSS reclamation) and commits it otherwise
// (correctness fallback). See runtime_arena_core.cpp for full semantics.
void arena_commit_scope(arena_t* arena);

// SW-74: move every block owned by @p src into @p dst's adopted-block list.
//
// Zero-copy ownership transfer, the native analogue of the bytecode VM's
// vm_evac_promote_all_blocks() (lib/backend/vm_region_evac.c). @p src is left
// with no blocks at all, so the arena_destroy() that follows releases only its
// scopes, mutex and control struct. The moved blocks are never allocated from
// again (allocation only bumps dst->current_block) and are never rewound by a
// scope pop, so promotion cannot alias live data or be undone; they are freed
// when @p dst is destroyed or reset.
//
// Used by region_destroy() to promote a PINNED region — one a continuation
// that may outlive it was captured inside — into the arena that encloses it,
// instead of leaking the chain outright. Returns the number of bytes moved
// (0 if either arena is NULL, if they are the same arena, or if src holds
// nothing). dst's total_allocated absorbs the moved bytes, so the retention
// stays visible to ESHKOL_ARENA_REPORT=1 rather than disappearing from the
// accounting the way an outright leak did.
size_t arena_adopt_blocks(arena_t* dst, arena_t* src);
/** True if @p ptr points into memory allocated after the innermost scope
 *  mark on @p arena (i.e. it would be reclaimed if that scope were popped). */
int arena_top_scope_contains(const arena_t* arena, const void* ptr);
/** True if @p ptr points into ANY live allocation of @p arena (any block,
 *  below that block's high-water mark). Unlike arena_top_scope_contains this
 *  is scope-independent: it answers "is this address memory this arena owns
 *  and has handed out", which is the precondition for dereferencing an
 *  integer that MIGHT be an arena pointer. */
int arena_contains(const arena_t* arena, const void* ptr);
void eshkol_arena_iter_scope_end(arena_t* arena, const eshkol_tagged_value_t* vals, uint64_t n);

// Per-thread arena management (v1.2)
// Each thread gets its own arena for lock-free allocation during parallel execution.
// Worker arenas are merged into the parent arena when parallel tasks complete.

/** Get or create the current thread's arena.
 *  Returns the thread-local arena if in a worker thread, else the global arena. */
arena_t* arena_get_thread_local(void);

/** Create a new thread-local arena for the current thread.
 *  @param size_hint  Initial block size (0 = default 1MB) */
arena_t* arena_create_thread_local(size_t size_hint);

/** Merge all blocks from src arena into dest arena.
 *  After merge, src is empty (reset) but still valid for reuse.
 *  This is used to collect worker thread allocations into the main arena. */
void arena_merge_to_parent(arena_t* dest, arena_t* src);

/** Check if the current thread is a worker thread. */
int arena_is_worker_thread(void);

/** Initialise all thread-local runtime state for a worker thread.
 *  Eagerly creates the thread-local arena and touches AD tape stack / region
 *  stack TLS slots so they are ready before the first task executes.
 *  Safe to call multiple times (idempotent). */
void eshkol_thread_init_worker(size_t arena_size_hint);

/** Tear down all thread-local runtime state for a worker thread.
 *  Destroys the thread-local arena. Safe to call multiple times. */
void eshkol_thread_shutdown_worker(void);

// Statistics and debugging
size_t arena_get_used_memory(const arena_t* arena);
size_t arena_get_total_memory(const arena_t* arena);
size_t arena_get_block_count(const arena_t* arena);

/**
 * Legacy cons cell structure optimized for arena allocation, storing both
 * car and cdr as raw int64 (untagged). Superseded by arena_tagged_cons_cell_t
 * for new code that needs type information; kept for compatibility.
 */
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
ESHKOL_STATIC_ASSERT(sizeof(arena_tagged_cons_cell_t) == 32,
                     "Tagged cons cell must be exactly 32 bytes for optimal cache alignment");
ESHKOL_STATIC_ASSERT(sizeof(arena_cons_cell_t) == 16,
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

// Allocate symbol with object header (for symbol->string conversion)
// Returns pointer to symbol data (header is at offset -8)
void* arena_allocate_symbol_with_header(arena_t* arena, size_t length);

// Allocate closure with object header (for consolidated CALLABLE type)
// Returns pointer to closure data (header is at offset -8)
// name: bound procedure name from (define name ...) or NULL for anonymous lambdas
eshkol_closure_t* arena_allocate_closure_with_header(arena_t* arena, uint64_t func_ptr,
                                                      size_t num_captures, uint64_t sexpr_ptr,
                                                      uint64_t return_type_info,
                                                      const char* name);

// Convenience constructors
arena_tagged_cons_cell_t* arena_create_int64_cons(arena_t* arena,
                                                   int64_t car, uint8_t car_type,
                                                   int64_t cdr, uint8_t cdr_type);
arena_tagged_cons_cell_t* arena_create_mixed_cons(arena_t* arena,
                                                   eshkol_tagged_data_t car, uint8_t car_type,
                                                   eshkol_tagged_data_t cdr, uint8_t cdr_type);

// Type-safe data access functions
int64_t arena_tagged_cons_get_int64(const arena_tagged_cons_cell_t* cell, bool is_cdr);
/** Read the car (or cdr, if @p is_cdr) of @p cell reinterpreted as a double. */
double arena_tagged_cons_get_double(const arena_tagged_cons_cell_t* cell, bool is_cdr);
/** Read the car (or cdr, if @p is_cdr) of @p cell reinterpreted as a raw pointer value. */
uint64_t arena_tagged_cons_get_ptr(const arena_tagged_cons_cell_t* cell, bool is_cdr);

// Type-safe data setting functions
/** Set the car (or cdr, if @p is_cdr) of @p cell to an int64 @p value tagged with @p type. */
void arena_tagged_cons_set_int64(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                  int64_t value, uint8_t type);
/** Set the car (or cdr, if @p is_cdr) of @p cell to a double @p value tagged with @p type. */
void arena_tagged_cons_set_double(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                   double value, uint8_t type);
/** Set the car (or cdr, if @p is_cdr) of @p cell to a pointer @p value tagged with @p type. */
void arena_tagged_cons_set_ptr(arena_tagged_cons_cell_t* cell, bool is_cdr,
                                uint64_t value, uint8_t type);
/** Set the car (or cdr, if @p is_cdr) of @p cell to the null/empty-list value. */
void arena_tagged_cons_set_null(arena_tagged_cons_cell_t* cell, bool is_cdr);

// Type query functions
/** Get the type tag of the car (or cdr, if @p is_cdr) of @p cell. */
uint8_t arena_tagged_cons_get_type(const arena_tagged_cons_cell_t* cell, bool is_cdr);
/** Get the flags byte of the car (or cdr, if @p is_cdr) of @p cell. */
uint8_t arena_tagged_cons_get_flags(const arena_tagged_cons_cell_t* cell, bool is_cdr);
/** True if the car (or cdr, if @p is_cdr) of @p cell has the given @p type tag. */
bool arena_tagged_cons_is_type(const arena_tagged_cons_cell_t* cell, bool is_cdr, uint8_t type);

// Direct tagged value access functions (NEW in Phase 3B)
// These functions enable direct storage and retrieval of complete tagged_value structs
/** Store the complete tagged @p value into the car (or cdr, if @p is_cdr) of @p cell. */
void arena_tagged_cons_set_tagged_value(arena_tagged_cons_cell_t* cell,
                                         bool is_cdr,
                                         const eshkol_tagged_value_t* value);
/** Read the complete tagged value from the car (or cdr, if @p is_cdr) of @p cell. */
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

// The arena a tape-retained AD node must be allocated from: the recording
// tape's owner_arena when a tape is live, else the caller's current arena.
// An object the tape holds a pointer to lives exactly as long as the tape, so
// the three allocators below route through this instead of trusting the
// caller's (possibly nursery/region-scoped) arena. See the definition in
// runtime_autodiff.cpp for the full invariant and the ESH-0214e interaction.
arena_t* eshkol_ad_home_arena(arena_t* fallback);

// AD node allocation for computational graphs
ad_node_t* arena_allocate_ad_node(arena_t* arena);
ad_node_t* arena_allocate_ad_node_with_header(arena_t* arena);  // For consolidated CALLABLE type
ad_node_t* arena_allocate_ad_batch(arena_t* arena, size_t count);

// Runtime reverse-dispatch hook for AD_NODE_CUSTOM. The node's saved_tensors[0]
// is an eshkol_custom_vjp_t whose callback writes unscaled local partials; this
// helper applies the node upstream adjoint and accumulates into every input.
void eshkol_ad_node_custom_backward(void* node_ptr);

// Global tape pointer for AD operations (shared across JIT modules in REPL).
// Not thread_local due to cross-platform LLVM↔C TLS ABI constraints.
// Thread safety: AD tape stack (__ad_tape_stack) is thread_local.
extern ad_tape_t* __current_ad_tape;

// Global AD mode flag (shared across JIT modules in REPL).
extern bool __ad_mode_active;

/** Debug helper: logs the current AD mode (`__ad_mode_active`) to stderr,
 *  tagged with @p context. Has no effect on program behavior beyond the
 *  diagnostic print. */
void debug_print_ad_mode(const char* context);

/** Debug helper: logs @p ptr to stderr, tagged with @p context. Has no
 *  effect on program behavior beyond the diagnostic print. */
void debug_print_ptr(const char* context, void* ptr);

// Global shared arena for REPL mode (persistent across evaluations)
// NOTE: Actual type is std::atomic<arena_t*> — declared in C++ section below

// Global command-line arguments (for (command-line) procedure in REPL)
extern int32_t __eshkol_argc;
extern char** __eshkol_argv;

// Global arena for default allocations
extern arena_t* __global_arena;
arena_t* get_global_arena(void);

/** Get the global shared arena, bypassing per-thread override.
 *  Use when allocation MUST go into the shared arena (e.g. building result lists
 *  that will be returned to the main thread). */
arena_t* get_global_arena_shared(void);

// ===== OALR Phase A: thread memory context (ADR-0001, migration Phase A) =====
//
// A per-thread memory context makes "which arena do my allocations target right
// now" a thread-local property reached through an accessor, instead of a direct
// read of the process-shared __global_arena global. This is the DAG root the
// resident-concurrency line builds on. Phase A adds only the accessor and the
// allocation-domain slot; the full ABI-v2 memctx (region_top / residence /
// resident txn) and the 8->32B object-header change are DEFERRED to later OALR
// phases. Nothing here changes the object-header ABI.
#define ESHKOL_MEMORY_ABI_PHASE_A 1u

typedef struct eshkol_memctx {
    uint32_t abi_version;        // ESHKOL_MEMORY_ABI_PHASE_A
    uint32_t flags;              // reserved (0 in Phase A)
    uint64_t thread_id;          // owner-thread id, assigned lazily for diagnostics
    arena_t* allocation_domain;  // current allocation arena override; NULL => global
    void*    runtime_private;    // reserved for later phases
} eshkol_memctx_t;

/** Return the calling thread's memory context (never NULL; lazily initialized). */
eshkol_memctx_t* eshkol_memctx_current(void);

/**
 * Return the arena the calling thread's allocations should currently target.
 *
 * This is the Phase-A allocation accessor: generated code and runtime helpers
 * obtain the live arena here rather than loading __global_arena directly, so
 * `with-region` routing is a thread-local memctx update instead of a shared
 * write. While a work-stealing construct is active it defers to the shared
 * thread-safe process arena, preserving the #217 parallel-scope guard semantics
 * and keeping the accessor in agreement with any not-yet-migrated direct
 * __global_arena read.
 */
arena_t* eshkol_current_arena(void);

// Tape allocation and management
ad_tape_t* arena_allocate_tape(arena_t* arena, size_t initial_capacity);
void arena_tape_add_node(ad_tape_t* tape, ad_node_t* node);
void arena_tape_reset(ad_tape_t* tape);

// Tape query functions
ad_node_t* arena_tape_get_node(const ad_tape_t* tape, size_t index);
size_t arena_tape_get_node_count(const ad_tape_t* tape);

// ===== AD staged-kernel Phase A: instrumentation counters =====
// Machine-checkable proof that a vector/tensor gradient does ONE primal
// evaluation + ONE reverse sweep (not N of each). Only fields wired at real
// runtime sites are present; more may be added as later phases land. See
// docs/design/adr/0002-ad-staged-dense-kernels.md and lib/core/runtime_autodiff.cpp.
typedef struct {
    uint64_t primal_calls;             // user-function evaluations in a gradient
    uint64_t reverse_passes;           // backward sweeps actually executed
    uint64_t tape_allocations;         // reverse-mode tapes allocated
    uint64_t tape_nodes;               // AD nodes appended to tapes
    uint64_t finite_difference_evals;  // finite-difference evaluations on any AD path
} EshkolADCounters;

void eshkol_ad_counters_reset(void);
void eshkol_ad_counters_get(EshkolADCounters* out);
// Increment hooks invoked from emitted IR at the real event sites.
/** Increment the primal-call counter; invoked from emitted IR each time a
 *  user function is evaluated during a gradient computation. */
void eshkol_ad_count_primal(void);
/** Increment the reverse-pass counter; invoked from emitted IR each time a
 *  backward sweep is executed. */
void eshkol_ad_count_reverse(void);
/** Increment the finite-difference counter; invoked from emitted IR each
 *  time a finite-difference evaluation runs on an AD path. */
void eshkol_ad_count_fd(void);
// Individual readers (Scheme-builtin backends).
uint64_t eshkol_ad_counter_primal_calls(void);
uint64_t eshkol_ad_counter_reverse_passes(void);
uint64_t eshkol_ad_counter_tape_allocations(void);
uint64_t eshkol_ad_counter_tape_nodes(void);
/** Read the total count of finite-difference evaluations performed on any
 *  AD path since the last eshkol_ad_counters_reset(). */
uint64_t eshkol_ad_counter_finite_difference_evals(void);

// One-pass gradient support.
//   arena_tape_set_variables: populate the (formerly dead) ad_tape_t::variables /
//   num_variables so a single reverse sweep's per-input gradients can be read back
//   without replaying the loss per component.
void arena_tape_set_variables(ad_tape_t* tape, ad_node_t** vars, size_t n);
//   eshkol_ad_mixed_record_count: monotonic count of reverse-over-forward mixed
//   records. The one-pass gradient snapshots this around its single primal pass;
//   a nonzero delta means an inner forward-mode derivative ran (per-component seed
//   semantics are load-bearing) and the pass safely falls back to per-component replay.
uint64_t eshkol_ad_mixed_record_count(void);

// ===== ESH-0093: mixed-mode AD (reverse tape over inner forward derivative) =====
// A reverse-mode gradient pass publishes its active variable node here so an
// inner forward-mode derivative can seed it in the jet (e2) and record the
// mixed partial back onto the tape. See lib/core/runtime_autodiff.cpp.
void* eshkol_ad_seed_swap(void* node);           // publish; returns previous
double eshkol_ad_seed_flag(void* node);          // 1.0 iff node is the active seed
void* eshkol_ad_mixed_record(void* arena, void* tape, double value, double dseed);

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
    // ESH-0214c: the arena that outlives this region — where escaping values are
    // promoted to. Captured at region_push (BEFORE with-region codegen hijacks
    // the __global_arena slot to point at this region's arena), so it is the
    // TRUE enclosing arena: the parent region's arena when nested, or the real
    // process/global arena at top level. Must NOT be recomputed from
    // get_global_arena() at escape time, which by then returns this very region.
    arena_t* escape_base;
    // ESH-0214c: persistent forwarding map (old data ptr -> promoted copy) for
    // deep escape promotion, living for exactly this region's lifetime. Keys
    // reference memory in this region (or an enclosing one), values reference
    // fwd_target -- both strictly outlive the map, which region_destroy frees.
    // Persisting it across separate escapes preserves shared structure (two
    // stores of lists sharing a tail keep sharing it) and makes re-storing an
    // already-escaped object free. Opaque (std::unordered_map) -- only
    // runtime_regions.cpp touches it.
    void* fwd_map;
    arena_t* fwd_target;   // target arena the fwd_map entries were promoted into
    // #341: the arena displaced by this region's eshkol_region_enter (or the
    // REGION_NO_HIJACK sentinel when enter declined). `with-region` codegen keeps
    // this token in an SSA register and hands it back to eshkol_region_leave, but
    // a NON-LEXICAL close — a user-reachable region handle, or an unwind crossing
    // an open region — has no such register to read. Recording it on the region
    // makes the allocation-slot restore recoverable from the region stack alone,
    // which is what lets eshkol_region_unwind_to() close regions it did not open.
    arena_t* entry_saved_arena;
    // SW-59/SW-74: set by eshkol_region_pin_all() when a first-class
    // continuation that may outlive this region is captured while it is open.
    // Mirrors the bytecode VM's heap_region_pin_all() (lib/backend/vm_core.c):
    // a pinned region is never RECLAIMED, because a captured continuation's
    // raw stack snapshot (eshkol_continuation_capture_stack(),
    // runtime_continuations.cpp) may hold interior pointers into this region's
    // arena that are not walkable the way a with-region result value is.
    //
    // SW-74 changed what "never reclaimed" costs. It used to mean region_destroy()
    // skipped arena_destroy() and dropped the block chain on the floor — an
    // unowned, unbounded leak for the rest of the process. It now means the
    // block chain is PROMOTED into the enclosing arena (escape_base) via
    // arena_adopt_blocks(), exactly as the VM's vm_evac_promote_all_blocks()
    // splices a pinned region into its parent: the bytes are still not returned
    // at region exit, but they are owned, accounted, and released when the
    // enclosing scope ends. An escape-only capture — one the compiler proves
    // cannot outlive the frame that captured it, and which therefore cannot be
    // invoked after this region's dynamic extent ends — never sets this flag at
    // all (ESHKOL_CONT_FLAG_ESCAPE_ONLY, eshkol.h), so the overwhelmingly common
    // early-return idiom inside with-region reclaims in full.
    //
    // Never cleared once set: liveness of a first-class continuation is not
    // decidable without a tracing collector, so the pin lasts for the region's
    // remaining lifetime. See docs/reference/language/continuations.md for the
    // user-facing statement of the tradeoff.
    int pinned;
    // SW-74: 1 when this region was opened through the user-reachable region
    // HANDLE api (eshkol_region_handle_open with reclaim=1), 0 for a lexical
    // `with-region`.
    //
    // This is what makes the escape-only pin skip sound. An escape-only
    // continuation can only be invoked while the frame that captured it is
    // running, so it can outlive an enclosing region ONLY if that region can be
    // torn down from inside that frame. A lexical `with-region` cannot: its exit
    // is downstream of the `call/cc` it encloses, so by the time it runs the
    // continuation is already unreachable. A handle CAN: `(region-close h)` is
    // an ordinary call, so it can run inside the `call/cc` procedure and
    // cascade-close a region the capture is standing in (see
    // eshkol_region_handle_close(), runtime_regions.cpp). The other two
    // teardown routes — an exception unwind and a continuation invoke — both
    // destroy the capturing frame on their way past, so neither can strand a
    // live escape-only continuation.
    //
    // So: an escape-only capture skips the pin when every open region is
    // lexical, and pins exactly as before when any handle-owned region is open.
    uint8_t handle_owned;
};

// SW-74: non-zero when any region currently open on the calling thread's stack
// was opened through the region-handle API and can therefore be closed
// out-of-line, from anywhere, while a frame that captured a continuation inside
// it is still running. See eshkol_region_t::handle_owned for why an escape-only
// capture consults this before declining to pin.
int eshkol_region_any_handle_owned_open(void);

// SW-59: pin every region currently on the calling thread's region stack.
// Called once per continuation capture that may outlive its frame (see
// eshkol_make_continuation_state_flags(), runtime_continuations.cpp) whenever
// the stack is non-empty at capture time — exactly the condition the VM checks
// (`vm->heap.regions.depth > 0`) before its own heap_region_pin_all().
// Idempotent and safe to call with no regions open.
//
// Every OPEN region is pinned, not just the innermost: the continuation's stack
// snapshot may hold interior pointers into any of them, and a `call/cc` nested
// two `with-region`s deep can be re-entered after both have exited.
void eshkol_region_pin_all(void);

// Thread-local region stack (safe for parallel-map + with-region)
#define MAX_REGION_DEPTH 64
extern thread_local eshkol_region_t* __region_stack[MAX_REGION_DEPTH];
extern thread_local uint64_t __region_stack_depth;

// Thread-local AD tape stack for nested gradient operations (double-backward)
// MAX_TAPE_DEPTH must match runtime_autodiff.cpp and codegen_context.h
#define ESHKOL_ARENA_MAX_TAPE_DEPTH 32
extern thread_local ad_tape_t* __ad_tape_stack[ESHKOL_ARENA_MAX_TAPE_DEPTH];
extern thread_local uint64_t __ad_tape_depth;
extern thread_local uint64_t __ad_pert_level;  // ESH-0070 forward-mode perturbation level
// ESH-0190 Taylor-tower context. These are process globals rather than TLS;
// generated REPL modules resolve them through registerRuntimeSymbols().
extern uint64_t __ad_tower_active;
extern uint64_t __ad_tower_order;
extern thread_local void* __outer_ad_node_storage;
extern thread_local void* __outer_ad_node_to_inner;
extern thread_local void* __outer_grad_accumulator;
extern thread_local void* __inner_var_node_ptr;
extern thread_local uint64_t __gradient_x_degree;
extern thread_local void* __outer_ad_node_stack[ESHKOL_ARENA_MAX_TAPE_DEPTH];
extern thread_local uint64_t __outer_ad_node_depth;

// Region lifecycle functions
eshkol_region_t* region_create(const char* name, size_t size_hint);
void region_destroy(eshkol_region_t* region);

// Region stack management
void region_push(eshkol_region_t* region);
void region_pop(void);
eshkol_region_t* region_current(void);

// Thread-safe region allocation-scope routing (with-region codegen calls these
// around the body). eshkol_region_enter redirects the shared current-arena slot
// to the region's arena ONLY in single-threaded, non-parallel context (returns
// the displaced arena, or an opaque sentinel when it declines); eshkol_region_leave
// restores it. The parallel-scope guards suppress that hijack and pin the shared
// slot to the thread-safe process arena while a work-stealing construct
// (parallel-map/fold/execute/filter/for-each, async futures) may run on worker
// threads. See runtime_regions.cpp for the full rationale.
arena_t* eshkol_region_enter(eshkol_region_t* region);
void eshkol_region_leave(arena_t* saved);
void eshkol_parallel_scope_begin(void);
void eshkol_parallel_scope_end(void);

// ───────────────────────────────────────────────────────────────────────────
// #341: USER-REACHABLE REGION HANDLES — non-lexical scoped reclamation.
//
// `with-region` is the recommended default and stays unchanged: a lexical block
// cannot be left un-closed. But some loop shapes have no convenient lexical
// body — most concretely an autodiff training step, whose per-step AD tape the
// automatic per-iteration nursery (ESH-0214e) deliberately refuses to reclaim
// because a `gradient` op / `set!` / `tensor-set!` in the body disqualifies its
// static escape analysis. For those, a handle pair opens and closes the SAME
// region machinery with no lexical bracket:
//
//   (region-open [name] [size])  -> opaque exact-integer handle token
//   (region-close handle v ...)  -> deep-promoted v ... (region reclaimed)
//   (region-open? handle)        -> #t while the handle names a live region
//
// SAFETY. The token is a slot index plus a GENERATION counter, not a pointer:
// closing a handle bumps its slot's generation, so every stale token (double
// close, use after close, a fabricated integer) fails validation and raises a
// clean catchable error instead of touching freed memory. An out-of-order close
// (closing an outer handle while an inner one is live) is a DEFINED cascade: it
// closes the inner handles too, innermost first, and invalidates their tokens —
// the same operation an unwind performs, so there is exactly one teardown path.
// Kept values are deep-promoted level by level through the ESH-0214c/d escape
// evacuator (interior-pointer walk included), never shallow-copied.
//
// UNWIND. eshkol_region_unwind_to() is the teardown primitive shared by close,
// the exception path (eshkol_exception_handler_t::region_mark) and the
// continuation path (eshkol_continuation_state_t::region_mark). A raise or a
// call/cc escape crossing an open region closes it and promotes the in-flight
// value out first. Because it works off the region stack rather than the handle
// table, `with-region` gets the same guarantee (before this, a raise out of a
// with-region body leaked the region AND left the allocation slot hijacked).
// ───────────────────────────────────────────────────────────────────────────

// Maximum simultaneously-open user handles per thread. Bounded by the region
// stack itself: every open handle holds one region-stack entry.
#define ESHKOL_MAX_REGION_HANDLES MAX_REGION_DEPTH

// Operation status. The core handle operations NEVER raise: they return a status
// and the caller raises through its own substrate's mechanism
// (eshkol_runtime_fatal on native, vm_raise_error_msg in the bytecode VM) using
// the shared eshkol_region_handle_status_message() text. That is what makes a
// `guard` around a misused handle observe byte-identical output on both
// substrates — the same contract the bytevector bounds checks use.
#define ESHKOL_RH_OK           0  // success
#define ESHKOL_RH_ERR_STALE    1  // invalid, already-closed, or foreign-thread token
#define ESHKOL_RH_ERR_NOT_LIVE 2  // token decoded but its region left the stack
#define ESHKOL_RH_ERR_TOO_MANY 3  // handle table exhausted (a close is missing)
#define ESHKOL_RH_ERR_DEPTH    4  // region stack exhausted (a close is missing)
#define ESHKOL_RH_ERR_CREATE   5  // region could not be created/activated

// The canonical message for a status code. Identical on every substrate.
const char* eshkol_region_handle_status_message(int status);

// Open a region and return its handle token (>0), or 0 with *status set on
// failure. `name` may be NULL; `size_hint` 0 selects the default. `reclaim`
// selects the substrate contract: non-zero performs the real
// region_create/region_push/eshkol_region_enter (native), zero records a
// bookkeeping-only handle whose close reclaims nothing (the bytecode VM, whose
// handle surface is Stage-2 — its `with-region` reclaims through
// lib/backend/vm_region_evac.c; see tests/vm_parity/PARITY.tsv).
int64_t eshkol_region_handle_open(const char* name, uint64_t size_hint, int reclaim,
                                  int* status);

// Close the handle named by `token`, deep-promoting vals[0..n) out of every
// region it closes (in place). Returns an ESHKOL_RH_* status; never raises.
int eshkol_region_handle_close(int64_t token, eshkol_tagged_value_t* vals, uint64_t n);

// Non-raising liveness probe: 1 while `token` names a currently-open handle.
int eshkol_region_handle_live(int64_t token);

// Unwind mark / unwind for substrates with NO region stack (the bytecode VM,
// whose handles are bookkeeping-only). Native handles are retired by
// eshkol_region_unwind_to() off the region stack instead; these use a monotonic
// open-sequence number, so the two never interfere. The VM records the mark on
// its handler frame and calls the unwind from vm_dispatch_exception, giving a
// raise the same observable effect on handle liveness as on native.
uint64_t eshkol_region_handle_seq_mark(void);
void eshkol_region_handle_seq_unwind_to(uint64_t mark);

// Region-stack depth, the mark form used by the exception/continuation records.
uint64_t eshkol_region_mark(void);

// Close every region opened after `mark`, innermost first, deep-promoting
// vals[0..n) out of each level as it goes, restoring the allocation slot from
// each region's recorded entry_saved_arena, and invalidating the tokens of any
// handles that named the closed regions. Safe (a no-op) when the stack is
// already at or below `mark`.
void eshkol_region_unwind_to(uint64_t mark, eshkol_tagged_value_t* vals, uint64_t n);

// Close every region entered since a continuation was captured, promoting the
// value it delivers out of them first. Reads the mark from the continuation
// state itself (eshkol_continuation_state_t::region_mark), so the invoke path
// passes only the state pointer — the same shape as eshkol_unwind_dynamic_wind
// and eshkol_promise_eval_unwind_to next to which it is called.
void eshkol_region_unwind_for_continuation(void* state);

// Surface entry points, shared verbatim by native codegen and the bytecode VM so
// arity coercions and error text cannot diverge. Optional arguments are passed
// as NULL pointers rather than sentinels.
//   (region-open)          -> a = b = NULL
//   (region-open n)        -> a = n, b = NULL   (numeric a = size hint, else name)
//   (region-open 'nm n)    -> a = 'nm, b = n
void eshkol_region_open_builtin(eshkol_tagged_value_t* out,
                               const eshkol_tagged_value_t* a,
                               const eshkol_tagged_value_t* b,
                               int reclaim);
// (region-close handle v ...) -> the promoted v for one keep, a list for several,
// '() for none. `vals` is promoted in place before the region is reclaimed.
void eshkol_region_close_builtin(eshkol_tagged_value_t* out,
                                 const eshkol_tagged_value_t* handle,
                                 eshkol_tagged_value_t* vals,
                                 uint64_t n);
// (region-open? handle) -> #t while the handle names a live region. Never raises.
void eshkol_region_open_p_builtin(eshkol_tagged_value_t* out,
                                  const eshkol_tagged_value_t* handle);

// Region allocation - allocates in the current region
void* region_allocate(size_t size);
void* region_allocate_aligned(size_t size, size_t alignment);
void* region_allocate_zeroed(size_t size);

// Region-aware cons cell allocation
arena_tagged_cons_cell_t* region_allocate_tagged_cons_cell(void);

// Region escape - copy value from current region to parent (or global arena)
// Returns pointer to the escaped copy. Increments region->escape_count.
void* region_escape(const void* ptr, size_t size);
void* region_escape_string(const char* str);
arena_tagged_cons_cell_t* region_escape_tagged_cons_cell(const arena_tagged_cons_cell_t* cell);
eshkol_tagged_value_t region_escape_tagged_value(eshkol_tagged_value_t val);
void region_escape_tagged_value_into(eshkol_tagged_value_t* out, const eshkol_tagged_value_t* val);

// Promote loop-carried values out of an automatic iteration nursery, then
// recycle that nursery at the tail-call back edge.  The JIT registers this
// explicitly because static host binaries do not expose it through dlsym on
// every supported platform.
void eshkol_iter_nursery_recycle(eshkol_region_t* region,
                                 eshkol_tagged_value_t* vals,
                                 uint64_t n);

// Region write barrier (ESH-0214c): promote a value's in-region subgraph when it
// is stored (by set-car!/set-cdr!/vector-set!/hash-table-set!/global set!) into a
// destination that outlives the value's region. Fast path (no active region) is a
// single thread-local load + branch. See runtime_regions.cpp for full semantics.
void eshkol_region_write_barrier_into(eshkol_tagged_value_t* out,
                                      const void* dst,
                                      const eshkol_tagged_value_t* value);
// Range form for bulk copies (vector-copy!): promotes each copied slot in
// place. Fast path (no region) is a single thread-local load + branch.
void eshkol_region_write_barrier_range(const void* dst,
                                       eshkol_tagged_value_t* slots,
                                       uint64_t n);

// Representation-aware vector mutation. Eshkol exposes both Scheme vectors
// (inline tagged slots) and numeric tensor-backed #(...) literals through the
// R7RS vector API; vector-copy! must therefore bridge both layouts safely.
typedef enum eshkol_vector_copy_status {
    ESHKOL_VECTOR_COPY_OK = 0,
    ESHKOL_VECTOR_COPY_NULL = 1,
    ESHKOL_VECTOR_COPY_BOUNDS = 2,
    ESHKOL_VECTOR_COPY_TYPE = 3
} eshkol_vector_copy_status_t;
int32_t eshkol_vector_copy_mutating(void* dst, int64_t at,
                                    const void* src, int64_t start, int64_t end);

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
// name: bound procedure name from (define name ...) or NULL for anonymous lambdas
eshkol_closure_t* arena_allocate_closure(arena_t* arena, uint64_t func_ptr, size_t num_captures,
                                         uint64_t sexpr_ptr, uint64_t return_type_info,
                                         const char* name);

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
ESHKOL_STATIC_ASSERT(sizeof(eshkol_shared_header_t) == 24,
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

// Tensor element dtype tags (ESH-0020). Storage is always f64 bit patterns in
// `elements`; the dtype records the logical element precision so matmul/cast can
// dispatch and so casting can apply the appropriate precision reduction. f64 is
// the default (0) so existing tensors and all existing code are unaffected.
typedef enum eshkol_tensor_dtype {
    ESHKOL_TENSOR_DTYPE_F64  = 0,  // double (default)
    ESHKOL_TENSOR_DTYPE_F32  = 1,  // IEEE single
    ESHKOL_TENSOR_DTYPE_F16  = 2,  // IEEE half
    ESHKOL_TENSOR_DTYPE_BF16 = 3,  // bfloat16
    ESHKOL_TENSOR_DTYPE_I8   = 4,  // signed 8-bit integer
    // ESH-0121 (matmul-reshape Hessian): sentinel marking a "dual tensor" whose
    // `elements` array holds 16-byte tagged DUAL_NUMBER jets (not f64 bit
    // patterns). Produced by reshape of a Scheme vector of forward-mode duals
    // during the Hessian's forward-over-forward sweep and consumed by the
    // dual-aware matmul/tensor-sum paths so second-order terms are not dropped.
    // Well above the real precision codes so no numeric kernel misreads it.
    ESHKOL_TENSOR_DTYPE_DUAL = 64  // elements are tagged DUAL_NUMBER values
} eshkol_tensor_dtype_t;

// Tensor structure for multi-dimensional arrays
// Must match LLVM TypeSystem tensor_type layout:
// Fields are all 8 bytes for natural alignment (40 bytes total)
// NOTE: elements stored as int64_t bit patterns of doubles for compatibility
typedef struct eshkol_tensor {
    uint64_t* dimensions;     // idx 0: Pointer to dimension sizes array
    uint64_t  num_dimensions; // idx 1: Number of dimensions (rank)
    int64_t*  elements;       // idx 2: Element data (doubles stored as int64 bits)
    uint64_t  total_elements; // idx 3: Product of all dimensions
    uint64_t  dtype;          // idx 4: eshkol_tensor_dtype_t (0 = f64 default)
} eshkol_tensor_t;

// Compile-time size validation. Field indices 0-3 are unchanged from the
// original 32-byte layout, so every existing GEP into a tensor remains valid;
// dtype is appended at idx 4.
ESHKOL_STATIC_ASSERT(sizeof(eshkol_tensor_t) == 40,
                     "Tensor struct must be 40 bytes (4 core fields + dtype)");

// Allocate tensor with object header (for consolidated HEAP_PTR type)
// Returns pointer to tensor data (header is at offset -8)
// Does NOT allocate dims or elements arrays - caller must allocate separately
eshkol_tensor_t* arena_allocate_tensor_with_header(arena_t* arena);

// Allocate tensor with dimensions and elements arrays in one call
// Returns fully initialized tensor with dims and elements arrays allocated
eshkol_tensor_t* arena_allocate_tensor_full(arena_t* arena, uint64_t num_dims, uint64_t total_elements);

// Apply the logical tensor dtype's precision to one f64 value. Storage remains
// f64; this is the scalar companion to eshkol_tensor_apply_dtype().
double eshkol_tensor_reduce_precision_value(double value, int64_t dtype);

// Extract tensor elements (double bitpatterns) as int64 dimension values
int64_t eshkol_tensor_to_dims(const void* tensor_ptr, int64_t* dims_out, int64_t max_dims);

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
    arena_t* home_arena;                  // ESH-0039: arena the table was created in.
                                          // Resize re-allocates the backing arrays here
                                          // (NOT the transient region arena that happens
                                          // to be active during a set!) so a table created
                                          // outside a (with-region ...) survives region_pop.
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

#include <atomic>

// Global shared arena for REPL mode (persistent across evaluations)
// Atomic to synchronize writes (REPL init) and reads (runtime exception handlers)
extern "C" std::atomic<arena_t*> __repl_shared_arena;

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
