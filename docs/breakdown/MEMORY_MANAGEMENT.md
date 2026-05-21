# Memory Management in Eshkol

## Table of Contents
- [Overview](#overview)
- [OALR System (Ownership-Aware Lexical Regions)](#oalr-system-ownership-aware-lexical-regions)
- [Arena Allocation](#arena-allocation)
- [Object Headers](#object-headers)
- [Reference Counting](#reference-counting)
- [Linear Types](#linear-types)
- [Implementation Details](#implementation-details)
- [Usage Examples](#usage-examples)

---

## Overview

Eshkol uses **OALR (Ownership-Aware Lexical Regions)** - a memory management system combining:

1. **Arena allocation** - Fast bump-pointer allocation with batch deallocation
2. **Object headers** - 8-byte headers on all heap objects for metadata
3. **Reference counting** - Optional shared ownership for complex lifetimes
4. **Linear types** - Compile-time tracking of single-use resources

**No garbage collector.** Memory is deterministic and predictable.

---

## OALR System (Ownership-Aware Lexical Regions)

The OALR system provides **five ownership modes** for different memory management needs:

### Ownership Modes

| Mode | Syntax | Semantics | Use Case |
|------|--------|-----------|----------|
| **Default** | `(cons 1 2)` | Arena-allocated, no cleanup tracking | Most data structures |
| **Owned** | `(owned value)` | Linear type, must consume exactly once | Resources (files, sockets) |
| **Move** | `(move value)` | Transfer ownership to new binding | Transferring linear values |
| **Borrow** | `(borrow value body)` | Temporary read-only access | Reading without ownership |
| **Shared** | `(shared value)` | Reference-counted shared ownership | Complex shared data |
| **Weak** | `(weak-ref value)` | Non-owning reference | Breaking reference cycles |

**Implementation:** [`lib/backend/llvm_codegen.cpp:8234-8567`](lib/backend/llvm_codegen.cpp)

### Example: Linear Types

```scheme
;; File handle is linear - must be closed exactly once
(define file (owned (open-file "data.txt")))
(let ((contents (read-file file)))
  (close-file (move file))  ; Transfer ownership to close-file
  contents)
;; file is consumed - using it again would be a compile-time error
```

### Example: Shared Ownership

```scheme
;; Create shared data structure
(define tree (shared (make-tree 1000)))

;; Multiple references allowed
(define ref1 tree)
(define ref2 tree)

;; Automatically freed when last reference goes out of scope
```

---

## Arena Allocation

**Implementation:** [`lib/core/arena_memory.cpp`](lib/core/arena_memory.cpp) (6,186 lines)

Eshkol uses a **hybrid arena model**: a global arena (`__global_arena`) for the main thread, plus **per-thread arenas** (1 MB each, lazily allocated via `thread_local` storage) for parallel worker threads. The global arena grows automatically as needed; per-thread arenas provide contention-free allocation during parallel operations.

### Arena Structure

```c
typedef struct arena_block {
    uint8_t* data;              // Allocated memory block
    size_t size;                // Block size in bytes
    size_t used;                // Bytes currently used
    struct arena_block* next;   // Next block in chain
} arena_block_t;

typedef struct arena {
    arena_block_t* current;     // Current allocation block
    size_t block_size;          // Size of each block (8192 bytes)
    size_t total_allocated;     // Total bytes allocated
} arena_t;
```

### Allocation Strategy

```c
// lib/core/arena_memory.cpp:145-198
void* arena_alloc(size_t size) {
    // Round up to 8-byte alignment
    size = (size + 7) & ~7;
    
    // Try allocating from current block
    if (current_block->used + size <= current_block->size) {
        void* ptr = current_block->data + current_block->used;
        current_block->used += size;
        return ptr;
    }
    
    // Need new block
    size_t new_block_size = max(8192, size * 2);
    arena_block_t* new_block = malloc(sizeof(arena_block_t));
    new_block->data = malloc(new_block_size);
    new_block->size = new_block_size;
    new_block->used = size;
    new_block->next = current_block;
    current_block = new_block;
    
    return new_block->data;
}
```

**Key properties:**
- **Block size**: 8192 bytes (8KB) default
- **Alignment**: 8-byte alignment for all allocations
- **Growth**: New blocks allocated when current block is full
- **No individual free()**: Memory reused when arena is reset

---

## Object Headers

All heap-allocated objects have an **8-byte header at offset -8** from the data pointer:

```c
typedef struct eshkol_object_header {
    uint8_t  subtype;      // Type-specific subtype (0-255)
    uint8_t  flags;        // GC mark, linear, borrowed, etc.
    uint16_t ref_count;    // Reference count (0 = not ref-counted)
    uint32_t size;         // Object size in bytes (excluding header)
} eshkol_object_header_t;  // 8 bytes
```

### Memory Layout

```
┌──────────────────────┬────────────────────────┐
│  Header (8 bytes)    │  Object Data           │
│  at offset -8        │                        │
├──────────────────────┼────────────────────────┤
│ subtype   (1 byte)   │                        │
│ flags     (1 byte)   │  Variable size         │
│ ref_count (2 bytes)  │  depends on type       │
│ size      (4 bytes)  │                        │
└──────────────────────┴────────────────────────┘
                       ↑
                    data.ptr_val points here
```

### Accessing Headers

```c
// Get header from data pointer
void* data_ptr = tagged_val.data.ptr_val;
eshkol_object_header_t* header = ESHKOL_GET_HEADER(data_ptr);

// Read header fields
uint8_t subtype = header->subtype;
uint8_t flags = header->flags;
uint16_t ref_count = header->ref_count;
uint32_t size = header->size;

// Check object type
if (tagged_val.type == ESHKOL_VALUE_HEAP_PTR) {
    switch (subtype) {
        case HEAP_SUBTYPE_CONS:   /* cons cell */ break;
        case HEAP_SUBTYPE_STRING: /* string */ break;
        case HEAP_SUBTYPE_VECTOR: /* vector */ break;
        case HEAP_SUBTYPE_TENSOR: /* tensor */ break;
        // ...
    }
}
```

### Object Flags

```c
#define ESHKOL_OBJ_FLAG_MARKED    0x01  // GC mark bit
#define ESHKOL_OBJ_FLAG_LINEAR    0x02  // Linear type (must consume once)
#define ESHKOL_OBJ_FLAG_BORROWED  0x04  // Currently borrowed
#define ESHKOL_OBJ_FLAG_CONSUMED  0x08  // Linear value consumed
#define ESHKOL_OBJ_FLAG_SHARED    0x10  // Reference-counted
#define ESHKOL_OBJ_FLAG_WEAK      0x20  // Weak reference
#define ESHKOL_OBJ_FLAG_PINNED    0x40  // Pinned (no relocation)
#define ESHKOL_OBJ_FLAG_EXTERNAL  0x80  // External resource
```

**Example: Checking if object is shared**

```c
bool is_shared = (header->flags & ESHKOL_OBJ_FLAG_SHARED) != 0;
```

---

## Reference Counting

Objects marked with `ESHKOL_OBJ_FLAG_SHARED` use **reference counting** for automatic cleanup:

### Reference Count Operations

```c
// Increment reference count
uint16_t new_count = ESHKOL_INC_REF(data_ptr);

// Decrement reference count
uint16_t new_count = ESHKOL_DEC_REF(data_ptr);
if (new_count == 0) {
    // Last reference dropped - free object
    arena_free_object(data_ptr);
}
```

### Example: Shared Data Structure

```scheme
;; Create shared tree structure
(define tree (shared (make-binary-tree 10000)))

;; Multiple references - reference count = 3
(define ref1 tree)
(define ref2 tree)
(define ref3 tree)

;; When ref1, ref2, ref3 go out of scope, count decrements
;; When count reaches 0, tree is automatically freed
```

---

## Linear Types

Linear types ensure resources are **consumed exactly once** at compile-time. Attempting to use a consumed linear value triggers a compile-time error.

### Linear Type Rules

1. **Creation**: `(owned value)` marks value as linear
2. **Consumption**: `(move value)` transfers ownership, consuming original binding
3. **Single use**: Linear values can only be used once
4. **Compile-time checking**: Violations detected before code runs

### Example: File Handling

```scheme
;; Open file - returns linear handle
(define file (owned (open-output-file "output.txt")))

;; Write to file - borrows handle temporarily
(borrow file
  (write-string file "Hello, world!"))

;; Close file - consumes handle
(close-output-port (move file))

;; ERROR: file is consumed, cannot use again
;; (write-string file "More data")  ; Compile-time error
```

### Example: Processing Multiple Files

```scheme
;; Open input file (linear)
(define in-file (owned (open-input-file "data.txt")))

;; Open output file (linear)
(define out-file (owned (open-output-file "processed.txt")))

;; Process data (borrow both files)
(borrow in-file
  (borrow out-file
    (let ((line (read-line in-file)))
      (write-string out-file (string-upcase line)))))

;; Close both files (consume linear values)
(close-input-port (move in-file))
(close-output-port (move out-file))
```

---

## Implementation Details

### Source Files

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **Arena Core** | [`lib/core/arena_memory.cpp`](lib/core/arena_memory.cpp) | 6,186 | Arena allocation, object headers, display |
| **Arena Header** | [`lib/core/arena_memory.h`](lib/core/arena_memory.h) | 156 | Arena API, memory tracking |
| **Object Headers** | [`inc/eshkol/eshkol.h:274-547`](inc/eshkol/eshkol.h) | 274 | Header structure, access macros |
| **Memory Codegen** | [`lib/backend/memory_codegen.cpp`](lib/backend/memory_codegen.cpp) | 734 | OALR operator codegen |

### Allocation Functions

**Core allocator:**
```c
// lib/core/arena_memory.cpp:145
void* arena_alloc(size_t size);
```

**Type-specific allocators (with headers):**
```c
// lib/core/arena_memory.cpp:856-1234
void* arena_alloc_cons(void);           // 32-byte cons cell with header
void* arena_alloc_string(size_t len);   // String with header
void* arena_alloc_vector(size_t cap);   // Vector with header
void* arena_alloc_tensor(...);          // Tensor with header
void* arena_alloc_closure(...);         // Closure with header
```

### Cons Cell Layout (Phase 3B)

**32-byte tagged cons cell** with direct tagged value storage:

```c
// lib/core/arena_memory.h:67-72
typedef struct arena_tagged_cons_cell {
    eshkol_object_header_t header;  // 8 bytes (subtype=HEAP_SUBTYPE_CONS)
    eshkol_tagged_value_t car;      // 16 bytes (type + data)
    eshkol_tagged_value_t cdr;      // 16 bytes (type + data)
} arena_tagged_cons_cell_t;         // 40 bytes total
```

**Alignment:** 8-byte aligned for cache efficiency

---

## Usage Examples

### Example 1: Creating Objects

```scheme
;; Cons cell (automatic header creation)
(define pair (cons 1 2))
;; Allocated as:
;;   - Header: {subtype=HEAP_SUBTYPE_CONS, flags=0, ref_count=0, size=32}
;;   - Data: car={type=INT64, data=1}, cdr={type=INT64, data=2}

;; String (automatic header creation)
(define str "Hello, Eshkol!")
;; Allocated as:
;;   - Header: {subtype=HEAP_SUBTYPE_STRING, flags=0, ref_count=0, size=...}
;;   - Data: length=15, capacity=16, "Hello, Eshkol!\0"

;; Vector (automatic header creation)
(define vec (vector 1 2 3))
;; Allocated as:
;;   - Header: {subtype=HEAP_SUBTYPE_VECTOR, flags=0, ref_count=0, size=...}
;;   - Data: length=3, capacity=3, [elem0, elem1, elem2]
```

### Example 2: Linear Resource Management

```scheme
;; File handle is linear
(define file (owned (open-output-file "output.txt")))

;; Write to file (borrow = temporary access)
(borrow file
  (begin
    (write-string file "Line 1\n")
    (write-string file "Line 2\n")))

;; Close file (move = consume linear value)
(close-output-port (move file))

;; file is now consumed - compiler prevents further use
```

### Example 3: Shared Data with Reference Counting

```scheme
;; Create shared game state
(define game-state (shared (make-hash)))
(hash-set! game-state "player1" (make-player "Alice"))
(hash-set! game-state "player2" (make-player "Bob"))

;; Multiple systems can hold references
(define physics-ref game-state)  ; ref_count = 2
(define rendering-ref game-state) ; ref_count = 3
(define ai-ref game-state)        ; ref_count = 4

;; When all references go out of scope, automatically freed
;; No manual cleanup needed
```

### Example 4: Weak References

```scheme
;; Create shared object
(define cache (shared (make-hash)))

;; Create weak reference (doesn't prevent cleanup)
(define weak-cache-ref (weak-ref cache))

;; cache can be freed even though weak-cache-ref exists
;; Accessing weak-cache-ref returns nothing if object is freed
```

---

## Implementation Details

### Global Arena and Per-Thread Arenas

```c
// lib/core/arena_memory.cpp
extern arena_t* __global_arena;          // process-global, created threadsafe
thread_local arena_t* __thread_local_arena;  // per-worker-thread, created via
                                              // arena_create_thread_local
```

The global arena is created via `arena_create_threadsafe` (so its
`thread_safe` flag is true and the mutex protects concurrent
allocations). Per-thread arenas are created via plain `arena_create`
(no mutex), because each is accessed exclusively by its owning
thread.

`get_global_arena()` returns `__global_arena`. `arena_get_thread_local()`
returns `__thread_local_arena` when set, falling back to the global
arena otherwise. The selection of which arena a particular allocation
targets depends on the caller's context — in the parallel-primitives
code path, the global arena is passed explicitly (so result-list
marshaling happens in the caller's arena); inside a worker context,
the LLVM-codegen path uses `arena_get_thread_local()` so any heap
allocation from the closure body routes to the worker's TLS arena.

See "Per-Thread Arena Lifecycle" below for the lazy-init protocol,
eager TLS warmup at worker startup, and the
`arena_merge_to_parent` merge protocol.

### Block Growth Strategy

```c
// lib/core/arena_memory.cpp:178-185
// When current block is full, allocate new block
size_t new_size = max(8192, requested_size * 2);
arena_block_t* new_block = malloc(sizeof(arena_block_t));
new_block->data = malloc(new_size);
new_block->size = new_size;
new_block->next = arena->current;
arena->current = new_block;
```

**Properties:**
- Minimum block size: 8KB
- Growth factor: 2× requested size (doubling for large allocations)
- Block chaining: Blocks linked together for traversal

### Alignment

All allocations are **8-byte aligned** for optimal performance:

```c
// lib/core/arena_memory.cpp:152
size = (size + 7) & ~7;  // Round up to 8-byte boundary
```

### Memory Statistics

```c
// lib/core/arena_memory.cpp:2890-2950
typedef struct arena_stats {
    size_t total_allocated;      // Total bytes allocated
    size_t total_used;            // Bytes currently used
    size_t num_blocks;            // Number of arena blocks
    size_t largest_block;         // Largest block size
    size_t num_allocations;       // Total allocation count
} arena_stats_t;

// Get arena statistics
arena_stats_t arena_get_stats(void);
```

---

## Object Lifecycle

### 1. Allocation

```c
// Allocate cons cell with header
void* cons_ptr = arena_alloc(sizeof(eshkol_object_header_t) + 32);
eshkol_object_header_t* header = (eshkol_object_header_t*)cons_ptr;
header->subtype = HEAP_SUBTYPE_CONS;
header->flags = 0;
header->ref_count = 0;
header->size = 32;

// Data starts 8 bytes after header
void* data_ptr = (uint8_t*)cons_ptr + sizeof(eshkol_object_header_t);
```

### 2. Use

```c
// Access data (data_ptr from tagged value)
eshkol_tagged_value_t val;
val.type = ESHKOL_VALUE_HEAP_PTR;
val.data.ptr_val = (uint64_t)data_ptr;

// Get header from data pointer
eshkol_object_header_t* header = ESHKOL_GET_HEADER(data_ptr);
uint8_t subtype = header->subtype;
```

### 3. Cleanup

**Default (arena-allocated):** No explicit cleanup. Memory reused when arena is reset.

**Shared (ref-counted):** Automatically freed when ref_count reaches 0.

**Linear (owned):** Must be explicitly consumed via `move`.

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Arena allocation | O(1) amortized | Bump pointer, occasional block allocation |
| Reference increment | O(1) | Single memory write |
| Reference decrement | O(1) or O(n) | O(n) if triggers cleanup of large structure |
| Linear type check | O(1) | Compile-time only |

### Space Complexity

| Object Type | Header | Data | Total Overhead |
|-------------|--------|------|----------------|
| Cons cell | 8 bytes | 32 bytes | 8 bytes (25%) |
| String (n chars) | 8 bytes | 16 + n bytes | 8 bytes |
| Vector (n elems) | 8 bytes | 16 + 16n bytes | 8 bytes |
| Tensor | 8 bytes | 32 + 8n bytes | 8 bytes |
| Closure | 8 bytes | 40 bytes | 8 bytes (20%) |

**Header overhead:** Fixed 8 bytes per object, amortized over object size.

---

## Best Practices

### 1. Prefer Arena Allocation

```scheme
;; Good: Simple arena allocation
(define data (list 1 2 3 4 5))

;; Avoid: Unnecessary shared references
;; (define data (shared (list 1 2 3 4 5)))
```

### 2. Use Linear Types for Resources

```scheme
;; Good: File is linear
(define file (owned (open-output-file "data.txt")))
(write-data file data)
(close-output-port (move file))

;; Bad: File can leak if not closed
;; (define file (open-output-file "data.txt"))
;; (write-data file data)
;; ;; Forgot to close - file leaks
```

### 3. Minimize Shared References

```scheme
;; Good: Local ownership
(define (process-data input)
  (let ((temp (make-hash)))
    (hash-set! temp "key" input)
    (extract-result temp)))

;; Avoid unless needed: Shared ownership
;; (define (process-data input)
;;   (let ((temp (shared (make-hash))))
;;     ...))
```

### 4. Borrow Instead of Copy

```scheme
;; Good: Borrow large structure
(define (compute-stats data)
  (borrow data
    (length (filter positive? data))))

;; Wasteful: Copy large structure
;; (define (compute-stats data)
;;   (let ((copy (list-copy data)))
;;     (length (filter positive? copy))))
```

---

## Memory Safety Guarantees

Eshkol provides **compile-time memory safety** through:

1. **Linear type checking** - Prevents use-after-move
2. **Borrow checking** - Ensures borrowed values are not moved
3. **Lifetime tracking** - Shared references automatically freed
4. **No dangling pointers** - Weak references become `nothing` when freed

**These guarantees eliminate entire classes of memory bugs at compile time.**

---

## v1.1 Memory Patterns

This section documents the memory allocation and lifecycle patterns introduced in Eshkol v1.1, covering the consciousness engine (logic, inference, workspace), parallel execution, and GPU buffer management. All descriptions are grounded in the source code as it exists on the `refactor/optimisation` branch.

### 1. Consciousness Engine Allocation

All consciousness engine objects use the same `arena_allocate_with_header` primitive as bignums and rationals. Every heap-allocated object carries an 8-byte `eshkol_object_header_t` prepended before the data pointer:

```
Memory layout:  [header: 8 bytes][object data: variable]
                ^                ^
                |                +-- pointer returned to caller
                +-- accessible via ESHKOL_GET_HEADER(ptr) = ptr - 8
```

The header contains `{subtype: u8, flags: u8, ref_count: u16, size: u32}`. Each consciousness engine type has a dedicated subtype constant:

| Object             | Heap Subtype                  | Allocated via                     |
|---------------------|------------------------------|----------------------------------|
| Substitution        | `HEAP_SUBTYPE_SUBSTITUTION` (12) | `alloc_substitution(arena, capacity)` |
| Fact                | `HEAP_SUBTYPE_FACT` (13)         | `alloc_fact(arena, arity)`            |
| Knowledge Base      | `HEAP_SUBTYPE_KNOWLEDGE_BASE` (15) | `alloc_kb(arena)`                 |
| Factor Graph        | `HEAP_SUBTYPE_FACTOR_GRAPH` (16) | `arena_allocate_with_header`      |
| Workspace           | `HEAP_SUBTYPE_WORKSPACE` (17)    | `arena_allocate_with_header`      |

**Note:** Individual `eshkol_factor_t` structs do NOT get their own heap subtype. They are allocated with `arena_allocate_aligned` (no header) and stored as pointers inside the factor graph's `factors` array. Only the factor graph itself carries a header for type dispatch.

#### Substitution Copy-on-Extend

Substitutions follow a **copy-on-extend** discipline. `eshkol_extend_subst` never mutates the input substitution. Instead, it:

1. Allocates a **new** substitution via `alloc_substitution(arena, new_capacity)`.
2. `memcpy`s the old bindings (var IDs and terms) into the new allocation.
3. Appends the new binding.
4. Returns the new substitution pointer.

This means every call to `eshkol_extend_subst` produces a fresh arena allocation. The old substitution remains valid and unmodified, which is essential for backtracking during unification (failed unification branches don't corrupt the substitution passed in).

The capacity growth strategy is: if the new binding fits within the existing capacity, reuse that capacity; otherwise double the capacity (minimum 8 slots).

```
Substitution memory layout (variable-length):
[header: 8B][num_bindings: u32, capacity: u32]
            [var_ids: capacity * u64]
            [terms:   capacity * eshkol_tagged_value_t (16B each)]

Total data size = sizeof(eshkol_substitution_t) + capacity * (8 + 16) bytes
```

#### Temporary Substitutions During Unification

When `eshkol_unify` recurses (e.g., structural unification of facts), each recursive call may allocate intermediate substitutions via `eshkol_extend_subst`. If unification ultimately fails (returns NULL), all those intermediate substitutions remain allocated in the arena -- they are **not** freed. The arena's bump allocator has no per-object free; these temporaries persist until the arena is reset or destroyed.

For `kb_query`, which tries unification against every fact in the knowledge base, failed unification attempts accumulate dead substitutions in the arena. This is acceptable because:
- The arena is typically reset after a query cycle completes.
- Substitution sizes are small (typically tens of bytes).
- The arena grows in blocks (default 64KB for the global arena, 1MB for thread-local arenas), so fragmentation is not a concern.

#### Knowledge Base Mutation Strategy

`kb_assert!` **mutates the KB in place** -- it does NOT copy. The `eshkol_kb_assert` function:

1. Checks if `num_facts >= capacity`.
2. If growth is needed, allocates a **new** `facts` pointer array (double the capacity) via `arena_allocate_aligned`, copies old pointers, and swings the KB's `facts` field to point to the new array. The old pointer array is abandoned in the arena (never freed).
3. Stores the fact pointer at `facts[num_facts++]`.

The fact itself is not copied -- the KB stores a pointer to the fact's existing arena location. This means mutating a fact after assertion would affect the KB's view of it (though the API does not expose fact mutation).

Initial capacity is 16 (`KB_INITIAL_CAPACITY`), growing by 2x.

#### Factor Graph CPT Allocation and Mutation

When a factor is created (`eshkol_make_factor`), its conditional probability table (CPT) is allocated as a flat `double` array via `arena_allocate_aligned`:

```
CPT size = product(dims[0..num_vars-1]) doubles
Memory: arena_allocate_aligned(arena, cpt_size * sizeof(double), 8)
```

The CPT data is copied from the caller's input at creation time.

`fg_update_cpt!` (`eshkol_fg_update_cpt_tagged`) **mutates the CPT in place**: it overwrites the factor's existing `cpt` array with values extracted from the new tensor argument, entry by entry. No new allocation occurs for the CPT itself during update.

After CPT mutation, the function **nulls out** the message arrays (`fg->msg_fv = NULL; fg->msg_vf = NULL`). This forces the next `fg-infer!` call to re-allocate fresh message storage (via `allocate_messages`), causing belief propagation to reconverge from scratch with the new CPT. The old message arrays remain allocated in the arena but are unreferenced.

#### Message Array Allocation

Belief propagation messages are allocated lazily on the first call to `fg-infer!` via `allocate_messages`:

```
For each factor f with num_vars variables connected to the factor graph:
  For each factor-variable edge:
    msg_fv[edge] = alloc_doubles(arena, var_dim)  // factor-to-variable message
    msg_vf[edge] = alloc_doubles(arena, var_dim)  // variable-to-factor message
```

Messages are updated in-place during iteration (the BP loop overwrites `old_msg[s] = new_msg[s]`). Temporary per-iteration messages use `alloca` (stack allocation), not arena allocation, so BP iterations do not consume arena memory beyond the initial message allocation.

#### Workspace Module Storage

`eshkol_make_workspace` allocates the workspace struct and its inline module array as a single contiguous allocation:

```
data_size = sizeof(eshkol_workspace_t) + max_modules * sizeof(eshkol_workspace_module_t)
```

This is a fixed-capacity design: the maximum number of modules is set at creation time. The `content` buffer (a `double` array of dimension `dim`) is allocated separately via `arena_allocate_aligned`.

Module registration (`ws-register!`) copies the module name into the arena via `arena_allocate_aligned(arena, name_len + 1, 1)` and stores the process function as a tagged value (a closure). The module struct is filled in-place within the pre-allocated array.

The `ws-step!` finalization (`eshkol_ws_step_finalize`) uses stack-allocated arrays (max 16 modules) for salience scores and proposal pointers -- no arena allocation occurs during the step computation itself. The winning module's proposal tensor content is copied into the workspace's pre-allocated `content` buffer via element-wise conversion from int64 bit patterns to doubles.

### 2. Parallel Arena Isolation

#### Per-Thread Arena Architecture

Each worker thread in the thread pool gets its own arena, managed through `thread_local` storage in `thread_pool.cpp`:

```cpp
thread_local arena_t* tls_arena = nullptr;
thread_local bool tls_is_worker_thread = false;
thread_local size_t tls_arena_size = 0;
```

When a worker thread starts (in either `work_stealing_worker_func` or `legacy_worker_func`), it sets:
```cpp
tls_is_worker_thread = true;
tls_arena_size = pool->thread_arena_size;  // from config, default 1MB
```

The arena itself is **lazily created** on first call to `get_thread_local_arena()`:
```cpp
arena_t* get_thread_local_arena(void) {
    if (!tls_arena) {
        size_t size = tls_arena_size > 0 ? tls_arena_size : (1024 * 1024);  // 1MB default
        tls_arena = arena_create(size);  // NOT thread-safe variant -- single-thread access
    }
    return tls_arena;
}
```

Key design choice: per-thread arenas use `arena_create` (not `arena_create_threadsafe`), because each arena is only accessed by its owning thread. This avoids mutex overhead on every allocation within a worker.

The default configuration in `thread_pool.h` is:
```c
#define ESHKOL_THREAD_POOL_DEFAULT_CONFIG { \
    .num_threads = 0,              /* hardware_concurrency */  \
    .task_queue_capacity = 0,      /* unlimited */             \
    .thread_arena_size = 1024 * 1024,  /* 1MB per thread */    \
    .enable_metrics = true,                                    \
    .name = "default"                                          \
}
```

#### Arena Allocation During Parallel Operations

The current parallel primitives (`parallel-map`, `parallel-filter`, `parallel-fold`, `parallel-for-each`, `parallel-execute`) do **not** use per-thread arenas for task execution. Instead, the architecture works as follows:

1. **Task decomposition** happens on the main thread. The input Scheme list is converted to a `std::vector<eshkol_tagged_value_t>` (heap-allocated via the C++ allocator, not the arena).

2. **Task structs** (`llvm_parallel_map_task`) are allocated in a `std::vector` on the main thread's stack/heap. Each task contains decomposed i64 fields (closure pointer, item type, item data, result pointer) -- no tagged value structs cross the C/LLVM boundary.

3. **Workers** execute LLVM-generated functions (`__parallel_map_worker`, etc.) that reconstruct tagged values in pure LLVM IR and call the closure dispatcher. The worker writes its result via a pointer to a pre-allocated result slot in the main thread's `results` vector.

4. **Result marshaling** back to the main thread uses `vector_to_list(results, arena)`, which allocates cons cells in the **main thread's arena** (the global arena passed to the parallel function).

This means the per-thread arenas (`tls_arena`) are available but currently go unused by the built-in parallel primitives. They exist for user-level parallel code that might need arena allocation within worker tasks (e.g., closures that create heap objects during parallel-map). Any heap objects created by closures during parallel execution would be allocated in the global arena (passed through the LLVM codegen), not in thread-local arenas.

#### What Happens When an Arena Is Exhausted

When an arena's current block is full, `arena_allocate_aligned` allocates a **new block** via `malloc`:

```cpp
size_t new_block_size = (aligned_size > arena->default_block_size) ?
                        aligned_size : arena->default_block_size;
arena_block_t* new_block = create_arena_block(new_block_size);
new_block->next = arena->current_block;
arena->current_block = new_block;
```

Blocks form a singly-linked list. The new block becomes the current block; old blocks remain linked for later cleanup. If `malloc` fails, the allocation returns `nullptr`. There is no built-in overflow signal or exception -- callers must check for NULL returns.

For thread-local arenas, `arena_reset` is available via `thread_pool_reset_thread_arena()`. Reset frees all blocks except the original one and resets `used` to 0, effectively recycling the arena's memory for the next task.

#### Synchronization Guarantees

The parallel primitives provide the following guarantees:

- **No data races on results**: each task writes to its own dedicated result slot (`results[i]`). Slots are pre-allocated in a contiguous vector before task submission.
- **Future-based synchronization**: `future_get(futures[i])` blocks until task `i` completes, ensuring all results are visible before the main thread reads them.
- **Small list optimization**: lists with fewer than 4 elements bypass the thread pool entirely and execute sequentially on the main thread.
- **No arena synchronization**: the global arena is thread-safe (created via `arena_create_threadsafe` with a `pthread_mutex_t`), so concurrent allocations from closures are serialized. However, the parallel primitives themselves minimize contention by performing arena allocation only during result marshaling on the main thread.

#### AD Tape Isolation

For parallel autodiff, the AD tape state uses `thread_local` storage:

```cpp
thread_local ad_tape_t* __ad_tape_stack[MAX_TAPE_DEPTH];
thread_local uint64_t __ad_tape_depth = 0;
```

This ensures that parallel workers performing gradient computation do not corrupt each other's tape state. Each thread maintains its own tape stack independently.

### 3. GPU Buffer Marshaling

#### Arena-to-GPU Data Flow

Tensor data in Eshkol is arena-allocated as arrays of `int64_t` values that are actually `double` bit patterns (the `tensor_layout_t` struct). When GPU computation is needed (e.g., for matrix multiplication), the data flow is:

```
Arena tensor (int64 bit patterns)
    |
    v  (interpret as double* -- same bit pattern)
eshkol_gpu_wrap_host(host_ptr, size_bytes)
    |
    v
MTLBuffer (zero-copy if page-aligned, else copy)
    |
    v
GPU kernel execution
    |
    v
memcpy result back to host_ptr (if fallback allocation was used)
    |
    v
Arena tensor updated in-place
```

#### MTLBuffer Allocation Strategy

Metal buffers use **`MTLResourceStorageModeShared`** (unified memory) as the primary allocation mode. On Apple Silicon, shared mode means both CPU and GPU access the same physical memory -- no explicit DMA transfers are needed.

The `eshkol_gpu_wrap_host` function attempts zero-copy wrapping first:

```objc
id<MTLBuffer> buffer = [g_metal_device newBufferWithBytesNoCopy:host_ptr
                                                         length:size_bytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
```

This succeeds only if `host_ptr` is page-aligned. If it fails (arena allocations are 8-byte aligned, not page-aligned), the fallback path allocates a new MTLBuffer and copies data:

```objc
int result = metal_alloc(size_bytes, ESHKOL_MEM_UNIFIED, out);
memcpy(out->host_ptr, host_ptr, size_bytes);
```

The `metal_alloc` function supports three storage modes depending on `EshkolMemoryType`:

| Memory Type          | Storage Mode                | Use Case                     |
|----------------------|----------------------------|------------------------------|
| `ESHKOL_MEM_DEVICE`  | `MTLResourceStorageModePrivate` | GPU-only (no CPU access)     |
| `ESHKOL_MEM_HOST_PINNED` | Shared or Managed (based on unified memory detection) | Pinned host memory |
| Default              | `MTLResourceStorageModeShared`  | General purpose (unified)    |

#### Buffer Pool

To avoid repeated `MTLBuffer` allocation/deallocation overhead (significant for iterative workloads like ML training loops), gpu_memory.mm implements a **size-binned buffer pool**:

```
Allocation request (N bytes)
    |
    v
Round up to next power-of-2 -> bucket size
    |
    v
Check g_buffer_pool[bucket] for available buffer
    |
    +-- Found -> return pooled buffer (no Metal allocation)
    |
    +-- Empty -> [g_metal_device newBufferWithLength:bucket
                                            options:MTLResourceStorageModeShared]
```

Buffer return (`pool_release`) stores the buffer back in its size bucket, up to `POOL_MAX_PER_BUCKET = 8` buffers per bucket. Excess buffers are dropped and freed by ARC. The pool is drained (`pool_drain` / `g_buffer_pool.clear()`) during `metal_shutdown`.

The pool is used extensively within the matmul implementations for intermediate buffers (f32 conversion buffers, computation buffers) but NOT for the user's input/output buffers (those use `eshkol_gpu_wrap_host`).

#### Result Copy-Back

After GPU kernel execution, results are copied back to the host pointer. The copy-back strategy depends on whether zero-copy wrapping succeeded:

**Zero-copy case** (`newBufferWithBytesNoCopy` succeeded): The GPU writes directly to the arena tensor's memory. The MTLBuffer wraps the host pointer, so `[buffer contents]` IS the host pointer. No copy needed -- `buf_c.host_ptr == (void*)C` evaluates true, and the `memcpy` is skipped.

**Fallback case** (arena pointer not page-aligned): The MTLBuffer has its own allocation. After GPU completion, results must be explicitly copied:

```cpp
if (buf_c.host_ptr != (void*)C) {
    memcpy((void*)C, buf_c.host_ptr, M * N * sizeof(double));
}
```

For precision-converted kernels (sf64, df64, f32_simd), the copy-back always uses `memcpy(C->host_ptr, [result_buffer contents], elementsC * 8)` because the computation uses intermediate pooled buffers, not the original user buffer.

#### GPU Buffer Lifetime

GPU buffers are explicitly freed by calling `eshkol_gpu_free`, which:

1. For Metal: releases the `__bridge_retained` reference, allowing ARC to free the MTLBuffer. The underlying memory is freed when ARC drops the last reference.
2. For wrapped buffers (`flags & 1`): does NOT free the host pointer (it belongs to the arena). Only the MTLBuffer wrapper is released.
3. For non-wrapped buffers: `free(host_ptr)` is called.

The typical lifecycle in `eshkol_blas_matmul_dispatch`:
```
wrap_host(A) -> wrap_host(B) -> wrap_host(C) -> gpu_matmul -> copy_back_if_needed -> free(A,B,C)
```

GPU buffers are stack-local (`EshkolGPUBuffer buf_a, buf_b, buf_c`) and freed immediately after use. They do not persist beyond the matmul call. Intermediate pooled buffers (for f32 conversion, etc.) are returned to the pool via `pool_release`, not freed.

**Important lifetime note:** The arena tensor's memory outlives the GPU buffer. The GPU buffer is a transient wrapper. The arena tensor persists until arena reset/destroy. There is no risk of use-after-free because `eshkol_gpu_free` on a wrapped buffer only releases the MTLBuffer, not the underlying arena-owned memory.

---

## Platform-Specific ABI Considerations

Eshkol's runtime passes `eshkol_tagged_value_t` (a 16-byte struct) across function boundaries in several places. Platform ABIs differ in how they handle 16-byte aggregate return values and aggregate parameters, which required explicit platform dispatch in two functions.

### ARM64 Thunk Calling Convention (`call_thunk_closure`)

**File:** `lib/core/arena_memory.cpp:3908`

Dynamic-wind and `call/cc` thunks are zero-argument closures invoked through a trampoline (`call_thunk_closure`). The trampoline bridges a typed function pointer (stored as `void*` in the closure) back to a call returning `eshkol_tagged_value_t`.

The problem: the two dominant ABI families handle 16-byte aggregate return differently.
- **x86-64 (System V) and Windows x64**: The caller passes a hidden first argument — a pointer to the caller's return slot. The function writes the return value there and returns `void`.
- **AArch64**: Returns the 16-byte struct directly in register pairs (`x0:x1`). No hidden buffer is passed.

Without the fix, ARM64 would treat the caller-provided hidden buffer pointer as the first data argument, shifting all real arguments by one slot and corrupting the return value.

The fix introduces a compile-time branch:

```c
#if defined(__aarch64__) || defined(_M_ARM64)
    // ARM64: direct return in registers — no hidden buffer
    typedef eshkol_tagged_value_t (*fn0_t)(void);
    typedef eshkol_tagged_value_t (*fn1_t)(void*);
    // ... (fn2_t through fn4_t follow the same pattern)
    result = ((fn0_t)(uintptr_t)closure->func_ptr)();
#else
    // x86/Windows: hidden return buffer as first parameter
    typedef void (*fn0_t)(eshkol_tagged_value_t*);
    typedef void (*fn1_t)(eshkol_tagged_value_t*, void*);
    // ... (fn2_t through fn4_t follow the same pattern)
    ((fn0_t)(uintptr_t)closure->func_ptr)(&result);
#endif
```

`call_thunk_closure` is used by `call_thunk_from_tagged` (line ~3975) which is called during `call/cc` continuation unwinding and dynamic-wind before/after thunk invocation. See also [CONTINUATIONS.md](CONTINUATIONS.md) for the call/cc and dynamic-wind implementation that depends on this trampoline.

### Windows x64 Struct-by-Value Parameter Fix (`region_escape_tagged_value_into`)

**File:** `lib/core/arena_memory.cpp:1909`

The Windows x64 ABI requires that structs larger than 8 bytes be passed by pointer, not by value. `region_escape_tagged_value_into` previously took `eshkol_tagged_value_t val` by value (16 bytes), violating this convention and causing misaligned stack frames on Windows.

The fix changes the signature:

```c
// Before:
extern "C" void region_escape_tagged_value_into(
    eshkol_tagged_value_t* out,
    eshkol_tagged_value_t val);           // 16-byte struct by value — illegal on Windows x64

// After:
extern "C" void region_escape_tagged_value_into(
    eshkol_tagged_value_t* out,
    const eshkol_tagged_value_t* val);    // pointer — correct on all platforms
```

The corresponding LLVM codegen was updated to pass the address of the value instead of the value itself. This function is called whenever a region-allocated object (OALR ownership model) is escaped into external or shared memory when returning from a lexical scope.

---

## See Also

- [Type System](TYPE_SYSTEM.md) - Object headers, type tags, subtypes
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) - Memory codegen, LLVM backend
- [API Reference](../API_REFERENCE.md) - Memory management functions

---

## What Landed in v1.2 / v1.2.1

The v1.2 release reorganized the memory subsystem around four architectural
moves: per-thread arenas with explicit lifecycle (`arena_create_thread_local`,
`arena_merge_to_parent`), scoped sub-arenas with LIFO push/pop on the main
arena, a 512 MB stack configured at link time plus an `ESHKOL_STACK_SIZE`
env override, and integer-overflow hardening on every variable-size
allocation site (HARDENING.md §192). This section documents each, with
citations to the current source.

### The Object Header, Field by Field

`inc/eshkol/eshkol.h` §322–331 gives the exact layout:

```c
typedef struct eshkol_object_header {
    uint8_t  subtype;      // 1 byte  — HEAP_SUBTYPE_* (0..255)
    uint8_t  flags;        // 1 byte  — ESHKOL_OBJ_FLAG_* bitfield
    uint16_t ref_count;    // 2 bytes — shared/weak ref count (0 = not ref-counted)
    uint32_t size;         // 4 bytes — object data size, excluding the header
} eshkol_object_header_t;

ESHKOL_STATIC_ASSERT(sizeof(eshkol_object_header_t) == 8,
                     "Object header must be 8 bytes for alignment");
```

Total: 8 bytes, statically asserted. The header is prepended to every
heap-allocated object; the data pointer returned by `arena_allocate_with_header`
points to the byte immediately after the header. `ESHKOL_GET_HEADER(data_ptr)`
in §446 is the macro that walks backwards by `sizeof(eshkol_object_header_t)`
to recover the header pointer.

The 4-byte `size` field caps the maximum single-object data size at
`UINT32_MAX` = 4 GiB − 1 (see "Allocation Hardening" below). The
`ref_count` is `uint16_t`, so 65 535 shared references is the upper bound
before saturation. The `flags` byte is a bitfield of `ESHKOL_OBJ_FLAG_*`
constants (§315–319): MARKED, LINEAR, BORROWED, CONSUMED, SHARED, WEAK,
PINNED, EXTERNAL.

### Arena Block Size: 1024-Byte Floor, 8192-Byte Default

`arena_create` (`lib/core/arena_memory.cpp` §169–195) enforces a minimum
block size of 1024 bytes. The common case is `arena_create(8192)` from
`Arena` (the C++ wrapper, §2417). The REPL shared arena passes 8192 as
well. Per-thread worker arenas pass 1 MB. The global arena and per-thread
arenas are *not* a fixed size — the field is `default_block_size`,
applied when a *new* block needs to be allocated. The initial block is
created at arena creation time.

When the current block cannot satisfy an allocation,
`arena_allocate_aligned` (§290–339) allocates a new block sized
`max(aligned_size, arena->default_block_size)` and links it at the head
of the singly-linked block chain. Old blocks remain reachable for the
arena's lifetime (until `arena_reset` or `arena_destroy`).

### Allocation Hardening: SIZE_MAX and UINT32_MAX Guards

`arena_allocate_with_header` (`lib/core/arena_memory.cpp` §358–403)
guards against two integer-overflow classes flagged in HARDENING.md §192:

```c
// #192 CRITICAL: total_size overflow if data_size is near SIZE_MAX
if (data_size > SIZE_MAX - sizeof(eshkol_object_header_t) - 8) {
    eshkol_error("arena_allocate_with_header: data_size=%zu would overflow", data_size);
    return nullptr;
}

// #192 HIGH: header->size is uint32_t — silently truncating a 4GB+ alloc
//           to its low 32 bits makes downstream readers under-copy
if (data_size > UINT32_MAX) {
    eshkol_error("arena_allocate_with_header: data_size=%zu exceeds UINT32_MAX", data_size);
    return nullptr;
}
```

The first check prevents wrap-around: `data_size + sizeof(header) + 8`
must not exceed `SIZE_MAX`, so the comparison is rearranged as
`data_size > SIZE_MAX − sizeof(header) − 8`. Without this, a caller
passing `data_size = SIZE_MAX − 4` would see the total compute to a
small positive value (well under the requested data size), the alignment
round-up would succeed, and the caller would receive a buffer much
smaller than expected — an immediate heap-overflow primitive.

The second check protects the header's 4-byte `size` field. Without it,
a 5 GB allocation would succeed but record `size = 5G mod 2^32 ≈ 705 MB`,
causing every downstream user that walks the object by header size to
under-copy by 4 GB. Reject rather than truncate.

Similar guards appear at every variable-size allocation site in
`arena_memory.cpp`:

- `arena_allocate_multi_value` §421–434 — `count * sizeof(tagged_value)
  + sizeof(size_t)` overflow check.
- `arena_allocate_tagged_cons_batch` §913–919 — `count *
  sizeof(arena_tagged_cons_cell_t)` overflow check.
- `arena_allocate_string_with_header` §517 — `length + sizeof(header) +
  8` overflow check.
- `arena_allocate_vector_with_header` §563–566 — `capacity *
  sizeof(tagged_value) + header + 8` overflow check.
- `arena_allocate_tensor_full` §3734–3748 — both `num_dims *
  sizeof(uint64_t)` and `total_elements * sizeof(int64_t)` overflow
  checks before the two component allocations.

### Per-Thread Arena Lifecycle

The v1.2 per-thread arena API is declared in `lib/core/arena_memory.h`
§88–116:

```c
arena_t* arena_get_thread_local(void);
arena_t* arena_create_thread_local(size_t size_hint);
void     arena_merge_to_parent(arena_t* dest, arena_t* src);
int      arena_is_worker_thread(void);
void     eshkol_thread_init_worker(size_t arena_size_hint);
void     eshkol_thread_shutdown_worker(void);
```

`arena_create_thread_local` (§1752–1760 in `arena_memory.cpp`):

```c
arena_t* arena_create_thread_local(size_t size_hint) {
    if (__thread_local_arena) return __thread_local_arena;   // idempotent
    size_t block_size = size_hint > 0 ? size_hint : (1024 * 1024);
    __thread_local_arena = arena_create(block_size);
    return __thread_local_arena;
}
```

Default block size is 1 MB. The TLS variable is `__thread_local_arena`,
a `thread_local arena_t*`. The pool's `thread_arena_size` config value
(default 1 MB, see `inc/eshkol/backend/thread_pool.h` §49) flows into
this size_hint.

**Eager init at worker startup**. `eshkol_thread_init_worker`
(§1787–1814) is called from `work_stealing_worker_func`
(`lib/backend/thread_pool.cpp` §203) and `legacy_worker_func` (§383)
before the first task runs. It touches every thread-local slot:

- `__ad_tape_stack[MAX_TAPE_DEPTH]` (32 slots) and `__ad_tape_depth`
  — explicitly zero so a JIT-emitted load on a fresh worker thread sees
  a defined value even if the TLS image is not zero-filled.
- `__outer_ad_node_stack[MAX_TAPE_DEPTH]` and `__outer_ad_node_depth`
  — N-dimensional derivative state.
- `__outer_ad_node_storage`, `__outer_ad_node_to_inner`,
  `__outer_grad_accumulator`, `__inner_var_node_ptr`,
  `__gradient_x_degree` — double-backward storage.
- `__region_stack[MAX_REGION_DEPTH]` (64 slots) and
  `__region_stack_depth` — OALR region stack.
- The thread-local arena itself, via `arena_create_thread_local`.

The architectural rationale (recorded in the source comment at
§1762–1786): on POSIX+glibc, `thread_local` is lazily initialized on
first access; on macOS Mach-O, it goes through `__tlv_*` wrappers.
Explicitly touching each TLS slot at worker startup forces
initialization before the first user task runs, preventing "silent
zero" hazards where codegen reads a TLS slot before it has been
materialized.

**Merge on join**. `arena_merge_to_parent` (§1837–1871) is the
collect-results-back-to-parent primitive:

```c
void arena_merge_to_parent(arena_t* dest, arena_t* src) {
    if (!dest || !src || dest == src) return;
    if (dest->thread_safe) arena_lock(dest);

    // Append src's blocks to the END of dest's chain.
    // dest->current_block stays unchanged — dest continues allocating
    // from its own newest block. Src's blocks become "old" blocks
    // containing finalized data that won't be allocated into.
    if (src->current_block) {
        if (dest->current_block) {
            arena_block_t* dest_tail = dest->current_block;
            while (dest_tail->next) dest_tail = dest_tail->next;
            dest_tail->next = src->current_block;
        } else {
            dest->current_block = src->current_block;
        }
        dest->total_allocated += src->total_allocated;
        // Clear src without freeing blocks (now owned by dest)
        src->current_block = nullptr;
        src->total_allocated = 0;
    }

    if (dest->thread_safe) arena_unlock(dest);
}
```

The append is at the tail of dest's chain so that dest's
`current_block` (the head of the chain) keeps allocating into the same
block it was using before the merge. Src's blocks become read-only
"old" blocks. Src is cleared without freeing the blocks themselves —
they are now owned by dest. Src remains a valid `arena_t*` (with an
empty block list) and can be reused for the next batch of allocations,
or destroyed.

**Shutdown**. `eshkol_thread_shutdown_worker` (§1818–1835) destroys
the thread-local arena via `arena_destroy(__thread_local_arena)` and
nulls every TLS slot, so a future pthread re-use does not observe
stale tape or region pointers from a previous worker lifetime.

**Worker detection**. `arena_is_worker_thread` (§1877–1883) delegates
to `eshkol_thread_pool_is_worker` from `thread_pool.cpp`, declared as
a weak symbol so the arena layer compiles even if the thread-pool
layer is not linked.

**Current parallel-primitives use of TLS arenas**. The parallel
primitives (`parallel-map`, `parallel-filter`, `parallel-fold`,
`parallel-execute`) currently use the *global arena* — passed to
`vector_to_list` (`parallel_codegen.cpp` §186, 347) — for assembling
result lists on the main thread after all workers complete. Workers
write per-element results into a stack-allocated `std::vector<eshkol_tagged_value_t>`
in the calling thread, via pointer. The thread-local arenas are
available for use by closure bodies executing within workers (any
`arena_allocate*` call from a worker context routes to the per-thread
arena via the worker's TLS `__thread_local_arena`). They are not yet
used by the parallel primitives' own marshaling infrastructure.

### Scoped Sub-Arenas: arena_push_scope / arena_pop_scope

The header (`lib/core/arena_memory.h` §39–43, 83–85) declares an
`arena_scope_t` struct and a push/pop API:

```c
struct arena_scope {
    arena_block_t* block;  // Block at scope start
    size_t used;           // Used bytes at scope start
    arena_scope_t* parent; // Parent scope (LIFO stack)
};

void arena_push_scope(arena_t* arena);
void arena_pop_scope(arena_t* arena);
```

`arena_push_scope` (`arena_memory.cpp` §706–721) snapshots
`{current_block, current_block->used}` into a malloc-allocated
`arena_scope_t`, prepended to `arena->current_scope` (LIFO). No
allocation of arena memory happens during push; this is a pure
checkpoint.

`arena_pop_scope` (§723–792) restores arena state to the scope start:

```c
void arena_pop_scope(arena_t* arena) {
    if (!arena || !arena->current_scope) {
        eshkol_error("Attempted to pop arena scope with no matching push — "
                     "unbalanced scope operations risk memory corruption");
        return;  // Graceful: skip the pop rather than kill the process
    }

    arena_scope_t* scope = arena->current_scope;

    // Free any blocks allocated AFTER this scope (the chain walks back
    // toward the older blocks; we free everything between the current
    // head and the scope's saved block).
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
}
```

**LIFO discipline is the invariant.** Cross-scope release corrupts
memory; the function logs an error and skips the pop rather than
kill the process. There is no magic sentinel on the scope object
itself, but the matching wrapper type for AD tapes
(`AdTapeOwned`) carries one — `magic = 0x5045544f444151` ("ADOWNER"
in ASCII) — so the release path can verify it is operating on an
owned tape rather than an arena-borrowed one. `ad_tape_release` on a
non-owned tape silently no-ops.

**The Bug I scenario.** A theorem-proving downstream (Noesis Sigma)
was allocating one AD tape per Adam-optimizer iteration, ~75 000
tapes per discover run, none freed. The main arena grew until
`malloc` returned NULL; subsequent AD ops dereffed a NULL tape and
crashed. The fix (commit `fb86bf1`, 2026-04-20) wraps each owned
tape in `arena_push_scope` at creation and `arena_pop_scope` at
explicit release, on the main arena. No separate arena per tape;
just checkpoint/restore on the existing one. This is a textbook
use of the scope API — many short-lived allocations followed by
bulk release in LIFO order.

**Diagnostic poisoning**. When `ESHKOL_ARENA_POISON=1`, the pop
routine (§732–771) fills the released region with `0xCB` bytes before
freeing the blocks. Any later dereference of a stale pointer crashes
with an address containing `0xCB` in obvious positions, turning a
silent SEGV at a mangled-looking address into an immediately-recognizable
arena UAF. The byte was picked to be distinct from libc's
MallocPreScribble (0xAA), MallocScribble (0x55), and glibc's
`perturb_free` (0x42).

### 512 MB Stack for Deep Recursion

Two coordinated changes set the process stack to 512 MB:

**Link-time, macOS** (`CMakeLists.txt` §1358–1359):

```cmake
# 512MB stack for deep recursion (main thread stack set at link time on macOS)
target_link_options(eshkol-run PRIVATE "-Wl,-stack_size,0x20000000")
```

`0x20000000` = 512 × 1024 × 1024 bytes. macOS sets the main thread
stack size at link time; `setrlimit(RLIMIT_STACK, ...)` after process
start cannot increase it.

**Link-time, Linux** (`CMakeLists.txt` §1371–1372):

```cmake
# 512MB stack for deep recursion
target_link_options(eshkol-run PRIVATE "-Wl,-z,stack-size=536870912")
```

`536870912` = 512 × 1024 × 1024 bytes.

**Runtime, both platforms**: `eshkol_init_stack_size`
(`lib/core/arena_memory.cpp` §62–90) uses `setrlimit(RLIMIT_STACK,
...)` to raise the soft limit for spawned threads, and as a Linux
fallback if the link-time flag was not applied. `ESHKOL_STACK_SIZE`
env var overrides:

```c
extern "C" void eshkol_init_stack_size() {
#ifdef _WIN32
    return;   // Windows thread stack sizing is handled at link/thread creation time
#else
    const rlim_t default_stack = 512ULL * 1024 * 1024;  // 512MB
    rlim_t target = default_stack;
    const char* env_val = getenv("ESHKOL_STACK_SIZE");
    if (env_val) {
        char* end = nullptr;
        unsigned long long parsed = strtoull(env_val, &end, 0);
        if (end != env_val && parsed >= 1024 * 1024) target = (rlim_t)parsed;
    }
    struct rlimit rl;
    if (getrlimit(RLIMIT_STACK, &rl) == 0 && rl.rlim_cur < target) {
        rl.rlim_cur = target;
        if (rl.rlim_max != RLIM_INFINITY && rl.rlim_max < target) {
            rl.rlim_cur = rl.rlim_max;
        }
        setrlimit(RLIMIT_STACK, &rl);
    }
#endif
}
```

This is called from generated `main` functions and from REPL JIT
init (via `ADD_SYMBOL(eshkol_init_stack_size)` in
`repl_jit.cpp:664`). The env override has a 1 MiB floor to prevent
accidentally setting an unusably small stack.

**Worker thread stack**: workers do not get the 512 MB stack — they
get a separate sizing via `pthread_attr_setstacksize`. The default
is 16 MB (`thread_pool.cpp` §482), overridable by
`ESHKOL_WORKER_STACK_BYTES`. The floor is `PTHREAD_STACK_MIN`.

**Recursion-depth check**: independent of the OS stack size, the
runtime tracks Scheme-level recursion depth in
`lib/core/resource_limits.cpp` §261–266 and aborts when it exceeds
`ESHKOL_DEFAULT_MAX_STACK_DEPTH = 100000`
(`inc/eshkol/core/resource_limits.h` §41), overridable by
`ESHKOL_MAX_STACK`. This is checked at function-prologue level by
generated code, so it catches runaway recursion even if the OS
stack is large enough to absorb it. The two limits are
complementary: the OS stack prevents undefined behavior from
overrunning the actual stack; the recursion-depth check prevents
denial-of-service from intentionally deep recursion against an
untrusted Scheme program.

**Windows `jmp_buf` sizing**: `eshkol_jmp_buf_size` in
`lib/core/platform_runtime.cpp` §551–553 returns `sizeof(jmp_buf)`
at runtime. The codegen for `setjmp` allocates the buffer via
`builder->CreateAlloca(int8_type, eshkol_jmp_buf_size_result, "jmp_buf")`
rather than a compile-time constant, so the same generated IR
links correctly against different `jmp_buf` sizes across libc
implementations (glibc, musl, MSVCRT, Apple libc). See
CONTINUATIONS.md §`Platform-Specific setjmp Lowering` for the
companion architecture-specific intrinsic calls (`Intrinsic::sponentry`
on Windows ARM64, `Intrinsic::frameaddress(0)` on Windows x64).
