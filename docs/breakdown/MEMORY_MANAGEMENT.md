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

**Implementation:** [`lib/backend/llvm_codegen.cpp:8234-8567`](lib/backend/llvm_codegen.cpp:8234)

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

**Implementation:** [`lib/core/arena_memory.cpp:1-3210`](lib/core/arena_memory.cpp:1)

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
| **Arena Core** | [`lib/core/arena_memory.cpp`](lib/core/arena_memory.cpp:1) | 3,210 | Arena allocation, object headers, display |
| **Arena Header** | [`lib/core/arena_memory.h`](lib/core/arena_memory.h:1) | 156 | Arena API, memory tracking |
| **Object Headers** | [`inc/eshkol/eshkol.h:274-547`](inc/eshkol/eshkol.h:274) | 274 | Header structure, access macros |
| **Memory Codegen** | [`lib/backend/memory_codegen.cpp`](lib/backend/memory_codegen.cpp:1) | 734 | OALR operator codegen |

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

### Global Arena

```c
// lib/core/arena_memory.cpp:45-47
static arena_t* __global_arena = NULL;

// Initialized at program start
void arena_init(void) {
    __global_arena = create_arena(8192);  // 8KB initial block
}
```

**All allocations go through the global arena.** No per-function or per-scope arenas.

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

## See Also

- [Type System](TYPE_SYSTEM.md) - Object headers, type tags, subtypes
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) - Memory codegen, LLVM backend
- [API Reference](../API_REFERENCE.md) - Memory management functions
