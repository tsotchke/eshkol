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

Eshkol uses a **single global arena** (`__global_arena`) for all heap allocations. The arena grows automatically as needed.

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
| Closure | 8 bytes | 32 bytes | 8 bytes (25%) |

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

## See Also

- [Type System](TYPE_SYSTEM.md) - Object headers, type tags, subtypes
- [Compiler Architecture](COMPILER_ARCHITECTURE.md) - Memory codegen, LLVM backend
- [API Reference](../API_REFERENCE.md) - Memory management functions
