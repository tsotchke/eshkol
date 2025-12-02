# Eshkol Memory Management System

## Overview

Eshkol uses **Ownership-Aware Lexical Regions (OALR)** - a predictable, GC-free memory management system that combines the efficiency of arena allocation with the safety of ownership semantics.

### Design Principles

1. **Predictable**: No garbage collection pauses, deterministic cleanup
2. **Gradual**: Simple code needs no annotations; complexity only when needed
3. **Safe**: Compile-time ownership checking where possible
4. **Efficient**: Arena allocation within regions, minimal overhead
5. **Long-running friendly**: Explicit control for servers and persistent applications

---

## Memory Tiers

### Tier 1: Stack (Default)

The simplest and fastest allocation. Values are automatically freed when the scope exits.

```scheme
(define (compute x)
  (let ((temp (* x x)))      ; Stack-allocated
    (+ temp 1)))             ; temp freed here
```

**Characteristics**:
- Zero allocation overhead (bump pointer)
- Automatic cleanup on scope exit
- No annotation needed
- Used for: primitives, small temporaries, function locals

**Implementation**: Uses the existing arena with automatic scope push/pop.

---

### Tier 2: Region (Lexical Lifetime)

Batch allocation with lexical lifetime. All allocations within a region are freed together when the region exits.

```scheme
(define (process-batch items)
  (with-region
    (let* ((parsed (map parse-item items))
           (filtered (filter valid? parsed))
           (results (map transform filtered)))
      (summarize results))))  ; Entire region freed here
```

**Syntax**:
```scheme
(with-region body ...)              ; Anonymous region
(with-region 'name body ...)        ; Named region (can be referenced)
(with-region ('name size) body ...) ; Named region with size hint
```

**Characteristics**:
- All allocations use region's arena
- Single bulk deallocation at region exit
- Values can escape via return (compiler promotes them)
- Great for request handling, batch processing, transforms

**Implementation**:
```c
typedef struct eshkol_region {
    eshkol_arena_t* arena;          // Region's dedicated arena
    const char* name;               // Optional name (NULL for anonymous)
    struct eshkol_region* parent;   // Parent region (for nesting)
    size_t escape_count;            // Track escaping allocations
} eshkol_region_t;
```

---

### Tier 3: Owned (Linear/Affine Types)

Resources that must be explicitly consumed. The compiler ensures owned values are used exactly once (linear) or at most once (affine).

```scheme
;;; Creating owned resources
(define (open-connection host port)
  (owned (tcp-connect host port)))

;;; Using owned resources - must be consumed
(define (handle-request conn)
  (let ((response (http-get conn)))  ; conn is "moved" here
    (close conn)                      ; conn consumed (required)
    response))

;;; Error: owned value not consumed
(define (bad-example conn)
  (let ((response (http-get conn)))
    response))  ; ERROR: conn not consumed

;;; Transfer ownership
(define (transfer-ownership conn)
  (let ((new-owner (move conn)))     ; Explicit move
    (process new-owner)))            ; new-owner must be consumed
```

**Syntax**:
```scheme
(owned expr)              ; Mark value as owned
(move value)              ; Transfer ownership
(consume value)           ; Consume without returning
(borrow value body ...)   ; Temporarily borrow (read-only access)
```

**Owned Types** (resources that should always be owned):
- File handles: `(owned (open-file path))`
- Sockets: `(owned (tcp-connect host port))`
- Locks: `(owned (mutex-lock m))`
- Database connections: `(owned (db-connect url))`

**Characteristics**:
- Compile-time tracking of ownership
- Must be consumed (closed, freed, transferred)
- Prevents resource leaks
- Zero runtime overhead (all checking at compile time)

**Implementation**:
```c
// Compile-time ownership tracking
typedef struct {
    const char* name;           // Variable name
    bool consumed;              // Has been consumed?
    bool moved;                 // Has been moved?
    size_t defined_at;          // Source location
} ownership_state_t;
```

---

### Tier 4: Shared (Reference Counted)

For values with complex, dynamic lifetimes. Uses reference counting for deterministic cleanup.

```scheme
;;; Creating shared values
(define (create-cache)
  (shared (make-hash-table 1000)))

;;; Shared values can be freely copied
(define (use-cache cache)
  (let ((local-ref cache))        ; Increments ref count
    (cache-get local-ref "key"))) ; Decrements on scope exit

;;; Weak references (don't prevent cleanup)
(define (create-observer cache)
  (weak-ref cache))               ; Doesn't increment count
```

**Syntax**:
```scheme
(shared expr)             ; Create shared (ref-counted) value
(weak-ref value)          ; Create weak reference
(strong-ref weak)         ; Upgrade weak to strong (may fail)
(ref-count value)         ; Get current reference count (debugging)
```

**Cycle Detection**:
```scheme
;;; For cyclic structures, use weak references
(define (make-graph)
  (shared
    (let ((nodes (make-hash-table)))
      ;; Edges use weak refs to prevent cycles
      (hash-set! nodes 'a (list (weak-ref nodes)))
      nodes)))
```

**Characteristics**:
- Deterministic cleanup (when ref count hits zero)
- Small overhead (ref count increment/decrement)
- Supports weak references for cycles
- Used for: caches, shared state, graphs, long-lived data

**Implementation**:
```c
typedef struct eshkol_shared_header {
    uint32_t ref_count;         // Strong reference count
    uint32_t weak_count;        // Weak reference count
    void (*destructor)(void*);  // Custom cleanup function
    uint8_t flags;              // Flags (e.g., cycle-detected)
} eshkol_shared_header_t;
```

---

## Escape Analysis

The compiler automatically detects values that escape their scope and promotes them appropriately.

```scheme
;;; Compiler detects escape, allocates appropriately
(define (create-list n)
  (let ((result (make-vector n)))  ; Would be stack, but escapes
    result))                        ; Promoted to caller's region

;;; No escape - stays on stack
(define (sum-list lst)
  (let ((total 0))                 ; Stack allocated
    (for-each (lambda (x) (set! total (+ total x))) lst)
    total))                        ; Primitive, copied to return
```

**Escape Categories**:

1. **No Escape**: Value used only within its scope
   - Allocation: Stack or current region

2. **Return Escape**: Value returned from function
   - Allocation: Caller's region (promoted)

3. **Closure Escape**: Value captured by closure
   - Allocation: Shared (ref-counted)

4. **Global Escape**: Value stored in global/mutable location
   - Allocation: Shared (ref-counted)

---

## Memory Hierarchy Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TIER 1: STACK                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│  Default allocation. Fastest. Auto-freed on scope exit.                 │
│  Usage: (let ((x 42)) ...)                                              │
│  Overhead: Zero                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  TIER 2: REGION                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│  Scoped arena. Batch allocation/free. Great for request handling.       │
│  Usage: (with-region (process-data input))                              │
│  Overhead: Arena creation (~100 bytes header)                           │
├─────────────────────────────────────────────────────────────────────────┤
│  TIER 3: OWNED                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│  Linear types. Must be consumed. For resources (files, sockets).        │
│  Usage: (owned (open-file path))                                        │
│  Overhead: Zero (compile-time only)                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  TIER 4: SHARED                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│  Ref-counted. For caches, graphs, long-lived shared data.               │
│  Usage: (shared (make-hash-table 1000))                                 │
│  Overhead: 8 bytes header + inc/dec on copy                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Practical Examples

### Example 1: Web Server (Long-Running)

```scheme
(define (start-server port)
  ;; Shared cache - lives for server lifetime
  (let ((cache (shared (make-lru-cache 10000)))
        (db (shared (db-connect "postgres://..."))))

    (http-serve port
      (lambda (req)
        ;; Per-request region - freed after each request
        (with-region
          (let* ((path (request-path req))
                 (cached (cache-get cache path)))
            (if cached
                cached
                (let ((result (with-region 'query
                                ;; Nested region for DB query
                                (db-query db path))))
                  ;; Result escapes to outer region
                  (cache-set! cache path result)
                  result))))))))
```

### Example 2: File Processing (Resource Safety)

```scheme
(define (process-file path)
  (let ((file (owned (open-file path "r"))))  ; Owned resource
    (let ((contents (read-all file)))
      (close file)                             ; Must consume
      (with-region
        ;; Process in region - bulk cleanup
        (let* ((lines (string-split contents "\n"))
               (parsed (map parse-line lines))
               (filtered (filter valid? parsed)))
          (summarize filtered))))))

;;; Using with-resource (syntactic sugar)
(define (process-file-v2 path)
  (with-resource (file (open-file path "r"))  ; Auto-close on exit
    (with-region
      (let ((contents (read-all file)))
        (process contents)))))
```

### Example 3: Graph Processing (Cycles)

```scheme
(define (build-graph edges)
  (shared
    (let ((nodes (make-hash-table)))
      (for-each
        (lambda (edge)
          (let ((from (car edge))
                (to (cdr edge)))
            ;; Use weak refs for edges to avoid cycles
            (let ((from-node (or (hash-ref nodes from)
                                 (hash-set! nodes from (make-node from)))))
              (node-add-edge! from-node (weak-ref (hash-ref nodes to))))))
        edges)
      nodes)))
```

### Example 4: Tensor Computation (Batch Processing)

```scheme
(define (neural-forward network input)
  (with-region ('forward (* (network-size network) 1000))
    ;; All intermediate activations in this region
    (let loop ((layers (network-layers network))
               (activation input))
      (if (null? layers)
          activation  ; Final result escapes
          (let* ((layer (car layers))
                 (weights (layer-weights layer))
                 (bias (layer-bias layer))
                 ;; Intermediate tensors freed with region
                 (z (tensor-add (tensor-matmul weights activation) bias))
                 (a (tensor-relu z)))
            (loop (cdr layers) a))))))
```

---

## Runtime Support

### New Value Types

```c
typedef enum {
    // ... existing types ...
    ESHKOL_VALUE_REGION_PTR  = 13,  // Pointer to region
    ESHKOL_VALUE_OWNED_PTR   = 14,  // Owned resource pointer
    ESHKOL_VALUE_SHARED_PTR  = 15,  // Shared (ref-counted) pointer
} eshkol_value_type_t;
```

### New Runtime Functions

```c
// Region management
eshkol_region_t* region_create(const char* name, size_t hint);
void region_destroy(eshkol_region_t* region);
void* region_allocate(eshkol_region_t* region, size_t size);
eshkol_region_t* region_current(void);
void region_push(eshkol_region_t* region);
void region_pop(void);

// Ownership tracking (compile-time, but runtime for dynamic checks)
void owned_mark(void* ptr);
void owned_consume(void* ptr);
bool owned_is_consumed(void* ptr);

// Shared/ref-counted
void* shared_allocate(size_t size, void (*destructor)(void*));
void shared_retain(void* ptr);
void shared_release(void* ptr);
void* weak_ref_create(void* shared_ptr);
void* weak_ref_upgrade(void* weak_ptr);  // Returns NULL if released
uint32_t shared_ref_count(void* ptr);
```

---

## Compiler Changes

### New AST Operations

```c
typedef enum {
    // ... existing ops ...
    ESHKOL_WITH_REGION_OP,    // (with-region body ...)
    ESHKOL_OWNED_OP,          // (owned expr)
    ESHKOL_MOVE_OP,           // (move value)
    ESHKOL_BORROW_OP,         // (borrow value body ...)
    ESHKOL_SHARED_OP,         // (shared expr)
    ESHKOL_WEAK_REF_OP,       // (weak-ref value)
} eshkol_op_t;
```

### Ownership Analysis Pass

The compiler performs ownership analysis after parsing:

1. **Track Owned Values**: Mark variables holding owned values
2. **Check Consumption**: Ensure all owned values are consumed before scope exit
3. **Detect Moves**: Track ownership transfers
4. **Error Reporting**: Clear errors for ownership violations

```
Error: Owned value 'conn' not consumed
  at line 42: (define (bad-fn conn) ...)

  Owned values must be explicitly consumed with:
    - (close conn)      ; For file/socket resources
    - (move conn)       ; Transfer ownership
    - (consume conn)    ; Explicit consumption
```

### Escape Analysis Pass

1. **Build Escape Graph**: Track where values flow
2. **Classify Escapes**: No escape, return, closure, global
3. **Determine Allocation**: Stack, region, or shared
4. **Insert Promotions**: Add runtime calls for promotions

---

## Implementation Phases

### Phase 1: Regions (Foundation)
- Add `with-region` syntax and parsing
- Implement region stack in runtime
- Modify allocation to use current region
- Add region cleanup on scope exit

### Phase 2: Escape Analysis
- Implement escape analysis pass
- Add value promotion for returns
- Handle closure captures

### Phase 3: Owned Types
- Add `owned`, `move`, `consume`, `borrow` syntax
- Implement compile-time ownership tracking
- Add ownership violation errors
- Create `with-resource` macro

### Phase 4: Shared Types
- Add `shared`, `weak-ref` syntax
- Implement reference counting runtime
- Add weak reference support
- Implement cycle detection (optional)

### Phase 5: Integration
- Add memory profiling tools
- Optimize common patterns
- Documentation and examples

---

## Compatibility Notes

### Existing Code

All existing Eshkol code continues to work unchanged:
- Default allocation uses stack/region automatically
- No annotations required for simple programs
- Escape analysis handles most cases

### Gradual Adoption

```scheme
;;; Simple code - no changes needed
(define (add x y) (+ x y))

;;; When you need control - add annotations
(define (server port)
  (with-region
    (let ((cache (shared (make-cache))))
      ...)))
```

---

## Performance Characteristics

| Operation | Cost | Notes |
|-----------|------|-------|
| Stack alloc | O(1) | Bump pointer |
| Region alloc | O(1) | Bump pointer within region |
| Region free | O(1) | Single deallocation |
| Shared alloc | O(1) | malloc + header init |
| Shared retain | O(1) | Atomic increment |
| Shared release | O(1) | Atomic decrement + potential free |
| Ownership check | O(0) | Compile-time only |

---

*Document Version: 1.0*
*Last Updated: December 2025*
