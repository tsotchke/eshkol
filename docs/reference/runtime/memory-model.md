# Memory Model — Tagged Values, Arenas, and Regions

Eshkol uses **arena (region-based) memory management**, not a garbage collector.
Allocation is O(1) bump-pointer; reclamation is bulk and deterministic at region
exit. This is a deliberate design choice for real-time, financial, and embedded
workloads where latency predictability matters — Eshkol will never have a GC.

## Tagged values (16 bytes)

Every Eshkol value is a fixed-width tagged value (`inc/eshkol/eshkol.h`):

```c
typedef struct eshkol_tagged_value {
    uint8_t  type;      // eshkol_value_type_t
    uint8_t  flags;     // exactness + other flags
    uint16_t reserved;
    union { int64_t int_val; double double_val; uint64_t ptr_val; uint64_t raw_val; } data;
} eshkol_tagged_value_t;   // 16 bytes (asserted <= 16)
```

The layout is `{type:8, flags:8, reserved:16, data:64}`; natural alignment pads
the struct to 16 bytes. A cons cell stores two full 16-byte tagged values (car,
cdr). Type tags include int, double, bool, char, complex, heap-ptr, logic-var,
dual-number, etc.; the `flags` byte carries exactness for the numeric tower.

Heap objects (vectors, tensors, strings, records) are prefixed with an 8-byte
`eshkol_object_header_t` that records a subtype and flags — the value's `data`
field is a pointer to the payload after the header.

## Arena allocator

The arena API lives in `lib/core/arena_memory.h` (impl in `lib/core/`).

| Function | Purpose |
|----------|---------|
| `arena_create(block_size)` | New growable arena |
| `arena_create_threadsafe(size)` | Mutex-guarded arena (the global arena) |
| `arena_create_bounded(capacity)` | Single fixed block, no-grow (embedded seam); returns NULL on overflow |
| `arena_allocate(a, n)` / `_aligned` / `_zeroed` | Bump-pointer allocation |
| `arena_allocate_with_header(a, n, subtype, flags)` | Allocate a header'd heap object |
| `arena_push_scope(a)` / `arena_pop_scope(a)` | LIFO sub-scope save/restore |
| `arena_reset(a)` | Reset to empty |
| `arena_destroy(a)` | Free the arena |

### Hybrid arena model

- A single **global arena** (created at 64 KiB, thread-safe) backs the main
  thread.
- Each **parallel worker** gets its own **per-thread arena** (1 MB block, lazily
  allocated) via `arena_create_thread_local` — zero cross-thread contention.
  `arena_merge_to_parent` publishes results back.

## Regions and `with-region`

Regions are lexically-scoped arenas layered on top of the allocator
(`lib/core/runtime_regions.cpp`). A thread-local region stack tracks the active
region; allocations route to the current region's arena and are reclaimed when
the region exits.

### Surface syntax

```scheme
(with-region body ...)              ; anonymous region
(with-region 'name body ...)        ; named region
(with-region ('name size) body ...) ; named region + size hint (bytes)
```

At least one body expression is required. The value of the last body expression
is returned; because the region's arena is freed on exit, a returned heap value
is **deep-copied out** into the parent/global arena so it survives (the "escape"
mechanism). The compiler emits `region_create` → `region_push` → *body* →
`region_pop`; `region_pop` destroys the region arena, reclaiming everything
allocated inside.

```scheme
;; Allocations inside the body are freed at region exit; the result escapes.
(define total
  (with-region ('scratch 65536)
    (let loop ((i 0) (acc 0))
      (if (= i 1000000)
          acc
          (loop (+ i 1) (+ acc i))))))
```

### Region reclamation fix (PR #81 / ESH-0039)

Before this fix, `with-region` reclaimed **nothing**: the LLVM backend emitted
every allocation against the single global-arena global variable, so a body's
cons/vector/closure/tape allocations landed in the never-freed global arena while
`region_pop` freed an empty region. Long-running loops grew RSS unboundedly
(~4 GB/epoch → OOM). The fix routes all body allocations through the region arena
by swapping the arena slot on region entry and restoring it after `region_pop`
(all ~200 allocation sites now target the region arena transparently, funneled
through `currentArenaPtr()`; `arena_push_scope`/`pop_scope` honor the same slot).
Boundary preservation: objects created *outside* a region record their `home_arena`
so they survive `region_pop` even if mutated inside it (e.g. a hash table resized
inside a region re-allocates its backing arrays in its home arena). Measured: peak
RSS over 200k iterations dropped from ~3153 MB to ~42 MB.

> **Known follow-up:** per-thread sub-arena routing for parallel workers in the
> JIT codegen path is deferred — it requires making the shared arena slot
> thread-local, a broader ABI change.

## Stack and depth limits

- Region stack depth is bounded (`MAX_REGION_DEPTH`); overflow raises an error.
- The AD tape stack (`MAX_TAPE_DEPTH = 32`) is thread-local.
- See [environment variables](environment-variables.md) for `ESHKOL_MAX_HEAP`,
  `ESHKOL_MAX_STACK`, `ESHKOL_STACK_SIZE`, and `ESHKOL_WORKER_STACK_BYTES`.
