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

## Automatic per-iteration reclamation in resident loops (ESH-0214e)

`with-region` is **no longer required** to keep a long-running loop's RSS flat.
A tail-recursive loop (a top-level self-recursive `define` or a `named-let`)
whose body allocates transient per-iteration garbage now gets **automatic
per-iteration reclamation** — the compiler brackets each iteration in a scope
and reclaims it at the back edge, so a resident tick/daemon loop stays flat
without any explicit region annotation.

Earlier, this automatic reclamation was *all-or-nothing*: the static escape
analysis rejected a loop body outright the moment it contained any persistent
mutation, because a value the iteration allocates and then stores into
outer/persistent state (a knowledge base, a workspace, a growing list) would
dangle when the per-iteration scope was rewound. A loop mutating persistent
state on **every** iteration therefore got **no** reclamation and leaked one
iteration's transient garbage forever (measured at ~3,366 bytes/tick; ~355 MB
over 100,000 ticks).

Such a mutating-but-escape-safe loop is now lowered with a **per-loop nursery
region** instead of being rejected — reusing the same deep-transitive
escape-promotion path that `with-region` uses, not a second mechanism:

- every iteration allocation lands in the nursery arena;
- each structural mutation's **existing write barrier** (`vector-set!`,
  `vector-fill!`, `hash-table-set!`, `set-car!`, `set-cdr!`) promotes any
  persistent-mutation escapee **out** of the nursery at the store — the write
  barrier acts as the generational remembered set;
- each tail-call back edge promotes the loop-carried out-values out and then
  **resets** the nursery (a deterministic generational minor collection, no
  tracing pause);
- the loop exit escapes the result and tears the nursery down.

Reclamation is sound by the same invariant as `with-region`'s `region_pop`:
after promotion, no surviving object points into the reset span. `set!` is
deliberately **not** an admitted mutation channel — its barrier fires only for
globals, and proving a `set!` target is a global rather than a shadowing
enclosing-scope local needs lexical resolution this downward-only analysis
lacks; a loop whose only persistent mutation is a `set!` falls back to the
commit-only path (correctness preserved, reclamation forgone). Non-mutating
loops keep the existing arena-scope path unchanged.

After the fix, a tick loop that mutates persistent state every iteration is flat
at 34 MB — **identical to its explicit `with-region` twin** — with every stored
value reading back correct, on JIT and AOT, and clean under
`ESHKOL_ARENA_POISON=1`. `with-region` remains available and is still the right
tool for a scratch region whose entire contents should be freed at a lexical
boundary; it is simply no longer *required* to achieve flat RSS in a resident
loop.

## Parallel workers: commit-only reclamation

Work-stealing pool workers all share the single thread-safe process arena, whose
scope stack (`arena_push_scope` / `arena_pop_scope`) is intrinsically
single-threaded — a pop rewinds the shared bump pointer and frees everything
allocated since the matching push. So during **parallel execution**, scope
operations on a shared arena degrade to **commit-only**: allocations are
retained and the shared scope stack is never rewound. This makes `parallel-map`
of a closure that allocates and returns collections (or uses an internal
named-let loop, or a builtin such as `memv`) return results **identical to
serial `map`**, with no cross-worker corruption. Per-iteration reclamation is
deferred for the duration of parallel execution only; single-threaded loops and
per-worker/region arenas keep full reclamation, so the flat-RSS behavior above
is unchanged. This is the standard "commit over reclaim = correctness over
throughput" fallback: the reclamation is what's traded away, never correctness.

> **Known follow-up:** per-thread sub-arena routing for parallel workers in the
> JIT codegen path (so each worker can reclaim its *own* scope stack rather than
> only committing) is deferred — it requires making the shared arena slot
> thread-local, a broader ABI change. Correctness does not depend on it.

## Stack and depth limits

- Region stack depth is bounded (`MAX_REGION_DEPTH`); overflow raises an error.
- The AD tape stack (`MAX_TAPE_DEPTH = 32`) is thread-local.
- See [environment variables](environment-variables.md) for `ESHKOL_MAX_HEAP`,
  `ESHKOL_MAX_STACK`, `ESHKOL_STACK_SIZE`, and `ESHKOL_WORKER_STACK_BYTES`.
