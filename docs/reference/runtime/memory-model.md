# Memory Model — Tagged Values, Arenas, and Regions

Eshkol uses **arena (region-based) memory management**, not a garbage collector.
Allocation is O(1) bump-pointer; on the native engine reclamation is bulk and
deterministic at region exit. This is a deliberate design choice for real-time,
financial, and embedded workloads where latency predictability matters — Eshkol
will never have a GC.

## Scope of the no-GC commitment

The commitment is precise, and the precision is what makes it useful. **It is a
statement about Eshkol's own semantics.** Eshkol values are never traced, never
reference-counted behind your back, and never reclaimed by a collector you did
not invoke. No Eshkol program pauses for a collection, ever.

It is *not* a statement that no collector may run anywhere in an Eshkol process.
A garbage-collected guest language — Python, Common Lisp — can be **hosted** on
regions: the guest heap is a region, the region supplies the arena, the byte
ceiling and the bulk teardown, and the guest's own collector traces the guest's
own object graph inside it. Eshkol traces nothing; the guest brings its collector
and pays for it only inside its own region, on a declared budget.

| | Eshkol's own values | A hosted guest's values |
|---|---|---|
| Traced by Eshkol | Never | Never |
| Traced at all | Never | By the guest's collector, inside the guest's region |
| Reclaimed | At region exit, in bulk | By the guest, whenever the guest decides |
| Pauses Eshkol code | Never | Never — only the guest pauses |
| Cycles reclaimed mid-scope | No (see below) | Yes, by the guest |

So the boundary is: **no GC for Eshkol, permanently; guest collectors welcome
inside their own regions.** Status: hosting is **Planned** — the adapter ABI and
its falsifier are specified in
[ADR-0011](../../design/adr/0011-guest-collector-adapter.md), sequenced behind
the OALR ABI v2 work of [ADR-0001](../../design/adr/0001-oalr-concurrent-resident.md).

Within Eshkol's own semantics one limit follows from the same theorem and is not
a defect: an unreachable **cycle** allocated inside a region is not reclaimed
until that region exits. Arbitrary aliasing, arbitrary mutation, and automatic
cycle reclamation cannot be had together without tracing, reference accounting,
or an ownership restriction. Eshkol chooses the region bound. Programs that build
cyclic garbage in a hot loop should scope it with `with-region` or a
`region-open`/`region-close` handle.

## Which engine reclaims

**Read this before relying on any reclamation claim on this page.** Eshkol has
two execution engines and they do not currently share reclamation:

| Engine | Binary | `with-region` | `region-open` / `region-close` |
|---|---|---|---|
| **Native** (LLVM JIT and AOT) | `eshkol-run` | **Yes** — the full OALR contract on this page: region arenas, escape promotion, the mutation write barrier, and per-loop nursery reclamation. | **Yes** |
| **Bytecode VM** | `eshkol-vm-standalone-test`, `eshkol-run --profile hosted-vm` | **Yes**, as of the Stage-1 region evacuator. Reclamation is real and measured; the escape semantics below hold. | **Not yet** — the handle protocol, its validation and its errors are identical to native, but a close reclaims no VM heap. Stage-2. |

Measured on the bytecode VM with the Stage-1 evacuator
(`tests/memory/vm_region_flat_rss_test.sh`, the fixture
`tests/memory/vm_region_growth_watchdog_test.esk` swept by iteration count):

| Iterations x 120 conses | Peak RSS |
|---|---|
| 1 000 | 26 MB |
| 4 000 | 26 MB |
| 16 000 | 26 MB |
| 16 000, evacuator disabled (`ESHKOL_VM_REGION_EVAC=0`) | 796 MB |

Sixteen times the work costs two megabytes: the curve is flat. Before the
evacuator the same fixture peaked at 1,503,297,536 bytes *with* the
`with-region` wrapper and 1,504,067,584 bytes *without* it — the same to within
0.06%, because the form reclaimed nothing (ledger entry SW-14 in
`.icc/silent-wrong-ledger.yaml`).

**Outside a region the VM still does not reclaim.** The global arena grows
monotonically, so a long-running VM workload that never opens a region can still
exhaust the host — and it says so, through a heap-budget diagnostic. See
[Environment variables](environment-variables.md) and
[Substrate support](#substrate-support) below.

### What the VM port does differently, and what that costs

The VM evacuator matches the native engine's *semantics*, not its
implementation. Native copies the escaping subgraph (Cheney-style, with a
forwarding map and a mutation write barrier at every store into a longer-lived
slot). The VM **marks from its root set** and sweeps at arena-block granularity
instead, because a VM value addresses the heap by a small integer index rather
than by pointer: a copying evacuator there would have to rewrite indices, and a
missed rewrite aliases a live object and produces a wrong *value* rather than a
crash. Marking moves nothing, so `eq?` identity, shared structure and cycles all
survive without special handling.

Three Stage-1 limits follow, and none of them can produce a dangling reference —
each degrades toward retaining memory:

- an escaping object with an **out-of-line payload** (a vector's element array,
  a bignum's limbs) keeps the arena block that payload occupies; escaping
  cons/closure structure is copied out exactly;
- a **continuation captured inside a region** pins that region, so it is
  promoted whole rather than freed;
- a subtype the evacuator does not classify pins its region and says so once.

**Unless a section says otherwise, every reclamation statement on this page
describes the NATIVE engine.**

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
the region exits. Both engines reclaim, by different mechanisms — see
[Which engine reclaims](#which-engine-reclaims).

### Surface syntax

```scheme
(with-region body ...)              ; anonymous region
(with-region 'name body ...)        ; named region
(with-region ('name size) body ...) ; named region + size hint (bytes)
```

At least one body expression is required, and the region name/size specifier is
**declarative**, not an expression: `('name size)` names and sizes the arena, it
is not a call. Non-final body expressions are evaluated for effect and their
values discarded, exactly as in `begin`.

The value of the last body expression is returned. Because the region's arena is
freed on exit, a returned heap value survives — natively by being **deep-copied
out** into the parent/global arena (the "escape" mechanism), on the bytecode VM
by being found reachable from the operand stack and promoted out of the swept
set. The native compiler emits
`region_create` → `region_push` → `eshkol_region_enter` → *body* →
`eshkol_region_unwind_to`, the single shared teardown primitive that also serves
`region-close`, a `raise` crossing an open region, and a continuation escape: it
promotes the kept result while the region is still current, restores the
allocation slot, then pops. Because that promotion **dispatches on the result's
tagged type**, `with-region` packs its body result before handing over the slot;
a raw unpacked literal would carry an uninitialised tag into the evacuator.

### Substrate support

Native (JIT and AOT) implements the full contract above. On the **bytecode VM**
`with-region` is value- and effect-transparent — all three spellings evaluate the
body identically and return the same value, pinned across the native, `vm-src`
and `vm-eskb` axes by `tests/vm_parity/corpus/with_region_lowering.esk` — **and,
since the Stage-1 region evacuator, it reclaims.** The compiler brackets the body
with a region push and a single teardown call
(`lib/backend/vm_compiler.c::compile_form_with_region`, native ids 2213/2214);
the teardown is the one path a `raise` crossing the body and a continuation
transfer out of it also take, so the structured and unstructured surfaces cannot
drift apart — the same discipline native keeps around
`eshkol_region_unwind_to()`.

What the VM does *not* yet reclaim is the user-reachable handle surface: a VM
`region-close` remains bookkeeping-only (see
[User-reachable region handles](#user-reachable-region-handles-341) below and
`tests/vm_parity/PARITY.tsv`), and it announces that at the point of use. That
is Stage-2.

The reclamation claim is measured, not asserted:
`tests/memory/vm_region_flat_rss_test.sh` sweeps the fixture's iteration count
and requires the peak-RSS curve to stay flat, requires a 2x separation against
the same binary with `ESHKOL_VM_REGION_EVAC=0`, and requires the printed answer
to be identical either way. Interior coverage — that every heap subtype a program
can build inside a region survives the pop with its contents intact — is gated
separately by `tests/memory/vm_region_evac_subtype_coverage_test.sh`, which reads
every promoted value back under `ESHKOL_ARENA_POISON=1`.

Outside a region the VM heap still grows monotonically, and still says so: the
arena is sampled as it grows and crossing `ESHKOL_VM_HEAP_BUDGET_MB` prints a
diagnostic naming the size, the budget and the mechanism that reclaims. It
changes no answer and is configurable — see
[Environment variables](environment-variables.md). The behaviour is gated by
`tests/memory/vm_region_growth_watchdog_test.sh`, which also pins that the
watchdog does *not* fire for a `with-region` loop that gets its memory back.

```scheme
;; Allocations inside the body are freed at region exit and the result escapes.
;; Both engines: the same total, and a flat RSS curve across iterations.
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

**Native engine only.** The AUTOMATIC half of that sentence — the per-loop
nursery that reclaims without the program asking — describes `eshkol-run` alone.
The bytecode VM has no nursery: a resident VM loop reclaims when it is wrapped in
an explicit `with-region`, and not otherwise. See
[Which engine reclaims](#which-engine-reclaims).

### What is flat, and what is not

"Flat" above is a statement about **transient** garbage, and it is exact: the
nursery reclaims every byte an iteration allocates and then drops. It is not a
statement about bytes the iteration deliberately **publishes**. Those are
different quantities, and a resident loop can have both, so the condition is
worth stating precisely rather than leaving to inference:

| Per-tick behaviour | Steady-state retention | Measured |
|---|---|---|
| Allocates transient garbage and drops it | **exactly 0 bytes/tick** | a 200-element vector per tick, 1.6 M ticks: byte-identical arena total at 200 k and 1.6 M ticks |
| Stores an immediate (or an already-persistent object) into persistent state — `vector-set!`, `hash-table-set!`, `set-cdr!`, `set!` | **exactly 0 bytes/tick** | all four channels, byte-identical arena total across an 8× horizon |
| Allocates a **fresh** heap object each tick and publishes it into persistent state | **size of the published object per tick** | 48 bytes per cons cell; 144 bytes for an 8-element vector |
| `raise`s and catches a condition each tick | **72 bytes per `raise`** | 200 k → 1.6 M raises, 71.996 bytes/raise |

Rows three and four are not leaks — they are the no-GC design boundary, and
they are the cases where a resident loop's memory is not flat. When a tick allocates a
fresh value and stores it into a persistent slot, that value must outlive the
tick, so the write barrier promotes it out of the nursery into the enclosing
arena. The value it *supersedes* — the previous occupant of that slot — is dead
the instant it is overwritten, but nothing can prove that: reclaiming it needs
either a tracing collector (which Eshkol does not have, by design) or a
uniqueness proof that the dead value is unshared. So the published bytes
accumulate at exactly the rate the program publishes them.

The fourth row is the same shape one level down: each `raise` allocates a
condition object, and for `(error ...)` that object is what the handler's
variable is bound to — user-reachable, so it cannot be recycled on the way out
of the handler without the same liveness proof. A resident loop whose error
boundary *fires* every tick therefore grows at 72 bytes/tick even though a loop
whose boundary merely *stands* every tick is exactly flat. Note the asymmetry:
entering a `guard` costs nothing (that was SW-53, fixed); raising through one
costs a condition object.

The gate fixture for this section (`tests/memory/iter_scope_partial_reclaim_test.esk`)
is a third-row program: it publishes a 3-element and a 2-element list every tick,
five cons cells, and therefore retains **240.0 bytes/tick exactly**. The
100 000-tick figure quoted above is measured and correct; what it is flat *in* is
the ~3,366 bytes/tick of transient garbage the nursery removed — a factor of
14 — not in the 240 bytes/tick the program asks to keep. Over a long enough
horizon the published bytes dominate: the same fixture measures ~385 MB of arena
at 1.6 M ticks.

**Writing a genuinely flat resident loop.** Keep the third row out of the tick.
Publish immediates or pre-allocated objects into persistent slots and mutate
those in place (`vector-set!` a number, or `set-car!`/`set-cdr!` the cells of a
list allocated once before the loop) instead of publishing a freshly consed
value each tick. Such a loop retains exactly zero bytes per tick, indefinitely —
this is gated at an 8× horizon by `tests/memory/resident_longrun_flat_gate.sh`.

**Build item — reclaiming superseded published values and spent conditions.**
Making rows three and four flat too is open work, not a documented limit to live
with forever. Both need the same missing ingredient: a liveness result strong
enough to retire an object whose last reference has just been overwritten (row
three: the previous occupant of a persistent slot) or has just gone out of scope
(row four: a condition object the handler did not keep). The natural home is a
store barrier driven by an ownership/uniqueness proof from the OALR ownership
layer; a compacting pass over persistent slots is the alternative. Tracked as
ledger entry SW-53's build item in
[`.icc/silent-wrong-ledger.yaml`](../../../.icc/silent-wrong-ledger.yaml).

## User-reachable region handles (#341)

`with-region` is the **recommended default and should stay your first choice**.
It is RAII-shaped: the region is bound to a lexical block, so it is unwind-safe
and impossible to forget to close. Reach for the handle API below only when the
shape you need genuinely has no convenient lexical body.

The motivating shape is an **autodiff training step**. The per-iteration nursery
of the previous section deliberately refuses that loop: its static escape
analysis disqualifies any body containing a `gradient`/derivative op, a `set!`,
or a `tensor-set!`, and a training step trips all three. That exclusion is by
design, not a bug — the nursery is aimed at pure or structurally-mutating tick
loops. The machinery, however, works perfectly: region entry swaps the arena the
AD tape is allocated from, so a per-step region reclaims the whole tape. What was
missing was a **non-lexical surface** for the same machinery.

### Surface

```scheme
(region-open)                 ; anonymous, default size    -> handle
(region-open 1048576)         ; size hint in bytes         -> handle
(region-open 'step)           ; debug name                 -> handle
(region-open 'step 1048576)   ; name + size hint           -> handle

(region-close handle)             ; reclaim, return '()
(region-close handle v)           ; reclaim, return the deep-promoted v
(region-close handle v1 v2 ...)   ; reclaim, return a list of promoted values

(region-open? handle)         ; #t while the handle names a live region
```

A **handle is an opaque exact integer token**. Do not do arithmetic on it, print
it as meaningful data, or store it past its close — treat it as an opaque value
whose only legal uses are `region-close` and `region-open?`. A single numeric
argument to `region-open` is read as the size hint; a single non-numeric argument
is read as the debug name. Handles are **per-thread**, like the region stack.

**On the native engine**, everything allocated between `region-open` and its
`region-close` is reclaimed at
the close. The values named in the `region-close` result list are
**deep-promoted** out first — the same transitive escape evacuator `with-region`
uses for its result, interior-pointer walk included (ESH-0214c/d), so shared
structure stays shared and no promoted object retains a pointer into the freed
arena. Values *not* named are genuinely unreachable afterwards.

Unlike `with-region`, the keep list is **explicit and mandatory**: there is no
lexical body for escape analysis to scan, so the runtime cannot infer what you
meant to keep. Anything you do not name is freed.

> **Everything means everything.** While a handle is open, the current
> allocation arena *is* the region's, so **every** value materialised in that
> window lives there — including quoted list/vector literals, interned results,
> and values bound by an enclosing `let` whose initialiser runs inside the
> window. A value is only safe past the close if it is named in the keep list or
> was created before the open. This is the one place the unstructured form is
> genuinely sharper than `with-region`, whose lexical body makes the window
> obvious at a glance; keep the window between `region-open` and `region-close`
> short and mechanical for the same reason.

```scheme
;; WRONG — `'(1 2 3)` is built while the region is open, so region-close frees it.
(let ((h (region-open)))
  (equal? '(1 2 3) (region-close h (compute))))   ; compares against freed memory

;; RIGHT — bind the close's result, then compare (or build the literal first).
(let* ((h    (region-open))
       (kept (region-close h (compute))))
  (equal? (list 1 2 3) kept))
```

```scheme
;; A training loop. `params` is handed forward; the ~123 MB/step AD tape is not.
(let loop ((step 0) (params initial-params))
  (if (= step total-steps)
      params
      (let ((h (region-open 'step)))
        (let* ((g   (gradient loss params))
               (new (update params g learning-rate)))
          (loop (+ step 1) (region-close h new))))))
```

### Safety contract

The API is unstructured, so misuse is possible in a way it is not with
`with-region`. Every misuse has a defined, memory-safe outcome — a clean
catchable error or a defined operation — and none can produce a dangling pointer
or a silently wrong answer. Each row is gated by
`tests/memory/region_handle_safety_test.esk`, run under `ESHKOL_ARENA_POISON=1`.

| Situation | Outcome |
|---|---|
| **Double close** | Clean catchable error: `region-close: invalid or already-closed region handle`. Nothing is freed twice. |
| **Use after close** (closing a handle whose region already went away) | Same clean catchable error. `(region-open? h)` is `#f` and never raises. |
| **Close out of order** (closing an outer handle while an inner one is live) | **Defined cascade**: closes the inner regions too, innermost first, promoting the kept values out at every level, and invalidates the inner handles' tokens. This is deliberately the same operation an unwind performs, so there is no ordering rule to get wrong. |
| **Never closed** | Bounded. Each open handle holds one region-stack entry; after 64 the next `region-open` raises `region-open: too many open region handles — a region-close is missing`. No unbounded growth, no corruption, and the runtime stays usable. |
| **Fabricated / stale integer passed as a handle** | Fails validation, clean catchable error. A token carries a **generation** counter alongside its slot, so a reused slot never accepts an old token, and `0`/negative values are never handles. |
| **Handle carried to another thread** | Fails validation there. Clean error, never a cross-thread free. |
| **Value not in the keep list, read after close** | Unreachable. Under `ESHKOL_ARENA_POISON=1` the arena is stamped with `0xCB` before release, so a missed escape root crashes on an obvious sentinel address instead of reading plausible stale data. |

### Unwind semantics

**A non-local exit that crosses an open region closes it.** This holds for both
`raise`/`guard` and a `call/cc` continuation escape, and for both `with-region`
bodies and open handles:

- the **in-flight value** — the raised value, or the value delivered to the
  continuation — is deep-promoted out of every region being closed *before* any
  arena is freed, so a handler never receives a pointer into freed memory;
- the allocation-routing slot is restored before the arena dies, so no
  post-unwind allocation can target a dead arena;
- the tokens of the closed handles are invalidated, so `(region-open? h)` reads
  `#f` in the handler and a later `region-close` raises cleanly rather than
  tearing down an unrelated region.

Mechanically the region depth is recorded as a mark next to the existing
`wind_mark` / `promise_mark` on the exception-handler record and on the captured
continuation state, and the unwind runs in the same place those two do. This is
also a **fix for `with-region`**: before this change a `raise` out of a
`with-region` body jumped past `region_pop` *and* `eshkol_region_leave`, leaking
the region and leaving the allocation slot pointing at an arena that was never
freed.

There is exactly **one teardown path**. Explicit close, out-of-order close,
`with-region` exit, a raise, and a continuation escape all call the same
`eshkol_region_unwind_to()` primitive — promote the kept values one level out,
restore the allocation slot, pop — so the structured and unstructured surfaces
cannot drift apart.

### Interaction with the automatic nursery

A loop containing `region-open`/`region-close` is **not** given an automatic
per-iteration nursery: the iter-scope analysis is an allowlist, and these
builtins are not on it. Your explicit handles win, and the two reclamation
mechanisms never nest unexpectedly.

### Measured

161-parameter MLP (14 → 10 → 1, tanh), full-batch `gradient` per step, fresh
parameter vector written each step; peak RSS in MB via `/usr/bin/time -l`, native
AOT. One step's tape is ~126 MB, so ~131 MB is the single-step floor. Source and
driver: `tests/memory/region_handle_training_rss.esk` /
`region_handle_training_rss_test.sh`.

| steps | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| plain loop (no scoping) | 632 | 1258 | 2510 | 5013 |
| `region-open`/`region-close`, default size hint | 343 | 605 | 847 | 879 |
| `with-region` per step, sized region | 131 | 132 | 132 | 132 |
| `region-open`/`region-close`, sized region | **132** | **132** | **132** | **132** |

All four produce **bit-identical** trained parameters. Reclamation is complete:
instrumenting the surviving arena shows it grows by exactly 2592 bytes per step —
the promoted 161-double parameter vector and nothing else.

**Size the region.** The difference between the two handle rows is not
reclamation — both reclaim everything — it is the *allocator's* high-water mark.
With the default 64 KiB hint, each step's arena reaches ~126 MB by adding
geometrically doubling blocks, and freeing that block sequence every step walks
the peak upward. A hint large enough to hold one step's work in a **single block**
keeps the address ranges stable and makes peak RSS flat. This applies equally to
`with-region`, and is the single most effective tuning knob for either form.

### Substrate support

Native (JIT and AOT) implements the full contract above. On the **bytecode VM**
the handle protocol, its validation and its error text are identical — the same C
implementation backs both — but a close still reclaims no VM heap. This is now
the ONE region surface the VM does not reclaim: the Stage-1 evacuator brackets
`with-region`, whose lexical extent tells the teardown exactly where the region
ends, whereas a handle can be closed out of order, from another dynamic extent,
or not at all. Wiring the handle surface through the same evacuator is Stage-2.
See `tests/vm_parity/PARITY.tsv`. Observable program output is byte-identical
(`tests/vm_parity/corpus/region_handle_contract.esk` pins it); only the
reclamation is absent, and `region-open` on the VM prints a one-time note saying
so. `ESHKOL_VM_REGION_QUIET=1` silences it.

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

## What LeakSanitizer sees, and what it is allowed to ignore

The `linux-x64-asan-ubsan` CI lane runs with `detect_leaks=1`. Two things had
to be true before that meant anything, and both were fixed in the leak audit:

- **A leak had to be able to fail the build.** `eshkol-run` supplied
  `__lsan_default_options()` returning `exitcode=0`, so LeakSanitizer printed
  its findings and the process exited 0 regardless. Measured before the fix: a
  hello-world `eshkol-run hello.esk -o hello` reported **248 387 bytes leaked
  in 28 748 allocations and exited 0**. The override is gone; a leak now fails.
- **The REPL had to be visible at all.** `repl_clean_exit()` ends the process
  with `std::_Exit()` — deliberately, so that static and thread-local
  destructors do not run while JIT worker threads may still hold libsystem
  locks. But LeakSanitizer installs its whole-process check as an `atexit`
  handler, so `_Exit` skipped it: the one long-lived process this project
  ships produced no leak output whether it leaked or not. The REPL now runs
  `__lsan_do_leak_check()` explicitly, after teardown and before `_Exit`.

### The audited state

Every real workload — an AOT compile, the compiled program, `-r` JIT, `--vm`,
the REPL, and the agent-FFI test binaries — was run under ASan+LSan over
`hello.esk`, `examples/h2_vibrational.esk`, `examples/autodiff.esk` and
`examples/tensors.esk`. The reports resolve to 17 distinct allocation sites:

| category | sites | disposition |
|---|---|---|
| **Runtime, VM, arena, compiled programs** | 0 | Nothing. These paths are leak-clean: every report from a compiled binary or a `--vm` run came from platform framework init, not from Eshkol code. |
| **Compiler front-end AST** | 8 | Retained for process lifetime by design (`eshkol_ast_t` has no destructor), the convention clang/rustc/gcc use. Named individually with a reason in `.icc/lsan-suppressions.txt`. Retires with epic #182. |
| **In-process JIT and driver** | 3 | **Fixed — see below.** All three grew with the work done, none was process-init. |
| **LLVM ORC JIT** | 1 | Third-party: `DynamicLibrarySearchGenerator` holds a `dlopen` handle for the life of the JITDylib. Suppressed, scoped to that class. |
| **Platform frameworks** | 7 | macOS `libobjc` / CoreFoundation / CFNetwork / libxpc process init. Not in the shipped suppression file — that file describes the Linux lane, where these frames do not exist. |

### The three that were real, and what they cost

None of these produced a wrong answer, so none is a silent-wrong entry; each is
growth, and each is now fixed rather than suppressed.

| leak | rate | cause | fix |
|---|---|---|---|
| `execute()` / `executeBatch()` result | **8 bytes per top-level form** under `-e`, `-r`, `(load …)` and `(import …)` | Both return the form's result as a heap-allocated `int64_t` on a caller-owns contract. Every `eshkol-repl` call site deleted it; six others did not — the two `eshkol-run` driver sites, the module-load and prefix-alias batches, and, worst, the top-level `SEQUENCE_OP` loop inside `execute()` itself, which assigned each sub-expression's result over `last_result` and dropped every one but the last. Measured at exactly n+1 allocations for n definitions. | All six sites delete it, like every other caller. |
| REPL variable storage | **16 bytes per distinct top-level variable** | `addModule()` allocates one 16-byte aligned slot per variable (`posix_memalign`) and registers its address as an absolute JIT symbol, keeping it in `repl_var_storage_` as a raw `void*`. Destroying the map dropped the pointers. Its sibling map `forward_ref_slots_` was already freed in `~ReplJITContext`; this one was missed. | Freed in `~ReplJITContext`, after `jit_.reset()` so nothing can still name the address. |
| driver path strings | ~1.8 KB per invocation | `compiled_files` and `output` hold `char*` of mixed ownership — some `argv[]` slices, some `strdup()`ed here — so neither could be freed as a whole. | The driver-owned strings come from a static, reachable driver-lifetime store. Rooted, not suppressed. |

A fourth defect was fixed alongside them without being a leak in its own right:
`addModule()` decided whether a `__repl_fwd_` slot already existed by asking
the JIT to resolve the symbol, which also answers "no" when the slot exists but
its module's `ResourceTracker` was evicted by hot reload. It then allocated a
*second* slot for a name whose address is baked into every module already
linked against it. It now reuses the existing slot, which is both leak-free and
the address stability the mechanism depends on.

### The one retention that is not flat

The front-end suppressions are scoped to one function each, but within a
function they are total, so a new leak inside `parse_list()` would be swallowed
by them. `tests/memory/leak_audit_gate.sh` closes that: it measures front-end
retention per REPL input line at two horizons **with suppressions off** and
gates on the slope.

Measured at 10 / 20 / 40 input lines: 21 171 / 37 451 / 70 009 bytes, i.e.
**1628 bytes per input line, linear across a 4× span**. A batch compile reaps
that at exit; a long-running REPL session does not.

The figure is expected to move when the front end legitimately changes shape,
and the gate is sensitive enough to show it: the audit first measured 1546.0
bytes per line, and rebasing onto a later master raised it to 1628 — the window
contained the NodeId identity substrate, which attaches identity data to every
parsed node, so more retained per form is the expected consequence rather than
a regression. Both numbers are recorded in the gate so the increase is
accounted for rather than silently absorbed by its tolerance. This is growth, not a wrong answer, so it is a defect rather
than a silent-wrong, and the honest statement of it is: **`eshkol-run` is
bounded by its input, `eshkol-repl` is not**. The number is pinned in the gate
so it can go down freely and cannot go up. Giving `eshkol_ast_t` a real owner
(epic #182) is what makes it zero; that is a 353-site ownership change across
the parser and macro expander, tracked separately.
