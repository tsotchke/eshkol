# ADR-0011: Hosted guest collectors over OALR regions

- **Status:** Proposed
- **Date:** 2026-08-25
- **Decision owners:** Eshkol memory/runtime architecture
- **Cluster:** OALR / language hosting / interop
- **Applies to:** the native LLVM AOT/JIT engine first; the bytecode VM implements the same semantic contract at its own stage
- **Supersedes:** Blocker 1 of `ESHKOL-CL-AUTOLITH-FEASIBILITY-2026-08-25` ("Eshkol has no garbage collector and has ruled one out permanently")
- **Related:** ADR-0000 (unified trajectory), ADR-0001 (concurrent resident OALR), ADR-0004 (types / ownership HIR), ADR-0006 (modules), ADR-0009 (DBSP)

---

## Decision

Eshkol will host garbage-collected guest language runtimes by giving each guest
heap **a region of its own**. The region supplies the arena, the lifetime bound,
the allocation seam, the teardown trigger, and the residence discipline. The
guest supplies the tracing over its own object graph. Eshkol never traces the
guest graph and never frees an individual guest object; the guest never traces,
moves, or frees an Eshkol object.

The two heaps meet at exactly one place: a **pinned handle table**, owned by the
adapter, allocated in the guest's region, and a mandatory root of every guest
collection. Eshkol-to-guest edges are handles and nothing else. Guest-to-Eshkol
edges are opaque tagged values carrying a **residence obligation** — the referent
must live strictly outside the guest's region.

This is an adapter, not a retreat from the no-GC commitment. It is available
because of a scope distinction that ADR-0001's theorem states but does not
make explicit:

> **ADR-0001 constrains what Eshkol's own semantics can be. It does not constrain
> what a runtime hosted on top of Eshkol's regions can be.**

The theorem in ADR-0001 reads: *arbitrary R7RS aliasing + mutation + automatic
reclamation of unreachable cycles cannot simultaneously be provided with no
tracing, no reference accounting, and no ownership restriction*
([`0001-oalr-concurrent-resident.md:18`](0001-oalr-concurrent-resident.md)). Every
term in it quantifies over the *Eshkol* object graph. A CPython heap or a Common
Lisp heap is a different graph, with a different collector, in a different
arena, paid for by a different program. Hosting one costs Eshkol nothing it has
promised not to pay: no tracing over Scheme values, no pauses in Scheme code, no
reachability walk that a Scheme program did not ask for by instantiating a guest.

The commitment therefore sharpens rather than weakens: **Eshkol's own semantics
are GC-free, permanently. A guest brings its own collector and pays for it only
inside its own region.**

---

## 1. Source baseline: what regions already provide

The adapter is unusually cheap because four of the five things a guest heap needs
are already shipped and load-bearing. Every citation below is against
`origin/master` at `e65cb5b5`.

| Requirement | Already present | Evidence |
|---|---|---|
| An arena with a bulk, O(1) teardown | `arena_create` / `arena_destroy`, block-chain free with optional poison | [`lib/core/runtime_arena_core.cpp:112`](../../../lib/core/runtime_arena_core.cpp#L112), [`:191`](../../../lib/core/runtime_arena_core.cpp#L191) |
| A hard byte ceiling that fails deterministically instead of growing | `arena_create_bounded` — single fixed block, `arena_allocate*` returns NULL on exhaustion | [`lib/core/runtime_arena_core.cpp:147`](../../../lib/core/runtime_arena_core.cpp#L147) |
| One allocation seam a guest must respect | `eshkol_current_arena()` — the thread-local allocation accessor all generated code routes through | [`lib/core/runtime_regions.cpp:176`](../../../lib/core/runtime_regions.cpp#L176), [`lib/core/arena_memory.h:361`](../../../lib/core/arena_memory.h#L361) |
| A **generation-counter handle protocol** with defined stale-token behavior | `region-open` / `region-close` / `region-open?`: "a slot index plus a GENERATION counter, not a pointer; closing a handle bumps its slot's generation, so every stale token ... fails validation and raises a clean catchable error instead of touching freed memory" | [`lib/core/arena_memory.h:520-545`](../../../lib/core/arena_memory.h#L520), [`:572`](../../../lib/core/arena_memory.h#L572), [`:577`](../../../lib/core/arena_memory.h#L577) |
| **One** teardown path shared by lexical exit, explicit close, `raise`, and `call/cc` escape | `eshkol_region_unwind_to()` — "the ONLY code that" tears a region down, reached from `with-region`, `region-close`, `eshkol_exception_handler_t::region_mark`, and `eshkol_continuation_state_t::region_mark` | [`lib/core/runtime_regions.cpp:1863`](../../../lib/core/runtime_regions.cpp#L1863), [`:1971`](../../../lib/core/runtime_regions.cpp#L1971), [`inc/eshkol/eshkol.h:1351`](../../../inc/eshkol/eshkol.h#L1351), [`:1498`](../../../inc/eshkol/eshkol.h#L1498) |
| A copying collector, already, over a restricted graph | "The evacuator is a Cheney-style copying collector restricted to the escaping subgraph" | [`lib/core/runtime_regions.cpp:868`](../../../lib/core/runtime_regions.cpp#L868) |

That last row is the architectural point in one line. Eshkol already runs a
Cheney copy — bounded, root-directed, forwarding-mapped, cycle-terminating —
whenever a value escapes a region
([`lib/core/runtime_regions.cpp:859-881`](../../../lib/core/runtime_regions.cpp#L859)).
The difference between that and a guest collector is **scope and trigger**, not
algorithm: Eshkol's copy is driven by the escape boundary and runs at region
exit; a guest's copy is driven by its own root set and runs when the guest
decides. Two Cheney collectors over two disjoint arenas do not interfere. They
are already the same shape.

The fifth requirement — a per-region finalizer — is the one thing absent.
`region_destroy` frees the arena wholesale with zero callbacks
([`lib/core/runtime_regions.cpp:436`](../../../lib/core/runtime_regions.cpp#L436)).
The only per-object destructor in the tree is refcount-driven and orthogonal to
regions (`eshkol_shared_header_t::destructor`,
[`lib/core/arena_memory.h:704`](../../../lib/core/arena_memory.h#L704),
invoked from `shared_release`). Section 6 specifies the region-scoped form.

### The two object-model bits that were reserved for this

```c
#define ESHKOL_OBJ_FLAG_PINNED    0x40  // Pinned in memory (no relocation)
#define ESHKOL_OBJ_FLAG_EXTERNAL  0x80  // External resource (needs explicit cleanup)
```
[`inc/eshkol/eshkol.h:482-483`](../../../inc/eshkol/eshkol.h#L482)

Both are declared and **read nowhere in the evacuator today**. They are the
correct bits, reserved for exactly this purpose, and the adapter is what makes
them load-bearing. `heap_subtype_t` currently allocates 0..25
([`inc/eshkol/eshkol.h:545-547`](../../../inc/eshkol/eshkol.h#L545)), leaving
26..255 for guest subtypes.

---

## 2. Required semantics

The adapter is governed by seven invariants. They extend ADR-0001's ten; where
they overlap, ADR-0001 wins.

1. **Guest containment.** Every byte of a guest heap is allocated from the arena
   of exactly one region, its **anchor region**. A guest never allocates through
   `malloc`, `mmap`, or `sbrk` for heap storage.
2. **No Eshkol tracing.** Eshkol's evacuator never enters a guest object. The
   guest handle subtype classifies as a leaf and is never deep-walked.
3. **No guest tracing of Eshkol.** A guest's tracer treats an
   `eshkol_tagged_value_t` as 16 opaque bytes: copied verbatim, never followed,
   never rewritten.
4. **Handles are the only inbound edge.** No raw pointer into a guest heap may
   ever be stored in an `eshkol_tagged_value_t`. Violation is a defined,
   detected fault, not undefined behavior (Section 4.1).
5. **Residence obligation on outbound edges.** An Eshkol value reachable from a
   guest object must reside strictly outside the anchor region: an ancestor
   region, the immortal domain, or a resident generation. Violation is caught at
   publication, and audited at collection in checked builds.
6. **Anchored lifetime.** A guest heap dies with its anchor region, through the
   one teardown path, wholesale. Individual guest objects are never freed by
   Eshkol.
7. **Budgeted.** A guest heap declares its byte ceiling at attach. Exceeding it
   is a deterministic guest-side allocation failure, never an Eshkol-side
   unbounded growth. `arena_create_bounded`
   ([`lib/core/runtime_arena_core.cpp:147`](../../../lib/core/runtime_arena_core.cpp#L147))
   is the mechanism.

Invariant 7 is what makes the whole thing acceptable to a real-time user. A
program that instantiates no guest pays nothing. A program that instantiates one
pays a *declared, bounded, inspectable* amount, in a region it named.

---

## 3. The interface

### 3.1 ABI

```c
#define ESHKOL_GUEST_ABI_V1 1u

typedef struct eshkol_guest_heap eshkol_guest_heap_t;

/* {generation:32, index:32}. Never a pointer. 0 is the null handle.
 * Same construction as the shipped region-handle token
 * (arena_memory.h:520-545): closing bumps the generation, so every stale
 * token fails validation instead of touching freed memory. */
typedef uint64_t eshkol_guest_handle_t;

typedef enum eshkol_guest_flags {
    ESHKOL_GUEST_ANCHORED   = 0,      /* handles may not outlive the anchor */
    ESHKOL_GUEST_PROMOTABLE = 1u<<0,  /* heap may be re-anchored outward     */
    ESHKOL_GUEST_MOVING     = 1u<<1,  /* collector relocates objects         */
    ESHKOL_GUEST_THREADED   = 1u<<2,  /* guest may create its own threads    */
} eshkol_guest_flags_t;

typedef struct eshkol_guest_vtable {
    uint32_t    abi_version;      /* ESHKOL_GUEST_ABI_V1 */
    const char *name;             /* "cpython-3.13", "ecl-24.5", ... */
    uint32_t    flags;            /* eshkol_guest_flags_t */

    /* Run one collection over the guest's own graph, rooted at the handle
     * table the adapter owns. Returns bytes reclaimed. Eshkol calls this; it
     * never enters it. */
    size_t (*collect)(void *guest, eshkol_guest_heap_t *h);

    /* Re-anchor: copy the live set into slabs the adapter has allocated from
     * `target`, rewriting the handle table. MOVING guests only; NULL otherwise,
     * which forces ANCHORED. */
    int    (*relocate)(void *guest, eshkol_guest_heap_t *h, arena_t *target);

    /* The anchor region is dying. Release OS resources the guest owns OUTSIDE
     * the arena (fds, dlopen handles, foreign mallocs, threads). Must not free
     * arena memory: arena_destroy does that in one step. Must not raise. */
    void   (*on_teardown)(void *guest, eshkol_guest_heap_t *h);

    /* Enumerate guest -> Eshkol edges so the adapter can audit residence. */
    void   (*enumerate_outbound)(void *guest,
                                 void (*visit)(void *ctx, const eshkol_tagged_value_t *),
                                 void *ctx);
} eshkol_guest_vtable_t;

/* Attach a guest heap to the CURRENT region. `budget` is a hard ceiling in
 * bytes; the adapter takes it from arena_create_bounded, so guest allocation
 * fails deterministically at the ceiling instead of growing Eshkol's footprint.
 * Registers the heap on the region's teardown list. */
int eshkol_guest_attach(const eshkol_guest_vtable_t *vt, void *guest,
                        size_t budget, eshkol_guest_heap_t **out);

/* Guest allocation. The ONLY way a guest obtains heap storage. */
void *eshkol_guest_alloc(eshkol_guest_heap_t *h, size_t n, size_t align);

/* Root protocol. A pin is an affine capability: pin/unpin must balance, and an
 * unbalanced pin at teardown is a diagnosable leak, not a crash. */
eshkol_guest_handle_t eshkol_guest_pin  (eshkol_guest_heap_t *h, void *obj);
void                  eshkol_guest_unpin(eshkol_guest_heap_t *h, eshkol_guest_handle_t);
void                 *eshkol_guest_deref(eshkol_guest_heap_t *h, eshkol_guest_handle_t);

/* Root enumeration, for the guest's tracer. The handle table IS the root set. */
void eshkol_guest_roots(eshkol_guest_heap_t *h,
                        void (*visit)(void *ctx, void **slot), void *ctx);

/* Publish an Eshkol value into the guest, checking the residence obligation.
 * Returns non-zero and does not publish if `v` lives at-or-inside the anchor. */
int eshkol_guest_publish(eshkol_guest_heap_t *h, eshkol_tagged_value_t v,
                         eshkol_tagged_value_t *slot);

size_t eshkol_guest_collect(eshkol_guest_heap_t *h);
```

### 3.2 The allocation contract

`eshkol_guest_alloc` does **not** forward to `eshkol_current_arena()`. It draws
from the heap's own bounded arena, captured at attach. This is deliberate and it
is the difference between an adapter and a leak: `eshkol_current_arena()` follows
the *Eshkol* program's region nesting
([`lib/core/runtime_regions.cpp:176`](../../../lib/core/runtime_regions.cpp#L176)),
and during a parallel scope it is pinned to the shared process arena. A guest
whose allocation followed that would scatter its heap across arenas with
unrelated lifetimes, and its collector would have no slab to sweep. The guest
heap is anchored once, at attach, and stays anchored.

The corollary is a hard rule for guest embedders: **a guest runtime that
allocates through its own `malloc` is not hosted, it is merely linked.** Every
guest in Section 8 is viable precisely because it already has an allocator seam
(CPython's arena allocator; Boehm's `GET_MEM`) that can be pointed at
`eshkol_guest_alloc`.

### 3.3 Interaction with Eshkol's write barriers

Eshkol's structural write barrier promotes a value's in-region subgraph when it
is stored into a destination that outlives it
([`lib/core/runtime_regions.cpp:1671`](../../../lib/core/runtime_regions.cpp#L1671)).
Its fast path is a single thread-local load and branch
([`:1681`](../../../lib/core/runtime_regions.cpp#L1681)), then a
`region_index_owning` comparison of value against destination.

The interaction with a guest's own barriers is **none, by construction**, and
this is the cleanest consequence of the handle rule:

- A guest's write barrier (generational card marking, CPython's `Py_INCREF`,
  Boehm's dirty bits) fires on stores between two *guest* objects. Those objects
  are not `eshkol_tagged_value_t`, are never destinations Eshkol's barrier sees,
  and live in an arena Eshkol's evacuator will not enter. Eshkol's barrier is not
  invoked and its fast path is not even reached.
- Eshkol's barrier fires on stores between two *Eshkol* values. Guest handles
  participate as **leaves**: a handle-carrying value stores like a port stores.
- The one crossing case — storing a handle into a destination that outlives the
  anchor region — is not a barrier interaction but a **lifetime** question, and
  Section 6.2 answers it.

Two barriers over two disjoint object sets do not compose badly, because they
never see the same store.

There is, however, a real hazard in the existing barrier that the ABI must dodge.
The closure evacuation path carries a conservative heuristic: an `INT64` capture
whose value reinterpreted as a pointer lands inside a dying region is treated as
a 16-byte mutable cell and copied
([`lib/core/runtime_regions.cpp:1379`](../../../lib/core/runtime_regions.cpp#L1379)).
The in-source justification is that "the other producers of INT64-packed
pointers in capture slots are `GlobalVariable` addresses and JIT/AOT code
addresses, neither of which is ever inside a region arena." A guest handle stored
as a bare `INT64` would be a new producer, and the assumption behind that
heuristic would no longer hold by construction. **Therefore guest handles get a
dedicated heap subtype, not a bare integer** — see Section 4.1. This is a small
decision that would have been an ugly bug.

---

## 4. The boundary

Cross-heap references are where FFI/GC integrations die. The decision is
**asymmetric**, because the two directions have genuinely different hazards.

### 4.1 Eshkol to guest: handles only

A reference from Eshkol into a guest heap is a value of a new heap subtype:

```c
HEAP_SUBTYPE_GUEST_HANDLE = 26,   /* first free slot; eshkol.h:545 */

typedef struct eshkol_guest_ref {
    eshkol_guest_heap_t   *heap;    /* which guest                 */
    eshkol_guest_handle_t  handle;  /* {generation:32, index:32}   */
} eshkol_guest_ref_t;
```

allocated through `arena_allocate_with_header`
([`lib/core/arena_memory.h:99`](../../../lib/core/arena_memory.h#L99)) with
`ESHKOL_OBJ_FLAG_EXTERNAL | ESHKOL_OBJ_FLAG_PINNED`
([`inc/eshkol/eshkol.h:482-483`](../../../inc/eshkol/eshkol.h#L482)), and
classified by a new explicit arm in `evac_kind_for`
([`lib/core/runtime_regions.cpp:1024`](../../../lib/core/runtime_regions.cpp#L1024)):

```c
case HEAP_SUBTYPE_GUEST_HANDLE: return EVAC_GUEST_HANDLE;
```

`EVAC_GUEST_HANDLE` copies the two-word payload verbatim and walks nothing. It is
behaviorally identical to the treatment ports already receive — "wraps an OS
fd/FILE\*; handle intentionally shared, not copied"
([`lib/core/runtime_regions.cpp:1026-1028`](../../../lib/core/runtime_regions.cpp#L1026)) —
which is the right precedent, because a guest handle is exactly that: a name for
a resource whose identity must survive copying and whose storage must not be
duplicated.

It is deliberately **not** left to fall through the `default: return EVAC_LEAF`
arm ([`:1115`](../../../lib/core/runtime_regions.cpp#L1115)). That arm would do
the right thing today by accident, and ADR-0001 invariant 6 forbids exactly that:
*"Every pointer-carrying allocation has a header and an exact layout descriptor.
'Unknown means leaf' is forbidden."* The existing debug watchdog that fires when
a watch-listed subtype is leaf-copied out of a region gets a `GUEST_*` entry of
the same shape.

**Why handles and not raw pointers, decisively.** Three independent reasons, in
increasing order of how expensive they are to discover the hard way:

1. **Moving collectors.** A Cheney or compacting guest relocates objects on every
   cycle. A raw pointer held on the Eshkol side is a use-after-move the instant
   the guest collects. Falsifier E2 (Section 9) demonstrates exactly this: the
   pinned handle resolves correctly to the moved object while the raw pointer
   captured before the collection is stale.
2. **Root discovery.** Eshkol has no shadow stack, no root set, and no
   conservative stack scanner. Ownership is probed by address-range scanning of
   the thread's region stack
   ([`lib/core/runtime_regions.cpp:897`](../../../lib/core/runtime_regions.cpp#L897)),
   which answers "which region owns this address," not "who is pointing at it."
   A guest collector needs the second question answered, and nothing in the tree
   can answer it. The handle table answers it by construction: it *is* the
   answer, maintained by the mutator at pin time. This is not a workaround; for a
   runtime with no stack maps, an explicit root table is the only sound design.
3. **The evacuator would misread it.** This is the concrete failure mode, and it
   is worth spelling out because it is silent rather than loud. The guest heap's
   slabs are allocated from the anchor region's arena, so
   `region_index_owning` classifies a guest object as *owned by that region*
   ([`lib/core/runtime_regions.cpp:897`](../../../lib/core/runtime_regions.cpp#L897)).
   If a raw guest pointer were stored in a tagged heap value and then escaped,
   `evac_value` would find it inner to the boundary
   ([`:1203`](../../../lib/core/runtime_regions.cpp#L1203)) and hand it to
   `evac_object` ([`:1137`](../../../lib/core/runtime_regions.cpp#L1137)), which
   reads an `eshkol_object_header_t` from the eight bytes *before* the pointer —
   guest payload, not a header. Three outcomes follow, all bad and none of them a
   clean crash: `h->size == 0` leaves the pointer in place and it goes stale at
   the guest's next collection; a garbage `h->size` above 256 MB trips the
   plausibility guard, warns, and also leaves it in place; a plausible-but-wrong
   size memcpies an arbitrary run of guest bytes into the parent arena and
   publishes a pointer to it. **A wrong answer, not a fault.** The handle
   subtype removes the possibility rather than detecting it.

**Failure mode if a user violates the rule.** They cannot, from Eshkol source:
the surface exposes no operation that yields a raw guest address. They can from
hand-written C through `eshkol_ffi.h`. The mitigations, in order of strength:
the debug watchdog fires on any `GUEST_*` subtype taking a leaf path it should
not; arena poisoning fills freed region bytes with `0xCB`
([`lib/core/runtime_arena_core.cpp:191`](../../../lib/core/runtime_arena_core.cpp#L191))
so a stale guest read is loud under a poison build; and `eshkol_guest_deref` of a
stale handle returns NULL with a violation counter incremented, because the
generation bumped — the same defined-error contract `region-close` already
provides for stale region tokens
([`lib/core/arena_memory.h:527-533`](../../../lib/core/arena_memory.h#L527)).

### 4.2 Guest to Eshkol: opaque values under a residence obligation

A guest object may embed an `eshkol_tagged_value_t` directly. The guest tracer
copies those 16 bytes and does not follow them. No handle indirection is needed
in this direction, because Eshkol values do not move once allocated — the
evacuator copies on *escape*, and an escape only ever moves a value **outward**,
to an arena that outlives the current one. So the obligation is not "do not
move," it is "do not die":

> **Residence obligation.** An Eshkol value reachable from a guest object must
> reside strictly outside the anchor region.

If it holds, the referent's arena outlives the guest heap by construction, and no
promotion can ever invalidate a guest-held pointer, because promotion targets are
selected by `region_escape_target` walking outward from `escape_base`
([`lib/core/runtime_regions.cpp:748`](../../../lib/core/runtime_regions.cpp#L748))
and a value already outside the anchor is never a promotion source.

Enforcement is two-layer. `eshkol_guest_publish` checks
`region_index_owning(ptr)` against the anchor index and refuses inward
references at the point of publication — a clean error at the moment of the
mistake, in every build. In checked builds, `enumerate_outbound` is additionally
walked at each collection and at teardown, so a guest that smuggles a value past
`publish` is caught at the next collection rather than at the eventual dangle.

The common case costs nothing: an Eshkol value handed to a guest is almost
always either immortal (interned symbols, string literals, the global arena) or
resident in a region enclosing the one where the guest was instantiated, which is
the natural shape of `(with-region 'python ...)`.

### 4.3 Alternatives rejected

- **Handles in both directions.** Symmetric and safe, and rejected for cost:
  Eshkol values are stable, so the indirection buys nothing and would put a table
  lookup on every guest field read. Asymmetry here is earned, not sloppy.
- **A shared immutable subset neither side collects.** Attractive on paper and
  useless in practice for the two named targets. Both Common Lisp and Python are
  mutation-centric; an immutable-only boundary would exclude the CLOS instance,
  the Python object, the hash table, and the list — that is, everything anyone
  wants to pass. It survives as a *fast path*: an immortal-resident Eshkol value
  needs no residence check at all, and the adapter special-cases it.
- **Forbidding guest-to-Eshkol entirely.** Kills the use case. The reason to host
  Python is to call Eshkol from it.
- **Conservative stack scanning on the Eshkol side (Boehm-style) to discover
  roots.** Would let raw pointers work for non-moving guests. Rejected: it
  imports precisely the property ADR-0001 rules out — a reachability walk over
  Eshkol memory that the program did not request — and it does so for *all*
  programs, including those with no guest. The whole value of this design is that
  the cost is confined to the region that asked for it.

---

## 5. Cycles: why the composition holds

The one thing regions genuinely cannot do is reclaim an unreachable cycle
mid-region. A guest collector can, inside its own region. The argument that this
composes is four steps, and each step is checkable against code that exists.

**Step 1 — the guest's trace terminates and is complete over its own graph.**
Standard, and independent of Eshkol. A tracing collector with a forwarding
pointer or a mark bit visits each object once; cycles terminate because a
revisited object answers with its forwarding pointer. This is the same argument
Eshkol's own evacuator relies on
([`lib/core/runtime_regions.cpp:868-872`](../../../lib/core/runtime_regions.cpp#L868)).

**Step 2 — the guest's root set is complete.** The guest's roots are its own
stack and statics, plus the adapter's handle table. Completeness of the second
part is the invariant that matters, and it is structural rather than analytic: a
handle exists only because `eshkol_guest_pin` created it, and pinning is the only
operation that produces an Eshkol-side reference at all (Section 4.1). There is
no second channel to miss. Contrast a raw-pointer design, whose root-set
completeness would depend on finding every raw pointer anywhere in the Eshkol
heap and stack — which, given that Eshkol has no stack maps and no root registry,
is not a property anyone could establish.

**Step 3 — Eshkol's evacuator never enters the guest graph.** `evac_kind_for`
returns `EVAC_GUEST_HANDLE` for the handle subtype, which walks nothing, so the
worklist in `region_evacuate_value`
([`lib/core/runtime_regions.cpp:1236`](../../../lib/core/runtime_regions.cpp#L1236))
never receives a guest object. No other path can introduce one, because
`evac_value` only ever follows pointers it extracts from tagged values
([`:1203`](../../../lib/core/runtime_regions.cpp#L1203)) and raw pointer fields
of *known* layouts (`evac_object_ptr`,
[`:1224`](../../../lib/core/runtime_regions.cpp#L1224)) — and no known layout has
a guest field.

**Step 4 — the guest's collection cannot invalidate an Eshkol value.** Guest
collection moves and frees guest objects only. Embedded Eshkol values are copied
verbatim as opaque bytes (invariant 3) and their referents live outside the
anchor region (invariant 5, Section 4.2), so no Eshkol object's liveness or
address is a function of anything the guest does.

Therefore: guest cycles are reclaimed by the guest, mid-region, while Eshkol's
resident set is untouched and Eshkol's trace count stays zero. Falsifier F1
(Section 9) exhibits this at 600,000 objects through a 256 KiB heap with a host
arena delta of exactly zero.

### 5.1 What would have to change — and it is a short list

The argument above is sound against the shipped evacuator, but it is not free.
Six changes are required, and none of them is speculative:

1. **`evac_kind_for` gains an `EVAC_GUEST_HANDLE` arm**
   ([`lib/core/runtime_regions.cpp:1024`](../../../lib/core/runtime_regions.cpp#L1024)).
   Explicit, not by fallthrough, per ADR-0001 invariant 6.
2. **`region_destroy` gains a teardown sweep**
   ([`lib/core/runtime_regions.cpp:436`](../../../lib/core/runtime_regions.cpp#L436)),
   reached through `eshkol_region_unwind_to`
   ([`:1971`](../../../lib/core/runtime_regions.cpp#L1971)) so that a `raise` or
   `call/cc` escape out of a guest's region tears the guest down exactly once, on
   the same path everything else uses.
3. **The write barrier gains a `GUEST_HANDLE` case**
   ([`lib/core/runtime_regions.cpp:1671`](../../../lib/core/runtime_regions.cpp#L1671))
   implementing the anchor rule of Section 6.2. Without it, a handle stored
   outward is copied as a leaf into the outer arena and its table dies underneath
   it — the escape bug in its purest form. **This is required, not optional.**
4. **`ESHKOL_OBJ_FLAG_EXTERNAL` and `ESHKOL_OBJ_FLAG_PINNED` become read**
   ([`inc/eshkol/eshkol.h:482-483`](../../../inc/eshkol/eshkol.h#L482)), where
   today they are declared and inspected nowhere.
5. **Guest handles must not be bare `INT64`**, to keep the closure-capture
   heuristic's stated assumption true
   ([`lib/core/runtime_regions.cpp:1379`](../../../lib/core/runtime_regions.cpp#L1379)).
   Section 4.1 already takes this route; it is recorded here because it is a
   constraint discovered from the implementation, not a preference.
6. **`region_index_owning` becomes a hot path**
   ([`lib/core/runtime_regions.cpp:897`](../../../lib/core/runtime_regions.cpp#L897)).
   It is a linear scan of the region stack, and each entry walks a block chain.
   A guest that publishes many values pays for it in `eshkol_guest_publish`. This
   is a **cost**, not a correctness issue, and ADR-0001's residence tagging in
   ABI v2 removes it by making residence a metadata read instead of a search —
   which is one of the reasons this ADR is sequenced behind Stage 4 (Section 7).

---

## 6. Lifetime and teardown

### 6.1 Wholesale, never walked

A guest heap dies with its anchor region, in one `arena_destroy`
([`lib/core/runtime_arena_core.cpp:191`](../../../lib/core/runtime_arena_core.cpp#L191)).
No walk, no finalizer storm, no collection at exit. This is the single largest
advantage arena hosting has over hosting on a general-purpose GC heap, and it is
worth being precise about why: a guest heap's teardown cost on a conventional
host is O(live objects) because every object may have a finalizer and the host
must find them. Here it is O(blocks), because the guest's own objects are
*inside* the slabs and the slabs are freed by the block chain.

`vt->on_teardown` runs before `arena_destroy` and exists for exactly one purpose:
resources the guest owns **outside** the arena — file descriptors, `dlopen`
handles, sockets, threads, foreign `malloc`s in libraries the guest links. It
must not free arena memory and must not raise. Measured teardown for a heap that
had processed 600,000 objects: **8.8 microseconds** (Section 9, E1).

The hook is installed on the region and reached through
`eshkol_region_unwind_to`, described in-tree as the one teardown path
([`lib/core/runtime_regions.cpp:1863`](../../../lib/core/runtime_regions.cpp#L1863)),
which is what makes the guarantee hold for all four ways a region can close:
lexical `with-region` exit, explicit `region-close`, a `raise` crossing the
region ([`inc/eshkol/eshkol.h:1351`](../../../inc/eshkol/eshkol.h#L1351)), and a
`call/cc` escape crossing it
([`inc/eshkol/eshkol.h:1498`](../../../inc/eshkol/eshkol.h#L1498)). A guest
runtime that leaked its interpreter state on the exception path would be a
serious defect; the existing unwind unification means the adapter gets that case
right by construction rather than by remembering to.

### 6.2 Escape: promote the heap, not the object

The existing promotion path is the model, and the question is whether it extends
to a guest object escaping outward. It does not extend directly, and the reason
is instructive: **a guest object's transitive closure is, in general, the guest
heap.** Promoting one guest object outward would require the guest to perform a
partial trace and produce a self-contained sub-heap — which no real collector
offers and which would in any case be a different heap with different identity.

So the disposition is per-heap, declared at attach:

**`ESHKOL_GUEST_ANCHORED` (default).** A handle may be stored anywhere at or
inside the anchor region. Storing one into a destination that outlives the
anchor raises a clean, catchable error at the write barrier: *guest handle
escaped its heap's anchor region*. This is deliberately the same shape as an
affine-capability violation in ADR-0004's ownership HIR, and it is the shape
ADR-0001 invariant 3 already demands of every residence-crossing edge. No silent
dangle is possible, because the barrier is the only way such a store reaches
memory.

**`ESHKOL_GUEST_PROMOTABLE` (opt-in, `MOVING` guests only).** The barrier
re-anchors the **whole heap** to the escape target: the adapter allocates fresh
slabs from the target arena, calls `vt->relocate`, which performs a collection
whose to-space is the new slabs and which rewrites the handle table, and updates
the anchor. Cost is O(live), which is exactly the cost model
`region_evacuate_value` already imposes on an escaping Eshkol subgraph
([`lib/core/runtime_regions.cpp:1236`](../../../lib/core/runtime_regions.cpp#L1236)) —
so promotion is not a new cost class, it is the existing one applied to a bigger
unit.

The honest consequence, stated plainly because it decides Section 8: **a
non-moving guest can only be `ANCHORED`.** CPython cannot relocate objects
(interior pointers to `PyObject` fields are pervasive and `id()` is defined as
the address), and Boehm-collected runtimes cannot either. So both concrete
targets in Section 8 are anchored-only, and hosting them well means anchoring
them at a *long-lived* region — the process region, or a resident session
generation from ADR-0001 §6 — rather than at a short lexical scope. That is a
natural fit for an interpreter, which wants process lifetime anyway, and it is a
real constraint on the pattern `(with-region 'py (py-eval ...))` returning a
Python object: it returns a handle that is valid only inside, and must convert to
an Eshkol value to cross out. Conversion at the boundary is the right discipline
for a language boundary regardless.

---

## 7. Position in the trajectory

ADR-0000 sequences the program in fourteen stages. The adapter has two hard
dependencies and one soft one, and the dependencies are the schedule — no dates
are proposed here.

**Hard: Stage 4 (v1.4.1), "OALR ABI v2 and portable tail transfer."** Two of ABI
v2's deliverables are load-bearing for this ADR and nothing substitutes for them.
The layout-descriptor requirement — *"every pointer-bearing layout registers a
descriptor or startup fails"* — is what turns Section 4.1's new `evac_kind_for`
arm from a special case bolted onto a hardcoded `switch` into a registration.
Building the adapter first would mean adding a sixteenth hand-written case to a
classifier that ADR-0000 has already scheduled for replacement. Second, residence
becomes object metadata under ABI v2, which is what retires the
`region_index_owning` linear scan that Section 5.1(6) flags as the adapter's one
real cost.

**Hard: Stage 3 (v1.4.0), "Resource-sound systems profile."** The pin is an
affine capability and the anchor rule is an `outlives` obligation. Stage 3
delivers exactly those: *"generative `Region`/`Cap`/`Own`/`Borrow`/`Shared`/
`Weak`; outlives plus non-lexical loans; deterministic drop/close HIR; affine
typestate handles for files and sockets."* An affine typestate handle for a
socket and an affine pin handle for a guest object are the same construct; the
adapter should consume that machinery, not duplicate it. Without it, unbalanced
pins are a runtime leak report instead of a compile error.

**Soft: Stage 6 (v1.5.1), "resident sessions begin."** Not required for the
adapter, required for the *interesting* uses of it. Section 6.2 concludes that
both concrete targets are anchored-only and want a long-lived anchor. A resident
session generation is the principled long-lived anchor; without one, hosting a
CPython interpreter means anchoring at the process region, which works and is
merely inelegant.

**Not a dependency: Stages 2 and 8-14.** The identity substrate (Stage 2) is
needed by a *front end* that compiles CL or Python source, not by the adapter,
which is a runtime artifact. This distinction matters for Section 8: the adapter
and the front end are separable projects, and the adapter is much the smaller one.

**Proposed landing.** The ABI and the four evacuator/teardown changes of
Section 5.1 land as a slice **inside Stage 4**, because they are consumers of the
same ABI break and should fail with it rather than after it. The guest-hosting
profile — budget accounting, the residence audit in checked builds, the
`eshkol-run` surface for instantiating a guest — lands **after Stage 6**. Front
ends are downstream of both and are not scheduled by this ADR.

---

## 8. The two concrete targets

### 8.1 Common Lisp, and what happens to the Autolith NO-GO

`ESHKOL-CL-AUTOLITH-FEASIBILITY-2026-08-25` returned NO-GO on "run Autolith on
Eshkol" and CONDITIONAL GO on a scoped CL-subset front end, on three blockers.
That study was correct on its own terms and its measurements stand. What changes
is that its first blocker was scoped to one of two possible routes, and the
adapter both dissolves it on that route and opens a second route the study did
not price.

**Blocker 1 — no GC. DISSOLVED.** The study's reasoning was: *"Transpiled CL code
carries no region annotations, so all allocation lands in the global arena, which
only grows... The options are (a) give Eshkol a GC, which the memory model
explicitly and permanently rejects, or (b) hand-annotate region boundaries
throughout the CL source."* The disjunction is incomplete. There is a third
option, which is this ADR: the CL heap becomes a region with a CL collector
inside it, un-annotated CL code allocates into it, and cycles are reclaimed by
the CL collector without Eshkol tracing anything. Preconditions: Stage 4 and
Stage 3 per Section 7, and — because no CL collector relocates — the CL heap is
`ANCHORED` at a process-lifetime or resident-session region.

**Blocker 2 — the live self-modifying image. PARTIALLY dissolved, and the part
that dissolves is the surprising one.** The study is right that the image is the
thesis of Autolith, and right that `src/self` needs a reflectively-readable,
settable global environment with snapshot and rollback. Those requirements are
untouched by this ADR: they are environment-model and front-end problems.

But *image serialization* — `save-lisp-and-die`, which the study lists as
ABSENT-BLOCKED — becomes structurally **easier** under arena hosting than under a
general GC heap, not harder. A guest heap is a bounded, known set of slabs with a
known base address and a known handle table. Dumping it is "write the slabs and
the table, relocate on load," which is close to what a Cheney to-space already
is. Dumping a conventional GC heap requires walking an arbitrary object graph and
reconstructing the collector's internal state. The generation-checkpoint scheme
Autolith layers on top (`sbcl-generations`) maps onto a sequence of slab
snapshots straightforwardly. So Blocker 2 moves from *architecturally precluded
by the memory model* to *blocked on a reflective environment model*, which is a
different and cheaper kind of blocked — and its hardest-sounding component is the
one that improves.

The study's finding on ADR-0000 Stage 14a is unaffected and should be preserved
as written: capsules confine plasticity to declared interfaces and Autolith's
plasticity is undeclared by construction. Stage 14a does not deliver Phase 3, and
this ADR does not change that.

**Blocker 3 — scale. SURVIVES, entirely.** 141,729 LOC of Lisp in `src/` plus
pinned dependencies, plus the Quicklisp transitive closure. Nothing in a memory
model touches this. Any plan built on the brief's original "~10k LOC" figure
remains wrong by more than an order of magnitude.

**The surface blockers survive too.** CLOS, `loop`, CL-conformant `format`, the
`NIL`/`T` generalized-boolean representation, multiple-value arity semantics,
pathnames, `defcstruct`-shaped FFI, Gray streams — all of the study's §4
EMULATABLE and ABSENT rows are front-end work and are unchanged. The adapter
resolves one row of that table ("Garbage collection — ABSENT permanently —
BLOCKED") and no other.

**The second route, which the study did not price.** The study evaluated exactly
one strategy: transpile CL source to Eshkol (call it **R1**). Under R1 the
adapter removes Blocker 1 and leaves Blockers 2 and 3 and the whole §4 surface
list, so R1's 21-35 engineer-month estimate is essentially unchanged — it was
never dominated by memory.

The adapter makes a second strategy available. **R2: host an embeddable CL
implementation's runtime as a guest.** ECL and Clasp are designed for embedding,
expose a C API, and carry their own collector (Boehm, in ECL's case), whose
`GET_MEM` seam is precisely the `eshkol_guest_alloc` hook this ADR specifies.
Under R2, CLOS, `loop`, `format`, the condition system, `defmacro`, pathnames,
packages, and the reader are **not implemented — they are already there**. The
entire §4 table collapses to an FFI-surface question. R2's cost is dominated by
the embedding and the boundary, not by reimplementing Common Lisp.

R2 is not free and its risks are different in kind, so it should be stated
adversarially: Boehm is a *conservative* collector that scans for anything
resembling a pointer, and confining it to a region arena requires it to be built
without stack scanning and with an explicit root set, which is a supported but
non-default configuration; a conservatively-scanned guest arena could retain
garbage indefinitely, so invariant 7's byte ceiling is what keeps it honest; and
R2 delivers a CL that runs *beside* Eshkol rather than *as* Eshkol, so Eshkol's
AD, tensors, and gradients reach it only through the boundary, not natively.
That last point is decisive for which route serves which goal.

**Sequenced YES, with preconditions.**

| Phase | Deliverable | Preconditions |
|---|---|---|
| G0 | Falsifier F1 — a toy tracing collector reclaiming cycles inside a real Eshkol arena with a flat host resident set | none. **COMPLETE**, Section 9 |
| G1 | Adapter ABI, `EVAC_GUEST_HANDLE`, the teardown sweep, the barrier anchor case | Stage 4 (ABI v2 layout descriptors), Stage 3 (affine handles) |
| G2 | F3 — ECL or Clasp attached to a region, `GET_MEM` routed to `eshkol_guest_alloc`, a cyclic CLOS graph collected inside the region, RSS flat, teardown bulk | G1 |
| G3 | Boundary: CL values to and from Eshkol values; `eshkol_guest_publish` residence enforcement over the real CL heap; the FFI-surface shims the study's §4 enumerates | G2 |
| G4 | Autolith's agent core, headless, on R2 | G3, plus Autolith's Quicklisp closure loading under the embedded implementation — the real long pole, and Blocker 3 is still Blocker 3 |
| — | Autolith's live self-modifying image | reflective environment model + snapshot/rollback. Still blocked. Slab-dump is a route to `save-lisp-and-die` specifically |

The verdict this supersedes with: **NO-GO on R1-for-Autolith stands, for
Blocker 3 rather than Blocker 1. R2 is newly available, is not blocked by the
memory model, and is the route to evaluate if running Autolith is the goal. A
CL-subset front end (R1) remains CONDITIONAL GO justified on Eshkol's own merits,
exactly as the study concluded.**

### 8.2 CPython

CPython is the easier target and the better first real guest, for a reason that
generalizes: **refcounting maps onto arena lifetime far more naturally than
tracing does.**

A refcount reaching zero is a *local, immediate, deterministic* reclamation
decision. It needs no global root set, no stop-the-world pause, no reachability
walk, and no cooperation from the host. It reclaims into CPython's own free
lists, which live inside the guest slab. Every property Eshkol's memory model
advertises — determinism, no pauses, bounded footprint, no host tracing —
survives refcounting untouched. The cycle collector then handles what refcounting
structurally cannot, on CPython's own schedule, inside CPython's own arena, and
it is precisely the component this ADR exists to make possible.

The shape of the adapter for CPython:

- **Allocation seam.** CPython already has the exact abstraction this ADR needs:
  a pluggable arena allocator underneath `obmalloc`, plus the `PyMem_*`
  allocator hooks. The adapter installs an arena allocator whose alloc calls
  `eshkol_guest_alloc` and whose free is a no-op that lets `obmalloc` recycle
  within the slab. CPython's own pools give it intra-slab reuse that a bump arena
  cannot provide, which is exactly the "guest pays for its own reclamation inside
  its own region" thesis.
- **Flags.** `ESHKOL_GUEST_ANCHORED | 0` — not `MOVING`, because `id()` is
  defined as the object address and interior `PyObject` pointers are pervasive.
  `relocate` is NULL. Per Section 6.2 the interpreter anchors at a
  process-lifetime or resident-session region.
- **Roots.** The handle table is the root set, and the pin protocol *is*
  `Py_INCREF`/`Py_DECREF`: `eshkol_guest_pin` increfs, `eshkol_guest_unpin`
  decrefs. A pinned handle is a strong reference by exactly the mechanism CPython
  already uses. `vt->collect` calls `PyGC_Collect`.
- **Boundary, Eshkol to Python.** A `HEAP_SUBTYPE_GUEST_HANDLE` whose handle
  slot holds a pinned `PyObject *`. Because CPython does not move, the handle
  costs one indirection and buys the generation check and the root registration —
  both of which are still required, the second one especially.
- **Boundary, Python to Eshkol.** An `eshkol_tagged_value_t` inside a small
  extension type (a capsule-shaped wrapper), opaque to CPython's tracer,
  published through `eshkol_guest_publish` so the residence obligation is
  checked. Its `tp_traverse` reports no references, which is correct: it holds
  no Python references.
- **Threads and subinterpreters.** A guest that creates threads declares
  `ESHKOL_GUEST_THREADED` and each guest thread entering Eshkol must call
  `eshkol_thread_init_worker`
  ([`lib/core/arena_memory.h:153`](../../../lib/core/arena_memory.h#L153)) or its
  region stack is uninitialized thread-local storage. Per-interpreter isolation
  maps onto the model unusually cleanly — one region, one guest heap, one
  interpreter — and is the natural unit of both budget and teardown.
- **Budget.** `arena_create_bounded` gives the interpreter a hard ceiling. A
  Python heap that exceeds it raises `MemoryError` inside Python, which is a
  defined Python outcome, rather than growing an Eshkol process without bound.
  For an embedded or real-time deployment this is a significantly stronger
  guarantee than CPython offers standalone.

The single genuinely hard part is not memory at all: it is that CPython holds
global interpreter state in process globals, so multiple guest heaps in sibling
regions are only as isolated as the interpreter's own isolation support. That is
a CPython property, not an Eshkol one, and it bounds how many independent Python
guests a process can host — not whether one can be hosted.

---

## 9. Falsifier F1 — built, run, measured

An ADR with a working falsifier is worth ten without one. F1 is built and its
numbers are below.

**Source:** [`0011-gc-adapter-falsifier/`](0011-gc-adapter-falsifier/) — self
contained, no CMake, no LLVM, no configure step.

```
docs/design/adr/0011-gc-adapter-falsifier/build-and-run.sh
```

**What it is.** A toy mini-Lisp guest with a Cheney semispace copying collector
whose two semispaces are one `arena_allocate` each from a real Eshkol `arena_t`.
The arena implementation is `lib/core/runtime_arena_core.cpp`, **unmodified**,
compiled and linked directly along with its mutex and diagnostics shims, the
tagged-cons helpers, and the logger. The only substitution is
`harness_stubs.cpp`, which supplies the process-wide byte-accounting and
interrupt symbols the arena calls but this experiment does not exercise; they
affect no bump placement, block sizing, scope rewind, or teardown, and every
number below is read from `arena_get_total_memory()` and the OS resident-size
counter. Every guest object lives inside the two slabs; Eshkol's allocator sees
exactly two allocations for the guest's entire life regardless of how many
collections run. The prototype implements the Section 3 ABI —
`eshkol_guest_vtable_t`, the generation-counter handle table, `pin`/`unpin`/
`deref`, `collect`, `on_teardown`, `enumerate_outbound` — as written.

Output below is from a run at `origin/master` `e65cb5b5`, arm64 macOS.

**What it proves and what it does not.** It establishes the memory-model claim:
that a guest collector hosted in a region reclaims cycles while the host arena
stays flat, that the handle protocol survives a moving collector, and that
teardown is bulk. It does **not** establish that a *real* language runtime can be
made to allocate this way — that is falsifier F3 (ECL) and F4 (CPython), both
unbuilt and both gated on Section 7's Stage 4. It also does not exercise the
`with-region` integration, because the required `evac_kind_for` and teardown
changes of Section 5.1 do not exist yet; that is F2.

### E1 — cycles reclaimed, host arena flat

200,000 rounds, each allocating a 3-cycle with extra sharing and dropping every
reference, through a 256 KiB (6,553-slot) semispace.

```
rounds                 200000 (600000 guest objects allocated)
guest collections      91
guest bytes reclaimed  23852920
live objects after GC  0 of 6553 slots
host arena  start      65536 B
host arena  after attach 655374 B
host arena  peak       655374 B
host arena  end        655374 B
wall                   2.2 ms
```

- 600,000 objects passed through a 6,553-slot heap with zero allocation
  failures. A collector that could not reclaim cycles would have exhausted
  from-space within three rounds — every object allocated is part of a cycle and
  nothing is pinned, so cycle reclamation is the *only* mechanism that can free
  anything here.
- Host arena peak equals host arena end equals the post-attach figure. **The
  delta across 91 guest collections is exactly zero bytes.**
- Eshkol performed zero traces and zero per-object frees.
- Teardown, for a heap that had processed 600,000 objects: **8.8 microseconds**,
  one `arena_destroy`.

### E2 — the handle protocol under a moving collector

The experiment that justifies mandating handles in the ABI.

```
objects before GC      130
objects after  GC      2
pinned object id       0xBEEF (expected 0xBEEF)
raw ptr before/after   0x138018000 / 0x138030000
```

- The pinned object survived, resolved through its handle to the **moved**
  address, with identity preserved and its 2-cycle partner intact and
  self-consistent.
- 128 objects of dead cycles reclaimed in one pass; the surviving 2 are the
  pinned cycle.
- The raw pointer captured before the collection is **stale** — a
  use-after-move. This is the failure a raw Eshkol-to-guest edge produces, shown
  rather than asserted.
- Unpinning and collecting again dropped the pinned cycle to 0 objects.
- Dereferencing the now-stale handle returned NULL with the violation counter at
  1: the generation check caught it. A detected fault, not a wild pointer —
  matching the contract `region-close` already provides for stale region tokens.

### E3 — guest-to-Eshkol edges

A guest object in one region holds an `eshkol_tagged_value_t` naming a real
`arena_tagged_cons_cell_t` in an outer arena, through a guest collection and
region teardown.

```
outbound edges found   1 (foreign-resident: 1)
Eshkol value after GC  car = 42
```

- The guest object survived the move carrying its Eshkol value copied verbatim;
  the tracer stepped over it and never followed it.
- The Eshkol object was neither moved nor freed by the guest.
- `enumerate_outbound` located the edge for a residence audit.
- After the guest's region was torn down, the outer-region Eshkol object was
  still live and still read 42.

### E4 — resident set under sustained allocation

8.4 million guest objects through a declared 2 MiB heap (2 × 1 MiB semispace),
against a permanent live set of 512 pinned cycles.

```
declared guest heap     2097152 B (2 x 1048576 semispace)
guest objects allocated 8401024
guest collections       333
host arena  2228238 -> 2228238 B (delta 0)
RSS pre-warmup          2097152 B
RSS after warmup (rss0) 4145152 B (+2048000 = heap first-touch)
RSS after 8M allocs     4145152 B (delta 0)
pinned live cycles intact 512 of 512
```

- Host arena delta across 8 million allocations and 333 collections: **zero
  bytes**.
- Process RSS rises by 2,048,000 bytes — the first-touch cost of the declared
  2,097,152-byte heap — and then **stops**. The subsequent 8 million allocations
  move it by zero. Resident set tracks the *declared budget*, not the allocation
  volume. This is invariant 7 measured.
- All 512 pinned live cycles intact and self-consistent after 333 relocations.

### F2, F3, F4 — the falsifiers not yet built

- **F2 (gated on G1).** A `with-region` fixture: a guest heap attached inside a
  region, a handle stored, the region exited by each of the four paths — normal
  exit, `region-close`, `raise`, `call/cc` escape — with the teardown hook
  observed to run exactly once on each. Kills the teardown design if
  `eshkol_region_unwind_to` does not in fact unify all four.
- **F3 (gated on G2).** ECL or Clasp attached to a region with Boehm's `GET_MEM`
  routed to `eshkol_guest_alloc`; a cyclic CLOS instance graph reclaimed inside
  the region; RSS flat over a long session; bulk teardown. This is the experiment
  that decides Section 8.1's R2, and it is the highest-value unbuilt one.
- **F4.** CPython under a bounded arena allocator, `PyGC_Collect` reclaiming a
  cyclic graph inside the region, `MemoryError` at the declared ceiling rather
  than host growth.

---

## 10. Non-goals

- **A garbage collector for Eshkol values.** Not now, not later. ADR-0001's
  theorem is unchallenged and this ADR does not weaken it by one term.
- **Automatic region inference for guest code.** The guest heap is declared and
  anchored explicitly. Nothing infers it.
- **Cross-guest references.** Two guest heaps do not reference each other
  directly; they meet through Eshkol values, subject to Section 4.2.
- **A CL or Python front end.** Separate work, downstream, not scheduled here.
- **Making a guest's pauses disappear.** A guest's collector pauses the guest.
  Eshkol code outside the guest's region is unaffected, and that is the whole of
  the claim. Anyone who instantiates a Python heap in a hard-real-time loop gets
  Python's latency inside that loop, and should not.

---

## 11. Consequences

**Positive.**

- The "no-GC closes doors permanently" risk is retired as stated. GC-hosted guest
  languages become a planned capability with a specified mechanism.
- The no-GC commitment becomes *more* precise, not less: it is a statement about
  Eshkol's own semantics, with a named boundary, rather than a blanket that
  happened to also exclude hosting.
- Two reserved object-model bits and a shipped generation-handle protocol become
  load-bearing rather than aspirational.
- Guest memory becomes *budgeted and inspectable* in a way it is not on a
  conventional host: a hosted CPython has a hard ceiling and a bulk teardown,
  which standalone CPython does not.

**Negative, stated plainly.**

- The evacuator, the write barrier, and `region_destroy` all grow a case. Three
  of the most correctness-critical functions in the runtime become slightly
  larger. Section 5.1 enumerates them so the review surface is bounded and
  known.
- `region_index_owning` becomes hot on the publish path until ABI v2's residence
  metadata retires it.
- Both concrete targets are anchored-only, which constrains the region patterns
  available to a user hosting them (Section 6.2).
- An unbalanced pin is a leak until Stage 3's affine handles make it a compile
  error.
- A conservatively-collected guest can retain garbage indefinitely inside its
  budget. The budget is what keeps that from becoming an Eshkol problem, which
  is why invariant 7 is an invariant and not a convenience.

---

## References

- [ADR-0000, unified architectural trajectory](0000-unified-trajectory.md) — Stage 3 (v1.4.0), Stage 4 (v1.4.1), Stage 6 (v1.5.1)
- [ADR-0001, concurrent resident-grade OALR](0001-oalr-concurrent-resident.md) — the theorem, residence, ABI v2, layout descriptors, invariants 3/6/9
- [ADR-0004, type system trajectory](0004-type-system-trajectory.md) — affine capabilities, `outlives`, ownership HIR
- [Memory model reference](../../reference/runtime/memory-model.md) — arenas, regions, escape promotion, user-reachable region handles
- `ESHKOL-CL-AUTOLITH-FEASIBILITY-2026-08-25` (private study) — the measured Autolith profile, the SBCL macroexpansion experiment, and the NO-GO this ADR supersedes on Blocker 1
- Falsifier F1: [`0011-gc-adapter-falsifier/`](0011-gc-adapter-falsifier/)
