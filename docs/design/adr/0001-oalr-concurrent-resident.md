# ADR-0001: Concurrent, resident-grade OALR

- **Status:** Proposed
- **Date:** 2026-07-09
- **Decision owners:** Eshkol memory/runtime architecture
- **Cluster:** OALR / concurrency / residence
- **Applies to:** LLVM AOT/JIT runtime first; the VM must implement the same semantic contract
- **Memory premise:** no autonomous garbage collector

## Decision

Eshkol will replace ambient, process-shared arena selection with an explicit **thread memory context**. Every OS thread gets its own lexical-region stack and private allocation domains. A lexical region is never concurrently bumped, rewound, or destroyed by another thread. Shared destinations are append-only/sharded or explicitly synchronized; they never carry a shared LIFO scope stack.

Escape will mean **lifetime extension with preserved Scheme identity and aliasing**, not “copy the value as it looked at the first store.” Same-thread outward stores are recorded in a typed escape ledger and evacuated as one graph when the source region closes. Cross-thread escape is an explicit freeze/move into a sealed transfer capsule, published with release/acquire ordering. Deep evacuation uses one complete layout table and one forwarding relation per transaction; it is atomic with respect to region destruction and never returns a pointer into a region that will be freed.

Resident programs will use an explicit **resident session** with two bounded state spaces, per-thread scratch regions, registered roots, a single-writer copy-on-write transaction, and epoch/RCU reader leases. Commit evacuates the final reachable state into the inactive space, atomically swaps the root table, waits before reusing the old space, and discards overwritten or unreachable history wholesale. This gives a hard steady-state bound proportional to the live state and peak scratch, not to daemon uptime.

This is not a background or heuristic GC. Reclamation occurs only at lexical exit, ownership transfer, or an explicit resident commit, over compiler/program-declared roots, with a configured byte budget. Algorithmically, resident commit is a Cheney-style copying pass. That distinction is intentional and unavoidable: **arbitrary R7RS aliasing + mutation + automatic reclamation of unreachable cycles cannot simultaneously be provided with no tracing, no reference accounting, and no ownership restriction.** If “no GC” is interpreted as “no reachability walk at all,” resident state must instead be affine/unique (with explicit reference counting for shared subgraphs); general Scheme mutation cannot be promised forever-flat.

## Source baseline and diagnosis

The prompt names `lib/core/arena_memory.cpp`. In this checkout that file has been split: raw blocks/scopes live in [`lib/core/runtime_arena_core.cpp:1`](../../../lib/core/runtime_arena_core.cpp#L1), and the C++ wrapper lives in [`lib/core/runtime_arena_cpp.cpp:1`](../../../lib/core/runtime_arena_cpp.cpp#L1). The stale source comment still names the former file at [`lib/core/runtime_regions.cpp:10`](../../../lib/core/runtime_regions.cpp#L10). This ADR describes the current split implementation.

The present system is partway between two models:

| Current fact | Evidence | Consequence |
|---|---|---|
| The lexical region stack is already TLS. | [`lib/core/runtime_regions.cpp:24`](../../../lib/core/runtime_regions.cpp#L24), [`lib/core/arena_memory.h:339`](../../../lib/core/arena_memory.h#L339) | This part should be retained, but hidden behind a memory-context ABI. It does **not** make allocation selection thread-safe. |
| Runtime `get_global_arena()` prefers a worker TLS arena. | [`lib/core/runtime_regions.cpp:61`](../../../lib/core/runtime_regions.cpp#L61) | Runtime callers can be thread-private. |
| Generated code loads `__global_arena` directly. | [`lib/backend/llvm_codegen.cpp:5519`](../../../lib/backend/llvm_codegen.cpp#L5519) | It bypasses the TLS selection above. |
| `with-region` saves and overwrites that global slot for its whole body. | [`lib/backend/llvm_codegen.cpp:29946`](../../../lib/backend/llvm_codegen.cpp#L29946), [`lib/backend/llvm_codegen.cpp:29960`](../../../lib/backend/llvm_codegen.cpp#L29960) | Two threads entering `with-region` race, allocate into one another's regions, restore stale values, and can destroy an arena still in use. |
| An `arena_t` owns one `current_scope` linked stack. | [`lib/core/arena_memory.h:46`](../../../lib/core/arena_memory.h#L46), [`lib/core/arena_memory.h:53`](../../../lib/core/arena_memory.h#L53) | Independent dynamic call stacks cannot safely share it. |
| Bump allocation locks a thread-safe arena. | [`lib/core/runtime_arena_core.cpp:190`](../../../lib/core/runtime_arena_core.cpp#L190) | The lock protects the bump pointer only. |
| Scope push/pop/commit do not lock and destructively rewind the arena. | [`lib/core/runtime_arena_core.cpp:273`](../../../lib/core/runtime_arena_core.cpp#L273), [`lib/core/runtime_arena_core.cpp:291`](../../../lib/core/runtime_arena_core.cpp#L291), [`lib/core/runtime_arena_core.cpp:358`](../../../lib/core/runtime_arena_core.cpp#L358) | Concurrent scope operations corrupt the linked stack; even per-operation locking would not make interleaved allocations rewindable. |
| Codegen works around the scope race by rejecting loops reachable from parallel callbacks. | [`lib/backend/llvm_codegen.cpp:23768`](../../../lib/backend/llvm_codegen.cpp#L23768), [`lib/backend/llvm_codegen.cpp:24301`](../../../lib/backend/llvm_codegen.cpp#L24301) | Safety depends on a conservative whole-compilation-unit analysis with an acknowledged REPL horizon, rather than on the runtime model. |
| The current deep escape is a graph copy with a forwarding map. | [`lib/core/runtime_regions.cpp:622`](../../../lib/core/runtime_regions.cpp#L622), [`lib/core/runtime_regions.cpp:685`](../../../lib/core/runtime_regions.cpp#L685) | Sharing and cycles can be retained within the copied snapshot. |
| Its owner lookup scans only the calling thread's active region stack. | [`lib/core/runtime_regions.cpp:646`](../../../lib/core/runtime_regions.cpp#L646), [`lib/core/runtime_regions.cpp:658`](../../../lib/core/runtime_regions.cpp#L658) | It cannot classify a pointer owned by a foreign worker or an inactive resident generation. |
| One region-level forwarding map is cleared when the promotion target changes. | [`lib/core/runtime_regions.cpp:831`](../../../lib/core/runtime_regions.cpp#L831) | It is not a general identity relation for several incomparable or cross-thread destinations. |
| Unsupported layouts silently become shallow leaves. | [`lib/core/runtime_regions.cpp:721`](../../../lib/core/runtime_regions.cpp#L721), [`lib/core/runtime_regions.cpp:736`](../../../lib/core/runtime_regions.cpp#L736) | A resident-grade claim is impossible while graph-bearing subtypes can retain dying interior pointers. |
| Mutable closure cells are guessed from an integer that resembles an arena address. | [`lib/core/runtime_regions.cpp:941`](../../../lib/core/runtime_regions.cpp#L941) | Pointer provenance and exact tracing are not represented in the ABI. |
| Evacuation allocation failure can return the old pointer. | [`lib/core/runtime_regions.cpp:752`](../../../lib/core/runtime_regions.cpp#L752), [`lib/core/runtime_regions.cpp:787`](../../../lib/core/runtime_regions.cpp#L787) | The caller can free the source region and publish a dangling pointer after an out-of-memory condition. |

The current per-iteration mechanism also exposes the residence problem. It performs a shallow check of loop out-values and relies on static exclusion of mutation ([`lib/core/runtime_arena_core.cpp:386`](../../../lib/core/runtime_arena_core.cpp#L386)); if one value escapes, it commits **all** allocation since the mark ([`lib/core/runtime_arena_core.cpp:427`](../../../lib/core/runtime_arena_core.cpp#L427)). That is a safe short-term fallback, not resident reclamation.

The new mutating-loop fixture proves that transient region garbage is reclaimed and that 64 final vector entries remain readable ([`tests/memory/region_mutating_loop_flat_rss_test.esk:21`](../../../tests/memory/region_mutating_loop_flat_rss_test.esk#L21), [`tests/memory/region_mutating_loop_flat_rss_test.esk:30`](../../../tests/memory/region_mutating_loop_flat_rss_test.esk#L30)). It does not prove forever-flat persistent mutation: every replacement list is promoted to the immortal arena, and the 300 MB gate merely separates that finite run from the former multi-gigabyte leak ([`tests/memory/region_mutating_loop_flat_rss_test.sh:15`](../../../tests/memory/region_mutating_loop_flat_rss_test.sh#L15)). A daemon that overwrites the same 64 slots forever still accumulates all retired promoted lists.

There is a second semantic bug hidden by the word “promotion.” Suppose `x` is region-local, `x` is stored in an outer vector, and then `(set-car! x ...)` executes before region exit. The current first store copies `x`; the persistent forwarding entry makes later stores reuse that copy ([`lib/core/runtime_regions.cpp:831`](../../../lib/core/runtime_regions.cpp#L831)), but mutation through the original pointer still changes the original. The two Scheme aliases have diverged. The implementation preserves graph topology **at copy time**, not alias semantics through the rest of the region activation.

## Required semantics

The design is governed by the following invariants.

1. **Private-domain ownership.** At any instant exactly one thread may allocate, rewind, mutate allocator metadata, or perform evacuation reads over an unsealed private domain.
2. **No ambient cross-thread allocation selector.** Generated allocation obtains the current domain from the calling thread's memory context. `__global_arena` is never temporarily overwritten.
3. **Residence-aware edges.** A pointer stored in destination residence `D` must refer to an object whose residence outlives `D`, or the store must be remembered and repaired before the source can die.
4. **No foreign rewind.** An arena with a foreign borrow, published root, or unjoined task cannot be reset or destroyed.
5. **Publication closes mutation rights.** A private mutable graph crosses a thread boundary only by consuming/moving it, or by freezing it. Shared mutation uses an explicitly shared/resident abstraction and synchronization.
6. **Complete tracing.** Every pointer-carrying allocation has a header and an exact layout descriptor. “Unknown means leaf” is forbidden.
7. **One identity, one forwarding result per transaction.** All roots evacuated in one exit/export/commit share a forwarding relation. Cycles terminate and aliases converge on one target object.
8. **All-or-nothing promotion.** Failure leaves the source live and the destination unpublished. It never degrades to an old pointer followed by source destruction.
9. **No hidden resident roots.** A raw Scheme pointer cannot live across a resident commit unless it is in the registered root table or covered by a reader lease. FFI pointers must be pinned/copied under an explicit lease.
10. **Bounded history.** Resident mode may block on a stale reader or fail a budgeted commit; it may not allocate an unbounded sequence of retired generations to preserve progress.

OALR guarantees memory safety, ownership transfer, and lifetime discipline. It does not make unsynchronized Scheme mutation a race-free operation. Concurrent writes to one shared logical object still require a mutex, channel ownership, atomics, or the single-writer resident transaction.

## 1. Thread memory contexts, not a locked shared stack

Each attached thread has one runtime-owned TLS pointer. LLVM calls an accessor instead of declaring or loading allocator TLS symbols itself; this avoids the existing C/LLVM TLS ABI ambiguity and gives JIT, AOT, callbacks, and foreign-created threads one contract.

```c
#define ESHKOL_MEMORY_ABI_V2 2u

typedef struct eshkol_memctx eshkol_memctx_t;
typedef struct eshkol_residence eshkol_residence_t;
typedef struct eshkol_region_frame eshkol_region_frame_t;
typedef struct eshkol_resident_txn eshkol_resident_txn_t;

struct eshkol_memctx {
    uint32_t abi_version;
    uint32_t flags;
    uint64_t thread_id;
    eshkol_region_frame_t *region_top; /* this thread only */
    eshkol_residence_t *allocation_domain;
    eshkol_resident_txn_t *resident_txn; /* nullable */
    void *runtime_private;
};

struct eshkol_region_frame {
    eshkol_region_frame_t *previous;
    eshkol_residence_t *residence;
    eshkol_residence_t *previous_allocation_domain;
    void *escape_ledger;
    uint64_t cookie;
};

typedef struct eshkol_region_token {
    eshkol_region_frame_t *frame;
    uint64_t cookie;
} eshkol_region_token_t;

int eshkol_mem_thread_attach(uint32_t flags, eshkol_memctx_t **out);
int eshkol_mem_thread_detach(void);
eshkol_memctx_t *eshkol_memctx_current(void);
arena_t *eshkol_current_arena(void);
int eshkol_region_enter(const char *name, size_t size_hint, uint32_t flags,
                        eshkol_region_token_t *out);
int eshkol_region_leave(eshkol_region_token_t token,
                        eshkol_tagged_value_t **root_slots,
                        size_t root_count);
```

`eshkol_thread_init_worker()` already eagerly initializes the TLS arena and region stack ([`lib/core/runtime_regions.cpp:112`](../../../lib/core/runtime_regions.cpp#L112)); it becomes a compatibility wrapper around `eshkol_mem_thread_attach`. Foreign threads entering Eshkol must attach or receive a deterministic error. Detach asserts an empty region stack, no resident lease, and no unsealed transfer.

`__global_arena` remains, during migration, as an immutable pointer to the process immortal domain for old runtime objects. It is not “current.” All generated allocation sites, including the helpers that currently use a direct global load, call or inline `eshkol_current_arena()`. Entering `with-region` changes only `memctx->allocation_domain` and restores it from the TLS frame.

### Why a mutex is not an alternative

Locking each push/pop cannot turn one stack into two dynamic call stacks:

```text
T1: push A       allocate a1       ...             pop A
T2:        push B       allocate b1       pop B
```

The operations may be individually serialized while their lifetimes remain interleaved. If T1 pops while B is logically active, LIFO is wrong; if B has popped first, rewinding A may still reclaim interleaved `b1`. Holding the lock from push through pop would serialize arbitrary Scheme execution, deadlock re-entrant callbacks, and defeat `parallel-map`. Therefore:

- lexical and iteration stacks are per-thread;
- private arenas use no allocator lock and assert their owner;
- a shared immortal domain may use locked allocation or, preferably, per-thread append-only shards;
- **scope/rewind is never legal on a shared arena**.

## 2. Separate region, arena, and residence

The current structures conflate three ideas. V2 gives them distinct meanings:

- **Arena:** a block allocator owned by one thread, with marks and bulk rewind.
- **Region frame:** a dynamic lexical capability on one thread; it selects an arena and records outward edges.
- **Residence:** a stable lifetime/ownership domain visible in object metadata. It can be lexical, iteration scratch, transfer, resident generation, shared-pinned, or immortal.

```c
typedef enum eshkol_residence_kind {
    ESH_RES_LEXICAL = 1,
    ESH_RES_ITERATION,
    ESH_RES_TRANSFER,
    ESH_RES_RESIDENT_GEN,
    ESH_RES_SHARED_PINNED,
    ESH_RES_IMMORTAL
} eshkol_residence_kind_t;

typedef enum eshkol_residence_state {
    ESH_RES_OPEN = 1,      /* owner may allocate/mutate */
    ESH_RES_SEALING,
    ESH_RES_SEALED,        /* immutable or exclusively transferable */
    ESH_RES_RETIRED,
    ESH_RES_DEAD
} eshkol_residence_state_t;

struct eshkol_residence {
    uint64_t id;
    uint64_t owner_thread;             /* 0 for shared */
    eshkol_residence_kind_t kind;
    _Atomic uint32_t state;
    eshkol_residence_t *parent;        /* lexical outlives edge, if any */
    arena_t *arena;                     /* private while OPEN */
    void *policy;                       /* capsule/session/shared policy */
};
```

The `parent` relation is a lifetime relation, not a license for another thread to allocate in the parent arena. Cross-thread residences may be incomparable. APIs therefore pass an explicit target residence; they do not infer it by scanning the caller's region stack or by asking which active arena contains `dst`.

### Tokenized arena marks

`arena_t::current_scope` is removed. A mark is a value owned by the region frame that created it:

```c
typedef struct arena_mark {
    arena_block_t *block;
    size_t used;
    size_t total_allocated;
    uint64_t arena_cookie;
    uint64_t owner_thread;
} arena_mark_t;

int arena_mark(arena_t *private_arena, arena_mark_t *out);
int arena_rewind(arena_t *private_arena, const arena_mark_t *mark);
```

Rewind verifies arena identity, owner, and that no seal/borrow was issued since the mark. Nesting belongs to the TLS frame stack, not to allocator-global mutable state. Dedicated lexical/iteration arenas can usually be reset as a whole; marks remain useful for compiler temporaries and nested nonescaping work.

The existing `arena_merge_to_parent` moves a source block chain while locking only the destination ([`lib/core/runtime_regions.cpp:194`](../../../lib/core/runtime_regions.cpp#L194)). V2 names the missing precondition: block adoption is valid only when the source residence is **sealed and exclusively owned**. The operation consumes the source token, so no concurrent allocator can still hold it.

## 3. Object and layout ABI v2

Cross-thread ownership cannot be discovered by scanning one thread's active arenas. Complete evacuation also cannot depend on an integer-looking-like-a-pointer heuristic. V2 makes residence, identity, and layout explicit while retaining the 16-byte `eshkol_tagged_value_t` ABI described at [`inc/eshkol/eshkol.h:177`](../../../inc/eshkol/eshkol.h#L177).

```c
typedef struct alignas(16) eshkol_object_header_v2 {
    uint32_t payload_size;
    uint16_t layout_id;          /* exact tracer/finalizer descriptor */
    uint8_t subtype;
    uint8_t flags;
    uint64_t object_id;          /* stable across evacuation; never reused live */
    eshkol_residence_t *home;
    uint64_t aux;                /* reserved; zero in ABI v2 */
} eshkol_object_header_v2_t;     /* 32 bytes */
```

The current header is 8 bytes and carries size/subtype/flags/refcount but no home or stable identity ([`inc/eshkol/eshkol.h:460`](../../../inc/eshkol/eshkol.h#L460)). Expanding it is an internal memory-ABI break, so AOT objects, the runtime, stdlib bitcode, and JIT modules must agree on `ESHKOL_MEMORY_ABI_V2`. The payload stays 16-byte aligned. `object_id` makes `eq?` stable across resident copies; pointer address is only a fast path when homes/epochs match.

Every traceable allocation, including closure capture cells and auxiliary arrays, gets a header. Raw numeric/byte buffers may use a nonpointer layout. No pointer-bearing buffer is headerless.

```c
typedef void (*eshkol_visit_slot_fn)(eshkol_tagged_value_t *slot, void *ctx);
typedef int  (*eshkol_trace_fn)(void *payload, eshkol_visit_slot_fn, void *ctx);
typedef void (*eshkol_finalize_fn)(void *payload);

typedef struct eshkol_layout_desc {
    uint16_t layout_id;
    uint16_t flags;       /* LEAF, MUTABLE, EXTERNAL, PINNED, ... */
    uint32_t min_size;
    eshkol_trace_fn trace;
    eshkol_finalize_fn finalize;
} eshkol_layout_desc_t;
```

Conses, vectors/records, multiple values, closures/environments, hash backing arrays, exceptions, tensors, promises, substitutions, knowledge bases, factor graphs, workspaces, DNC/SDNC, Taylor values, parameters, continuations, and future heap types must register a descriptor before they can be allocated in a reclaimable residence. Registration failure is a link/startup error. External resources are represented by stable linear/pinned handles; evacuation copies a reference to the handle, not the OS resource, and cannot double-finalize it.

A global range/page index may later replace the `home` pointer for compact objects, but it must provide the same O(1), cross-thread-safe residence query. Scanning `arena_block_t` lists is not the contract.

## 4. Alias-preserving lexical escape

### Deferred same-thread stores

For a private lexical region and a **thread-confined** outer destination, the write barrier does **not** immediately clone a mutable object. It performs the store and records the actual logical destination in the source region's escape ledger. Until region exit, the destination may point inward because the source remains live and only its owner thread may observe/mutate that private graph. All aliases therefore still refer to the same object.

```c
typedef struct eshkol_place {
    void *container;               /* NULL for a global/root cell */
    eshkol_residence_t *home;
    uint16_t layout_id;
    uint16_t field_or_card;
    uint32_t version;              /* detects resize/replacement */
    eshkol_tagged_value_t *stable_slot; /* only when descriptor says stable */
} eshkol_place_t;

int eshkol_store_tagged(eshkol_place_t dst,
                        const eshkol_tagged_value_t *value);
int eshkol_store_tagged_range(eshkol_place_t dst,
                              const eshkol_tagged_value_t *values,
                              size_t count);
```

Recording a typed place rather than only a raw slot matters for hash-table resize and other movable backing stores. At exit the layout descriptor re-resolves the place; an overwritten slot is skipped. Fixed cons/vector slots can use the stable fast path. This ABI replaces the current “return a possibly promoted temporary, then store it” barrier at [`lib/core/runtime_regions.cpp:1087`](../../../lib/core/runtime_regions.cpp#L1087), because that form cannot later repair the definitive slot.

A process global, channel, future, atomic cell, foreign buffer, or container with a foreign reader is **not** thread-confined. Such a destination never temporarily receives an inward raw pointer. The store must use one of three paths:

1. consume/freeze and eagerly export the value, ending source mutation rights;
2. stage the store in the region ledger and publish it only after close (owner-thread reads consult the pending-store overlay); or
3. update a resident/shared handle under its synchronization policy.

Thus a worker cannot make its open region graph visible merely by assigning a Scheme global. In legacy single-thread execution a global may use the staged path with no observable delay outside the owner; once a second thread is attached, globals are classified as shared publication points.

On every control-flow exit from `with-region`—normal result, exception cleanup, or supported continuation unwind—the runtime:

1. stops further source-region mutation;
2. gathers the body result and all still-live remembered places;
3. computes the least residence that outlives all destinations (normally parent or immortal);
4. evacuates all gathered roots with one forwarding table;
5. rewrites every destination and result slot;
6. validates that no repaired graph points into the source;
7. only then destroys/resets the source region.

The source is still live during steps 1–6, so an allocation or layout failure aborts the exit cleanly. The runtime may raise an out-of-memory condition while retaining/unwinding the source; it may not log and leave the old pointer in place.

This batching fixes snapshot-not-alias semantics. A store followed by mutation sees one source object; evacuation happens after the last mutation. Multiple outward aliases are rewritten to the same `object_id`/copy. A cycle is copied once. If destinations have incomparable residences, ordinary `with-region` cannot invent a common lifetime: the programmer/compiler must select an explicit shared capsule or immortal target.

### Barrier completeness

All pointer stores must use a typed barrier: `set!` into globals/captured cells, `set-car!`, `set-cdr!`, vector/record writes and fills/copies, hash keys/values and resize, promise memoization, exception irritants, knowledge/workspace mutation, continuation capture, runtime helper writes, and FFI callbacks. Static lifetime analysis may prove a barrier unnecessary, but it is an optimization. Correctness no longer depends on the present mutation blacklist.

Private region pointers may not be placed in atomics, channels, futures, or foreign memory through the deferred path. Those operations are publication boundaries and use transfer below.

## 5. Cross-thread borrowing, escape, and promotion

Thread stacks are not merged or inherited. A structured task receives:

- immutable/borrowed inputs whose owner is guaranteed live until join;
- its own thread memory context and task/iteration regions;
- an output transfer builder owned by that worker.

The producer, never the receiver, closes over and exports a private result. It is the only thread allowed to traverse an open source graph.

```c
typedef struct eshkol_transfer eshkol_transfer_t;

int eshkol_transfer_begin(size_t byte_budget, eshkol_transfer_t **out);
int eshkol_transfer_export_move(
    eshkol_transfer_t *transfer,
    eshkol_tagged_value_t *roots,
    size_t root_count);              /* consumes mutable source rights */
int eshkol_transfer_seal(eshkol_transfer_t *transfer); /* release */
int eshkol_transfer_take(
    eshkol_transfer_t *transfer,
    eshkol_memctx_t *receiver,
    eshkol_tagged_value_t *roots,
    size_t root_count);              /* acquire; consumes capsule token */
```

Export deep-copies only the result closure into a fresh producer-private capsule arena, or adopts an entire child arena when static ownership proves it contains no unwanted garbage and no outward pointers. Seal verifies closure, changes `OPEN -> SEALED` with release ordering, and revokes producer mutation. Take acquires the sealed roots and either:

- makes the capsule an owned child residence of the receiver;
- adopts its sealed blocks into a compatible receiver domain; or
- imports it as a resident transaction source; commit packs it into the next generation.

A capsule may retain an outward edge to an immutable input or immortal object only when that residence already outlives the receiver's use. The capsule records the dependency/reader lease, and take must complete before the producer releases it. Otherwise export copies the dependency into the capsule too. This makes the target decision depend on residence handles, never on which worker happened to hold an address.

There is no allocation into another thread's open private arena. A process-shared append-only target uses per-thread shards and publishes sealed chunks. A shared mutable slot accepts only an immutable capsule, a consumed unique value, or a resident/shared handle. Attempting to publish an ordinary aliased mutable private value is a compile-time ownership error where provable and a runtime `ESH_MEM_EALIAS` otherwise.

`parallel-map` currently lets workers write tagged results into caller-owned temporary slots and later builds the result list ([`lib/backend/parallel_codegen.cpp:474`](../../../lib/backend/parallel_codegen.cpp#L474), [`lib/backend/parallel_codegen.cpp:567`](../../../lib/backend/parallel_codegen.cpp#L567)). Under v2 each task slot carries a sealed capsule token as well as its tagged root. The joiner acquires every capsule before exposing the result list. Worker arenas can then reset after each task rather than retaining result graphs until thread-pool shutdown.

### Concurrency of deep evacuation

Deep evacuation is a transaction with explicit source and destination sets:

```c
typedef enum eshkol_promote_mode {
    ESH_PROMOTE_EXIT,
    ESH_PROMOTE_MOVE,
    ESH_PROMOTE_RESIDENT_COMMIT
} eshkol_promote_mode_t;

int eshkol_promote_roots(
    eshkol_memctx_t *owner,
    eshkol_residence_t *const *sources,
    size_t source_count,
    eshkol_residence_t *target,
    eshkol_tagged_value_t **root_slots,
    size_t root_count,
    eshkol_promote_mode_t mode);
```

- Source domains must be owner-private and stopped, sealed, or immutable under a reader lease.
- The target must be owner-private, a private transfer builder, or the caller's shard of a shared append-only domain.
- The forwarding key is `(source residence id, object id)`; the transaction has exactly one target.
- A work queue follows exact layout descriptors. No runtime recursion is required.
- Target allocations are staged behind an arena mark. On failure the target rewinds, the source stays live, and roots are unchanged.
- Roots are rewritten only after the traversal succeeds. Publication is the final release operation.
- A second thread never mutates the same forwarding table. Concurrent exports have disjoint private sources/capsules; shared immutable sources may be copied independently while retaining `object_id`.

The existing Cheney-style worklist is the right local mechanism ([`lib/core/runtime_regions.cpp:632`](../../../lib/core/runtime_regions.cpp#L632)); v2 changes its ownership, completeness, failure atomicity, and publication boundary.

## 6. Resident sessions: forever-flat state under mutation

### The resident contract

Resident mode is opt-in because ordinary process globals currently mean “immortal.” A session declares a finite root table and hard budgets. Raw object references are borrowed only for the duration of one transaction or read epoch.

```c
typedef struct eshkol_resident eshkol_resident_t;
typedef uint32_t eshkol_root_id_t;
typedef struct eshkol_read_lease eshkol_read_lease_t;

typedef struct eshkol_resident_limits {
    size_t max_live_bytes;
    size_t max_scratch_bytes_per_thread;
    size_t max_transfer_bytes;
    uint32_t max_threads;
    uint32_t max_retired_epochs; /* resident profile requires 1 */
} eshkol_resident_limits_t;

int eshkol_resident_create(const eshkol_resident_limits_t *,
                           eshkol_resident_t **out);
int eshkol_resident_root_create(eshkol_resident_t *,
                                const eshkol_tagged_value_t *initial,
                                eshkol_root_id_t *out);
int eshkol_resident_read_begin(eshkol_resident_t *, eshkol_read_lease_t **out);
void eshkol_resident_read_end(eshkol_read_lease_t *);
int eshkol_resident_write_begin(eshkol_resident_t *, eshkol_resident_txn_t **out);
int eshkol_resident_commit(eshkol_resident_txn_t *);
void eshkol_resident_abort(eshkol_resident_txn_t *);
```

One writer owns a transaction. Readers borrow the currently published immutable generation. A writer sees a mutable logical view implemented by transaction-local copy-on-write. Provisional COW objects live in a bounded **delta scratch** residence, not in the next published arena; this lets commit pack only the final live closure:

1. `write_begin` selects and resets inactive state space `B`, creates per-thread scratch plus a writer delta residence, and snapshots roots from published space `A`.
2. Normal allocations go to per-thread scratch, not to `A` or the immortal arena.
3. On mutation of an `A` object, `eshkol_resident_resolve_mutable` copies that object to delta once and records `A.object_id -> delta.object`. The mutation applies to delta.
4. Every transactional read, mutation, and `eq?` resolves through this map. Thus a second alias to the old address observes the new logical object; `object_id` preserves identity.
5. A value stored into the logical persistent state may still point to `A`, delta, ordinary scratch, or an imported capsule; that is legal only inside the uncommitted transaction.
6. `commit` evacuates all registered roots from `A` plus the transaction-owned sources into empty `B`, applying the COW forwarding relation. Abandoned intermediate deltas are not copied. The result is closed: no pointer in `B` targets `A`, delta/scratch, a worker arena, or an unsealed capsule.
7. The session publishes `(root_table_B, epoch+1)` with one release operation. New readers acquire `B`; old readers finish on immutable `A`.
8. Before the next reuse of `A`, the writer waits for the old epoch's reader count to reach zero. It then resets `A`, all scratch regions, transfer capsules, and transaction maps wholesale.

```text
                              writer transaction
 published roots ──> A ── resolve/COW ──> delta + scratch + capsules
       readers ─────> A (immutable)                    │
                                                        │ evacuate final roots
                                                        v
                                               B (next, private)
                                                        │ validate + release
                                                        v
                                       published roots ──> B
                                quiesce readers of A; reset A
```

The next iteration swaps the names A and B. There is no “old generation that is never reclaimed.” Replaced list versions that are unreachable from the final root table are not copied into `B`; their bytes disappear when `A` and scratch reset. Cycles and shared tails are handled by the same forwarding relation.

### Why this survives persistent mutation

Consider a resident program's long-lived vector whose 64 slots are overwritten forever. The vector and the 64 final lists are live state. During a tick, scratch may contain arbitrary temporary lists; the modified vector is COW'd to delta, and new slot values may initially live in scratch. Commit copies only the vector's final 64 targets and any other registered state into `B`. After readers leave `A`, every older overwritten list is reclaimed at once. Uptime does not appear in the bound.

With `L` the configured maximum published live graph, `S_t` each active thread's scratch cap, `X` the transfer cap, `I` true immortal/static memory, and `M` metadata, the steady-state bound is:

```text
RSS_managed <= I + 2*L + sum(S_t) + X + M
```

There may temporarily be one published old generation and one next generation; `max_retired_epochs == 1` forbids a third. A reader that never releases its lease causes backpressure at the next rotation. The runtime reports the reader/task responsible and blocks or times out according to session policy; it does not silently grow another generation. If `B` exceeds `max_live_bytes`, commit fails atomically, `A` remains published, and the application can shed state or terminate predictably.

### Mutation and identity details

- **Aliases:** all transaction operations resolve by `object_id`; mutation through either alias reaches the one COW copy.
- **`eq?`:** heap identity compares `object_id` in resident/transfer-capable layouts. Address equality remains a valid fast path but not the semantic definition.
- **Globals:** a resident global is a stable root-table slot, not an immortal object slot. `set!` updates the transaction root.
- **Nested objects:** commit walks exact descriptors, so a vector -> closure -> mutable capture -> hash -> cycle is one graph.
- **Parallel reads:** workers take an `A` read lease. They cannot mutate `A`.
- **Parallel results:** workers return sealed capsules; the writer adds them to the transaction source set and commit imports their live closure into `B`.
- **Parallel mutation:** partitioned unique state may be moved to workers and reunited; otherwise mutation is serialized through the writer or explicit synchronization.
- **Continuations:** a continuation cannot retain a transaction borrow past commit. Capturing one either rejects, closes over a sealed capsule, or promotes its exact frame layout into a separately budgeted residence.
- **FFI/pinning:** a raw pointer is valid only until its lease ends. Long-lived foreign references require an explicit pinned/linear handle and are charged outside `L`; the caller must release them.
- **Abort:** because `A` is immutable, abort resets `B` and scratch and leaves published roots unchanged. External side effects are outside the memory transaction and are not rolled back.

### Per-iteration regions outside resident mode

Named-let/TCO loops should open a private `ESH_RES_ITERATION` region, not push a mark on a potentially shared arena. At the back edge or loop exit, all loop-carried values and remembered outward places are batch-promoted into the parent, then the entire iteration region resets. An escaping value no longer forces `arena_commit_scope` to retain unrelated garbage. With complete barriers, the current whole-program “mutation-free and not parallel-reachable” analysis becomes a barrier-elision/performance analysis, not a memory-safety gate.

For daemon code, an iteration region is nested inside the resident write transaction: discardable work stays in scratch, while the committed live closure lands in `B`. Wrapping an iteration in `with-region` alone is insufficient for forever-flatness because its escape target is still immortal.

## 7. Ownership surface and R7RS compatibility

The runtime policy aligns with Eshkol's existing `owned`, `move`, `borrow`, `shared`, and `weak-ref` surface and object flags ([`inc/eshkol/eshkol.h:450`](../../../inc/eshkol/eshkol.h#L450)), but those flags must become enforced capabilities rather than advisory bits.

- `owned T@R` is affine: it may be moved into a task/capsule or destroyed, not silently copied as another mutable owner.
- `borrow T@R` cannot outlive the region/task/read lease and cannot cross resident commit.
- immutable shared borrows may cross threads while their residence is sealed and leased.
- mutable sharing requires a resident root/transaction, a synchronized shared handle, or explicit reference-counted storage.
- legacy unannotated Scheme values remain freely aliasable within one thread and one open lexical lifetime; barriers dynamically promote at boundaries.

Full R7RS code can continue using immortal globals and ordinary `with-region`; it receives safety but no promise that arbitrary immortal-state churn is bounded. The forever-flat guarantee is attached to the resident-session contract. This makes the guarantee testable instead of pretending that an arena can discover when the last arbitrary Scheme alias disappeared.

## 8. Relation to prior memory models

### Tofte–Talpin regions

[Tofte and Talpin's region calculus](https://ropas.snu.ac.kr/lib/dock/ToTa1997.pdf) supplies the core insight: infer or pass region parameters, allocate into lexically scoped regions, and reclaim a region at a statically justified end. OALR keeps bulk deallocation and region polymorphism as compiler goals. It differs in two important ways: Scheme escape/mutation requires dynamic remembered edges and evacuation, and concurrency means the dynamic region stack belongs to an execution context—not to a process arena. Resident generations are non-LIFO residences selected by an explicit session.

### Cyclone

[Cyclone's region system](https://www.cs.cornell.edu/projects/cyclone/papers/cyclone-regions.pdf) combines region subtyping/effects with stack, dynamic, and heap regions; its [unique and reference-counted pointer work](https://homes.cs.washington.edu/~djg/papers/old/unique-abstract.html) makes alias policy explicit. Eshkol should take the same lesson: lifetime and aliasability are separate dimensions. A mutex does not turn an aliasable shared region into a uniquely rewindable region. Transfer capsules correspond to unique dynamic regions; sealed borrows correspond to shared immutable access; resident roots are the controlled escape hatch for mutable identity.

### Linear and affine types

[Linear Regions Are All You Need](https://www.cs.cornell.edu/people/fluet/research/substruct-regions/ESOP06/esop06.pdf) shows how substructural capabilities can encode LIFO regions, dynamic regions, and unique pointers. Eshkol's owner token is affine in practice: it may be consumed at most once (dropping an unused arena is allowed), while a transfer seal/take token is linear at the ABI boundary. This is what makes zero-copy block adoption and deterministic destruction sound. General Scheme values remain aliasable; only ownership capabilities are affine.

### Rust lifetimes and borrowing

Rust references carry lifetimes/outlives relations, and its MIR borrow checker computes regions over control flow ([Rust compiler guide](https://rustc-dev-guide.rust-lang.org/borrow_check.html)). Eshkol should emulate the separation between a zero-cost checked borrow and ownership of storage, especially for task inputs and resident reader leases. It should not copy Rust wholesale: Scheme graphs, mutation, continuations, and dynamic typing require barriers and runtime checks where proof is unavailable. As in Rust, preventing lifetime UAF is not the same as synchronizing a logical data race.

### Cheney copying and generational collection

[Cheney's nonrecursive compacting algorithm](https://doi.org/10.1145/362790.362798) provides the worklist/forwarding mechanism already recognizable in Eshkol's deep escape. [Ungar's generation scavenging](https://doi.org/10.1145/390011.808261) contributes the young/old observation, survivor copying, and remembered older-to-younger stores.

OALR deliberately borrows mechanisms, not the autonomous collector policy:

- a lexical/iteration region is analogous to a nursery, but its destruction point is explicit;
- the escape ledger is analogous to a remembered set, but it is scoped to a known source region;
- promotion copies an explicit escaping closure, not all roots of a global heap;
- resident A/B rotation copies from an explicit registered root table at `commit`, not on an allocator-triggered GC schedule;
- hard budgets and reader backpressure replace “grow the heap and collect later.”

Calling resident commit “not GC” must not obscure the algorithm: it is reachability copying over an explicitly managed residence. The architectural claim is deterministic scope and root ownership, not the absence of tracing instructions.

## 9. Rejected alternatives

### Lock the existing global region/scope stack

Rejected. Per-operation locking does not restore LIFO across interleaved thread lifetimes or make rewind distinguish allocations. Scope-duration locking serializes user code and is not robust across exceptions, callbacks, or nested parallelism.

### Keep TLS stacks but continue swapping `__global_arena`

Rejected. This is the current contradiction: region selection is TLS while generated allocation is process-global. Fixing only the stack leaves the decisive race.

### Make every arena thread-safe and lock scope operations

Rejected. A bump allocator mutex prevents duplicate allocation ranges; it cannot make one thread's rollback skip another thread's interleaved bytes. Shared domains need shards or no rewind.

### Eagerly copy on every outward store

Rejected for same-thread mutable values. It produces snapshot-not-alias behavior unless every later access follows forwarding indirection. Deferred repair at lexical close is cheaper and preserves ordinary Scheme behavior. Eager export remains correct when publication consumes or freezes source mutation rights.

### Persist all promotions in the immortal arena

Rejected for resident state. It prevents UAF but converts every overwritten version into permanent residence, so memory is proportional to uptime.

### Reference-count every Scheme object

Rejected as the default. It taxes every edge update, needs atomic counts across threads, and does not reclaim cycles without an additional cycle algorithm. Explicit `shared` handles remain useful for objects whose lifetime cannot fit lexical, transfer, or resident epochs.

### Reset an arena when shallow loop out-values do not point into it

Rejected as a general rule. Mutation can hide an inward pointer transitively in an old container; foreign tasks and continuations can hide roots too. Exact barriers plus batch promotion make the proof local and dynamic.

### Add an unbounded queue of resident generations for slow readers

Rejected. It preserves throughput by abandoning the forever-flat guarantee. Resident mode applies backpressure after one retired generation.

## 10. Migration plan

### Phase A — expose the true allocation context

1. Add memory ABI versioning and `eshkol_memctx_current`/`eshkol_current_arena`.
2. Convert all LLVM codegen modules and runtime helpers from direct `__global_arena` loads to the accessor.
3. Make `with-region` push/pop only TLS context state; remove the global save/store sequence.
4. Add owner assertions and thread attach/detach diagnostics.

Exit gate: concurrent `parallel-map` callbacks may nest `with-region` without global-slot writes, races, or allocator ownership violations.

### Phase B — private tokenized scopes

1. Replace `arena_scope_t *current_scope` with value marks owned by frames.
2. Give every TCO iteration a private/reusable scratch region.
3. Seal/consume the source token for block adoption.
4. Remove the parallel-reachability safety exclusion after tests prove the runtime invariant; retain analysis for optimization.

Exit gate: TSAN stress with nested TCO loops on every worker; no shared scope metadata and no committed unrelated iteration garbage.

### Phase C — exact graph ABI and alias-preserving exits

1. Introduce header/layout ABI v2 and convert every pointer-bearing producer.
2. Replace legacy shallow escape entry points with `eshkol_promote_roots` wrappers.
3. Change mutation lowering to typed stores/places and add escape ledgers.
4. Route normal, exceptional, and continuation-supported exits through one close operation.
5. Make promotion failure atomic and test injected OOM at every allocation.

Exit gate: store-then-mutate, mutate-through-two-aliases, cycles, shared tails, hash resize, closures with mutable captures, and every registered subtype remain correct after source destruction.

### Phase D — structured cross-thread transfer

1. Add transfer builders and capsule tokens to thread-pool task/result ABIs.
2. Borrow inputs until join; export worker results before task completion.
3. Reset worker task arenas after capsule publication.
4. Reject raw private pointers at channel/future/atomic/FFI publication boundaries.

Exit gate: nested parallel map/fold/filter/execute with region allocation; randomized work stealing; cancellation; producer OOM; and worker shutdown cannot expose or free a foreign live pointer.

### Phase E — resident sessions

1. Add root tables, A/B spaces, writer transactions, COW resolution, and reader epochs.
2. Integrate globals/loop-carried state selected for resident mode.
3. Import worker capsules into the next generation and enforce join/lease closure at commit.
4. Add byte budgets, stale-reader diagnostics, and atomic abort.

Exit gate: a mutating daemon runs at least 10 million iterations with constant live shape; managed RSS stays within the formula above after warm-up, every final state read is correct, alias/`eq?` tests pass, and a stalled reader causes bounded backpressure rather than a third generation.

## 11. Verification matrix

At minimum, CI must cover:

| Axis | Required cases |
|---|---|
| Thread isolation | 1/2/N workers; nested `with-region`; nested TCO; work stealing; caller warmup racing worker entry |
| Escape identity | store then mutate source; mutate via outer alias before close; two destinations; shared tail; self-cycle; mutually recursive closures |
| Layout completeness | one escape/commit fixture for every pointer-bearing `layout_id`; startup failure for unregistered layout |
| Cross-thread transfer | borrowed input; unique move; immutable share; illegal mutable alias; cancellation before/after seal; receiver abandonment |
| Resident correctness | root replacement; in-place pair/vector/hash mutation; cycles; closure captures; exception/abort; import capsule |
| Boundedness | constant live graph for 10M+ ticks; forced max-live OOM; stale reader; maximum workers/scratch; no third epoch |
| Failure atomicity | allocation failure at header copy, raw buffer copy, worklist growth, root rewrite staging, seal, and commit validation |
| Tooling | ASan/UBSan/TSan, arena poison, deterministic stress scheduler, ownership/residence event log |

Useful counters are per-residence, not just per-arena: allocated bytes, live bytes at commit, copied objects/bytes, escape-ledger entries, capsule bytes, COW hits, commit duration, reader wait, high-water RSS, and rejected publication attempts. A debug build should print an ownership path such as `thread 7 / txn 41 / scratch 3 -> resident B` for any dangling or illegal edge.

## Consequences

### Positive

- `with-region` and iteration reclamation become structurally thread-safe rather than conservatively disabled in parallel code.
- Deep escape preserves cycles, sharing, **and post-store alias semantics**.
- Worker results acquire an explicit lifetime; worker arenas can be reclaimed per task.
- Resident memory is bounded by configured live/scratch/transfer capacity, independent of uptime and overwrite count.
- OOM, stale readers, illegal publication, and unsupported layouts fail at defined boundaries instead of producing delayed SIGSEGVs.
- The runtime has one memory model across AOT, JIT, REPL, callbacks, and eventually VM execution.

### Costs and constraints

- The object header grows and all runtime/stdlib/JIT components require an ABI transition.
- Every pointer-bearing layout needs an exact descriptor; runtime authors cannot hide tagged pointers in raw buffers.
- Same-thread mutation barriers record outward places; resident mutation adds COW resolution and commit work proportional to live state.
- Cross-thread mutable publication becomes more explicit. Some formerly accidental behavior will be rejected.
- Forever-flat resident mode can block behind readers and cannot promise both bounded space and unbounded reader latency.
- FFI pins, external resources, explicit shared/RC objects, and true immortal data must be budgeted separately.

## Final architectural boundary

**Lexical regions answer “when may this thread bulk-free scratch?” Transfer capsules answer “who owns this graph after a thread handoff?” Resident epochs answer “which bounded version of persistent state is published?”** An arena is only the allocator underneath those answers. It must never again be used as the global lifetime model.
