# OALR object ABI v2 — migration inventory

- Status: Phase A landed (definitions + layout pinning); Phases B/C not started
- Governs: ADR-0000 Stage 4 (v1.4.1), ADR-0001 §3 "Object and layout ABI v2"
- Measured at: `origin/master` 73cc7cbb, arm64-apple-darwin24.1.0, LLVM 21
- Every count below is reproducible with the commands in [§7](#7-reproducing-the-counts)

---

## 1. What this document is

ADR-0001 §3 replaces the 8-byte `eshkol_object_header_t` with a 32-byte header
that adds an exact **layout descriptor id**, a **stable object identity**, and
the object's **residence (home)**. The layout id is the field the deep-walk
evacuator needs: ADR-0001 §1 principle 6 forbids "unknown means leaf", and
today's evacuator still has to *guess* — `evac_kind_for()` in
`lib/core/runtime_regions.cpp` decides how deeply to copy from the subtype byte
plus an "integer that resembles an arena address" heuristic, with a documented
watchlist of subtypes it knowingly leaves shallow (ESH-0214d/e).

Growing the header is an **internal memory-ABI break**. Nothing about it is
local: the header is a *prefix every translation unit computes independently*,
and it is invisible in every symbol signature, so a half-migrated toolchain does
not fail to link — it produces garbage. This document is the honest inventory of
what must move, and the argument for why it must move **all at once**.

## 2. What Phase A landed (this PR)

Strictly additive; default off; the live object ABI is unchanged.

| Artifact | What it does |
|---|---|
| `inc/eshkol/memory_abi_v2.h` | Defines `eshkol_object_header_v2_t` (32 B, 16-B aligned) exactly as ADR-0001 §3 writes it, plus `eshkol_layout_desc_t`, the layout flag bits, and the opaque `eshkol_residence_t`. |
| same file | Static-asserts **both** layouts: v1 is 8 bytes with `subtype@0`/`flags@1`/`size@4`; v2 is 32 bytes with `payload_size@0`/`layout_id@4`/`subtype@6`/`flags@7`/`object_id@8`/`home@16`/`aux@24`. A migration PR that moves a field fails to compile. |
| same file | `ESHKOL_MEMORY_ABI_ACTIVE`, `eshkol_object_header_active_t`, `ESHKOL_OBJECT_HEADER_SIZE`, `ESHKOL_OBJECT_PAYLOAD_ALIGN`. |
| `lib/core/memory_abi_v2.cpp` | `eshkol_memory_abi_active()` — the ABI version the *runtime archive* was built with, so a mismatch against a caller becomes observable. |
| `CMakeLists.txt` | `option(ESHKOL_MEMORY_ABI_V2 ... OFF)` → whole-project `ESHKOL_MEMORY_ABI_V2_ENABLED=1`, and a configure-time `Eshkol object ABI: v1/v2` status line. |
| `tests/core/memory_abi_v2_test.cpp` | CTest `memory_abi_v2_test`. Checks both layouts, the descriptor record, that the runtime and the caller agree, and that the flag state selects the ABI it claims. Passes in **both** flag states. |

Note the naming: `ESHKOL_MEMORY_ABI_V2` is the ABI **version number** (value 2,
as ADR-0001 writes it). `ESHKOL_MEMORY_ABI_V2_ENABLED` is the **build switch**.
Deliberately different spellings so a test for one can never be mistaken for the
other.

## 3. The inventory

Five distinct classes of site, in increasing order of how badly they break.

### Class A — references to the header type (154 references, 40 files)

Anything that names `eshkol_object_header_t`. Mechanically retypeable to
`eshkol_object_header_active_t` **once the fields line up** — but v2 renames
`size`→`payload_size` and drops `ref_count` entirely, so every field access has
to be looked at, not sed'ed.

| File | Refs | Note |
|---|---:|---|
| `lib/core/runtime_object_alloc.cpp` | 19 | the allocator itself; see Class E |
| `lib/core/runtime_taylor.c` | 13 | C translation unit — needs the C-compatible header |
| `lib/core/runtime_regions.cpp` | 10 | the evacuator; the *reason* for v2 |
| `inc/eshkol/eshkol.h` | 9 | type + macros + one inline reader |
| `lib/core/runtime_tensor_index.cpp` | 7 | |
| `lib/core/runtime_tensor_alloc.cpp` | 7 | |
| `lib/core/runtime_deep_equal.cpp` | 7 | |
| `lib/core/logic.cpp` | 7 | |
| `lib/core/introspection.cpp` | 6 | |
| `lib/core/runtime_autodiff.cpp` | 5 | |
| 30 further files | ≤4 each | incl. `lib/agent/c/*.c`, `lib/ffi/eshkol_ffi.cpp`, 3 test files |

`ref_count` deserves a specific callout: v2 has no counterpart, because ADR-0001
subsumes refcounting into residence ownership plus the transfer-capsule
protocol (§5). Every `ref_count` read/write is therefore a **semantic** port,
not a rename, and must be enumerated before any mechanical pass runs.

### Class B — header-size arithmetic (43 sites, 14 files)

`sizeof(eshkol_object_header_t)` used as an *offset* or an *allocation
overhead*. These are the sites that silently compute the wrong address if the
size changes under them.

| File | Sites |
|---|---:|
| `lib/core/runtime_object_alloc.cpp` | 14 |
| `lib/core/runtime_regions.cpp` | 6 |
| `lib/core/runtime_taylor.c` | 4 |
| `inc/eshkol/eshkol.h` | 3 |
| `lib/core/symbol_intern.cpp`, `runtime_tensor_alloc.cpp`, `runtime_hash_table.cpp`, `runtime_exceptions_hosted.cpp`, `runtime_closure_alloc.cpp`, `runtime_autodiff.cpp` | 2 each |
| `runtime_string.cpp`, `runtime_continuations.cpp`, `agent_system_builtins.c`, `eshkol_ffi.h` | 1 each |

These are the *easy* ones: they already spell the size symbolically, so
retargeting them at `ESHKOL_OBJECT_HEADER_SIZE` is correct by construction.

### Class C — header access macros (121 uses, 32 files)

`ESHKOL_GET_HEADER` / `ESHKOL_GET_DATA_PTR` / `ESHKOL_GET_SUBTYPE` /
`ESHKOL_SET_SUBTYPE` / `ESHKOL_GET_FLAGS` / `ESHKOL_SET_FLAGS` /
`ESHKOL_HAS_FLAG`. Because these are macros over `sizeof(...)`, redefining them
against the active header migrates all 121 at once — **provided** every object
in the process has the same header. That proviso is the whole argument of §5.

### Class D — hard-coded `-8` header offsets in emitted LLVM IR (45 sites, 9 files)

This is the class that makes the migration a compiler change rather than a
runtime change. Generated code does not call the macros; it emits the offset as
an IR constant:

```cpp
// lib/backend/arithmetic_codegen.cpp:2552
Value* header_ptr = ctx_.builder().CreateGEP(
    ctx_.int8Type(), heap_ptr, llvm::ConstantInt::get(ctx_.int64Type(), -8));
Value* subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), header_ptr, "heap_subtype");
```

| File | Sites |
|---|---:|
| `lib/backend/llvm_codegen.cpp` | 15 |
| `lib/backend/autodiff_codegen.cpp` | 13 |
| `lib/backend/collection_codegen.cpp` | 9 |
| `lib/backend/hash_codegen.cpp` | 2 |
| `lib/backend/arithmetic_codegen.cpp` | 2 |
| `lib/backend/tensor_shape_codegen.cpp` | 1 |
| `lib/backend/tensor_creation_codegen.cpp` | 1 |
| `lib/backend/tagged_value_codegen.cpp` | 1 |
| `lib/backend/string_io_codegen.cpp` | 1 |

Note two independent constants are baked in at each site, not one: the **header
size** (`-8`) *and* the **field offset within the header** (subtype at +0 today;
at +6 under v2, since `payload_size` and `layout_id` precede it). A site that
loads `subtype` today needs `-32 + 6 = -26`, not `-32`. Nothing in the build
currently checks that the frontend's idea of the offset matches the runtime's;
the static assertions added in Phase A are the first half of that check, and
Phase B owes the second half — a single `headerFieldOffset()` helper in the
codegen context, so the constant exists once.

The bytecode VM is **not** in this class: `lib/backend/vm_*.c` contains **zero**
references to `eshkol_object_header_t` — the VM carries its own `Value` /
`HeapObject` representation. VM/native parity for ABI v2 is therefore a question
about the *boundary* (ESKB, FFI, shared runtime helpers), not about the VM's
own object model.

### Class E — allocation entry points (141 calls, 43 files)

Every heap object is born through one of ten `*_with_header` constructors:

| Entry point | Calls |
|---|---:|
| `arena_allocate_with_header` | 68 |
| `arena_allocate_string_with_header` | 57 |
| `arena_allocate_cons_with_header` | 35 |
| `arena_allocate_vector_with_header` | 17 |
| `arena_allocate_closure_with_header` | 16 |
| `arena_allocate_tensor_with_header` | 14 |
| `arena_allocate_ad_node_with_header` | 14 |
| `arena_allocate_symbol_with_header` | 12 |
| `eshkol_make_exception_with_header` | 4 |
| `arena_hash_table_create_with_header` | 3 |

(Counts include declarations and the definitions themselves; the funnel is what
matters — the *header write itself* happens in only a handful of places.)

That funnel is the good news: `arena_allocate_with_header` is where the header
is actually stamped, and it is one function. The bad news is in the same
function:

```c
size_t total_size = sizeof(eshkol_object_header_t) + data_size;
total_size = (total_size + 7) & ~((size_t)7);
void* raw = arena_allocate_aligned(arena, total_size, 8);
```

ABI v2 requires **16-byte** payload alignment (the header is `alignas(16)` so
that every `eshkol_tagged_value_t` slot in the payload lands aligned). So the
allocator's alignment argument, the round-up mask, and every arena block's own
base alignment have to move from 8 to 16 in the same change. Arena block
bookkeeping is not in Class A-D and is easy to miss.

Every constructor also has to supply a `layout_id`, which does not exist yet.
Per ADR-0001 §3 the descriptor registry must be populated for conses,
vectors/records, multiple values, closures/environments, hash backing arrays,
exceptions, tensors, promises, substitutions, knowledge bases, factor graphs,
workspaces, DNC/SDNC, Taylor values, parameters and continuations — and
registration failure must be a **startup error**, so partial registration is not
a valid intermediate state either.

## 4. Sites that are not counted above but still break

- **Precompiled stdlib bitcode / object cache.** `lib/repl/repl_jit.cpp:2617`
  keys the cached stdlib object on a *content hash of the stdlib sources* plus
  the target triple: `"stdlib-jit-v4-" + utohexstr(content_hash) + "-" + triple`.
  It does **not** include the object ABI. Flipping `ESHKOL_MEMORY_ABI_V2` and
  re-running therefore reuses a stdlib object compiled against the other header
  — a silent miscompile, not a cache miss. Phase B must fold the ABI version
  into that key (and into the AOT JIT-run cache key built by
  `makeJitRunCacheKey`, `exe/eshkol-run.cpp:575`) **before** any allocation site
  moves.
- **Emitted AOT objects and `--shared-lib` artifacts** linked against a runtime
  built with the other flag state. Same failure mode; needs the startup ABI
  check that `eshkol_memory_abi_active()` now makes possible.
- **The `.eskb` bytecode format**, to the extent it embeds runtime-object
  assumptions at the FFI boundary.
- **The WASM and macos-x64 lite lanes**, which ADR-0000 Stage 4 names as the
  explicit blast-radius falsifier: "if they cannot rebuild green under ABI v2,
  ABI v2 does not ship in this cycle."

## 5. Why the migration cannot be staged one subsystem at a time

Deliverable (d) of this work item was to migrate one self-contained subsystem
behind the flag as a proof of concept, *if it is clean to do so*. It is not, and
the reason is structural rather than a matter of effort.

**The header size is the thing you must know before you can read the header.**
`ESHKOL_GET_HEADER(data)` is `data - sizeof(header)`, and `subtype` — the only
field that says what kind of object this is — lives *inside* that header. So to
find out whether a given object is a v1 object or a v2 object, you must first
read its subtype; to read its subtype you must first know whether it is a v1 or
a v2 object. There is no discriminator at a fixed negative offset from the
payload, and there cannot be one without a *third* format that both versions
agree on.

Concretely, a "cons cells only" migration fails on the first `car`:
`runtime_deep_equal.cpp`, `runtime_display_hosted.cpp`, `logic.cpp` and 29 other
files all reach a generic `eshkol_tagged_value_t` whose `ESHKOL_VALUE_HEAP_PTR`
could be either kind, and each of the 45 Class-D IR sites emits a *constant*
offset chosen at compile time for a value whose kind is only known at run time.

The two escapes both cost more than the migration they were meant to de-risk:

1. **Tag the pointer.** Steal a low bit of `data.ptr_val` to mean "v2 object".
   The tagged value is 16 bytes with 8 bytes of payload, so the bit exists — but
   this touches every producer and consumer of `ptr_val`, i.e. a superset of
   Classes A-E, and it is throwaway work: the bit is deleted again when the
   migration completes.
2. **Version by residence.** Make the arena, not the object, carry the version,
   and branch on the arena at every header access. That is a runtime load and a
   branch in front of *every* `car`, and it does not help the Class-D IR sites,
   which would each become a diamond.

The honest conclusion is the one ADR-0000 §6 already anticipated when it named
"OALR ABI v2 blast radius" as a top risk: the header change is **atomic by
construction**. It ships as one coordinated change across the allocator, the
evacuator, the 45 codegen sites, the stdlib cache keys and the lite lanes, or it
does not ship. What *can* be staged is everything around it, which is what
Phase A does: the layouts are pinned, the version is observable at both compile
time and run time, and the flag exists so the eventual atomic change can be
developed and tested without being on by default.

## 6. Phase B/C order of operations

The order is forced by §4 and §5, not chosen:

1. **Cache-key correctness first.** Fold `ESHKOL_MEMORY_ABI_ACTIVE` into the
   stdlib object cache key (`repl_jit.cpp`) and the JIT-run cache key
   (`eshkol-run.cpp`). Without this, every later step can be poisoned by a stale
   artifact and the failure looks like a codegen bug.
2. **Hard startup ABI check.** Generated programs assert
   `eshkol_memory_abi_active() == ESHKOL_MEMORY_ABI_ACTIVE` at startup and abort
   loudly on mismatch. Ship this while v1 is still live, so it is exercised
   before it is needed.
3. **One header-offset helper in the codegen context**, replacing all 45
   Class-D constants with `headerFieldOffset(Field::Subtype)` etc. This is a
   pure refactor under v1 and can land and be verified independently.
4. **Layout-descriptor registry** + registration for every type ADR-0001 §3
   lists, with a startup check that every registrable subtype has a descriptor.
   Still under v1: the registry is simply unused.
5. **Arena alignment 8 → 16** including block base alignment.
6. **The atomic header change** behind the flag, then the ADR-0000 Stage-4 exit
   gate: full suite green in both flag states, plus WASM + macos-x64 lite lanes
   rebuilt green under v2.
7. Escape ledgers, deferred same-thread stores, transfer capsules (ADR-0001
   §4-5) — these consume `layout_id` and `home` and cannot start before step 6.

Steps 1-5 are all strictly additive under ABI v1 and can each be their own PR.

## 7. Reproducing the counts

```sh
B=origin/master
# Class A
git grep -c 'eshkol_object_header_t' $B -- 'lib/**' 'inc/**' 'tests/**' 'exe/**'
# Class B
git grep -c 'sizeof(eshkol_object_header_t)' $B -- 'lib/**' 'inc/**' 'tests/**' 'exe/**'
# Class C
git grep -c -E 'ESHKOL_(GET_HEADER|GET_DATA_PTR|GET_SUBTYPE|SET_SUBTYPE|GET_FLAGS|SET_FLAGS|HAS_FLAG)' \
    $B -- 'lib/**' 'inc/**' 'tests/**' 'exe/**'
# Class D
git grep -c -- ', -8)' $B -- 'lib/backend/**'
# Class E
git grep -o -E '[a-z_]+_with_header' $B -- 'lib/**' 'inc/**' | sed 's/.*://' | sort | uniq -c | sort -rn
# VM is uninvolved (expect no output)
git grep -l 'eshkol_object_header_t' $B -- 'lib/backend/vm_*.c'
```
