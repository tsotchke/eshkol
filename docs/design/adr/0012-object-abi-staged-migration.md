# ADR-0012: Object ABI — discrimination, enforcement, and the staged migration

- **Status:** Accepted through Stage 2 (COMPLETE); Proposed for Stages 3-6
- **Date:** 2026-08-25
- **Decision owners:** Eshkol memory/runtime architecture
- **Cluster:** OALR / object model / toolchain integrity
- **Applies to:** LLVM AOT and JIT, the runtime archive, the bytecode VM, the WASM lane, persisted formats, and the public embedding ABI
- **Relates to:** ADR-0001 Phase C (header/layout ABI v2), ADR-0000 §6 (blast radius), `docs/design/ABI_V2_MIGRATION_INVENTORY.md` (Phase A, #478)

## Relationship to ABI v2 Phase A

Phase A (#478) landed the destination and the switch: `eshkol_object_header_v2_t`
with both layouts statically asserted, `ESHKOL_MEMORY_ABI_ACTIVE`,
`ESHKOL_OBJECT_HEADER_SIZE`, and a project-wide `ESHKOL_MEMORY_ABI_V2` option
that is off by default. It also recorded, in its own words, the problem this ADR
exists to solve: *"The header prefix appears in no symbol signature, so a
half-migrated toolchain does not fail to link, it produces garbage."*

This ADR supplies the enforcement layer for that switch, and is coupled to it
rather than parallel to it. The guard symbol's name is derived from
`ESHKOL_MEMORY_ABI_ACTIVE`, so configuring with `-DESHKOL_MEMORY_ABI_V2=ON`
renames the symbol on its own. The flip is protected by construction: nobody has
to remember to protect it, and a build half-flipped between the two layouts
cannot link. Phase A's stated Phase-B goal — "make a mismatch a hard startup
error rather than a silent miscompile" — is met one stage earlier and one stage
stronger, at link time rather than at startup, so the bad artifact is never
produced rather than being caught while running.

Phase A's inventory is a hand survey, reproducible from the commands in its §7.
The machine inventory below supersedes it as the enforced artifact and confirms
its class D count exactly — 45 sites across 9 files — while finding three
classes it could not: see the table.

## Context

Every heap-allocated Eshkol object is a payload preceded by an
`eshkol_object_header_t`. The pointer that circulates through compiled code, the
runtime, the FFI and the embedding API points at the payload; the header is
reached by subtracting its size. `subtype` — the field that says what kind of
object this is — lives inside that header.

One consequence governs everything that follows.

> **An object carries no discriminator for its own layout.** Given a payload
> pointer there is no test that distinguishes an object laid out one way from an
> object laid out another. Reading the field that would tell you already assumes
> the answer.

So two halves of a toolchain that disagree about the layout do not fail to
understand each other. They agree, confidently, and read the wrong bytes.
Nothing crashes; the program returns wrong answers. The failure has no
signature, no log line, and no stack trace.

That is the property this ADR is about. It is not a difficulty to be worked
around on the way to a header change — it is the reason the header change is the
largest single schedule risk in the project, and the thing that has to be
neutralised before the change is attempted rather than during it.

### What the machine inventory found

The site inventory is produced by `scripts/abi_header_inventory.py`, is
regenerated from `git ls-files` on every run, and is snapshotted at
`docs/design/abi/header-site-inventory.json` with a readable companion beside
it. Seventeen detectors across three layers — lexical token matching, libclang
semantic resolution over `compile_commands.json`, and the compiler's own emitted
IR — each recording *how* it found what it found, so the finding method is
auditable and rerunnable rather than trusted.

The layers are load-bearing in different ways. The **lexical** layer is what CI
enforces, because it must run anywhere with no build and no third-party
dependency. The **semantic** layer resolves what token matching cannot — a
member access through a typedef, a `sizeof *hdr`, a cast written through an
alias — and it refuses to report a clean result when a translation unit failed
to parse, because a scan that finds nothing because it read nothing is the
worst possible answer. The **emitted** layer is ground truth: it compiles a
corpus and counts header offsets that actually reach the IR, so a source
construct that produces an offset by a route no detector models still shows up.

The relevant result is not the total. It is which classes exist:

| Class | What | Found by |
|---|---|---|
| A | `ESHKOL_GET_HEADER` family uses | token match on the macro names |
| B | references to the header type | token match |
| C | `sizeof(header)` as an offset or overhead | token match |
| D | `-8` baked into emitted LLVM IR as a constant | regex over the code generator |
| **D2** | **`-7`, `-6`, `-4` — field offsets *inside* the header, in emitted IR** | **derived offset set, not a `-8` search** |
| E | calls through the `*_with_header` constructor family | naming convention |
| F | structs that structurally duplicate the header under another name | field-sequence comparison |
| G | field access through a header-typed pointer | token match, file-scoped |
| H | the header re-implemented in JavaScript for the WASM lane | JS token and offset match |
| I | layout dependence inside a public installed header | derived from A-G |
| J | the *second* prefix-header ABI (`eshkol_shared_header_t`, 24 bytes) | token match |
| **K** | **raw byte arithmetic that rebuilds the header, naming neither the type nor the macro** | **byte-pointer cast followed by `- 8`** |
| L | the layout restated in comments and published documentation | raw-text match, comments included |
| **M** | **persistent caches whose key omits the object ABI** | **detector for an absence** |
| S1-S3 | member access, `sizeof`, and casts resolved by type rather than by spelling | libclang over `compile_commands.json` |
| R | header offsets that actually reach emitted LLVM IR | `--dump-ir` over a corpus |

At the commit this ADR was written against, the lexical layer reports 816 sites
across 98 files; adding the semantic layer brings the enumerated total to 1273,
and the emitted layer counts 66 header GEPs across the corpus. The published
hand survey reported 504. The gap is not carelessness in that survey — it is
three classes the search it used could not express.

Those three are the argument for machining this inventory.

**D2 is invisible to the published search.** The header size is 8, so a survey
that greps for `-8` finds the sites that read `subtype` and none of the sites
that read `flags` (`-7`), `ref_count` (`-6`) or `size` (`-4`). A migration that
changes the header size and diligently fixes every `-8` leaves this class
reading the wrong field of the right header — a subtler and later-surfacing
corruption than getting the header wrong outright. The detector derives the
offset set from the layout rather than sweeping a range, so it reports these ten
sites and no unrelated small constants.

**K contains no token any type-directed or macro-directed search can match.**
`uint8_t subtype = *(ptr - 8);` reads as ordinary byte arithmetic. Nine of these
exist, one of them in the FFI layer that embedders link against.

**M is a detector for something that is not there.** A cache key that omits the
object ABI hands back an artifact built against the other layout, and the
program runs. Absence is the hardest property to keep an eye on by hand, and it
is the one that turns a correct migration into a wrong-answer bug on someone
else's machine three weeks later.

Two further findings from the same pass, recorded because they change the plan:

- `VmObjectHeader` (`lib/backend/vm_arena.h`) structurally duplicates the object
  header with no static assertion tying the two together, **and its subtype
  numbering has already diverged** — `VM_SUBTYPE_STRING` is 2 where
  `HEAP_SUBTYPE_STRING` is 1. Two definitions of one ABI, already disagreeing.
- `eshkol_shared_header_t` is a second, independent 24-byte prefix header using
  the identical `data - sizeof(header)` idiom and the identical field names
  (`ref_count`, `flags`). A migration that pattern-matches on the idiom rather
  than on the type will convert these by accident.

## The discriminator problem

The staging question, stated precisely: **during a migration, how do objects of
the old layout and objects of the new layout coexist, when the tag that would
distinguish them lives inside the part that moved?**

Four mechanisms were considered.

### Option 1 — a version bit in the tagged value

`eshkol_tagged_value_t` has a spare `uint8_t flags` and a 16-bit `reserved`
field. There is room for a layout bit without stealing a pointer bit at all.

Rejected, and the reason is structural rather than economic. **The discriminator
would live on the reference, and references are erased at the C boundary.** The
runtime is full of functions that take a bare `void*` payload pointer — the FFI
layer, the evacuator, the display path, the deep-equal walk. At exactly those
sites there is no tagged value in hand to consult. A discriminator that is
absent where the decision is made is not a discriminator.

The secondary costs confirm it: every construction site must set the bit, every
one of the fifty-five class D/D2 sites becomes a diamond rather than a GEP, and
the whole apparatus is deleted the day the migration completes.

### Option 2 — a sentinel magic word at a fixed negative offset

Reserve a word immediately below the payload, identical in both layouts, holding
a magic that names the layout. `*(uint32_t *)(p - 4)` then answers the question
for any payload pointer, tagged or bare — which is exactly what Option 1 could
not do.

The v1 header has no room: all eight bytes are occupied, and `-4` is `size`. The
magic would have to go *below* the v1 header, at `-12`, which those bytes do not
belong to — they belong to whatever was allocated before. Making them belong to
the object means over-allocating and writing the magic in every allocator: a
change to allocation, not to the payload-offset relationship, and so genuinely
additive with respect to all 767 read sites.

So it is implementable. It is rejected anyway, for a reason worth stating
plainly because it is what makes the whole problem smaller than it looks:

> **In-memory coexistence is not a real requirement.** Two object layouts can
> only meet inside one process if a mixed toolchain was allowed to link. A
> process either has one layout or is already broken. The discriminator would
> exist solely to let a broken process limp — and limping is worse than
> stopping, because limping is silent.

Coexistence *is* a real requirement in one place: data that outlives the
process. Persisted `.eskb` weight files and the knowledge-base format carry
subtype bytes on disk and are read back by a later build. That is where a
discriminator belongs, and those formats already have version fields — which
Stage 5 makes carry the object ABI explicitly.

Paying eight bytes on every heap object, forever, to solve an in-memory problem
that should not be permitted to arise is the wrong trade.

### Option 3 — atomic flip, with a fingerprint that makes a mixed link fail loudly

Accept that the header change is atomic by construction: one build, one layout,
no coexistence. Then spend the effort not on making mixed builds work, but on
making them **impossible to produce silently**.

Encode the layout in the *name* of a guard symbol. Every participant emits a
reference to it; the runtime defines exactly one. Halves that disagree name
different symbols, and the link fails with an error naming the layout the code
expected.

This is the recommendation, and it is landed. `inc/eshkol/abi_fingerprint.h`
derives the symbol name from four numbers — ABI version, header size, subtype
offset, payload alignment — by token pasting, and selects those numbers from
`ESHKOL_MEMORY_ABI_ACTIVE`, so the name cannot fail to change when the layout
does. `lib/core/abi_fingerprint.c` holds the single definition. `MemoryCodegen`
plants the reference into every emitted LLVM module as a data relocation held
live by `llvm.used`, so it survives every optimisation level.

Measured, on the same link command, varying one thing:

| Object | Runtime | Result |
|---|---|---|
| compiled with `-DESHKOL_MEMORY_ABI_V2_ENABLED=1` | v1 runtime | **link refused**, naming `eshkol_object_abi_v2_h32_s6_a16` |
| same object, guard reference stripped | v1 runtime | links clean and silent — the world before this change |
| compiled without the flag | v1 runtime | links |

The middle row is the point. That was the existing behaviour, and it is what a
half-migrated toolchain would have done.

Note that the first row required no edit to produce: setting the flag the
migration will eventually set is what renames the symbol.

The mechanism closes in both directions and covers all four ways the halves can
drift apart — a stale object file, a stale cached JIT artifact, a stale
installed runtime, a `--shared-lib` artifact built by an older compiler. It
costs one pointer per module and nothing at run time.

It also closes the loop from the other side. The fingerprint numbers are bound
to the real layout by static assertions on `offsetof`, compiled by every
participant. Change the struct without changing the numbers and the build stops;
change the numbers and the symbol renames. There is no path that changes the
layout quietly.

**What the guard does not cover, stated rather than left to be discovered.** The
WASM lane resolves the runtime surface through JS-implemented imports instead of
linking the archive, so there is no second half to disagree with and an
undefined data symbol would simply fail the link; the guard is deliberately not
emitted for wasm32, and Stage 2 brings that lane under an equivalent check.
`dlopen()` consumers get no link-time check at all, which is why
`eshkol_abi_fingerprint_name()` and `eshkol_abi_runtime_header_size()` are
exported: a consumer that loads the runtime dynamically can perform at load time
the check a static link would have performed for it.

### Option 4 — one accessor, so the layout is known in one place

Necessary, and complementary rather than alternative. It reduces the number of
places that *know* the layout; it does nothing to detect two builds that know
different ones. Options 3 and 4 together are the plan: 4 shrinks the surface, 3
makes a mistake on the remaining surface loud.

### Decision

**Option 3 plus Option 4.** No in-memory discriminator. Coexistence is forbidden
rather than supported, and enforced at link time. Persisted formats carry an
explicit ABI field, because that is the only place where two layouts genuinely
have to meet.

The minimum bar this ADR sets, and the one Stage 0 has already cleared: *a
half-migrated toolchain must be impossible to link silently.* Turning a
silent-garbage failure into a loud-error failure is worth more than any amount
of staging cleverness, because a staged migration that is 96% complete is
indistinguishable, from the outside, from one that is finished.

## The sequenced plan

Every stage names its falsifier — the specific observation that would show the
stage did not do what it claims. A stage without one is not done, it is
unmeasured.

The canary throughout is the **lite lanes**. They are the cheapest full
build-and-run in CI, they cover macOS, Linux and Windows, and the WASM lite lane
is the one ADR-0000 §6 already names as the blast-radius falsifier for exactly
this change. Nothing proceeds to a broader lane on a red lite lane.

### Stage status

| Stage | Status | Completion evidence |
|---|---|---|
| Stage 0 — enforcement and inventory | COMPLETE | ABI inventory, layout pin, and native mixed-link fingerprint |
| Stage 1 — cache keys carry the ABI | COMPLETE | Run-cache and JIT stdlib cache keys include `ESHKOL_OBJECT_ABI_CACHE_TAG`; cache invalidation test |
| Stage 2 — WASM geometry guard | COMPLETE | WASM entry import checks pointer width, active object-header geometry, and tagged-value geometry; both JS glue files and the deliberate mismatch test pass |
| Stage 3 — the funnel | PROPOSED | Not started |
| Stage 4 — reconcile duplicate definitions | PROPOSED | Not started |
| Stage 5 — persisted formats declare their ABI | PROPOSED | Not started |
| Stage 6 — the flip | PROPOSED | Not started |

Rollback throughout is a flag flip: `-DESHKOL_MEMORY_ABI_V2=OFF` restores the v1
layout in one configure step, because the option is project-wide and the header
offset is a constant every translation unit computes independently.

### Stage 0 — enforcement and inventory (landed)

The machine inventory, the ratchet, the layout pin, and the link-time guard.
Nothing changes layout.

*Falsifier:* feed the ratchet a newly introduced raw header site and it stays
green; feed the guard an object built against a different layout and it links;
give the layout pin a wrong offset and it passes. All three were run, all three
went red as required, and the transcripts are in the originating PR.

Building the compiler with `-DESHKOL_MEMORY_ABI_V2=ON` and linking its output
against a v1 runtime is the migration's own worst case, and it is refused in
both directions. That exercise also found a real defect in the guard's first
form: `abi_fingerprint.h` selected its numbers from `ESHKOL_MEMORY_ABI_ACTIVE`
but did not include the header that defines it, so a translation unit that
included the fingerprint first fell through to the v1 branch and emitted a v1
guard inside a v2 build. The code generator was such a translation unit. The
header now takes that include itself and `#error`s rather than assuming, because
a correctness property that depends on include order is not a property.

### Stage 1 — cache keys carry the ABI

Thread the fingerprint into every persistent artifact cache key: the AOT
`run-<sha>` cache, the JIT stdlib object cache, and the unvalidated legacy
`stdlib.o` load path. Drives class M to zero.

*Falsifier:* build, run a program to populate the cache, bump
`ESHKOL_OBJECT_ABI_VERSION`, rebuild, run again — and observe a cache **miss**.
A hit means the key still does not distinguish the layouts, and the stage has
not landed regardless of what the diff says.

*Canary:* lite lanes plus a cold-cache and a warm-cache run.

### Stage 2 — the WASM lane and the second-implementation problem

The object header is re-implemented byte-for-byte in JavaScript, in two files,
and `scripts/check_wasm_imports.py` checks import *names* only — it says so in a
comment. Extend it to check the header geometry the JS stubs assume, and bring
the wasm32 lane under an equivalent of the guard. **COMPLETE:** every generated
wasm32 entry point calls `eshkol_wasm_abi_check` with the pointer width, active
object-header size/alignment/field offsets, and tagged-value size/alignment/
field offsets. Both JS glue files compare the complete tuple and throw before
user code runs when any value differs. The import checker validates both glue
objects and its negative test deliberately changes `objectHeaderSize` and
requires a failure.

*Falsifier:* change one offset in a JS stub and require the check to go red. It
currently would not.

*Canary:* the WASM lite lane, which is where this lands or does not.

### Stage 3 — the funnel

One accessor for the C and C++ sites; one `headerFieldOffset(field)` helper
replacing the fifty-five class D and D2 constants in the code generator. Classes
A, C, D, D2, G and K collapse into callers of two functions.

*Falsifier:* **emitted IR must be byte-identical** across the refactor for a
corpus of programs, compared with `--dump-ir` at each optimisation level. This
is a strong and cheap proof of behavioural identity, and it is available for
exactly this class of change; a refactor of the code generator that cannot
produce it has changed something it did not mean to.

*Canary:* lite lanes, then the AD and tensor suites, which exercise the densest
concentration of class D2 sites.

### Stage 4 — reconcile the duplicate definitions

`VmObjectHeader` gets a static assertion tying it to the canonical header, and
its already-divergent subtype numbering is either reconciled or made explicitly
independent with a compile-time cross-check. `eshkol_shared_header_t` is
documented as a distinct ABI and excluded from the migration by name rather than
by hope.

*Falsifier:* a test asserting `VM_SUBTYPE_STRING == HEAP_SUBTYPE_STRING` fails
today. Either the numbering is reconciled and it passes, or the independence is
deliberate and the test asserts the mapping table instead. Silence is not an
option that remains available.

### Stage 5 — persisted formats declare their ABI

The knowledge-base format writes raw subtype bytes to disk and its version field
would not catch a subtype renumbering. The `.eskb` and QLMW formats pin
in-memory geometry that is not in their headers. Each gains an explicit object-ABI
field, checked on read.

*Falsifier:* write a file under one ABI, read it under another, and require a
diagnostic rather than data.

*Canary:* the model-IO and knowledge-base round-trip tests.

### Stage 6 — the flip

`ESHKOL_MEMORY_ABI_V2=ON` becomes the default, atomically, with everything above
in place. The guard makes every stale artifact in the world announce itself.

*Falsifier:* the full suite, then the pillars, then the lite lanes on all three
platforms. Any silent numerical difference — as opposed to a loud failure — means
a site was missed, and the inventory is the list of places to look.

*Rollback:* `-DESHKOL_MEMORY_ABI_V2=OFF`.

## Consequences

The inventory is regenerable and ratcheted, so it cannot rot into a
96%-correct list: `scripts/abi_header_inventory.py check` fails on a new site in
a file the baseline does not list, and the snapshot is rebuilt by a documented
command rather than maintained by hand.

The header layout is pinned by a test that reads the fields both through the
accessor and as raw bytes at negative offsets — the two routes that generated
code and the runtime respectively take — so a divergence between them is caught
rather than averaged.

Mixed links are dead. That is the change that makes the rest of the migration a
matter of work rather than of luck: every remaining stage can be attempted, and
if it is attempted wrongly, the toolchain says so at link time instead of the
program saying nothing at all.

Class L — the layout restated in prose, including in the published
specification — is now counted. It is a documentation work-list rather than a
code hazard, and it is tracked in the same inventory so it is not rediscovered
later as a surprise.
