# Frames and address spaces composed: model, proofs, native correspondence

This experiment composes two parts of tsotchke/eshkol's K2 kernel, the frame pool and the
address spaces of `lib/kernel/k2_mmu_frames.c`, into one Lean model. It proves the
composed invariants and isolation, and compares the model with native C trace by trace.

The model follows native as of commit `0c047a37b`. That commit:

- partitions kernel and user ASIDs at initialization;
- records, per frame, the user address space that maps it;
- replaces the permission union with exact read, write and execute counts.

Results for the earlier native code are kept below under
[Historical results](#historical-results-before-0c047a37b). The files that produced them
are preserved at the tag `historical/frame-mapping-composition-before-asid-partition`.

```sh
CC=cc LEAN_BIN=<lean 4.12.0> ./run_frame_mapping_correspondence.sh        # seeds 1 20261126 777
CC=cc LEAN_BIN=<lean 4.12.0> ./run_frame_mapping_negative_controls.sh     # 10 controls
LEAN_BIN=<lean 4.12.0> ../../../proofs/lean/mutation/run_mutation_manifests.sh
```

## Files

| File | Content |
|---|---|
| `proofs/lean/EshkolKernelFrameMappings.lean` | The composed model, its invariants and the proofs (core Lean 4.12.0, no `sorry`) |
| `check_axioms.lean` | Axioms of the headline theorems |
| `generate_frame_mapping_traces.c` | Initialization probes and random native traces, with the full state, native-side invariants, and a classification of each refusal |
| `FrameMappingCorrespondence.lean` | Replays a trace on the model; compares every probe, and the status and complete state after every operation |
| `run_frame_mapping_negative_controls.sh` | Replays the traces against each manifest mutant |
| `check_asid_collision_refused.c` | Directed check: the earlier ASID collision is refused at initialization and at map |
| `check_kernel_mapping_admits_user.c` | Directed check: the earlier kernel-last-ASID refusal no longer occurs |
| `run_frame_mapping_correspondence.sh` | Builds the harness, reports axioms, replays both configurations, runs the directed checks |
| `proofs/lean/mutation/manifests/EshkolKernelFrameMappings.mutants.json` | 10 mutants, with expected outcomes for both the proofs and the differential |

## Model

**Actual relation.** A global list of entries (space, page, frame, permissions), plus a
ghost generation recorded at mapping time. Native does not store the generation.

**Recorded summaries.** Native's per-frame metadata is modelled as recorded fields,
separate from the relation:

| Field | Native | Meaning |
|---|---|---|
| `refcount` | `map_refcount` | entries of the frame |
| `userRefcount` | `user_map_refcount` | entries of the frame in user spaces |
| `reads`, `writes`, `execs` | `read_refcount`, `write_refcount`, `exec_refcount` | entries with R, W, X |
| `perms` | `mapped_perms` | R, W, X recomputed from the three counts |
| `ownerAsid`, `ownerSpace` | `mapped_asid`, `mapped_user_aspace` | the user space mapping the frame, while `userRefcount` is positive |

`ownerSpace` is the index of an address space. Native compares a pointer; the harness
prints that pointer as the index of the space it points to.

**Statuses.** Every operation returns native's exact status, with native's checks in
native's order. The operations are initialize a space, allocate (user or kernel), free,
map, remap and unmap.

**Invariants.** `Coherent` is the conjunction of fourteen separately named invariants:

| Invariant | Relation |
|---|---|
| `RefcountExact` | `refcount` equals the number of entries of the frame |
| `UserRefcountExact` | `userRefcount` equals the number of user-space entries |
| `ReadCountExact`, `WriteCountExact`, `ExecCountExact` | each count equals the number of entries with that bit |
| `PermsExact` | `perms` is recomputed from the counts |
| `OwnerExact` | no owner is recorded when `userRefcount` is 0. Otherwise the owner is a user space, recorded with that space's ASID |
| `UserEntriesOwned` | every user entry lies in its frame's recorded owner space |
| `ReferentialIntegrity` | every entry names an allocated frame within the pool |
| `KernelFramesNotInUser` | no user space maps a kernel frame |
| `EntriesSatisfyPolicy` | every entry passes the permission policy and the x86-64 encoding check |
| `NoWriteExecuteAlias` | no frame is mapped writable in one entry and executable in another |
| `GenerationCurrent` | every entry's generation equals its frame's current generation (temporal safety) |
| `PageUnique` | at most one entry per (space, page) |

**Guarantee, with no assumption about ASIDs.** `SingleUserSpace`: all user entries of a
frame lie in one address space. It follows from `UserEntriesOwned` alone. Two
consequences are proved:

- `MutableIsolation`: no frame is mapped in two user ASIDs when either mapping is writable.
- `SpaceIsolation`: two distinct user spaces never share a frame, even when their ASIDs
  are equal.

The earlier model needed the environmental assumption `UniqueAsids` for isolation. The
present one does not, because native now keys ownership on the space, not on the ASID.

## Theorems

| Theorem | Statement |
|---|---|
| `reachable_coherent` | every state reachable from initialization is `Coherent`, for any list of spaces |
| `coherent_single_user_space` | `Coherent → SingleUserSpace` |
| `reachable_mutable_isolation` | every reachable state satisfies `MutableIsolation` |
| `reachable_space_isolation` | every reachable state satisfies `SpaceIsolation` |
| `owner_check_asid_redundant` | on a coherent state, native's ownership check refuses exactly when the recorded owner space differs. Its ASID comparison never decides alone |
| `init_refuses_kernel_space_with_user_asid` | initialization refuses a kernel space with ASID 2, the earlier collision configuration |
| `init_refuses_user_space_with_kernel_asid` | initialization refuses a user space with ASID 0 |
| `init_accepts_shared_user_asid` | initialization does not prevent two user spaces from sharing an ASID |

The isolation theorems hold for any list of spaces, including lists that initialization
would refuse. The ASID partition is therefore not needed for them. What the partition
protects lies outside the model; see the abstraction boundaries.

`check_axioms.lean` reports the axioms:

- the three reachability theorems use `propext`, `Classical.choice` and `Quot.sound`;
- `owner_check_asid_redundant` and the initialization theorems use none.

The repository's audit, `scripts/check_os_proof_axioms.lean`, admits exactly these three
axioms.

## Abstraction boundaries

These parts of native are outside the model, and no result here covers them:

- page-table capacity (512 entries per space, `PT_FULL`);
- PTE encoding beyond one rule. On x86-64, a kernel space refuses the U bit with
  `PRIV_VIOLATION`, after the policy check.
- the TLB. Native's software TLB lives in each space, and its entries carry the space's
  ASID. A hardware TLB is shared by all spaces and tagged by ASID. Whether two user spaces
  may share an ASID is therefore a hardware TLB question, which this model does not
  address. The kernel/user partition matters there too.
- virtual-address alignment and canonicality, and the other MMU modes. The harness uses
  x86-64 4-level paging only.
- `mode` and other arguments of `k2_aspace_init`. The model covers its ASID partition only.

## Results

### Correspondence

The harness uses three spaces: space 0 is the kernel, spaces 1 and 2 are users. There are
two configurations, and initialization admits both:

- **distinct:** ASIDs 0, 1 and 2.
- **shared:** ASIDs 0, 2 and 2. The two user spaces share an ASID.

Each configuration ran 3 seeds × 2,000 traces × 80 operations, 960,000 operations in all.
The model replayed every trace with full agreement: every operation had the same status
and the same complete state.

Each trace file also begins with 12 initialization probes:

- both kinds of space;
- ASIDs 0, 1, 2, 7, `K2_INVALID_INDEX − 1` and `K2_INVALID_INDEX`.

Native and `Space.initStatus` agree on all 72 probes, 12 in each of the six files.

### Native state, checked against the actual relation

All thirteen native-side invariants hold in all 960,000 states of both configurations:

- the exact counts, the recomputed permissions and the owner record;
- user entries owned, referential integrity, kernel frames not in user spaces;
- no write/execute alias, single user space, mutable isolation.

**Native's behaviour does not depend on how user ASIDs are assigned.** For seed 1, the two
configurations give identical statuses, invariant bits and refusal classes. Their full
states are identical apart from the recorded owner ASID. Ownership is decided by the
space, so equal ASIDs change nothing the model observes.

### Precision

Every `PERM_ALIAS` refusal is classified against the actual mappings. The two
configurations give the same counts:

| Class | Count | Meaning |
|---|---|---|
| genuine: write/execute | 12,101 | a real W/X alias would result |
| cross-space, writable | 2,638 | a user mapping shared with another user space, one of them writable |
| cross-space, read-only | 3,493 | read-only sharing with another user space; refused by design |
| unexplained | 0 | |

Both precision classes of the earlier code are gone:

- **The kernel last ASID** no longer blocks user mappings. `check_kernel_mapping_admits_user.c`
  checks this directly.
- **The stale permission union** was replaced by exact counts. Remap now excludes the
  remapped entry's own bits before checking the other mappings.

Read-only sharing between user spaces is refused, as the authors state they intend.

### Directed checks

`check_asid_collision_refused.c` takes the earlier reproducer apart:

| Step | Before `0c047a37b` | Now |
|---|---|---|
| kernel space initialized with ASID 2 | accepted | `INVALID_ARG` |
| A (ASID 1) maps F read-only; kernel maps F; B (ASID 2) maps F writable | `OK`: isolation broken | `PERM_ALIAS` |
| A and B share ASID 2; A maps F read-only; B maps F writable or read-only | not tested | `PERM_ALIAS`, both |

### Mutants

| Mutant | Proofs | Differential |
|---|---|---|
| `map-skips-owner-check` | REFUTED (`reachable_space_isolation`) | diverges |
| `map-compares-owner-asid-only` | REFUTED (`reachable_space_isolation`), in the shared-ASID configuration | diverges |
| `map-compares-owner-space-only` | PROOF_BROKEN, as declared | agrees |
| `map-skips-write-execute-alias-check` | REFUTED (`reachable_coherent`) | diverges |
| `remap-skips-write-execute-alias-check` | REFUTED | diverges |
| `map-skips-kernel-alias-check` | REFUTED | diverges |
| `free-skips-still-mapped-check` | REFUTED | diverges |
| `unmap-keeps-refcount` | REFUTED | diverges |
| `unmap-keeps-owner` | REFUTED | diverges |
| `map-records-no-owner` | REFUTED | diverges |

**`map-compares-owner-asid-only`** is the ASID-keyed ownership rule. Its witness lets two
user spaces with the same ASID share a frame writably.

**`map-compares-owner-space-only`** drops the ASID comparison. It cannot change behaviour,
for two independent reasons:

- `owner_check_asid_redundant` proves that it changes no verdict on a coherent state;
- the differential agrees on all traces of seed 1.

It is classified PROOF_BROKEN only because the preservation proofs quote the exact text of
the check.

The native ASID comparison is not proposed for removal. The theorem covers coherent states
of the model only, and the comparison still guards states outside them, such as corrupted
metadata.

## Obligation generation, revisited

The [historical finding](#finding-composition-generates-structural-obligations-not-the-security-policy)
asked whether composing two verified models generates their shared obligations. Then, two
structural rules generated six of ten invariants. The security property needed an
invariant found by a failed proof, and an environmental assumption.

The same two rules applied to the present model:

| Structural relationship | Invariants |
|---|---|
| A field summarising another structure's state | the five count invariants, `PermsExact`, `OwnerExact`, `UserEntriesOwned` |
| A reference into another structure | `ReferentialIntegrity`, `GenerationCurrent`, `KernelFramesNotInUser` |
| Not generated | `PageUnique`, `EntriesSatisfyPolicy` (local to address spaces), `NoWriteExecuteAlias` (the intended policy) |

Eleven of fourteen are generated, and the isolation guarantee follows from one of them,
`UserEntriesOwned`, with no further invariant and no assumption.

The rule generates `UserEntriesOwned` only once the summarised query is named, here the
set of spaces of a frame's user entries. Choosing that query is the security decision. It
was made in native's design by `0c047a37b`, not by the rule.

**Interpretation.** A structural rule yields the security property when the system records
a summary of exactly the quantity the policy constrains. The earlier last-ASID summary did
not record that quantity, and structure alone could not reach the policy.

**Scope.** This is one composition in two versions, and the rules remain candidates.

## Limits

- **Finite trace agreement, not a refinement proof.** The theorems are about the model.
- The model is single-threaded. Native's locking is not modelled.
- Mutants are hand-written. Survival and coverage are relative to this manifest.
- The negative controls replay seed 1 only.

## Historical results (before 0c047a37b)

These results are for native code before `0c047a37b`. They were produced by the model,
harness and manifest at the tag `historical/frame-mapping-composition-before-asid-partition`.
They are kept as recorded evidence and were not rerun against the current code. In
particular, the ASID collision and both precision classes below describe the earlier code
only.

### Model

**Actual relation.** A global list of entries (space, page, frame, permissions), plus a
ghost generation recorded at mapping time.

**Recorded summaries.** Native's per-frame summaries are separate fields:

- `refcount`
- `permUnion`
- `lastAsid`

**Statuses.** Every operation returns native's exact status, with native's checks in
native's order. The operations are allocate (user or kernel), free, map, remap and unmap.

**Invariants.** `Coherent` is the conjunction of ten separately named invariants:

| Invariant | Relation |
|---|---|
| `RefcountExact` | `refcount` equals the number of entries for the frame |
| `PermUnionCovers` | `permUnion` covers the W and X bits of every entry (an overapproximation) |
| `LastAsidSound` | for each user entry, `lastAsid` is that entry's ASID or some kernel space's ASID |
| `ReferentialIntegrity` | every entry names an allocated frame within the pool |
| `GenerationCurrent` | every entry's generation equals its frame's current generation (temporal safety) |
| `KernelFramesNotInUser` | no user space maps a kernel frame |
| `PageUnique` | at most one entry per (space, page) |
| `EntriesSatisfyPolicy` | every entry passes the permission policy and the x86-64 encoding check |
| `NoWriteExecuteAlias` | no frame is mapped writable in one entry and executable in another |
| `UserAsidsAgree` | two user entries of a frame share an ASID, or a kernel space shares one of their ASIDs |

**Assumption and guarantee.**

- **Assumption:** `UniqueAsids`, an environmental assumption that native does not enforce.
- **Guarantee:** `MutableIsolation`. No frame is mapped in two user ASIDs when either
  mapping is writable. Read-only sharing is permitted.
- **Native's stricter rule:** `SingleUserAsid` refuses read-only sharing as well. It is
  reported as a precision property of native, not as the guarantee.

### Theorems

| Theorem | Statement |
|---|---|
| `reachable_coherent` | every reachable state is `Coherent`, with or without unique ASIDs |
| `unique_asids_single_user_asid` | `UniqueAsids ∧ Coherent → SingleUserAsid` |
| `mutable_isolation` | `UniqueAsids ∧ Coherent → MutableIsolation` |
| `reachable_mutable_isolation` | every reachable state from unique ASIDs satisfies `MutableIsolation` |
| `asid_collision_breaks_isolation` | without unique ASIDs, a reachable state violates `MutableIsolation` |
| `kernel_last_asid_refuses_read_only_user` | native refuses a read-only user mapping of a frame mapped only by the kernel |

The first four depend on `propext` and `Quot.sound` only; see `check_axioms.lean`.

### Abstraction boundaries

These parts of native are outside the model, and no result here covers them:

- page-table capacity (512 entries per space, `PT_FULL`);
- PTE encoding beyond one rule. On x86-64, a kernel space refuses the U bit with
  `PRIV_VIOLATION`, after the policy check.
- the TLB;
- virtual-address alignment and canonicality;
- the other MMU modes. The harness uses x86-64 4-level paging only.

### Results

#### Correspondence

The harness uses three spaces: space 0 is the kernel, spaces 1 and 2 are users. There are
two configurations:

- **unique:** ASIDs 0, 1 and 2.
- **collide:** the kernel space has ASID 2, the same as user space 2.

Each configuration ran 3 seeds × 2,000 traces × 80 operations. The model replayed every
trace with full agreement: 960,000 operations in all, every one with the same status and
the same complete state.

Statuses exercised:

- **map:** all seven refusals plus `OK`.
- **remap:** `VA_NOT_MAPPED`, `PRIV_VIOLATION`, `WX_VIOLATION`, `INVALID_ARG`, `PERM_ALIAS`
  and `OK`.
- **free:** every status except `INVALID_ARG`. The harness never offers an out-of-range
  index.
- **unmap:** both of its statuses.

#### Native state, checked against the actual relation

| | unique | collide |
|---|---|---|
| States checked | 480,000 | 480,000 |
| `MutableIsolation` violated | 0 | 4,644 |
| `SingleUserAsid` violated | 0 | 10,338 |
| All other invariants | hold | hold |

**The cross-ASID hypothesis is reproduced.** If a kernel space shares an ASID with a user
space, native lets that user space map writable a frame that another user space maps.
Shortest native witness, 5 operations; each map line gives the space, page and permissions:

```
alloc_user                      => frame 0
map  space 1 page 2  R|U        => OK   last_asid 1
map  space 0 page 0  R|K        => OK   last_asid 2   (the kernel space's ASID is 2)
map  space 0 page 4  R          => OK
map  space 2 page 5  R|W|U      => OK   user 1 read-only and user 2 writable share frame 0
```

`asid_collision_breaks_isolation` proves the same in the model, so `UniqueAsids` is
necessary, not merely convenient.

**Status in the current tree: a conditional failure, not a demonstrated vulnerability.**

- **The booted system satisfies the assumption.** The only caller of `k2_aspace_init` is
  `k_microkernel_converge_boot` (`inc/eshkol/kernel/k_kernel_policy.h`). It creates
  `K_POL_MAX_SERVICES` user spaces with ASIDs `i + 1`, which are distinct, and no kernel
  space. So the collision is unreachable there.
- **Nothing enforces it.** `k2_aspace_init` accepts any ASID and keeps no registry, so the
  obligation rests on every future caller.

#### Precision

Every `PERM_ALIAS` refusal is classified against the actual mappings. In the unique
configuration there are no safety failures. The refusals divide as follows:

| Class | Count | Meaning |
|---|---|---|
| genuine: write/execute | 11,215 | a real W/X alias would result |
| genuine: isolation | 2,213 | a writable cross-ASID user mapping would result |
| precision: kernel last ASID | 6,036 | the kernel mapped the frame last, so every user map is refused |
| precision: read-only sharing | 2,864 | native's `SingleUserAsid` rule refuses read-only sharing |
| precision: stale permission union | 947 | the union keeps W or X from a mapping already removed |

`reproduce_kernel_last_asid_refusal.c` reproduces the first precision class in three
operations. `kernel_last_asid_refuses_read_only_user` states it in the model.

**A stale user last ASID was not observed**, in either configuration. Under unique ASIDs it
appears unreachable: every successful map overwrites `lastAsid`, and a user map with a
different ASID is refused, so no mapping in another ASID can outlive it. This argument is
not proved.

#### Mutants

| Mutant | Proofs | Differential |
|---|---|---|
| `map-skips-last-asid-check` | REFUTED (`reachable_mutable_isolation`) | diverges |
| `map-skips-write-execute-alias-check` | REFUTED (`reachable_coherent`) | diverges |
| `remap-skips-write-execute-alias-check` | REFUTED | diverges |
| `map-skips-kernel-alias-check` | REFUTED | diverges |
| `free-skips-still-mapped-check` | REFUTED | diverges |
| `unmap-keeps-refcount` | REFUTED | diverges |
| `map-records-no-last-asid` | REFUTED | diverges |
| `unmap-keeps-last-asid` (precision) | SURVIVED, as expected | diverges |

The precision mutant leaves safety intact, so no safety proof can see it. Only the
differential does.

### Finding: composition generates structural obligations, not the security policy

**Question.** When two verified models are composed, are their shared verification
obligations generated systematically?

**Answer: partly.** Two rules, applied to the composition, generate six of the ten
invariants in `Coherent`. They do not generate the other four, including both that carry
the security property.

**Generated (6).**

| Structural relationship | Obligations | Invariants |
|---|---|---|
| A field summarising another structure's state | exactness, coverage or soundness against the actual query | `RefcountExact`, `PermUnionCovers`, `LastAsidSound` |
| A reference into another structure | existence, currency, type compatibility | `ReferentialIntegrity`, `GenerationCurrent`, `KernelFramesNotInUser` |

**Not generated (4).**

| Invariant | Origin |
|---|---|
| `PageUnique` | local to address spaces; holds of either model alone |
| `EntriesSatisfyPolicy` | local to address spaces; holds of either model alone |
| `NoWriteExecuteAlias` | the intended policy across mappings |
| `UserAsidsAgree` | a failed proof of `MutableIsolation` |

**How `UserAsidsAgree` was found.**

1. **Insufficient premise.** The proof of isolation from the six generated invariants
   failed: once a kernel space's ASID is recorded last, `LastAsidSound` says nothing about
   two user entries.
2. **Candidate invariant.** The missing premise was stated as a new invariant.
3. **Independent establishment.** The new invariant was proved preserved by every
   operation and true initially. It is not an assumption added to make the isolation proof
   succeed.

**The environmental assumption.** The composition also made `UniqueAsids` explicit, a
hypothesis native's checks leave implicit. `asid_collision_breaks_isolation` proves it
necessary, and native reproduces the failure when it does not hold.

**Interpretation.** Structural coherence of a composition does not imply the purpose of the
composition. The generated layer stops at structure. The security policy, and the
intermediate invariants it needs, came from stating the guarantee and attempting its
derivation.

**Scope.** One composition. The two rules have not been tested on a second, so they are
candidates for a generator, not a method.

### Limits

- **Finite trace agreement, not a refinement proof.** The theorems are about the model.
- The model is single-threaded. Native's locking is not modelled.
- Mutants are hand-written. Survival and coverage are relative to this manifest.
