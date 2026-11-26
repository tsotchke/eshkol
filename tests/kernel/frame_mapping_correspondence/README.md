# Frames and address spaces composed: model, proofs, native correspondence

This experiment composes two parts of tsotchke/eshkol's K2 kernel, the frame pool and the
address spaces of `lib/kernel/k2_mmu_frames.c`, into one Lean model. It proves the
composed invariants and isolation, and compares the model with native C trace by trace.
It also asks whether composing two verified models generates their shared verification
obligations systematically.

```sh
CC=cc LEAN_BIN=<lean 4.12.0> ./run_frame_mapping_correspondence.sh        # seeds 1 20261126 777
CC=cc LEAN_BIN=<lean 4.12.0> ./run_frame_mapping_negative_controls.sh     # 8 controls
LEAN_BIN=<lean 4.12.0> ../../../proofs/lean/mutation/run_mutation_manifests.sh
```

## Files

| File | Content |
|---|---|
| `proofs/lean/EshkolKernelFrameMappings.lean` | The composed model, its invariants and the proofs (core Lean 4.12.0, no `sorry`) |
| `check_axioms.lean` | Axioms of the headline theorems |
| `generate_frame_mapping_traces.c` | Random native traces with the full state, native-side invariants, and a classification of each refusal |
| `FrameMappingCorrespondence.lean` | Replays a trace on the model and compares the status and the complete state after every operation |
| `run_frame_mapping_negative_controls.sh` | Replays the traces against each manifest mutant |
| `reproduce_kernel_last_asid_refusal.c` | Directed test of one precision failure |
| `reproduce_asid_collision.c` | Directed test of the ASID collision, with a unique-ASID control |
| `run_frame_mapping_correspondence.sh` | Builds the harness, reports axioms, replays both configurations, runs the reproducer |
| `proofs/lean/mutation/manifests/EshkolKernelFrameMappings.mutants.json` | 8 mutants, with expected outcomes for both the proofs and the differential |

## Model

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

## Theorems

| Theorem | Statement |
|---|---|
| `reachable_coherent` | every reachable state is `Coherent`, with or without unique ASIDs |
| `unique_asids_single_user_asid` | `UniqueAsids ∧ Coherent → SingleUserAsid` |
| `mutable_isolation` | `UniqueAsids ∧ Coherent → MutableIsolation` |
| `reachable_mutable_isolation` | every reachable state from unique ASIDs satisfies `MutableIsolation` |
| `asid_collision_breaks_isolation` | without unique ASIDs, a reachable state violates `MutableIsolation` |
| `kernel_last_asid_refuses_read_only_user` | native refuses a read-only user mapping of a frame mapped only by the kernel |

The first four depend on `propext` and `Quot.sound` only; see `check_axioms.lean`.

## Abstraction boundaries

These parts of native are outside the model, and no result here covers them:

- page-table capacity (512 entries per space, `PT_FULL`);
- PTE encoding beyond one rule. On x86-64, a kernel space refuses the U bit with
  `PRIV_VIOLATION`, after the policy check.
- the TLB;
- virtual-address alignment and canonicality;
- the other MMU modes. The harness uses x86-64 4-level paging only.

## Results

### Correspondence

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

### Native state, checked against the actual relation

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

### Precision

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

### Mutants

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

## Finding: composition generates structural obligations, not the security policy

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

## Limits

- **Finite trace agreement, not a refinement proof.** The theorems are about the model.
- The model is single-threaded. Native's locking is not modelled.
- Mutants are hand-written. Survival and coverage are relative to this manifest.
