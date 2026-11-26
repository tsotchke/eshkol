# Specification-mutation testing for the Lean proofs

`MutationRunner.lean` answers the question "which defects can these proof obligations
see?". For each declared mutation it changes the model, recompiles, and classifies the
result. It is a single file that uses only the Lean core library.

## Run

```sh
LEAN_BIN=<lean 4.12.0> ./run_mutation_manifests.sh       # self-test and every manifest
LEAN_BIN=<lean 4.12.0> lean --run MutationRunner.lean manifests/EshkolKernelK2.mutants.json
```

The exit status is non-zero if a target does not compile clean, or if any mutant's outcome
differs from the `expect` recorded in its manifest. The manifests are therefore a regression
gate. A repair that turns `SURVIVED` into `REFUTED` is a change to record in the manifest,
together with the witness that refutes it.

## Manifest

```json
{ "target": "<path relative to the manifest>",
  "mutants": [ { "id": "...", "description": "...",
                 "old": "<text occurring exactly once>", "new": "...",
                 "witness": "<theorem proving ¬ type_of% @<refutes>, optional>",
                 "witness_name": "...", "refutes": "<theorem of the target>",
                 "equivalent": "<reason, optional>", "expect": "<outcome, optional>" } ] }
```

The target file is never written. Mutant and witness copies are written beside it and
removed after classification; set `KEEP_MUTANTS=1` to keep them.

## Outcomes

| Outcome | Meaning |
|---|---|
| `INVALID` | An error lies outside every theorem; the mutant model is ill-formed. |
| `REFUTED` | The witness proves the negation of the named theorem's statement on the mutant, with no error and no `sorryAx`. |
| `PROOF_BROKEN` | Errors lie only inside theorems. The change is detected, but no counterexample was supplied. |
| `SURVIVED` | The mutant compiles clean. No obligation detects the change. |
| `EQUIVALENT` | Declared neutral, and it compiles clean. It is `EQUIVALENT_CONTRADICTED` if a proof breaks. |
| `MANIFEST_ERROR` | `old` is not unique, or `refutes` names no theorem of the target. |

**Target binding.** After each witness, the runner appends its own line,
`example : ¬ type_of% @<refutes> := <witness_name>`. A witness counts only if that line
elaborates. This binds the counterexample to the exact statement of the theorem it claims to
contradict, not merely to its name. `#print axioms` confirms that the witness does not depend
on the (possibly broken) proof. The runner reads only the witness's own axiom report. Other
`#print axioms` output in the target is ignored, such as a report on a theorem whose proof the
mutation broke. A witness whose report is missing is not counted.

## Verification of the runner

`selftest/` has one fixture and eleven mutants, one per outcome. Four of them are decoys
that must not count as refutations:

- a false witness;
- a witness that borrows a broken proof;
- a witness naming a theorem the target does not have;
- a witness that is true but is not the negation of the theorem it names.

The fixture also prints its own axiom report for `double_even`. In `expect-refuted`, the
mutation breaks that proof, so the report names `sorryAx`, and the valid witness must still
count as a refutation. An earlier runner scanned the whole compiler output for `sorryAx`.
It classified this case `PROOF_BROKEN`.

## Manifests (Lean 4.12.0)

| Target | Refuted | Proof broken | Survived |
|---|---|---|---|
| `EshkolKernelPolicy.lean` | 6 | 0 | 2 |
| `EshkolKernelK1.lean` | 2 | 1 | 0 |
| `EshkolKernelK2.lean` (with `FramePool.Accounted`) | 4 | 0 | 0 |
| `EshkolKernelK3.lean` | 0 | 1 | 0 |
| `EshkolCompositorAligned.lean` | 10 | 0 | 2 |

- **`EshkolKernelPolicy` survivors:** writes escaping their surface, and writes under a
  foreign badge. These are recorded as the current state of the model.
- **`EshkolCompositorAligned` survivors:** a wrong rejection status code and a wrong slot
  limit. No theorem states either. The native correspondence test detects both (see
  `tests/kernel/compositor_correspondence/`).

The counts are coverage against these enumerated mutants only. They are not an adequacy
score for the specifications.
