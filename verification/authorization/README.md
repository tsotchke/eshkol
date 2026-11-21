# Contract and authorization R0

One gate decides whether a consumer may take a particular action with proposed bytes. The consumer chooses the contract and registers independent observations. The producer supplies only the candidate. Existing Eshkol programs and compiler entry points keep their public interfaces.

| Candidate | Consumer decision | Defining observation |
|---|---|---|
| Complete LLVM function for Gram | Export the conditional omission for IR review. Keep the guarded comparison for strict lowering. | Reverse the rewrite and recover the entire pinned reference function. Changing one operand fails at the binding check. Helper effects for the ordinary path remain open. |
| Reordered 23-gate Steane preparation | Dispatch depth 8 under ideal equality. Under the biased single-fault model, a budget of 999/1,000,000 selects depth 9. A budget of 1/1,000 admits depth 8. | All 128 signed columns agree. After source gate 12, a YY fault flips the recovered logical X readout only in the candidate. Across 225 paired events the risk increase is exactly 1/1,000. |
| Regrouped P5 heat solver | Refuse prescribed arithmetic. Authorize the specified sine-rod solve with grid-point error at most 1/100,000. | One step on the tie triple gives ½ + 2⁻⁵³ where the prescribed order gives ½. A separate rational enclosure covers every native field value on the requested grid. |

## Reproduce

Use Linux x86-64, Python 3.10 or later, the official Lean 4.35.0 release and the Eshkol v1.3.6-evolve x64 lite compiler with its matching runtime, LLVM 21 libraries and native linker. Put the two executables on PATH or set LEAN and ESHKOL_RUN. Run from the Eshkol repository root.

```sh
python3 -B verification/authorization/replay.py --output authority.json
```

The replay compiles the bundled P5 source in a private directory. It submits the same ELF to both policies, checks 206 points after 3,362 steps at time 1/50, then executes the released bytes and checks the returned field against the historical P5 digest. The independent analytic enclosure uses rational bounds for the sine and exponential. Its maximum applies only at these grid points on this input. The exact-arithmetic checker can disprove an all-input claim with a single triple. Agreement on that triple leaves the claim open.

The Steane observer uses exact signed integer columns with a common Hadamard normalization. The risk observer checks syndrome and logical X recovery using independent Pauli propagation and dense replay. The biased law assigns probability 1/100 to one fault. Conditional on a fault, it gives 9/10 of its mass uniformly to the 225 events and 1/10 to gate 12 YY. Dispatch in this replay runs the selected tape on the host simulator. The Gram fixture is the complete extracted guarded function. Selection releases its bytes to the comparison consumer. The runtime and epoch mutation controls exercise the gate with supplied identities.

## Read a decision

Start with the returned status. A denial includes the first failed rule, required and observed values, and the witness location. An inconclusive decision gives the missing premise or failed check. The receipt links each exercised fallback to its independent acceptance. The budget boundary reuses the same two Steane evidence IDs across two different decisions.

The scope digest covers the typed policy, its permission class and generation, the gate implementation, the artifact digest and the union of declared checker dependencies. The classes are IR review, lowering, native release, execution and publication. The consumer owns the active contract. Replacing it advances its generation, including when restoring an earlier budget. Each evidence ID covers the measured value, level, observation kind, subject, candidate and relevant input digests. A decision links to its scope and ordered evidence IDs. IDs use SHA-256 over sorted-key UTF-8 JSON with compact separators. Byte leaves use SHA-256 directly. Replaying recomputes observations. The issuer privately retains the payload and ordered evidence IDs. A single-use ticket identifies that entry. Consumption reconstructs the binding from those IDs, the consumer's current scope and the bytes expected at its entry point. Release also compares the complete immutable payload with those expected bytes. A receipt, another gate's ticket or a spent ticket yields no artifact.

## Cross-contract controls

The receipt records each attempted transfer, the issuing decision, the requested artifact, the fresh decision on that request and the point of refusal.

| Existing permission | New request | Fresh decision |
|---|---|---|
| Q3 ideal operator | Preserve every paired fault syndrome and logical Pauli | Denied. The first mismatch in gate-ID and Pauli order is gate 12 IY. The syndrome changes from 001111 to 111111. |
| Q3 increase at most 1/1,000 | Tighten to 999/1,000,000 on the same issuer | Old ticket expires. Reassessment denies the measured increase of 1/1,000. Restoring the old budget requires another assessment. |
| P5 numerical accuracy | Exact prescribed arithmetic on the same ELF | Denied at the tie triple. |
| Gram IR review | Native release of the same proposal | Inconclusive at the missing helper effects. |
| Gram proposal with its original ticket and receipt | Substitute the guarded function under the same IR-review policy | Ticket refused. The guarded bytes obtain their own review decision. |

## Proof and trust boundary

`Verificatum.Authority.authorize_sound` proves coverage of every required typed claim at the issued scope, conditional on sound checked records and sufficient consumer policy. `consume_bound` binds released artifact and current scope. `release_sound` composes them. Lean reports `propext` and `Quot.sound` for the first and third results and no axioms for the second. The hypotheses include the actual meaning of the checked claims. [Lean model](Authority.lean)

The normal gate calls the two functions in `kernel.py`. The replay parses that file with the Python AST, translates its admitted expressions and control flow into Lean, and checks three theorems against that translation.

- `decision_refines` covers every finite list of obligations and every checker-result function. The loop agrees with the independent specification on empty input, the first failed obligation and complete success.
- `decision_authorizes` relates success to the original scoped-evidence judgment, assuming the observer returns checked exactly for bound evidence and cannot issue authority itself.
- `release_refines` covers all four byte sequences supplied to release. It establishes agreement with modeled consumption on the binding and exact artifact. It uses `propext`. The decision theorems also use `Quot.sound`.

The replay compares actual CPython results with Lean on 121 decision sequences and 81 byte-release cases. It changes the loop's stop test and replaces the release conjunction with a disjunction in separate scratch translations. Each mutant must translate successfully and fail its corresponding theorem. [Python kernel](kernel.py) · [Subset translator](extract_kernel.py) · [Refinement proofs](Refinement.lean)

The proof assumes ordinary CPython semantics for the admitted subset and faithful translation of its AST. Snapshot construction, hashing and serialization, the observer, ticket ownership and atomic lookup in the sequential host remain trusted. The host trusts the Python consumer, its checkers, input manifests, epoch provider, interpreter, native compiler, linked libraries, OS and hardware. The host must supply a coherent snapshot, advance epochs across mutations, and use the released bytes for the declared action. This experiment assumes sequential execution and a stable toolchain and loader environment during one replay. Refinement of that surrounding Python code, concurrent revocation, persistent certificate validation and production handoff remain open. A future NativeEffect or ArenaLedger adapter must bind its real dependencies, including helper implementations and allocation generations, before granting execution authority.

## Cost of adding a consumer

The receipt counts nonblank, noncomment lines in each adapter, including contract declarations, producer helpers and execution glue. The shared gate and Lean model contain no consumer imports or domain cases. Existing domain mathematics stays in the fixture checkers. This records integration size on three consumers. Person-hours and proof effort for a new mathematical checker require a prospective measurement on a fourth consumer.

The Steane preparation, decoder and matrix-derived Pauli checker carry the accompanying MIT notice. [Fixture license](fixtures/steane.LICENSE)
