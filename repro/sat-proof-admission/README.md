# SAT input and proof admission

This contribution extends the existing `core.reasoning.sat` from #895; it adds
no second solver. Complete DIMACS headers, declared counts, variable bounds,
exact decimal literals and zero terminators are checked. Structured proofs
must have proper finite data and explicit `add`/`delete` tags. Source clauses
normalize repeated literals and ignore tautologies, matching solver semantics.
Proof clauses preserve their first literal (the RAT pivot) when deduplicated;
tautological proof steps refuse. Deletion removes exactly one normalized clause.

The bounded gate runs the existing SAT suite, 49 admission controls and 352
independent brute-force cases on uncached in-process JIT and AOT-O0. Separate
Python unit propagation and RAT checks validate PHP(4,3), PHP(5,4) and a proof
with two genuine non-RUP RAT additions. Frozen model/proof data is replayed
with the solver, brute-force helper and all formula producers replaced by
error stubs. It checks those stubs are active before independent replay.

```sh
python3 scripts/run_sat_proof_gate.py --eshkol-bin /opt/eshkol/bin/eshkol-run --results /tmp/sat-proof.json
python3 scripts/sat_proof_gate_self_test.py
```

The gate always forces `ESHKOL_JIT_CACHE=0`. Its readiness control intercepts
the first real child boundary for unset/enabled callers, preserves caller
state, and rejects equal but false native receipts. CMake registration is
opt-in through `ESHKOL_SAT_PROOF_ADMISSION_TESTS`.

The source-DIMACS contract follows the [DRAT format specification](https://github.com/marijnheule/drat-trim/blob/master/README.md).
This checker honors unit deletions; DRAT-trim's optional unit-deletion behavior
is not silently substituted. The RAT rule is unchanged. This is finite native
qualification with a trusted parser, arithmetic and compiler, not a
proof-assistant formalization, solver-throughput result or full-project gate.
