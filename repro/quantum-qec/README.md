# Compile signed stabilizers to qcircuit and certify finite QEC decoders

Signed symplectic Clifford preparation into the existing qcircuit IR, finite weight-one decoder certificates, independent matrix-derived Pauli replay, and core Lean theorem emission.

Base: `022890ca5426e01b3133a600030dd1a10b32e735` (`feat/v20-quantum-types`). This is one independently reviewable cut.

## Run

```sh
python3 repro/quantum-qec/self_test.py
python3 repro/quantum-qec/gate.py --eshkol-bin /path/to/eshkol-run --lean /path/to/lean --results repro/quantum-qec/results.json
```

The gate forces `ESHKOL_JIT_CACHE=0` in every native child, even when its caller
sets it to 1. It requires complete expected output and uncached in-process JIT /
AOT-O0 equality, freezes the tested sources for the duration of the run, and
executes producer-free replay with disabled producers exercised as negative
controls. Boundary self-tests launch no native compiler or workload.

Steane, Shor and surface-d3 preparations run through the actual circuit simulator. Independent replay checks 78 identity/weight-one recoveries. A 128-qubit preparation uses 577 gates without a state vector. Lean accepts six finite theorems and rejects all three no-op decoder controls.

## Check received witnesses

Run `python3 repro/quantum-qec/oracle.py repro/quantum-qec/results.json --lean-dir /tmp/qec-lean`, then run `lean /tmp/qec-lean/code0.lean` (and code1/code2). This path uses neither Eshkol nor a decoder producer. The full gate additionally requires Lean to reject false decoder theorems.

## Limits

Finite identity/weight-one Pauli recovery modulo global phase; no threshold, full distance, fault-tolerant syndrome-extraction, hardware, or general quantum-program correctness claim. Signed circuit preparation is checked independently; the Lean theorems concern phase-free syndrome recovery. The new compiler qreg linearity rules and Moonlab native hardware lowering are not qualified by these tests.

Qualification uses the published v1.3.6-evolve Linux x64 lite compiler/runtime in
Ubuntu 22.04/gcc-11. These are native library-level qualifications against pinned
feature-branch source interfaces. No new compiler build, full upstream CI/CTest,
VM/WASM, Windows, GPU or hardware qualification is implied. The small standalone
CTest project checks this cut's registration and boundary readiness separately.
