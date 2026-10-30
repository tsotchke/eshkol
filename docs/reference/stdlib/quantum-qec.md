# Compile signed stabilizers to qcircuit and certify finite QEC decoders

Signed symplectic Clifford preparation into the existing qcircuit IR, finite weight-one decoder certificates, independent matrix-derived Pauli replay, and core Lean theorem emission.

```scheme
(require core.quantum.qec)
(define code (steane-qec))
(define circuit (stabilizer-prepare 7 (qec-code-checks code)))
(verify-stabilizer-preparation 7 (qec-code-checks code) circuit)
(verify-qec-certificate (qec-certificate code) 7 (qec-code-checks code))
```

Generators are optional-sign `IXYZ` strings. Wire zero follows the existing
most-significant tensor-factor convention. Anticommutation, non-Hermitian
input, and a derived `-I` are refused. The preparation checker conjugates each
original signed generator backwards through the received Clifford circuit and
requires a positive diagonal Pauli on `|0...0>`.

A decoder certificate is `#(qec-decoder-v1 n generators table)`, where each table
row is `(syndrome-bits recovery-string)`. Replay binds to the requested code and
generates identity plus every single-qubit X/Y/Z error itself. Recovery times
error must lie in the stabilizer span, modulo phase. Literal identity would be
the wrong condition for degenerate Shor/surface recoveries.

`make-qec-code` constructs a first-wins weight-one syndrome table. Construction
does not assert correctability: only successful replay admits that finite claim.

## Scope

Finite identity/weight-one Pauli recovery modulo global phase; no threshold, full distance, fault-tolerant syndrome-extraction, hardware, or general quantum-program correctness claim. Signed circuit preparation is checked independently; the Lean theorems concern phase-free syndrome recovery. The new compiler qreg linearity rules and Moonlab native hardware lowering are not qualified by these tests.

## Independent replay

Run `python3 repro/quantum-qec/oracle.py repro/quantum-qec/results.json --lean-dir /tmp/qec-lean`, then run `lean /tmp/qec-lean/code0.lean` (and code1/code2). This path uses neither Eshkol nor a decoder producer. The full gate additionally requires Lean to reject false decoder theorems.

Selected MIT contribution; see `repro/quantum-qec/LICENSE`.
