---
kind: explanation
status: current
owner-area: types
since: v2.0-starlight
sources:
  - inc/eshkol/types/quantum_types.h
  - lib/types/quantum_types.cpp
  - lib/types/quantum_index.cpp
  - lib/types/type_checker.cpp
  - lib/types/hott_types.cpp
  - lib/frontend/parser.cpp
  - lib/core/quantum.esk
  - lib/core/quantum/circuit.esk
  - lib/core/quantum/optimize.esk
  - lib/core/quantum/route.esk
  - lib/core/quantum/register.esk
  - lib/core/quantum/qaoa.esk
  - lib/core/quantum/moonlab.esk
  - lib/agent/c/agent_quantum.c
  - tests/typesystem/qreg_corpus_gate.sh
---
# ADR-0048: Quantum register types, quantum regions and circuit compilation

**Status:** Proposed
**Date:** 2026-10-30
**Scope:** the v2.0-starlight quantum items of the roadmap: register types
`qreg<n>` with compile-time dimensions on the linear judgment that makes
`Qubit` no-cloning; `define-quantum-region` scoping of allocation and
deallocation; a circuit IR with optimisation (gate fusion, commutation-based
cancellation, phase folding, ZX-calculus resynthesis) and qubit mapping
(SABRE routing); exact unitary-equivalence checks; lowering to Moonlab; QAOA
with Eshkol AD through Moonlab's exact adjoint; and the interface the v1.9
quantitative type theory takes over.
**Depends on:** ADR-0004 (one quantitative dependent type system; "quantum
types are a specialization"), ADR-0041 (`core.monoidal.zx`), the Moonlab
integration (`docs/design/MOONLAB_INTEGRATION.md`, the custom-VJP node).

## Context

`Qubit` has been a linear type since v1.3.4: `TypeChecker::analyzeLinearUses`
rules on every binder and a violation is fatal on both engines. ADR-0004
places quantum registers in the same machinery: `QReg ρ n` is a linear
dependent family, `qalloc : Π (ρ : QRegion) (n : Nat). QReg ρ n`, a region
closes only when every qubit was measured or released, and there is no second
"quantum borrow checker". The checker that exists today has TypeIds without
indices, a gradual relation, and no dependent function types in use. Moonlab
runs circuits and differentiates parametric circuits exactly
(`moonlab_diff_*`), and `core.monoidal.zx` (ADR-0041) evaluates Clifford+T
circuits exactly over cyclotomic fields and simplifies ZX graphs.

## Decision

1. **Four types, one linear judgment.** `qreg<e>` (a register of e qubits,
   `TYPE_FLAG_LINEAR`), `qidx<e>` (the address of one qubit of a `qreg<e>`, a
   natural below e), `qregion` (the capability of one region) and
   `qregion-body` (a procedure of one `qregion` whose result is classical).
   Instances `qreg<3>`, `qreg<n+1>` are interned per canonical index
   (`TypeEnvironment::internIndexedType`, ids 58000-59999) as nominal
   subtypes of their family carrying its flags, so every rule of the linear
   analysis (no-cloning, no-discard, alias chains, branches, closures,
   containers) applies to registers unchanged. The bare family `qreg` is a
   register of a size known only at run time: still linear, compatible with
   every dimension. Dimension checking is gradual; linearity is not.

2. **Indices are linear expressions over naturals.** `IndexExpr` is
   `c + a1*v1 + ... + ak*vk` with a canonical spelling, written `qreg<n+1>`
   or `(qreg (+ n 1))`. At an application the callee's index variables are
   renamed apart and unified with the arguments' indices (one unknown per
   pattern is solved, `n+1 = 3` gives `n = 2`; a pattern with two unknowns
   waits for another argument), then substituted into the result. A variable
   spelled like an integer parameter is that argument's compile-time value:
   `(qalloc qr 3) : qreg<3>`, `(qalloc qr k) : qreg` for a run-time k.
   Inside a procedure generic in n, n is rigid: passing its `qreg<n>` where
   `qreg<3>` is required is a mismatch.

3. **Findings are fatal.** A dimension mismatch, a `qidx` literal out of
   range or repeated within one gate ("a gate cannot act on a copy of its own
   qubit"), a register that leaves its region, a region capability returned
   inside a procedure, an undeclared or mis-sized continuation parameter, and
   a linear value computed and discarded in a sequence go through
   `reportLinearViolation`: the compile stops in every mode but `--unsafe`.
   The bytecode engine runs the same checker (`linear_check_bridge`), so both
   engines refuse the same programs.

4. **Signatures keep their dependency.** `quantum::Signature` records each
   parameter's name and `SpecType` (plain, indexed, arrow, type variable). A
   definition whose annotations mention a quantum type is bound under a
   dependent function type identified by its signature key
   (`makeDependentFunctionType`), and the registry finds its signature
   through that TypeId. The core.quantum primitives are bound at checker
   construction from a table written in the annotation language, so a program
   checks the same with and without the library's source in view (JIT
   compiles modules separately; AOT merges them). `lib/core/quantum/
   register.esk` declares every primitive with exactly that signature over an
   untyped implementation, and a program's own definition of a name shadows
   the primitive.

5. **Regions are syntax.** `(define-quantum-region R (name . formals) body)`
   and `(with-quantum-region (R) body)` are rewritten by the reader into
   `define`, `lambda ((R : qregion))` and a call of
   `call-with-quantum-region`, whose parameter type `qregion-body` makes the
   checker reject a quantum result and a returned procedure that mentions R.
   `qalloc` takes the region explicitly, so allocation outside a region does
   not type. At run time a region owns one backend state and the circuit
   applied to it; every register operation consumes its record; a region
   closes only with no live register.

6. **One circuit IR.** `core.quantum.circuit`: gate lists on wires (the gate
   names and meanings of core.monoidal.zx, plus RX, RY, RZ, RZZ, CCX, CCZ and
   parameter references `(qparam k c)`), a binary64 complex state-vector
   simulator that AD differentiates through every angle, and equivalence
   three ways: exact over Q(zeta_m) through `circuit->unitary` (up to a
   global phase or exactly), a ZX certificate (`c ; d^-1` simplifies to bare
   wires), and binary64 within a tolerance. ZX rewriting is core.monoidal.zx's;
   nothing here re-implements it.

7. **Optimisation passes are each sound, and checked so.** Gate fusion,
   commutation-based cancellation (seven commutation rules, each proved in
   the source and checked exactly), phase folding over CNOT+phase regions
   (sum-over-paths parities), and ZX resynthesis of Clifford+T windows (a
   window is replaced only when its simplified graph is in frontier form and
   `qcircuit-equivalent-exact?` proves the extraction equal at lower cost).
   A region compiles its pending gates at each measurement.

8. **Qubit mapping is SABRE.** Coupling graphs with all-pair distances; the
   front-layer / extended-set heuristic with decay and reverse-traversal
   initial layouts; a shortest-path baseline. `qrouting-verify` proves a
   routed circuit equal to the logical circuit embedded through the initial
   layout and followed by the inverse of the final permutation. A region
   given a coupling graph keeps a logical-to-physical layout across
   measurements.

9. **Moonlab through the existing custom-VJP node.** A circuit-expectation
   program (gates on Moonlab's set, Pauli-sum observable, rotations bound to
   `coef * theta_k`) shares the handle space of the Stage S2 Hamiltonians, so
   `vqe-energy-primitive` evaluates it with `moonlab_diff_forward` and, under
   `gradient`, records one custom-VJP node whose backward pass is
   `moonlab_diff_backward_pauli_sum` chained through the coefficients. One
   primitive, two program kinds; no new compiler builtin. Circuits are
   lowered to {H X Y Z CNOT RX RY RZ} exactly up to a global phase. Regions
   can run on a Moonlab backend.

10. **QAOA.** MaxCut and weighted MaxCut: the cost as a Pauli sum, the
    parametric circuit (`RZZ(-gamma w)`, `RX(2 beta)`), exact expectations
    and AD gradients on the state vector, the p = 1 closed forms (Wang,
    Hadfield, Jiang, Rieffel 2018; the general Ising form), brute-force
    optima, Adam ascent; the same program on Moonlab with Moonlab's gradient.

## Interface for the quantitative type theory (v1.9)

The v1.9 elaborator replaces the use-count analysis with quantitative usage
arithmetic (multiplicities 0, 1, omega). This ADR fixes what it takes over:

- **Multiplicities.** A `qreg<e>` or `Qubit` binder has multiplicity 1; an
  index variable (n in `qreg<n>`) has multiplicity 0 (compile-time only,
  erased; it never appears in generated code); a `qregion` binder has
  multiplicity omega inside its region and must not occur in the region's
  result. `qidx<e>` values are unrestricted naturals.
- **Signatures.** `quantum::Signature` / `SpecType` are the elaborator's input
  for every quantum primitive and definition: parameter names, indexed
  families with `IndexExpr` indices, arrows (continuations), type variables.
  `Signature::key` identifies a dependent signature.
- **Index terms.** `IndexExpr` (linear over naturals, canonical spelling,
  substitution, renaming apart) is the index language; the elaborator's index
  solver replaces `Registry::checkApplication`'s one-unknown solving and adds
  the side condition `e >= 0` for `qreg<n-k>` (today a run-time check in
  `qsplit`).
- **Judgment failures.** `TypeChecker::reportLinearViolation` is the single
  fatal channel; a QTT usage error and every finding of Decision 3 report
  there, so both engines keep refusing the same programs.
- **Regions.** `qregion-body` is the region-closing rule `ρ ∉ FV(result)`;
  under QTT it becomes the region's effect row (`Quantum ρ`, ADR-0045) and
  the escape check on the result type.

## Consequences

- Registers inherit every linear rule with no second analysis; a named `let`
  or `do` over a register is reported undecidable exactly as for `Qubit`
  (loop-carried accounting is ADR-0004 work).
- A register's size known only at run time is checked at run time (index
  ranges, split points, distinct qubits), with the same messages.
- The ZX resynthesis fires on frontier-form windows; general circuit
  extraction (Backens et al. 2021) is Planned for v1.4.1. Phase folding does
  not fold RZZ parities.
- The Moonlab backend creates its state at a region's first measurement; a
  later `qalloc` is refused (a growing Moonlab state is Planned for v1.4.1).
- The VM runs the checker and the library; the custom-VJP node is
  native-only, as for `vqe-energy-primitive` (PARITY.tsv).

## Alternatives considered

- **A new hott kind for indexed families.** The spelled form `qreg<e>` is a
  symbol to the reader, and the checker interns its instances; a structured
  type-application kind belongs to the type-extension work and would replace
  the spelling without changing the TypeIds.
- **A separate quantum borrow checker.** ADR-0004 rules it out; the linear
  analysis already decides cloning and discarding.
- **Measuring part of a register into a pair.** A pair holding a register
  would launder linearity through an untyped container; `qsplit` passes the
  parts to a continuation whose parameters are declared linear.
- **A new codegen builtin for circuit expectations.** The VQE custom-VJP node
  already carries an opaque Moonlab VJP; giving programs its handle space
  keeps one primitive.

## Verification

- `tests/typesystem/qreg_*.esk` through `tests/typesystem/qreg_corpus_gate.sh`
  (CTest `quantum_types_corpus`): 14 refusals and 3 accepted programs, native
  and bytecode engines.
- `tests/stdlib/quantum_circuit_test.esk`, `quantum_optimize_test.esk`,
  `quantum_route_test.esk`, `qaoa_test.esk`, `quantum_register_test.esk`
  (CTest `*_jit`, `*_aot`).
- `tests/quantum/qaoa_moonlab_crosscheck_test.esk` (CTest
  `qaoa_moonlab_crosscheck`, quantum builds).
- `scripts/run_quantum_types_gate.sh`, criterion `v20_quantum_types`.

## References

ADR-0004; ADR-0041; E. Farhi, J. Goldstone, S. Gutmann, arXiv:1411.4028
(2014); Z. Wang, S. Hadfield, Z. Jiang, E. Rieffel, PRA 97, 022304 (2018);
G. Li, Y. Ding, Y. Xie, ASPLOS 2019 (SABRE); M. Amy, D. Maslov, M. Mosca,
IEEE TCAD 33 (2014) (phase folding); R. Duncan, A. Kissinger, S. Perdrix,
J. van de Wetering, Quantum 4 (2020); M. Backens et al., Quantum 5 (2021);
R. Atkey, "Syntax and semantics of quantitative type theory" (LICS 2018);
P. Selinger, "Towards a quantum programming language" (2004); nLab pages
"quantum circuit", "ZX-calculus", "linear type theory".
