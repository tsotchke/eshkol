---
kind: explanation
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/monoidal.esk
  - lib/core/monoidal/scalars.esk
  - lib/core/monoidal/finvect.esk
  - lib/core/monoidal/network.esk
  - lib/core/monoidal/structure.esk
  - lib/core/monoidal/vect.esk
  - lib/core/monoidal/diagram.esk
  - lib/core/monoidal/zx.esk
  - tests/stdlib/monoidal_test.esk
  - tests/stdlib/zx_test.esk
  - tests/quantum/zx_moonlab_crosscheck_test.esk
---
# ADR-0041: `core.monoidal` — monoidal categories, string diagrams and the ZX-calculus

**Status:** Proposed
**Date:** 2026-10-29
**Scope:** the library that computes with monoidal, braided, symmetric,
compact closed and dagger structure and monoidal functors; FinVect over exact
carriers; tensor networks; string diagrams with evaluation and a normal form;
and the ZX-calculus with its rewrite rules, simplification and Clifford+T
circuits; its boundary with `core.category` (ADR-0036), `core.cyclotomic`
and Moonlab (`agent.quantum`).
**Depends on:** ADR-0036 (`core.category`), ADR-0033 (`core.sheaf`: the
finite-category representation and the Z and Q coefficient rings).

## Context

`core.category` computes with finite categories, functors and (co)limits but
had no tensor product. Monoidal structure is where category theory meets
linear algebra and quantum computation: coherence (Mac Lane's pentagon and
triangle, Joyal-Street's hexagons), string diagrams as the calculus of
monoidal categories (Joyal-Street, Selinger), and the ZX-calculus as a
complete graphical language for qubit maps (Coecke-Duncan, van de
Wetering). Nothing in `lib/` or `examples/` offered a monoidal category, a
string diagram, a tensor network or a ZX diagram; `agent.quantum` reaches
Moonlab's floating-point state-vector simulator, and `core.cyclotomic` gives
exact Q(zeta_n), the field in which every Clifford+T amplitude lies.

## Decision

1. **Structures are records of procedures over a category; laws are
   checks.** A monoidal category is
   `#(monoidal-category ops tensor tensor-arrows unit associator left-unitor right-unitor braiding name)`
   over category operations `#(category-ops objects arrows source target compose identity equal? inverse)`.
   The same checks (pentagon, triangle, naturality, invertibility, the two
   hexagons, symmetry, snake equations, dagger laws, monoidal-functor
   associativity and unit squares, the braiding square) serve every
   instance and return witnesses `(law object ...)`.

2. **Finite categories are core.category's.** `finite-category-ops` reads a
   core.sheaf.site category; `finite-monoidal-category` takes the tensor as a
   core.category functor on `category-product C C`, so its functoriality is
   also `functor-laws?`. Instances: discrete monoidal categories of finite
   monoids, meet-semilattices (cartesian), and the skeletal categories of a
   finite group with a 3-cocycle omega and optional braiding c, where the
   pentagon is the cocycle condition and the hexagons the abelian cocycle
   condition. No second category type exists.

3. **FinVect over an exact carrier.** Objects are lists of dimensions with
   tensor `append` (strict); a linear map is `#(linear-map S dom cod rows)`.
   The carriers are core.sheaf.module's Z and Q (exact Scheme numbers, rows a
   core.integer_linalg matrix) and core.cyclotomic's Q(zeta_n), behind one
   descriptor `#(exact-scalars ...)` (core.monoidal.scalars). `core.ring`
   carriers (ADR-0034) are not on this library's base; the descriptor offers
   exactly the operations of a ring carrier, and moving onto core.ring is
   Planned for v1.4.1. The compact structure is the self-dual one of the
   standard basis and the dagger the conjugate transpose, so FinVect over
   Q(zeta_n) is dagger compact. A bracketed FinVect (objects binary trees)
   gives a monoidal category that is not strict, with the flattening functor
   as Mac Lane's strictification.

4. **One contraction engine.** Tensors carry labelled legs;
   `network-contract` contracts every repeated label exactly (greedy order,
   zero entries skipped). String diagrams and ZX graphs both evaluate through
   it.

5. **String diagrams: two evaluations and a normal form.** A diagram is a
   typed term (boxes, identities, composition, tensor, swaps, cups, caps).
   It evaluates compositionally in FinVect and, independently, as the
   tensor network of its open graph. Its normal form is the open graph with
   boxes numbered canonically (breadth first from the boundary; closed
   components from the root of least encoding), which decides equality in
   the free symmetric monoidal category with self-dual compact structure
   (graph isomorphism fixing the boundary, Joyal-Street and Kelly-Laplaza).

6. **ZX graphs as data, rules with their scalars.** A ZX graph has Z, X, H
   and boundary nodes and plain or Hadamard edges. Its matrix is exact over
   Q(zeta_n) (Q(zeta_8) for multiples of pi/4). Every rule returns the
   rewritten graph and the exact scalar `sqrt(2)^a e^{i pi p} q` with
   [[g]] = s [[g']]: fusion, identity, self-loops, colour change, H boxes,
   pi-copy, state copy, bialgebra, Hopf, Euler decomposition, Hadamard Hopf
   and self-loops, local complementation and pivoting. The scalars of local
   complementation and pivoting depend on how many Hadamard edges are added
   and removed (each carries 1/sqrt 2), and are stated that way. A rule
   applied outside its pattern is refused, never approximated.

7. **Circuits two ways.** A Clifford+T circuit becomes a ZX graph with its
   scalar, and is also run gate by gate on exact state vectors; the suites
   compare the two, before and after simplification.

8. **Moonlab through Pauli expectations.** `agent.quantum` exposes Moonlab's
   state through gates and <Z_q>; Moonlab works in binary64. The cross-check
   compares, for each circuit, exact Pauli expectations of the ZX state with
   Moonlab's (basis rotation, CNOT parity, <Z>), each within 1e-10, and for
   up to three qubits all 4^n strings (state tomography, fidelity 1 within
   1e-9). It runs in builds with `-DESHKOL_QUANTUM_ENABLED=ON`; the gate
   records it as its own event and as a skip with the reason elsewhere.

## Consequences

- Exhaustive checks are exact: 1142 rule instances over the phases k pi/4,
  the 256 candidate associators on Z/2 (8 pass the pentagon, the |Z^3| that
  core.category.linear's group cohomology predicts), 1024 monoidal-functor
  candidates, and the abelian 3-cocycles on (Z/2, Z/4) (4, the quadratic
  forms).
- Checks over FinVect are over chosen objects and arrows; their cost grows
  with the cube of the dimensions of the tensor products of the chosen
  objects, so the suites use dimensions up to 3.
- Simplification removes interior Clifford spiders by local complementation
  and pivoting; the boundary variants of pivoting, phase gadgets and circuit
  extraction from simplified graphs are Planned for v1.4.1.
- Comparing amplitudes with Moonlab directly (rather than through Pauli
  expectations) needs an amplitude accessor in agent.quantum and is Planned
  for v1.4.1.
- The library is outside the `(require stdlib)` closure; the VM prelude and
  its generated tables are unchanged.

## Alternatives considered

- **Monoidal categories only on finite categories.** FinVect is the
  instance string diagrams and the ZX-calculus need; a record of procedures
  over category operations serves both without a second notion of category.
- **Floating-point evaluation.** Clifford+T amplitudes lie in Q(zeta_8);
  exact evaluation makes rule soundness an equality, not a tolerance, and
  leaves the tolerance to the comparison with Moonlab where it belongs.
- **Equality of diagrams by evaluation only.** Evaluation equality is
  semantic and depends on the interpretation; the graph normal form decides
  the syntactic equality of the free structure, and the suites check that
  it implies equal values.
- **A dense rule set without scalars.** ZX up to scalars hides the
  normalisation of Hadamard edges; stating each scalar keeps every rewrite an
  exact equation, and the simplifier's result times its scalar is the
  circuit.

## Verification

- `tests/stdlib/monoidal_test.esk` (CTest `monoidal_test_jit`,
  `monoidal_test_aot`, 34 checks).
- `tests/stdlib/zx_test.esk` (CTest `zx_test_jit`, `zx_test_aot`, 13 checks:
  1142 of 1142 rule instances sound; 30 random Clifford+T circuits against
  the state-vector semantics before and after simplification).
- `tests/quantum/zx_moonlab_crosscheck_test.esk` (CTest
  `zx_moonlab_crosscheck`, quantum builds).
- `scripts/run_monoidal_gate.sh`, criterion `v14_monoidal_zx`, with the
  reference pages `monoidal.md` and `zx.md` through the documentation
  example harness.

## References

S. Mac Lane, *Categories for the Working Mathematician*, 2nd ed. (1998),
VII and XI; A. Joyal, R. Street, "The geometry of tensor calculus I", Adv.
Math. 88 (1991), and "Braided tensor categories", Adv. Math. 102 (1993);
G. M. Kelly, M. L. Laplaza, "Coherence for compact closed categories", JPAA
19 (1980); P. Selinger, "Dagger compact closed categories and completely
positive maps" (2007), and "A survey of graphical languages for monoidal
categories", LNP 813 (2011); P. Etingof, S. Gelaki, D. Nikshych, V. Ostrik,
*Tensor Categories* (2015); B. Coecke, R. Duncan, "Interacting quantum
observables", New J. Phys. 13 (2011); B. Coecke, A. Kissinger, *Picturing
Quantum Processes* (2017); J. van de Wetering, "ZX-calculus for the working
quantum computer scientist" (2020); R. Duncan, A. Kissinger, S. Perdrix,
J. van de Wetering, "Graph-theoretic simplification of quantum circuits with
the ZX-calculus", Quantum 4 (2020); nLab pages "monoidal category",
"braided monoidal category", "symmetric monoidal category", "monoidal
functor", "compact closed category", "dagger-compact category", "string
diagram", "2-group", "ZX-calculus", "Frobenius algebra".
