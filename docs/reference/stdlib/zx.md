---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/monoidal/zx.esk
  - tests/stdlib/zx_test.esk
  - tests/quantum/zx_moonlab_crosscheck_test.esk
---

# `core.monoidal.zx` — the ZX-calculus, exactly

**Source:** [zx.esk](../../../lib/core/monoidal/zx.esk).
**Require:** `(require core.monoidal.zx)` or `(require core.monoidal)`. Not
part of `(require stdlib)`.
**Design record:** [ADR-0041](../../design/adr/0041-core-monoidal-string-diagrams-zx.md).
**Built on:** [`core.monoidal`](monoidal.md) (FinVect, tensor networks,
string diagrams) and [`core.cyclotomic`](../../../lib/core/cyclotomic.esk).

The ZX-calculus is a graphical language for linear maps between qubits:
diagrams of Z and X spiders with phases and Hadamard boxes, and rewrite rules
that preserve their meaning. `core.monoidal.zx` holds ZX graphs as data,
evaluates them exactly as matrices over the cyclotomic field Q(zeta_8) (for
phases that are multiples of pi/4; Q(zeta_n) in general), states every rule
with the exact scalar it costs, simplifies graphs with the graph-like
strategy of Duncan, Kissinger, Perdrix and van de Wetering, and translates
Clifford+T circuits into ZX graphs next to an independent exact
state-vector semantics.

Sources: B. Coecke, R. Duncan, "Interacting quantum observables: categorical
algebra and diagrammatics", New J. Phys. 13 (2011); B. Coecke, A. Kissinger,
*Picturing Quantum Processes* (2017), ch. 9; J. van de Wetering,
"ZX-calculus for the working quantum computer scientist", arXiv:2012.13966
(2020); R. Duncan, A. Kissinger, S. Perdrix, J. van de Wetering,
"Graph-theoretic simplification of quantum circuits with the ZX-calculus",
Quantum 4, 279 (2020); P. Selinger, "A survey of graphical languages for
monoidal categories" (2011); nLab: [ZX-calculus](https://ncatlab.org/nlab/show/ZX-calculus),
[Frobenius algebra](https://ncatlab.org/nlab/show/Frobenius+algebra),
[dagger-compact category](https://ncatlab.org/nlab/show/dagger-compact+category).

## Representation and semantics

| Value | Representation |
|---|---|
| phase | an exact rational p, the angle p pi (mod 2) |
| graph | `#(zx-graph nodes edges inputs outputs next)`; nodes `#(id kind phase)` with kind `Z`, `X`, `H` (two legs) or `B` (boundary, one leg); edges `#(eid a b type)` with type `plain` or `hadamard`; parallel edges and self-loops allowed |
| scalar | `#(zx-scalar a p q)` = sqrt(2)^a e^{i pi p} q, q an exact rational |
| circuit | a list of gates `(H q)` `(X q)` `(Y q)` `(Z q)` `(S q)` `(Sdg q)` `(T q)` `(Tdg q)` `(P q p)` `(XP q p)` `(CNOT c t)` `(CZ a b)` `(SWAP a b)`; qubit 0 is the first tensor factor |

With k legs, Z(a) has entry 1 at |0...0>, e^{i a} at |1...1> and 0
elsewhere; X(a) = |+...+><+...+| + e^{i a}|-...-><-...-|, whose entry at a bit
string b is 2^{-k/2}(1 + e^{i a}(-1)^{|b|}); H = (1/sqrt 2)[[1, 1], [1, -1]]; a
Hadamard edge is a plain edge carrying an H box. `(zx-evaluate g S)` contracts
the graph's tensor network from its inputs to its outputs.
`(zx-graph-from nodes edges inputs outputs)` builds a graph from lists
`(id kind phase)` and `(a b)` / `(a b hadamard)`.

```scheme
(require stdlib)
(require core.monoidal.zx)
(define K (cyclotomic-scalars 8))
;; Z(pi/2) on one wire is diag(1, i); X(pi) is the NOT gate
(linear-map-equal? (zx-spider-map K 'Z 1/2 1 1) (circuit->unitary K 1 '((S 0))))      ;; => #t
(linear-map-equal? (zx-spider-map K 'X 1 1 1) (circuit->unitary K 1 '((X 0))))       ;; => #t
;; CNOT is sqrt 2 times a Z spider joined to an X spider
(define cnot (zx-graph-from '((0 B) (1 B) (2 B) (3 B) (4 Z 0) (5 X 0))
                            '((0 4) (1 5) (4 5) (4 2) (5 3)) '(0 1) '(2 3)))
(linear-map-equal? (linear-map-scale (zx-evaluate cnot K) (scalars-sqrt2 K)) (circuit->unitary K 2 '((CNOT 0 1))))  ;; => #t
```

## Rewrite rules

Every rule is a procedure `(rule g node ...)` returning `(g' . s)` with
[[g]] = s [[g']] exactly; a graph outside the rule's pattern is refused with
an error that names the rule and the condition.

| Rule | Procedure | Pattern | Scalar |
|---|---|---|---|
| spider fusion (f) | `(zx-fuse g u v)` | same colour, at least one plain edge between them | 1 |
| identity (id) | `(zx-remove-identity g v)` | phase 0, two legs | 1 |
| self-loop | `(zx-remove-self-loop g v)` | a plain self-loop | 1 |
| colour change (h) | `(zx-color-change g v)` | any spider; every leg toggles plain/Hadamard | 1 |
| H box to edge | `(zx-h-box->edge g h)` | an H box | 1 |
| HH = id (hh) | `(zx-hh-cancel g h1 h2)` | two H boxes on one plain edge | 1 |
| pi-copy (pi) | `(zx-pi-copy g n v)` | a two-legged phase-pi spider of the other colour on a plain edge into v (phase a) | e^{i a} |
| state copy (c) | `(zx-state-copy g s v)` | a one-legged phase-(a pi) spider of the other colour (a = 0, 1) into v (phase b) with n other legs | sqrt(2)^{1-n} e^{i pi a b} |
| bialgebra (b) | `(zx-bialgebra g z x)` | phase-free Z and X spiders, three legs each, one plain edge between them | sqrt 2 |
| Hopf (ho) | `(zx-hopf g u v)` | two parallel plain edges between spiders of different colours | 1/2 |
| Euler (eu) | `(zx-euler g h)` | an H box becomes Z(pi/2) X(pi/2) Z(pi/2) | e^{-i pi/4} |
| Hadamard Hopf | `(zx-hadamard-hopf g u v)` | two parallel Hadamard edges between Z spiders | 1/2 |
| Hadamard self-loop | `(zx-hadamard-self-loop g v)` | a Hadamard self-loop on Z(a); the phase becomes a + pi | 1/sqrt 2 |
| local complementation (lc) | `(zx-local-complement g v)` | an interior Z(+-pi/2) whose k edges are single Hadamard edges to Z spiders | sqrt(2)^{d-(k-1)} e^{+-i pi/4} |
| pivoting (p) | `(zx-pivot g u v)` | adjacent interior Z spiders with phases 0 or pi and single Hadamard edges | sqrt(2)^{d+1-\|A\|-\|B\|-2\|C\|} (-1)^{a_u a_v} |

In the last two rows d is the number of Hadamard edges the rule adds minus
the number it removes (each Hadamard edge carries its factor 1/sqrt 2), and
A, B and C are the neighbours of u only, of v only, and of both.

```scheme
(require stdlib)
(require core.monoidal.zx)
(define K (cyclotomic-scalars 8))
;; the bialgebra rule on a Z and an X spider with two inputs and two outputs
(define g (zx-graph-from '((0 B) (1 B) (2 B) (3 B) (4 Z 0) (5 X 0)) '((0 4) (1 4) (4 5) (5 2) (5 3)) '(0 1) '(2 3)))
(define r (zx-bialgebra g 4 5))
(list (zx-spider-count g) (zx-spider-count (car r)))   ;; => (2 4)
(zx-scalar-sqrt2-power (cdr r))                        ;; => 1
(zx-equal-up-to? g (car r) (cdr r) K)                  ;; => #t
;; pi-copy: a NOT through Z(3 pi/4) costs e^{3 i pi/4}
(define p (zx-graph-from '((0 B) (1 B) (2 B) (3 X 1) (4 Z 3/4)) '((0 3) (3 4) (4 1) (4 2)) '(0) '(1 2)))
(define q (zx-pi-copy p 3 4))
(list (zx-scalar-phase (cdr q)) (zx-equal-up-to? p (car q) (cdr q) K))   ;; => (3/4 #t)
```

## Simplification

`(zx-to-graph-like g)` turns H boxes into Hadamard edges, recolours X
spiders, fuses spiders along plain edges and removes self-loops and parallel
Hadamard edges, so that every spider is a Z spider and spiders are joined by
single Hadamard edges. `(zx-simplify g)` then also removes identities and
applies local complementation and pivoting until none applies. Both return
`(g' . s)` with [[g]] = s [[g']]; `zx-graph-like?` tests the form.

```scheme
(require stdlib)
(require core.monoidal.zx)
(define K (cyclotomic-scalars 8))
(define gates '((H 0) (CNOT 0 1) (S 1) (CNOT 1 2) (H 2) (CZ 0 2) (H 1) (S 0) (CNOT 2 0)))
(define r (circuit->zx 3 gates))
(define s (zx-simplify (car r)))
(list (zx-spider-count (car r)) (zx-graph-like? (car s)))   ;; => (10 #t)
(< (zx-spider-count (car s)) (zx-spider-count (car r)))     ;; => #t
;; the simplified graph still is the circuit, scalar included
(linear-map-equal? (circuit->unitary K 3 gates)
                   (linear-map-scale (zx-evaluate (car s) K) (zx-scalar->element K (zx-scalar-mul (cdr r) (cdr s)))))  ;; => #t
;; CNOT twice simplifies to two bare wires
(zx-spider-count (car (zx-simplify (car (circuit->zx 2 '((CNOT 0 1) (CNOT 0 1)))))))   ;; => 0
```

## Circuits

`(circuit->zx n gates)` returns `(g . s)` with U = s [[g]]: a phase gate is a
Z spider, `X` and `XP` are X spiders, `H` toggles the next edge to a Hadamard
edge, `CNOT` is a Z spider joined to an X spider (s gains sqrt 2), `CZ` two Z
spiders joined by a Hadamard edge (sqrt 2), `Y` is Z(pi) then X(pi) (s gains
i) and `SWAP` crosses the wires. `(circuit->unitary S n gates)` and
`(circuit-state S n gates)` compute the same circuit gate by gate on exact
state vectors, independently of the ZX graph; `(pauli-expectation S n psi
paulis)` and `(circuit-pauli-expectation S n gates paulis)` give
<psi|P|psi> for a Pauli string such as `(X Z I)`.

```scheme
(require stdlib)
(require core.monoidal.zx)
(define K (cyclotomic-scalars 8))
(define ghz '((H 0) (CNOT 0 1) (CNOT 1 2)))
(define r (circuit->zx 3 ghz))
(linear-map-equal? (circuit->unitary K 3 ghz) (linear-map-scale (zx-evaluate (car r) K) (zx-scalar->element K (cdr r))))  ;; => #t
(scalars-equal? K (circuit-pauli-expectation K 3 ghz '(X X X)) (scalars-one K))   ;; => #t
;; <X> after H and T is cos(pi/4) = 1/sqrt 2, exactly
(scalars-equal? K (circuit-pauli-expectation K 3 '((H 0) (T 0)) '(X I I)) (scalars-inverse-sqrt2 K))  ;; => #t
```

`(zx-box-interpretation S)` returns the dimensions and box maps that make
the ZX generators, named `(Z p m n)`, `(X p m n)` and `H`, boxes of
[`core.monoidal.diagram`](monoidal.md#string-diagrams), so a ZX term written
as a string diagram evaluates compositionally as well.

## Verification

`tests/stdlib/zx_test.esk` (CTest `zx_test_jit`, `zx_test_aot`): the
generators against their matrices; every rule on every instance over the
phases k pi/4, k = 0..7 (fusion along one or two edges in four shapes, the
identity rule for every pair of edge types, colour change with up to three
legs, H boxes, pi-copy and state copy with up to three further legs, the
bialgebra in both orientations, Hopf and Hadamard Hopf for all 64 phase
pairs, local complementation for every edge pattern among up to three
neighbours, pivoting for all neighbourhood sizes up to two), each checked as
an exact equality [[lhs]] = s [[rhs]]; refusals outside the patterns; 30
random Clifford+T circuits on up to four qubits against the state-vector
semantics, before and after simplification; and the ZX generators as
string-diagram boxes.

`tests/quantum/zx_moonlab_crosscheck_test.esk` (builds with
`-DESHKOL_QUANTUM_ENABLED=ON`, CTest `zx_moonlab_crosscheck`) compares
circuits on one to five qubits with Moonlab's state-vector simulator through
[`agent.quantum`](../../design/MOONLAB_INTEGRATION.md): Moonlab works in
binary64 and agent.quantum exposes its state through gates and <Z_q>, so the
comparison is on Pauli expectations, each within 1e-10, with full state
tomography (fidelity 1 within 1e-9) for up to three qubits.
