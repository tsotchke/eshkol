---
kind: reference
status: current
owner-area: stdlib
since: v2.0-starlight
sources:
  - lib/core/quantum.esk
  - lib/core/quantum/circuit.esk
  - lib/core/quantum/optimize.esk
  - lib/core/quantum/route.esk
  - lib/core/quantum/register.esk
  - lib/core/quantum/qaoa.esk
  - lib/core/quantum/moonlab.esk
  - lib/types/quantum_types.cpp
  - tests/typesystem/qreg_corpus_gate.sh
  - tests/stdlib/quantum_register_test.esk
  - tests/stdlib/quantum_optimize_test.esk
  - tests/stdlib/quantum_route_test.esk
  - tests/stdlib/qaoa_test.esk
  - tests/quantum/qaoa_moonlab_crosscheck_test.esk
---

# `core.quantum` — quantum registers, regions, circuit compilation and QAOA

**Source:** [quantum.esk](../../../lib/core/quantum.esk) and
[`lib/core/quantum/`](../../../lib/core/quantum/).
**Require:** `(require core.quantum)`; the Moonlab backend and Moonlab
expectations need `(require core.quantum.moonlab)` in a build configured with
`-DESHKOL_QUANTUM_ENABLED=ON`. Not part of `(require stdlib)`.
**Design record:** [ADR-0048](../../design/adr/0048-quantum-register-types-and-circuit-compilation.md).
**Built on:** the linear `Qubit` judgment, [`core.monoidal.zx`](zx.md) for
exact semantics and ZX rewriting, and [Moonlab](../../design/MOONLAB_INTEGRATION.md).

## Register types

| Type | Meaning |
|---|---|
| `qreg<e>`, `(qreg e)` | a register of e qubits; linear: used exactly once |
| `qreg` | a register whose size is known only at run time (still linear) |
| `qidx<e>` | a qubit address of a `qreg<e>`: a natural below e |
| `qregion` | the capability of one quantum region |
| `qregion-body` | a procedure of one `qregion` with a classical result |

An index e is a natural, a name, or a sum of them with natural
coefficients: `qreg<3>`, `qreg<n>`, `qreg<n+1>`, `qreg<m+n>`, `qreg<2*n>`;
`(qreg (+ n 1))` is the same type. Dimensions unify at every call; a name
spelled like an integer parameter stands for that argument's compile-time
value, so `(qalloc qr 3)` has type `qreg<3>`.

The checker rejects, in every mode but `--unsafe` and on both engines:

| Program | Diagnostic |
|---|---|
| a `qreg<2>` passed where `qreg<3>` is declared | `quantum register dimension mismatch in argument 1 of 'f': expected qreg<3>, got qreg<2>` |
| a register used twice | `linear variable 'r' was consumed more than once` |
| a register never measured or released | `linear variable 'r' was not consumed` |
| a register computed and dropped in a sequence | `a linear value of type qreg<1> is computed and discarded` |
| `(qh r 2)` on a `qreg<2>` | `qubit index 2 in argument 2 of 'qh' is out of range for qreg<2>` |
| `(qcnot r 1 1)` | `qubit 1 is used twice by one 'qcnot' ... (no-cloning)` |
| a region body returning a register | `the body of a quantum region returns qreg<2>: quantum data cannot leave the region that owns it` |
| a region body returning a procedure that mentions the region | `the quantum region 'qr' escapes its region inside a returned procedure` |
| a body whose register size differs from its declared result | `quantum register dimension mismatch: the result of 'g' is declared qreg<3> but is qreg<2>` |
| an undeclared `qsplit` continuation parameter | `parameter 1 of the procedure passed to 'qsplit' receives qreg<1>; declare it with that type` |

```text
(define (ghz3 (r : qreg<3>)) : qreg<3>
  (qcnot (qcnot (qh r 0) 0 1) 1 2))
(define-quantum-region qr (run) (qmeasure (ghz3 (qalloc qr 2))))
;; [ERROR] Type error: quantum register dimension mismatch in argument 1 of
;; 'ghz3': expected qreg<3>, got qreg<2>
;; Refusing to generate code for a program that violates linear (no-cloning) typing
```

## Regions and primitives

`(define-quantum-region R (name formals ...) body ...)` defines a procedure
whose body runs in a fresh quantum region named R;
`(with-quantum-region (R) body ...)` is the expression form. Both are
`call-with-quantum-region` applied to `(lambda ((R : qregion)) body ...)`.
The region owns one simulator state; it closes on every exit, and on a normal
exit only when every register allocated in it was measured or released.

| Primitive | Type |
|---|---|
| `(qalloc qr n)` | `qregion integer -> qreg<n>` |
| `(qh r i)` `qx` `qy` `qz` `qs` `qsdg` `qt` `qtdg` | `qreg<n> qidx<n> -> qreg<n>` |
| `(qp r i p)` | P(p): diag(1, e^{i p pi}) |
| `(qrx r i theta)` `qry` `qrz` | `qreg<n> qidx<n> number -> qreg<n>` |
| `(qcnot r c t)` `qcz` `qswap` | `qreg<n> qidx<n> qidx<n> -> qreg<n>` |
| `(qrzz r a b theta)` | e^{-i theta Z(x)Z/2} |
| `(qccx r a b t)` `qccz` | three-qubit gates |
| `(qapply-circuit r c)` | apply a qcircuit of the register's width |
| `(qmeasure r)` | `qreg<n> -> list` of bits, consumes r |
| `(qmeasure-bit r)` | `qreg<1> -> integer` |
| `(qrelease r)` | measure and forget |
| `(qsplit r k f)` | `(f first-k rest)` with `f : (-> qreg<k> qreg<n-k> a)` |
| `(qjoin a b)` | `qreg<m> qreg<n> -> qreg<m+n>` |
| `(qreg->qubit r)` / `(qubit->qreg q)` | a `Qubit` is a one-qubit register |

```scheme
(require stdlib)
(require core.quantum)
(define-quantum-region qr (flip-first n)
  (let* ((r (qalloc qr n))
         (r (qx r 0)))
    (qmeasure r)))
(flip-first 3)   ;; => (1 0 0)
(with-quantum-region (qr) (qmeasure-bit (qx (qalloc qr 1) 0)))   ;; => 1
```

`(set-quantum-region-options! alist)` sets `(seed . k)` (a reproducible
measurement stream), `(optimize . #f)` (run the recorded gates as written),
`(coupling . cg)` (route every compiled block on the coupling graph cg; the
region keeps the logical-to-physical layout) and `(backend . b)`
(`statevector-backend`, the default, or `moonlab-backend`).
`(qregion-gate-log qr)` is the circuit a region has executed so far.

## Circuits

A circuit is `#(qcircuit n gates)` (`make-qcircuit`) with gates `(H q)`
`(X q)` `(Y q)` `(Z q)` `(S q)` `(Sdg q)` `(T q)` `(Tdg q)` `(P q p)`
`(XP q p)` `(RX q th)` `(RY q th)` `(RZ q th)` `(CNOT c t)` `(CZ a b)`
`(SWAP a b)` `(RZZ a b th)` `(CCX a b t)` `(CCZ a b c)`; wire 0 is the most
significant bit. An angle is a real or `(qparam k c)`, the parameter
c * theta_k; `qcircuit-bind` substitutes a vector of values, which may be AD
values inside `gradient`.

| Procedure | Result |
|---|---|
| `(qcircuit-run c)` | the final binary64 state vector from |0..0> |
| `(qcircuit-unitary c)` | the unitary, as a vector of columns |
| `(qstate-pauli-expectation n st paulis)` | <psi|P|psi> |
| `(qcircuit-equivalent-exact? c d)` | equal up to a global phase, exactly over Q(zeta_m) |
| `(qcircuit-equal-exact? c d)` | equal, exactly |
| `(qcircuit-zx-identity? c)` | #t when ZX simplification reduces c to bare wires (a proof) |
| `(qcircuit-equivalent-float? c d)` | equal up to a global phase within 1e-9 |
| `(qcircuit-lower c)` | the same circuit on {H X Y Z CNOT RX RY RZ} |
| `(qcircuit-stats c)` | gate count, two-qubit count, T-count, depth |

## Optimisation

`(qopt-optimize c [options])` repeats gate fusion (`qopt-fuse`),
commutation-aware cancellation (`qopt-cancel`), phase folding
(`qopt-phase-fold`) and ZX resynthesis of Clifford+T windows
(`qopt-zx-resynthesize`, through `core.monoidal.zx`) until the cost
(two-qubit count, T-count, gate count) stops falling. Every pass keeps the
unitary up to a global phase; `'((verify . #t))` checks each pass.

```scheme
(require stdlib)
(require core.quantum)
(define c (make-qcircuit 2 '((H 0) (CNOT 0 1) (T 1) (T 1) (CNOT 0 1) (CNOT 0 1))))
(qcircuit-stats c)                         ;; => ((gates . 6) (two-qubit . 3) (t-count . 2) (depth . 6))
(qcircuit-gates (qopt-optimize c))         ;; => ((H 0) (CNOT 0 1) (S 1))
(qcircuit-equivalent-exact? c (qopt-optimize c))   ;; => #t
(qcircuit-zx-identity? (qcircuit-concat (qopt-optimize c) (qcircuit-inverse c)))   ;; => #t
```

## Qubit mapping

`(coupling-line n)`, `(coupling-ring n)`, `(coupling-grid rows cols)`,
`(coupling-graph n edges)` build coupling graphs. `(qroute c cg [options])`
routes with SABRE (front layer, extended set, decay, reverse-traversal
initial layout) and returns a record with the routed circuit on physical
wires, the initial and final layouts and the number of SWAPs inserted;
`(qrouting-verify r c)` proves it equal to c, and
`(qrouting-respects-coupling? r cg)` checks every two-qubit gate.

```scheme
(require stdlib)
(require core.quantum)
(define c (make-qcircuit 3 '((H 0) (CNOT 0 2) (CNOT 1 2))))
(define r (qroute c (coupling-line 3)))
(list (qrouting-respects-coupling? r (coupling-line 3)) (qrouting-verify r c))   ;; => (#t #t)
```

## QAOA

`(maxcut-problem n edges)` with edges `(u v)` or `(u v w)`;
`(qaoa-parametric-circuit problem p)` (parameters gamma_1..gamma_p,
beta_1..beta_p; `RZZ(-gamma w)` per edge, `RX(2 beta)` per qubit);
`(qaoa-expectation problem gammas betas)`, exact on the state vector;
`(qaoa-objective problem p)` for `gradient`; `(qaoa-p1-expectation problem
gamma beta)`, the p = 1 closed form; `(maxcut-brute-force problem)`;
`(qaoa-optimize problem p)`, Adam ascent on exact AD gradients;
`(maxcut-pauli-terms problem)`, the cost as a Pauli sum.

```scheme
(require stdlib)
(require core.quantum)
(define ring (maxcut-problem 4 '((0 1) (1 2) (2 3) (3 0))))
(maxcut-brute-force ring)   ;; => (4 (0 1 0 1) (1 0 1 0))
(< (abs (- (qaoa-expectation ring '(0.4) '(1.1)) (qaoa-p1-expectation ring 0.4 1.1))) 1e-12)   ;; => #t
```

With `(require core.quantum.moonlab)`, `(qaoa-moonlab-program problem p)`
builds the same circuit and observable as a Moonlab program and
`(moonlab-expectation program theta)` evaluates it with Moonlab; inside
`(gradient (lambda (th) (moonlab-expectation program th)) theta)` the
derivative is Moonlab's exact adjoint, recorded as one custom-VJP node.

## Verification

`tests/typesystem/qreg_*.esk` (CTest `quantum_types_corpus`, native and
bytecode engines); `tests/stdlib/quantum_circuit_test.esk`,
`quantum_optimize_test.esk`, `quantum_route_test.esk`, `qaoa_test.esk`,
`quantum_register_test.esk` (CTest `*_jit`, `*_aot`);
`tests/quantum/qaoa_moonlab_crosscheck_test.esk` (CTest
`qaoa_moonlab_crosscheck`, quantum builds); gate
`scripts/run_quantum_types_gate.sh`.
