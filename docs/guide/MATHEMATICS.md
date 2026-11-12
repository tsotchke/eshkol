---
kind: guide
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/sheaf.esk
  - lib/core/topos.esk
  - lib/core/ring.esk
  - lib/core/category.esk
  - lib/core/simplicial.esk
  - lib/core/homological.esk
---

# Computing mathematics in Eshkol

This is a reader's guide to the mathematics libraries: sites and sheaves, toposes with
Kripke–Joyal semantics, rings as a functor-of-points value, finite category theory with Kan
extensions, and the simplicial and homological algebra built on top of them. Every example
below is copied verbatim from the library's own reference page or from the test suite that
exercises it — nothing here is a new API. Each library decides its claims exactly: every
quantifier ranges over a finite set, so answers come from enumeration and exact linear algebra,
never from sampling.

For independent confirmation, see the [nLab conformance report](../reports/NLAB_CONFORMANCE.md):
51 pages, 141 claims quoted from nLab and turned into finite statements, 132 of them decided and
passing in native JIT and AOT.

Two libraries arrive later in the v1.4.x line and are covered at the end: Gröbner bases and
monoidal categories with the ZX calculus. A third, Lawvere theories, is noted alongside
`core.category` for the same reason.

## Sites and sheaves — `core.sheaf`

`(require core.sheaf)` loads sheaf theory on finite sites: finite categories and posets, sieves
and Grothendieck topologies, presheaves of sets and of abelian groups, the sheaf condition as a
gluing operation, sheafification by the plus construction, and Cech and derived-functor
cohomology. Design record: [ADR-0033](../design/adr/0033-core-sheaf-finite-sites.md). Reference:
[sheaf.md](../reference/stdlib/sheaf.md).

```scheme
(require stdlib)
(require core.sheaf)
;; the pseudocircle: a = 0, b = 1 below c = 2 and d = 3
(define X (make-finite-space 4 (list (cons 0 2) (cons 1 2) (cons 0 3) (cons 1 3))))
(define site (finite-space-site X))
(define C (site-category site))
(display (finite-space-opens X)) ;; => (() (1) (0) (0 1) (0 1 3) (0 1 2) (0 1 2 3))
(display (grothendieck-topology? site)) ;; => #t
;; a "topology" in which {a, b} has no cover fails the maximal-sieve axiom there
(define ab (category-object-of-label C (list 0 1)))
(define without-ab (make-site C (lambda (c S) (and (not (= c ab)) (site-covering? site c S)))))
(display (car (grothendieck-topology-violations without-ab))) ;; => (maximal 3)
```

`finite-space-site` builds the Alexandrov topology of a finite preorder; `grothendieck-topology?`
checks the three SGA 4 II.1.1 axioms by enumeration and names any violation exactly.

## Toposes and Kripke–Joyal semantics — `core.topos`

`(require core.topos)` builds presheaf toposes and the internal language they interpret: the
subobject classifier Omega, Lawvere–Tierney topologies, exponentials and power objects, and
Kripke–Joyal forcing for first-order and intuitionistic higher-order formulas. Design record:
[ADR-0035](../design/adr/0035-kripke-joyal-semantics.md). Reference:
[topos.md](../reference/stdlib/topos.md).

```scheme
(require stdlib)
(require core.sheaf.site)
(require core.topos)

;; the walking arrow 0 -> 1, presheaf topos
(define C (poset-category 2 (lambda (x y) (<= x y))))
(define T (presheaf-topos C))
(define L (make-topos-language T '() '()))

(define lem (topos-formula L '() '(forall (p Omega) (or (holds p) (not (holds p))))))
(topos-valid? lem)          ;; => #f
(topos-counter-witness lem) ;; => (1 ()) refuted at stage 1, no free variables
(topos-cross-check lem)     ;; => (2 #t) two instances, forcing agrees with Omega

(define dn (topos-formula L '((p Omega)) '(implies (holds p) (not (not (holds p))))))
(topos-valid? dn)           ;; => #t
```

The law of excluded middle fails on the walking-arrow topos (it is not Boolean); double negation
still holds, as intuitionistic logic predicts. `topos-cross-check` compares forcing against the
internal truth value Omega directly, instance by instance.

## Rings — `core.ring`

`(require core.ring)` is the ring-carrier protocol: a commutative ring with identity given as
data, so a program written once against the generic operations `r+ r- r* r-neg r-inv r/ r-expt
r= r-int` runs over any carrier — ordinary, exact, modular, dual-number, Taylor-jet, polynomial,
or group-ring — and takes that carrier's R-points. Design record:
[ADR-0034](../design/adr/0034-ring-carriers-functor-of-points.md). Reference:
[ring.md](../reference/stdlib/ring.md).

```scheme
(require core.ring)
;; f(x) = x^3 - 2x + 5, written once against the carrier R
(define (f R x) (r+ R (r- R (r-expt R x 3) (r* R 2 x)) 5))
(display (f integers 10)) ;; => 985
(newline)
(display (f rationals 2/3)) ;; => 107/27
(newline)
(display (f flonums 0.5)) ;; => 4.125
(newline)
(display (f (make-zmod 7) 3)) ;; => 5
(newline)
(define J (make-jets rationals 3))
;; f(2 + e) in Q[e]/e^3: the Taylor coefficients f(2), f'(2), f''(2)/2
(display (jet-coefficients (f J (jet-seed J 2)))) ;; => (9 10 6)
(newline)
(define Qx (make-polys rationals))
;; f at the generic point x: its coefficients, lowest degree first
(display (ring-poly-coefficients (f Qx (ring-poly-variable Qx)))) ;; => (5 -2 0 1)
(newline)
;; over the native tower the compiler's own AD operators see through it
(display (derivative (lambda (x) (f numeric-tower x)) 2)) ;; => 10
```

The same `f` evaluated over the integers, the rationals, IEEE flonums, Z/7, the jet ring
Q[e]/e^3, and the polynomial ring Q[x] — one definition, six interpretations, each exact in its
own carrier.

## Finite categories and Kan extensions — `core.category`

`(require core.category.functor)` (loaded by `core.category`) is finite category theory on
`core.sheaf.site`'s finite-category value: functor categories, comma categories, the Grothendieck
construction, adjunctions found through terminal objects of comma categories, monads, and Kan
extensions (Lan ⊣ F* ⊣ Ran). Design record:
[ADR-0036](../design/adr/0036-core-category-finite-categories.md). Reference:
[category.md](../reference/stdlib/category.md).

```scheme
(require stdlib)
(require core.category.functor)
(define Sq (commutative-square-category))      ;; a: 0 -> 1, b: 0 -> 2, c: 1 -> 3, d: 2 -> 3, c a = d b
(list (category-object-count Sq) (category-arrow-count Sq) (length (category-hom Sq 0 3)))  ;; => (4 9 1)
(length (functors-between (walking-arrow-category) Sq))                                     ;; => 9
(define C3 (poset-category 3 (lambda (x y) (<= x y))))
(define FC (functor-category (walking-arrow-category) C3))
(list (category-object-count FC) (category-arrow-count FC))                                ;; => (6 20)
;; the inclusion of {0, 2} into 0 < 1 < 2 and its right adjoint, found through comma categories
(define inc (inclusion-functor C3 (list 0 2)))
(define A (right-adjoint (cdr inc)))
(functor-objects (adjunction-right A))           ;; => #(0 0 1)
(adjunction-violations A)                        ;; => ()
(monad-laws-violations (adjunction-monad A))     ;; => ()
(category-nerve-cohomology (cyclic-group-category 2) 4)  ;; => ((0 1 ()) (1 0 ()) (2 0 (2)) (3 0 ()) (4 0 (2)))
```

`right-adjoint` takes the terminal object of each comma category and returns `#f` when one is
missing; `adjunction-violations` checks functoriality, naturality of unit/counit, and both
triangle identities. The last line is the integral cohomology of the classifying space of Z/2.

Lawvere theories (semigroups through Heyting algebras, built on `core.theory`'s free models and
`Omega_J(c)`) are a separate library, `core.theory` — see below, arrives with v1.4.x.

## Simplicial and homological algebra — `core.simplicial`, `core.homological`

`(require core.simplicial)` is simplicial sets, Kan complexes, nerves and classifying spaces, the
bar construction, and standard constructions (`standard-simplex`, `simplicial-sphere`,
`real-projective-space`, `lens-space`). `(require core.homological)` is chain/cochain complexes,
homology and cohomology with torsion, chain maps, mapping cones, and long exact sequences. Design
record: [ADR-0037](../design/adr/0037-simplicial-and-homological-algebra.md). Reference:
[simplicial.md](../reference/stdlib/simplicial.md), [homological.md](../reference/stdlib/homological.md).

```scheme
(require stdlib)
(require core.simplicial)
(define D2 (standard-simplex 2))
(display (map (lambda (n) (simplicial-set-count D2 n)) (list 0 1 2 3))) ;; => (3 3 1 0)
(define s (simplex (list 0 1 2) 2))
(display (simplicial-set-face D2 1 s)) ;; => #(simplex (0 1) (0 2))
(display (simplicial-set-degeneracy D2 0 (simplex (list 0 2) 1))) ;; => #(simplex (0 0 1) (0 2))
(display (simplicial-set-operator D2 s (list 0 0 2))) ;; => #(simplex (0 0 1) (0 2))
(display (length (simplicial-set-simplices D2 3))) ;; => 15
```

```scheme
(require stdlib)
(require core.ring)
(require core.homological)
;; the cellular chain complex of RP^2: Z <-0- Z <-2- Z
(define C (make-chain-complex integers (list 1 1 1) (list (vector (vector 0)) (vector (vector 2)))))
(display (homology C)) ;; => ((0 1 ()) (1 0 (2)) (2 0 ()))
(display (cohomology (complex-dual C))) ;; => ((0 1 ()) (1 0 ()) (2 0 (2)))
(display (homology (complex-change-ring C (make-zmod 2)))) ;; => ((0 1 ()) (1 1 ()) (2 1 ()))
(display (homology-generators C 1)) ;; => ((2 . #(1)))
```

`RP^2` has `H_0 = Z`, `H_1 = Z/2` (the torsion that distinguishes it from a sphere), and
`H_2 = 0`; reducing coefficients to `Z/2` turns every group into `Z/2`, matching the universal
coefficient theorem.

## Lawvere theories — arrives with v1.4.x

Algebraic theories from semigroups through Heyting algebras, presented as Lawvere theories with
free models deciding the word problem, and `Omega_J(c)` models reached through `core.theory.topos`:
library `core.theory`, [PR #878](https://github.com/tsotchke/eshkol/pull/878). Planned for
v1.4.x.

## Gröbner bases — arrives with v1.4.x

Gröbner bases over polynomial rings `K[x1..xn]`, with Q as the default field computed by
multimodular arithmetic: library `core.groebner`, [PR #880](https://github.com/tsotchke/eshkol/pull/880).
Planned for v1.4.x.

## Monoidal categories and ZX — arrives with v1.4.x

Monoidal coherence on finite categories, FinVect over Z, Q and cyclotomic fields, string diagrams
with a normal form, and the ZX calculus for quantum circuits: library `core.monoidal`,
[PR #881](https://github.com/tsotchke/eshkol/pull/881). Planned for v1.4.x.
