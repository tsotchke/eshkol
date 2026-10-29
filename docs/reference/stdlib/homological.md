---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/homological.esk
  - lib/core/homological/complex.esk
  - lib/core/homological/abelian.esk
  - lib/core/homological/spectral.esk
  - lib/core/homological/simplicial.esk
  - tests/stdlib/homological_test.esk
---

# `core.homological` — homological algebra over ring carriers

`(require core.homological)` loads one library for exact homological
algebra: chain and cochain complexes over the ring carriers of
[`core.ring`](ring.md) (`Z`, `Q`, `F_p` and other fields), homology with
torsion and explicit generators, chain maps, homotopies, mapping cones and
long exact sequences, double complexes, finitely presented abelian groups
with `Hom`, tensor, `Ext` and `Tor` over `Z`, spectral sequences of filtered
complexes page by page, and the cochain algebra of simplicial sets: the
Alexander–Whitney cup product, cup-i products, Steenrod squares, the
Bockstein, and group cohomology through the bar construction.

Integer linear algebra is that of [`core.integer_linalg`](integer_linalg.md)
(the Smith normal form; ranks and invariant factors are computed without
building the transforms), so torsion is exact. Design record:
[ADR-0037](../../design/adr/0037-simplicial-and-homological-algebra.md).

| Module | Contents |
|---|---|
| [`core.homological`](../../../lib/core/homological.esk) | loads all of the modules below |
| [`core.homological.complex`](../../../lib/core/homological/complex.esk) | coefficients, complexes, homology, generators, class tests, duals, tensor products, chain maps, homotopies, cones, long exact sequences, double and total complexes, Q/Z coefficients |
| [`core.homological.abelian`](../../../lib/core/homological/abelian.esk) | finitely presented abelian groups, lattices, subquotients, `Hom`, tensor, `Ext`, `Tor`, universal coefficients |
| [`core.homological.spectral`](../../../lib/core/homological/spectral.esk) | filtered complexes, pages `E_r`, differentials, convergence, the two spectral sequences of a double complex |
| [`core.homological.simplicial`](../../../lib/core/homological/simplicial.esk) | normalized and Moore chains of simplicial sets, cochains, cup and cup-i products, Steenrod squares, the Bockstein, group (co)homology |

The modules are not part of `(require stdlib)`; require them explicitly.
Simplicial sets come from [`core.simplicial`](simplicial.md).

## Coefficients

A coefficient ring is a carrier of `core.ring`: `integers`, `rationals`,
`(make-zmod p)` for a prime `p`, or any carrier that declares `field?`.
Over `Z` ranks and torsion come from the Smith normal form; over `Q` from
the same engine after clearing denominators; over a field from Gauss–Jordan
elimination with the carrier's own operations. `Z/m` for composite `m`,
jets and polynomial rings are refused by name.

| Procedure | Meaning |
|---|---|
| `(coefficient-kind R)` | `'Z`, `'Q` or `'field` |
| `coefficient-reduce` `coefficient-reduce-vector` `coefficient-reduce-matrix` | images under `Z -> R` (and `Q -> R`) |
| `(coefficient-rank R A)` `(coefficient-kernel R A ncols)` | rank and a kernel basis |
| `(coefficient-solve R B ncols v)` `(coefficient-in-span? R B ncols v)` | a solution of `B y = v`, or `#f` |
| `coefficient-matrix-mul` `coefficient-matrix-vector` `coefficient-vector-add` `coefficient-vector-scale` `coefficient-zero-vector?` | arithmetic in `R` |

## Complexes

`#(complex kind R dims maps)`: `kind` is `'cochain` (`D_k : C^k -> C^{k+1}`)
or `'chain` (`d_k : C_k -> C_{k-1}`); matrices are those of
`core.integer_linalg` (vectors of row vectors), `#f` is the zero map. A
complex known through degree `N` may carry the differential that leaves
(cochain) or enters (chain) its top degree, which is how truncations of
infinite complexes such as bar constructions are represented exactly.

| Procedure | Meaning |
|---|---|
| `(make-cochain-complex R dims Ds)` | `Ds = (D_0 ... )`, of length `N` or `N + 1` |
| `(make-chain-complex R dims ds)` | `ds = (d_1 ... )`, of length `N` or `N + 1` |
| `complex-kind` `complex-ring` `complex-dims` `complex-dim` `complex-top-degree` `complex-map` `complex-in-map` `complex-out-map` | access |
| `(complex-d-squared-violations C)` | degrees where `d d` is not zero |
| `(homology C [kmax])` `(cohomology C [kmax])` | `((k betti torsion) ...)`; over a field `betti` is the dimension |
| `homology-betti` `homology-torsion` `homology-group-order` `(homology-rank-over H k p)` | reading a table |
| `(homology-generators C k)` | `((order . cycle) ...)`, order 0 for a free summand |
| `homology-cycle?` `homology-boundary?` | class tests (over `Z` a lattice test) |
| `(homology-classes-independent? C k vs)` `(homology-coordinates C k v)` | over a field |
| `(complex-dual C)` `(complex-change-ring C R)` `(complex-shift C s)` `(complex-direct-sum C D)` `(complex-tensor C D)` | constructions |
| `(complex-euler-characteristic C)` | alternating sum of dimensions |

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

## Chain maps, cones and long exact sequences

| Procedure | Meaning |
|---|---|
| `(make-chain-map C D fs)` | `fs = (f_0 ... f_N)`, `f_k : C_k -> D_k` |
| `chain-map-violations` `chain-map-compose` `chain-map-identity` | checks and composition |
| `(chain-homotopy-violations f g hs)` | degrees where `f - g /= d h + h d` |
| `(induced-map f k)` | over a field, the matrix of `H_k(f)` in the bases of `homology-generators` |
| `(mapping-cone f)` | cochain map: `Cone^n = A^n (+) B^{n-1}`, `D(a, b) = (D a, f a - D b)`; chain map: `Cone_n = A_{n-1} (+) B_n`, `d(a, b) = (-d a, f a + d b)` |
| `mapping-cone-inclusion` `mapping-cone-projection` | the maps of the cone sequence |
| `(long-exact-sequence f nmax)` | over a field, per degree `(n dim H(K) dim H(A) dim H(B) rank p rank f rank i)` |
| `(long-exact-sequence-exact? f nmax)` | exactness at every node, from the ranks |

For the restriction `C(X) -> C(Y)` along a subcomplex the cochain cone
computes the relative cohomology `H^n(X, Y)`.

```scheme
(require stdlib)
(require core.ring)
(require core.simplicial)
(require core.homological)
(define F2 (make-zmod 2))
(define rp4 (real-projective-space 4))
(define rp2 (simplicial-skeleton rp4 2))
(define CX (simplicial-cochain-complex rp4 4 F2))
(define CY (simplicial-cochain-complex rp2 4 F2))
;; restriction of cochains: each simplex of RP^2 is a simplex of RP^4
(define f (make-chain-map CX CY (map (lambda (n) (let ((M (integer-matrix (complex-dim CY n) (complex-dim CX n))))
                                                   (if (> (complex-dim CY n) 0) (integer-matrix-set! M 0 0 1))
                                                   M))
                                     (iota 5))))
(display (chain-map-violations f)) ;; => ()
(display (map cadr (long-exact-sequence f 4))) ;; => (0 0 0 1 1)
(display (long-exact-sequence-exact? f 3)) ;; => #t
```

## Double complexes and Q/Z coefficients

| Procedure | Meaning |
|---|---|
| `(make-double-complex R P Q dim h v)` | `C^{p,q}` for `0 <= p <= P`, `0 <= q <= Q`; `h : C^{p,q} -> C^{p+1,q}`, `v : C^{p,q} -> C^{p,q+1}` commuting |
| `(double-complex-violations X)` | failing squares and `h h`, `v v` |
| `(total-complex X nmax)` | `Tot^n = (+)_{p+q=n} C^{p,q}`, `D = h + (-1)^p v`, blocks ordered by `p` |
| `total-complex-offsets` | where each block sits in `Tot^n` |
| `(qz-coboundary-test C n N)` `(qz-coboundary? C n U N)` | whether the cochain `U/N` with values in `Q/Z` is a coboundary: with `e` the exponent of the torsion of `coker D_{n-1}`, exactly when `e U` lies in `im D_{n-1} + N e Z^m` |
| `(torsion-exponent A)` | the largest invariant factor |

## Abelian groups, Ext and Tor

A finitely presented abelian group is `#(fp-group n R)`, the quotient of
`Z^n` by the columns of `R`. Kernels, cokernels and subquotients are
computed with lattices (a basis from the Smith normal form, a quotient from
the Smith normal form of coordinates). `Tor` and `Ext` come from a free
resolution `0 -> Z^r -> Z^n -> A -> 0`; results are `(betti torsion)`.

| Procedure | Meaning |
|---|---|
| `(make-fp-group n R)` `(abelian-group betti torsion)` | presentations |
| `fp-group-invariants` | `(betti torsion)` |
| `(fp-hom-well-defined? A B F)` `(fp-kernel A B F)` `(fp-cokernel A B F)` `(fp-subquotient A B C f g)` | maps and their (co)kernels, homology at `B` |
| `lattice-basis` `lattice-contains?` `lattice-quotient-invariants` | lattices by generators |
| `abelian-hom` `abelian-tensor` `abelian-tor` `abelian-ext` | the functors over `Z` |
| `(universal-coefficients-cohomology H)` | `H^k = Z^{b_k} (+) T_{k-1}` from an integral homology table |
| `(universal-coefficients-mod-p H p)` | `dim H^k(X; F_p)` from it |

```scheme
(require stdlib)
(require core.homological)
(define Z4 (abelian-group 0 (list 4)))
(define Z6 (abelian-group 0 (list 6)))
(display (list (abelian-tor Z4 Z6) (abelian-ext Z4 Z6))) ;; => ((0 (2)) (0 (2)))
(display (abelian-ext (abelian-group 1 (list 2)) (abelian-group 1 '()))) ;; => (0 (2))
(display (universal-coefficients-mod-p (list (list 0 1 '()) (list 1 0 (list 2)) (list 2 0 '())) 2)) ;; => ((0 1 ()) (1 1 ()) (2 1 ()))
```

## Spectral sequences

A filtered complex is `#(filtered-complex C filtration)`: a cochain complex
and, per degree, the filtration degree of each basis vector (`F^p C^n` is
spanned by the vectors of degree `>= p`). With `n = p + q`,
`Z_r^{p,n} = {x in F^p C^n : D x in F^{p+r}}`,
`B_r^{p,n} = D Z_{r-1}^{p-r+1,n-1} + Z_{r-1}^{p+1,n}` and
`E_r^{p,q} = Z_r / B_r` (McCleary, 2.2), computed over `Z` as lattices
(torsion visible) and over a field by ranks. The condition on `D x` needs
the filtration one degree up, so a complex answers for degrees below its top.

| Procedure | Meaning |
|---|---|
| `(make-filtered-complex C filtration)` | `filtration : n -> vector of degrees` |
| `(filtered-complex-violations X)` | entries of `D` that lower the filtration |
| `(spectral-sequence-entry X r p q)` | `E_r^{p,q}` as `(betti torsion)` |
| `(spectral-sequence-page X r nmax)` | the nonzero entries `(p q betti torsion)` of total degree `<= nmax` |
| `(spectral-sequence-differential-rank X r p q)` | over a field, the rank of `d_r : E_r^{p,q} -> E_r^{p+r,q-r+1}` |
| `(spectral-sequence-page-violations X rmax nmax)` | where `E_{r+1}` differs from the cohomology of `(E_r, d_r)` (a field) |
| `(spectral-sequence-limit-page X)` | the page from which every differential vanishes |
| `(filtration-graded-cohomology X p n)` | `F^p H^n / F^{p+1} H^n` |
| `(spectral-sequence-convergence-violations X nmax)` | where `E_infinity^{p,q}` and `gr^p H^{p+q}` differ (exact over `Z`) |
| `(double-complex-filtered-complex X nmax by)` | `Tot(X)` filtered by columns (`'columns`) or rows (`'rows`) |

```scheme
(require stdlib)
(require core.ring)
(require core.simplicial)
(require core.homological)
;; the skeletal filtration of RP^5: E_2 = E_infinity = H^*(RP^5; Z) through degree 4
(define C (simplicial-cochain-complex (real-projective-space 5) 5))
(define X (make-filtered-complex C (lambda (n) (make-vector (complex-dim C n) n))))
(display (spectral-sequence-page X 2 4)) ;; => ((0 0 1 ()) (2 0 0 (2)) (4 0 0 (2)))
(display (spectral-sequence-convergence-violations X 4)) ;; => ()
```

## Cochains of simplicial sets

Chains and cochains are normalized: the basis in degree `n` is the list of
nondegenerate n-simplices, a cochain a vector on it.

| Procedure | Meaning |
|---|---|
| `(simplicial-chain-complex X nmax [R])` `(simplicial-cochain-complex X nmax [R])` | degrees `0 .. nmax` with the differential needed at `nmax` |
| `(simplicial-homology X nmax [R])` `(simplicial-cohomology X nmax [R])` | (co)homology, default over `Z` |
| `(simplicial-moore-complex X nmax [R])` | the complex on every simplex, degenerate ones included |
| `simplicial-boundary-matrix` | `d_n` over `Z` |
| `cochain-value` `cochain-from-function` `cochain-basis-vector` | cochains |
| `(simplicial-coboundary X u n [R])` | `delta u` |
| `(cup-product X u p v q [R])` | `(u cup v)(x) = u(front_p x) v(back_q x)` |
| `(cup-i-product X u p v q i)` | over `F_2`, the Medina-Mardones formula for Steenrod's cup-i |
| `(steenrod-square X u n k)` | `Sq^k u = u cup_{n-k} u` |
| `(bockstein X u n p)` | `(delta u~)/p mod p` for a mod-p cocycle |
| `(group-cochain-complex G nmax R [module])` `(group-cohomology G nmax R [module])` | normalized bar cochains with values in `R^r`, `G` acting through `module` |
| `(make-group-module r action)` `(trivial-group-module r)` | `action : g -> r x r` integer matrix |
| `(group-chain-complex G nmax [R])` `(group-homology G nmax [R])` | chains of `BG` |

```scheme
(require stdlib)
(require core.ring)
(require core.simplicial)
(require core.homological)
(define F2 (make-zmod 2))
(display (group-homology (symmetric-group 3) 3)) ;; => ((0 1 ()) (1 0 (2)) (2 0 ()) (3 0 (6)))
(display (simplicial-homology (lens-space 7 2) 3)) ;; => ((0 1 ()) (1 0 (7)) (2 0 ()) (3 1 ()))
(define BZ2 (classifying-space (cyclic-group 2)))
(define x (cochain-from-function BZ2 1 (lambda (t) 1)))
(define x2 (cup-product BZ2 x 1 x 1 F2))
(define x3 (cup-product BZ2 x2 2 x 1 F2))
;; Sq^1 x^3 = x^4 (C(3,1) is odd) and Sq^1 x^2 = 0
(display (list (steenrod-square BZ2 x3 3 1) (steenrod-square BZ2 x2 2 1))) ;; => (#(1) #(0))
(define sign (make-group-module 1 (lambda (g) (vector (vector (if (= g 0) 1 -1))))))
(display (group-cohomology (cyclic-group 2) 3 integers sign)) ;; => ((0 0 ()) (1 0 (2)) (2 0 ()) (3 0 (2)))
```

## Worked programs

- [`examples/mathematics_cohomology_rings_cup_products.esk`](../../../examples/mathematics_cohomology_rings_cup_products.esk):
  cohomology rings of `K(Z/p,1)` and `B(Z/2)^n`, the Bockstein, and the
  twists of `(Z/2)^3` through the Q/Z coboundary test.
- [`examples/mathematics_group_cohomology_order_eight.esk`](../../../examples/mathematics_group_cohomology_order_eight.esk):
  integral homology of the groups of order eight, normalized against
  unnormalized bar complexes.
- [`examples/mathematics_cohomology_rings_nonabelian.esk`](../../../examples/mathematics_cohomology_rings_nonabelian.esk):
  the mod-2 cohomology rings of D_4 and Q_8, which the ring structure separates.
- [`examples/mathematics_group_cohomology_sweep.esk`](../../../examples/mathematics_group_cohomology_sweep.esk):
  integral homology receipts of a chosen finite group (driven by
  `scripts/run_math_sweep.sh`).
- [`examples/mathematics_two_groups_postnikov.esk`](../../../examples/mathematics_two_groups_postnikov.esk):
  cohomology of 2-group nerves over `Z/2` and `Z`.
- [`examples/mathematics_descent_hypercohomology.esk`](../../../examples/mathematics_descent_hypercohomology.esk):
  the descent double complex of a non-Leray cover and the hypercohomology
  of a two-term complex of sheaves as total complexes.
- [`tests/stdlib/homological_test.esk`](../../../tests/stdlib/homological_test.esk):
  exhaustive small cases with independent cross-checks (CTest
  `homological_jit`, `homological_aot`).

## Scope

Complexes are finite-dimensional in each degree and matrices are dense.
Steenrod squares are mod 2; reduced powers at odd primes, the Serre spectral
sequence of a fibration and effective homology of infinite complexes are
Planned for v1.4.1 on the same representation.

## Sources

- C. A. Weibel, *An Introduction to Homological Algebra* (1994), 1.1–1.5, 2.4,
  3.1–3.6, 5.4–5.7.
- J. McCleary, *A User's Guide to Spectral Sequences*, 2nd ed. (2001), 2.2–2.4.
- A. Hatcher, *Algebraic Topology* (2002), 2.1–2.2, 3.1–3.2, 3.E, 4.L.
- K. S. Brown, *Cohomology of Groups*, GTM 87 (1982), I.5, III.1.
- N. E. Steenrod, "Products of cocycles and extensions of mappings", Ann. of
  Math. 48 (1947); A. M. Medina-Mardones, "New formulas for cup-i products
  and fast computation of Steenrod squares", Comput. Geom. 109 (2023).
- P. Deligne, "Théorie de Hodge III", Publ. IHES 44 (1974), 5.3.
- H. Cohen, *A Course in Computational Algebraic Number Theory*, GTM 138
  (1993), 2.4.
- nLab: [chain complex](https://ncatlab.org/nlab/show/chain+complex),
  [mapping cone](https://ncatlab.org/nlab/show/mapping+cone),
  [spectral sequence](https://ncatlab.org/nlab/show/spectral+sequence),
  [Steenrod square](https://ncatlab.org/nlab/show/Steenrod+square),
  [group cohomology](https://ncatlab.org/nlab/show/group+cohomology),
  [Ext](https://ncatlab.org/nlab/show/Ext).
