---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/effective.esk
  - lib/core/effective/chain.esk
  - lib/core/effective/perturbation.esk
  - lib/core/effective/simplicial.esk
  - lib/core/effective/homotopy.esk
  - lib/core/effective/hypercover.esk
  - tests/stdlib/effective_test.esk
  - tests/stdlib/effective_homotopy_test.esk
  - tests/stdlib/effective_hypercover_test.esk
  - tests/stdlib/effective_pi6_s3_test.esk
---

# `core.effective` — effective homology, homotopy groups and hypercovers

`(require core.effective)` loads one library for spaces that are not of
finite type: reductions between chain complexes and the basic perturbation
lemma, reductions produced by discrete vector fields, the Eilenberg–Zilber
reduction (by its vector field and by the Alexander–Whitney, shuffle and
Shih formulas), simplicial groups and twisted cartesian products,
classifying spaces `W-bar G` with their reduction onto the bar
construction, Eilenberg–MacLane spaces `K(Z, n)` and `K(Z/m, n)` with
effective homology, cohomology classes as maps into `K(pi, n)`, the
Whitehead tower and the homotopy groups of simply connected simplicial sets
of finite type, and hypercovers of finite spaces with descent for mapping
spaces into 2-types and 3-types.

It is built on [`core.simplicial`](simplicial.md) and
[`core.homological`](homological.md) (ADR-0037) and on the finite spaces of
[`core.sheaf`](sheaf.md). Design: [ADR-0038](../../design/adr/0038-effective-homology.md).

## Modules

| Module | What it provides |
|---|---|
| `core.effective.chain` | linear combinations, locally effective complexes (a basis per degree, or none), homology of effective complexes, reductions, their composition and exact identity checks |
| `core.effective.perturbation` | `perturb-reduction` (basic perturbation lemma, with the filtration it lowers checked on every cell), `perturb-small`, `vector-field-reduction`, `collapse-matching`, `finite-complex-reduction`, words, `tensor-complex`, `tensor-reduction`, `bar-complex`, `bar-reduction` |
| `core.effective.simplicial` | `simplicial-effective-chains`, effective spaces, Eilenberg–Zilber (`eilenberg-zilber-reduction`, `eilenberg-zilber-formula-reduction`, `alexander-whitney`, `eilenberg-maclane-shuffle`, `shih-homotopy`), `product-space`, simplicial groups, `k-pi-1-group`, `classifying-group`, twisted products, `twisted-product-space`, `classifying-space-of`, `eilenberg-maclane-space` |
| `core.effective.homotopy` | `cochain-to-eilenberg-maclane`, `cocycle-twisting`, `homology-functional`, `whitehead-step`, `homotopy-groups`, `sphere-space` |
| `core.effective.hypercover` | `finite-space-hypercover`, `finite-space-cech-nerve`, `hypercover-violations`, `two-type-cardinalities`, `mapping-space-cardinality`, `postnikov-cardinalities` |

## Reductions and the perturbation lemma

A reduction `(f, g, h) : C => D` has `f g = 1`, `g f + d h + h d = 1` and
`f h = h g = h h = 0`; the maps act on cells and are extended linearly and
memoized. `reduction-violations r n big-cells small-cells` checks every
identity exactly on the given cells. `perturb-reduction r delta` returns the
reduction `(C, d + delta) => (D, d + f Sigma g)` with
`Sigma = sum (-1)^i (delta h)^i delta`; given `(key . drop)` it checks that
every term of `delta(c)` lowers `key` by at least `drop`, which is the
nilpotency hypothesis. A discrete vector field (a status procedure returning
`'critical`, `(source . target)` or `(target . source)`) gives a reduction
onto the critical cells by `vector-field-reduction`.

The Eilenberg–Zilber reduction of `Delta^1 x Delta^1`: five edges and two
triangles reduce to the tensor product, whose top cell lifts to the shuffle
combination.

```scheme
(require stdlib)
(require core.simplicial)
(require core.effective)
(define r (eilenberg-zilber-reduction (standard-simplex 1) (standard-simplex 1) #t))
(display (map (lambda (n) (effective-complex-rank (reduction-big r) n)) (list 0 1 2))) ;; => (4 5 2)
(display (map (lambda (n) (effective-complex-rank (reduction-small r) n)) (list 0 1 2))) ;; => (4 4 1)
(define top (car (effective-complex-basis (reduction-small r) 2)))
(display (word-letters top)) ;; => ((1 0 1) (1 0 1))
(display (length (reduction-violations r 2 (effective-complex-basis (reduction-big r) 2) (list top)))) ;; => 0
```

## Eilenberg–MacLane spaces

`(eilenberg-maclane-space m n)` is `K(Z, n)` for `m = 0` and `K(Z/m, n)`
otherwise, as `W-bar^(n-1) K(pi, 1)` with the reduction
`C(W-bar G) => Bar(C G) => Bar(E_G)`. Its small complex is of finite type,
so its homology is computed exactly: `K(Z, 2)` has the homology of
`CP^infinity`, and `K(Z/2, 2)` has `Z/4` in degree 4.

```scheme
(require stdlib)
(require core.effective)
(display (effective-space-homology (eilenberg-maclane-space 0 2) 4)) ;; => ((0 1 ()) (1 0 ()) (2 1 ()) (3 0 ()) (4 1 ()))
(display (effective-space-homology (eilenberg-maclane-space 2 2) 5)) ;; => ((0 1 ()) (1 0 ()) (2 0 (2)) (3 0 ()) (4 0 (4)) (5 0 (2)))
```

## Homotopy groups

`(homotopy-groups X nmax)` returns `pi_2 .. pi_nmax` of a simply connected
effective space as homology rows `(k betti torsion)`: each `pi_k` is
`H_k(X_k)` by Hurewicz, and `X_(k+1)` is the twisted product of `X_k` with
`K(pi_k, k-1)` classified by the cocycle dual to `H_k(X_k)`.

```scheme
(require stdlib)
(require core.effective)
(display (map homotopy-group-report (homotopy-groups (sphere-space 2) 3))) ;; => (Z Z)
(display (map homotopy-group-report (homotopy-groups (sphere-space 3) 4))) ;; => (0 Z Z/2)
```

Measured on Linux x64 (one core, after about 40 s of compilation):
`pi_4(S^2) = Z/2` in 0.36 s, `pi_5(S^2) = Z/2` in 5.8 s, `pi_5(S^3) = Z/2` in
2.0 s and `pi_6(S^3) = Z/12` in 146 s with 32.5 GB peak memory. A
non-cyclic `pi_k` (a product of fibres) is planned for v1.4.1.

## Hypercovers and descent (S6.3)

`(finite-space-hypercover X cover n)` builds, level by level through
coskeleta, a hypercover of a finite T_0 space by minimal opens (cones)
refining `cover`; `finite-space-cech-nerve` builds the Cech nerve by the same
recursion with connected components. On the pseudo-2-sphere, whose cover by
its two maximal opens has the pseudocircle as intersection, the Cech nerve
sees only a point while the hypercover gives the values of the space for
`BZ/2`, `K(Z/2, 2)` and the split 2-type.

```scheme
(require stdlib)
(require core.sheaf.site)
(require core.simplicial)
(require core.effective)
(define S2 (make-finite-space 6 (list (cons 0 2) (cons 1 2) (cons 0 3) (cons 1 3) (cons 2 4) (cons 3 4) (cons 2 5) (cons 3 5))))
(define cover (map (lambda (x) (finite-space-down-set S2 x)) (finite-space-maximal-points S2)))
(define H (finite-space-hypercover S2 cover 3))
(display (hypercover-level-counts H 3)) ;; => (2 4 28 196)
(display (two-type-cardinalities H 2)) ;; => (1/2 4 2)
(display (two-type-cardinalities (poset-nerve S2) 2)) ;; => (1/2 4 2)
(display (two-type-cardinalities (finite-space-cech-nerve S2 cover 3) 2)) ;; => (1/2 2 1)
```

`postnikov-cardinalities S` computes, for an effective space whose integral
homology has no 2-torsion in degrees 1 to 3, the 2-type values and the
homotopy cardinalities of `Map(S, K(B,3) x_k K(A,2))` for `A = Z/2`,
`B = Z/4` with `k = 0` and `k` the Pontryagin square, and for `A = B = Z/2`
with `k = 0` and `k = iota u iota`, by the fibration formula with the class
of `k(x)` decided on the 4-cycles of the space.

## References

F. Sergeraert, J. Rubio, "Constructive algebraic topology", Bull. Sci. Math.
126 (2002); the Kenzo program and its documentation; A. Romero, F.
Sergeraert, "Discrete vector fields and fundamental algebraic topology",
arXiv:1005.5685; J. P. May, *Simplicial Objects in Algebraic Topology*
(1967); S. Eilenberg, S. Mac Lane, "On the groups H(pi, n)" (1953–54);
J. Lurie, *Higher Topos Theory* (2009), 6.5.3–6.5.4; D. Dugger, S.
Hollander, D. Isaksen (2004); nLab, "effective homology", "homological
perturbation theory", "Whitehead tower", "hypercover".
