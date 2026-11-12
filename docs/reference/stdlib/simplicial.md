---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/simplicial.esk
  - lib/core/simplicial/sset.esk
  - lib/core/simplicial/group.esk
  - lib/core/simplicial/nerve.esk
  - lib/core/simplicial/kan.esk
  - tests/stdlib/simplicial_test.esk
---

# `core.simplicial` — simplicial sets, nerves and Kan complexes

`(require core.simplicial)` loads one library for simplicial sets as exact
data: simplices in Eilenberg–Zilber normal form, simplicial sets given by
their nondegenerate simplices and faces, Delta-complexes and simplicial
complexes, products, joins, cones and suspensions, finite groups and crossed
modules, nerves of finite categories, posets, groups and crossed modules,
McCord's finite models, horns and their fillers, the Kan and quasi-category
conditions, Kan's combinatorial homotopy groups, and the enumeration of
simplicial maps `X -> Y`.

Every quantifier ranges over a finite set: "every horn has a filler" is a
check over every compatible horn, a homotopy group is a computed quotient,
a mapping space is an enumeration. Homology, cohomology, cup products and
Steenrod squares of these simplicial sets are in
[`core.homological`](homological.md). Design record:
[ADR-0037](../../design/adr/0037-simplicial-and-homological-algebra.md).

| Module | Contents |
|---|---|
| [`core.simplicial`](../../../lib/core/simplicial.esk) | loads all of the modules below |
| [`core.simplicial.sset`](../../../lib/core/simplicial/sset.esk) | simplices, simplicial sets, operators, Delta-complexes, simplicial complexes, standard simplices, horns, spheres, lens spaces, `RP^n`, skeleta, products, joins, cones, suspensions |
| [`core.simplicial.group`](../../../lib/core/simplicial/group.esk) | finite groups by tables, quotients, crossed modules, `pi_1`, `pi_2` and the k-invariant |
| [`core.simplicial.nerve`](../../../lib/core/simplicial/nerve.esk) | nerves of finite categories and posets, order complexes, face posets, barycentric subdivision, McCord models, `BG`, nerves of crossed modules |
| [`core.simplicial.kan`](../../../lib/core/simplicial/kan.esk) | horns and fillers, Kan complexes and quasi-categories, Kan's homotopy groups, simplicial maps |

The modules are not part of `(require stdlib)`; require them explicitly.
Finite categories and finite spaces are those of
[`core.sheaf.site`](sheaf.md).

## Representation

All values are ordinary Eshkol data, so they print, compare with `equal?`
and serve as hash-table keys.

| Value | Shape | Notes |
|---|---|---|
| simplex | `#(simplex eta x)` | the n-simplex `eta^* x`: `x` is a nondegenerate m-simplex (any value), `eta` the list `(eta(0) ... eta(n))` of a monotone surjection `[n] -> [m]`; nondegenerate exactly when `eta` is the identity |
| simplicial set | `#(simplicial-set name nondegenerate face max-dim cache)` | `nondegenerate`: `n -> list of nondegenerate n-simplices`; `face`: `(n i x) -> d_i x` as a simplex; `max-dim` is the top dimension or `#f` |
| finite group | `#(finite-group n table inverse identity labels)` | elements `0 .. n-1`, `table[a][b] = a b` |
| crossed module | `#(crossed-module M N boundary action)` | `boundary : M -> N`, `action : (n m) -> n . m` |

By the Eilenberg–Zilber lemma every simplex is uniquely a degeneracy of a
nondegenerate simplex, so a simplicial set is determined by its
nondegenerate simplices and their faces (Kenzo's representation, Sergeraert
and Rubio). Every face, degeneracy and simplicial operator `theta : [k] ->
[n]` is computed from those by the simplicial identities; the order of
`nondegenerate` fixes the basis of chains and cochains.

## Simplices and simplicial sets

| Procedure | Meaning |
|---|---|
| `(simplex x m)` | the nondegenerate m-simplex `x` |
| `simplex-degree` `simplex-base` `simplex-base-degree` `simplex-surjection` | parts |
| `simplex-nondegenerate?` `simplex-degenerate?` `simplex=?` | tests |
| `(make-simplicial-set name nondegenerate face [max-dim])` | a simplicial set; `face` may return a bare value for a nondegenerate face |
| `(simplicial-set-from-all name all face degeneracy [max-dim])` | from every simplex with faces and degeneracies on values; `v` is degenerate when `v = s_j d_j v` for some `j` |
| `simplicial-set-nondegenerate` `simplicial-set-count` `simplicial-set-index` `simplicial-set-nondegenerate-ref` | the basis in degree n, its size, the index of a nondegenerate simplex |
| `(simplicial-set-face X i s)` `(simplicial-set-degeneracy X j s)` `(simplicial-set-faces X s)` | `d_i s`, `s_j s`, all faces |
| `(simplicial-set-operator X s theta)` | `theta^* s` for a monotone `theta` given as a list |
| `(simplicial-set-restrict X s vertices)` | the face spanned by an increasing vertex list |
| `(simplicial-set-front-face X s p)` `(simplicial-set-back-face X s q)` | the Alexander–Whitney faces `[0..p]` and `[n-q..n]` |
| `(simplicial-set-simplices X n)` | every n-simplex, degenerate ones included |
| `(simplicial-set-identity-violations X nmax)` | failures of `d_i d_j = d_{j-1} d_i` on nondegenerate simplices |
| `(simplicial-set-euler-characteristic X nmax)` | alternating count of nondegenerate simplices |
| `monotone-surjections` `monotone-injections` `ordered-subsets` | the operators of the simplex category |

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

## Constructions

| Procedure | Meaning |
|---|---|
| `(make-delta-complex vertex-count faces-1 faces-2 ...)` | a Delta-complex; entry `k` of `faces-n` lists `(d_0 ... d_n)` of the k-th n-simplex |
| `(delta-complex-from-tetrahedra V edge-ends tets)` | a 3-dimensional Delta-complex from tetrahedra `(v0 v1 v2 v3 e01 e02 e03 e12 e13 e23)`, triangles identified by their ordered edge triples |
| `(make-simplicial-complex facets)` | an ordered simplicial complex; simplices are increasing vertex lists |
| `(standard-simplex n)` `(simplex-boundary n)` `(horn n k)` | `Delta^n`, its boundary, `Lambda^n_k` |
| `(simplicial-sphere n)` | `S^n = Delta^n / boundary`: one vertex, one n-simplex |
| `(real-projective-space n)` | `RP^n` with one simplex in each dimension (the n-skeleton of `BZ/2`) |
| `(lens-space p q)` | `L(p,q)`: 3 vertices, `2p + 3` edges, `4p` triangles, `2p` tetrahedra (Hatcher, Example 2.43) |
| `(discrete-simplicial-set n)` `(point-simplicial-set)` | discrete sets |
| `(simplicial-skeleton X n)` | the n-skeleton |
| `(simplicial-product X Y)` | `X x Y`: pairs of n-simplices with no common degeneracy direction |
| `(simplicial-join X Y)` `(simplicial-cone X)` `(simplicial-suspension X)` | Joyal's join, `X * Delta^0`, `X * S^0` |

```scheme
(require stdlib)
(require core.simplicial)
(define (counts X n) (map (lambda (k) (simplicial-set-count X k)) (iota (+ n 1))))
;; Delta^1 x Delta^1: four vertices, five edges, two triangles (the shuffles)
(display (counts (simplicial-product (standard-simplex 1) (standard-simplex 1)) 2)) ;; => (4 5 2)
(display (counts (simplicial-product (simplicial-sphere 1) (simplicial-sphere 1)) 2)) ;; => (1 3 2)
(display (counts (lens-space 5 2) 3)) ;; => (3 13 20 10)
(display (simplicial-set-euler-characteristic (simplicial-suspension (simplex-boundary 2)) 2)) ;; => 2
```

## Finite groups and crossed modules

| Procedure | Meaning |
|---|---|
| `(make-finite-group n mul [labels])` | from a multiplication on `0 .. n-1`; identity and inverses are found |
| `group-order` `group-mul` `group-inv` `group-identity` `group-elements` `group-label` | access |
| `(cyclic-group n)` `(group-direct-product G H)` `(dihedral-group n)` `(quaternion-group)` `(symmetric-group n)` | constructors; a pair `(g, h)` is `g |H| + h`, `r^i s^j` in `D_n` is `2i + j`, `Q_8` is `2u + sign` |
| `group-axioms-violations` `group-abelian?` `group-element-order` `group-power` `group-commute?` | structure |
| `group-conjugacy-classes` `group-centralizer` `group-homomorphism?` | classes and maps |
| `(group-quotient G K)` | `(list G/K section projection)`, cosets numbered by their smallest representative |
| `(make-crossed-module M N boundary action)` | a crossed module; `(abelian-crossed-module a b c action)` is `Z/a -> Z/b`, `m -> c m`; `(group-crossed-module G)` is `1 -> G` |
| `(crossed-module-violations X)` | failures of the axioms: `boundary-not-a-homomorphism`, `peiffer`, `equivariance`, `not-an-automorphism`, `not-an-action` |
| `crossed-module-pi1` `crossed-module-pi2` `crossed-module-pi1-order` `crossed-module-pi2-order` | `coker d` (as a quotient group) and `ker d` |
| `(crossed-module-k-invariant X)` | `(list pi1 k)`: the 3-cocycle `k(a,b,c) = (s(a) . f(b,c)) f(a,bc) f(ab,c)^-1 f(a,b)^-1` with values in `ker d` (Baez–Lauda, Sec. 8) |

```scheme
(require stdlib)
(require core.simplicial)
(define twist (abelian-crossed-module 4 4 2 (lambda (n m) (if (even? n) m (modulo (- 0 m) 4)))))
(display (crossed-module-violations twist)) ;; => ()
(display (list (crossed-module-pi1-order twist) (crossed-module-pi2-order twist))) ;; => (2 2)
(define k (cadr (crossed-module-k-invariant twist)))
(display (list (k 0 0 0) (k 1 0 1) (k 1 1 1))) ;; => (0 0 2)
(display (length (group-conjugacy-classes (dihedral-group 4)))) ;; => 5
```

## Nerves and finite models

| Procedure | Meaning |
|---|---|
| `(finite-category-nerve C)` | the nerve of a finite category of `core.sheaf.site`; simplices `(x0 f1 ... fn)` |
| `(poset-nerve X)` `(order-complex n lt?)` | the nerve of a finite space: chains `x0 < ... < xn` |
| `(classifying-space G)` | `BG`, the normalized bar construction: tuples of non-identity elements, `d_i` multiplies `g_i g_{i+1}` |
| `(crossed-module-nerve X)` | the nerve of a 2-group (Duskin): edge labels in `N`, triangle labels in `M` with `g_ik = d(m_ijk) g_ij g_jk` and `m_ijl (g_ij . m_jkl) = m_ikl m_ijk` |
| `(crossed-module-simplices X n)` | every n-simplex of that nerve, by a pruned search |
| `(face-poset K nmax)` `(mccord-finite-model K nmax)` | the finite space of nondegenerate simplices ordered by faces, weakly equivalent to `|K|` for a simplicial complex `K` (McCord) |
| `(barycentric-subdivision K nmax)` | the order complex of the face poset |
| `nerve-string-normal-form` | the normal form of a string of arrows containing identities |

```scheme
(require stdlib)
(require core.simplicial)
(define (counts X n) (map (lambda (k) (simplicial-set-count X k)) (iota (+ n 1))))
(display (counts (classifying-space (quaternion-group)) 3)) ;; => (1 7 49 343)
(define K22 (abelian-crossed-module 2 1 0 (lambda (n m) m)))
(display (map (lambda (n) (length (crossed-module-simplices K22 n))) (iota 5))) ;; => (1 1 2 8 64)
(define pseudocircle (make-finite-space 4 (list (cons 0 2) (cons 1 2) (cons 0 3) (cons 1 3))))
(display (counts (poset-nerve pseudocircle) 2)) ;; => (4 4 0)
(display (counts (barycentric-subdivision (simplex-boundary 2) 1) 2)) ;; => (6 6 0)
```

## Horns, Kan complexes, homotopy groups

| Procedure | Meaning |
|---|---|
| `(simplicial-horns Y n k)` | every compatible horn `Lambda^n_k`, as lists of n faces |
| `(horn-filler-table Y n k)` `(horn-fillers Y n k horn)` | fillers, counted or listed |
| `(horn-filler-report Y n)` | per `k`: `(k horns min-fillers max-fillers)` over every compatible horn |
| `(horn-violations Y n k)` | the horns without a filler |
| `(kan-complex-through? Y nmax)` `(quasi-category-through? Y nmax)` | every horn, or every inner horn, of dimension `<= nmax` fills |
| `(basepoint-simplex v n)` `(kan-spheres Y v n)` | the degenerate basepoint and the n-simplices with every face at it |
| `(kan-homotopy-relation Y v n)` `(kan-homotopic? Y v n x y)` | Kan's relation: some `z` has `d_n z = x`, `d_{n+1} z = y`, other faces at the basepoint |
| `(kan-homotopy-classes Y v n)` | `pi_n(Y, v)` as classes of spheres, basepoint class first |
| `(kan-pi1-product Y x y)` | `d_1` of a filler of the inner horn `(y, _, x)` |
| `(simplicial-maps-fold X Y proc init)` `(simplicial-maps-count X Y)` | every simplicial map `X -> Y` for finite `X`, `proc` receiving `image : (n x) -> f(x)` |
| `(simplicial-map-order X)` | the enumeration order: vertices, then edges chosen to close triangles early, each higher simplex right after its faces |

```scheme
(require stdlib)
(require core.simplicial)
(define pseudocircle (make-finite-space 4 (list (cons 0 2) (cons 1 2) (cons 0 3) (cons 1 3))))
(define N (poset-nerve pseudocircle))
;; a quasi-category (inner horns fill uniquely) that is not a Kan complex
(display (horn-filler-report N 2)) ;; => ((0 20 0 1) (1 12 1 1) (2 20 0 1))
(display (list (quasi-category-through? N 3) (kan-complex-through? N 2))) ;; => (#t #f)
(define BS3 (classifying-space (symmetric-group 3)))
(display (length (kan-homotopy-classes BS3 '() 1))) ;; => 6
;; Map(T^2, BS_3) counts commuting pairs: |S_3| x 3 classes
(define torus (make-delta-complex 1 (list (list 0 0) (list 0 0) (list 0 0)) (list (list 0 2 1) (list 1 2 0))))
(display (simplicial-maps-count torus BS3)) ;; => 18
```

## Worked programs

- [`examples/mathematics_kan_complexes_horns.esk`](../../../examples/mathematics_kan_complexes_horns.esk):
  horn fillers of 2-group nerves, the pseudocircle as a quasi-category that
  is not Kan, Kan's `pi_1` and `pi_2` with the group law from fillers.
- [`examples/mathematics_two_groups_postnikov.esk`](../../../examples/mathematics_two_groups_postnikov.esk):
  crossed modules, their nerves and the k-invariant seen in `H^2`.
- [`examples/mathematics_group_cohomology_order_eight.esk`](../../../examples/mathematics_group_cohomology_order_eight.esk):
  classifying spaces of the groups of order eight and Dijkgraaf–Witten state
  sums as sums over `Map(M, BG)`.
- [`tests/stdlib/simplicial_test.esk`](../../../tests/stdlib/simplicial_test.esk):
  exhaustive small cases with independent cross-checks (CTest
  `simplicial_jit`, `simplicial_aot`).

## Scope

Simplicial sets are finite in each degree. `simplicial-set-simplices` lists
every simplex of a degree (`C(n, m)` degeneracies of each nondegenerate
m-simplex), so horn and homotopy computations are meant for low degrees;
chains and cochains use only nondegenerate simplices. A crossed-module
nerve enumerates all of its simplices in each degree requested. Planned
for v1.4.1: simplicial groups and the Kan loop group, minimal fibrations,
and Postnikov towers beyond 2-types, on the same representation.

## Sources

- J. P. May, *Simplicial Objects in Algebraic Topology* (1967), Sections 1–6, 21–23.
- P. G. Goerss, J. F. Jardine, *Simplicial Homotopy Theory* (1999), I.1–I.7.
- D. M. Kan, "A combinatorial definition of homotopy groups", Ann. of Math. 67 (1958).
- A. Joyal, "Quasi-categories and Kan complexes", JPAA 175 (2002); J. Lurie,
  *Higher Topos Theory* (2009), 1.1.2.
- J. Duskin, "Simplicial matrices and the nerves of weak n-categories I", TAC 9 (2002).
- R. Brown, P. J. Higgins, R. Sivera, *Nonabelian Algebraic Topology* (2011), 2.5;
  J. Baez, A. Lauda, "Higher-dimensional algebra V: 2-groups", TAC 12 (2004).
- M. C. McCord, Duke Math. J. 33 (1966); J. A. Barmak, LNM 2032 (2011).
- A. Hatcher, *Algebraic Topology* (2002), 2.1 and Example 2.43.
- F. Sergeraert, J. Rubio, "Constructive algebraic topology", Bull. Sci. Math. 126 (2002).
- nLab: [simplicial set](https://ncatlab.org/nlab/show/simplicial+set),
  [Kan complex](https://ncatlab.org/nlab/show/Kan+complex),
  [nerve](https://ncatlab.org/nlab/show/nerve),
  [crossed module](https://ncatlab.org/nlab/show/crossed+module),
  [horn](https://ncatlab.org/nlab/show/horn).
