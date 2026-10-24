---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/sheaf.esk
  - lib/core/sheaf/site.esk
  - lib/core/sheaf/glue.esk
  - lib/core/sheaf/presheaf.esk
  - lib/core/sheaf/module.esk
  - lib/core/sheaf/cohomology.esk
  - tests/stdlib/sheaf_test.esk
---

# `core.sheaf` — sheaf theory on finite sites

`(require core.sheaf)` loads one library for Grothendieck's sheaf theory on
finite sites: finite categories and posets, sieves and Grothendieck
topologies, presheaves of sets and of abelian groups, the sheaf condition as
a gluing operation that returns either the global section or the overlaps
where gluing fails, sheafification by the plus construction, and Cech and
derived-functor cohomology with exact integer or rational coefficients.

Every quantifier in the definitions ranges over a finite set, so every
statement the library makes is decided by enumeration and exact linear
algebra; nothing is sampled or approximated. Design record:
[ADR-0033](../../design/adr/0033-core-sheaf-finite-sites.md).

| Module | Contents |
|---|---|
| [`core.sheaf`](../../../lib/core/sheaf.esk) | loads all of the modules below |
| [`core.sheaf.site`](../../../lib/core/sheaf/site.esk) | finite categories, sieves, sites and topologies, finite spaces |
| [`core.sheaf.glue`](../../../lib/core/sheaf/glue.esk) | gluing results, covers of data by pieces |
| [`core.sheaf.presheaf`](../../../lib/core/sheaf/presheaf.esk) | presheaves of sets, matching families, plus construction |
| [`core.sheaf.module`](../../../lib/core/sheaf/module.esk) | presheaves of free Z-modules and Q-vector spaces |
| [`core.sheaf.cohomology`](../../../lib/core/sheaf/cohomology.esk) | Cech cohomology, sheaves on finite spaces by stalks, derived limits |
| [`core.integer_linalg`](../../../lib/core/integer_linalg.esk) | Smith normal form and lattices; see [integer_linalg.md](integer_linalg.md) |

The modules are not part of `(require stdlib)`; require them explicitly.
`core.topos` (internal language and forcing) is built on the site, sieve and
presheaf representation documented here.

## Representation

All values are ordinary Eshkol data (tagged vectors and lists), so they print,
compare with `equal?` and serialize.

| Value | Shape | Notes |
|---|---|---|
| finite category | `#(finite-category nobj arrows compose ids labels)` | objects are `0 .. nobj-1`; `arrows` is a vector of `#(src tgt name)`; arrows are referred to by index; `compose` maps `(g f)` to the index of `g ∘ f` |
| sieve on `c` | ascending list of arrow indices with target `c` | closed under precomposition |
| site | `#(site category covering? minimal)` | `covering?` decides `J(c)`; `minimal` is `#f` or a basis giving the smallest covering sieve |
| finite space | `#(finite-space n leq)` | a finite preorder; the opens are its down-sets, as ascending point lists |
| presheaf of sets | `#(set-presheaf C sets maps)` | `sets[c]` lists the elements of `F(c)`; `maps[a]` sends element indices of `F(tgt a)` to `F(src a)` |
| family on a sieve | alist `((f . x_f) ...)` | one entry per arrow of the sieve |
| module presheaf | `#(module-presheaf C ring ranks res bases)` | `ring` is `'Z` or `'Q`; `res[a]` is the matrix of `F(a)`, `rank(src a)` rows by `rank(tgt a)` columns |
| poset module | `#(poset-module X ring ranks maps)` | a sheaf on a finite space by its stalks: `maps` is `(x y) -> F(x) -> F(y)` for `y <= x` |
| gluing result | `#(gluing glued value)` or `#(gluing obstructed (obstruction ...))` | see [Gluing](#gluing) |

Composition is written `(category-compose C g f)` for `g ∘ f` ("first `f`,
then `g`"); presheaves are contravariant, `F(g ∘ f) = F(f) F(g)`.

## Sites

### Finite categories

| Procedure | Meaning |
|---|---|
| `(make-finite-category nobj arrows compose ids [labels])` | a category from its arrow table and composition procedure |
| `(make-category-from-presentation n gens normal-form max-len)` | from generators `(src tgt name)` and a normal form on words (lists of generator indices, `(g2 g1)` = `g2 ∘ g1`) |
| `(poset-category n leq? [labels])` | the thin category of a preorder; the arrow `x -> y` exists when `(leq? x y)` |
| `(category-opposite C)` | `C^op` |
| `category-object-count` `category-arrow-count` `category-arrows` | sizes and the arrow table |
| `arrow-source` `arrow-target` `arrow-name` `category-identity` | arrow data |
| `(category-compose C g f)` / `category-composable?` | `g ∘ f`, or `#f` when `tgt f ≠ src g` |
| `category-hom` `category-arrows-into` `category-arrows-from` `category-arrow-named` | hom-sets and arrow lookup |
| `category-object-label` `category-object-of-label` | labels (the open sets, for a site of opens) |
| `category-laws-violations` / `category-laws?` | `(identity a)` and `(associativity h g f)` failures |
| `category-thin?` `category-thin-arrow` `category-meet` | thin categories: the unique arrow, and the meet (product) of two objects |
| `finite-category?` | recognizer |

### Sieves

| Procedure | Meaning |
|---|---|
| `(sieve? C c S)` | `S` is a sieve on `c` |
| `(sieve-maximal C c)` | all arrows into `c` |
| `(sieve-generated C c arrows)` | the smallest sieve containing the arrows |
| `(sieve-pullback C f S)` | `f*S = {g : f ∘ g ∈ S}` on `src f` |
| `sieve-intersection` `sieve-union` `sieve-contains?` `sieve-subset?` `sieve-domains` | set operations |
| `(sieves-on C c)` | every sieve on `c` (enumerates subsets: small categories) |

### Sites and topologies

A Grothendieck topology is given by a decision procedure `covering?`, so a
site never has to list its covering sieves; they are enumerated only when an
operation quantifies over them.

| Procedure | Meaning |
|---|---|
| `(make-site C covering? [minimal])` | a site from a covering predicate `(c S) -> boolean` |
| `(make-site-from-sieves C J)` | `J[c]` lists the covering sieves of `c` |
| `(trivial-site C)` / `(dense-site C)` | only maximal sieves cover / the dense (double negation) topology |
| `(finite-space-site X)` | the opens of a finite space, with `S` covering `U` when the domains of `S` cover `U` |
| `site-category` `site-covering?` `site-covering-sieves` `site?` | access |
| `(site-minimal-covering-sieve s c)` | the intersection of all covering sieves of `c` (itself covering) |
| `(grothendieck-topology-violations s)` | the failures of the three axioms (SGA 4 II.1.1), each named: `(maximal c)`, `(stability c S f)`, `(transitivity c S R)` |
| `grothendieck-topology?` | no violations |

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
(define broken (make-site C (lambda (c S) (and (not (= c ab)) (site-covering? site c S)))))
(display (car (grothendieck-topology-violations broken))) ;; => (maximal 3)
```

### Finite spaces

A finite preorder with its Alexandrov topology (McCord 1966; Barmak, LNM
2032). Opens are down-sets; `U_x = {y : y <= x}` is the smallest open
containing `x`.

| Procedure | Meaning |
|---|---|
| `(make-finite-space n covers)` | from pairs `(x . y)` meaning `x <= y`; the transitive closure is taken |
| `finite-space-size` `finite-space-leq?` `finite-space-lt?` `finite-space?` | the order |
| `finite-space-down-set` `finite-space-up-set` | `U_x` and the closure of `x` |
| `finite-space-open?` `finite-space-opens` | opens, in a fixed enumeration order |
| `finite-space-maximal-points` `finite-space-minimal-points` | extremal points |
| `(finite-space-product X Y)` | the product order; `(x, y)` is encoded `x * |Y| + y` |
| `(finite-space-components X U)` | connected components of `U` |
| `(finite-space-covers X U)` | all open covers of `U` by nonempty opens |
| `(finite-space-finest-cover X U)` | the cover by the `U_x`, `x ∈ U`, which refines every cover |
| `(finite-space-chains X k allowed)` | strict chains `x_0 > ... > x_k` in `allowed` |
| `sheaf-subsets` `sheaf-subsets-of-size` `sheaf-intersect-all` `sheaf-set-equal?` | finite-set helpers used throughout |

## Presheaves of sets

| Procedure | Meaning |
|---|---|
| `(make-set-presheaf C sets restrict)` | `restrict` is `(a x) -> F(a)(x)`; identities are filled in |
| `(make-set-presheaf-from-indices C sets maps)` | from index tables directly |
| `constant-set-presheaf` `(representable-presheaf C c)` | the constant presheaf; Yoneda's `Hom(-, c)` |
| `presheaf-category` `presheaf-elements` `presheaf-element-count` `presheaf-element-index` | access |
| `presheaf-restrict` `presheaf-restrict-index` | `F(a)` on elements or indices |
| `presheaf-functoriality-violations` / `presheaf-functorial?` | `(identity c)` and `(composite g f)` failures |
| `(presheaf-natural? F G eta)` | naturality of a family of index maps |
| `(matching-families F c S)` | all matching families for the sieve `S` on `c` |
| `family-incompatibilities` `matching-family?` `restriction-family` | the compatibility condition `F(g) x_f = x_{f ∘ g}` |
| `(family-amalgamations F c S fam)` | the sections restricting to the family |
| `(glue-family F c S fam)` | the unique amalgamation, or the obstruction |
| `(presheaf-sheaf-violations site F)` | `(c S not-separated)` and `(c S not-glueing k)` for every covering sieve |
| `presheaf-sheaf?` `presheaf-separated?` | no violations / no separatedness violations |
| `(presheaf-plus site F)` / `(sheafify site F)` | `F^+` and `F^{++}` |
| `(presheaf-unit-map site F Fplus)` | the unit `F -> F^+` |

On a finite site the covering sieves of `c` are closed under intersection, so
the colimit in `F^+(c) = colim_R Match(R, F)` is attained at the smallest
covering sieve and `F^+(c) = Match(R_min(c), F)` (Mac Lane–Moerdijk III.5).

```scheme
(require stdlib)
(require core.sheaf)
(define X (make-finite-space 4 (list (cons 0 2) (cons 1 2) (cons 0 3) (cons 1 3))))
(define site (finite-space-site X))
(define C (site-category site))
;; the constant presheaf with value {p, q} is not a sheaf; its sheafification
;; has 2^(number of components) sections on every open
(define F (constant-set-presheaf C (list 'p 'q)))
(define aF (sheafify site F))
(display (map (lambda (c) (presheaf-element-count aF c)) (iota 7))) ;; => (1 2 2 4 2 2 2)
(display (list (presheaf-sheaf? site F) (presheaf-sheaf? site aF))) ;; => (#f #t)
```

## Gluing

Every gluing operation returns a gluing result. `glued?`, `glued-value` and
`gluing-obstructions` read it; `gluing-success` and `gluing-failure` build it.
An obstruction is a list headed by a symbol:

| Obstruction | Meaning |
|---|---|
| `(disagree i j overlap value-i value-j)` | two local results differ on their overlap |
| `(incomplete i missing)` | a local result omits part of its piece |
| `(incompatible f g)` | a family on a sieve fails `F(g) x_f = x_{f ∘ g}` |
| `(no-amalgamation)` | the family is compatible but no global section exists |
| `(not-unique n)` | compatible, with `n > 1` global sections (`infinite` for modules) |

### Gluing data split over pieces

The same condition applies to any data split over pieces (shards, devices,
time windows): local results that agree on every pairwise overlap glue to one
global result, and a disagreement is reported with the overlap where it
happens. `(make-cover pieces overlap restrict same?)` describes the pieces:
`overlap` is `(i j) -> descriptor or #f`, `restrict` is
`(i overlap value) -> value on the overlap` and `same?` compares restricted
values. `(glue-cover cover locals assemble)` checks every overlap and calls
`assemble` on the local results only when they match;
`(cover-disagreements cover locals)` lists the failures.
`(keyed-cover domains)` and `(glue-keyed-sections domains locals)` cover the
common case of pieces that each own a list of keys, with local results as
association lists.

```scheme
(require stdlib)
(require core.sheaf)
(define domains (list (cons 'shard-1 (list 1 2 3)) (cons 'shard-2 (list 3 4))))
(define r (glue-keyed-sections domains
            (list (cons 'shard-1 (list (cons 1 'a) (cons 2 'b) (cons 3 'c)))
                  (cons 'shard-2 (list (cons 3 'c) (cons 4 'd))))))
(display (glued-value r)) ;; => ((1 . a) (2 . b) (3 . c) (4 . d))
(define bad (glue-keyed-sections domains
              (list (cons 'shard-1 (list (cons 1 'a) (cons 2 'b) (cons 3 'c)))
                    (cons 'shard-2 (list (cons 3 'z) (cons 4 'd))))))
(display (gluing-obstructions bad)) ;; => ((disagree shard-1 shard-2 (3) (c) (z)))
```

## Presheaves of abelian groups

A module presheaf assigns `R^rank(c)` to each object, `R = Z` or `Q`, and an
exact matrix to each arrow. Over `Z` the sheaf condition asks that
`F(c) -> Match(S, F)` be an isomorphism of lattices, decided with the Smith
normal form; over `Q` it is the same statement for spans.

| Procedure | Meaning |
|---|---|
| `(make-module-presheaf C ring rank res)` | `rank` is `c -> rank`, `res` is `a -> matrix` (`#f` for an identity) |
| `module-presheaf-category` `module-presheaf-ring` `module-presheaf-rank` `module-presheaf-restriction` `module-presheaf-bases` `module-presheaf?` | access |
| `module-presheaf-functoriality-violations` / `module-presheaf-functorial?` | `(shape a)`, `(identity c)`, `(composite g f)` failures |
| `constant-module-presheaf` | `R^r` with identity restrictions |
| `(module-matching-basis F S)` | a basis of `Match(S, F)` in `⊕_{f ∈ S} F(src f)` |
| `module-restriction-to-sieve` `module-sieve-offsets` | the map `F(c) -> ⊕ F(src f)` and its block layout |
| `(module-glue F c S fam)` | the unique section restricting to a family of vectors, or the obstruction |
| `module-family-incompatibilities` | the failing composites |
| `module-separated-on-sieve?` `module-sheaf-on-sieve?` | the conditions for one sieve |
| `module-separated-on-family?` `module-sheaf-on-family?` | for a family of arrows, through the sieve it generates |
| `module-presheaf-sheaf-violations` / `module-presheaf-sheaf?` | over every covering sieve |
| `(module-presheaf-plus site F)` / `module-sheafify` | `F^+` (with its bases) and `F^{++}` |
| `(module-presheaf-unit site F Fplus c)` | the matrix of `F(c) -> F^+(c)` |
| `ring-kernel` `ring-rank` `ring-solve` `ring-same-span?` `ring-cochain-cohomology` | linear algebra over the coefficient ring |

```scheme
(require stdlib)
(require core.sheaf)
(define X (make-finite-space 4 (list (cons 0 2) (cons 1 2) (cons 0 3) (cons 1 3))))
(define site (finite-space-site X))
(define C (site-category site))
(define (obj U) (category-object-of-label C U))
;; the constant sheaf Z, built from its stalks
(define Z (poset-module->module-presheaf site (constant-poset-module X)))
;; local sections 3 on {a} and 5 on {b} glue on {a, b} to the section (3, 5)
(define ab (obj (list 0 1)))
(define a->ab (category-thin-arrow C (obj (list 0)) ab))
(define b->ab (category-thin-arrow C (obj (list 1)) ab))
(define e->ab (category-thin-arrow C (obj '()) ab))
(define S (sieve-generated C ab (list a->ab b->ab)))
(display (glued-value (module-glue Z ab S (list (cons e->ab (vector)) (cons a->ab (vector 3)) (cons b->ab (vector 5)))))) ;; => #(3 5)
```

## Cohomology

### Sheaves on finite spaces by their stalks

A sheaf of abelian groups on a finite T_0 space is the same as a functor on
its poset (stalks `F(x)` and restrictions `F(x) -> F(y)` for `y <= x`). Its
cohomology is the derived limit over the poset, computed from the
cosimplicial replacement (Bousfield–Kan XI.6–7)

    C^n = ∏_{x_0 > ... > x_n} F(x_n),
    (du)(x_0 > ... > x_{n+1}) = Σ_{i <= n} (-1)^i u(… without x_i)
                              + (-1)^{n+1} F(x_n -> x_{n+1}) u(x_0 > ... > x_n),

with Smith normal forms, so integer torsion is visible. Results are lists
`((k betti torsion) ...)`.

| Procedure | Meaning |
|---|---|
| `(make-poset-module X ring rank maps)` | stalk ranks and restriction matrices |
| `(poset-local-system X m [ring])` / `(constant-poset-module X [ring])` | rank-one systems from integers `m(x, y)` |
| `poset-module-space` `poset-module-ring` `poset-module-rank` `poset-module-map` `poset-module?` | access |
| `poset-module-functoriality-violations` / `poset-module-functorial?` | `F(x -> z) = F(y -> z) F(x -> y)` |
| `(poset-module-cochain-complex F kmax [allowed])` | `(dims Ds)` of the cosimplicial replacement (on the points `allowed`) |
| `(poset-module-cohomology F kmax [allowed])` | `H^0 .. H^kmax` |
| `(poset-module-cohomology-generators F k)` | explicit cocycles generating `H^k` |
| `(poset-module-sections F D)` / `poset-module-restrict-section` | `Γ(D)` for a down-set `D` and restriction of sections |
| `(poset-module-cech-complex F cover kmax)` / `poset-module-cech-cohomology` | the Cech complex of a cover by down-sets |
| `(poset-module->module-presheaf site F)` | `U ↦ Γ(U)` on the site of opens |
| `(finite-space-euler-characteristic X kmax)` | the alternating count of chains |

### Cech cohomology

`C^k = ⊕_{i_0 < ... < i_k} F(U_{i_0} ∩ ... ∩ U_{i_k})` with the alternating
restriction differential (Godement II.5; Bredon III.4). The same engine,
`(cech-complex-from-nerve n kmax open-of rank-of restrict)`, serves module
presheaves on any thin site with meets (`module-cech-complex`,
`module-cech-cohomology`) and poset modules (intersections of point sets).
`(cohomology-generators complex k)` returns a list of `(order . cocycle)`:
order `0` for an infinite cyclic summand, `d > 1` for a `Z/d` summand.

On a Leray cover (every finite intersection acyclic) Cech cohomology equals
sheaf cohomology; on other covers it can differ, and the library computes
both.

```scheme
(require stdlib)
(require core.sheaf)
(define X (make-finite-space 4 (list (cons 0 2) (cons 1 2) (cons 0 3) (cons 1 3))))
;; constant Z and the local system with monodromy -1 around the pseudocircle
(define Z (constant-poset-module X))
(define twisted (poset-local-system X (lambda (x y) (if (and (= x 3) (= y 1)) -1 1))))
(display (poset-module-cohomology Z 2)) ;; => ((0 1 ()) (1 1 ()) (2 0 ()))
(display (poset-module-cohomology twisted 2)) ;; => ((0 0 ()) (1 0 (2)) (2 0 ()))
;; the cover by the two maximal opens is Leray: Cech gives the same groups
(define cover (list (finite-space-down-set X 2) (finite-space-down-set X 3)))
(display (poset-module-cech-cohomology twisted cover 1)) ;; => ((0 0 ()) (1 0 (2)))
;; the Z/2 class has an explicit cocycle of order 2
(display (car (car (poset-module-cohomology-generators twisted 1)))) ;; => 2
```

## Worked programs

- [`examples/mathematics_finite_site_sheafification.esk`](../../../examples/mathematics_finite_site_sheafification.esk):
  the sheaf condition on every cover of every open of the pseudocircle, the
  plus construction, the unit map, and Cech `H^1` of the presheaf and its
  sheafification.
- [`examples/mathematics_finite_spaces_sheaf_cohomology.esk`](../../../examples/mathematics_finite_spaces_sheaf_cohomology.esk):
  McCord's finite models of the circle, sphere and torus, local systems with
  Kunneth torsion, and Cech descent on Leray covers.
- [`tests/stdlib/sheaf_test.esk`](../../../tests/stdlib/sheaf_test.esk): exhaustive
  small cases with independent cross-checks (CTest `sheaf_jit`, `sheaf_aot`).

## Scope

The sites are finite: every operation enumerates objects, arrows, sieves or
chains, and `sieves-on` enumerates subsets of the arrows into an object, so
quantifying over all sieves is meant for sites with tens of arrows. Coverings
are decided by a predicate, so large sites are usable as long as only the
needed covers are examined. Coefficients are free modules over `Z` or `Q`;
Cech cohomology through `module-cech-complex` requires a thin site with meets.

## Sources

- A. Grothendieck, M. Artin, J.-L. Verdier, *Théorie des topos et cohomologie
  étale des schémas* (SGA 4), Exposés I–V, LNM 269/270 (1972).
- R. Godement, *Topologie algébrique et théorie des faisceaux* (1958).
- S. Mac Lane, I. Moerdijk, *Sheaves in Geometry and Logic* (1992), I–III.
- G. E. Bredon, *Sheaf Theory*, 2nd ed., GTM 170 (1997), II–III.
- B. Iversen, *Cohomology of Sheaves* (1986), II–III.
- A. K. Bousfield, D. M. Kan, *Homotopy Limits, Completions and
  Localizations*, LNM 304 (1972), XI.
- M. C. McCord, "Singular homology groups and homotopy groups of finite
  topological spaces", Duke Math. J. 33 (1966); J. A. Barmak, *Algebraic
  Topology of Finite Topological Spaces and Applications*, LNM 2032 (2011).
