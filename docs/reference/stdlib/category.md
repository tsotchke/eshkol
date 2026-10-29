---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/category.esk
  - lib/core/category/finite.esk
  - lib/core/category/functor.esk
  - lib/core/category/set.esk
  - lib/core/category/linear.esk
  - lib/core/category/descent.esk
  - tests/stdlib/category_test.esk
  - tests/stdlib/category_s62_test.esk
  - examples/mathematics_finite_categories_kan_extensions.esk
  - examples/mathematics_sheafification_descent_kan.esk
  - examples/mathematics_homotopy_colimits_grothendieck.esk
---

# `core.category` — finite category theory

**Source:** [category.esk](../../../lib/core/category.esk) and the five modules in
[lib/core/category/](../../../lib/core/category/).
**Require:** `(require core.category)` for everything, or one layer:
`core.category.finite`, `core.category.functor`, `core.category.set`,
`core.category.linear`, `core.category.descent` (each loads the layers it
builds on). Not part of `(require stdlib)`.
**Design record:** [ADR-0036](../../design/adr/0036-core-category-finite-categories.md).

`core.category` computes with finite categories as data: functors and natural
transformations checked by their laws, functor categories, comma and slice
categories, limits and colimits, ends and coends, the Yoneda lemma,
categories of elements and Grothendieck constructions, left and right Kan
extensions by the pointwise and the (co)end formulas, adjunctions with both
triangle identities, monads, the nerve and its cohomology, derived limits,
and descent and sheafification on the sites of [`core.sheaf`](sheaf.md).
Every law is a check that returns its failures, so a statement that does not
hold names its witness.

Sources: S. Mac Lane, *Categories for the Working Mathematician*, 2nd ed.
(1998) (CWM); E. Riehl, *Category Theory in Context* (2016); G. M. Kelly,
*Basic Concepts of Enriched Category Theory* (1982; TAC Reprints 10, 2005)
for ends, coends and Kan extensions as weighted (co)limits; S. Mac Lane,
I. Moerdijk, *Sheaves in Geometry and Logic* (1992) for sieves, the plus
construction and a -| i; A. K. Bousfield, D. M. Kan, LNM 304 (1972) and
R. W. Thomason (1979) for derived limits and homotopy colimits. Each
section below names the nLab page of its construction.

## Representation

There is one category type, core.sheaf.site's
`#(finite-category nobj arrows compose ids labels)`: objects `0 .. nobj-1`,
arrows named by their index, `(category-compose C g f)` = g o f. Every
constructor below returns that value, so a category from `core.category` is
a site's category, a topos's base, or a shape for a limit without
conversion.

| Value | Representation |
|---|---|
| functor | `#(functor C D objects arrows)`: vectors object -> object, arrow -> arrow |
| natural transformation | vector, object c -> the component arrow F c -> G c |
| Set-valued functor | `#(set-functor C sets maps)`: `sets[c]` the element list, `maps[a]` the index map X(source) -> X(target) |
| presheaf on C | core.sheaf's `set-presheaf`; `presheaf->set-functor` reads it as a set-functor on C^op, sharing the data |
| transformation of set-functors | vector, object c -> vector of indices X(c) -> Y(c) |
| limit in Set | list of families, each a vector c -> index |
| colimit in Set | `#(set-colimit ...)` with `set-colimit-size`, `set-colimit-class co c i` |
| comma (F \| G) | a category with object labels `(a b u)`, u : F a -> G b, arrow names `(f g)` |
| product C x D | object (x, y) is `x * |ob D| + y`, labelled `(x y)`; arrow `(f g)` is `f * |ar D| + g` |
| category diagram | `#(category-diagram C categories functors)`, a strict functor C -> Cat |
| adjunction | `#(adjunction L R unit counit)`, L -\| R |
| monad | `#(monad C T unit multiplication)` |
| linear diagram | core.sheaf's `module-presheaf` (ranks over Z or Q, restriction matrices) |

## Categories, functors, adjunctions

nLab: [category](https://ncatlab.org/nlab/show/category),
[free category](https://ncatlab.org/nlab/show/free+category),
[functor category](https://ncatlab.org/nlab/show/functor+category),
[comma category](https://ncatlab.org/nlab/show/comma+category),
[adjunction](https://ncatlab.org/nlab/show/adjunction),
[monad](https://ncatlab.org/nlab/show/monad),
[nerve](https://ncatlab.org/nlab/show/nerve). CWM I-II, IV, VI; Riehl 1, 4, 5.

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

`right-adjoint L` takes the terminal object of each comma category (L | d)
(Riehl 4.6, CWM IV.1 Theorem 2) and returns `#f` when one is missing;
`left-adjoint` dually. `adjunction-violations` checks the functor laws, the
naturality of unit and counit and both triangle identities;
`adjunction-hom-bijection-violations` checks D(Lc, d) ~ C(c, Rd) by counting
and by the two explicit maps. The last line is the integral cohomology of the
nerve of B(Z/2), Quillen's classifying space BZ/2.

**`core.category.finite`:** `discrete-category` `terminal-category`
`free-category` `walking-arrow-category` `parallel-pair-category`
`span-category` `cospan-category` `commutative-square-category`
`monoid-category` `cyclic-group-category` `category-product`
`category-product-object` `category-product-arrow` `category-coproduct`
`category-full-subcategory` `category-full-subcategory-inclusion`
`category-arrow-list` `category-object-list` `category-identity?`
`category-isomorphism?` `category-inverse` `category-groupoid?`
`category-initial-objects` `category-terminal-objects` `category-nerve`
`category-nerve-coboundary` `category-nerve-cohomology`.

**`core.category.functor`:** `make-functor` `functor-from-procedures`
`category-functor?` `functor-source` `functor-target` `functor-object`
`functor-arrow` `functor-objects` `functor-arrows` `functor-laws-violations`
`functor-laws?` `functor-equal?` `identity-functor` `functor-compose`
`constant-functor` `object-functor` `terminal-functor` `functor-opposite`
`inclusion-functor` `product-projections` `functors-between`
`natural-transformation-violations` `natural-transformation?`
`natural-transformations` `identity-transformation` `transformation-compose`
`whisker-left` `whisker-right` `horizontal-compose` `natural-isomorphism?`
`functor-category` `functor-category-functors` `comma-category`
`comma-domain-projection` `comma-codomain-projection` `slice-category`
`coslice-category` `make-category-diagram` `category-diagram?`
`category-diagram-base` `category-diagram-category` `category-diagram-functor`
`category-diagram-violations` `grothendieck-construction`
`grothendieck-projection` `make-adjunction` `adjunction?` `adjunction-left`
`adjunction-right` `adjunction-unit` `adjunction-counit`
`adjunction-violations` `adjunction-hom-bijection-violations` `right-adjoint`
`left-adjoint` `make-monad` `monad?` `monad-category` `monad-functor`
`monad-unit` `monad-multiplication` `adjunction-monad`
`monad-laws-violations` `monad-algebras`.

The Grothendieck construction of a category diagram P has objects `(c x)`,
x an object of P(c), and arrows `(a phi)` with phi : P(a)(x) -> x'
(SGA 1 VI; nLab [Grothendieck construction](https://ncatlab.org/nlab/show/Grothendieck+construction));
its nerve is the homotopy colimit of the nerves (Thomason). The
[homotopy-colimit example](../../../examples/mathematics_homotopy_colimits_grothendieck.esk)
computes suspensions and Borel constructions this way.

## Set-valued functors, Yoneda and Kan extensions

nLab: [limit](https://ncatlab.org/nlab/show/limit),
[colimit](https://ncatlab.org/nlab/show/colimit),
[end](https://ncatlab.org/nlab/show/end),
[Yoneda lemma](https://ncatlab.org/nlab/show/Yoneda+lemma),
[category of elements](https://ncatlab.org/nlab/show/category+of+elements),
[Kan extension](https://ncatlab.org/nlab/show/Kan+extension),
[pointwise Kan extension](https://ncatlab.org/nlab/show/pointwise+Kan+extension).
CWM III, IX, X; Riehl 2, 3, 6; Kelly 2-4.

```scheme
(require stdlib)
(require core.category.set)
;; a Z/2-set: the generator swaps x and y and fixes z
(define BZ2 (cyclic-group-category 2))
(define X (make-set-functor-from-names BZ2 (list (list 'x 'y 'z)) (list (cons 1 (vector 1 0 2)))))
(define F (terminal-functor BZ2))
(set-functor-sizes (set-lan F X))                ;; => (2)
(set-functor-sizes (set-ran F X))                ;; => (1)
(set-functor-elements (set-lan F X) 0)           ;; => ((0 0 x) (0 0 z))
(set-natural-isomorphism? (set-lan F X) (set-lan-coend F X) (set-lan-comparison F X))  ;; => #t
(set-lan-adjunction-violations F X (set-lan F X))  ;; => ()
(set-natural-transformation-count X X)           ;; => 3
(yoneda-violations X 0)                          ;; => ()
;; the pullback of p p q <- {0 1 2} and {0 1} -> q p
(define P (make-set-functor-from-names (cospan-category) (list (list 'p 'q) (list 0 1 2) (list 0 1))
                                       (list (cons (list 0) (vector 0 0 1)) (cons (list 1) (vector 1 0)))))
(list (set-limit-size P) (set-colimit-size (set-colimit P)))  ;; => (3 2)
```

Along B(Z/2) -> 1 the left Kan extension is the orbit set and the right one
the fixed points. An element of Lan_F G(d) is printed as the representative
`(c u x)` of its class: an object c, an arrow u : F c -> d and x in G(c).
`set-lan` and `set-ran` use the comma-category formulas; `set-lan-coend`
and `set-ran-end` use Lan_F G = int^c D(Fc, -) x G(c) and
Ran_F G = int_c G(c)^{D(-, Fc)}; the comparisons are natural isomorphisms.
`set-lan-adjunction-violations F G Y` checks Lan -| F^* at G and Y (unit
and counit natural, both triangle identities), and
`set-ran-adjunction-violations F Y G` checks F^* -| Ran.
`set-natural-transformations` solves Nat(X, Y) as a limit over the category of
elements of X; `set-natural-transformations-end` uses the end
int_c Y(c)^X(c). `yoneda-violations X c` compares Nat(C(c, -), X) with X(c)
by counting and through the two explicit maps.

**`core.category.set`:** `make-set-functor` `make-set-functor-from-indices`
`make-set-functor-from-names` `set-functor?` `set-functor-category`
`set-functor-elements` `set-functor-element-count` `set-functor-element-index`
`set-functor-sizes` `set-functor-apply` `set-functor-apply-index`
`set-functor-functoriality-violations` `set-functor-functorial?`
`presheaf->set-functor` `set-functor->presheaf` `set-functor-restrict`
`constant-set-functor` `corepresentable-set-functor` `hom-set-functor`
`set-functor-product` `set-natural?` `set-natural-transformations`
`set-natural-transformation-count` `set-identity-transformation`
`set-transformation-compose` `set-transformation-identity?`
`set-transformation-bijective?` `set-natural-isomorphism?` `set-limit`
`set-limit-size` `set-limit-element-values` `set-cone-count` `set-colimit`
`set-colimit-size` `set-colimit-class` `set-colimit-representatives` `set-end`
`set-coend` `set-coend-class` `set-natural-transformations-end`
`yoneda-element` `yoneda-transformation` `yoneda-violations`
`presheaf-yoneda-violations` `elements-category` `elements-projection`
`presheaf-elements-category` `presheaf-elements-projection`
`discrete-opfibration?` `discrete-fibration?` `set-lan` `set-ran`
`set-lan-coend` `set-ran-end` `set-lan-comparison` `set-ran-comparison`
`set-lan-unit` `set-lan-counit` `set-lan-map` `set-ran-unit` `set-ran-counit`
`set-ran-map` `set-restrict-map` `set-lan-adjunction-violations`
`set-ran-adjunction-violations`.

Ends and coends take a bifunctor H : C^op x C -> Set as a set-functor on
`(category-product (category-opposite C) C)`; `hom-set-functor` is C(-, -).

## Linear diagrams and derived limits

nLab: [Ab](https://ncatlab.org/nlab/show/Ab),
[derived functor](https://ncatlab.org/nlab/show/derived+functor),
[group cohomology](https://ncatlab.org/nlab/show/group+cohomology).
CWM V.2; C. Weibel, *An Introduction to Homological Algebra* (1994), 2.6 and 6.

```scheme
(require stdlib)
(require core.category.linear)
;; Z with the sign action of Z/2: no invariants, coinvariants Z/2
(define BZ2 (cyclic-group-category 2))
(define sign (make-module-presheaf BZ2 'Z (lambda (c) 1) (lambda (a) (if (= a 1) (vector (vector -1)) #f))))
(define K (module-presheaf-colimit sign))
(list (module-limit-rank (module-presheaf-limit sign)) (module-colimit-rank K) (module-colimit-torsion K))  ;; => (0 0 (2))
(module-presheaf-derived-limits sign 3)          ;; => ((0 0 ()) (1 0 (2)) (2 0 ()) (3 0 (2)))
```

A diagram is a core.sheaf `module-presheaf` on C (limits over C^op); a
covariant diagram J -> Mod is the module presheaf on J^op with the same
matrices. `module-presheaf-derived-limits F kmax` is the cohomology of the
normalized cosimplicial replacement, so over B(G) it is group cohomology
H^*(G; M) (the second line: H^*(Z/2; Z^-)) and over a poset it is core.sheaf's
derived limit. `module-presheaf-map-hypercohomology F G phi kmax` takes a
two-term complex F -> G.

**`core.category.linear`:** `module-presheaf-limit` `module-limit-rank`
`module-limit-basis` `module-limit-component` `module-presheaf-colimit`
`module-colimit-rank` `module-colimit-torsion`
`module-presheaf-limit-relations` `module-presheaf-colimit-relations`
`module-presheaf-restrict` `poset-module->point-presheaf`
`module-presheaf-cosimplicial-complex` `module-presheaf-derived-limits`
`module-presheaf-map-hypercohomology` `module-cohomology-mod-p-dimensions`.

## Descent and sheafification

nLab: [sieve](https://ncatlab.org/nlab/show/sieve),
[sheafification](https://ncatlab.org/nlab/show/sheafification),
[plus construction on presheaves](https://ncatlab.org/nlab/show/plus+construction+on+presheaves),
[reflective subcategory](https://ncatlab.org/nlab/show/reflective+subcategory),
[idempotent monad](https://ncatlab.org/nlab/show/idempotent+monad),
[Cech cover](https://ncatlab.org/nlab/show/Cech+cover),
[descent](https://ncatlab.org/nlab/show/descent).
Mac Lane-Moerdijk I.4, III.4-III.5, V.3; Johnstone, *Elephant*, A4.4 and C2.

```scheme
(require stdlib)
(require core.category.descent)
;; the site of opens of the pseudocircle (7 opens, the empty one first)
(define X (make-finite-space 4 (list (cons 0 2) (cons 1 2) (cons 0 3) (cons 1 3))))
(define site (finite-space-site X))
(define C (site-category site))
(define F (constant-set-presheaf C (list 'p 'q)))
(define aF (sheafify site F))
(map (lambda (c) (presheaf-element-count aF c)) (iota 7))  ;; => (1 2 2 4 2 2 2)
(sheafification-adjunction-violations site F aF) ;; => ()
(sheafification-monad-violations site F)         ;; => ()
(define ab (category-object-of-label C (list 0 1)))
(define S (site-minimal-covering-sieve site ab))
(list (length (sieve-matching-limit F ab S)) (sieve-descent-violations F ab S))  ;; => (2 ())
;; Cech cohomology of the cover by the two maximal opens, as lim^n over the nerve category of the cover
(define M (poset-module->point-presheaf (constant-poset-module X)))
(define cover (list (finite-space-down-set X 2) (finite-space-down-set X 3)))
(module-presheaf-derived-limits (cdr (finite-space-cover-sections-presheaf M cover)) 1)  ;; => ((0 1 ()) (1 1 ()))
```

`sieve-matching-limit F c S` is Match(S, F) computed as the limit of F over
the category of the sieve (the full subcategory of C/c on S); it is the value
at S of the right Kan extension of F along the Yoneda embedding, and on a
thin site `presheaf-ran-along-inclusion` gives it as Ran along the inclusion
of the sieve's objects. `cech-equalizer` is the equalizer form for a family of
arrows. `sheafification-counit` glues twice; `sheafification-adjunction-violations
site F G` checks a -| i for a presheaf F and a sheaf G: naturality, both triangle
identities, |Nat(F, G)| = |Nat(aF, G)| and the explicit bijection
psi |-> psi eta. `sheafification-monad-violations` checks the unit,
right-unit and associativity laws of i a. For finite spaces,
`finite-space-cover-descent-category` is the Grothendieck construction of
J |-> U_J, whose derived limits give sheaf cohomology for every cover, and
`finite-space-cech-nerve-category` is the face category of the Cech nerve.
`tests/stdlib/category_s62_test.esk` uses these to reproduce the numbers of
the finite-site sheafification, finite-space cohomology, descent and
2-descent examples.

**`core.category.descent`:** `sieve-category` `sieve-matching-limit`
`sieve-descent-violations` `presheaf-ran-along-inclusion` `cech-equalizer`
`cech-descent-violations` `presheaf-sheaf-violations-by-limit`
`presheaf-plus-map` `sheafification-unit` `sheafify-map`
`sheafification-counit` `sheafification-adjunction-violations`
`sheafification-monad-violations` `category-components`
`cover-nerve-category` `cech-derived-limits`
`finite-space-cover-descent-category` `finite-space-cech-nerve-category`
`cech-nerve-cell-counts` `module-sieve-limit` `module-sheaf-at-by-limit`
`module-sieve-unit-shape` `finite-space-cover-sections-presheaf`
`finite-space-sections`.

## Sizes

Functors, natural transformations between functors, functor categories and
the sieves of a site are enumerated, so they suit small categories. Limits,
Kan extensions and natural transformations of Set-valued functors use
propagating searches whose cost follows the size of the answer. Derived
limits build the cosimplicial replacement up to the requested degree, so
their cost grows with the number of chains of arrows. Mapping spaces into
2-groups with a nontrivial action or k-invariant are Planned for v1.4.1 with
the simplicial layer (ADR-0037); limits and colimits over a `core.ring`
carrier are Planned for v1.4.1.
