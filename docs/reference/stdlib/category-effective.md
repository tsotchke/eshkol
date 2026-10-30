---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/category/effective.esk
  - lib/core/category/presented.esk
  - lib/core/category/standard.esk
  - lib/core/category/effective_kan.esk
  - tests/stdlib/category_effective_test.esk
  - tests/nlab/nlab_category_effective_test.esk
  - tests/formal/coq/StringRewriting.v
---

# Effective and finitely presented categories

**Source:** [effective.esk](../../../lib/core/category/effective.esk),
[presented.esk](../../../lib/core/category/presented.esk),
[standard.esk](../../../lib/core/category/standard.esk),
[effective_kan.esk](../../../lib/core/category/effective_kan.esk).
**Require:** `core.category.effective`, `core.category.presented`,
`core.category.standard`, `core.category.effective_kan` (each loads the ones
it builds on). They are not loaded by `(require core.category)` and are not
part of `(require stdlib)`.
**Design:** [ADR-0054](../../design/adr/0054-effective-and-presented-categories.md).
**Finite categories:** [category.md](category.md) (ADR-0036).

These modules extend [`core.category`](category.md) from finite categories to
infinite ones. A category is *effectively given*: its objects come from an
enumerator or a membership test, each hom-set is produced on demand (a finite
list, or a lazy enumerator when it is infinite), and composition, identities
and equality of arrows are procedures. This is the model of
`core.effective`'s locally effective chain complexes applied to categories. A
finite category is a special case, and the conversion goes both ways.

```scheme
(require core.category.presented)
(require core.category.standard)

;; Z^2 as a one-object category presented by generators and a relator
(define Z2 (presented-group '(a b) '((a b (inv a) (inv b)))))
(presented-category-status Z2)                                  ; => complete
(map (lambda (n) (length (enumerator-stage (presented-hom Z2 0 0) n))) '(0 1 2 3))
;; => (1 4 8 12): the normal forms of length n

;; <a, b | aba = bab> has no finite complete system on {a, b}
(define KN (presented-monoid '(a b) '(((a b a) (b a b))) 40 400))
(presented-equal KN '(a b a a) '(b a b a))                       ; => equal
(presented-equal KN '(a b) '(b a))                               ; => undecided

;; Delta: products exist only with a terminal factor
(ordinal-product 'delta 0 2)       ; => (2 (2 0 (0 0 0)) (2 2 (0 1 2)))
(ordinal-product 'delta 1 1)
;; refused: limit-does-not-exist ... (candidate 3 test-object 1 cones 9 maps-into-candidate 10)
```

## What is exact and what is a sample

A law about an infinite category cannot be checked by listing it. Every
`*-violations` procedure therefore takes the objects and arrows to check,
checks exactly those, and returns the list of failures (empty when every
law holds).

When the category is **presented** by generators and relations, the checks
are complete:

- a functor out of it is determined by the images of the generators, and it
  is a functor exactly when those images satisfy the relations;
- a transformation is natural exactly when it is natural on the generators.

Those checks have `generator` in their names, and the suites state which
range every other check covers.

## Refusals

A refusal is an R7RS error object. Its message is the name of the condition,
and its irritants name the operation and the witness. A refusal never
returns a substitute value. `category-refusal?`, `category-refusal-kind` and
`category-refusal-irritants` read refusals, and `category-refuse` raises one.

| Kind | When |
|---|---|
| `enumeration-infinite` | listing an enumerator without a finite bound |
| `category-not-finite` | converting a category with infinitely many objects or an infinite hom-set |
| `objects-not-enumerated` | asking for the objects of a category that has only a membership test |
| `not-composable` | composing arrows whose endpoints do not match |
| `composite-outside-hom` | a composite or identity that is missing from the hom-set enumeration |
| `nerve-not-enumerable` | listing the nondegenerate simplices of the nerve of an infinite category |
| `presentation-undecided-within-budget` | normal forms needed, but Knuth-Bendix completion stopped at its budget |
| `limit-does-not-exist` | the (co)limit provably does not exist; the witness is in the irritants |
| `limit-not-computable` | the carrier or category has no algorithm here (for example Mat over Z/6) |
| `comma-category-not-finite` | a Kan extension whose comma category is not known to be finite |
| `hom-not-enumerated` | a matrix hom-set over a carrier without an element enumeration |

## `core.category.effective`

**Enumerators.** `make-enumerator stage [bound]` builds an enumerator: stage
`n` lists the items of size `n`. With a bound `N`, every stage `>= N` is empty
and the enumerator is finite. A plain list is also accepted wherever an
enumerator is expected.

| Procedure | Meaning |
|---|---|
| `make-enumerator` `list->enumerator` `as-enumerator` `enumerator?` | construction and coercion |
| `enumerator-stage` `enumerator-bound` `enumerator-finite?` | stage n; the bound (or #f) |
| `enumerator-up-to` `enumerator-take` `enumerator-find` | the items of stages 0..n; at most k items; the first item with a property |
| `enumerator->list` `enumerator-count` | every item (refused when infinite) |
| `enumerator-map` `enumerator-filter` `enumerator-append` | stage-wise maps |
| `enumerator-product` `enumerator-concat` | pairs, by total stage; a dependent union |
| `naturals-enumerator` `integers-enumerator` | 0, 1, 2, ... and 0, 1, -1, 2, -2, ... |

**Categories.** `make-effective-category name objects object? hom compose
identity source target arrow=? [props]`. `props` is an association list. It
may declare `'generators`, `'relations` (pairs of composable lists of
generators, in composition order), `'thin`, and `'finite-category` (the
`core.sheaf.site` value the category came from).

| Procedure | Meaning |
|---|---|
| `effective-category?` `effective-category-name` `effective-category-objects` `effective-category-object?` | accessors |
| `effective-category-hom` `effective-category-hom-list` `effective-category-hom-up-to` | hom(a, b) as an enumerator, as a list (refused when infinite), or its stages 0..n |
| `effective-category-compose` `effective-category-compose-list` `effective-category-identity` `effective-category-identity?` | composition (refused unless the arrows compose) and identities |
| `effective-category-source` `effective-category-target` `effective-category-arrow=?` | endpoints and equality of arrows |
| `effective-category-property` `effective-category-with-properties` | declared structure |
| `effective-category-object-list` `effective-category-arrows-up-to` `effective-category-finite?` | listings |
| `effective-category-laws-violations E objs n` | identity, endpoint and associativity laws on the arrows of stages 0..n between objs |
| `finite-category->effective` `effective->finite-category` | conversion both ways (the second numbers arrows hom by hom; arrow names are the effective arrows, object labels the effective objects) |
| `effective-opposite` `effective-full-subcategory` `effective-product-category` | constructions |

**Functors, transformations and adjunctions.** `make-effective-functor C D
object arrow [name [props]]`. A transformation is a procedure from objects to
components. `make-effective-adjunction F G unit counit` builds `F -| G`, with
`unit c : c -> G F c` and `counit d : F G d -> d`.

| Procedure | Meaning |
|---|---|
| `effective-functor?` `effective-functor-name` `effective-functor-source` `effective-functor-target` `effective-functor-object` `effective-functor-arrow` `effective-functor-property` | accessors (a functor may declare `'fibre` and `'cofibre` for Kan extensions) |
| `effective-identity-functor` `effective-functor-compose` `effective-functor-opposite` | constructions |
| `effective-functor-violations F objs arrows` | identities, endpoints and composites on the given objects and arrows |
| `effective-functor-generator-violations F [objs]` | complete check out of a presented source: identities, generator endpoints, every relation |
| `finite-functor->effective` `effective-functor->finite` | conversion of functors between finite categories |
| `effective-transformation-violations F G alpha objs arrows` `effective-transformation-generator-violations F G alpha [objs]` | naturality on given arrows, or on the generators (complete) |
| `effective-identity-transformation` `effective-transformation-compose` `effective-whisker-left` `effective-whisker-right` | the 2-categorical operations |
| `effective-adjunction?` `effective-adjunction-left` `effective-adjunction-right` `effective-adjunction-unit` `effective-adjunction-counit` | accessors |
| `effective-adjunction-violations A objsC objsD arrowsC arrowsD` | both triangle identities at the given objects, naturality of unit and counit on the given arrows |
| `effective-adjunction-transpose` `effective-adjunction-cotranspose` `effective-adjunction-hom-bijection-violations A pairs n` | the bijection D(F c, d) ~ C(c, G d): both round trips on hom stages 0..n, and every transpose inside its hom-set when that hom-set is finite |

**Set-valued functors and nerves.**

| Procedure | Meaning |
|---|---|
| `make-effective-set-functor C object arrow` `effective-set-functor?` `effective-set-functor-category` `effective-set-functor-object` `effective-set-functor-arrow` `effective-set-functor-violations` | covariant functors to finite sets |
| `effective-category-nerve E` | a `core.simplicial.sset` simplicial set of infinite type; its faces and degeneracies are computed for every category, while listing its nondegenerate simplices needs a finite category |
| `effective-nerve-simplices E n bound` | the nondegenerate n-simplices built from hom stages 0..bound |
| `effective-nerve-boundary E n x` | the normalized boundary, as `core.effective.chain` linear combinations |
| `effective-nerve-chain-data E` | `(basis d)` for `core.effective.chain`'s `make-effective-complex`: basis is #f in degrees that are not enumerable (a locally effective complex) |

## `core.category.presented`

Words list generator indices or names in composition order: `(g2 g1)` is
`g2 o g1`, the convention of `core.sheaf.site`'s
`make-category-from-presentation`. A name that is an exact integer is read as
an index.

Completion uses the shortlex order: shorter words come first, and words of
the same length compare lexicographically by generator index. It reduces
both sides of each equation, orients the pair, inter-reduces the rules,
queues critical pairs, and stops at the budget. The default budget is 256
rules and 4096 equations.

| Procedure | Meaning |
|---|---|
| `make-presentation nobj gens relations [labels]` `presentation?` `presentation-object-count` `presentation-generators` `presentation-generator-count` `presentation-relations` `presentation-generator-index` `presentation-generator-name` `word-source` `word-target` | presentations (gens are `(source target name)`; relations `(lhs rhs)`; malformed input is an error) |
| `shortlex<?` `critical-pairs r1 r2` `knuth-bendix equations max-rules max-steps` | the completion engine |
| `rewriting-system?` `rewriting-system-status` `rewriting-system-complete?` `rewriting-system-rules` `rewriting-system-steps` `rewriting-system-pending` `rewrite-normal-form` `word-reducible?` | rewriting systems |
| `present-category P [max-rules [max-steps]]` `presented-category?` `presented-category-presentation` `presented-category-system` `presented-category-status` `presented-category-complete?` | the presented category |
| `presented-normal-form` `presented-word` `presented-equal PC u v [object]` | `'equal`, `'distinct` or `'undecided` (see below) |
| `presented-arrow` `presented-generator-arrow` | arrows `(source target normal-word)` |
| `presented-hom` `presented-hom-finite?` `presented-hom-count` | hom-sets by normal form and length; finiteness decided by the automaton of irreducible words |
| `presented->effective` `presented->finite-category` `presented-site-category` | the effective view; a finite category (refused when a hom-set is infinite); the same category through `core.sheaf.site` with the completed normal form |
| `presented-monoid names relations [budget]` `presented-group names relators [budget]` `free-monoid` `free-category-on-quiver` | monoids, groups (inverse of `x` is named `(inv x)`), free monoids and free categories |
| `presented-functor PC D objects images` `presented-functor-violations` `presented-transformation-violations` | functors out of a presented category and their complete checks |

`presented-equal` answers:

- `'equal`: whenever the reduced forms coincide. This is sound with any
  rule set, because every rule is a consequence of the relations.
- `'distinct`: with a complete system; for different endpoints; or for words
  of different lengths when every relation preserves length.
- `'undecided`: otherwise.

The theorems behind these answers are machine-checked in
[`tests/formal/coq/StringRewriting.v`](../../../tests/formal/coq/StringRewriting.v):
Newman's lemma, uniqueness of normal forms, soundness of rewriting and of
every completion move, and the normal-form decision procedure. All are
proved in Rocq 9 with no axioms.

## `core.category.standard`

| Procedure | Category |
|---|---|
| `poset-effective-category` `naturals-poset` `integers-poset` `rationals-poset` `divisibility-poset` `rationals-enumerator` | countable posets; arrows `(a . b)`; meets, joins, top and bottom declared |
| `poset-meet` `poset-join` `poset-top` `poset-bottom` | products, coproducts, terminal and initial objects of a poset (refused by name when absent) |
| `floor-ceiling-adjunctions` | `ceiling -| (Z -> Q) -| floor`, as a pair of adjunctions |
| `monoid-effective-category` `naturals-additive-monoid` `integers-additive-group` | one-object categories; arrows are elements |
| `simplex-category` `finord-category` `finset-category` `monotone-maps` | Delta (`[m] = {0..m}`), FinOrd and the skeleton of FinSet (`n = {0..n-1}`); arrows `(m n images)` |
| `delta-face` `delta-degeneracy` `delta-cosimplicial-identity-violations N` | faces, degeneracies, and every cosimplicial identity up to `[N]` |
| `delta-normal-form` `delta-from-normal-form` | the unique factorization `d^(i1)..d^(is) s^(j1)..s^(jt)` |
| `delta-truncated-presentation N` `delta-truncated-functor` | Delta_{<=N} by faces, degeneracies and identities, and its functor to Delta |
| `ordinal-terminal` `ordinal-initial` `ordinal-product` `ordinal-coproduct` | (co)products in Delta (`'delta`) and FinOrd (`'finord`): constructed or refused with a counting witness |
| `pointed-finset-category` `pointed-set-adjunction` | pointed finite sets and free -\| forgetful from FinSet |
| `finset-terminal` `finset-initial` `finset-product` `finset-coproduct` `finset-equalizer` `finset-coequalizer` `finset-pullback` `finset-pushout` | finite limits and colimits in FinSet, as `(apex leg ...)` |
| `finset-limit J objs arrs` `finset-colimit J objs arrs` `finset-cones` `finset-factor` | any finite diagram (J a `core.sheaf.site` category), cones from a test object, and factorizations |
| `matrix-category R` `matrix-category-ring` `make-matrix-arrow` `matrix-arrow-matrix` `matrix-identity-arrow` | Mat_R over a `core.ring` carrier; arrow `(n m M)` is an m x n matrix |
| `mat-kernel` `mat-cokernel` `mat-equalizer` `mat-coequalizer` `mat-product` `mat-coproduct` `mat-pullback` `mat-pushout` `mat-terminal` `mat-initial` `mat-limit` `mat-colimit` `mat-rank` | (co)limits for R = Z or a field |
| `mat-factor-through-mono` `mat-factor-through-epi` | the unique factorization through a kernel or cokernel, or #f |

Over Z, kernels are saturated lattices, computed with the Smith normal form
through `core.integer_linalg`. The cokernel in Mat_Z is `Z^m / sat(im A)`:
a map into a free module that kills `im A` also kills its saturation, so
this quotient is the colimit among free modules. The module cokernel, with
its torsion, is `core.category.linear`'s.

## `core.category.effective_kan`

| Procedure | Meaning |
|---|---|
| `effective-comma-objects F d` `effective-cocomma-objects F d` | the objects of (F \| d) and (d \| F) |
| `effective-lan F G` `effective-ran F G` | pointwise Kan extensions as effective set functors |
| `effective-lan-unit` `effective-ran-unit` `effective-ran-counit` `effective-restrict` | units, counits, and restriction F^* H = H o F |
| `effective-set-limit sets constraints` | compatible families of a finite diagram of finite sets |
| `truncated-simplex-category n` `simplex-inclusion-functor n` | Delta_{<=n} and its inclusion |
| `nerve-presheaf E` `simplex-presheaf n` | the nerve of a finite category and Delta[n] as presheaves on Delta |
| `truncate-presheaf X n` `skeleton Y n` `coskeleton Y n` `presheaf-unit-bijective? X n m` | truncation, `sk_n` (Lan), `cosk_n` (Ran), and whether `X_m -> (cosk_n tr_n X)_m` is a bijection |

A comma category (F | d) is finite in either of two cases:

- the source has finitely many objects;
- the functor declares `'fibre` (`d` -> a list containing every `c` with
  `D(F c, d)` nonempty), or `'cofibre` for Ran.

Each listed object is checked to be an object of the source. Completeness of
the list is part of the functor's declaration.

## Verification

- [`tests/stdlib/category_effective_test.esk`](../../../tests/stdlib/category_effective_test.esk)
  (CTest `category_effective_{jit,aot}`): 51 checks with independent
  cross-checks: closed forms, brute force, `core.category.kan`,
  `core.sheaf.site`, `core.integer_linalg`, `core.simplicial`.
- [`tests/nlab/nlab_category_effective_test.esk`](../../../tests/nlab/nlab_category_effective_test.esk)
  (CTest `nlab_category_effective_{jit,aot}`): 13 nLab claims (see
  [NLAB_CONFORMANCE.md](../../reports/NLAB_CONFORMANCE.md)).
- [`tests/formal/coq/StringRewriting.v`](../../../tests/formal/coq/StringRewriting.v):
  `rocq c -no-glob StringRewriting.v` prints "Closed under the global context"
  for each theorem. The formal gate checks every `tests/formal/coq/*.v`.
