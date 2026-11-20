---
kind: explanation
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/category.esk
  - lib/core/category/finite.esk
  - lib/core/category/functor.esk
  - lib/core/category/set.esk
  - lib/core/category/kan.esk
  - lib/core/category/linear.esk
  - lib/core/category/descent.esk
  - lib/core/category/cohomology.esk
  - tests/stdlib/category_test.esk
  - tests/stdlib/category_s62_test.esk
---
# ADR-0036: `core.category` — finite category theory as the shared base layer

**Status:** Proposed
**Date:** 2026-10-29
**Scope:** the library that computes with finite categories, functors,
natural transformations, limits and colimits, ends and coends, Kan
extensions, adjunctions, monads and the Yoneda lemma; its boundary with
`core.sheaf` (ADR-0033), whose finite-category representation it uses, and
with `core.topos` (ADR-0035); and the S6.2 statement that descent is a Kan
extension and sheafification a reflector.
**Depends on:** ADR-0033 (`core.sheaf`: the finite-category, sieve, site and
presheaf representations), ADR-0035 (`core.topos`).

## Context

The mathematics examples compute with categories as data: categories from
presentations, functors checked on composites, limits as compatible
families, colimits as quotients, comma categories, Kan extensions with their
units and counits, Grothendieck constructions, nerves, and derived limits.
Three programs (finite categories and Kan extensions, sheafification and
descent, abelian sheaves on a finite site) each carried their own copy of a
category type, functor checks, limits, colimits, comma categories and Kan
extensions; the homotopy-colimit program carried its own Grothendieck
construction and nerve. `core.sheaf.site` already fixed one representation of
a finite category, and `core.topos` builds on it, but nothing reusable
offered functors, natural transformations, (co)limits, Kan extensions or
adjunctions on that representation. The monoidal and ZX lane (ADR-0041),
the simplicial and homological lane (ADR-0037) and the Lawvere-theory lane
(ADR-0039) all need that layer.

## Decision

1. **One category type: core.sheaf.site's.** Every category is
   `#(finite-category nobj arrows compose ids labels)`. `core.category`
   adds constructors (standard shapes, free categories, monoid and group
   categories, products, coproducts, full subcategories, functor categories,
   comma and slice categories, categories of elements, Grothendieck
   constructions), all of which return that value. No second category type
   exists; `core.sheaf` and `core.topos` are unchanged.

2. **Layers, loaded together by `(require core.category)`.**
   `core.category.finite` (constructions, connected components, the nerve
   with its integral cohomology), `core.category.functor` (functors, natural
   transformations, enumeration of functors and transformations, functor
   categories, comma, slice and sieve categories, category diagrams and their
   Grothendieck construction, adjunctions, adjoints found through comma
   categories, monads and their algebras), `core.category.set` (Set-valued
   functors, limits, colimits, ends, coends, Yoneda, categories of
   elements), `core.category.kan` (Kan extensions by both formulas,
   Lan -| F^* -| Ran, descent as a right Kan extension on thin sites),
   `core.category.linear` (diagrams of free Z-modules and Q-vector spaces:
   limits, colimits with torsion, derived limits, hypercohomology of a
   two-term complex), `core.category.descent` (the sheaf condition and Cech
   descent as limits, sheafification as the reflector a -| i and its monad)
   and `core.category.cohomology` (abelian descent through nerve categories of
   covers and Grothendieck constructions). A program requires only the
   layers it uses; the AOT compile time grows with the layers loaded. They
   are ordinary modules outside the `(require stdlib)` closure; the VM
   prelude and its generated tables are unchanged.

3. **Values are data, laws are checks.** A functor is
   `#(functor C D objects arrows)`; a natural transformation is a vector of
   components. Nothing is assumed: `functor-laws-violations`,
   `natural-transformation-violations`, `adjunction-violations` (both
   triangle identities), `adjunction-hom-bijection-violations`,
   `monad-laws-violations`, `yoneda-violations`, and the Kan-extension
   adjunction checks each return the list of failures, so a law that fails
   names its witness.

4. **Presheaves are Set-functors on the opposite category, without
   copying.** A covariant `#(set-functor C sets maps)` uses the index
   conventions of core.sheaf's `set-presheaf`; `presheaf->set-functor` shares
   the sets and maps of a presheaf with the set-functor on `C^op`. A
   transformation is the same component vector in both readings, so
   core.sheaf's `presheaf-natural?` and core.category's `set-natural?` agree.

5. **Natural transformations are a limit.** Nat(X, Y) is the limit of Y over
   the category of elements of X, found by one propagating search over the
   elements of X (an arrow forces a component value). Enumerating products of
   function spaces is never needed; the end formula
   Nat(X, Y) = int_c Y(c)^X(c) and a brute-force count are the test suite's
   independent cross-checks.

6. **Kan extensions two ways.** The pointwise formulas
   Lan_F G(d) = colim_{(F | d)} G and Ran_F G(d) = lim_{(d | F)} G are
   computed over comma categories; the coend and end formulas
   Lan_F G = int^c D(Fc, -) x G(c) and Ran_F G = int_c G(c)^{D(-, Fc)} are
   computed separately, and `set-lan-comparison` / `set-ran-comparison` give
   the natural isomorphisms between them. The units and counits of
   Lan -| F^* -| Ran come with all four triangle identities.

7. **Derived limits over any finite category.** `module-presheaf-derived-limits`
   is the cohomology of the normalized cosimplicial replacement
   (Bousfield-Kan) over the nerve of `C^op`, over Z with torsion or over Q.
   Over a poset it agrees with core.sheaf's derived limits; over BG it is
   group cohomology. Cech cohomology of a cover is lim^n over the nerve
   category of the cover, and cohomological descent is lim^n over the
   Grothendieck construction of J |-> U_J, whose nerve is the homotopy
   colimit of the cover (Thomason).

8. **S6.2.** Descent is the limit over the category of a covering sieve,
   which is the value at the sieve of the right Kan extension of F along the
   Yoneda embedding; on thin sites it is also the right Kan extension of F
   restricted to the sieve's objects. Cech descent is its equalizer form.
   Sheafification is the plus construction twice, with the counit given by
   gluing twice; a -| i is checked with both triangle identities, the
   universal property counted and as an explicit bijection, and the
   sheafification monad's laws.

## Consequences

- The finite-categories, sheafification and homotopy-colimit examples are
  written on the library with byte-identical output in JIT and AOT, and
  shrink from 504, 562 and 367 lines to 202, 210 and 158.
- `tests/stdlib/category_s62_test.esk` reproduces through the categorical
  operators the numbers of the finite-site sheafification program, the
  finite-space cohomology program, the descent and hypercohomology program,
  and of the 2-descent program the Cech nerve and order complex cell counts,
  the Leray conditions, and the mapping-space cardinalities of the untwisted
  2-types (BZ/p, K(Z/p,2) and their product) on both sides of each cover,
  including the failure of descent on the pseudo-2-sphere.
- Enumeration bounds the sizes: functors, transformations and functor
  categories are enumerated, so they suit small categories; limits, Kan
  extensions and natural transformations use propagating searches and
  scale with the size of the answer.
- Mapping spaces into 2-groups with a nontrivial action or k-invariant
  (crossed modules) are Planned for v1.4.1 with the simplicial layer
  (ADR-0037), which provides simplicial maps into nerves of crossed modules.
- Limits and colimits over a `core.ring` carrier (ADR-0034) are Planned for
  v1.4.1, once that library shares a base with this one; until then the
  linear layer uses core.sheaf.module's Z and Q coefficients.
- Moving `core.sheaf.cohomology`'s poset derived limits onto
  `module-presheaf-derived-limits`, and the plus construction on morphisms
  into `core.sheaf.presheaf`, are Planned for v1.4.1; today the test suites
  check that both routes agree.

## Alternatives considered

- **A new category record with typed objects.** It would duplicate
  core.sheaf.site and force conversions at every boundary with core.sheaf
  and core.topos. Integer objects with labels serve both.
- **Kan extensions only by the pointwise formula.** The (co)end formulas are
  the ones the enriched and monoidal layers need (Kelly); computing both and
  comparing them is also the strongest test of either.
- **Natural transformations by enumerating component functions.** Exponential
  in the sizes of the sets even when the answer is small; the propagating
  search over the category of elements is exact and fast.
- **Derived limits only over posets.** Group cohomology, Grothendieck
  constructions of covers and nerve categories are not posets in general;
  one cosimplicial replacement covers them all.

## Verification

- `tests/stdlib/category_test.esk` (CTest `category_jit`, `category_aot`):
  brute-force cross-checks of nerves, functors (counts |ob C|, |ob C|^2,
  |ar C| from 1, 1 + 1 and the walking arrow; homomorphism counts between
  cyclic groups), the arrow category as [2, C], natural transformations into
  posets, the interchange law, slices and coslices, the Grothendieck
  construction against the category of elements, right and left adjoints
  against exhaustive Galois-partner search, the diagonal's adjoints
  (meet and join), monad laws and algebras, equalizers, coequalizers and
  pullbacks against direct counts for every pair of maps, cones by
  |lim|^k, orbits and fixed points as Lan and Ran by both formulas,
  Lan -| F^* -| Ran triangle identities, three-way natural-transformation
  counts, Yoneda counted and explicit, full faithfulness of the Yoneda
  embedding, co-Yoneda, limits and colimits of modules with torsion, group
  cohomology as derived limits, derived limits against core.sheaf, and the
  sheaf condition by limits against core.sheaf on a non-thin site.
- `tests/stdlib/category_s62_test.esk` (CTest `category_s62_jit`,
  `category_s62_aot`): the S6.2 statement, written before the code at the top
  of the file, and the numbers of the four programs.
- The rewritten examples (CTest `cat_finite_categories_kan_*`,
  `cat_sheafification_descent_kan_*`, `hocolim_grothendieck_borel_*`) diffed
  byte for byte against the previous programs.

## References

S. Mac Lane, *Categories for the Working Mathematician*, 2nd ed. (1998);
E. Riehl, *Category Theory in Context* (2016); G. M. Kelly, *Basic Concepts
of Enriched Category Theory* (1982; TAC Reprints 10, 2005); S. Mac Lane,
I. Moerdijk, *Sheaves in Geometry and Logic* (1992); P. T. Johnstone,
*Sketches of an Elephant* (2002); A. K. Bousfield, D. M. Kan, LNM 304
(1972); R. W. Thomason, Math. Proc. Camb. Phil. Soc. 85 (1979); D. Quillen,
"Higher algebraic K-theory I", LNM 341 (1973); C. Weibel, *An Introduction
to Homological Algebra* (1994); nLab pages "category", "functor", "natural
transformation", "limit", "colimit", "end", "comma category", "Kan
extension", "adjunction", "monad", "Yoneda lemma", "Grothendieck
construction", "nerve", "sheafification", "descent".
