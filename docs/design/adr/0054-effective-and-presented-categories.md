---
kind: explanation
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
# ADR-0054: effective and finitely presented categories

**Status:** Proposed
**Date:** 2026-10-30
**Scope:** category theory with infinite categories. Effectively given
categories (objects and arrows produced on demand), finitely presented
categories decided by Knuth-Bendix completion, the standard infinite
categories (monoids and groups, countable posets, Delta, FinOrd, FinSet,
Mat_R), functors, transformations and adjunctions between them, limits and
colimits where they exist and are computable, Kan extensions along functors
with finite comma categories, and nerves as locally effective simplicial
sets.
**Depends on:** ADR-0036 (`core.category`: finite categories), ADR-0033
(`core.sheaf.site`'s finite-category representation), ADR-0037
(`core.simplicial`: simplicial sets of infinite type, explicit truncation),
ADR-0034 (`core.ring` carriers), ADR-0038 (`core.effective`: the locally
effective model this layer imitates).

## Context

`core.category`, `core.topos`, `core.sheaf` and the Kan extensions of
ADR-0036 list every object and every arrow of a category. Infinite objects
appear in Eshkol only in homotopy theory (`core.effective`, where a chain
complex may be only locally effective: each degree is produced on demand and
the differential is a procedure) and in the HoTT layer. Plain category theory
had no way to say "the monoid N", "the simplex category" or "the category of
integer matrices", although most of category theory is about such
categories. The owner asked for infinite categories in ordinary 1-category
theory.

## Decision

1. **Effective categories, by analogy with locally effective complexes.**
   A category is `#(effective-category name objects object? hom compose
   identity source target arrow=? props)`. The objects are an *enumerator*
   (stages of finite lists, with an optional bound that makes the enumeration
   finite) and/or a membership predicate. `hom a b` returns an enumerator,
   finite or not. Composition, identities and decidable equality of arrows
   are procedures. Declared structure (`'generators`, `'relations`, `'thin`,
   and the finite category a value came from) goes in an association list.
   A finite category is a special case: `finite-category->effective` and
   `effective->finite-category` convert both ways, the second refusing by
   name (`category-not-finite`, with the infinite hom-set as witness) when an
   enumeration is not finite.

2. **Finitely presented categories decide equality by completion, under a
   budget, and never guess.** A presentation is a finite quiver with
   relations between parallel paths (`core.sheaf.site`'s word convention).
   `present-category` runs shortlex Knuth-Bendix completion with
   inter-reduction and a budget (rules alive, equations processed). A
   complete system decides equality by normal forms and enumerates hom-sets
   lazily by normal form and word length. An incomplete one still answers
   `'equal` soundly (every rule is a consequence of the relations) and
   `'distinct` only by a sound invariant (length, when every relation
   preserves it); otherwise `'undecided`. Everything that needs normal forms
   refuses an incomplete system with `presentation-undecided-within-budget`.
   Hom-set finiteness is decided exactly from the automaton of irreducible
   words (a state is an object and the last m - 1 generators): hom(a, b) is
   infinite exactly when a cycle lies on a path from a to b. Monoids and
   groups are one-object presentations; groups add formal inverses.

3. **The standard infinite categories are data.** (N, <=), (Z, <=), (Q, <=)
   and divisibility; (N, +) and (Z, +); Delta and FinOrd as monotone maps;
   the skeleton of FinSet; pointed finite sets; Mat_R for a `core.ring`
   carrier R (objects n, arrows m x n matrices, hom-sets enumerated by entry
   height for Z, Q and Z/m). Delta's faces, degeneracies, cosimplicial
   identities and epi-mono normal form are library procedures, and the
   presentation of Delta_{<=N} by faces, degeneracies and the identities is
   a library value that completion turns into Delta_{<=N}.

4. **Checks say exactly what they checked.** Every `*-violations` procedure
   takes the objects and arrows it checks and returns the list of failures.
   Out of a presented category the checks are complete: a functor is an
   assignment of the generators that respects the relations, and a
   transformation is natural when it is natural on the generators
   (`effective-functor-generator-violations`,
   `effective-transformation-generator-violations`, `presented-functor-violations`).
   Elsewhere the docs and the test names state the finite range checked.
   Adjunctions carry unit and counit; the triangle identities, naturality
   and the hom bijection (with membership of every transpose in its hom-set
   when that hom-set is finite) are checked.

5. **Limits where they exist and are computable; refusals by name
   otherwise.** FinSet: products, coproducts, equalizers, coequalizers,
   pullbacks, pushouts and limits and colimits of any finite diagram, with
   cone and factorization enumerators for the universal property. Mat_R for
   R = Z or a field: kernels, cokernels, (co)equalizers, biproducts,
   pullbacks, pushouts, and finite limits and colimits as one kernel or
   cokernel; over Z kernels are saturated lattices (Smith normal form) and
   the cokernel is the torsion-free quotient Z^m / sat(im A), which is the
   colimit among free modules. Other carriers: `limit-not-computable`.
   Delta and FinOrd: the (co)products that exist (a terminal or initial
   factor) are constructed; for every other pair the unique candidate forced
   by counting at the point is refuted by counting at the next test object,
   and the refusal `limit-does-not-exist` carries that witness. Posets:
   meets, joins, top and bottom as declared; a declared absence refuses with
   `limit-does-not-exist`.

6. **Kan extensions when comma categories are finite.** `effective-lan` and
   `effective-ran` compute Lan_F G(d) = colim_{(F | d)} G and
   Ran_F G(d) = lim_{(d | F)} G pointwise, for Set-valued functors with
   finite values. The source may be infinite: the functor then declares the
   objects over (or under) each d (`'fibre`, `'cofibre`), and each must be
   an object; otherwise, or when a hom-set involved is infinite, the
   refusal is `comma-category-not-finite`. Restriction along Delta_{<=n} ->
   Delta gives truncation; Lan and Ran along it give the skeleton and the
   coskeleton of simplicial presheaves. The left Kan extension along
   (N, <=) -> (Z, <=) is computed at every integer while the right one is
   refused, which is the behaviour the comma categories dictate.

7. **Nerves are locally effective simplicial sets.**
   `effective-category-nerve` is a `core.simplicial.sset` simplicial set of
   infinite type whose faces and degeneracies are computed for every
   category; listing the nondegenerate simplices of a degree needs a finite
   category and is otherwise refused (`nerve-not-enumerable`), with bounded
   listings by hom stage. `effective-nerve-chain-data` returns the pair
   (basis, d) that `core.effective.chain`'s `make-effective-complex` takes
   (basis #f in degrees that are not enumerable, the normalized boundary as
   a linear combination in core.effective's convention). That is the bridge
   to effective homology: with `core.effective` (ADR-0038) on the same base,
   `(apply make-effective-complex "nerve" (effective-nerve-chain-data E))`
   is the nerve's locally effective chain complex. Planned for v1.4.1: that
   one-line constructor in `core.effective`, once both libraries share a
   base.

8. **Machine-checked foundations.** `tests/formal/coq/StringRewriting.v`
   (Rocq 9, no axioms) proves Newman's lemma, uniqueness of normal forms for
   confluent relations, soundness of rewriting and of every completion move
   (orienting, critical pairs, simplification, returning and dropping
   rules), and that with a confluent rule set that joins the relations two
   words are congruent exactly when their normal forms coincide. These are
   the theorems behind the `'equal` and `'distinct` answers.

## Consequences

- Infinite categories are ordinary values: the free monoid on one generator
  is checked isomorphic to (N, +), Z^2 and the free group of rank 2 have
  their normal forms counted by closed formulas, and Delta_{<=3} is
  recovered from its presentation.
- `<a, b | aba = bab>` has no finite complete system on {a, b}
  (Kapur-Narendran); the library reports the budget, answers what the
  partial system proves, and refuses the rest. Choosing orders other than
  shortlex by generator index (weights, recursive path orders) is Planned
  for v1.4.1.
- Enumeration still bounds what is practical: hom stages grow with the
  alphabet, and coskeleta in high degrees search large comma categories.
- Kan extensions along functors with infinite comma categories (for example
  Ran along (N, <=) -> (Z, <=)) remain refused; computing them through
  finite presentations of the colimit or limit is Planned for v1.4.2.
- The elementary-topos structure of the FinSet skeleton (exponentials and the
  subobject classifier as effective constructions) is Planned for v1.4.1.

## Alternatives considered

- **Extend `#(finite-category ...)` with lazy fields.** Every finite
  algorithm indexes arrows by integers; a second representation with an
  explicit conversion keeps them exact and leaves `core.sheaf` and
  `core.topos` unchanged.
- **Streams instead of staged enumerators.** Stages make the order
  deterministic, give a natural notion of size (word length, entry height,
  degree) and carry finiteness as a bound that the code can test.
- **Answer `'distinct` from bounded search when completion stops.** That
  guesses; the decision procedure answers only what a proof supports.
- **Hard-code nonexistence of (co)products in Delta.** The counting witness
  is computed from the hom-sets, so the refusal is itself checkable.

## Verification

- `tests/stdlib/category_effective_test.esk` (CTest `category_effective_jit`,
  `category_effective_aot`; pending build node): 51 checks with independent
  cross-checks: closed forms (4n normal forms of Z^2, 4 * 3^(n-1) for the free
  group, binomial hom counts of Delta and FinOrd, n^m for FinSet,
  (2s+1)^k - (2s-1)^k for Mat_Z), permutation orders against S3,
  core.sheaf.site's presentation route, core.category.kan's Lan and Ran on
  finite functors, core.integer_linalg's rational kernels, brute-force
  kernels over Z/5, finite-category-nerve's counts, and d d = 0.
- `tests/nlab/nlab_category_effective_test.esk` (CTest
  `nlab_category_effective_{jit,aot}`; pending build node): 13 nLab claims.
- `tests/formal/coq/StringRewriting.v`: checked with `rocq c` (Rocq 9.3.0);
  picked up by the formal gate's `tests/formal/coq/*.v` glob.

## References

D. E. Knuth, P. B. Bendix (1970); M. H. A. Newman, Ann. of Math. 43 (1942);
R. V. Book, F. Otto, *String-Rewriting Systems* (1993); D. Kapur,
P. Narendran, Theoret. Comput. Sci. 35 (1985); D. F. Holt, B. Eick,
E. A. O'Brien, *Handbook of Computational Group Theory* (2005);
F. Sergeraert, J. Rubio, Bull. Sci. Math. 126 (2002); S. Mac Lane,
*Categories for the Working Mathematician*, 2nd ed. (1998); E. Riehl,
*Category Theory in Context* (2016); P. Goerss, J. Jardine, *Simplicial
Homotopy Theory* (1999); P. Gabriel, M. Zisman (1967); H. Cohen, GTM 138
(1993); nLab pages "free monoid", "free category", "simplex category",
"coskeleton", "adjoint functor", "FinSet", "matrix".
