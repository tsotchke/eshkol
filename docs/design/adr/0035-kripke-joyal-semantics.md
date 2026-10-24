---
kind: explanation
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/topos.esk
  - tests/stdlib/topos_test.esk
  - tests/stdlib/topos_corpus_test.esk
  - tests/stdlib/topos_corpus_sheaf_test.esk
  - tests/stdlib/topos_reductions_test.esk
  - tests/stdlib/topos_jtranslation_test.esk
  - tests/stdlib/topos_local_ring_test.esk
  - examples/mathematics_presheaf_topos_internal_logic.esk
---
# ADR-0035: Kripke-Joyal semantics for local truth

**Status:** Proposed
**Date:** 2026-10-24
**Scope:** what it means in Eshkol for a proposition to hold of data that is
only locally available. That is, the semantics of assertions, contracts and
logical queries whose subject is assembled from pieces: shards, devices,
time windows, opens of a space, fibres of a computation over Spec Z. It also
covers the library that computes this semantics now (`core.topos`), the
proposed `locally` modality, and its relation to the HoTT layer and to the
ring carriers of ADR-0034.
**Depends on:** ADR-0033 (`core.sheaf`: finite sites, sieves, presheaves,
gluing), ADR-0034 (ring carriers, the functor-of-points evaluation model),
ADR-0004 (one quantitative dependent type system).

## Context

Eshkol already computes with local data. `core.sheaf` (ADR-0033) glues
results over covers and names the overlaps where gluing fails. `core.dbsp`
maintains Z-set views over changing inputs, and the distributed and device
layers split work over shards. ADR-0034 proposes running exact computations
in every fibre F_p and gluing by the Chinese remainder theorem. What none of
these has is a semantics for *propositions* about such data. A Boolean test
evaluated on the assembled whole is the only reading available, and it is
the wrong one, for two reasons.

1. A property can hold of every piece without holding of the whole. On
   Spec(Z/12) every element x has x or 1 - x invertible *locally*: on
   D(x) and D(1 - x), which cover. Globally neither holds for x = 3.
   Locality is a theorem about the structure sheaf, but a two-valued check at
   the global stage refutes it.
2. A property can fail to be decidable from the data available at a stage.
   Excluded middle fails for the generic subobject of the walking arrow. A
   program that branches on such a proposition as if it were decided has
   invented information.

The mathematics that handles both is the internal logic of a topos
(Mac Lane-Moerdijk ch. VI; Johnstone, *Elephant*, D1 and D4). A proposition
about data of type X in context U is evaluated at stage U on a generalized
element `alpha : y(U) -> X`. Disjunction and existence are *local*: they
hold when they hold on a cover. Implication and universal quantification
range over all later stages. This is Kripke-Joyal forcing (Joyal; Osius
1975). For the trivial topology on a poset it is Kripke's semantics (1965),
and for a topology of bars on a tree it is Beth's (1956).

The HoTT layer (`inc/eshkol/types/hott_types.h`) has a universe U2 of
propositions (`Eq`, `LessThan`, `Bounded`, `Subtype`). They are erased at
runtime and their meaning is not tied to any semantics of local data. Eshkol
has no special form for assertions over local data.

## Decision

### 1. The facility now: `core.topos` (v1.4.0-connection)

`core.topos` builds the topos Sh(C, J) of any finite site from `core.sheaf`
(with the trivial topology, Psh(C)) and gives it the full Mitchell-Benabou
language:

- **Types:** caller-supplied sheaves, 1, Omega (closed sieves), products,
  exponentials `B^A(U) = Nat(y(U) x A, B)` and power objects `Omega^A`.
- **Terms:** variables, declared morphisms and constants (naturality checked),
  pairing and projections, application, lambda abstraction, comprehension
  `{x : A | phi}`, and `truth phi : Omega`.
- **Formulas:** equality, true and false, the connectives, and quantifiers
  over every type, including Omega and P(A).

It computes two semantics independently:

- **Forcing** by the Kripke-Joyal clauses, with covering sieves enumerated
  (the minimal ones suffice, since J(U) is upward closed). Nothing is
  reduced to the identity cover.
- **The Omega-valued interpretation**, compositional through the Heyting
  algebra of Omega_J. Implication is found by enumeration, ∃ is the closure
  of the union of images f_!, and ∀ is the intersection of the right adjoints
  ∀_f of pullback.

`topos-cross-check` verifies the forcing theorem (U forces phi(alpha) iff
[[phi]]_U(alpha) is maximal) on every stage and environment. The suites
check:

- the forcing theorem on an exhaustively enumerated corpus on five sites
  (walking arrow, pseudocircle, BZ/2, sheaves on the pseudocircle space, a
  tree with bars);
- the Kripke reduction against an independent Kripke evaluator, and the Beth
  reduction against an independent Beth evaluator with bars computed from
  paths and from unions of opens;
- the soundness of intuitionistic higher-order logic: 27 theorems forced on
  four sites;
- the failure of excluded middle, double-negation elimination and De Morgan's
  law, each with an explicit counter-witness stage, on the non-Boolean
  sites. De Morgan's law holds on the walking arrow, a De Morgan topos.
  All of these laws hold on the Boolean BZ/2;
- the Lawvere-Tierney bijection by exhaustion;
- the flagship: the structure sheaf of Spec(Z/n) is internally a local ring.

Reference: [topos.md](../../reference/stdlib/topos.md).

### 2. Local truth is forcing

The meaning Eshkol adopts for "proposition phi holds of local data alpha in
context U" is **U forces phi(alpha) in Sh(C, J)**, where:

- **C** is the category of contexts of availability: shards and their
  inclusions, devices and their sub-meshes, time windows, opens, or affine
  schemes Spec R with their localizations. An arrow V -> U says that V is a
  more specific context than U.
- **J** says when a family of more specific contexts jointly determines U:
  shards that partition a key range, devices that cover a mesh, windows that
  cover an interval, opens whose union is U, localizations that generate the
  unit ideal.
- **Data of type X** is a sheaf on (C, J). Its sections over U are the
  results available in context U, and restriction is the passage to a more
  specific context. Gluing is the sheaf condition, already decided and
  explained by `core.sheaf.glue`.
- **A generalized element** alpha : y(U) -> X is a section over U.

Consequences of this choice, each now a computed fact in the suites:

- **Monotonicity.** A proposition forced in context U stays forced in every
  more specific context.
- **Local character.** A proposition forced on every member of a cover is
  forced on the whole. Assertions over sharded data can therefore be checked
  shard by shard and glued, and the global answer is the forcing at the
  global stage.
- **Sound reasoning.** Every theorem of intuitionistic higher-order logic
  holds, so a derivation in that logic is a valid argument about local data
  whatever the site.
- **No invented decisions.** Excluded middle is not assumed. A branch on an
  undecided proposition is not a valid step. On a Boolean site (presheaves on
  a group, or a dense topology on a finite poset) the classical laws are
  forced again and the library shows that they are.

### 3. The `locally` modality is the Lawvere-Tierney topology j

A language of `core.topos` may carry a second, *modal* topology J' on the
same category. `(locally p)` is forced at U iff some J'-covering sieve has
every member forcing p. Its Omega-valued meaning is the J'-closure: the
Lawvere-Tierney topology j' applied to [[p]]. On data, the corresponding
operation is sheafification for J' (`sheafify` in `core.sheaf`), whose unit
is the map into the J'-local reading.

The precise statement adopted is the Lawvere-Tierney translation: for a
J-sheaf X, **Sh(C, J) forces phi at U iff Psh(C) forces phi^j at U**.
Here phi^j applies `locally` to the atoms, to false, to every disjunction and
to every existential, and leaves conjunction, implication and the universal
quantifier unchanged (Johnstone A4.4, D4.6; Mac Lane-Moerdijk V.3).
`tests/stdlib/topos_jtranslation_test.esk` checks this on the whole first-order
corpus over the open-cover topology of the pseudocircle space. The same
corpus shows that the untranslated presheaf reading differs on some formulas.

So "phi holds locally" is a modal operator with an exact semantics. It is
idempotent, it preserves conjunction and truth, and it is the identity on
the propositions of a topos whose own topology is J'. This is the meaning a
language-level `locally` form will have (section 6).

### 4. Relation to the HoTT layer

The universe U2 of `hott_types.h` holds propositions that are erased at
runtime. Under this record their meaning in a context Gamma of types is a
**subobject** [[phi]] of Gamma. Equivalently, it is a map Gamma -> Omega:
in HoTT terms a mere proposition, in the topos the subobject that forcing
computes. Concretely:

- `Eq` at type A is the equality predicate, with truth value the closed
  sieve of arrows along which the two sides agree. For a sheaf this is
  already closed; for a separated presheaf it is the closure.
- Propositional truncation ||A|| (existence of an inhabitant) is the image of
  A -> 1. Its forcing clause is the local existential: an inhabitant on a
  cover, not a global one.
- The `locally` modality of section 3 is, on propositions, the lex modality
  given by the Lawvere-Tierney topology. Its modal propositions are exactly
  the j-closed ones, and on types it is sheafification, a left exact
  reflective subuniverse (Rijke, Shulman and Spitters, "Modalities in
  homotopy type theory", 2020). The planned type-level `locally` of
  ADR-0004's trajectory is this modality, not a new primitive.
- Erasure is unaffected. A U2 proposition carries no runtime data. When its
  context is a finite site, forcing is a decision procedure for it, and the
  proof obligation "phi holds at U" is the computed fact `topos-forces?`.

### 5. Relation to ADR-0034 (ring carriers): generalized elements are R-points

ADR-0034 reads a numeric program built from ring operations as a morphism of
functors of points. It has a value at an R-point for every carrier R, and
every ring map commutes with it. The link is this:

- Take for C the opposite of a category of rings (finitely presented
  algebras, or the localizations of a fixed ring) and for J the Zariski
  topology: a family of localizations R -> R[1/f_i] covers when the f_i
  generate the unit ideal. A stage is U = Spec R, and y(Spec R) is the functor
  Hom(R, -).
- A sheaf X on this site is a functor of points that satisfies Zariski
  descent. By the Yoneda lemma, a generalized element alpha : y(Spec R) -> X
  is exactly an element of X(R), **an R-point of X**. Restriction of alpha
  along V -> U, that is, along R -> R', is base change of the point.
- A term of the internal language is a morphism of sheaves, so its value at
  an R-point is its value computed over the carrier R. The naturality that
  `make-topos-language` checks for every declared morphism is ADR-0034's
  requirement that every ring map commute with the program. Evaluating a
  program over a carrier (ADR-0034) is evaluating a term at a stage (this
  record). A ring carrier is an R for which stages Spec R are available.
- A formula forced at Spec R is a statement about the R-point that holds
  Zariski-locally on R. The flagship test exhibits this on the stages of
  Spec(Z/12) and Spec(Z/30). The affine line O, whose R-points are the
  elements of R, is internally a local ring, though Z/12 and Z/30 are not
  local rings. Internally, "x or 1 - x is invertible" is forced through the
  cover by D(x) and D(1 - x).
- ADR-0034's computation over Spec Z (section 7 there) is the sheaf condition
  of this site at work. For squarefree M = p_1 ... p_k, Spec(Z/M) is the
  discrete space of the primes p_i. The structure sheaf has
  O(Spec Z/M) = Z/M, the product of the F_{p_i} by CRT. The flagship test
  checks that O is a sheaf on exactly this site: CRT gluing of fibre results
  is the gluing of sections of O. A property of the glued result that is
  checked per fibre is forced at the global stage by local character. For
  squarefree M the test also shows that O is internally a field (x = 0 or x
  is invertible, forced through the fibres). For M = 12 the nilpotent 2 in
  Z/4 refutes that statement at the stage D(3).

### 6. Language adoption, staged along the release ladder

Each rung ships only what its gate proves. Work not yet in the tree is
Planned for the rung named. No compiler, VM or runtime source changes in
v1.4.0-connection.

| Release | Stage | Gate |
|---|---|---|
| v1.4.0-connection | `core.topos` (sections 1-3): the internal language, forcing and the Omega-valued interpretation, Lawvere-Tierney topologies, the `locally` modality over a modal topology, and this record. | `topos_test`, `topos_corpus_test`, `topos_corpus_sheaf_test`, `topos_reductions_test`, `topos_jtranslation_test` and `topos_local_ring_test` on JIT and AOT, the rewritten presheaf-topos example with unchanged output, on Linux x86-64 and macOS arm64. |
| v1.4.1 | Planned for v1.4.1: sites given by a coverage (generating families), so that forcing ranges over generating covers without enumerating every covering sieve. Forcing over the keyed covers of `core.sheaf.glue` (shards, devices, time windows): `topos-check-locally`, which checks a formula piece by piece and glues the verdict with the obstruction report of the gluing layer. | The corpus cross-check on coverage-generated sites equal to the enumerated sites; glue-layer verdicts equal to forcing at the global stage on sharded fixtures. |
| v1.4.2 | Planned for v1.4.2: an assertion surface as library macros (no compiler change). `(define-site name ...)`, and `(assert-forced site stage phi)` that reports the counter-witness stage and environment when it fails. `(locally p)` usable in those assertions. | The assertion forms produce the same verdicts as `topos-forces?` on the corpus; counter-witness reports match `topos-counter-witness`. |
| v1.4.3 | Planned for v1.4.3: incremental forcing on DBSP. The forcing tables of a formula are maintained as a DBSP view while sections and covers change, so a long-running assertion over streaming local data is re-decided by differences. | Incremental tables equal to recomputation after every change on randomized fixtures; differences bounded by the changed stages. |
| v1.4.4 | Planned for v1.4.4: VM and browser VM parity of the library, and the assertion forms on the VM path. | VM parity rows for every `core.topos` export; the corpus on the VM equal to native. |
| v1.4.5-accelerate | Planned for v1.4.5-accelerate: batched forcing. Truth values are already bit masks of sieves, so meets, joins, closures and the adjoint tables become vector bit operations on CPU vector units, GPU and XLA, for sites with many stages. | Batched truth values bit-identical to the scalar library on the corpus; throughput measured against the scalar path. |
| v1.5.0 | Planned for v1.5.0: the type-level reading of section 4 (ADR-0004). U2 propositions are interpreted as subobjects; `locally` is a lex modality on types (sheafification); forcing is the decision procedure for propositions whose context is a finite site; ring carriers (ADR-0034, v1.5.0) supply the stages Spec R. | Type-checker conformance suite for the modality laws (idempotent, preserves finite products, unit is the sheafification unit); every v1.4.x assertion re-expressed at the type level with unchanged verdicts. |

## Alternatives rejected

- **Two-valued truth on the assembled whole.** It rejects true local
  statements (the flagship) and decides undecided ones. It is the reading
  this record replaces.
- **Kripke semantics only (the trivial topology).** It is the special case
  in which every cover is trivial. Disjunction and existence must then be
  decided at the stage, so locality of the structure sheaf fails and
  statements about sharded data could only be made after assembly.
- **Beth semantics only.** It covers posets with bars, but not sites with
  non-trivial automorphisms such as presheaves on a group, nor Zariski
  sites. Forcing on general sites contains it, and the suites check the
  reduction.
- **Checking local truth by sampling stages or environments.** On finite
  sites forcing is decidable exactly, so a sampled answer would be a
  weaker claim than the one available.

## Consequences

- `core.topos` is the one place that decides local truth. `core.sheaf`
  remains the one place for sites, presheaves and gluing (ADR-0033), and
  `core.topos` consumes it without re-implementing it.
- The presheaf-topos example now uses the library, and its output is
  byte-identical to before.
- Exponentials and power objects are enumerated, so quantifying over them
  costs time exponential in the size of the site. The v1.4.5-accelerate rung
  addresses throughput, and the coverage rung (v1.4.1) addresses the number
  of covers. Neither changes the semantics.

## Evidence

- `tests/stdlib/topos_test.esk`: types, Heyting adjunction, the
  Lawvere-Tierney bijection (4 of 108 endomaps on the walking arrow, and the
  inverse), the classifier and exponential adjunction by counting, soundness
  and the classical-law counter-witnesses.
- `tests/stdlib/topos_corpus_test.esk` and `topos_corpus_sheaf_test.esk`:
  forcing against the Omega-valued truth on the enumerated corpus over five
  sites (three presheaf toposes, two sheaf toposes).
- `tests/stdlib/topos_reductions_test.esk`: the Kripke and Beth reductions
  against independent evaluators.
- `tests/stdlib/topos_jtranslation_test.esk`: the j-translation theorem.
- `tests/stdlib/topos_local_ring_test.esk`: the internally local structure
  sheaf of Spec(Z/12) and Spec(Z/30), the covering witnesses D(x) and
  D(1 - x), the refutation in the presheaf topos, and the field statements
  (squarefree n versus the nilpotent 2 in Z/4).

## References

- S. Mac Lane, I. Moerdijk, *Sheaves in Geometry and Logic* (1992),
  ch. III, V and VI (VI.5-VI.7 the Mitchell-Benabou language and Kripke-Joyal
  semantics; VI.10 the generic local ring).
- P. T. Johnstone, *Sketches of an Elephant* (2002), A4.4 (local operators),
  D1 (the internal language), D4 (higher-order logic; D4.6 De Morgan toposes).
- G. Osius, "A note on Kripke-Joyal semantics for the internal language of
  topoi", in *Model Theory and Topoi*, LNM 445 (1975).
- S. Kripke, "Semantical analysis of intuitionistic logic I" (1965).
- E. W. Beth, "Semantic construction of intuitionistic logic" (1956).
- A. Kock, *Synthetic Differential Geometry* (2nd ed. 2006), III.11
  (the internally local ring of the Zariski topos).
- E. Rijke, M. Shulman, B. Spitters, "Modalities in homotopy type theory",
  LMCS 16 (2020).
- A. Grothendieck, J. Dieudonne, EGA I (1960), the functor of points.
