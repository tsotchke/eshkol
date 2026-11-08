---
kind: reference
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

# `core.topos` — the internal language of a topos, with Kripke-Joyal forcing

**Source:** [topos.esk](../../../lib/core/topos.esk).
**Require:** `(require core.topos)`; builds on [`core.sheaf`](sheaf.md)
(`core.sheaf.site`, `core.sheaf.presheaf`), no compiler changes. It is not part of
`(require stdlib)`.
**Design record:** [ADR-0035](../../design/adr/0035-kripke-joyal-semantics.md).

For a finite site (C, J) from `core.sheaf`, `core.topos` builds the topos
Sh(C, J). With the trivial topology this is the presheaf topos Set^{C^op}. It
then gives the topos its Mitchell-Benabou internal language. Formulas of
higher-order intuitionistic logic are written as s-expressions over the objects
of the topos. They are interpreted in two independent ways, both computed
exactly by enumeration over the finite site:

- **Kripke-Joyal forcing** `U ||- phi(alpha)`, by recursion on the formula over
  stages U and generalized elements `alpha : y(U) -> X`;
- **the Omega-valued interpretation** `[[phi]]`, a closed sieve at every stage,
  computed through the Heyting algebra of the subobject classifier.

The forcing theorem says that `U ||- phi(alpha)` holds iff `[[phi]]_U(alpha)` is
the maximal sieve, that is, iff `alpha` factors through the subobject `[[phi]]`.
`topos-cross-check` verifies this for a given formula on every stage and every
environment.

Sources: S. Mac Lane, I. Moerdijk, *Sheaves in Geometry and Logic* (1992),
I.4, I.6, III.7, V.1, chapter VI (the Mitchell-Benabou language VI.5,
Kripke-Joyal semantics VI.6 for presheaves and VI.7 for sheaves) and chapter
VIII (the Zariski topos and local rings); P. T. Johnstone, *Sketches of an
Elephant* (2002),
D1 (the internal language) and D4 (the higher-order logic of toposes, local
operators, De Morgan toposes D4.6); A. Joyal's forcing semantics as written
up by G. Osius, "A note on Kripke-Joyal semantics for the internal language
of topoi" (1975); S. Kripke, "Semantical analysis of intuitionistic logic I"
(1965); E. W. Beth, "Semantic construction of intuitionistic logic" (1956).

## A first program

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

Stage 1 has three sieves, so `Omega(1)` has three elements. The middle sieve
`{0 -> 1}` is neither forced nor refuted at stage 1. It is the
generalized element that refutes excluded middle and double-negation
elimination.

## Toposes

| Procedure | Contract |
|---|---|
| `(make-topos site)` | The topos Sh(C, J) of a `core.sheaf` site. The covering sieves must satisfy the three Grothendieck axioms (checked; violations raise). |
| `(presheaf-topos C)` | `(make-topos (trivial-site C))`: presheaves on a finite category. |
| `(topos? x)` `(topos-site T)` `(topos-category T)` | Inspection. |

## Types (objects of the topos)

Every type is a finite presheaf in `core.sheaf`'s representation (element
lists per object and an index vector per arrow). `(topos-type-presheaf X)`
returns it, so all of `core.sheaf` applies (`presheaf-sheaf?`,
`presheaf-natural?`, ...). A type constructor called twice with the same
arguments returns the same object (`eq?`).

| Procedure | Contract |
|---|---|
| `(topos-object T F)` | A caller-supplied presheaf `F`. It must be a sheaf for the topology of `T` (checked with `presheaf-sheaf-violations`; a non-sheaf raises). Use `sheafify` from `core.sheaf` first when needed. |
| `(topos-terminal T)` | The terminal sheaf 1. |
| `(topos-omega T)` | The subobject classifier: `Omega_J(c)` is the set of J-closed sieves on c (all sieves for the trivial topology); `Omega(f)(S) = f^* S`. Elements are ascending arrow lists. |
| `(topos-product T A B)` | Pointwise product; elements are lists `(a b)`. |
| `(topos-exponential T A B)` | `B^A(U) = Nat(y(U) x A, B)`, enumerated by backtracking with propagation along restrictions; restriction is precomposition. |
| `(topos-power T A)` | `P(A) = Omega^A`; an element at U is the classifying map of a closed subobject of `y(U) x A`. |
| `(topos-morphisms T A B)` | Every natural transformation `A -> B`, each a vector over objects of index vectors. |
| `(topos-type-size X c)` `(topos-type-elements X c)` `(topos-type-element X c i)` `(topos-type-index X c v)` | Elements of `X(c)`. |
| `(topos-type-restrict X a v)` | `X(a)(v)` for an arrow `a : d -> c` and `v` in `X(c)`. |
| `(topos-type? x)` | Whether `x` is a type of some topos. |
| `(topos-type-kind X)` `(topos-type-factors X)` | `base`, `terminal`, `omega`, `product` or `exponential`; the factors `(A B)` of a product or of `B^A`. |
| `(topos-exponential-apply T E U theta a)` | Evaluation `B^A x A -> B` at stage U: `theta_U(id_U, a)`. |

Types are owned by the topos and kept in its table. Create them before
entering a `with-region` (see [Memory](#memory)).

## Omega, closure and Lawvere-Tierney topologies

| Procedure | Contract |
|---|---|
| `(topos-closed-sieves T c)` | `Omega_J(c)`, ascending arrow lists. |
| `(topos-closure T c S)` `(topos-closed? T c S)` | `cl(S) = {f : V -> c | f^* S covers V}`. |
| `(topos-omega-top T c)` `(topos-omega-bottom T c)` | The maximal sieve; the closure of the empty sieve. |
| `(topos-omega-meet T c S R)` `(topos-omega-join T c S R)` | Intersection; closure of the union. |
| `(topos-omega-implies T c S Q)` | The largest closed `R` with `R meet S <= Q`, found by enumerating `Omega_J(c)`. |
| `(topos-omega-not T c S)` `(topos-omega-boolean? T c)` | Pseudo-complement; whether `Omega_J(c)` is Boolean. |
| `(topos-lawvere-tierney T)` | The Lawvere-Tierney topology `j : Omega -> Omega` of the presheaf topos on C induced by J, `j_c(S) = {f : f^* S covers}`: a vector over objects of index maps on `(sieves-on C c)`. |
| `(topos-lawvere-tierney? C j)` | The axioms: natural, `j true = true`, `j j = j`, `j` preserves meets. |
| `(topos-lawvere-tierney-site C j)` | The Grothendieck topology of `j`: S covers c iff `j_c(S)` is maximal (Mac Lane-Moerdijk V.1.2, the inverse bijection). |
| `(topos-classifying-map T X A)` | For a sub-presheaf A (per-object lists of element indices), the map `X -> Omega_J`, `x |-> cl{f : X(f) x in A}`. |
| `(topos-classified-subobject T X chi)` | The pullback of true along `chi`. |

## Languages

A language fixes the sorts and the symbols that formulas may use. Every
declared symbol is checked to be a morphism of the topos when the language is
built: functions and constants must be natural, and predicates must define
sub-presheaves.

| Procedure | Contract |
|---|---|
| `(make-topos-language T sorts declarations [modal-site])` | `sorts` is an alist `(name . type)`; `declarations` is a list of the three kinds below. |
| `(topos-function name arg-types result-type proc)` | `proc` maps `(U x ...)`, element values at U, to the result value at U. |
| `(topos-constant name type proc)` | `proc` maps U to a value at U: a global element. |
| `(topos-predicate name arg-types pred)` | `pred` maps `(U x ...)` to a boolean and must define a sub-presheaf. |

The optional `modal-site` is a second Grothendieck topology J' on the
category of `T`. It gives meaning to `(locally p)`: p holds on a J'-cover (see
[the modality](#the-locally-modality)). Argument and result types are type
expressions (below). A predicate is used
through its classifying map into `Omega_J`, so on a non-trivial site `P(x)` is
forced at U exactly when `P` holds on a cover of U. `(topos-language-type L e)`
returns the type denoted by a type expression, and `(topos-language-topos L)`
returns the topos.

## Formulas, terms and types

**Type expressions:** a declared sort name, `Omega`, `1`, `(* A B)`, `(fun A B)`
(the exponential `B^A`), `(P A)`, or a type object.

**Terms:** variables; declared constants; `(f t ...)` for a declared function;
`(pair s t)` `(fst t)` `(snd t)`; `(app f t)` for `f : (fun A B)`;
`(lambda (x A) t)`; the comprehension `(set (x A) phi) : (P A)`; and
`(truth phi) : Omega`, the truth value of a formula.

**Formulas:** `true` `false` `(= s t)` `(and phi ...)` `(or phi ...)`
`(implies phi psi)` `(iff phi psi)` `(not phi)` `(forall (x A) phi)`
`(exists (x A) phi)` (several binders `((x A) (y B))` nest) `(in t s)` for
`s : (P A)`, `(holds t)` for `t : Omega`, and `(R t ...)` for a declared
predicate, and `(locally phi)` when the language has a modal topology.
Quantifiers range over every type, including `Omega`, `P(A)` and
exponentials, so the logic is higher order.

`(topos-formula L context phi)` type-checks and compiles `phi`; `context` lists
the free variables as `((x A) ...)`, and `(topos-formula-context F)` returns
them as `((x . type) ...)`. Ill-typed formulas, unknown symbols and
unbound variables raise. The compiled formula is immutable.

## Queries

Environments are lists of element values in the order of the context. The
value for `x : A` at stage U is an element of `A(U)`, a generalized element
`y(U) -> A` by the Yoneda lemma; a value outside `A(U)` raises.

| Procedure | Result |
|---|---|
| `(topos-forces? F U [env])` | Kripke-Joyal forcing `U ||- phi(env)`. |
| `(topos-truth F U [env])` | The Omega-valued truth `[[phi]]_U(env)`, a closed sieve on U. |
| `(topos-environments F U)` | Every environment at stage U. |
| `(topos-extension F)` | For each stage, the environments that force `phi`: the subobject `[[phi]]`. |
| `(topos-valid? F)` `(topos-counter-witness F)` | Whether every stage forces `phi` for every environment, and otherwise the first `(stage env)` that does not. |
| `(topos-cross-check F)` | `(instances #t)` when forcing and the Omega-valued truth agree everywhere and every truth value is a closed sieve; otherwise `(instances #f stage env forced truth)`. |
| `(topos-monotone? F)` `(topos-local? F)` | Monotonicity under restriction, and local character: if the sieve of arrows along which `phi` is forced covers U, then U forces `phi`. |
| `(topos-forcing-report F)` | `(instances forcing-theorem? monotone? local?)`: the three checks above on one memo context. |
| `(topos-instance-count F)` | The number of (stage, environment) pairs. |

### The forcing clauses

For a stage U and an environment `alpha` (Mac Lane-Moerdijk VI.7.1):

| Formula | `U ||- ...` iff |
|---|---|
| `s = t` | `s(alpha) = t(alpha)` in `X(U)` |
| `false` | the empty sieve covers U |
| `p and q` | U forces both |
| `p or q` | some covering sieve S of U has every member `f : V -> U` forcing `p(alpha f)` or `q(alpha f)` |
| `p implies q` | for every `f : V -> U`, if V forces `p(alpha f)` then V forces `q(alpha f)` |
| `not p` | for every `f : V -> U`, if V forces `p(alpha f)` then the empty sieve covers V |
| `exists y : Y. p` | some covering sieve S of U has, for every member `f : V -> U`, a `beta` in `Y(V)` with V forcing `p(alpha f, beta)` |
| `forall y : Y. p` | for every `f : V -> U` and every `beta` in `Y(V)`, V forces `p(alpha f, beta)` |
| `holds t` | `t(alpha)` is the maximal sieve on U |

The covering sieves are enumerated. Only the minimal covering sieves are
needed, because J(U) is upward closed. The clauses are not reduced to the
identity cover. For the trivial topology on a poset these clauses are
Kripke's; for a non-trivial topology on a poset they are Beth's. Both
reductions are checked in `tests/stdlib/topos_reductions_test.esk` against
independent Kripke and Beth evaluators.

### The Omega-valued interpretation

`[[s = t]]` is the closure of `{f : X(f) s = X(f) t}`, meet is
intersection, and join is the closure of the union. Implication is the
largest closed R with `R meet S <= T`, found by enumeration. `[[exists y p]]` is
the closure of the union of the images `f_!([[p]](alpha f, beta))`, and
`[[forall y p]]` is the intersection of the right adjoints
`forall_f([[p]](alpha f, beta))` of pullback. None of these steps call the
forcing evaluator.

### The `locally` modality

`(locally p)` is forced at U iff some covering sieve of the modal topology J'
has every member forcing p. Its Omega-valued meaning is the J'-closure of
[[p]], which is the Lawvere-Tierney topology j' applied to the truth value. Over the
presheaf topos on C, with J' the topology of a site, this gives the
Lawvere-Tierney translation: for a J'-sheaf X, `Sh(C, J')` forces phi at U
iff `Psh(C)` forces phi^j at U. Here phi^j wraps atoms, `false`, every `or` and
every `exists` in `locally`, and turns `(not p)` into
`(implies p^j (locally false))`. `tests/stdlib/topos_jtranslation_test.esk` checks
the theorem on its whole first-order corpus over the open-cover topology of a
finite space.

## Examples

Excluded middle fails on the walking arrow; on the Boolean topos of
presheaves on Z/2 it holds:

```scheme
(require stdlib)
(require core.sheaf.site)
(require core.topos)
(define (z2-nf w) (if (even? (length w)) '() (list 0)))
(define BZ2 (make-category-from-presentation 1 (list (list 0 0 's)) z2-nf 2))
(define T (presheaf-topos BZ2))
(define L (make-topos-language T '() '()))
(topos-valid? (topos-formula L '() '(forall (p Omega) (or (holds p) (not (holds p)))))) ;; => #t
```

On a finite space with the open-cover topology, disjunction is local. The
pseudocircle's open `{0, 1}` is disconnected, and the section of the
locally constant sheaf that is 0 on one point and 1 on the other forces
`x = zero or x = one`, although neither disjunct holds on the whole open:

```scheme
(require stdlib)
(require core.sheaf.site)
(require core.sheaf.presheaf)
(require core.topos)
(define space (make-finite-space 4 (list (cons 0 2) (cons 1 2) (cons 0 3) (cons 1 3))))
(define site (finite-space-site space))
(define C (site-category site))
(define T (make-topos site))
(define two (topos-object T (sheafify site (constant-set-presheaf C '(0 1)))))
(define top (category-object-of-label C '(0 1 2 3)))
(define (global i U) (topos-type-restrict two (car (category-hom C U top)) (topos-type-element two top i)))
(define L (make-topos-language T (list (cons 'X two))
                               (list (topos-constant 'zero 'X (lambda (U) (global 0 U)))
                                     (topos-constant 'one 'X (lambda (U) (global 1 U))))))
(define F (topos-formula L '((x X)) '(or (= x zero) (= x one))))
(define ab (category-object-of-label C '(0 1)))
(map (lambda (e) (topos-forces? F ab e)) (topos-environments F ab)) ;; => (#t #t #t #t)
(topos-valid? F)            ;; => #t
(cadr (topos-cross-check F)) ;; => #t
```

### Flagship: the structure sheaf of Spec(Z/n) is internally local

`tests/stdlib/topos_local_ring_test.esk` builds the Zariski site of Z/n.
Its objects are the basic opens `D(f)` of Spec(Z/n) under inclusion, and a
family covers `D(f)` when its members generate the unit ideal of the
localization `O(D(f)) = (Z/n)[1/f] = Z/m`. The structure sheaf is `D |-> O(D)`,
restriction is reduction mod m', and its ring operations are declared
morphisms. The test checks that the topology from unit ideals agrees with
covers by unions, and that the structure sheaf is a sheaf (the Chinese
remainder theorem). It then forces
`(forall (x O) (or (exists (y O) (= (mul x y) one)) (exists (y O) (= (mul (sub one x) y) one))))`
at every stage of Spec(Z/12) and of Spec(Z/30). The reason is that x and 1 - x
generate the unit ideal, so `D(x)` and `D(1 - x)` cover; x is a unit on
`D(x)` and 1 - x is a unit on `D(1 - x)`. The global rings are not local: in
Z/12 the elements 3, 4, 9 and 10 have neither x nor 1 - x a unit. With the
trivial topology on the same poset, the same formula fails at the global
stage, because the disjunction there must be decided at the stage itself. The
same test shows the role of reducedness. For squarefree n = 30, O is
internally a field in both senses (`x = 0 or x invertible`, and
`not x = 0 implies x invertible`). For n = 12 the stage `D(3) = {2}`, where
`O = Z/4`, refutes both, with the nilpotent 2 as the witness.

## Memory

A query builds its own memo tables, and the compiled formula keeps none. Wrapping a
query in `with-region` releases everything it allocated, for example
`(with-region (car (topos-cross-check (topos-formula L ctx phi))))`.

Types are memoized in the topos. A type first mentioned inside a region
would be stored in the topos's table, so create every type a formula may
name (e.g. `(topos-language-type L '(P X))`) before entering the region. The
corpus test does this.

## Representation and cost

Truth values are computed on bit masks of the arrows into a stage, and the
public procedures take and return ascending arrow lists. Covering is a
hash lookup. Closure is tabulated over all masks at stages with at most 12
incoming arrows, and implication is tabulated over `Omega_J(c)` when it has at
most 64 elements; larger stages compute the same values directly. Masks are
exact integers, so the number of arrows into a stage is not bounded.
Exponentials and power objects are enumerated, so their size, and with it the
cost of quantifying over them, grows quickly with the site. The library is
meant for exact computation on small finite sites.

## Tests

- `tests/stdlib/topos_test.esk`: types, Omega and its Heyting algebra (the
  adjunction at every stage), the Lawvere-Tierney bijection by exhaustion
  (4 topologies out of 108 endomaps on the walking arrow, and back), the
  subobject classifier and exponential adjunction by counting, 27 theorems
  of intuitionistic higher-order logic forced on four sites, and the
  classical laws: excluded middle and double-negation elimination are refuted
  with a counter-witness stage on the walking arrow, the pseudocircle and the
  pseudocircle space. De Morgan's law for not-and is refuted on the latter two
  and holds on the walking arrow (a De Morgan topos), and all of these laws
  hold on BZ/2.
- `tests/stdlib/topos_corpus_test.esk` and `topos_corpus_sheaf_test.esk`:
  every formula up to a size bound over fixed atoms, first-order (size 4,
  1980 formulas) and higher-order (quantifiers over X, P(X) and Omega; size 3,
  276 formulas, and size 2, 32 formulas, where P(X) has 40 or more elements at
  a stage). They check forcing against the Omega-valued truth, monotonicity and
  local character at every stage and every environment of the presheaf
  toposes of the walking arrow, the pseudocircle and BZ/2, and of the sheaf
  toposes of the pseudocircle space and a tree with bars.
- `tests/stdlib/topos_reductions_test.esk`: the Kripke and Beth reductions
  against independent evaluators on the first-order corpus.
- `tests/stdlib/topos_jtranslation_test.esk`: the j-translation theorem of
  [the modality](#the-locally-modality) on the first-order corpus.
- `tests/stdlib/topos_local_ring_test.esk`: the flagship above.
- [`examples/mathematics_presheaf_topos_internal_logic.esk`](../../../examples/mathematics_presheaf_topos_internal_logic.esk)
  uses the library for Omega, exponentials, the Heyting operations and
  Lawvere-Tierney topologies.

Registered in CTest as `<suite>_jit` and `<suite>_aot` for the six suites
above.

## Notes

- Planned for v1.4.1: forcing over sites given by a coverage (generating
  families instead of all covering sieves), so that larger sites need not
  enumerate their covering sieves. The language-level adoption of
  Kripke-Joyal semantics is staged in ADR-0035.
