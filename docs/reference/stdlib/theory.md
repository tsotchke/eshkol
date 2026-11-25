---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/theory.esk
  - lib/core/theory/equational.esk
  - lib/core/theory/free.esk
  - lib/core/theory/finite.esk
  - lib/core/theory/lawvere.esk
  - lib/core/theory/topos.esk
  - lib/core/ring.esk
  - tests/stdlib/theory_test.esk
  - tests/stdlib/theory_topos_test.esk
---

# `core.theory` — algebraic theories and their models

`(require core.theory)` loads algebraic theories as first-class data
([ADR-0039](../../design/adr/0039-lawvere-theories.md)). A **theory** is a
signature (operation symbols with arities) and a list of equational laws; a
**model** is a carrier given by its operations. The library checks laws
(exhaustively on finite models, on seeded samples on infinite ones), finds
homomorphisms, forms products and quotients by congruences, builds the free
model on finitely many generators with a decided word problem, and presents
each theory as its **Lawvere theory**: the category with finite products whose
morphisms n -> m are m-tuples of terms in n variables.

```scheme
(require core.theory)
;; Z/4 under addition, as a model of the theory of groups
(define Z4 (cyclic-group 4))
(display (model-law-violations Z4)) ;; => ()
(newline)
;; the theory of groups proves (xy)^-1 = y^-1 x^-1, but not xy = yx
(display (theory-word-problem group-theory '(inv (mul x y)) '(mul (inv y) (inv x)))) ;; => #t
(newline)
(display (theory-word-problem group-theory '(mul x y) '(mul y x))) ;; => #f
(newline)
;; homomorphisms Z/4 -> Z/2: x -> 0 and x -> x mod 2
(display (length (model-homomorphisms Z4 (cyclic-group 2)))) ;; => 2
```

`core.theory` is not part of `(require stdlib)`. Its parts can be required
alone: `core.theory.equational` (theories, models, laws, constructions;
it loads `core.ring`), `core.theory.free` (free models and the word problem),
`core.theory.finite` (finite models), `core.theory.lawvere` (the Lawvere
category and interpretations), and `core.theory.topos` (the truth values of a
`core.topos` topos as a Heyting algebra; it loads `core.topos`, so the
umbrella does not).

## Terms, equations and theories

A **term** is a variable (a symbol) or an application `(op t ...)` of an
operation of that arity; a constant is written `(one)`. Operation names are
any data compared with `equal?`, so a family such as the scalars of a module,
`(scale r)`, is a set of operation names.

| Procedure | Meaning |
|---|---|
| `(make-signature '((op arity) ...))`, `signature?`, `signature-operations`, `signature-arity` | a signature |
| `(well-formed-term? S-or-T t)` | every application has its operation's arity |
| `(term-variables t)` | variables in order of first appearance |
| `(term-substitute t '((x . s) ...))` | simultaneous substitution |
| `(term-size t)`, `(term-operations t)` | size; operations used |
| `(make-equation name lhs rhs)`, `equation?`, `equation-name`, `equation-lhs`, `equation-rhs`, `equation-variables` | an equational law; its variables are those of the left side then the right |
| `(make-theory name signature equations [kind parameters])` | a theory; equations may be `(name lhs rhs)` lists; an ill-formed side is refused |
| `theory?`, `theory-name`, `theory-signature`, `theory-equations`, `theory-operations`, `theory-arity`, `theory-kind`, `theory-parameters` | accessors; the kind names the decision procedure of `core.theory.free` |
| `(theory-extend T name equations [operations])` | more equations and operations (kind `user`) |

A theory with no equations has kind `absolutely-free`: its models are the
algebras of the signature, and its free model is the term model.

### The catalogue

| Theory | Signature | Laws |
|---|---|---|
| `semigroup-theory` | `mul` | associativity |
| `monoid-theory` | `one mul` | left and right identity, associativity |
| `group-theory` | `one mul inv` | identities, left and right inverses, associativity |
| `abelian-group-theory` | `zero add neg` | identity, inverse, commutativity, associativity |
| `ring-theory` | `zero one add neg mul` | abelian group, monoid (both identities), left and right distributivity |
| `commutative-ring-theory` | `zero one add neg mul` | the eight laws of `core.ring`'s law gate, in its order |
| `(module-theory R)` | `zero add neg` and `(scale r)` for each r of a finite ring model R | abelian group, `1x = x`, `r(x + y)`, `(r + s)x`, `(rs)x` |
| `lattice-theory` | `meet join` | commutativity, associativity, absorption |
| `distributive-lattice-theory` | `meet join` | lattice laws and distributivity |
| `boolean-algebra-theory` | `bottom top meet join complement` | bounded distributive lattice with complements |
| `heyting-algebra-theory` | `bottom top meet join implies` | bounded lattice, `x -> x = 1`, `x /\ (x -> y) = x /\ y`, `y /\ (x -> y) = y`, `x -> (y /\ z) = (x -> y) /\ (x -> z)` |

## Models and the law check

`(make-model T name 'operations alist ['elements list] ['equal? proc] ['member? proc] ['extra alist])`
builds a model; an operation of arity 0 is given by its value, every other by
a procedure. A model with `elements` is finite; its elements are canonical
representatives. A missing operation is refused.

| Procedure | Meaning |
|---|---|
| `model?`, `model-theory`, `model-name`, `model-elements`, `model-finite?`, `model-size`, `model-operation`, `model-equal?`, `model-member?`, `model-extra`, `model-ref` | accessors; `model-extra` is the alist given as `'extra`, `model-ref` one of its values |
| `(model-evaluate M t env)` | the value of a term under an alist of variable values |
| `(term->procedure M t vars)` | the term compiled to a procedure of one vector of values |
| `(model-law-violations M [samples])` | the laws checked under every assignment of the samples (every element when M is finite) |
| `(model-satisfies? M [samples])` | no violations |
| `(equations-violations M equations samples)` | the same check for any list of equations over M's operations |

The law check extends an assignment one variable at a time and, at each
prefix, checks the equations with exactly that many variables. A violation is
`(law value ...)`, the values in the order of the law's variables:

```scheme
(require core.theory)
;; subtraction is not associative and has no left identity
(define Z3- (make-model group-theory "Z/3 under subtraction"
  'operations (list (cons 'one 0) (cons 'mul (lambda (a b) (modulo (- a b) 3)))
                    (cons 'inv (lambda (a) a)))
  'elements '(0 1 2)))
(display (car (model-law-violations Z3-))) ;; => (associativity 0 0 1)
(newline)
(display (length (model-law-violations Z3-))) ;; => 20
```

### Ring carriers are models

Every `core.ring` carrier is a model of `commutative-ring-theory`:
`(ring-carrier->model R [elements])` takes the carrier's own `zero one add neg
mul` and `equal?`; the model of `Z/m` lists its m elements when m <= 4096
(a larger one is checked on samples, like an infinite carrier). The law check
of this library is `core.ring`'s (`law-violations`, `law-term->procedure`),
and the laws of `commutative-ring-theory` are `core.ring`'s
`ring-law-equations`, the laws its `ring-law-violations` checks before the
carrier protocol's own checks (the unit group, `from-integer` as a ring map,
the characteristic). Carriers, their operations and every call site of
`core.ring` are unchanged.

```scheme
(require core.ring)
(require core.theory)
(display (model-law-violations (ring-carrier->model (make-zmod 6)))) ;; => ()
(newline)
;; Z on seeded samples across the int64 boundary
(display (model-satisfies? (ring-carrier->model integers)
                           '(0 1 -1 9223372036854775807 -9223372036854775808 18446744073709551616))) ;; => #t
```

## Homomorphisms, products, congruences and quotients

| Procedure | Meaning |
|---|---|
| `(model-homomorphism-violations f M N [samples])` | the tuples on which f fails to commute with an operation, as `(op argument ...)` |
| `(model-homomorphism? f M N [samples])` | none |
| `(model-homomorphisms M N)` | every homomorphism of finite models, as graphs (alists over M's elements); found from the images of a generating set |
| `(model-isomorphic? M N)` | a bijective homomorphism exists |
| `(finite-map-apply M graph x)` | apply a graph, by M's equality |
| `(model-generators M)`, `(model-subalgebra M generators)`, `(model-submodel M generators)` | a generating set; the generated subalgebra |
| `(model-product M ...)`, `(model-projection P i)` | the product (elements are lists) and its projections |
| `(model-congruence? M partition)` | the classes are respected by every operation |
| `(model-congruence-closure M pairs)` | the least congruence containing the pairs |
| `(model-kernel f M N)` | the partition by images, a congruence |
| `(model-quotient M partition [name])`, `(model-quotient-map M Q)` | the quotient (refused unless the partition is a congruence) and its projection |

```scheme
(require core.theory)
(define groups (small-groups))
(define D4 (list-ref groups 12))
;; the centre of D4 is {1, r^2}; r^2 is encoded 4
(define Q (model-quotient D4 (model-congruence-closure D4 (list (cons 0 4)))))
(display (model-size Q)) ;; => 4
(newline)
(display (model-isomorphic? Q (list-ref groups 4))) ;; => #t
(newline)
(display (model-isomorphic? (model-product (cyclic-group 2) (cyclic-group 3)) (cyclic-group 6))) ;; => #t
```

## Interpretations

An interpretation S -> T sends each operation of S to a term of T:
`(make-interpretation S T (list (list op (x ...) term) ...))`
(`interpretation?`, `interpretation-source`, `interpretation-target`).
`(interpretation-translate I t)` translates terms; `(model-reduct I M)` makes
a T-model an S-model (the forgetful functor); `(interpretation-violations I)`
lists the S-laws whose translation T does not prove, decided by T's word
problem. It is empty exactly when I is a morphism of theories (of Lawvere
theories). `core.theory.lawvere` provides `monoid->semigroup`,
`group->monoid`, `abelian-group->group`, `ring->abelian-group`,
`ring->monoid`, `commutative-ring->ring`, `lattice->heyting-algebra`,
`lattice->distributive-lattice`, `distributive-lattice->boolean-algebra` and
`heyting-algebra->boolean-algebra` (implication as `(join (complement x) y)`),
all morphisms; `group->abelian-group` and `ring->commutative-ring` read the
same symbols in the commutative theory and are not, so the reduct along them
satisfies the laws exactly when the model is commutative.

```scheme
(require core.theory)
(display (interpretation-violations heyting-algebra->boolean-algebra)) ;; => ()
(newline)
(display (interpretation-violations ring->commutative-ring)) ;; => (multiplicative-commutativity)
(newline)
(define S3 (list-ref (small-groups) 7))
(display (model-satisfies? (model-reduct group->abelian-group S3))) ;; => #f
```

## Free models and the word problem

`(theory-free-model T generators)` is the free T-model on a list of generator
symbols; its equality decides the word problem.

| Kind | Elements | Equality |
|---|---|---|
| absolutely free | terms | syntactic |
| semigroups, monoids | words | equal words |
| groups | freely reduced words of `(g . 1)` and `(g . -1)` | equal reduced words |
| abelian groups | integer vectors Z^n | equal vectors |
| rings | Z<X>: integer combinations of words, as `core.dbsp` Z-sets | equal Z-sets |
| commutative rings | Z[X]: `core.ring`'s monoid ring (`zset-group-ring`) on exponent vectors | equal Z-sets |
| lattices | terms | Whitman's conditions for s <= t, both ways |
| distributive lattices, Boolean algebras | truth tables over the 2^n assignments (the varieties are generated by the two-element algebra) | equal tables; listed when at most 4096 elements |
| Heyting algebras | terms | s |- t and t |- s in intuitionistic propositional logic, decided by Dyckhoff's G4ip |
| modules over a finite ring R | vectors R^n | equal vectors |

| Procedure | Meaning |
|---|---|
| `(free-model-generator F g)` | the element of generator g |
| `(free-element->term F x)` | a term over the generators denoting x |
| `(free-model-lift F M images)` | the homomorphism F -> M extending the alist of generator images |
| `(theory-word-problem T s t)` | whether T proves s = t |
| `(theory-normal-form T t [generators])` | the free-model element t denotes |
| `(lattice-leq? s t)`, `(heyting-provable? s t)` | the two decision procedures on terms |

A theory of kind `user` with equations (for instance one made by
`theory-extend`) has no decision procedure here and is refused by name.

```scheme
(require core.theory)
(define F (theory-free-model commutative-ring-theory '(x)))
;; (x + 1)(x - 1) = x^2 - 1 in Z[x]
(display (free-element->term F (theory-normal-form commutative-ring-theory
                                 '(mul (add x (one)) (add x (neg (one)))) '(x)))) ;; => (add (neg (one)) (mul x x))
(newline)
;; free lattices are not distributive; free Heyting algebras refute excluded middle
(display (theory-word-problem lattice-theory '(meet x (join y z)) '(join (meet x y) (meet x z)))) ;; => #f
(newline)
(display (theory-word-problem heyting-algebra-theory '(join x (implies x (bottom))) '(top))) ;; => #f
(newline)
(display (model-size (theory-free-model distributive-lattice-theory '(a b c)))) ;; => 18
```

## The Lawvere theory

`(lawvere-category T)` is L_T (`lawvere-category?`,
`lawvere-category-theory`; morphisms answer `lawvere-morphism?`). A morphism n -> m is
`(lawvere-morphism L n terms)`, m terms in the variables
`(lawvere-variables n)` = `x1 ... xn`.

| Procedure | Meaning |
|---|---|
| `lawvere-source`, `lawvere-target`, `lawvere-terms` | the morphism's data |
| `(lawvere-identity L n)` | the variables |
| `(lawvere-compose g f)` | g after f: f's components substituted for g's variables |
| `(lawvere-projections L n m)` | the projections of the product n + m |
| `(lawvere-pairing f g)` | the morphism into the product |
| `(lawvere-morphism=? f g)` | componentwise equality by the word problem |
| `(lawvere-apply M f xs)` | the model M as a product-preserving functor: f acting on M^n |
| `(lawvere-free-homomorphism f)` | `(F(m) F(n) h)`: f : n -> m as the homomorphism h : F(m) -> F(n) of free models (L_T is the opposite of finitely generated free models) |
| `(lawvere-truncation T N)` | when the free models on at most N generators are finite, the full subcategory on 0 .. N as a `core.sheaf.site` finite category, the representation `core.category` builds on |

```scheme
(require core.theory)
(define L (lawvere-category group-theory))
(define f (lawvere-morphism L 2 '((mul x1 x2) (inv x1))))
(define g (lawvere-morphism L 2 '((mul x2 x1))))
;; g f (x1, x2) = x1^-1 (x1 x2) = x2
(display (lawvere-morphism=? (lawvere-compose g f) (lawvere-morphism L 2 '(x2)))) ;; => #t
(newline)
(display (lawvere-apply (cyclic-group 5) f '(2 4))) ;; => (1 3)
```

The symbolic category part is what L_T needs; finite truncations use the
shared finite-category representation, so `category-laws?` of
`core.sheaf.site` and the constructions of `core.category` apply to them.

## Finite models

| Constructor | Model |
|---|---|
| `(finite-group name n mul)` | a group on 0 .. n-1 from its multiplication (identity and inverses found) |
| `(cyclic-group n)`, `(dihedral-group m [name])`, `(quaternion-group)`, `(group-direct-product G H)` | Z/n, D_m of order 2m, Q8, G x H |
| `(small-groups)` | the fourteen groups of order at most eight: 1, Z2, Z3, Z4, Z2xZ2, Z5, Z6, S3, Z7, Z8, Z4xZ2, Z2xZ2xZ2, D4, Q8 |
| `(zmod-ring-model n)`, `(matrix-ring-model R k)` | Z/n through `core.ring`; k x k matrices over a finite ring model |
| `(make-module R name elements zero add neg scale)`, `(regular-module R)` | a finite R-module; R over itself |
| `(finite-lattice name elements leq)`, `(lattice-order L)` | meet and join of a finite order (refused unless a lattice) |
| `(chain-lattice k)`, `(divisor-lattice n)`, `(diamond-lattice)`, `(pentagon-lattice)` | chains, divisors under gcd and lcm, M3, N5 |
| `(finite-heyting-algebra name elements leq)` | with relative pseudo-complements (refused when one does not exist) |
| `(powerset-boolean-algebra n)`, `(divisor-boolean-algebra n)` | subsets of n points; divisors of a squarefree n |

## Finite coalgebras

A coalgebra for F X = O x X^A (a Moore machine) is
`(make-coalgebra states alphabet output step)` (`coalgebra?`,
`coalgebra-states`, `coalgebra-alphabet`, `(coalgebra-output C x)`,
`(coalgebra-step C x a)`).
`coalgebra-homomorphism?` checks outputs and transitions;
`coalgebra-bisimilarity` is the largest bisimulation by partition
refinement and `(coalgebra-bisimilar? C x y)` asks it of two states; `coalgebra-minimize` is the quotient by it (the image in the final
coalgebra); `coalgebra-sum` is the disjoint sum and
`(coalgebra-behaviour C x n)` the outputs after every word of length at most n.

```scheme
(require core.theory)
;; count the letter a modulo 3, on six states
(define C (make-coalgebra '(0 1 2 3 4 5) '(a)
                          (lambda (x) (if (= 0 (modulo x 3)) 1 0))
                          (lambda (x l) (modulo (+ x 1) 6))))
(display (coalgebra-bisimilarity C)) ;; => ((0 3) (1 4) (2 5))
(newline)
(display (coalgebra-states (coalgebra-minimize C))) ;; => (0 1 2)
```

## Truth values of a topos

`(require core.theory.topos)` adds `(topos-omega-model T c)`: the closed
sieves Omega_J(c) of a `core.topos` topos with its own meet, join, implication
and bounds, a finite model of `heyting-algebra-theory`;
`(topos-omega-boolean-model T c)` reads it with complement `x -> bottom` as a
candidate Boolean algebra, lawful exactly when `topos-omega-boolean?` holds;
and `(topos-omega-restriction T f)` is pullback of sieves along f, a
homomorphism of Heyting algebras.

```scheme
(require core.sheaf.site)
(require core.topos)
(require core.theory)
(require core.theory.topos)
(define T (presheaf-topos (poset-category 2 (lambda (x y) (<= x y)))))
(define Omega (topos-omega-model T 1))
(display (model-size Omega)) ;; => 3
(newline)
(display (model-satisfies? Omega)) ;; => #t
(newline)
(display (model-satisfies? (topos-omega-boolean-model T 1))) ;; => #f
```

## Refusals

Refusals are R7RS error objects; `theory-refusal?`, `theory-refusal-kind`
and `theory-refusal-irritants` read them.

| Kind | When |
|---|---|
| `theory-ill-formed-term` | a term with an unknown operation or an arity that does not match |
| `theory-missing-operation` | a model or interpretation without one of the operations |
| `theory-infinite-model` | the elements of an infinite model, or a law check without samples |
| `theory-not-a-congruence` | a quotient by a partition the operations do not respect |
| `theory-no-decision-procedure` | a free model of a theory with equations and no decision procedure |
| `theory-not-a-lattice` | a finite order without meets or joins, or without a relative pseudo-complement |

## References

- F. W. Lawvere, *Functorial Semantics of Algebraic Theories*, PhD thesis,
  Columbia University (1963); Reprints in Theory and Applications of
  Categories 5 (2004).
- J. Adamek, J. Rosicky, E. M. Vitale, *Algebraic Theories: A Categorical
  Introduction to General Algebra*, Cambridge Tracts in Mathematics 184
  (2011).
- nLab, "Lawvere theory", <https://ncatlab.org/nlab/show/Lawvere+theory>.
- S. Burris, H. P. Sankappanavar, *A Course in Universal Algebra* (1981),
  chapter II.
- P. M. Whitman, "Free lattices", Annals of Mathematics 42 (1941).
- R. Dyckhoff, "Contraction-free sequent calculi for intuitionistic logic",
  Journal of Symbolic Logic 57 (1992).
- J. J. M. M. Rutten, "Universal coalgebra: a theory of systems",
  Theoretical Computer Science 249 (2000).
- S. Mac Lane, I. Moerdijk, *Sheaves in Geometry and Logic* (1992).
