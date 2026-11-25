---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/monoidal.esk
  - lib/core/monoidal/scalars.esk
  - lib/core/monoidal/finvect.esk
  - lib/core/monoidal/network.esk
  - lib/core/monoidal/structure.esk
  - lib/core/monoidal/vect.esk
  - lib/core/monoidal/diagram.esk
  - tests/stdlib/monoidal_test.esk
---

# `core.monoidal` — monoidal categories and string diagrams

**Source:** [monoidal.esk](../../../lib/core/monoidal.esk) and the modules in
[lib/core/monoidal/](../../../lib/core/monoidal/).
**Require:** `(require core.monoidal)` for everything, or one layer:
`core.monoidal.scalars`, `core.monoidal.finvect`, `core.monoidal.network`,
`core.monoidal.structure`, `core.monoidal.vect`, `core.monoidal.diagram`,
`core.monoidal.zx` (each loads the layers it builds on). Not part of
`(require stdlib)`.
**Design record:** [ADR-0041](../../design/adr/0041-core-monoidal-string-diagrams-zx.md).
**ZX-calculus:** [zx.md](zx.md).

`core.monoidal` computes with tensor products exactly. Monoidal, braided,
symmetric, compact closed and dagger structures are data, and every
coherence law is a check that returns its violations: the pentagon and the
triangle, the two hexagons, symmetry, the snake equations, the dagger laws,
and the coherence of monoidal functors. The structures live on the finite
categories of [`core.category`](category.md) and on FinVect, the category of
finite-dimensional spaces over an exact carrier (Z, Q or a cyclotomic field
Q(zeta_n)). String diagrams are a syntax with two evaluations in FinVect
(compositional, and as a tensor network) and a normal form that decides
equality in the free symmetric monoidal category with self-dual compact
structure.

Sources: S. Mac Lane, *Categories for the Working Mathematician*, 2nd ed.
(1998) (CWM), VII and XI; A. Joyal, R. Street, "The geometry of tensor
calculus I", Adv. Math. 88 (1991), and "Braided tensor categories", Adv.
Math. 102 (1993); G. M. Kelly, M. L. Laplaza, "Coherence for compact closed
categories", JPAA 19 (1980); P. Selinger, "A survey of graphical languages
for monoidal categories", in *New Structures for Physics*, LNP 813 (2011);
B. Coecke, A. Kissinger, *Picturing Quantum Processes* (2017).

## Representation

| Value | Representation |
|---|---|
| carrier | `#(exact-scalars ...)`: `(integer-scalars)`, `(rational-scalars)`, `(cyclotomic-scalars n)`; elements are exact numbers or core.cyclotomic elements |
| FinVect object | a list of positive dimensions `(d1 ... dk)`; `()` is the unit; tensor is `append` |
| linear map | `#(linear-map S dom cod rows)`, rows a vector of rows (a core.integer_linalg matrix over Z and Q) |
| tensor | `#(tensor S legs dims entries)`, legs labelled, entries flat, first leg most significant |
| category operations | `#(category-ops objects arrows source target compose identity equal? inverse)` |
| monoidal category | `#(monoidal-category ops tensor tensor-arrows unit associator left-unitor right-unitor braiding name)` |
| monoidal functor | `#(monoidal-functor M N objects arrows mu epsilon)` |
| string diagram | `#(diagram kind dom cod data)`, wires typed by any datum |
| open graph | `#(open-graph inputs outputs nodes edges loops)` |

The basis vector |i1 ... ik> of `(d1 ... dk)` has the flat index
((i1 d2 + i2) d3 + ...) + ik, so `(linear-map-tensor f g)` is the Kronecker
product with f's factor first. The compact structure is the symmetric
self-dual one of the standard basis: the cup on A is sum_a |a a> and the cap
its transpose; the dagger is the conjugate transpose.

## FinVect

nLab: [Vect](https://ncatlab.org/nlab/show/Vect),
[compact closed category](https://ncatlab.org/nlab/show/compact+closed+category),
[dagger-compact category](https://ncatlab.org/nlab/show/dagger-compact+category).
CWM VII.1; Coecke-Kissinger 4-5; Selinger 2007.

```scheme
(require stdlib)
(require core.monoidal.finvect)
(define Q (rational-scalars))
(define f (make-linear-map Q '(2) '(2) (vector (vector 1 2) (vector 3 4))))
(define g (make-linear-map Q '(2) '(2) (vector (vector 0 1) (vector 1 0))))
(linear-map-rows (linear-map-compose g f))                 ;; => #(#(3 4) #(1 2))
(linear-map-codomain (linear-map-tensor f g))              ;; => (2 2)
(vector-ref (linear-map-rows (linear-map-tensor f g)) 1)   ;; => #(1 0 2 0)
;; the snake equation (cap (x) id) o (id (x) cup) = id
(define snake (linear-map-compose (linear-map-tensor (linear-map-cap Q '(3)) (linear-map-identity Q '(3)))
                                  (linear-map-tensor (linear-map-identity Q '(3)) (linear-map-cup Q '(3)))))
(linear-map-equal? snake (linear-map-identity Q '(3)))     ;; => #t
;; the swap is natural: sigma o (f (x) g) = (g (x) f) o sigma
(linear-map-equal? (linear-map-compose (linear-map-swap Q '(2) '(2)) (linear-map-tensor f g))
                   (linear-map-compose (linear-map-tensor g f) (linear-map-swap Q '(2) '(2))))  ;; => #t
(linear-map-rows (linear-map-inverse f))                   ;; => #(#(-2 1) #(3/2 -1/2))
```

| Procedure | Meaning |
|---|---|
| `(linear-map S dom cod entry)` | the map with entries `(entry r c)` |
| `(make-linear-map S dom cod rows)` | from a vector of rows |
| `linear-map-identity`, `linear-map-zero` | identities and zero maps |
| `(linear-map-compose g f)` | g o f; `(linear-map-compose-all f1 ... fk)` in diagram order |
| `(linear-map-tensor f g)`, `linear-map-tensor-all` | Kronecker product |
| `(linear-map-swap S A B)`, `(linear-map-permutation S obj perm)` | the braiding and wire permutations (output j is input `perm[j]`) |
| `(linear-map-cup S A)`, `(linear-map-cap S A)` | the compact structure |
| `linear-map-dagger`, `linear-map-transpose`, `linear-map-conjugate` | dagger structure |
| `linear-map-equal?`, `linear-map-proportional`, `linear-map-inverse` | equality, the scalar c with f = c g, Gauss-Jordan inverse (#f when none) |
| `linear-map-trace`, `linear-map-apply`, `linear-map-state`, `linear-map-effect`, `linear-map-basis-state`, `linear-map-scalar` | traces, application, states and effects |

The carriers: `(integer-scalars)` and `(rational-scalars)` use exact Scheme
numbers, `(cyclotomic-scalars n)` uses [`core.cyclotomic`](../../../lib/core/cyclotomic.esk)
with `scalars-root` for zeta_n^k, `scalars-phase` for e^{i pi p},
`scalars-sqrt2` (8 | n) and `scalars-conj` for complex conjugation.

## Tensor networks

nLab: [string diagram](https://ncatlab.org/nlab/show/string+diagram). R. Penrose (1971).

```scheme
(require stdlib)
(require core.monoidal.network)
(define Q (rational-scalars))
(define f (make-linear-map Q '(2) '(2) (vector (vector 1 2) (vector 3 4))))
(define g (make-linear-map Q '(2) '(2) (vector (vector 0 1) (vector 1 0))))
;; g o f: f's output leg and g's input leg share the label m
(define gf (network-evaluate Q (list (linear-map->tensor f '(m) '(x)) (linear-map->tensor g '(y) '(m))) '(y) '(x)))
(linear-map-rows gf)                                        ;; => #(#(3 4) #(1 2))
;; a self-loop is a trace
(tensor-scalar-value (network-contract Q (list (linear-map->tensor f '(z) '(z))) '()))  ;; => 5
```

`(network-contract S tensors open)` contracts every label that occurs twice
(greedily, the pair with the smallest result first) and returns the tensor
with legs `open`; `(network-evaluate S tensors outs ins)` reads it as the map
ins -> outs. `tensor-permute`, `tensor-contract`, `tensor-trace`,
`linear-map->tensor` and `tensor->linear-map` are the steps.

## Monoidal categories

nLab: [monoidal category](https://ncatlab.org/nlab/show/monoidal+category),
[pentagon identity](https://ncatlab.org/nlab/show/pentagon+identity),
[braided monoidal category](https://ncatlab.org/nlab/show/braided+monoidal+category),
[symmetric monoidal category](https://ncatlab.org/nlab/show/symmetric+monoidal+category),
[monoidal functor](https://ncatlab.org/nlab/show/monoidal+functor),
[2-group](https://ncatlab.org/nlab/show/2-group). CWM VII.1-2, XI.1-2;
Joyal-Street 1993; Etingof et al., *Tensor Categories* (2015), 2.1-2.6, 8.1.

A monoidal category is a record of procedures over a category given by its
operations; the laws are checked over the listed objects and arrows (all of
them for a finite category). The constructors on finite categories are
`(finite-monoidal-category C T unit a l r [braiding])` with `T` a
core.category functor on `(category-product C C)`,
`(monoid-monoidal-category n mul e)`, `(meet-monoidal-category P)` and
`(cocycle-monoidal-category n mul e m omega [c])`: the skeletal category of
a finite group G with every Hom(g, g) = Z/m, tensor the group law, associator
omega(g, h, k). Its pentagon holds exactly when omega is a 3-cocycle, and a
braiding c satisfies the hexagons exactly when (omega, c) is an abelian
3-cocycle.

```scheme
(require stdlib)
(require core.monoidal.structure)
(define (z2 a b) (modulo (+ a b) 2))
;; omega(1,1,1) = 1 is the non-trivial class of H^3(Z/2; Z/2)
(define M (cocycle-monoidal-category 2 z2 0 2 (lambda (x y z) (* x y z))))
(monoidal-category-violations M)          ;; => ()
(monoidal-strict? M)                      ;; => #f
;; a function that is not a cocycle fails the pentagon, with witnesses
(define N (cocycle-monoidal-category 2 z2 0 2 (lambda (x y z) (if (and (= x 1) (= y 1) (= z 0)) 1 0))))
(car (pentagon-violations N))             ;; => (pentagon 1 1 0 0)
;; the semion: Z/2 with values in Z/4, omega(1,1,1) = 2, c(1,1) = 1
(define semion (cocycle-monoidal-category 2 z2 0 4 (lambda (x y z) (* 2 x y z)) (lambda (x y) (* x y))))
(braided-monoidal-violations semion)      ;; => ()
(symmetry-violations semion)              ;; => ((symmetry 1 1))
```

| Check | Law |
|---|---|
| `monoidal-tensor-violations` | the tensor is a functor (types, identities, interchange) |
| `monoidal-naturality-violations`, `monoidal-isomorphism-violations` | the associator and unitors are natural isomorphisms |
| `pentagon-violations`, `triangle-violations` | Mac Lane's coherence axioms |
| `monoidal-category-violations` | all of the above |
| `braiding-naturality-violations`, `hexagon-violations`, `symmetry-violations` | braided and symmetric structure |
| `braided-monoidal-violations`, `symmetric-monoidal-violations` | the whole structure |
| `compact-closed-violations M cup cap` | the two snake equations (self-dual objects) |
| `dagger-violations M dagger`, `dagger-compact-violations M dagger cup cap` | dagger laws, unitary structure maps, cap = cup^dagger, symmetric cups |
| `monoidal-functor-violations`, `monoidal-functor-associativity-violations`, `monoidal-functor-unit-violations`, `braided-monoidal-functor-violations` | monoidal functors |

Each check returns a list of witnesses `(law object ...)`, empty when the law
holds.

## FinVect as a monoidal category

`(finvect-monoidal-category S objects arrows)` is the strict symmetric
monoidal FinVect_S, checked over the given objects and arrows;
`finvect-dagger`, `(finvect-cup S)` and `(finvect-cap S)` give its dagger
compact structure. `(bracketed-finvect-monoidal-category S objects arrows)`
is the same category on bracketed objects (an integer dimension, `()`, or
`#(x y)`), where (x (x) y) (x) z and x (x) (y (x) z) are different objects
joined by the associator; `(bracketed-objects-of-size factors)` lists every
bracketing. `(flatten-monoidal-functor S B F)` forgets the brackets (a strong
monoidal equivalence, Mac Lane's coherence theorem) and
`(base-change-monoidal-functor M N T phi)` changes the carrier.

```scheme
(require stdlib)
(require core.monoidal.vect)
(define Q (rational-scalars))
(define f (make-linear-map Q '(2) '(2) (vector (vector 1 2) (vector 3 4))))
(define F (finvect-monoidal-category Q (list '() '(2)) (list f)))
(symmetric-monoidal-violations F)                                         ;; => ()
(dagger-compact-violations F finvect-dagger (finvect-cup Q) (finvect-cap Q))  ;; => ()
(length (bracketed-objects-of-size (list 2 2 2 2)))                       ;; => 5
(define B (bracketed-finvect-monoidal-category Q (list '() 2 (vector 2 '())) (list (make-bracketed-arrow 2 2 f))))
(list (symmetric-monoidal-violations B) (monoidal-strict? B))             ;; => (() #f)
```

## String diagrams

nLab: [string diagram](https://ncatlab.org/nlab/show/string+diagram),
[compact closed category](https://ncatlab.org/nlab/show/compact+closed+category).
Joyal-Street 1991; Kelly-Laplaza 1980; Selinger 2011.

A diagram is built from `(diagram-box name dom cod)`,
`(diagram-identity types)`, `(diagram-then d1 d2 ...)` (first d1),
`(diagram-tensor d1 d2 ...)`, `(diagram-swap A B)`, `(diagram-cup A)` and
`(diagram-cap A)`; a composite whose wires do not match is refused.
`(diagram-evaluate d S dims boxes)` evaluates it in FinVect, `dims` and
`boxes` being procedures or association lists from wire types to dimensions
and from box names to linear maps. `diagram-evaluate-network` evaluates the
same diagram as the tensor network of its open graph.

`(diagram-normal-form d)` is the open graph with its boxes numbered
canonically (breadth first from the boundary, closed components from the
root with the least encoding), and two diagrams have the same normal form
exactly when their open graphs are isomorphic: equality in the free
symmetric monoidal category with self-dual compact structure on the boxes.
Equal normal forms imply equal values in every interpretation;
`diagram-equal-by-evaluation?` is the semantic comparison in one
interpretation.

```scheme
(require stdlib)
(require core.monoidal.diagram)
(define Q (rational-scalars))
(define f (diagram-box 'f '(A) '(B)))
(define g (diagram-box 'g '(B) '(A)))
(define dims (list (cons 'A 2) (cons 'B 2)))
(define boxes (list (cons 'f (make-linear-map Q '(2) '(2) (vector (vector 1 2) (vector 3 4))))
                    (cons 'g (make-linear-map Q '(2) '(2) (vector (vector 0 1) (vector 1 0))))))
;; interchange: (f (x) g) ; (g (x) f) = (f ; g) (x) (g ; f)
(define d1 (diagram-then (diagram-tensor f g) (diagram-tensor g f)))
(define d2 (diagram-tensor (diagram-then f g) (diagram-then g f)))
(diagram-equal-by-normal-form? d1 d2)                 ;; => #t
(diagram-equal-by-evaluation? d1 d2 Q dims boxes)     ;; => #t
;; a snake straightens; a closed loop is the dimension
(define snake (diagram-then (diagram-tensor (diagram-identity '(A)) (diagram-cup '(A)))
                            (diagram-tensor (diagram-cap '(A)) (diagram-identity '(A)))))
(diagram-equal-by-normal-form? snake (diagram-identity '(A)))   ;; => #t
(linear-map-scalar-value (diagram-evaluate (diagram-then (diagram-cup '(B)) (diagram-cap '(B))) Q dims boxes))  ;; => 2
(linear-map-equal? (diagram-evaluate d1 Q dims boxes) (diagram-evaluate-network d1 Q dims boxes))  ;; => #t
(diagram-equal-by-normal-form? (diagram-then f g) (diagram-identity '(A)))  ;; => #f
```

`diagram-dagger` mirrors a diagram (a box f becomes `(dagger f)`),
`diagram-boxes` lists the boxes, and `diagram->open-graph` gives the graph:
nodes, edges between ports `(b in k)`, `(b out k)`, `(n i in k)`,
`(n i out k)`, and the wire types of closed loops.

## Verification

`tests/stdlib/monoidal_test.esk` (CTest `monoidal_test_jit`,
`monoidal_test_aot`) checks FinVect over Q and Q(zeta_8) as a symmetric
dagger compact category, the bracketed FinVect and the flattening functor,
discrete, cartesian and cocycle monoidal categories, the pentagon against
group cohomology (exactly 8 of the 256 functions (Z/2)^3 -> Z/2 satisfy it,
the number |Z^3| that H^k(Z/2; F_2) computed by `core.category.linear`
predicts), monoidal functors between cocycle categories (associativity
exactly when omega' = omega - delta nu, 128 of 1024 cases), braidings
(4 of 16 normalized pairs on (Z/2, Z/4), the quadratic forms), and string
diagrams: the axioms by normal form and by evaluation, compositional against
tensor-network evaluation on random diagrams, and random rewrites by the
axioms preserving normal form and value.
