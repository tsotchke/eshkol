---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/ring.esk
  - lib/core/algebraic_domain.esk
  - tests/stdlib/ring_test.esk
  - tests/stdlib/ring_ieee_test.esk
  - lib/core/theory/equational.esk
---

# `core.ring` — ring carriers

`(require core.ring)` loads the ring-carrier protocol of
[ADR-0034](../../design/adr/0034-ring-carriers-functor-of-points.md) (the
functor-of-points evaluation model), at library level. A **ring carrier** is a
commutative ring with identity given as data. A program written once against
the generic operations `r+ r- r* r-neg r-inv r/ r-expt r= r-int` is a
polynomial (or rational) map, and running it over a carrier `R` takes its
`R`-points: ordinary, exact, modular, dual-number, Taylor-jet, polynomial and
group-ring evaluation are the same program over different carriers.

```scheme
(require core.ring)
;; f(x) = x^3 - 2x + 5, written once against the carrier R
(define (f R x) (r+ R (r- R (r-expt R x 3) (r* R 2 x)) 5))
(display (f integers 10)) ;; => 985
(newline)
(display (f rationals 2/3)) ;; => 107/27
(newline)
(display (f flonums 0.5)) ;; => 4.125
(newline)
(display (f (make-zmod 7) 3)) ;; => 5
(newline)
(define J (make-jets rationals 3))
;; f(2 + e) in Q[e]/e^3: the Taylor coefficients f(2), f'(2), f''(2)/2
(display (jet-coefficients (f J (jet-seed J 2)))) ;; => (9 10 6)
(newline)
(define Qx (make-polys rationals))
;; f at the generic point x: its coefficients, lowest degree first
(display (ring-poly-coefficients (f Qx (ring-poly-variable Qx)))) ;; => (5 -2 0 1)
(newline)
;; over the native tower the compiler's own AD operators see through it
(display (derivative (lambda (x) (f numeric-tower x)) 2)) ;; => 10
```

`core.ring` is not part of `(require stdlib)`. Its companion
[`core.multimodular`](multimodular.md) computes exact answers over Spec Z in the
fibres F_p.

## Carriers

| Carrier | Ring | Elements | Notes |
|---|---|---|---|
| `integers` | Z | exact integers (int64 promoting to bignum) | units +-1; `r/` by a non-unit promotes to Q (R7RS), by zero refuses; ordered; euclidean |
| `rationals` | Q | exact rationals | a field; ordered; an inexact operand is refused, never rounded |
| `flonums` | binary64 | IEEE 754 doubles, including -0.0, the infinities, NaN | the **IEEE tier**: specified per operation, `lawful?` is `#f` (see [The IEEE tier](#the-ieee-tier)) |
| `numeric-tower` | the native tower | anything the built-in operators accept | R7RS contagion; carries the compiler's AD values, so `derivative`, `gradient` and `taylor` differentiate programs written on `core.ring`; not lawful |
| `(make-zmod m)` | Z/m, m >= 1 | integers in `[0, m)` | a field exactly when m is prime (decided by deterministic Miller-Rabin below 3317044064679887385961981; above it a modulus not proven composite is refused) |
| `(make-jets K n [e])` | K[e]/e^n | `n` coefficients in K | n = 2: dual numbers (forward-mode AD); n = k+1: k-jets; a unit when the constant term is a unit of K |
| `(make-polys K [x])` | K[x] | coefficient lists, lowest degree first | a unit when the constant term is a unit and every other coefficient is nilpotent; euclidean when K is a field |
| `(zset-group-ring op identity inverse [name] [options])` | Z[G], G abelian | [`core.dbsp`](dbsp.md) Z-sets | the Z-set sum is the ring sum; the product is the DBSP join on the trivial key emitting `(op g h)`; the unit group is decided exactly for a finite G (`'elements`) or a torsion-free G (`'torsion-free #t`), see [Jets, polynomials and group rings](#jets-polynomials-and-group-rings) |
| `(algebraic-domain owner)` | Q(alpha) and its exact Taylor / hyper-dual lifts | algebraic elements and AD values | from [`core.algebraic_domain`](algebraic_linalg.md); see [Interoperation](#interoperation) |

Parameterised carriers whose parameters are canonical data are interned once
per process, so `(eq? (make-zmod 7) (make-zmod 7))` holds. Carriers that take
an object as a parameter (an algebraic field, an AD context, a group given by
procedures) compare by that object's identity with `ring-carrier=?` and are not
retained by the registry. The optional generator symbol of `make-jets` (default
`e`) and variable of `make-polys` (default `x`) name different carriers:
`Q[e]/e^2` and `Q[d]/d^2` meet only by explicit nesting,
`(make-jets (make-jets rationals 2 'e) 2 'd)`.

## The descriptor

A carrier is a versioned descriptor. `(ring-ref R key)` reads a field:

| Key | Meaning |
|---|---|
| `zero` `one` `add` `neg` `mul` `sub` | the identities and ring operations (`sub` is derived) |
| `from-integer` | the unique ring map Z -> R |
| `equal?` `unit?` `inverse` | decidable equality; the unit group (`inverse` refuses a non-unit) |
| `characteristic` | n >= 0 with ker(Z -> R) = nZ |
| `field?` `lawful?` `exact?` `tier` | every nonzero element a unit; the ring laws hold as equalities; elements exact; `lawful`, `ieee` or `tower` |
| `nilpotency` `base` `primal` | n for K[e]/e^n, the base carrier, the map e -> 0 |
| `from-rational` `admit-rational` `constant` | the implicit map Q -> R (a Q-algebra), the explicit partial map Q -> Z/m, the map K -> R for a carrier over K |
| `order` `euclidean` `analytic` `nilpotent?` `divide` | optional capabilities; absent capabilities are `#f` |
| `integral-domain` `exact-quotient` | no zero divisors (declared by Z and by K[x] over a domain; every field is one); `(q a b)` = a / b when b divides a |
| `name` `family` `family-id` `parameters` `abi-version` | identity |

`(ring-carrier? x)` recognises a descriptor. `(ring-carrier-name R)`,
`(ring-carrier-family R)`, `(ring-carrier-family-id R)`,
`(ring-carrier-parameters R)` and `(ring-carrier-abi-version R)` read the same
fields directly, and `(ring-carrier-extra R key)` reads a family's own data
(`#f` when absent). The descriptor ABI version is `(ring-abi-version)`, 1: adding
a family or a capability keeps it, changing the descriptor layout or an
element representation increments it. Families have stable ids that are never
reused, listed by `(ring-families)`: 1 `integers`, 2 `rationals`, 3 `flonums`,
4 `zmod`, 5 `jets`, 6 `polys`, 7 `group-ring`, 8 `algebraic`,
9 `numeric-tower`, 64 `user`. `(ring-registered-carriers)` lists the interned
instances.

```scheme
(require core.ring)
(display (ring-ref (make-zmod 12) 'characteristic)) ;; => 12
(newline)
(display (ring-ref (make-zmod 13) 'field?)) ;; => #t
(newline)
(display (ring-carrier-family-id (make-jets rationals 2))) ;; => 5
(newline)
(display (ring-carrier-name (make-jets (make-zmod 7) 6))) ;; => Z/7[e]/e^6
(newline)
(display (eq? (make-zmod 7) (make-zmod 7))) ;; => #t
```

## Generic operations

Every operation takes the carrier first and admits each operand into it
(see [Coercion](#coercion)).

| Procedure | Meaning |
|---|---|
| `(r0 R)` `(r1 R)` | zero and one |
| `(r+ R a b)` `(r- R a b)` `(r* R a b)` `(r-neg R a)` | ring operations |
| `(r-int R n)` | the image of the exact integer n |
| `(r= R a b)` `(r-zero? R a)` `(r-one? R a)` | equality with the carrier's `equal?` |
| `(r-unit? R a)` `(r-inv R a)` | the unit group; `r-inv` refuses a non-unit (`ring-not-a-unit`) |
| `(r/ R a b)` | `a * b^-1`; the IEEE tier divides once, and Z promotes a non-unit divisor to Q |
| `(r-expt R a n)` | `a^n` for an exact integer n (negative n needs a unit) |
| `(r-sum R xs)` `(r-product R xs)` | folds |
| `(r< R a b)` | needs the `order` capability (Z, Q, binary64, the tower) |
| `(r-quotient R a b)` `(r-remainder R a b)` `(r-gcd R a b)` | need `euclidean` (Z; K[x] over a field); the gcd is nonnegative over Z and monic over K[x] |
| `(r-analytic R name x)` | `exp log sin cos tan atan sqrt` on binary64 and the tower |
| `(ring-primal R x)` | the constant term of a jet, which branches and comparisons read |

## Coercion

An operand of `(r+ R a b)` is admitted into R by the lattice of ADR-0034
section 5:

1. an element of R is used as it is;
2. an exact integer enters every carrier through `from-integer` (Z is
   initial), so a program constant means the same thing in every ring;
3. an exact rational enters a Q-algebra (Q, Q[e]/e^n, Q[x], Q(alpha) and its
   lifts) implicitly; it enters Z/m only by the explicit `(ring-admit R q)`,
   which maps n/d to n * d^-1 and refuses when d is not a unit mod m;
4. an element of K enters K[e]/e^n, K[x] and the other carriers over K as a
   constant, along any chain of bases;
5. an exact number meeting a flonum is rounded once to nearest, ties to even
   (R7RS contagion); a flonum never enters a lawful carrier;
6. otherwise there is no common carrier and the operation raises
   `ring-no-common-carrier` naming both carriers.

Elements of Z, Q, binary64, jets, polynomials and algebraic carriers record
their carrier in their representation. An element of Z/m or of a group ring is
a bare integer or Z-set, which by itself is an element of Z or of the free
Z-module on its rows; `(ring-element R x)` records that it lives in R,
`(ring-element? x)` recognises such a located element, and
`(ring-element-value x)` returns the bare value. `(ring-member? R x)` holds
when x is already in R's own representation. `(ring-carrier-of x)` is the
carrier a value names by itself. `(ring-join A B)` is the carrier both enter,
and `(ring-binary op a b)` (op one of `add sub mul div`) and
`(ring-equal? a b)` dispatch on located operands through the join.

```scheme
(require core.ring)
(define F7 (make-zmod 7))
(define F11 (make-zmod 11))
(display (r+ F7 5 4)) ;; => 2
(newline)
(display (r-inv F7 3)) ;; => 5
(newline)
(display (ring-admit F7 1/3)) ;; => 5
(newline)
(display (ring-element-value (ring-binary 'add (ring-element F7 5) 4))) ;; => 2
(newline)
(display (ring-binary 'add 1/4 0.5)) ;; => 0.75
(newline)
(display (ring-carrier-name (ring-join integers (make-jets F7 2)))) ;; => Z/7[e]/e^2
(newline)
(write (guard (e ((ring-refusal? e) (list (ring-refusal-kind e) (ring-refusal-irritants e))))
  (ring-binary 'add (ring-element F7 3) (ring-element F11 3)))) ;; => (ring-no-common-carrier (ring-join "Z/7" "Z/11"))
(newline)
(write (guard (e ((ring-refusal? e) (list (ring-refusal-kind e) (ring-refusal-irritants e))))
  (r+ F7 1/2 1))) ;; => (ring-no-common-carrier (r+ "Q" "Z/7"))
(newline)
(write (guard (e ((ring-refusal? e) (list (ring-refusal-kind e) (ring-refusal-irritants e))))
  (r+ rationals 0.5 1))) ;; => (ring-no-common-carrier (r+ "R (binary64)" "Q"))
```

## Refusals

A refusal is an R7RS error object whose message names the condition and whose
irritants name the operation, the carrier and the element. It never returns a
substitute value; a carrier of fractions is a separate carrier reached by an
explicit map, never by a division that happened to fail.

| Condition | Irritants | Raised by |
|---|---|---|
| `ring-not-a-unit` | operation, carrier, element | `r-inv` / `r/` of a non-unit; `ring-admit` of n/d with d not a unit; elimination without a unit pivot |
| `ring-no-common-carrier` | operation, carrier, carrier | rule 6 of [Coercion](#coercion) |
| `ring-not-an-element` | operation, carrier, value | a value that is in no carrier (a symbol, a string) |
| `ring-missing-capability` | operation, carrier, capability | `r<` without `order`, `r-gcd` without `euclidean`, `r-analytic` without `analytic`, a unit question in Z[G] for an undeclared G (`unit-group`), the law check on a carrier that is not lawful |
| `ring-no-ring-map` | operation, source, target | a reduction or composition with no ring map |

`(ring-refusal? e)`, `(ring-refusal-kind e)` (a symbol) and
`(ring-refusal-irritants e)` read them; `guard` with `error-object?` catches
them like any other error.

```scheme
(require core.ring)
(write (guard (e ((ring-refusal? e) (list (ring-refusal-kind e) (ring-refusal-irritants e))))
  (r-inv (make-zmod 6) 4))) ;; => (ring-not-a-unit (r-inv "Z/6" 4))
(newline)
(write (guard (e ((ring-refusal? e) (list (ring-refusal-kind e) (ring-refusal-irritants e))))
  (r< (make-zmod 7) 1 2))) ;; => (ring-missing-capability (r< "Z/7" order))
(newline)
(display (r/ integers 1 3)) ;; => 1/3
```

## Jets, polynomials and group rings

| Procedure | Meaning |
|---|---|
| `(jet-seed J a)` | the generic point `a + e` of K[e]/e^n |
| `(make-jet J coefficients)` `(jet-constant J c)` | a jet from its coefficients (padded with zeros) or a constant |
| `(jet-coefficients x)` `(jet-coefficient x i)` | the coefficients, lowest order first |
| `(ring-poly P coefficients)` `(ring-poly-variable P)` | a polynomial from coefficients; the variable x |
| `(ring-poly-coefficients p)` `(ring-poly-degree p)` `(ring-poly-leading p)` | coefficients (lowest first, no trailing zeros), degree (-1 for zero), leading coefficient |
| `(ring-poly-eval p c)` | Horner evaluation at c in the coefficient carrier |

The unit group of Z[G] is decided exactly. Every `+-g` is a unit, and an
element whose augmentation (the sum of its weights, the ring map Z[G] -> Z)
is not +-1 is not. Beyond that the answer depends on G, which the caller
declares after the optional name:

- `'elements (g0 g1 ...)` declares G finite with these elements (closure
  under the operation and the inverse is checked). Then u is a unit exactly
  when the integer matrix of multiplication by u on the basis G has
  determinant +-1, and its inverse is the exact solution of u x = 1. In
  Z[C5], u = -1 + t + t^4 is a unit with inverse -1 + t^2 + t^3.
- `'torsion-free #t` declares G torsion-free; then Z[G] is a domain whose
  units are exactly the `+-g` (Z[t,1/t] is the case G = Z).

With neither declaration any other unit question is refused with
`ring-missing-capability` (capability `unit-group`), never guessed.

```scheme
(require core.ring)
(define Z4x (make-polys (make-zmod 4)))
;; 1 + 2x is a unit of Z/4[x]: 2 is nilpotent mod 4
(display (r-unit? Z4x (ring-poly Z4x '(1 2)))) ;; => #t
(newline)
(display (ring-poly-coefficients (r-inv Z4x (ring-poly Z4x '(1 2))))) ;; => (1 2)
(newline)
(define Qx (make-polys rationals))
(display (ring-poly-coefficients (r-gcd Qx (ring-poly Qx '(-1 0 1)) (ring-poly Qx '(1 2 1))))) ;; => (1 1)
(newline)
(define L (zset-group-ring (lambda (g h) (+ g h)) 0 (lambda (g) (- g)) "Z[t,1/t]" 'torsion-free #t))
(define t (zset-singleton 1 1))
;; (t + 1)(t - 1) = t^2 - 1, as (weight row) entries
(display (zset-entries (r* L (r+ L t 1) (r- L t 1)))) ;; => ((-1 0) (1 2))
(newline)
(display (zset-entries (r-inv L t))) ;; => ((1 -1))
(newline)
;; Z[C5]: a unit that is not +-g
(define C5 (zset-group-ring (lambda (g h) (modulo (+ g h) 5)) 0 (lambda (g) (modulo (- g) 5))
                            "Z[C5]" 'elements '(0 1 2 3 4)))
(define u (zset-from-weighted '((-1 0) (1 1) (1 4))))
(display (r-unit? C5 u)) ;; => #t
(newline)
(display (zset-entries (r-inv C5 u))) ;; => ((-1 0) (1 2) (1 3))
```

## Ring maps

A ring map is first-class: `(make-ring-map source target proc [name])`,
`(ring-map? x)`,
`(ring-map-apply phi x)`, `(ring-map-source phi)`, `(ring-map-target phi)` and
`(ring-map-compose g f)` (g after f).

| Constructor | Map |
|---|---|
| `(ring-base-change A B)` | the implicit lattice map A -> B (Z -> F_p, Q -> Q[e]/e^n, Q -> binary64, K -> K[x], ...) |
| `(zmod-reduction A B)` | Z/m -> Z/d for d dividing m, or Z -> Z/d; explicit because it forgets information |
| `(jet-primal J)` | K[e]/e^n -> K, e -> 0 |
| `(ring-poly-evaluation P S a)` | K[x] -> S, x -> a; with `a = (jet-seed J c)` it is K[x] -> K[e]/e^n, the Taylor expansion at c |
| `(group-ring-character Zg S chi)` | Z[G] -> S induced by a character chi from G to the units of S |

```scheme
(require core.ring)
(define Qx (make-polys rationals))
(define J (make-jets rationals 4))
;; x -> 1/2 + e: x^3 expanded at 1/2
(define phi (ring-poly-evaluation Qx J (jet-seed J 1/2)))
(display (jet-coefficients (ring-map-apply phi (ring-poly Qx '(0 0 0 1))))) ;; => (1/8 3/4 3/2 1)
(newline)
(display (ring-map-apply (zmod-reduction (make-zmod 49) (make-zmod 7)) 45)) ;; => 3
(newline)
(display (ring-map-apply (ring-base-change integers (make-zmod 7)) (expt 10 30))) ;; => 1
```

## Linear algebra over a carrier

`(ring-det K matrix)` is one determinant whose elimination follows the
carrier's declared structure:

- over a field, Gaussian elimination with unit pivots (every nonzero pivot);
- over an integral domain with exact quotients (Z, K[x] over a domain, any
  carrier declaring `integral-domain` and `exact-quotient`), fraction-free
  Bareiss elimination: any nonzero pivot, each step divided exactly by the
  previous pivot, so no unit pivot is ever needed (a unimodular integer
  matrix such as `((2 3) (3 5))` has determinant 1 though its first column
  holds no unit);
- over any other carrier (Z/m for composite m, jets) pivots must be units: a
  column whose remaining entries are all zero gives determinant zero, and a
  column with nonzero entries but no unit is refused with `ring-not-a-unit`.

`(ring-solve K A b)` needs inverses and keeps unit pivots: a system without
a unit pivot, or a singular one, is refused with `ring-not-a-unit`. Matrices
are vectors or lists of rows. Over the fields Z/p these are the eliminations
[`core.multimodular`](multimodular.md) runs in each fibre.

```scheme
(require core.ring)
(display (ring-det rationals '((2 1) (1 3)))) ;; => 5
(newline)
(display (ring-det integers '((2 3) (3 5)))) ;; => 1
(newline)
(define Zx (make-polys integers))
(define x (ring-poly-variable Zx))
(display (ring-poly-coefficients (ring-det Zx (list (list x 2) (list 3 x))))) ;; => (-6 0 1)
(newline)
(display (ring-det (make-zmod 7) '((2 1) (1 3)))) ;; => 5
(newline)
(display (ring-solve (make-zmod 7) '((2 1) (1 3)) '(1 0))) ;; => #(2 4)
```

## The law check and user carriers

`(ring-law-violations R samples [integer-samples])` checks, with the carrier's
own `equal?`, commutativity and associativity of `add` and `mul`,
distributivity, the identities and additive inverses on every triple of
samples; `a * a^-1 = 1` for every sampled unit and a `ring-not-a-unit` refusal
for every sampled non-unit; that `from-integer` is a ring map on the integer
samples (by default they cross the int64 boundary and include bignums); and
the declared characteristic. It returns the violations as lists
`(law element ...)`, the empty list when every law held, and refuses the IEEE
tier and the native tower, which are not rings.

`(make-ring-carrier name key value ...)` builds a user carrier from the
required fields `zero one add neg mul from-integer equal? unit? inverse` and
any optional capability of [the descriptor](#the-descriptor). Its `inverse`
is guarded by its `unit?`, so a non-unit is refused by name.

```scheme
(require core.ring)
(define Z5 (make-ring-carrier "Z/5 by hand" 'characteristic 5 'field? #t
  'zero 0 'one 1
  'add (lambda (a b) (modulo (+ a b) 5)) 'neg (lambda (a) (modulo (- a) 5))
  'mul (lambda (a b) (modulo (* a b) 5)) 'from-integer (lambda (n) (modulo n 5))
  'equal? = 'unit? (lambda (a) (not (= a 0))) 'inverse (lambda (a) (modulo (expt a 3) 5))))
(display (ring-law-violations Z5 '(0 1 2 3 4))) ;; => ()
(newline)
;; subtraction in place of addition is caught at once
(define bad (make-ring-carrier "bad" 'characteristic 0
  'zero 0 'one 1 'add (lambda (a b) (- a b)) 'neg (lambda (a) (- a)) 'mul *
  'from-integer (lambda (n) n) 'equal? = 'unit? (lambda (a) (= (abs a) 1)) 'inverse (lambda (a) a)))
(display (car (ring-law-violations bad '(1 2 3)))) ;; => (additive-inverse 1)
```

The ring laws checked here are `ring-law-equations`, a list of
`(name lhs rhs)` laws over the operation symbols `zero one add neg mul`, and
the check is `(law-violations op-of arity-of same laws samples)`: every law
under every assignment of the samples, extended one variable at a time, with
the laws of each number of variables checked at that prefix; `op-of` gives an
operation's procedure (its value at arity 0) and `arity-of` its arity.
`(law-term->procedure op-of arity-of term vars)` compiles one term to a
procedure of a vector of values. [`core.theory`](theory.md) checks the laws of
every algebraic theory with them.

```scheme
(require core.ring)
(define Z4 (make-zmod 4))
(define (op-of op) (ring-ref Z4 op))
(define (arity-of op) (cdr (assq op '((zero . 0) (one . 0) (add . 2) (neg . 1) (mul . 2)))))
(display (law-violations op-of arity-of (lambda (a b) (= a b)) ring-law-equations '(0 1 2 3))) ;; => ()
(newline)
;; x * x = x fails at 2 and 3 in Z/4
(display (law-violations op-of arity-of (lambda (a b) (= a b)) (list (list 'idempotent '(mul x x) 'x)) '(0 1 2 3))) ;; => ((idempotent 2) (idempotent 3))
```

`make-family-carrier` builds a carrier of a registered family (core.algebraic_domain
uses it), and `ring-register-recognizer!` lets a family name the carrier of its
own element representation.

## The IEEE tier

`flonums`, and jets or polynomials over it, are specified per operation and
never called a ring:

- elements are binary64 values including -0.0, the infinities and NaN;
  `equal?` is numeric `=`, so NaN is not equal to itself;
- `add`, `sub`, `mul` and division are the IEEE operations, rounded to
  nearest, ties to even, applied once: for finite operands the result is the
  exact rational result rounded once (an infinity when that rounding
  overflows); division by zero gives an infinity or NaN and does not raise;
  negation is the sign flip;
- an exact operand is rounded once, in exact integer arithmetic, whatever the
  size of its numerator and denominator;
- there is no implicit reassociation and no contraction into fused
  multiply-add;
- jet coefficients over binary64 are computed by the lawful recurrences with
  every coefficient operation rounded.

```scheme
(require core.ring)
(display (r+ flonums 0.1 0.2)) ;; => 0.30000000000000004
(newline)
(display (r/ flonums 1.0 0.0)) ;; => +inf.0
(newline)
(display (r-neg flonums 0.0)) ;; => -0.0
(newline)
(display (r= flonums (r/ flonums 0.0 0.0) (r/ flonums 0.0 0.0))) ;; => #f
(newline)
(display (r+ flonums 1/3 0.0)) ;; => 0.3333333333333333
```

`tests/stdlib/ring_ieee_test.esk` checks this specification on 20,592
operations against an exact reference rounding and pins a digest of the
binary64 bit patterns, so native JIT and AOT agree bit for bit.

## Interoperation

- **`core.algebraic_domain`.** `(algebraic-domain owner)` returns a carrier of
  the `algebraic` family for a number field Q(alpha) or one of its exact Taylor
  or hyper-dual lifts, and `algebraic-domain-ref` keeps its coefficient
  vocabulary (`copy add sub mul inverse equal? zero? unit? coefficients element
  zero one`, and `owner kind dimension parent depth`). The same domain answers
  the generic operations, and algebraic elements name their carrier.
- **`core.symbolic` coefficient fields.** `(coefficient-field->ring field)` maps
  a coefficient-field descriptor `#(eshkol-coefficient-field p)` to
  `(make-zmod p)` (or `rationals` for p = 0), and `(ring->coefficient-field R)`
  goes back. The elements are the same values (canonical representatives in
  `[0, p)`), so they pass between the two without conversion.
- **`core.dbsp`.** `zset-group-ring` reuses the Z-set sum, negation and join;
  any carrier's additive group can be handed to `make-dbsp-group` of
  `core.dbsp.circuit` as `(lambda () (r0 R))`, `(lambda (a b) (r+ R a b))`,
  `(lambda (a) (r-neg R a))` and `(lambda (a) (r-zero? R a))`.

`exact-prime?` decides primality exactly below 3317044064679887385961981 (and
for larger n proven composite by a witness).

## Tests

```text
ctest --test-dir build --output-on-failure -R '^(ring|ring_ieee|multimodular)_(jit|aot)$'
```

`ring_test` runs the law check exhaustively on every element of Z/n for
n = 1..13, three small jet carriers, a nested jet carrier and Z/4[x] in degree
at most 1 (21,810 triples), and with a fixed seed on twelve infinite carriers
(Z across the int64 boundary, Q, Q[e]/e^4, Q[x], F_101[x], Z/10^30, Z/2^64,
Z/7^16, Z[t,1/t], Q(sqrt 2) and its Taylor lift, F_7[e]/e^6), plus
characteristic, nilpotency, the unit groups (including Z[C5] against the
circulant determinant over all 243 elements with coefficients in {-1, 0, 1}),
every refusal, the coercion lattice and the ring maps. `examples/mathematics_functor_of_points.esk` runs
one program over eight carriers and the fibres of Spec Z on this library.

The native runtime registry (`ring_carriers.def`, `eshkol_ring_carrier_t`) is
Planned for v1.4.2, code generation through it for v1.4.3, the VM and browser
VM on the same registry for v1.4.4, and carriers in the type system for v1.5.0,
as the ADR's staged plan lists.
