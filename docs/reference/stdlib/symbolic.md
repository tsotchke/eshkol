# `core.symbolic` — symbolic polynomials and truncated power series over Q and GF(p)

**Source**: [`lib/core/symbolic.esk`](../../../lib/core/symbolic.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.symbolic)`

Symbolic *values* over the exact tower — int64, bignum, exact rational —
complementing `core.exact_linalg`'s exact linear algebra at points. Where
that module evaluates numerically, this one represents a residual (or
any algebraic expression) AS a value: a sparse multivariate polynomial
or a truncated multivariate power series that can be added, multiplied,
differentiated, integrated, composed and queried to an arbitrary order,
instead of only sampled at a point.

Coefficients live in a field: the rationals Q (the default, with the
exact tower's contagion rules) or a prime field GF(p). Series values
carry the weighted l1 coefficient norm of the analytic space B_rho, and
`series-of` turns an ordinary procedure — one that uses the built-in AD
operators internally — into a polynomial-valued value whose coefficients
can be collected and solved.

## Representation

A **polynomial** is `(eshkol-poly vars terms)`: `vars` is the sorted
list of variable symbols the polynomial is expressed over (canonical,
regardless of the order the caller introduced them in); `terms` is a
canonical, zero-coefficient-free, duplicate-free list of
`(expvec . coeff)` pairs, `expvec` a `(vector e0 e1 ... en-1)` of
nonnegative exact integers and `coeff` a nonzero number. Terms are
ordered by ascending total degree then lexicographically on the
exponent vector — a total order, so `poly=?` is a normal-form check, not
a symbolic-equivalence search.

A polynomial over GF(p) appends the prime: `(eshkol-poly vars terms p)`,
and stores every coefficient as its canonical representative, an exact
integer in `[0, p)`. A Q polynomial keeps the 3-element shape above.

A **truncated power series** is `(eshkol-series vars terms order)`,
structurally the same except an `expvec` entry MAY be negative (a
Laurent / principal-part term — see below), and `order` records a
truncation bound: terms are dropped once the sum of their NONNEGATIVE
exponent parts exceeds the bound (so a Laurent term is never truncated
away by an upper bound on the *regular* part). A series over GF(p) is
`(eshkol-series vars terms order p)`.

**Build exponent data with plain lists** — every public constructor
takes exponents as a Scheme list (`(list 1 0 2)`), never a `#(...)`
vector literal; a numeric `#(...)` literal auto-promotes to an
f64-backed tensor (the same `core.exact_linalg` gotcha), and this module
never asks the caller to build a raw exponent vector directly.

## Exactness contract

Every function here is built from the generic arithmetic operators
(`+ - * / zero? = < >`) plus repeated squaring for exponentiation, and
never writes a floating-point literal. R7RS contagion does the rest: an
all-exact input stays exact; one inexact coefficient demotes only the
terms it touches. `series-exp` / `series-log` / `series-sin` /
`series-cos` / `series-sqrt` derive their Taylor coefficients from exact
rationals (`1/n!`, generalized-binomial coefficients via `/`), so an
exact-coefficient input series produces an exact-coefficient result — no
`sin`/`cos`/`exp`/`log`/`sqrt` call on a floating-point value ever
happens inside this module.

**Known limitation (inherited).** `eshkol_rational_t`'s
numerator/denominator are `int64_t`, not bignums (see
[EXACT_ARITHMETIC.md §5](../../breakdown/EXACT_ARITHMETIC.md)) — a
rational step that would overflow int64 silently degrades to double.
`poly-expt` / `series-inverse` / the Taylor transcendentals are exact
exactly as long as intermediate numerators/denominators stay in int64
range.

## The "1+x" convention for `series-log` / `series-sqrt` / `series-inverse`

`series-exp`, `series-sin`, `series-cos` take an argument series `f`
with `f(0) = 0` and return `exp(f)`, `sin(f)`, `cos(f)` directly — those
three are entire, well-defined at the origin. `series-log`,
`series-sqrt` and `series-inverse` are singular AT the origin, so they
instead take `f` with `f(0) = 0` and return `log(1+f)`, `sqrt(1+f)`,
`1/(1+f)` — the standard "log1p"/"sqrt1p" convention.
**`(series-log x-series)` means `log(1+x)`, not `log(x)`.** Feeding
`series-log` a series whose constant term is already `1` (intending
`log(2+x)`) is a mistake the function catches: it raises, because ITS
OWN argument's constant term must be zero.

```scheme
(require stdlib)
(define x (series (list 'x) (list (cons (list 1) 1)) 6)) ; x itself
(define log1px (series-log x))    ; log(1+x)
(define onepx (series-exp log1px)) ; = 1+x again, exactly
(display (series-coeff onepx (list (cons 'x 1)))) (newline) ; 1
(display (series-coeff onepx (list (cons 'x 2)))) (newline) ; 0
```

## Coefficient fields: Q and GF(p)

`(coefficient-field-gf p)` names the prime field GF(p);
`(coefficient-field-q)` names Q. `poly->field` and `series->field` move a
value into a field: Q to GF(p) reduces each exact rational `n/d` to
`n * d^-1 mod p` (raising when `p` divides `d`), GF(p) to Q lifts the
canonical representatives, and a value already in the target field is
returned unchanged. Every operation then stays in the operands' field;
mixing fields in one operation raises instead of converting implicitly.

GF(p) arithmetic is the exact integer tower followed by `modulo p`
(int64 promotes to bignum, so large primes work), and division multiplies
by the Fermat inverse `a^(p-2)`. The modulus is proved prime with the
deterministic Miller-Rabin bases 2 through 41, exact for every `p` below
`3317044064679887385961981`; a larger modulus is refused. Series
transcendentals over GF(p) reduce their rational Taylor coefficients into
the field, so they are defined up to the order where a denominator such
as `k!` becomes divisible by `p`, and raise beyond it.

```scheme
(require stdlib)
(define F7 (coefficient-field-gf 7))
(define x (poly->field (poly-var 'x) F7))
(define one (poly->field (poly-const 1) F7))
;; Frobenius: (x + 1)^7 = x^7 + 1 in characteristic 7
(display (poly=? (poly-expt (poly+ x one) 7) (poly+ (poly-expt x 7) one))) (newline) ;; => #t
;; 1/2 is 4 in GF(7)
(display (poly-coeff (poly->field (poly-const 1/2) F7) '())) (newline)          ;; => 4
(display (coefficient-field-characteristic (poly-field x))) (newline)          ;; => 7
```

## Coefficient norm on series values

`(series-norm s rho)` is the weighted l1 norm `sum |c_a| rho^a` of the
analytic space B_rho. `rho` is one positive weight applied to the total
degree, or an alist `((var . weight) ...)` with one weight per variable;
a Laurent term `tau^-k` contributes `rho^-k`. The norm is
submultiplicative, `|f g| <= |f| |g|` (truncation only drops terms),
which is the property a contraction argument on B_rho needs. Exact
coefficients and an exact weight give an exact norm. The norm is defined
over Q; a GF(p) value raises.

```scheme
(require stdlib)
;; 1 - 2x + 3/4 x y, known to total degree 6
(define f (series (list 'x 'y) (list (cons (list 0 0) 1) (cons (list 1 0) -2) (cons (list 1 1) 3/4)) 6))
(display (series-norm f 1/2)) (newline)                                  ;; => 35/16
(display (series-norm f (list (cons 'x 1/2) (cons 'y 3)))) (newline)     ;; => 25/8
(display (<= (series-norm (series* f f) 1/2) (* (series-norm f 1/2) (series-norm f 1/2)))) (newline) ;; => #t
```

## Polynomial-valued duals: `series-of`, collect, solve

`(series-of f vars point order)` applies an ordinary procedure to a
symbolic point and returns the truncated series of `f` in the
displacements of `vars` from `point`, with coefficient
`(d^a f)(point) / a!`. `f` is written with the generic arithmetic and may
use the built-in AD operators itself: each coefficient is computed by
nested passes of `taylor` and `derivative-n`, and the passes inside `f`
nest under them as levels of the recursive Taylor carrier
([ADR-0027](../../design/adr/0027-recursive-taylor-level-carrier.md)). At
an exact point the coefficients are exact; a transcendental inside `f`
demotes only the coefficients it touches. When `f` is a polynomial of
total degree at most `order` in `vars`, the result is that polynomial
exactly, and `series->poly` converts it. A vector-valued `f` gives a
vector of series, one per component.

The graded value is then **collected** and **solved**:
`(poly-collect p vars)` and `(series-collect s vars)` group terms by the
exponents of the given variables (a symbol or a list), returning an
ascending list of `(exponents . coefficient)`, the coefficient a
polynomial (or series) in the remaining variables; Laurent exponents of
a grading parameter are allowed in `series-collect`.
`(poly-linear-system equations unknowns)` reads each equation as `= 0`,
requires it to be affine in `unknowns` and free of every other variable,
and returns `(cons A b)` in `core.exact_linalg`'s row-vector format.
`(poly-solve-linear equations unknowns)` returns every solution as
`(particular . basis)`, alists from unknowns to values, raising on an
inconsistent system; over Q it is `exact-affine-solve`, over GF(p) the
same elimination in the field.

```scheme
(require stdlib)
;; ansatz u(x) = a x^2 + b x, residual u'' + u' - (2 + 2x)
(define (ansatz a b) (lambda (x) (+ (* a x x) (* b x))))
(define (residual a b x)
  (let ((u (ansatz a b)))
    (- (+ (derivative-n u x 2) (derivative-n u x 1)) (+ 2 (* 2 x)))))
(define r (series->poly (series-of residual (list 'a 'b 'x) (list 0 0 0) 3)))
(display (poly->string r)) (newline)              ;; => 2*a*x + 2*a + b - 2*x - 2
(define eqs (map cdr (poly-collect r 'x)))        ; coefficients of x^0 and x^1
(display (car (poly-solve-linear eqs (list 'a 'b)))) (newline)  ;; => ((a . 1) (b . 0))
```

## AD interoperation

Eshkol closures cannot be introspected at runtime, and a `define-syntax`
macro defined inside a stdlib module does not propagate across the
precompiled-stdlib boundary into user source (see
`lib/core/testing.esk`'s header for the identical constraint) — so
neither "pass a closure and call it with a magic symbolic value" nor
"expose this as a macro" is available the way the compiler's own
built-in `diff` special form works (a compile-time AST rewrite over the
literal expression; the parser creates the `diff` AST and the LLVM path
rewrites it in `simplifySymbolicAST` (`lib/frontend/parser.cpp`,
`lib/backend/llvm_codegen.cpp`).
`poly-derivative-of` / `series-derivative-of` instead take the
expression as a **quoted datum**:

```scheme
(poly-derivative-of   '(+ (* x x) 1) 'x)          ; the polynomial x^2+1
(series-derivative-of '(sin (* 2 x)) 'x 8)        ; sin(2x) to order 8
```

Supported heads: `+ - * /` (division needs a literal numeric divisor for
`poly-derivative-of`; `series-derivative-of` allows a general divisor
via `series-inverse`), `expt`/`pow` (a literal exact nonnegative-integer
exponent), and — `series-derivative-of` only — `sin cos exp log sqrt`
under the "1+x" convention above. The only symbol the expression may
contain is the bound variable itself; every other head or free symbol
**raises loudly** via `(error ...)` rather than silently falling back to
a numeric or wrong answer.

## Functions

### Polynomials

#### `(poly? x)`
`#t` iff `x` is a polynomial value.

#### `(poly-const c)` / `(poly-var sym)`
The constant polynomial `c`, or the polynomial consisting of the single
variable `sym` to the first power.

#### `(poly vars term-specs)`
General constructor. `term-specs` is a list of `(exponents . coeff)`
pairs, `exponents` a list of nonnegative exact integers the same length
as `vars`, in `vars`' order (not necessarily canonical — the constructor
canonicalizes). A negative exponent raises: use `series` for Laurent
data.

```scheme
(require stdlib)
(define p (poly (list 'x 'y) (list (cons (list 2 0) 1) (cons (list 0 0) -1)))) ; x^2 - 1
(display (poly->string p)) (newline) ; x^2 - 1
```

#### `(poly+ a b)` / `(poly- a b)` / `(poly* a b)`
Ring addition, subtraction and multiplication. `a` and `b` may be
expressed over different (or overlapping) variable sets; the result is
expressed over their union.

#### `(poly-expt p n)`
`p^n` for a nonnegative exact integer `n`, via repeated squaring
(`O(log n)` polynomial multiplications).

#### `(poly-scale p c)`
`p` with every coefficient multiplied by the scalar `c`.

#### `(poly-eval p bindings)`
Evaluates `p` exactly at a point. `bindings` is a list of `(var . value)`
pairs covering every variable of `p`; a missing binding raises. Exact in,
exact out.

#### `(poly-deriv p var)`
The partial derivative of `p` with respect to `var` (`0` if `var` doesn't
appear in `p`).

#### `(poly-degree p)`
The total degree of `p` (`0` for the zero polynomial).

#### `(poly-coeff p var-exp-alist)`
The coefficient of the monomial named by `var-exp-alist` (a list of
`(var . exp)` pairs; a variable of `p` omitted from the alist is taken
to have exponent `0`). `0` if that monomial doesn't appear.

#### `(poly=? a b)`
Normal-form equality: both sides are canonicalized over the UNION of
their variables before comparing term-for-term, so a spurious extra
variable (e.g. the zero polynomial that results from cancelling one out)
never causes a false negative. Coefficients compare with `=`.

#### `(poly->string p)`
A human-readable rendering, highest degree first, e.g. `"x^2 + 2*x*y - 3"`.

### Truncated multivariate power series

#### `(series? x)`
`#t` iff `x` is a series value.

#### `(series vars term-specs order)`
General constructor, structurally like `poly` except exponents may be
negative (Laurent terms) and `order` is a truncation bound: a bare
integer (total-degree bound) or a bare list of integers/`#f` parallel to
`vars` (per-variable bound; `#f` = unbounded in that variable).

```scheme
(require stdlib)
;; 1 + x + x^2, known to total degree 6
(define s (series (list 'x) (list (cons (list 0) 1) (cons (list 1) 1) (cons (list 2) 1)) 6))
;; x-bound 2, y-bound 5 (per-variable order)
(define t (series (list 'x 'y) (list (cons (list 3 0) 1)) (list 2 5)))
(display (series-coeff t (list (cons 'x 3) (cons 'y 0)))) (newline) ; 0 -- truncated
```

#### `(series+ a b)` / `(series- a b)` / `(series* a b)`
Ring operations, truncated to the tighter of `a` and `b`'s order bounds.
`series*` is correct up to that combined order PROVIDED both operands
are themselves accurate to at least that order — the standard
truncated-series-arithmetic caveat.

#### `(series-compose g f)`
`g(f(x))`, truncated to `f`'s order. `g` must be single-variable; `f`
must have zero constant term (composition around a nonzero point isn't
supported).

#### `(series-deriv s var)` / `(series-integrate s var)`
Partial derivative / antiderivative (zero integration constant) with
respect to `var`. `series-integrate` raises on a `var^-1` term (its
antiderivative is a logarithm, not a Laurent series term), and over
GF(p) on a term whose new exponent is divisible by `p`.

#### `(series-inverse s)`
`1/s` for a single-variable series `s` with nonzero constant term `c0`,
via the standard recurrence `r0 = 1/c0`,
`r_n = -(1/c0) * sum_{k=1}^n c_k * r_(n-k)`. Exact throughout when every
`c_k` is exact.

#### `(series-coeff s var-exp-alist)`
The coefficient at the exponents named by `var-exp-alist` (same
convention as `poly-coeff`) — this is the first-class way to ask "what
is the coefficient of `tau^-1`": `(series-coeff s (list (cons 'tau -1)))`.

#### `(series-truncate s new-order)`
Re-truncates to the tighter of `s`'s existing bound and `new-order`;
never fabricates accuracy `s` didn't already have.

#### `(series->poly s)`
`s` as a polynomial; raises if `s` has any negative exponent.

#### `(series-exp f)` / `(series-sin f)` / `(series-cos f)`
`exp(f)` / `sin(f)` / `cos(f)` for a series `f` with zero constant term.

#### `(series-log f)` / `(series-sqrt f)`
`log(1+f)` / `sqrt(1+f)` for a series `f` with zero constant term — see
the "1+x convention" section above.

#### `(series-lowest-order s var)`
The minimum exponent of `var` among `s`'s nonzero terms (may be negative
— the Laurent leading order). `#f` if `s` is the zero series or `var`
doesn't occur in `s`.

#### `(series-singular-part s var)`
The sub-series of terms with negative `var` exponent — the Laurent
principal part with respect to `var`.

```scheme
(require stdlib)
(define lau (series (list 'tau) (list (cons (list -1) 1) (cons (list 0) 2) (cons (list 1) 3)) 5))
(display (series-lowest-order lau 'tau)) (newline)                            ; -1
(display (series-coeff (series-singular-part lau 'tau) (list (cons 'tau -1)))) (newline) ; 1
```

### Coefficient fields

#### `(coefficient-field-q)` / `(coefficient-field-gf p)`
The field descriptors for Q and GF(p). `coefficient-field-gf` raises
unless `p` is an exact prime below `3317044064679887385961981`.

#### `(coefficient-field? x)` / `(coefficient-field-characteristic field)`
Recognizes a field descriptor; the characteristic is `0` for Q and `p`
for GF(p).

#### `(poly-field p)` / `(series-field s)`
The coefficient field of a polynomial or series.

#### `(poly->field p field)` / `(series->field s field)`
The same value over `field`: Q to GF(p) reduces, GF(p) to Q lifts the
canonical representatives, the same field is the identity, and two
different prime fields raise.

### Coefficient norm

#### `(series-norm s rho)`
The weighted l1 norm of a series or polynomial over Q, with one weight
`rho` or an alist of per-variable weights; see "Coefficient norm on
series values" above.

### Polynomial-valued duals

#### `(series-of f vars point order)`
The truncated series of `f` about `point` in the variables `vars` (a list
of distinct symbols, one per argument of `f`), to total order `order`,
computed with the built-in AD operators; a vector of series for a
vector-valued `f`.

#### `(poly-collect p vars)` / `(series-collect s vars)`
Terms grouped by the exponents of `vars`, as an ascending list of
`(exponents . coefficient)`.

#### `(poly-linear-system equations unknowns)`
`(cons A b)` for equations (polynomials, or series without Laurent
terms) affine in `unknowns`.

#### `(poly-solve-linear equations unknowns)`
`(particular . basis)`: a particular solution and a basis of the
homogeneous solutions, each an alist over `unknowns`.

### AD interoperation

#### `(poly-derivative-of expr var)`
Evaluates the quoted expression `expr` (built from `+ - * /` and
literal-integer `expt`/`pow`, with `var` as its only free symbol) into
the polynomial it denotes.

#### `(series-derivative-of expr var order)`
Evaluates the quoted expression `expr` (built from `+ - * /`,
literal-integer `expt`/`pow`, and `sin cos exp log sqrt` under the "1+x"
convention, with `var` as its only free symbol) into the series it
denotes, truncated to total order `order`.

```scheme
(require stdlib)
(display (poly->string (poly-derivative-of '(+ (* x x) 1) 'x))) (newline) ; x^2 + 1
(display (series-coeff (series-derivative-of '(sin x) 'x 6) (list (cons 'x 3)))) (newline) ; -1/6
```

## Performance note

Multiplying two FULLY SATURATED order-8, 3-variable series (each
holding all `C(8+3,3) = 165` monomials of total degree <= 8 —
`series*` computes the full 165x165 = 27,225-pair cartesian product
before canonicalizing and truncating back down to 165 terms) measured
about 47ms per call on the reference build (`eshkol-run -r`, 10 calls
averaged). A partially-filled series (most realistic uses — few
variables' worth of nonzero terms) is proportionally faster: order-8
multiplications that stay in the tens-of-terms range (e.g. squaring a
4-term series twice, to 10 then 35 terms) run effectively instantly.

**A related runtime characteristic worth flagging, not a defect in this
module**: chaining ~15-30+ allocating `series*`/`poly*` calls through a
growing accumulator in a single top-level loop (each iteration's result
depending on the previous, so nothing is ever reclaimed) hits the OALR
runtime's `Region stack overflow (max depth: 64)` and crashes — a
handful of calls at full order-8/3-variable saturation is fine (see the
measurement above); it is specifically a long chain of *retained*
intermediate results in one unbounded owning scope that overflows,
consistent with `lib/core/list/sort.esk`'s documented "OALR regions
retain every allocation until their owner pops" constraint. This
module's own code contains no non-tail recursion and no `for-each`/`map`
(checked); a caller chaining many series operations in one top-level
loop should bound retention with `with-region` per iteration. Repro
saved at `.scratch/defects/region_stack_overflow_chained_series_mul_repro.esk`
in the development worktree.
