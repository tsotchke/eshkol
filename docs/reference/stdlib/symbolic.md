# `core.symbolic` — symbolic polynomials and truncated power series over the exact tower

**Source**: [`lib/core/symbolic.esk`](../../../lib/core/symbolic.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.symbolic)`

Symbolic *values* over the exact tower — int64, bignum, exact rational —
complementing `core.exact_linalg`'s exact linear algebra at points. Where
that module evaluates numerically, this one represents a residual (or
any algebraic expression) AS a value: a sparse multivariate polynomial
or a truncated multivariate power series that can be added, multiplied,
differentiated, integrated, composed and queried to an arbitrary order,
instead of only sampled at a point.

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

A **truncated power series** is `(eshkol-series vars terms order)`,
structurally the same except an `expvec` entry MAY be negative (a
Laurent / principal-part term — see below), and `order` records a
truncation bound: terms are dropped once the sum of their NONNEGATIVE
exponent parts exceeds the bound (so a Laurent term is never truncated
away by an upper bound on the *regular* part).

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

## AD interoperation

Eshkol closures cannot be introspected at runtime, and a `define-syntax`
macro defined inside a stdlib module does not propagate across the
precompiled-stdlib boundary into user source (see
`lib/core/testing.esk`'s header for the identical constraint) — so
neither "pass a closure and call it with a magic symbolic value" nor
"expose this as a macro" is available the way the compiler's own
built-in `diff` special form works (a compile-time AST rewrite over the
literal expression; see `lib/backend/eshkol_compiler.c`'s `diff`
handling and `simplifySymbolicAST` in `lib/backend/llvm_codegen.cpp`).
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
antiderivative is a logarithm, not a Laurent series term).

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
