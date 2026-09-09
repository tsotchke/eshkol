# `core.exact_linalg` — exact rational linear algebra and torus averaging

**Source**: [`lib/core/exact_linalg.esk`](../../../lib/core/exact_linalg.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.exact_linalg)`

Linear algebra over the *scalar* exact tower — int64, bignum, exact
rational — on plain Scheme vectors-of-vectors, never tensors. An n x m
matrix is `(vector row0 row1 ... row(n-1))` where each `rowI` is a
length-m vector; a vector `(vector x0 x1 ... x(k-1))` is a plain
column/point vector. Compare `lib/math.esk` (`det`/`inv`/`solve`), which
works on `f64`-backed tensors and seeds `pi`/`e`/`epsilon` as inexact
constants — that module is untouched by this one; the two are
deliberately parallel, not merged.

**Build matrices and vectors with the `vector` procedure, never with a
`#(...)` literal.** The reader auto-promotes a numeric `#(...)` vector
literal — flat or nested — into an f64-backed shaped-tensor value: its
entries silently become inexact, and for a nested literal the outer
`vector-length` reports the *flattened* element count rather than the row
count. `#(1 2)` and `#(#(1 2) #(3 4))` are therefore both the wrong way to
build input for this module; `(vector 1 2)` and
`(vector (vector 1 2) (vector 3 4))` are the right way, and every example
below uses that form.

## Exactness contract

Every function in this module is built only from the generic arithmetic
operators (`+ - * / zero? < =`) and never writes a floating-point literal.
R7RS numeric contagion does the rest: feed every function here an
all-exact matrix (integers, bignums, and/or exact rationals) and every
result is exact (`(exact? result)` is `#t`); let one entry be inexact
(`1.0` instead of `1`, or a value that came from `sqrt`/`sin`/division by
an inexact quantity elsewhere) and the affected results demote to inexact
automatically, the same way `(+ 1 2.0)` demotes to `3.0`. `exact-matrix?`
is the one function that *inspects* exactness, as a query predicate; every
other function accepts a mixed exact/inexact matrix and lets contagion
decide the result — it does not require `exact-matrix?` to hold on its
input.

**Known limitation.** `eshkol_rational_t` stores its numerator and
denominator as plain `int64_t`, not bignums (see
[EXACT_ARITHMETIC.md §5](../../breakdown/EXACT_ARITHMETIC.md)). A rational
arithmetic step whose numerator or denominator would overflow `int64`
silently falls back to double arithmetic at the runtime level — the
R7RS-mandated "never fail, degrade" rule for exact rationals, not a defect
in this module. `exact-solve`/`exact-inverse`/`exact-nullspace` on
rational inputs are exact exactly as long as every intermediate pivot
stays within `int64` range; there is no arbitrary-precision rational type
underneath to fall back on for larger systems. `exact-det` sidesteps this
for **integer** inputs specifically by using Bareiss fraction-free
elimination, whose every division is guaranteed (Sylvester's identity) to
divide evenly — the running matrix never holds a non-integral
intermediate value when the input is all-integer, which is the strongest
exactness guarantee this module offers.

## Functions

### `(exact-matrix? m)`
`#t` iff `m` is a non-empty vector of equal-length vectors whose every
entry is an exact number (`(and (number? x) (exact? x))`).

### `(exact-matrix-ref m i j)`
The entry at row `i`, column `j` — `(vector-ref (vector-ref m i) j)`.

### `(exact-matrix-mul a b)`
The matrix product of an `ra x ca` matrix `a` and a `ca x cb` matrix `b`
(errors if the inner dimensions disagree).

### `(exact-matrix-transpose m)`
The transpose of an `r x c` matrix (a fresh `c x r` matrix; `m` is not
mutated).

### `(exact-det m)`
The determinant of a square matrix via **Bareiss fraction-free Gaussian
elimination**. See the exactness note above for why integer inputs never
generate an intermediate fraction.

```scheme
(require stdlib)
(define m3 (vector (vector 1 2 3) (vector 2 5 4) (vector 1 0 5)))
(display (exact-det m3)) (newline)          ; -2
(display (exact? (exact-det m3))) (newline) ; #t
(define m2-cov (vector (vector 1 1/4) (vector 1/4 3/10)))
(display (exact-det m2-cov)) (newline)      ; 19/80
```

### `(exact-solve a b)`
The exact solution vector `x` of `a x = b`, for a square, **full-rank**
`a`, via Gauss-Jordan elimination with exact pivoting (any nonzero pivot
works — exact arithmetic has no numerical-stability reason to prefer the
largest-magnitude one). **Raises** `(error "exact-solve: singular matrix"
a)` on a rank-deficient `a` — via `(error ...)`, catchable with `guard`
like the rest of the stdlib — rather than returning a fabricated value.

### `(exact-inverse a)`
The exact inverse of a square, full-rank matrix, via Gauss-Jordan
elimination on `[a | I]`. **Raises** `(error "exact-inverse: singular
matrix" a)` on a singular `a`.

### `(exact-rank m)`
The rank of `m` (square or rectangular) via row reduction.

### `(exact-nullspace m)`
A list of exact basis vectors spanning `ker(m)` — one per free (non-pivot)
column of the reduced row echelon form of `m`. `'()` when `m` has full
column rank (trivial nullspace).

```scheme
(require stdlib)
(define m-singular (vector (vector 1 2) (vector 2 4)))
(display (exact-rank m-singular)) (newline)      ; 1 (rows proportional)
(display (exact-nullspace m-singular)) (newline) ; (#(-2 1))
```

### `(torus-average f n)`
The discrete angular mean of `f` over `n` equally spaced points on a
circle: `(1/n) * sum_{k=0}^{n-1} (f k)`. `f` receives the sample index `k`
(`0 <= k < n`), **not** an angle — placing that index on the circle is the
caller's choice. The weight `1/n` is always an exact rational, so if `f`
returns an exact value at every sample (e.g. an exact rational function of
`k/n`) the average is exact. If `f` samples a transcendental quantity
(`sin`/`cos` of an angle built from `pi`, etc.) its return values are
inexact, and ordinary R7RS contagion demotes the sum — and hence the
average — to inexact; **that demotion is expected**, not a bug: this
module never introduces inexactness itself, and there is no way to make a
transcendental sample exact. `n` must be a positive exact integer.

### `(torus-average-2d f n m)`
The discrete mean of `f` over an `n x m` grid of equally spaced points on
the torus T²: `(1/(n*m)) * sum_{i,j} (f i j)`. Same exactness contract as
`torus-average`: `f` receives the two grid indices (`0 <= i < n`,
`0 <= j < m`), the weight `1/(n*m)` is always an exact rational, and
inexactness enters only through what `f` itself returns. `n` and `m` must
both be positive exact integers.

```scheme
(require stdlib)
;; f(k) = k over 4 points: (0+1+2+3)/4 = 3/2, exact.
(display (torus-average (lambda (k) k) 4)) (newline)             ; 3/2
;; f(i,j) = i+j over a 2x3 grid: 9/6 = 3/2, exact.
(display (torus-average-2d (lambda (i j) (+ i j)) 2 3)) (newline) ; 3/2
```
