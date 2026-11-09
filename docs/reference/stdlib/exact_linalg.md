# `core.exact_linalg` — exact rational linear algebra and torus averaging

**Source**: [`lib/core/exact_linalg.esk`](../../../lib/core/exact_linalg.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.exact_linalg)`

Linear algebra over the exact tower — int64, bignum, exact rational — on
Scheme vectors of row vectors **and on exact tensors**. An n x m matrix is
`(vector row0 row1 ... row(n-1))` where each `rowI` is a length-m vector, or
a rank-2 tensor; a right-hand side is a vector or a rank-1 tensor. An exact
tensor (`(tensor-cast X 'exact)`, ADR-0055) carries exact rationals in tensor
storage; the native kernels `tensor-det`, `tensor-inverse`, `tensor-solve`,
`linear-solve`, `matmul` and `transpose` compute over it exactly, and this
module accepts it too, so the same system can be solved by two independent
implementations and the answers compared. `lib/math.esk`'s `det` / `inv` /
`solve` follow the same contagion rule (exact seeds, exact pivots).

**Build row matrices with the `vector` procedure, never with a `#(...)`
literal.** The reader auto-promotes a numeric `#(...)` vector literal — flat
or nested — into an f64-backed tensor value: its entries become inexact, and
for a nested literal the outer `vector-length` reports the *flattened*
element count rather than the row count. Build exact data with `(vector ...)`
/ `(list ...)` and, for tensor storage, convert it explicitly with
`(tensor-cast X 'exact)`.

## Exactness contract

Every function in this module is built only from the generic arithmetic
operators (`+ - * / zero? < =`) and never writes a floating-point literal.
R7RS numeric contagion does the rest: feed every function here an
all-exact matrix (integers, bignums, and/or exact rationals) and every
result is exact (`(exact? result)` is `#t`); let one entry be inexact
(`1.0` instead of `1`, or a value that came from `sqrt`/`sin`/division by
an inexact quantity elsewhere) and the affected results are inexact, the
same way `(+ 1 2.0)` is `3.0`. `exact-matrix?` is the one function that
*inspects* exactness, as a query predicate; every other function accepts a
mixed exact/inexact matrix and lets contagion decide the result. A tensor
operand gets a tensor answer: an exact tensor when every entry of the answer
is exact, an f64 tensor otherwise.

Rationals are bignum-capable: a numerator or denominator past int64 range is
carried as a bignum, so `exact-solve` / `exact-inverse` / `exact-nullspace`
and the native exact kernels stay exact at any size (the 8 x 8 Hilbert
determinant, `1/365356847125734485878112256000000`, is computed exactly by
both). `exact-det` uses Bareiss fraction-free elimination, whose every
division is guaranteed (Sylvester's identity) to divide evenly, so an integer
matrix never produces an intermediate fraction.

## Exact tensors

An exact tensor stores each element as an exact rational of the scalar tower
in tensor storage (dtype `exact`, ADR-0055). It is a tensor (`tensor?` is
`#t`, `(tensor-dtype t)` is `exact`), built only by an explicit conversion:

| Construction | Result |
|---|---|
| `(tensor-cast X 'exact)` | `X` a tensor, list, vector or rectangular nest of them: exact elements kept whole, a finite double converted as `exact` converts it, an infinity or NaN refused |
| `(make-tensor shape fill :dtype 'exact)` | every slot the exact `fill`; an inexact fill is refused |
| `(tensor-cast t 'f64)` | the explicit conversion out: each element as `inexact` gives it |
| `(tensor ...)`, `#(...)` | unchanged: an f64 tensor (`(tensor 1/3)` stores the nearest double) |

Operations with an exact rule:

| Operation | On exact operands |
|---|---|
| `tensor-ref`, `vector-ref`, `vref`, `tensor-get`, `tensor-shape`, `reshape` | the exact element / an exact view |
| `tensor-set!` | stores an exact number whole; an inexact value is refused, the slot unchanged |
| `tensor-add` `tensor-sub` `tensor-mul` `tensor-div`, `+ - * /` on two tensors | exact, broadcasting; division by an exact 0 entry raises |
| `tensor-sum` | the exact sum |
| `matmul` | exact; a rank-1 operand is promoted as for f64 tensors |
| `transpose` | exact |
| `tensor-det` | exact, by Bareiss fraction-free elimination; a singular matrix gives exact `0` |
| `tensor-inverse`, `tensor-solve`, `linear-solve` | exact, by fraction-free elimination of `[A \| B]` and exact back substitution; several right-hand-side columns are allowed; a singular matrix raises, decided from an exact zero pivot, never a tolerance |
| `exact-det` `exact-solve` `exact-inverse` `exact-rank` `exact-nullspace` `exact-matrix-mul` `exact-matrix-transpose` | accept exact tensors (this module) |

Contagion is decided from the operand dtypes:

| Operands | Result |
|---|---|
| exact and exact | exact tensor (or exact rational for `tensor-det`, `tensor-sum`) |
| exact and f64 (or another precision dtype) | that inexact dtype, computed on the projections; for a solve, by the f64 LU routine |
| exact and a forward-mode jet tensor | jet tensor |
| exact tensor and a list or vector | the collection converts as `(tensor X)` does (f64), so the result is inexact; pass an exact tensor to stay exact |

Every other tensor kernel — the BLAS-backed and f64 LU / QR / Cholesky / SVD
routines, the elementwise transcendental functions, reductions along an axis,
`einsum` — refuses an exact tensor with a catchable error that names the
operation and `(tensor-cast t 'f64)`; an exact tensor is never rounded
implicitly. The bytecode VM refuses `'exact` in `tensor-cast` and
`make-tensor` explicitly; exact tensors there are planned for v1.4.1.

```scheme
(require stdlib)
(define A (tensor-cast (list (list 1 2 3) (list 2 5 4) (list 1 0 5)) 'exact))
(tensor-det A)                                   ; -2
(tensor-inverse A)                               ; exact entries, A * A^-1 = I exactly
(define x (tensor-solve A (tensor-cast (list 1 2 3) 'exact)))
(matmul A x)                                     ; #(1 2 3), exact
(exact-solve A (vector 1 2 3))                   ; the same x, by core.exact_linalg
(define N (tensor-cast (list (list 1 1) (list 1 (+ 1 (/ 1 (expt 10 40))))) 'exact))
(tensor-det N)                                   ; 1/10000000000000000000000000000000000000000
(tensor-add A (tensor-cast A 'f64))              ; an f64 tensor, by contagion
```

`ctest --test-dir build --output-on-failure -R exact_tensor` runs
`tests/stdlib/exact_tensor_test.esk` under JIT and AOT: construction,
stores, the exact kernels cross-checked against this module, bignum and
bignum-rational systems, exact singularity, contagion, refusals,
`lib/math.esk` on exact data, and the Navier-Stokes mechanization steps
7, 22, 24, 34, 59 and 68.

## Functions

### `(exact-matrix? m)`
`#t` iff `m` is a non-empty vector of equal-length vectors whose every
entry is an exact number (`(and (number? x) (exact? x))`), or a rank-2
exact tensor.

### `(exact-matrix-ref m i j)`
The entry at row `i`, column `j` — `(vector-ref (vector-ref m i) j)`, or
`(tensor-ref m i j)` for a tensor.

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

## Exact affine solution families

`(exact-affine-solve A b [columns])` returns `(particular . nullspace-basis)`.
Every solution is the particular vector plus an arbitrary rational linear
combination of the basis vectors. Rectangular and singular consistent systems
are admitted; inconsistent systems raise. `columns` fixes the ambient width
when `A` is an empty outer vector (default zero), and must agree with nonempty
rows. All inputs must be exact rational generic vectors. Inputs are preserved
and returned vectors own their storage. Arithmetic that loses exactness raises
before a result is returned. The existing `exact-solve` contract is unchanged.

```scheme
(require core.exact_linalg)
(exact-affine-solve (vector (vector 2 1)) (vector 1))
; particular #(1/2 0), nullspace basis (#(-1/2 1)); build with vector procedures.
```

Run `ctest --test-dir build --output-on-failure -R exact_affine`. The arithmetic
suite independently checks all 729 ternary two-dimensional systems against
determinants and augmented minors, plus rectangular/empty shapes and ownership.
