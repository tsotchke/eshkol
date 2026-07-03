# `math` — linear algebra, numerical methods, and statistics

**Source**: [`lib/math.esk`](../../../lib/math.esk)
**Require**: `(require math)` — **must be required individually**. Despite operating on the same tensor/vector data as the auto-loaded signal/ml modules, `math` is **NOT** pulled in by `(require stdlib)` (there is no `(require math)` line in `lib/stdlib.esk`). Calling `dot` etc. after only `(require stdlib)` fails with *"called undefined function 'dot'"*.

Numerical algorithms implemented in Eshkol itself: matrix determinant/inverse/solve (Gaussian / Gauss-Jordan elimination with partial pivoting), 3-D cross product, dot/normalize, power-iteration eigenvalue estimate, Simpson integration, Newton root-finding, and variance/std/covariance statistics.

## Data model — everything is a tensor/vector written `#(...)`

Matrices are **flat** tensors in row-major order: an n×n matrix is a length-n² tensor, and `n` is passed explicitly. Vectors are `#(...)` literals (or the result of `make-vector`/`vector`). Elements are read with `vref` and are treated as `double`.

> **Do not pass lists.** These functions index with `vref` and length with `vector-length`; a Scheme `(list …)` is a different heap layout and is misread → **SIGSEGV**. See [Known issues](#known-issues).

Constants `pi`, `e`, `epsilon` (`1e-15`) are defined internally but are **not** in the `provide` list, so user code cannot reference them after `(require math)`.

## Functions

### `(mat-ref M cols row col)`
Element at `(row, col)` of a matrix `M` stored flat with `cols` columns. Computes `(vref M (+ (* row cols) col))`.

```scheme
(require math)
(display (mat-ref #(1.0 2.0 3.0 4.0) 2 1 0)) (newline)
```
```
3
```

### `(tensor-copy T)`
Returns a fresh mutable `make-vector` copy of tensor/vector `T` (used internally so the elimination routines don't clobber the caller's data).

```scheme
(display (tensor-copy #(1.0 2.0 3.0))) (newline)
```
```
#(1 2 3)
```

### `(det M n)`
Determinant of the `n`×`n` matrix `M` (flat, row-major) via Gaussian elimination with partial pivoting, O(n³). Returns a `double`. A singular matrix (pivot below `epsilon`) yields `0.0`.

```scheme
(display (det #(4.0 3.0 6.0 3.0) 2)) (newline)
```
```
-6
```

### `(inv M n)`
Inverse of the `n`×`n` matrix `M` via Gauss-Jordan elimination. Returns the inverse as a flat vector, or `#f` if singular.

```scheme
(display (inv #(4.0 7.0 2.0 6.0) 2)) (newline)
(display (inv #(1.0 2.0 2.0 4.0) 2)) (newline)   ;; singular
```
```
#(0.6 -0.7 -0.2 0.4)
#f
```

### `(solve A b n)`
Solve `A x = b` for the `n`-vector `x` (LU / Gaussian elimination with partial pivoting). `A` is flat `n`×`n`, `b` is length `n`. Returns `x`, or `#f` if singular.

```scheme
(display (solve #(2.0 1.0 1.0 3.0) #(3.0 5.0) 2)) (newline)
```
```
#(0.8 1.4)
```

### `(cross u v)`
3-D cross product `u × v`. Both must be length-3 vectors; returns a length-3 vector.

```scheme
(display (cross #(1.0 0.0 0.0) #(0.0 1.0 0.0))) (newline)
```
```
#(0 0 1)
```

### `(dot u v)`
Dot product of two vectors (loops to `(vector-length u)`; if `v` is shorter it reads out of bounds).

```scheme
(display (dot #(1.0 2.0 3.0) #(4.0 5.0 6.0))) (newline)
```
```
32
```

Edge cases: passing lists SIGSEGVs — see [Known issues](#known-issues).

### `(normalize v)`
Unit vector in the direction of `v` (divides by `norm`). If `‖v‖ < epsilon` the input is returned unchanged (avoids divide-by-zero).

```scheme
(display (normalize #(3.0 4.0))) (newline)
(display (normalize #(0.0 0.0))) (newline)   ;; zero vector returned as-is
```
```
#(0.6 0.8)
#(0 0)
```

### `(mat-vec-mul A x rows cols)`
Matrix-vector product `A·x`. `A` is flat `rows`×`cols`, `x` is length `cols`; returns length `rows`.

```scheme
(display (mat-vec-mul #(1.0 2.0 3.0 4.0) #(1.0 1.0) 2 2)) (newline)
```
```
#(3 7)
```

### `(power-iteration A n max-iters tolerance)`
Estimate the dominant eigenvalue of the `n`×`n` matrix `A` by the power method, stopping after `max-iters` or when successive Rayleigh quotients differ by less than `tolerance`.

```scheme
(display (power-iteration #(2.0 0.0 0.0 3.0) 2 100 1e-9)) (newline)
```
```
3
```

### `(integrate f a b n)`
Numerical integral of `f` over `[a, b]` using Simpson's rule with `n` intervals (`n` should be even).

```scheme
(display (integrate (lambda (x) (* x x)) 0.0 1.0 100)) (newline)
```
```
0.333333
```

### `(newton f df x0 tolerance max-iters)`
Newton–Raphson root of `f` with derivative `df`, starting from `x0`. Stops when `|x_{k+1} − x_k| < tolerance`, on `max-iters`, or if `|df| < epsilon` (returns current `x`).

```scheme
(display (newton (lambda (x) (- (* x x) 2.0))
                 (lambda (x) (* 2.0 x))
                 1.0 1e-10 100)) (newline)
```
```
1.41421
```

### `(variance v)`
Population variance of vector `v` (divides by `len`, not `len−1`). Uses `tensor-mean` for the mean.

```scheme
(display (variance #(1.0 2.0 3.0 4.0 5.0))) (newline)
```
```
2
```

### `(std v)`
Standard deviation, `(sqrt (variance v))`.

```scheme
(display (std #(1.0 2.0 3.0 4.0 5.0))) (newline)
```
```
1.41421
```

### `(covariance u v)`
Population covariance of `u` and `v` (divides by `len` of `u`).

```scheme
(display (covariance #(1.0 2.0 3.0) #(4.0 5.0 6.0))) (newline)
```
```
0.666667
```

## Known issues

### Passing a list where a tensor/vector is expected → SIGSEGV

All `math` functions index with `vref`/`vector-length`, which assume the tensor/vector heap layout. A Scheme list is a cons chain with a different layout; reading it as a tensor dereferences garbage and crashes. This is the tensor-op-on-non-tensor class tracked as **ESH-0069** ("Tensor ops SIGSEGV on non-tensor (vector) input instead of raising a type error").

```scheme
;; repro.esk
(require math)
(display (dot (list 1.0 2.0 3.0) (list 4.0 5.0 6.0))) (newline)
```
```
[Eshkol] fatal signal: SIGSEGV (segmentation fault) — terminating
```
Workaround: always pass `#(...)` literals or the result of `(vector …)` / `(make-vector …)`, never `(list …)`.
