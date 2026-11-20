---
kind: reference
status: current
owner-area: ad
since: v1.4.0-connection
sources:
  - lib/core/ml/diffopt.esk
  - lib/core/ml/gradient_estimators.esk
  - tests/ml/diffopt_test.esk
  - docs/design/adr/0031-differentiable-optimization.md
---
# `core.ml.diffopt` — differentiating through optimization and discrete solvers

**Source**: [`lib/core/ml/diffopt.esk`](../../../lib/core/ml/diffopt.esk)
**Require**: `(require core.ml.diffopt)` — not auto-loaded by `(require stdlib)`.
The tape layers also need `(require core.ad.tape)`; `softmax`, `softmax-vjp` and
`argmax-onehot` come from [`core.ml.gradient_estimators`](ml_gradient_estimators.md).
**Design**: [ADR-0031](../../design/adr/0031-differentiable-optimization.md).
**Guide**: [Automatic Differentiation §12](../../guide/AUTOMATIC_DIFFERENTIATION.md#12-differentiating-through-optimization-and-discrete-solvers).

A layer whose output is the *solution* of an optimization problem — a ranking,
a shortest path, a top-k set, a ridge fit, a fixed point, a projection onto the
simplex — has a pullback defined by the problem rather than by the solver's
instructions. This module provides the four standard rules (Blondel and Roulet,
*The Elements of Differentiable Programming*, chapters 11, 13 and 14). Each rule
is one pure vector-Jacobian-product function plus one recorder that puts it on
the [`core.ad.tape`](ad_tape.md) with `record-op!`, so these layers compose with
every other tape op in one reverse pass.

| Rule | Pure pullback | Tape layer |
|---|---|---|
| Blackbox combinatorial solver (Vlastelica et al. 2020) | `blackbox-vjp` | `blackbox-solve` |
| Perturbed optimizer (Berthet et al. 2020) | `perturbed-estimate`, `perturbed-vjp` | `perturbed-solve`, `perturbed-argmax`, `perturbed-top-k`, `perturbed-max` |
| Fenchel-Young loss | prediction − target | `perturbed-fenchel-young`, `fenchel-young-softmax`, `fenchel-young-sparsemax` |
| Implicit function theorem | `implicit-root-vjp` | `implicit-root`, `implicit-fixed-point`, `implicit-argmin` |
| Simplex projections | `softmax-vjp`, `sparsemax-vjp` | `tape-softmax`, `tape-sparsemax` |

Vectors are Scheme vectors and matrices are vectors of row vectors (the
[`core.exact_linalg`](exact_linalg.md) convention). Arithmetic is generic, so
exact input stays exact through every pure pullback. The numeric tape seeds with
`1.0` and accumulates in f64, so a gradient read back with `tape-gradient` is
f64; read exact cotangents from the pure functions. No pullback here evaluates a
finite difference: `(ad-finite-difference-evals)` stays `0`.

Run the examples with the library path set, in JIT or AOT:

```
ESHKOL_PATH=lib ./build/eshkol-run -r example.esk -L build
ESHKOL_PATH=lib ./build/eshkol-run example.esk -o example -L build && ./example
```

Both modes print the outputs shown.

## Blackbox combinatorial solvers

A solver maps a cost vector `w` to `y(w) = argmin_{y in Y} <w, y>`. Its exact
derivative is zero almost everywhere; the blackbox rule differentiates the
piecewise-affine interpolation of the loss instead, at the cost of one extra
solver call:

    dL/dw = -(y(w) - y(w + lambda g)) / lambda,   g = dL/dy,  lambda > 0.

### `(blackbox-vjp solver w y g lambda)`
The pullback above. `y` must be `(solver w)`. Returns a vector shaped like `w`.

### `(blackbox-solve t w-node solver lambda)`
Records `y = (solver (node-value w-node))` on tape `t` with the blackbox
pullback. Returns the output node.

### `(rank-solver w)`
Normalized descending rank: the largest weight gets `1/n`, the smallest `1`,
ties break by index. By the rearrangement inequality this minimizes `<w, y>`
over the permutations of `(1/n, 2/n, ..., 1)`. Exact rationals.

### `(grid-path-solver rows cols)`
Returns a solver for vertex costs on a `rows × cols` grid (row-major vector):
`w ↦` the indicator of the minimum-cost monotone path (moves right or down)
from the top-left cell to the bottom-right cell. Dynamic programming over the
grid DAG, so negative perturbed costs are exact; ties prefer the move down.

```scheme
;; diffopt-blackbox.esk — blackbox pullbacks of a ranking and a shortest path
(require core.ad.tape)
(require core.ml.diffopt)
(define w (vector 9/10 1/10 1/2 3/10))
(define y (rank-solver w))
(display y) (newline)
;; g = dL/dy for L = |y - (3/4 1/4 1 1/2)|^2 / 2, lambda = 2
(display (blackbox-vjp rank-solver w y (vector -1/2 3/4 -1/2 1/4) 2)) (newline)
(define path (grid-path-solver 2 2))
(display (path (vector 1 1 5 1))) (newline)
(display (blackbox-vjp path (vector 1 1 5 1) (vector 1 1 0 1) (vector 0 5 0 0) 1)) (newline)
;; the same rule on the tape, inside a loss
(define target (vector 3/4 1/4 1 1/2))
(display
  (with-tape "bb"
    (lambda ()
      (let* ((t (current-tape))
             (wn (tape-input t w))
             (yn (blackbox-solve t wn rank-solver 2))
             (d (vector-map - (node-value yn) target))
             (L (record-op! t (list yn) 0
                  (lambda (g) (list (vector-map (lambda (x) (* g x)) d))))))
        (car (tape-gradient t L (list wn)))))))
(newline)
```
```
#(1/4 1 1/2 3/4)
#(1/4 -3/8 1/4 -1/8)
#(1 1 0 1)
#(0 -1 1 0)
#(0.25 -0.375 0.25 -0.125)
```

The second line is the exact pullback: `w + 2g = (-1/10 8/5 -1/2 4/5)` ranks to
`(3/4 1/4 1 1/2)`, and `((3/4 1/4 1 1/2) − y)/2 = (1/4 −3/8 1/4 −1/8)`. The last
line is the same pullback read back from the f64 tape.

## Perturbed optimizers and Fenchel-Young losses

A solver here maps `theta` to `argmax_{y in C} <y, theta>` (for example
`argmax-onehot` for the simplex, `(top-k-indicator k)` for the top-k polytope).
Adding noise `eps Z` and averaging gives a differentiable map:

    y_eps(theta) = E[argmax <y, theta + eps Z>]
    J y_eps(theta) = (1/eps) E[y*(theta + eps Z) grad-nu(Z)^T]

with `nu` the negative log density of `Z`: `grad-nu(z) = 1 − exp(−z)` for
Gumbel noise and `z` for standard Normal noise. With Gumbel noise and the
argmax solver, `y_eps(theta) = softmax(theta / eps)` exactly. The expectations
are estimated by Monte Carlo over a fixed sample, and the forward value and the
pullback use the same draws.

### `(perturbation-noise kind m n seed)`
`m` draws of length-`n` noise, `kind` `'gumbel` or `'normal`, from the isolated
generator `(make-prng seed)`. The same seed gives the same draws in JIT and AOT;
the shared `(random)` generator is untouched. Returns a perturbation value.

### `(make-perturbation kind samples)`, `(perturbation-kind p)`, `(perturbation-samples p)`
Build a perturbation from explicit noise vectors, and read it back.

### `(perturbed-estimate solver theta eps p)`
Returns `#(y-eps ys f-eps)`: the Monte Carlo `y_eps`, the vector of per-sample
solutions, and the perturbed maximum
`F_eps = mean_m <y_m, theta + eps Z_m>`.

### `(perturbed-vjp p eps ys g)`
The Monte Carlo pullback `(J y_eps)^T g = (1/(eps M)) sum_m <y_m, g> grad-nu(Z_m)`,
given the per-sample solutions `ys` from `perturbed-estimate`.

### `(perturbed-solve t theta-node solver eps p)`, `(perturbed-argmax t theta-node eps p)`, `(perturbed-top-k t theta-node k eps p)`
Record `y_eps` on the tape with the Monte Carlo pullback. `perturbed-argmax` and
`perturbed-top-k` fix the solver.

### `(perturbed-max t theta-node solver eps p)`
Record the scalar `F_eps(theta)`; its gradient is `y_eps`.

### `(top-k-indicator k)`
Solver for the top-k polytope: `theta ↦` the 0/1 indicator of the `k` largest
entries, ties preferring the lower index.

### `(perturbed-fenchel-young t theta-node y solver eps p)`
The perturbed Fenchel-Young loss `F_eps(theta) − <theta, y>`, whose gradient is
`y_eps(theta) − y`. The term `eps Ω(y)` does not depend on `theta` and is
omitted.

### `(fenchel-young-softmax t theta-node y)`
The Shannon-entropy Fenchel-Young loss
`logsumexp(theta) − <theta, y> + sum_i y_i log y_i` (the cross-entropy for a
one-hot `y`), gradient `softmax(theta) − y`. Nonnegative, zero exactly at
`y = softmax(theta)`.

### `(fenchel-young-sparsemax t theta-node y)`
The sparsemax loss `<theta, p> − |p|²/2 + |y|²/2 − <theta, y>` with
`p = sparsemax(theta)`, gradient `p − y`. Exact on exact input.

```scheme
;; diffopt-perturbed.esk — perturbed argmax, its Jacobian, Fenchel-Young
(require core.ad.tape)
(require core.ml.gradient_estimators)
(require core.ml.diffopt)
(define theta (vector 1.0 0.5 -0.3))
(define noise (perturbation-noise 'gumbel 20000 3 2026))
(define est (perturbed-estimate argmax-onehot theta 1.0 noise))
(display (vector-ref est 0)) (newline)          ; Monte Carlo y_eps
(display (softmax theta)) (newline)             ; its exact expectation
(define g (vector 0.3 -1.0 0.7))
(display (perturbed-vjp noise 1.0 (vector-ref est 1) g)) (newline)
(display (softmax-vjp (softmax theta) g)) (newline)
;; Fenchel-Young loss on the tape: gradient = y_eps - y
(display
  (with-tape "fy"
    (lambda ()
      (let* ((t (current-tape)) (tn (tape-input t theta))
             (L (perturbed-fenchel-young t tn (vector 0.0 1.0 0.0)
                                         argmax-onehot 1.0 noise)))
        (car (tape-gradient t L (list tn)))))))
(newline)
```
```
#(0.5304 0.32630000000000003 0.1433)
#(0.5321802894514172 0.3227836620470283 0.14503604850155455)
#(0.18868803792627045 -0.3035662283804597 0.11019171678737594)
#(0.1924387029989099 -0.30289878717696445 0.11046008417805454)
#(0.5304 -0.6737 0.1433)
```

The Monte Carlo estimate and its Jacobian-vector product agree with the exact
softmax and softmax pullback to the sampling error of 20000 draws.

## Implicit differentiation

For `x*(theta)` defined by `F(x*, theta) = 0` with `d_x F` invertible, the
implicit function theorem gives the pullback of `g` as `−(d_theta F)^T u` with
`(d_x F)^T u = g`. The adjoint system is solved by `diffopt-linear-solve`:
exactly when every entry is exact, by the f64 `linear-solve` otherwise. `x*`
comes from any solver and is not differentiated through.

### `(implicit-root-vjp jx jt g)`
`jx = d_x F` (`n × n`), `jt = d_theta F` (`n × p`) at the root; returns
`−jt^T u` with `jx^T u = g`.

### `(implicit-root t F x-star theta-node [jac-x jac-theta])`
Records `x*` as a function of `theta`. `F` is `(lambda (x theta) ...)`
returning a vector. `jac-x` and `jac-theta`, when given, are procedures
`(x theta) → matrix`; written with exact arithmetic they make the pullback
exact. Otherwise the compiled `jacobian` supplies them (f64). `theta` may be a
scalar or a vector; for a scalar `theta`, `jac-theta` may return a vector.

### `(implicit-fixed-point t T x-star theta-node [jac-x jac-theta])`
`x* = T(x*, theta)`: the root of `T(x, theta) − x`. Optional Jacobians are those
of `T`.

### `(implicit-argmin t f x-star theta-node [hess-x cross])`
`x* = argmin_x f(x, theta)`: the root of `grad_x f`. Optional procedures give
`d²f/dx²` and `d²f/(dx dtheta)`; by default the compiled `hessian` and the
Jacobian of the compiled `gradient` supply them.

### `(diffopt-linear-solve a b)`, `(diffopt-jacobian f x)`
The adjoint solve described above, and the compiled `jacobian` returned as a
vector of rows.

```scheme
;; diffopt-implicit.esk — ridge least-squares sensitivity, exact and compiled
(require core.ad.tape)
(require core.exact_linalg)
(require core.ml.diffopt)
;; w*(lambda) = argmin |Xw - y|^2/2 + lambda |w|^2/2, at lambda = 1/3
(define X (vector (vector 1 2) (vector 3 -1) (vector 0 1/2)))
(define Y (vector 1 2/3 -1))
(define (ridge-matrix l)            ; X^T X + lambda I = d_w F
  (let ((m (exact-matrix-mul (exact-matrix-transpose X) X)))
    (vector (vector (+ (vector-ref (vector-ref m 0) 0) l) (vector-ref (vector-ref m 0) 1))
            (vector (vector-ref (vector-ref m 1) 0) (+ (vector-ref (vector-ref m 1) 1) l)))))
(define XtY (vector (+ 1 2 0) (+ 2 -2/3 -1/2)))
(define w* (exact-solve (ridge-matrix 1/3) XtY))
(display w*) (newline)
;; g^T dw*/dlambda with g = (2, -3/7): d_lambda F = w*
(display (implicit-root-vjp (ridge-matrix 1/3)
                            (vector (vector (vector-ref w* 0)) (vector (vector-ref w* 1)))
                            (vector 2 -3/7)))
(newline)
;; the same sensitivity on the tape, partial Jacobians from compiled AD
(define (objective w l)
  (let ((r0 (- (+ (vector-ref w 0) (* 2.0 (vector-ref w 1))) 1.0))
        (r1 (- (- (* 3.0 (vector-ref w 0)) (vector-ref w 1)) (/ 2.0 3.0)))
        (r2 (- (* 0.5 (vector-ref w 1)) -1.0)))
    (+ (* 0.5 (+ (* r0 r0) (* r1 r1) (* r2 r2)))
       (* 0.5 l (+ (* (vector-ref w 0) (vector-ref w 0)) (* (vector-ref w 1) (vector-ref w 1)))))))
(display
  (with-tape "ridge"
    (lambda ()
      (let* ((t (current-tape))
             (ln (tape-input t (/ 1.0 3.0)))
             (wn (implicit-argmin t objective
                   (vector (exact->inexact (vector-ref w* 0)) (exact->inexact (vector-ref w* 1))) ln))
             (L (record-op! t (list wn) 0 (lambda (g) (list (vector (* g 2.0) (* g (/ -3.0 7.0))))))))
        (car (tape-gradient t L (list ln)))))))
(newline)
```
```
#(633/2041 418/2041)
#(-1457082/29159767)
-0.04996891779004962
```

The exact pullback equals the closed form `−g^T (X^T X + lambda I)^{-1} w*`;
the compiled-AD default reproduces it in f64 (`−1457082/29159767 ≈
−0.0499689177900496`).

## Smooth relaxations: softmax and sparsemax

### `(sparsemax z)`
The Euclidean projection of `z` onto the probability simplex,
`p = max(z − tau, 0)`, with `tau` chosen so that `p` sums to 1. Exact on exact
input; may return exact zeros (a sparse distribution).

### `(sparsemax-jacobian z)`, `(sparsemax-vjp p g)`
`J = diag(s) − s s^T / |S|` for the support indicator `s` of `p = sparsemax(z)`;
the pullback is `s ⊙ (g − mean_{i in S} g_i)`.

### `(softmax-jacobian z)`
`diag(p) − p p^T` at `p = softmax(z)`. The pullback is
[`softmax-vjp`](ml_gradient_estimators.md).

### `(tape-softmax t z-node)`, `(tape-sparsemax t z-node)`
Tape layers with the pullbacks above.

```scheme
;; diffopt-sparsemax.esk — exact sparsemax, its Jacobian and its loss
(require core.ad.tape)
(require core.ml.diffopt)
(display (sparsemax (vector 1/2 1/3 -1))) (newline)
(display (sparsemax-jacobian (vector 1/2 1/3 -1))) (newline)
(display
  (with-tape "fy"
    (lambda ()
      (let* ((t (current-tape)) (tn (tape-input t (vector 1/2 1/3 -1))))
        (node-value (fenchel-young-sparsemax t tn (vector 0 1 0)))))))
(newline)
```
```
#(7/12 5/12 0)
#(#(1/2 -1/2 0) #(-1/2 1/2 0) #(0 0 0))
49/144
```

## Engines

Native JIT and AOT run the whole module. The bytecode VM runs everything except
`perturbation-noise`, which draws from `make-prng`; pass explicit noise with
`make-perturbation` there. Seeded noise on the VM is Planned for v1.4.1.
Reaching these pullbacks from the compiled `gradient` operator, through the
native custom-VJP node, is Planned for v1.4.1.

## References

- M. Blondel and V. Roulet, *The Elements of Differentiable Programming*, 2024.
- M. Vlastelica et al., *Differentiation of Blackbox Combinatorial Solvers*, ICLR 2020.
- Q. Berthet et al., *Learning with Differentiable Perturbed Optimizers*, NeurIPS 2020.
- M. Blondel, A. F. T. Martins and V. Niculae, *Learning with Fenchel-Young Losses*, JMLR 2020.
- A. F. T. Martins and R. F. Astudillo, *From Softmax to Sparsemax*, ICML 2016.
