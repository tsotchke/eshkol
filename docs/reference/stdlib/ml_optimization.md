# `ml.optimization` — gradient-based optimizers

**Source**: [`lib/ml/optimization.esk`](../../../lib/ml/optimization.esk)
**Require**: `(require ml.optimization)` — also **auto-loaded** by `(require stdlib)`.

Unconstrained optimizers that minimize a scalar objective `f: vector → scalar`, using the built-in `gradient` function (forward-mode AD). Provides gradient descent, Adam, L-BFGS, and nonlinear conjugate gradient, plus a backtracking line search and two small tensor helpers. All points and gradients are vectors written `#(...)`.

Optimizer functions take the objective and start point positionally, then accept hyper-parameters as an **optional positional tail** (`. opts`) — pass them in order (e.g. `lr`, then `max-iter`, then `tol`); omitted ones fall back to defaults.

## Tensor helpers

### `(tensor-dot a b)`
Dot product of two vectors.

```scheme
(require ml.optimization)
(display (tensor-dot #(1.0 2.0 3.0) #(4.0 5.0 6.0))) (newline)
```
```
32
```

### `(tensor-norm v)`
L2 norm, `(sqrt (tensor-dot v v))`.

```scheme
(display (tensor-norm #(3.0 4.0))) (newline)
```
```
5
```

## Optimizers

All examples minimize `f(x,y) = x² + y²` (minimum at the origin):

```scheme
(define (quad v) (let ((x (vref v 0)) (y (vref v 1))) (+ (* x x) (* y y))))
```

### `(gradient-descent f x0 [lr [max-iter [tol]]])`
Basic gradient descent. Defaults: `lr = 0.01`, `max-iter = 1000`, `tol = 1e-8` (on gradient norm). Returns the optimized point.

```scheme
(display (gradient-descent quad #(5.0 5.0) 0.1 1000 1e-8)) (newline)
```
```
#(3.10827e-09 3.10827e-09)
```

### `(adam f x0 [lr [beta1 [beta2]]])`
Adam (adaptive moment estimation). Optional tail sets `lr` (default `0.001`), `beta1` (`0.9`), `beta2` (`0.999`); `epsilon = 1e-8`, `max-iter = 1000`, `tol = 1e-8` are fixed internally.

```scheme
(display (adam quad #(5.0 5.0) 0.1)) (newline)
```
```
#(4.7806e-11 4.7806e-11)
```

### `(l-bfgs f x0 [max-iter [tol [m]]])`
Limited-memory BFGS with two-loop recursion, Armijo line search, and a curvature safeguard. Defaults: `max-iter = 200`, `tol = 1e-8`, history size `m = 10`.

```scheme
(display (l-bfgs quad #(5.0 5.0))) (newline)
```
```
#(0 0)
```

### `(conjugate-gradient f x0 [max-iter [tol [restart-interval]]])`
Nonlinear conjugate gradient (Fletcher–Reeves) with periodic restart. Defaults: `max-iter = 200`, `tol = 1e-8`, `restart-interval = (vector-length x0)`.

```scheme
(display (conjugate-gradient quad #(5.0 5.0))) (newline)
```
```
#(0 0)
```

### `(line-search f x d grad [alpha0 [c1 [rho]]])`
Backtracking line search returning a step size `alpha` satisfying the Armijo condition `f(x + α·d) ≤ f(x) + c1·α·gradᵀd`. Defaults: `alpha0 = 1.0`, `c1 = 1e-4`, `rho = 0.5`, and a fixed `max-iter = 30` (returns the current `alpha` if not satisfied).

```scheme
(display (line-search quad #(1.0 1.0) #(-1.0 -1.0) #(2.0 2.0))) (newline)
```
```
1
```

## Internal helpers (not in `provide`)

`vec-add`, `vec-sub`, `vec-scale`, `vec-neg`, `vec-zeros`, `vec-sq`, `adam-apply-update`, and `l-bfgs-direction` are elementwise vector utilities used by the optimizers and are not exported.

## Notes / caveats

- Inputs must be vectors (`#(...)`); like the rest of the tensor stack, passing a list will misread memory (see ESH-0069). All examples above run cleanly under both `-r` and via `(require stdlib)`.
- `gradient` is forward-mode AD; the objective `f` must be differentiable through the operations it uses.
