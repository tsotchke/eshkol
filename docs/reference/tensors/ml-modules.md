# ML / Numerical Library Modules

Three library modules built on the tensor and autodiff primitives. Unlike the
tensor builtins, these require an explicit `(require …)`. Every `provide` is
listed with its signature and a verified run example (outputs pasted as
printed).

---

## `ml.optimization` — gradient-based optimizers

```scheme
(require ml.optimization)
```

**Provides:** `gradient-descent` `adam` `l-bfgs` `conjugate-gradient`
`line-search` `tensor-dot` `tensor-norm`

Optimizers operate on **`vector`** points and use the builtin `gradient`.
`tensor-dot`/`tensor-norm` here are module-level vector helpers (they shadow the
tensor builtins of the same name inside the module).

| Function | Signature |
|----------|-----------|
| `gradient-descent` | `(gradient-descent f x0 [lr=0.01] [max-iter=1000] [tol=1e-8])` |
| `adam` | `(adam f x0 [lr=0.001] [beta1=0.9] [beta2=0.999])` — max-iter fixed 1000, tol 1e-8 |
| `l-bfgs` | `(l-bfgs f x0 . opts)` |
| `conjugate-gradient` | `(conjugate-gradient f x0 . opts)` |
| `line-search` | `(line-search f x d grad [alpha0=1.0] [c1=1e-4] [rho=0.5])` |
| `tensor-dot` | `(tensor-dot a b)` |
| `tensor-norm` | `(tensor-norm v)` |

```scheme
;; minimize f(v) = (v0-3)² + (v1-5)², start at (0,0)
(define (f v)
  (+ (* (- (vref v 0) 3.0) (- (vref v 0) 3.0))
     (* (- (vref v 1) 5.0) (- (vref v 1) 5.0))))

(gradient-descent f (vector 0.0 0.0))       ;; => #(3 5)
(adam f (vector 0.0 0.0) 0.1)               ;; => #(3 5)   (lr=0.1)
(tensor-dot (vector 1.0 2.0 3.0) (vector 4.0 5.0 6.0))  ;; => 32
```

> `adam`'s default learning rate is `0.001`; with the fixed 1000-iteration cap
> that under-converges on this quadratic (returns ≈`#(0.92 0.96)`). Pass a
> larger `lr` (e.g. `0.1`) to converge. `gradient-descent` converges with its
> defaults.

---

## `core.manifold` — Riemannian geometry

```scheme
(require core.manifold)
```

**Provides:** `make-euclidean-manifold` `make-hyperbolic-manifold`
`make-spherical-manifold` `manifold-exp-map` `manifold-log-map`
`manifold-distance` `manifold-parallel-transport` `manifold-curvature`
`manifold-dimension` `manifold-type` `metric-component` `manifold-metric`
`manifold-metric-inverse` `christoffel-symbol` `manifold-christoffel`
`manifold-sectional-curvature` `manifold-scalar-curvature` `ricci-component`
`manifold-ricci` `riemann-component`

| Function | Signature |
|----------|-----------|
| `make-euclidean-manifold` / `make-hyperbolic-manifold` / `make-spherical-manifold` | `(make-…-manifold dim)` |
| `manifold-type` | `(manifold-type m)` → `euclidean` / `hyperbolic` / `spherical` |
| `manifold-dimension` | `(manifold-dimension m)` |
| `manifold-curvature` | `(manifold-curvature m)` → `0` / `-1` / `1` |
| `manifold-distance` | `(manifold-distance m a b)` |
| `manifold-exp-map` | `(manifold-exp-map m base v)` |

```scheme
(define e (make-euclidean-manifold 3))
(define h (make-hyperbolic-manifold 2))
(manifold-type e)         ;; => euclidean
(manifold-dimension e)    ;; => 3
(manifold-curvature h)    ;; => -1
```

Curvatures: euclidean `0`, hyperbolic `-1`, spherical `1`. The module also
exposes the metric tensor, Christoffel symbols, and sectional/scalar/Ricci/
Riemann curvature accessors listed above.

---

## `signal.fft` — Fast Fourier Transform

```scheme
(require signal.fft)
```

**Provides:** `fft` `ifft`

Radix-2 Cooley-Tukey. `fft` accepts a real or complex `vector` whose length is a
**power of 2** and returns a complex-valued vector; `ifft` inverts it.

```scheme
(define s (vector 1.0 2.0 3.0 4.0))
(fft s)          ;; => #(10 -2+2i -2 -2-2i)
(ifft (fft s))   ;; => #(1 2+5.72119e-18i 3 4-5.72119e-18i)
```

The forward transform is exact; the inverse round-trips to the input up to a
~1e-18 imaginary residue (floating-point round-off).

---

## See also

- [operations.md](operations.md) — the tensor builtins these modules build on
- [../ad/operators.md](../ad/operators.md) — the `gradient` the optimizers call
- [creation.md](creation.md) — vector vs tensor
