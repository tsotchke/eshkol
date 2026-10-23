# `core.manifold` — Riemannian manifolds for geometric ML

**Source**: [`lib/core/manifold.esk`](../../../lib/core/manifold.esk)
**Require**: `(require core.manifold)` — auto-loaded via `(require stdlib)` (it is listed in `lib/stdlib.esk`), but the examples below `(require core.manifold)` explicitly so they stand alone.

Constructors and operations for three constant-curvature Riemannian spaces:
Euclidean (flat, K=0), Hyperbolic ball (K=−1), and Spherical (K=+1,
stereographic), all in **one chart**: the ADR-0018 κ-stereographic chart in the
`α = 1` gauge, with metric `I/(1 + K‖x‖²/4)²` — the same chart as the VM
[geometry ops](geometry.md) and [`core.curvature`](curvature.md). The hyperbolic
ball has radius 2; the spherical chart covers the sphere except its pole. It provides the geometric-ML primitives (exponential/logarithmic
maps, geodesic distance, parallel transport) plus a full closed-form differential-
geometry surface (metric, inverse metric, Christoffel symbols, sectional/scalar
curvature, Ricci and Riemann tensors). Pure Scheme over the core math builtins, so
it runs identically under REPL/JIT and AOT.

## Data model

- **A manifold** is a 2-element vector `#(type dim)` where `type` is one of the symbols
  `euclidean`, `hyperbolic`, `spherical`. Do not construct it by hand — use the
  constructors below. Inspect with `manifold-type` / `manifold-dimension`.
- **Points and tangent vectors** are 1-D numeric vectors. Use `#(...)` (vector literal)
  or `(vector ...)` / `(make-vector n x)`. Both forms verified to work. See
  [Known issues](#known-issues) — passing a plain `list` silently returns a wrong result.
- **Matrices** returned by `manifold-metric`, `manifold-metric-inverse`, `manifold-ricci`
  are vectors of row-vectors (index `(vector-ref (vector-ref M i) j)` for entry `M[i][j]`).
- The **rank-3 Christoffel tensor** from `manifold-christoffel` is indexed `[k][i][j]`
  (a vector of `k` slices, each an `n×n` matrix).

## Functions

### `(make-euclidean-manifold dim)` / `(make-hyperbolic-manifold dim)` / `(make-spherical-manifold dim)`
Construct a manifold of the given integer dimension. Hyperbolic is the ball of
radius 2 (K=−1); spherical is the stereographic chart (K=+1); Euclidean is flat
(K=0). For a curvature other than ±1, or for curvature derivatives, use
[`core.curvature`](curvature.md).
Returns the `#(type dim)` object.

```scheme
;; example.esk
(require core.manifold)
(define H (make-hyperbolic-manifold 3))
(display (manifold-type H)) (newline)
(display (manifold-dimension H)) (newline)
```
```
hyperbolic
3
```

### `(manifold-type m)`
Returns the type symbol (`euclidean` / `hyperbolic` / `spherical`).

### `(manifold-dimension m)`
Returns the integer dimension.

### `(manifold-curvature m)`
Returns the constant sectional curvature K as a float: `-1.0` hyperbolic, `1.0`
spherical, `0.0` euclidean.

```scheme
(require core.manifold)
(display (manifold-curvature (make-hyperbolic-manifold 3))) (newline)
(display (manifold-curvature (make-euclidean-manifold 3)))  (newline)
(display (manifold-curvature (make-spherical-manifold 3)))  (newline)
```
```
-1.0
0.0
1.0
```

### `(manifold-distance m a b)`
Geodesic distance between points `a` and `b`: `artan_B(‖(−a) ⊕_B b‖)` with
`B = −K/4`, where `artan_B` is `artanh(√B r)/√B` on the ball and `atan(√−B r)/√−B` on
the sphere chart; Euclidean is the L2 norm of `a−b`.

```scheme
(require core.manifold)
(define a #(0.1 0.2 0.0))
(define b #(0.3 -0.1 0.0))
(display (manifold-distance (make-hyperbolic-manifold 3) a b)) (newline)
(display (manifold-distance (make-euclidean-manifold 3)  a b)) (newline)
(display (manifold-distance (make-spherical-manifold 3)  a b)) (newline)
```
```
0.3654157126323245
0.36055512754639896
0.3557995806927712
```

Edge case: `(manifold-distance m x x)` returns `0.0`. Points are vectors; see
[Points are vectors](#points-are-vectors).

### `(manifold-exp-map m base v)`
Exponential map: move from point `base` along tangent vector `v`, returning the endpoint
point: `base ⊕_B tan_B(λ‖v‖)·v/‖v‖` (`tanh` on the ball, `tan` on the sphere
chart); Euclidean is `base + v`. A near-zero tangent returns `base` unchanged.

```scheme
(require core.manifold)
(define H (make-hyperbolic-manifold 3))
(define a #(0.1 0.2 0.0))
(define v #(0.05 -0.03 0.02))
(display (vector-ref (manifold-exp-map H a v) 0)) (newline)
```
```
0.1501046611543684
```

### `(manifold-log-map m base p)`
Logarithmic map: the tangent vector at `base` pointing toward `p` (inverse of
`manifold-exp-map`). Euclidean is `p − base`. Returns the zero vector when `p` coincides
with `base`.

```scheme
(require core.manifold)
(define H (make-hyperbolic-manifold 3))
(define a #(0.1 0.2 0.0))
(define v #(0.05 -0.03 0.02))
;; exp then log round-trips back to v
(display (vector-ref (manifold-log-map H a (manifold-exp-map H a v)) 0)) (newline)
```
```
0.049999999999999885
```

### `(manifold-parallel-transport m base-a base-b v)`
Transport tangent vector `v` from `base-a` to `base-b`. Euclidean is the identity;
hyperbolic and spherical use `(λ_a/λ_b)·gyr[base-b,−base-a]v` with the linear
closed form for gyration — the exact constant-curvature gyrovector transport for
this chart, not a first-order rescaling.

```scheme
(require core.manifold)
(define H (make-hyperbolic-manifold 3))
(define a #(0.1 0.2 0.0))
(define b #(0.3 -0.1 0.0))
(define v #(0.05 -0.03 0.02))
(display (vector-ref (manifold-parallel-transport H a b v) 0)) (newline)
```
```
0.050375696202531635
```

### `(manifold-sectional-curvature m)`
Sectional curvature — equal to `manifold-curvature` for these constant-curvature spaces.
Takes only the manifold (constant across all points/planes).

```scheme
(require core.manifold)
(display (manifold-sectional-curvature (make-hyperbolic-manifold 3))) (newline)
```
```
-1.0
```

### `(manifold-scalar-curvature m)`
Scalar curvature R = K·n·(n−1), where n is the dimension. Takes only the manifold.

```scheme
(require core.manifold)
(display (manifold-scalar-curvature (make-hyperbolic-manifold 3))) (newline)  ; -1*3*2
(display (manifold-scalar-curvature (make-euclidean-manifold 3)))  (newline)
(display (manifold-scalar-curvature (make-spherical-manifold 3)))  (newline)
```
```
-6.0
0.0
6.0
```

### `(metric-component m x i j)`
Single metric-tensor entry g_ij at point `x`. All three spaces are conformally flat:
g_ij = λ(x)²·δ_ij, where λ = 1/(1 + K‖x‖²/4): 1 (euclidean), 1/(1−‖x‖²/4)
(hyperbolic), 1/(1+‖x‖²/4) (spherical). Off-diagonal entries are 0.

```scheme
(require core.manifold)
(define H (make-hyperbolic-manifold 3))
(display (metric-component H #(0.1 0.2 0.0) 0 0)) (newline)
```
```
1.0254766864284566
```

### `(manifold-metric m x)`
Full n×n metric tensor at `x` as a vector of row-vectors. For n ≥ 48 the rows are
materialized in parallel via `parallel-map`.

```scheme
(require core.manifold)
(define H (make-hyperbolic-manifold 3))
(display (vector-ref (vector-ref (manifold-metric H #(0.1 0.2 0.0)) 0) 0)) (newline)
```
```
1.0254766864284566
```

### `(manifold-metric-inverse m x)`
Full n×n inverse metric g^ij at `x` (diagonal 1/λ², off-diagonal 0), same vector-of-rows
layout.

```scheme
(require core.manifold)
(define H (make-hyperbolic-manifold 3))
(display (vector-ref (vector-ref (manifold-metric-inverse H #(0.1 0.2 0.0)) 0) 0)) (newline)
```
```
0.9751562500000004
```

### `(christoffel-symbol m x i j k)`
Single Christoffel symbol Γ^k_ij at `x` (connection coefficient). Closed form for
conformally-flat metrics: Γ^k_ij = δ_ik ∂_j lnλ + δ_jk ∂_i lnλ − δ_ij ∂_k lnλ.
Symmetric in the lower indices i,j. Zero everywhere for Euclidean. Argument order is
`i j k` (lower, lower, upper).

```scheme
(require core.manifold)
(define H (make-hyperbolic-manifold 3))
(display (christoffel-symbol H #(0.1 0.2 0.0) 0 0 0)) (newline)
(display (christoffel-symbol (make-euclidean-manifold 3) #(0.1 0.2 0.0) 0 1 0)) (newline)
```
```
0.050632911392405056
0.0
```

### `(manifold-christoffel m x)`
Full rank-3 Christoffel tensor at `x`, indexed `[k][i][j]` (a length-n vector of n×n
matrix slices). For n ≥ 48 the k-slices are built in parallel.

```scheme
(require core.manifold)
(define H (make-hyperbolic-manifold 3))
(display (vector-length (manifold-christoffel H #(0.1 0.2 0.0)))) (newline)
```
```
3
```

### `(ricci-component m x i j)`
Single Ricci-tensor entry. Einstein form for constant curvature: Ric_ij = K·(n−1)·g_ij.

```scheme
(require core.manifold)
(define H (make-hyperbolic-manifold 3))
(display (ricci-component H #(0.1 0.2 0.0) 0 0)) (newline)  ; -1*2*g_00
```
```
-2.050953372856913
```

### `(manifold-ricci m x)`
Full n×n Ricci tensor at `x` as a vector of row-vectors.

```scheme
(require core.manifold)
(define H (make-hyperbolic-manifold 3))
(display (vector-ref (vector-ref (manifold-ricci H #(0.1 0.2 0.0)) 0) 0)) (newline)
```
```
-2.050953372856913
```

### `(riemann-component m x i j k l)`
Single Riemann-curvature-tensor entry. Constant-curvature form:
R_ijkl = K·(g_ik·g_jl − g_il·g_jk). Takes four lower indices i,j,k,l.

```scheme
(require core.manifold)
(define H (make-hyperbolic-manifold 3))
(display (riemann-component H #(0.1 0.2 0.0) 0 1 0 1)) (newline)  ; = -λ⁴
```
```
-1.051602434408287
```

## Points are vectors

All operations index points with `vector-ref` / `vector-length`. Pass `#(...)`,
`(vector ...)`, or `(make-vector n x)`; a list argument raises a catchable type error
(`vector-length: expected vector or tensor, got pair`).

## Notes

- The full regression suite for this module lives at
  [`tests/manifold/manifold_test.esk`](../../../tests/manifold/manifold_test.esk) (exp/log
  round-trip, curvature closed forms, Christoffel symmetry, parallel materializers at
  dim ≥ 48). Additional geometric-surface coverage is in
  [`tests/vm/geometric_surface_regression.esk`](../../../tests/vm/geometric_surface_regression.esk).
- The internal `mf-*` helpers (`mf-dot`, `mf-mobius-add`, `mf-lambda`, `mf-build-matrix`,
  `mf-build-rank3`, …) are not exported and are not part of the public API.
- All 20 provided symbols were verified by running under the `-r` (REPL/JIT) path.
