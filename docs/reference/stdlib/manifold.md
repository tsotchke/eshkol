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

A space that needs more than one chart — the whole sphere, hyperbolic space
in two models, a triangulated surface — is an **atlas** glued from charts:
see [Atlases](#atlases-manifolds-glued-from-charts-coremanifoldatlas)
(`core.manifold.atlas`), where the chart of this module is one chart of the
sphere atlas.

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

## Atlases: manifolds glued from charts (`core.manifold.atlas`)

**Source**: [`lib/core/manifold/atlas.esk`](../../../lib/core/manifold/atlas.esk)
**Require**: `(require core.manifold.atlas)` (not part of `(require stdlib)`; it
loads `core.sheaf.glue` and `core.sheaf.cohomology`, whose gluing results it
returns). Design: [ADR-0018, amendment "Charts as gluing data"](../../design/adr/0018-signed-curvature-stereographic-geometry.md#amendment-v140-connection-charts-as-gluing-data).

The functions above work in **one** chart. A manifold is the colimit of its
charts: an atlas is the gluing data — chart domains `φ_i(U_i) ⊂ Rⁿ`,
transition maps `g_{i→j} = φ_j ∘ φ_i⁻¹` on overlaps, and the cocycle condition

```
g_{j→k} ∘ g_{i→j} = g_{i→k}   on φ_i(U_i ∩ U_j ∩ U_k),      g_{i→i} = id
```

which is exactly what makes "`x ~ g_{i→j}(x)`" an equivalence relation. This
module checks that condition, glues points, tangent vectors, covectors,
bilinear forms, functions and vector fields along it, and computes the
topology of the gluing on the nerve of the cover with
[`core.sheaf`](sheaf.md).

### Data model

| Value | Constructor | Meaning |
|---|---|---|
| chart | `(make-chart name dim domain? [metric])` | `domain?` decides membership in `φ(U)`; `metric` (optional) maps a point to an `n × n` matrix |
| transition | `(make-transition from to map overlap?)` | `map` sends `from`-coordinates to `to`-coordinates; `overlap?` decides `φ_from(U_from ∩ U_to)` |
| atlas | `(make-atlas dim carrier charts transitions samples)` | `samples`: alist chart name → list of points (the sample net) |
| point of the glued space | `(atlas-point atlas chart x)` | a chart representative |

Points are vectors built with `vector` (not `#(...)` literals, which are f64
tensors). Give both directions of every transition the gluing uses.

**Carriers.** `'exact`: coordinates are exact rationals, every check is an
equality, and a transition that leaves the exact numbers is reported.
`(list 'float tol)`: two points (or Jacobians, or metrics) agree when every
entry satisfies `|a_i − b_i| ≤ tol · max(1, max|a_j|, max|b_j|)`;
`(atlas-default-tolerance)` is `2⁻⁴⁰`.

**What is checked.** Every check runs over the sample net: for each sample
`x` of chart `i` and all charts `j, k`, the cocycle and inverse identities,
domain consistency (`g_{i→j} x ∈ U_j ∩ U_k` exactly when `x ∈ U_i ∩ U_k`),
the chain rule `g'_{j→k}(g_{i→j} x) · g'_{i→j}(x) = g'_{i→k}(x)` with
Jacobians from forward-mode AD (`derivative` along each axis — exact at exact
points), and, when the charts carry metrics, that every transition is an
isometry `g'ᵀ G_j(g x) g' = G_i(x)`. Each visit to a sample runs in its own
`with-region`, so long checks stay flat in memory.

**Results.** Every check returns a `core.sheaf` gluing result,
`#(gluing glued value)` or `#(gluing obstructed (obstruction ...))`:

| Obstruction | Meaning |
|---|---|
| `(cocycle (i j k) x via-j direct)` | `g_{j→k} g_{i→j} x ≠ g_{i→k} x` |
| `(identity i x image)` | an explicit `g_{i→i}` moves `x` |
| `(domain (i j k) x)` | the overlaps disagree about `x` (`k = j`: `g_{i→j} x` is outside chart `j`) |
| `(outside i x)` | a sample of chart `i` is not in its domain |
| `(inexact (i j) x value)` | an exact atlas produced an inexact value |
| `(jacobian-cocycle (i j k) x lhs rhs)` | the chain rule fails |
| `(metric (i j) x pulled-back metric)` | `g_{i→j}` is not an isometry |
| `(orientation-varies (i j) x y)` | `det g'_{i→j}` changes sign on the overlap |
| `(degenerate (i j) x)` | `det g'_{i→j}(x) = 0` |
| `(monodromy loop)` | a closed path of charts along which the orientation reverses |
| `(disagree i j overlap vi vj)` | (from `core.sheaf`) local data differ on an overlap |

### Space forms: the ADR-0018 chart as one chart of an atlas

`(sphere-atlas n K)` is the sphere of curvature `K > 0` with the `2(n+1)`
stereographic charts from the poles `±R e_k` (`R = 1/√K`), named `e<k>-` and
`e<k>+`. Chart `e<n>-` (pole `−R e_n`) is the chart of `core.manifold` and
[`core.curvature`](curvature.md); every chart carries the metric
`I/(1 + K|x|²/4)²`. The atlas is exact when `K` is the square of a rational
and `(float 2⁻⁴⁰)` otherwise; an optional third argument is the list of
coordinate values of the sample grid.

```scheme
(require core.manifold.atlas)
(require core.sheaf.glue)
(define S2 (sphere-atlas 2 1))
(display (atlas-chart-names S2)) (newline)
(display (atlas-carrier S2)) (newline)
(display (glued-value (atlas-check S2))) (newline)
(display (atlas-transition-map S2 'e2- 'e2+ (vector 1/2 1/3))) (newline)
(display (glued-value (atlas-orientation S2))) (newline)
```
```
(e0- e0+ e1- e1+ e2- e2+)
exact
((charts . 6) (cocycle-checks . 3588) (jacobian-checks . 3588) (metric-checks . 732))
#(72/13 48/13)
((e0- . 1) (e0+ . -1) (e1- . -1) (e1+ . 1) (e2- . 1) (e2+ . -1))
```

The antipodal charts differ by the inversion `x ↦ 4x/(K|x|²)` (here
`(1/2, 1/3) ↦ (72/13, 48/13)`). The orientation is the choice of sign per
chart that makes every transition orientation-preserving.

Tangent vectors move by the Jacobian of the transition (exact here), covectors
by its inverse transpose, bilinear forms by `J⁻ᵀ B J⁻¹`:

```scheme
(require core.manifold.atlas)
(define S2 (sphere-atlas 2 1))
(define x (vector 1/2 1/3))
(define v (vector 3 -5))
(define w (atlas-transport-vector S2 'e2- 'e0+ x v))
(display w) (newline)
(display (atlas-transport-vector S2 'e0+ 'e2- (atlas-transition-map S2 'e2- 'e0+ x) w)) (newline)
(display (atlas-jacobian S2 'e2- 'e2+ x)) (newline)
```
```
#(-79776/7225 118368/7225)
#(3 -5)
#(#(-720/169 -1728/169) #(-1728/169 720/169))
```

`(space-form-atlas n K)` covers every sign: the sphere atlas for `K > 0`; one
chart `euclidean` with the identity metric for `K = 0`; for `K < 0` the
ADR-0018 ball of radius `2/√(−K)` (chart `ball`) glued to the upper
half-space (chart `half-space`, metric `I/(−K z_n²)`) by an inversion, which
reverses orientation. `(space-form-conformal-factor K x)` is the one
expression `1/(1 + K|x|²/4)` the core chart uses for every sign, so `K = 0`
is the Euclidean chart exactly:

```scheme
(require core.manifold.atlas)
(require core.sheaf.glue)
(define x (vector 1/3 -1/5))
(display (map (lambda (K) (space-form-conformal-factor K x)) (list 1 1/4 0 -1/4 -1))) (newline)
(display (atlas-chart-names (space-form-atlas 2 0))) (newline)
(display (atlas-metric (space-form-atlas 2 0) 'euclidean x)) (newline)
(define H2 (space-form-atlas 2 -1))
(display (atlas-chart-names H2)) (newline)
(display (atlas-transition-map H2 'ball 'half-space (vector 0 0))) (newline)
(display (glued-value (atlas-orientation-signs H2))) (newline)
(display (glued? (atlas-check H2))) (newline)
```
```
(450/467 1800/1817 1 1800/1783 450/433)
(euclidean)
#(#(1 0) #(0 1))
(ball half-space)
#(0 1)
(((ball half-space) . -1))
#t
```

`(sphere-atlas->ambient K chart x)` and `(sphere-atlas<-ambient K chart p)`
convert between chart coordinates and the embedded sphere `|p|² = 1/K` (the
pole of a chart is refused).

### Points, functions and vector fields on the glued space

```scheme
(require core.manifold.atlas)
(require core.sheaf.glue)
(define S2 (sphere-atlas 2 1))
;; the height p_2 of the embedded sphere, written in each chart
(define (height name) (lambda (x) (vector-ref (sphere-atlas->ambient 1 name x) 2)))
(define r (atlas-glue-functions S2 (map (lambda (n) (cons n (height n))) (atlas-chart-names S2))))
(display (glued? r)) (newline)
(define f (glued-value r))
(display (f (atlas-point S2 'e0+ (vector 1/2 1/3)))) (newline)
(define p (atlas-point S2 'e2+ (vector 0 0)))
(display (atlas-point-charts S2 p)) (newline)
(display (atlas-point=? S2 p (atlas-point S2 'e0- (vector 0 -2)))) (newline)
```
```
#t
48/157
(e0- e0+ e1- e1+ e2+)
#t
```

`atlas-glue-functions` and `atlas-glue-vector-fields` build a `core.sheaf`
cover whose overlaps are the samples of each pairwise overlap and call
`glue-cover`: local data that agree glue to a procedure on atlas points, and
data that do not are reported with `disagree` obstructions naming the charts.
`(atlas-recoordinatize atlas chart h h-inv)` changes one chart's coordinates
(a Cech coboundary): the result glues to the same space, with the chart's
metric pushed forward.

### Building an atlas, and a broken one

```scheme
(require core.manifold.atlas)
(require core.sheaf.glue)
;; the real line from two charts, L = (-inf, 1) and R = (0, inf)
(define (ident x) x)
(define (line shift)
  (make-atlas 1 'exact
              (list (make-chart 'L 1 (lambda (x) (< (vector-ref x 0) 1)))
                    (make-chart 'R 1 (lambda (x) (> (vector-ref x 0) 0))))
              (list (make-transition 'L 'R (lambda (x) (vector (+ (vector-ref x 0) shift))) (lambda (x) (> (vector-ref x 0) 0)))
                    (make-transition 'R 'L ident (lambda (x) (< (vector-ref x 0) 1))))
              (list (cons 'L (list (vector -1) (vector 1/2)))
                    (cons 'R (list (vector 1/2) (vector 2))))))
(display (atlas-check-cocycle (line 0))) (newline)
(display (atlas-check-cocycle (line 1/10))) (newline)
```
```
#(gluing glued (cocycle-checks 2))
#(gluing obstructed ((cocycle (L R L) #(1/2) #(3/5) #(1/2)) (cocycle (R L R) #(1/2) #(3/5) #(1/2))))
```

With the shift, `g_{R→L} ∘ g_{L→R}` is not the identity: the witness names
the triple `(L R L)`, the point, and the two values.

### The nerve, orientation and cohomology

The nerve of the cover is the set of chart index sets whose common overlap
holds a sample; `(atlas-nerve atlas)` lists each simplex with its witness
`(chart-index . point)`. `(atlas-nerve-space atlas)` is the nerve as a
`core.sheaf` finite space (its points are the overlaps `U_J`, ordered by
inclusion). The orientation sheaf is the rank-one local system whose
restrictions are the signs of `det g'_{i→j}`:

- `(atlas-orientation-signs atlas)` — the sign on every nerve edge;
- `(atlas-w1 atlas)` — `(w1 cochain trivial? evidence)`: the Z/2 cochain, and
  either the orientation that trivializes it or a loop of charts on which it
  is 1;
- `(atlas-orientation atlas)` / `(atlas-orientable? atlas)` — the global
  section of the orientation sheaf, or the `monodromy` loop;
- `(atlas-cech-cohomology atlas kmax coefficients)` with coefficients `'Z` or
  `'orientation` — Cech cohomology of the cover, `((k betti torsion) ...)`;
  `(atlas-cech-complex ...)` is the complex itself (for
  `cohomology-generators`);
- `(atlas-nerve-cohomology atlas kmax coefficients)` — derived cohomology of
  the local system on the nerve space, an independent path to the same
  groups;
- `(atlas-mod2-betti atlas kmax)` — dimensions of `H^k(N; Z/2)` from the
  integral groups (universal coefficients).

For a good cover these are the groups of the manifold. The vertex-star atlas
of a triangulated surface is one: `(simplicial-surface-atlas triangles)`
maps the open star of each vertex piecewise linearly onto a fan of exact
rational points, the transitions are affine on each triangle, and the nerve
is the triangulation itself. The 6-vertex projective plane:

```scheme
(require core.manifold.atlas)
(require core.sheaf.glue)
(define rp2 (simplicial-surface-atlas
             (list (list 1 2 3) (list 1 3 4) (list 1 4 5) (list 1 5 6) (list 1 6 2)
                   (list 2 3 5) (list 3 4 6) (list 4 5 2) (list 5 6 3) (list 6 2 4))))
(display (glued? (atlas-check rp2))) (newline)
(display (gluing-obstructions (atlas-orientation rp2))) (newline)
(display (atlas-cech-cohomology rp2 2 'Z)) (newline)
(display (atlas-cech-cohomology rp2 2 'orientation)) (newline)
(display (atlas-nerve-cohomology rp2 2 'orientation)) (newline)
(display (atlas-mod2-betti rp2 2)) (newline)
```
```
#t
((monodromy (1 2 4 1)))
((0 1 ()) (1 0 ()) (2 0 (2)))
((0 0 ()) (1 0 (2)) (2 1 ()))
((0 0 ()) (1 0 (2)) (2 1 ()))
(1 1 1)
```

`RP²` is not orientable: the loop `1 → 2 → 4 → 1` reverses orientation.
`H*(RP²; Z) = Z, 0, Z/2`; with orientation coefficients `0, Z/2, Z` (twisted
Poincaré duality); with Z/2 coefficients `1, 1, 1`.

### Function table

| Function | Returns |
|---|---|
| `chart?` `chart-name` `chart-dimension` `chart-domain?` `chart-metric` | chart accessors |
| `transition?` `transition-from` `transition-to` `transition-map` `transition-overlap?` | transition accessors |
| `atlas?` `atlas-dimension` `atlas-carrier` `atlas-chart-names` `atlas-chart-count` `atlas-chart` `atlas-chart-index` `atlas-samples` `atlas-transition` | atlas accessors |
| `(atlas-overlap? A from to x)` / `(atlas-transition-map A from to x)` | overlap membership / `g_{from→to}(x)` (refused outside the overlap) |
| `(atlas-same-point? A x y)` | equality under the atlas carrier |
| `(atlas-check A)` | all checks; glued value: the counts of checks made |
| `(atlas-check-cocycle A)` / `(atlas-cocycle-violations A)` | the cocycle, identity and domain checks |
| `(atlas-jacobian A from to x)` / `(atlas-jacobian-violations A)` | the Jacobian of a transition / chain-rule failures |
| `(atlas-metric A chart x)` / `(atlas-metric-violations A)` | a chart metric / isometry failures |
| `atlas-point` `atlas-point?` `atlas-point-chart` `atlas-point-local` `atlas-point-coordinates` `atlas-point-charts` `atlas-point-normalize` `atlas-point=?` | points of the glued space |
| `atlas-transport-vector` `atlas-transport-covector` `atlas-transport-bilinear` | `(A from to x value)`: tensors moved to chart `to` |
| `atlas-glue-functions` `atlas-glue-vector-fields` `atlas-recoordinatize` | gluing local data; changing one chart |
| `atlas-nerve` `atlas-nerve-space` `atlas-orientation-signs` `atlas-w1` `atlas-orientation` `atlas-orientable?` `atlas-local-system` `atlas-cech-complex` `atlas-cech-cohomology` `atlas-nerve-cohomology` `atlas-mod2-betti` | the nerve and its cohomology |
| `sphere-atlas` `sphere-atlas-core-chart` `sphere-atlas-chart-name` `sphere-atlas->ambient` `sphere-atlas<-ambient` `space-form-atlas` `space-form-conformal-factor` `simplicial-surface-atlas` | constructors |

Checks run over the sample net, not over every point of an overlap. Turning
a sample check of a rational transition of bounded degree into a proof on
the whole overlap is Planned for v1.4.1. The suite is
[`tests/manifold/manifold_atlas_test.esk`](../../../tests/manifold/manifold_atlas_test.esk)
(CTest `manifold_atlas_jit`, `manifold_atlas_aot`).

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
