# `core.curvature` — signed-curvature stereographic geometry

```scheme
(require core.curvature)
```

One chart for every sign of the sectional curvature `K`, in the canonical-flat
gauge `alpha = 1` ([ADR-0018](../../design/adr/0018-signed-curvature-stereographic-geometry.md)):

```
g_K(x) = I / (1 + K |x|^2 / 4)^2          lambda_x = 1 / (1 + K |x|^2 / 4)
```

| `K` | the chart is | notes |
|---|---|---|
| `K < 0` | the open ball of radius `2/sqrt(-K)` | a point on or outside the boundary is refused |
| `K = 0` | Euclidean `R^n` | distance `‖x − y‖`, gradients unscaled |
| `K > 0` | the sphere of radius `1/sqrt(K)` minus one point | the missing point (the pole) is at infinity; the antipode of `x` is `−x/(K‖x‖²/4)` |

The family is **real-analytic in `K` through `K = 0`** at fixed chart
coordinates: the same formula serves every sign, and near zero the generalized
trigonometric ratios switch to a degree-10 series in the dimensionless
`q = K r²/4` for `|q| ≤ 2⁻⁸`. A learned curvature can therefore change sign,
and the curvature derivatives exist at zero.

Every function in this module calls the one f64 implementation in
[`inc/eshkol/backend/riemannian_core.h`](../../api/backend/riemannian_core.md),
the same code the bytecode VM's [geometry ops](geometry.md) and the AD bridge
run, through `lib/core/curvature_runtime.c`. A native program and a VM program
compute the same bits for the same inputs on the same platform.

## Engine availability

| Engine | Available |
|---|---|
| Native REPL/JIT (`eshkol-run -r`) | Yes |
| Native AOT (`eshkol-run -o`) | Yes |
| Bytecode VM | Planned for v1.4.1 (the VM reaches the same core through the geometry ops: `geodesic-distance`, `curvature-gradient`, `geodesic-attention-forward`, …) |

## Data model and errors

Points and tangent vectors are **f64 tensors** — a literal `#(0.3 -0.2 0.1)`,
`(make-tensor (list n) 0.0)`, or any op that returns one. A refused input raises
a catchable error naming the reason: a point outside the ball, a point at the
pole, an antipodal pair of the positive chart (the cut locus), a geodesic that
reaches the pole, a non-finite curvature or coordinate, a non-f64 tensor, or an
output that is not representable in f64. Nothing is clamped or projected
silently.

## Distances and curvature jets

### `(chart-distance x y K)`

The geodesic distance.

```scheme
(require core.curvature)
(define a #(0.9 0.0))
(define b #(-0.9 0.0))
(display (chart-distance a b -1.0)) (newline)   ; 1.9388011143762072
(display (chart-distance a b 0.0)) (newline)    ; 1.8
(display (chart-distance a b 1.0)) (newline)    ; 1.6914157045317628
```

The three values are one continuous family in `K`. At `K = −1` the points are
`0.9/2 = 0.45` of the ball radius out, and the distance is the classical
`4·artanh(0.45)`.

### `(chart-sq-distance x y K)`

`d(x, y)²`. This is the primitive attention scores and squared-distance losses
are built from: unlike `d`, it is smooth at `x = y`.

### `(chart-distance-jet x y K)` / `(chart-sq-distance-jet x y K)`

The **curvature jet** at fixed chart points: `(list d dd/dK d²d/dK²)` (for the
squared distance, of `d²`). Exact closed forms — the series is differentiated
termwise for `|q| ≤ 2⁻⁸` and the closed form otherwise — not difference
quotients. They are defined at `K = 0`, and the left and right limits agree
there.

```scheme
(require core.curvature)
(define x #(0.3 -0.2 0.1))
(define y #(-0.25 0.15 0.2))
(display (chart-distance-jet x y 0.0)) (newline)
; (0.6595452979136459 -0.009893179468704692 0.00024382565232245064)
(display (equal? (chart-distance-jet x y 1e-300)
                 (chart-distance-jet x y -1e-300)))
(newline)                                       ; #t
```

### `(chart-sq-distance-gradient x y K)`

`∂(d²)/∂x = −2 λ_x² log_x(y)`, the coordinate gradient. It is formed from the
log-map direction and never through `d`, so it is **exactly zero** at `x = y`
with no epsilon guard.

### `(chart-conformal-factor x K)`

`λ_x = 1/(1 + K‖x‖²/4)`; exactly `1.0` at `K = 0`.

## Maps and gyrovector operations

| Function | Returns |
|---|---|
| `(chart-exp-map x v K)` | `exp_x(v) = x ⊕ (λ_x tan_K(λ_x‖v‖)/(λ_x‖v‖) v)`; a geodesic that reaches the pole is refused |
| `(chart-log-map x y K)` | `log_x(y)`, the inverse of `chart-exp-map`; `λ_x‖log_x y‖ = d(x, y)` |
| `(chart-transport x y v K)` | parallel transport `(λ_x/λ_y) gyr[y, −x] v`, an isometry |
| `(chart-mobius-add x y K)` | `x ⊕_K y`, defined for every sign; `2 ⊗ x = x ⊕ x` |
| `(chart-mobius-scalar r x K)` | `r ⊗_K x` |
| `(chart-project x K)` | for `K < 0`, a point on or outside the ball rescaled strictly inside; otherwise a copy |
| `(chart-riemannian-grad g x K)` | `g / λ_x²` |

```scheme
(require core.curvature)
(define x #(0.3 -0.2 0.1))
(define v #(0.7 0.2 -0.3))
(define e (chart-exp-map x v 1.0))
(display e) (newline)                           ; #(1.03428734667149 0.06039184514954082 -0.2551709454539306)
(display (chart-log-map x e 1.0)) (newline)
; #(0.6999999999999998 0.19999999999999996 -0.2999999999999999)
```

## Squared-distance geodesic attention

Logits are `−d²(q_i, k_j)/(2t)` with a finite temperature `t` in
squared-distance units (default `sqrt(dim)`); the softmax is max-shifted by the
row's smallest squared distance. There is no curvature-dependent scale, so the
weights are continuous in `K` through zero, and at `K = 0` they are
`softmax(−‖q − k‖²/(2 sqrt(dim)))`.

| Function | Returns |
|---|---|
| `(chart-attention Q KEYS V K [t])` | the `[nq, dv]` outputs `Σ_j w_ij V_j` |
| `(chart-attention-weights Q KEYS V K [t])` | the `[nq, nk]` weights |
| `(chart-default-temperature dim)` | `sqrt(dim)` |

`Q` is `[nq, d]` (or one row `[d]`), `KEYS` is `[nk, d]`, `V` is `[nk, dv]`.

```scheme
(require core.curvature)
(define q #(0.3 -0.2 0.1))
(define keys (tensor (list (list 0.3 -0.2 0.1) (list -0.25 0.15 0.2))))
(define vals (tensor (list (list 1.0 0.0) (list 0.0 1.0))))
(display (chart-attention-weights q keys vals -1.0)) (newline)
; #((0.5323094147333125 0.4676905852666875))
```

## The embedded sphere

For `K > 0` the chart is isometric to the round sphere of radius `1/sqrt(K)` in
`R^{n+1}` (the model of the `spherical-*` ops):

| Function | Returns |
|---|---|
| `(chart->sphere x K)` | `(λ_x x, (1 − K‖x‖²/4) λ_x / sqrt(K))` |
| `(sphere->chart p K)` | the stereographic projection `2R p'/(R + p_n)` from the pole `p = −R e_n` |

## Convention ids and migration

Serialized geometry (checkpoints, curvature schedules, product factors,
attention temperatures) carries a convention id. The current one is
`STEREOGRAPHIC_V2_ALPHA1` (id 2, `alpha = 1`).

| Function | Returns |
|---|---|
| `(chart-convention-id)` / `(chart-alpha)` | `2` / `1.0` |
| `(make-geometry-header K)` | `(geometry-header 2 1.0 K)` |
| `(geometry-header-check h)` | `K`; raises for an unversioned record (id 0), a legacy record (id 1), an unknown id or a different `alpha` — data is never reinterpreted |
| `(migrate-legacy-ball-point x K)` | `x_v2 = 2 x_v1` for a point of the legacy ball (`alpha = 2`, radius `1/sqrt(−K)`); distances are preserved |
| `(migrate-legacy-ball-tangent v K)` | `v_v2 = 2 v_v1` (the differential of the point map), for tangent vectors and first-moment optimizer state; squared-norm state is invariant |
| `(migrate-legacy-temperature s d_ref)` | `t = d_ref / s`: a legacy linear score `−s·d` and the squared score `−d²/(2t)` have equal slope at the reference distance `d_ref` (softmax rows are invariant to a constant offset) |

Legacy positive-curvature data was ambient sphere coordinates; migrate it with
`sphere->chart`.

```scheme
(require core.curvature)
(define x1 #(0.4 -0.3 0.1))
(display (migrate-legacy-ball-point x1 -1.0)) (newline)       ; #(0.8 -0.6 0.2)
(display (geometry-header-check (make-geometry-header -1.0))) ; -1.0
(newline)
```

## Running the examples

```sh
eshkol-run -r example.esk               # JIT
eshkol-run -o example example.esk && ./example   # AOT
```

## Verification

- `tests/bridge/signed_curvature_reference_grid_test.cpp` checks the shared core
  against an independent binary128 reference at every finite binade of `K` of
  both signs (all subnormal binades, `DBL_TRUE_MIN`, `DBL_MAX`, and `0`):
  distance, squared distance, curvature jet, endpoint gradient and refusal
  class per case.
- `tests/geometry/signed_curvature_test.esk` runs this module under the JIT and
  AOT against the same reference values; `tests/vm/signed_curvature_surface.esk`
  runs the VM geometry ops against them.
- `scripts/run_signed_curvature_gate.sh` runs all three and records the
  `v14_signed_curvature_reference_grid` ICC event.

## See also

- [geometry.md](geometry.md) — the VM geometry ops on the same chart
- [manifold.md](manifold.md) — `core.manifold`
- [ADR-0018](../../design/adr/0018-signed-curvature-stereographic-geometry.md)
