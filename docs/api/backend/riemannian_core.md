# `eshkol/backend/riemannian_core.h`

Closed-form constant-curvature geometry (Poincare ball, Euclidean space, round sphere) in f64, shared by the VM's geometric opcodes. WHY THIS FILE EXISTS. `lib/backend/vm_geometric.c` used to implement the curved operations as their FLAT counterparts: `hyperbolic-exp-map` was vector addition, `hyperbolic-log-map` subtraction, `geodesic-distance` and `poincare-distance` the L2 distance, `mobius-add` addition, `mobius-scalar-mul` a scale, `parallel-transport` and `riemannian-grad` the identity. Each of them accepted a curvature argument and discarded it. That body was the one every shipped VM build compiled — every CI lane, every release binary, the WASM playground. The result was Euclidean answers returned under Riemannian names, with nothing in the output showing the argument had been dropped: the same plausible-wrong-number class `frechet_mean_core.h` documents for the Euclidean weighted average that used to stand in for the Frechet mean. That second legacy qLLM dispatch body has since been deleted outright — it did not compile against the current libsemiclassical_qllm ABI and was fp32 throughout — so the forms below are the ONE implementation of this geometry on the VM engine, not the default of two. WHY A HEADER AND NOT A LIBRARY TU. Identical reason to `inc/eshkol/backend/frechet_mean_core.h`: `lib/backend/vm_geometric.c` is a unity-build include consumed by `lib/backend/eshkol_vm.c`, which is also built as a single translation unit on its own (the `eshkol-vm-standalone-test` target), so a call to an external symbol would not link there. Static functions in a header give every caller ONE source of truth with no link edge. ═══ THE MODEL, IN ONE PLACE ═══════════════════════════════════════════════ Every entry point below takes `K,` the SECTIONAL CURVATURE, and dispatches on its sign. The chart and its normalisation are fixed by TWO constants and two accessors, and nothing else in this file or in vm_geometric.c open-codes them: ESHKOL_RM_LAMBDA0 the conformal factor at the origin of the ball chart ESHKOL_RM_FLAT_LAMBDA the conformal factor the K = 0 branch uses eshkol_rm_ball_param(c) the chart's ball parameter B, from c = -K eshkol_rm_lambda(x,c,n) the conformal factor at a point K < 0 Poincare ball. With c = -K and B = eshkol_rm_ball_param(c), the metric is g_x = lambda_x^2 <.,.> with lambda_x = LAMBDA0 / (1 - B |x|^2), B = c LAMBDA0^2 / 4, so the ball has Euclidean radius 1/sqrt(B) = 2/(LAMBDA0 sqrt(c)) and sectional curvature exactly -c. At the shipped LAMBDA0 = 2 this is B = c, radius 1/sqrt(c) and lambda_x = 2/(1-c|x|^2): the convention of Ganea et al., Nickel-Kiela and geoopt, which is what every in-tree call site already passes -- `(make-hyperbolic-manifold 2 -1.0)`, `(poincare-distance x y -1.0)` -- and what `eshkol_frechet_mean_compute` takes. K = 0 Flat R^n with the metric ESHKOL_RM_FLAT_LAMBDA^2 <.,.>. K > 0 Round sphere of radius R = 1/sqrt(K), points required to lie ON it. THE FAMILY IS DISCONTINUOUS AT K = 0 AS SHIPPED, AND THAT IS A KNOWN OPEN QUESTION, NOT AN OVERSIGHT. The c -> 0 limit of the ball branch is flat space with the metric LAMBDA0^2 <.,.>, because lambda_0 = LAMBDA0 for every c. The K = 0 branch instead uses FLAT_LAMBDA = 1, the CANONICAL Euclidean metric, which is what "K = 0 is Euclidean" means to a caller and what every existing flat-reduction test asserts. Those two cannot both hold: with LAMBDA0 = 2 and FLAT_LAMBDA = 1 the geodesic distance jumps by a factor of 2 as K crosses 0, and the Riemannian gradient by a factor of 4. Which of the two to keep is a CONVENTION RULING, not a bug fix, because it changes published numbers and the AD bridge's contract along with them. Setting FLAT_LAMBDA to LAMBDA0 is the entire change on this side: the family becomes real-analytic in K on K <= 0, `geodesic-distance` at K = 0 returns 2|x-y| and `riemannian-grad` returns g/4, and eshkol_rm_distance_dK stops refusing at K = 0. Setting LAMBDA0 to 1 is the other resolution -- a ball of radius 2/sqrt(c) whose flat limit is canonical -- and is also a one-line change here, though it moves every published hyperbolic constant. exp, log, parallel transport, projection and Mobius addition are NOT affected by either constant beyond the ball parameter: a CONSTANT conformal rescale of a metric leaves the Levi-Civita connection unchanged, so those five maps are the same maps under any LAMBDA0. Only distances, norms and the gradient conversion carry the factor. RELATION TO THE AD BRIDGE. `lib/bridge/qllm_bridge.cpp` (`ad_hyperbolic_distance`, `ad_poincare_exp_map`, `ad_poincare_log_map`, `ad_geodesic_attention`) computes the same ball formulas, but its entry points take a BALL PARAMETER c > 0, not a sectional curvature: it converts with `c = (curvature == 0) ? 1 : |curvature|`. For K < 0 that coincides with this file (c = -K) and the two agree. For K >= 0 IT DOES NOT: at K = 0 the bridge silently selects the c = 1 ball, and at K > 0 it selects a ball where this file selects a sphere. The bridge must branch on the sign of K and refuse K >= 0 on its Poincare-only entry points; that is a change to the bridge, tracked on its own lane, and until it lands the "VM and AD agree" claim holds only for K < 0. Copyright (C) tsotchke SPDX-License-Identifier: MIT

63 public symbol(s) — 25 documented, 38 undocumented.

Generated by `scripts/gen_api_docs.py`. Do not edit by hand.

## Symbols

### `eshkol_rm_dot_dd`

*Function* — line 287

```c
static double eshkol_rm_dot_dd(const double* a, const double* b, int n,
 double* lo) { ... }
```

<a,b> in double-double: returns the rounded sum and sets *`lo` to the residual, so that hi + lo is the dot product to about 32 digits. WHY THE BALL CHART NEEDS THIS. Every quantity in Poincare-ball geometry is built from 1 - B|x|^2 and 1 + B<x,y>, and those are exactly the quantities the chart destroys near the boundary: a point at Euclidean distance eps from the boundary has 1 - B|x|^2 ~ 2 eps, and computing it as one minus a rounded sum of squares knows it only to an ABSOLUTE 1e-16 -- eight significant digits at eps = 1e-9, and none at all at eps = 1e-16. Near-boundary points are not an edge case here; they are the entire reason to use the ball, because that is where hyperbolic embeddings put their leaves. Accumulating the sum exactly and closing with a fused multiply-add gives 1 - B|x|^2 to full RELATIVE precision instead, so the formulas downstream inherit a well-conditioned input.

### `eshkol_rm_one_minus_dot`

*Function* — line 305

```c
static double eshkol_rm_one_minus_dot(const double* a, const double* b, double B,
 int n) { ... }
```

1 - B<a,b>, to full relative precision. The fma performs the subtraction with a single rounding of the exact product, so a result of size 1e-12 is accurate to 1e-12 * eps rather than to eps.

### `eshkol_rm_one_plus_dot`

*Function* — line 314

```c
static double eshkol_rm_one_plus_dot(const double* a, const double* b, double B,
 int n) { ... }
```

1 + B<a,b>, to full relative precision.

### `eshkol_rm_one_minus_bnorm2`

*Function* — line 324

```c
static double eshkol_rm_one_minus_bnorm2(const double* x, double B, int n) { ... }
```

1 - B|x|^2, the ball chart's conformal denominator, to full relative precision.

### `eshkol_rm_axpby_exact`

*Function* — line 351

```c
static void eshkol_rm_axpby_exact(double p, const double* a, double q,
 const double* b, int n, double* out) { ... }
```

p*a + q*b, componentwise, with the two products formed exactly and summed with their residuals. The Mobius numerator is p x + q y with p and q BOTH of size 1 - B|x|^2, and for two nearly-antipodal near-boundary points the two terms cancel down to the square of that. Rounding each product to a double first throws the answer away even when p and q are themselves exact: at B = 1, x = 0.999999999, y = -0.999999998 the terms are 2e-9 and the numerator is 3e-18, so a 1e-25 rounding in each product is a 1e-7 relative error in the result. Two-product plus two-sum keeps it.

### `eshkol_rm_ball_param`

*Function* — line 365

```c
static double eshkol_rm_ball_param(double c) { ... }
```

The ball parameter B of the chart of curvature -c: the number for which the ball is |x|^2 < 1/B and Mobius addition is (+)_B.

### `eshkol_rm_lambda`

*Function* — line 371

```c
static double eshkol_rm_lambda(const double* x, double K, int n) { ... }
```

The conformal factor lambda_x of the metric of curvature `K` at

### `eshkol_rm_metric_norm`

*Function* — line 378

```c
static double eshkol_rm_metric_norm(const double* v, const double* x, double K,
 int n) { ... }
```

The Riemannian norm of tangent vector `v` at `x.`

### `eshkol_rm_tanh_over`

*Function* — line 388

```c
static double eshkol_rm_tanh_over(double z) { ... }
```

tanh(z)/z, analytic at 0 with value 1. The series is used near zero because the quotient is 0/0 there, which is exactly the c -> 0 corner of the exponential map.

### `eshkol_rm_psi`

*Function* — line 408

```c
static double eshkol_rm_psi(double w, double* d1, double* d2) { ... }
```

psi(w) = asinh(sqrt w)/sqrt w, analytic at 0 with value 1, together with psi'(w) and psi''(w). The hyperbolic distance is 2 sqrt(R) psi(c R) with R = |x-y|^2/P (see eshkol_rm_distance), which is the STABLE form: the arccosh(1 + 2cR) it replaces loses every digit of a small separation to the 1 + eps cancellation (at c = 1, x = 0, y = 1e-9 it returns exactly 0 where the distance is 2e-9), and psi is what makes the K -> 0 corner and the near-coincident corner one expression instead of two special cases.

**Parameters**

- `d1` — psi'(w), may be NULL.
- `d2` — psi''(w), may be NULL.

### `eshkol_rm_mobius_den`

*Function* — line 458

```c
static double eshkol_rm_mobius_den(const double* x, const double* y, double B,
 int n) { ... }
```

The Mobius denominator 1 + 2B<x,y> + B^2 |x|^2 |y|^2, evaluated as (1 + B<x,y>)^2 + B^2 (|x|^2|y|^2 - <x,y>^2) which is the same number and is accurate where the direct sum is not. WHY. For two interior points the denominator is bounded below by (1 - sqrt(B)|x| sqrt(B)|y|)^2 > 0, so it never vanishes -- but it can be ARBITRARILY SMALL, and the direct sum computes it as a difference of terms of size 1. At B = 1, x = 0.999999999, y = -0.999999998 the true denominator is about 9e-18 and every digit of it is lost; the old code then floored the result at 1e-15 without scaling the numerator, turning an exact quotient of 1/3 into 0.003. The grouping above is exact instead: 1 + B<x,y> is a subtraction of two nearby numbers, hence exact in f64, and the Gram term is non-negative and computed without cancellation.

### `eshkol_rm_mobius_den_negx`

*Function* — line 475

```c
static double eshkol_rm_mobius_den_negx(const double* x, const double* y,
 double B, int n) { ... }
```

The Mobius denominator of the pair (-x, y), i.e. (1 - B<x,y>)^2 + B^2(|x|^2|y|^2 - <x,y>^2), without materialising -x. Negating x flips the sign of <x,y> and leaves the Gram term alone.

### `eshkol_rm_mobius_add`

*Function* — line 508

```c
static void eshkol_rm_mobius_add(const double* x, const double* y, double B,
 int n, double* out) { ... }
```

Mobius addition on the ball of parameter `B` > 0: x (+)_B y = ((1 + 2B<x,y> + B|y|^2) x + (1 - B|x|^2) y) / (1 + 2B<x,y> + B^2 |x|^2 |y|^2) At B = 0 this is x + y, which is why the Euclidean branch of every caller is the same code with B = 0 rather than a separate special case. THE DENOMINATOR IS NOT FLOORED. For two points strictly inside the ball, writing p = sqrt(B)|x| < 1 and q = sqrt(B)|y| < 1, the denominator is 1 + 2pq cos(theta) + p^2 q^2 >= (1 - pq)^2 > 0, so it cannot vanish and there is nothing to guard. It used to be floored at 1e-15 without scaling the numerator, which for x = 0.999999999, y = -0.999999998 at B = 1 turned an exact quotient of 1/3 into 0.003: a valid interior sum, off by two orders of magnitude, with no diagnostic. Callers that could hand this arbitrary vectors -- parallel transport was the only one -- no longer do; transport uses the linear gyration form below.

### `eshkol_rm_gyration`

*Function* — line 545

```c
static void eshkol_rm_gyration(const double* a, const double* b, const double* w,
 double B, int n, double* out) { ... }
```

The gyration gyr[a,b]w on the ball of parameter `B,` in CLOSED LINEAR FORM: D = 1 + 2B<a,b> + B^2 |a|^2 |b|^2 A = -B^2 <a,w> |b|^2 + B <b,w> + 2 B^2 <a,b><b,w> C = -B^2 <b,w> |a|^2 - B <a,w> gyr[a,b]w = w + 2 (A a + C b) / D WHY NOT THE MOBIUS COMPOSITION. gyr[a,b]w is also -(a (+) b) (+) (a (+) (b (+) w)), and that is how transport used to compute it -- by feeding the TANGENT VECTOR w into Mobius POINT addition. A tangent vector is not constrained to the ball, so an intermediate denominator can vanish even though the gyration is a globally defined linear isometry. Counterexample at B = 1, x = y = (0.5), v = (2): every gyration in one dimension is the identity, so transport from a point to itself must return (2); the intermediate (-0.5) (+) 2 has denominator zero, the old floor turned its exact 0/0 cancellation into zero, and the routine returned (0.5). The form above is linear in w and its only denominator D is the Mobius denominator of the two POINTS a and b, which are validated ball points, so it is bounded below by (1 - sqrt(B)|a| sqrt(B)|b|)^2 > 0.

### `eshkol_rm_check_point`

*Function* — line 563

```c
static const char* eshkol_rm_check_point(const double* x, double K, int n) { ... }
```

Check that `x` is a legal point of the manifold of curvature `K.`

**Returns**

NULL when it is, else a reason naming what is wrong.

### `eshkol_rm_require_interior`

*Function* — line 595

```c
static const char* eshkol_rm_require_interior(const double* out, double K, int n) { ... }
```

Check that a computed ball point is STRICTLY interior. tanh rounds to exactly 1.0 for arguments beyond about 19, so both the exponential map and Mobius scalar multiplication can land on the boundary -- `exp_0(20)` at K = -1 returned exactly 1.0, and `mobius-scalar-mul(100, 0.5, -1)` likewise -- which is not a point of the open ball and is the one place these ops could return a value outside their own codomain. Refusing names the representability limit; returning the boundary point would hand the caller a point at which lambda is infinite and every subsequent log is zero.

**Returns**

NULL when strictly interior, else a reason.

### `eshkol_rm_check_tangent`

*Function* — line 609

```c
static const char* eshkol_rm_check_tangent(const double* x, const double* v,
 double K, int n) { ... }
```

Check that `v` is tangent to the sphere of curvature `K` at `x.` A no-op for K <= 0, where every vector is tangent.

**Returns**

NULL when tangent (or K <= 0), else a reason.

### `eshkol_rm_distance`

*Function* — line 653

```c
static const char* eshkol_rm_distance(const double* x, const double* y, double K,
 int n, double* out) { ... }
```

Geodesic distance between `x` and `y` on the manifold of curvature

**Returns**

NULL on success, else a reason.

### `eshkol_rm_exp_map`

*Function* — line 707

```c
static const char* eshkol_rm_exp_map(const double* x, const double* v, double K,
 int n, double* out, double* scratch) { ... }
```

Exponential map exp_x(v) on the manifold of curvature `K.` Hyperbolic: x (+)_B [ (lambda_x/LAMBDA0) tanh(z)/z * v ] with z = sqrt(B) lambda_x |v| / LAMBDA0. Written through tanh(z)/z rather than tanh(...)/(sqrt(B)|v|) so that the B -> 0 corner is the same expression (tanh(z)/z -> 1) instead of a 0/0 special case. Spherical: cos(|v|/R) x + R sin(|v|/R) v/|v|, with the tangency of v checked rather than assumed -- a v with a radial component would leave the sphere.

**Parameters**

- `scratch` — n doubles.

**Returns**

NULL on success, else a reason.

### `eshkol_rm_log_map`

*Function* — line 770

```c
static const char* eshkol_rm_log_map(const double* x, const double* y, double K,
 int n, double* out, double* scratch) { ... }
```

Logarithmic map log_x(y) on the manifold of curvature `K,` the inverse of eshkol_rm_exp_map(). Hyperbolic: the DIRECTION comes from the Mobius numerator of (-x) (+)_B y and the MAGNITUDE from the geodesic distance, |log_x(y)| = d(x,y)/lambda_x. Both halves matter: - the direction is taken from the numerator vector alone, never divided by the Mobius denominator, which is the quantity that goes to zero for two nearly-antipodal interior points; - the magnitude is d/lambda_x, computed by eshkol_rm_distance's stable asinh form, rather than (2/(sqrt(B) lambda_x)) artanh(sqrt(B)|u|). This op used to REFUSE whenever sqrt(B)|u| rounded to 1, on the grounds that no finite log existed. That was wrong: EVERY pair of strictly interior ball points has a unique finite log, and the rounding was a numerical failure of the artanh route, not a statement about the manifold. For x = 0.999999999, y = -0.999999999 at B = 1 the true distance is 4 artanh(x) ~ 42.83; the old code refused and the AD bridge, on the same input, clamped to 1 - 1e-12 and fabricated 28.32.

**Parameters**

- `scratch` — n doubles.

**Returns**

NULL on success, else a reason.

### `eshkol_rm_transport`

*Function* — line 843

```c
static const char* eshkol_rm_transport(const double* x, const double* y,
 const double* v, double K, int n,
 double* out, double* scratch) { ... }
```

Parallel transport of tangent vector `v` from `x` to `y` along the connecting geodesic. Hyperbolic: P_{x->y}(v) = (lambda_x / lambda_y) gyr[y, -x] v, with the gyration in its LINEAR closed form (eshkol_rm_gyration) rather than expanded through Mobius point addition -- see that function for the counterexample the expansion got wrong. Spherical: P_{x->y}(v) = v - (<y,v>/(R^2 + <x,y>)) (x + y), the closed form, which is exactly tangent at y by construction. Refuses a non-tangent v (the old code returned a radial v unchanged, which is not even tangent at the destination) and refuses antipodal endpoints, where the geodesic and hence the transport is not unique.

**Parameters**

- `scratch` — n doubles.

**Returns**

NULL on success, else a reason.

### `eshkol_rm_mobius_scalar`

*Function* — line 888

```c
static const char* eshkol_rm_mobius_scalar(double r, const double* x, double K,
 int n, double* out) { ... }
```

Mobius scalar multiplication r (x)_B x = (1/sqrt(B)) tanh(r artanh(sqrt(B)|x|)) x/|x|.

**Returns**

NULL, or a reason when `x` is not strictly inside the ball, or when the result would land on it (see eshkol_rm_require_interior).

### `eshkol_rm_project`

*Function* — line 924

```c
static const char* eshkol_rm_project(const double* x, double K, int n, double* out) { ... }
```

Project `x` onto the manifold of curvature `K.` Hyperbolic: rescale onto the open ball when `x` is on or outside it, leaving interior points untouched. Spherical: rescale to radius 1/sqrt(K). Euclidean: a copy.

**Returns**

NULL on success, else a reason (only when the input cannot be scaled, i.e. it is the origin on a sphere).

### `eshkol_rm_egrad_to_rgrad`

*Function* — line 983

```c
static const char* eshkol_rm_egrad_to_rgrad(const double* g, const double* x,
 double K, int n, double* out) { ... }
```

Convert a Euclidean gradient `g` at `x` into the Riemannian gradient. Conformal branches (K <= 0): g / lambda_x^2, with lambda from the ONE accessor -- so the K = 0 value follows ESHKOL_RM_FLAT_LAMBDA and moves with the convention ruling rather than being open-coded here. Spherical: the ambient gradient projected onto the tangent space at `x.`

**Returns**

NULL on success, else a reason.

### `eshkol_rm_distance_dK`

*Function* — line 1055

```c
static const char* eshkol_rm_distance_dK(const double* x, const double* y,
 double K, int n, double* d_out,
 double* d1_out, double* d2_out) { ... }
```

Geodesic distance between `x` and `y` AND its first two derivatives with respect to the sectional curvature K, at fixed points. WHY THIS EXISTS. `curvature-gradient` (852) returned the plain SUM of a tensor's elements, `curvature-hessian` (855) returned the constant 0.0, and `adaptive-curvature-step` (856) moved K by a fixed 0.01 times that sum. None of the three differentiated anything: the first two are not derivatives of any objective, and a Hessian that is identically zero is the assertion that every objective is affine in K, made without looking at one. This function is the measurement they now report -- exact closed-form d/dK and d^2/dK^2, not a difference quotient, so there is no step size to choose and no truncation error to bound. HYPERBOLIC BRANCH (K < 0, c = -K, B = ball_param(c)). With a = |x|^2, b = |y|^2, E = |x-y|^2, P(B) = (1 - B a)(1 - B b), Rr = E/P, w = B Rr, d = LAMBDA0 sqrt(Rr) psi(w) and the derivatives are the exact chain rule on that composition. Written through psi rather than arccosh because the two agree exactly and only this one stays accurate as B -> 0 and as the points approach each other. Coincident points are handled separately: E = 0 makes d identically zero in B, so every K-derivative is exactly zero. SPHERICAL BRANCH (K > 0). A point of the sphere of radius 1/sqrt(K) is NOT a point of the sphere of a different radius, so "hold the points fixed and vary K" is not a curve in any single manifold. The family this branch differen- tiates instead holds the pair at FIXED ANGULAR POSITION and lets the radius follow K: with theta fixed, d = theta K^(-1/2), so d' = -theta K^(-3/2)/2 and d'' = 3 theta K^(-5/2)/4. The spherical formulas are published only when both derivatives are representable in f64. At the positive subnormal floor, the fixed-angle analytic derivative can be outside f64 even while d itself is finite; that case is an explicit refusal, never a successful infinity. For theta = 0 the continuous value is d' = d'' = 0, including at that floor. K = 0 IS REFUSED WHILE ESHKOL_RM_FLAT_LAMBDA != ESHKOL_RM_LAMBDA0. The ball branch tends to LAMBDA0 |x-y| as K -> 0-, the flat branch returns FLAT_LAMBDA |x-y|, and the spherical branch diverges as K -> 0+. With the two constants unequal the family is discontinuous there, so no number is the derivative and returning one would be the plausible-wrong-number case this surface exists to exclude. If the convention ruling equalises them the ball branch below is already valid AT B = 0 -- psi is analytic there -- and this refusal becomes removable.

**Parameters**

- `d_out` — the distance itself (may be NULL).
- `d1_out` — d(distance)/dK.
- `d2_out` — d^2(distance)/dK^2.

**Returns**

NULL on success, else a reason.

## Undocumented

| Symbol | Kind | Line |
|---|---|---:|
| `ESHKOL_RM_LAMBDA0` | Macro | 108 |
| `ESHKOL_RM_FLAT_LAMBDA` | Macro | 114 |
| `ESHKOL_RM_SPHERE_TOL` | Macro | 120 |
| `ESHKOL_RM_TANGENT_TOL` | Macro | 128 |
| `ESHKOL_RM_PSI_SMALL` | Macro | 133 |
| `ESHKOL_RM_TAU_SMALL` | Macro | 136 |
| `eshkol_rm_dot` | Function | 138 |
| `eshkol_rm_norm` | Function | 144 |
| `eshkol_rm_scaled_product4` | Function | 165 |
| `eshkol_rm_scaled_norm2_times` | Function | 167 |
| `eshkol_rm_scaled_product4` | Function | 187 |
| `eshkol_rm_difference_norm2` | Function | 195 |
| `eshkol_rm_scaled_dot_factor` | Function | 213 |
| `eshkol_rm_points_equal` | Function | 231 |
| `eshkol_rm_sphere_antipodal` | Function | 239 |
| `eshkol_rm_sphere_angle` | Function | 251 |
| `eshkol_rm_check_output` | Function | 332 |
| `eshkol_rm_directional` | Struct | 1168 |
| `eshkol_rm_directional::value` | Variable | 1169 |
| `eshkol_rm_directional::tangent` | Variable | 1170 |
| `eshkol_rm_dadd` | Function | 1173 |
| `eshkol_rm_dsub` | Function | 1178 |
| `eshkol_rm_dneg` | Function | 1183 |
| `eshkol_rm_dmul` | Function | 1187 |
| `eshkol_rm_ddiv` | Function | 1193 |
| `eshkol_rm_dsqrt` | Function | 1199 |
| `eshkol_rm_dtanh_over` | Function | 1205 |
| `eshkol_rm_dnorm` | Function | 1221 |
| `eshkol_rm_dscaled_norm2_times` | Function | 1228 |
| `eshkol_rm_done_minus_bnorm2` | Function | 1234 |
| `eshkol_rm_done_plus_dot` | Function | 1240 |
| `eshkol_rm_done_minus_dot` | Function | 1248 |
| `eshkol_rm_dmobius_den` | Function | 1256 |
| `eshkol_rm_daxpby_exact` | Function | 1281 |
| `eshkol_rm_dmobius_add` | Function | 1296 |
| `eshkol_rm_distance_directional` | Function | 1318 |
| `eshkol_rm_exp_directional` | Function | 1371 |
| `eshkol_rm_log_directional` | Function | 1411 |
