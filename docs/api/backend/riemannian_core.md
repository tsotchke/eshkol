# `eshkol/backend/riemannian_core.h`

The ADR-0018 signed-curvature stereographic chart (one chart for every sign of K, alpha = 1) and the embedded round sphere, in f64, shared by the VM's geometric opcodes, the AD bridge and the native core.curvature module. WHY THIS FILE EXISTS. `lib/backend/vm_geometric.c` used to implement the curved operations as their FLAT counterparts: `hyperbolic-exp-map` was vector addition, `hyperbolic-log-map` subtraction, `geodesic-distance` and `poincare-distance` the L2 distance, `mobius-add` addition, `mobius-scalar-mul` a scale, `parallel-transport` and `riemannian-grad` the identity. Each of them accepted a curvature argument and discarded it. That body was the one every shipped VM build compiled — every CI lane, every release binary, the WASM playground. The result was Euclidean answers returned under Riemannian names, with nothing in the output showing the argument had been dropped: the same plausible-wrong-number class `frechet_mean_core.h` documents for the Euclidean weighted average that used to stand in for the Frechet mean. That second legacy qLLM dispatch body has since been deleted outright — it did not compile against the current libsemiclassical_qllm ABI and was fp32 throughout — so the forms below are the ONE implementation of this geometry on the VM engine, not the default of two. WHY A HEADER AND NOT A LIBRARY TU. Identical reason to `inc/eshkol/backend/frechet_mean_core.h`: `lib/backend/vm_geometric.c` is a unity-build include consumed by `lib/backend/eshkol_vm.c`, which is also built as a single translation unit on its own (the `eshkol-vm-standalone-test` target), so a call to an external symbol would not link there. Static functions in a header give every caller ONE source of truth with no link edge. ═══ THE MODEL, IN ONE PLACE (ADR-0018) ════════════════════════════════════ Every entry point below takes `K,` the SIGNED SECTIONAL CURVATURE, and evaluates ONE chart for every sign: the kappa-stereographic chart in the canonical-flat gauge alpha = 1 (docs/design/adr/0018-signed-curvature- stereographic-geometry.md). The chart and its normalisation are fixed by one constant and two accessors, and nothing else in this file or its callers open-codes them: ESHKOL_RM_ALPHA the conformal factor at the origin (alpha = 1) eshkol_rm_ball_param(c) the chart parameter B = c alpha^2 / 4, c = -K eshkol_rm_lambda(x,K,n) the conformal factor at a point The metric is g_K(x) = lambda_x^2 <.,.>, lambda_x = alpha / (1 - B |x|^2), B = -K alpha^2 / 4 (so rho = K/4 = -B at alpha = 1), and the gyrovector operations are Mobius addition (+)_B with the SAME signed B. Consequences, all of them by construction rather than by special case: K < 0 B > 0: the open ball of radius 1/sqrt(B) = 2/sqrt(-K). K = 0 B = 0: canonical Euclidean R^n, lambda = 1, (+)_0 = vector sum. K > 0 B < 0: the round sphere of radius 1/sqrt(K) minus one point (the POLE, at infinity in the chart). Every finite point is a point of the manifold; the antipode of x is -x/(|B| |x|^2), its CUT LOCUS. The family is real-analytic in K through K = 0 at fixed chart coordinates. The generalized trigonometric ratios (eshkol_rm_psi, eshkol_rm_tan_ratio, eshkol_rm_artan_ratio) are functions of one dimensionless signed argument and switch to a degree-10 series for |argument| <= ESHKOL_RM_Q_SERIES, so the K -> 0 corner and the K = 0 point are one expression. Positive cut loci and the negative ball boundary are explicit domain errors, never clamps. Legacy coordinates (LEGACY_BALL_V1: the alpha = 2 ball of radius 1/sqrt(-K), K = 0 with lambda = 1, K > 0 as an ambient sphere in R^{n+1}) are read only through the migration helpers at the end of this file: x_v2 = 2 x_v1 for ball points and tangent vectors, the stereographic projection for ambient sphere points. Serialized geometry carries ESHKOL_GEOMETRY_CONVENTION_* and is refused when the id is missing or does not match. The ambient round sphere survives as an EMBEDDED MODEL, not as a second chart: the `eshkol_rm_sphere_*` functions take points of R^{n+1} on the sphere of radius 1/sqrt(K) and back the ops whose contract is that embedding (`great-circle-distance`, `slerp`, `spherical-exp/log/project`, and the space-form SPHERICAL factor). eshkol_rm_sphere_to_chart and eshkol_rm_chart_to_sphere are the isometry between the two. exp, log, parallel transport, projection and Mobius addition depend on the chart parameter B only: a constant conformal rescale of a metric leaves the Levi-Civita connection unchanged. Distances, norms, the gradient conversion and the curvature derivatives carry alpha. RELATION TO THE AD BRIDGE. `lib/bridge/qllm_bridge.cpp` includes this header and calls these same f64 primitives for its hyperbolic distance, Poincare exponential/logarithmic maps, and geodesic-attention distance scores. The bridge validates that its public curvature is negative, converts it to the common ball parameter c = -K, and retains the common forward weights for its reverse rule. There is no second bridge-side forward formula to drift from this model: the VM and bridge distance/map forwards are one implementation. Copyright (C) tsotchke SPDX-License-Identifier: MIT

115 public symbol(s) — 50 documented, 65 undocumented.

Generated by `scripts/gen_api_docs.py`. Do not edit by hand.

## Symbols

### `eshkol_rm_sphere_direction_angle`

*Function* — line 647

```c
static const char* eshkol_rm_sphere_direction_angle(const double* x,
 const double* y,
 int n, double* out) { ... }
```

The great-circle angle between the DIRECTIONS of two nonzero vectors. `great-circle-distance` is documented (docs/reference/stdlib/geometry.md, id 819) as the scale-invariant angle between two arbitrary vectors, not the distance between two points already on the sphere: each input is normalized with the hypot norm first, so a pair of subnormals and a pair spanning 1e300/1e-300 give the same answer as their unit representatives. Computing it as `acos(<x,y>/(|x||y|))` instead — the raw dot product divided by the product of the norms — underflows both the dot product and |x||y| to zero for subnormal inputs and overflows |x||y| for large ones, so it reported an angle of acos(0/0) or acos(0) rather than the angle that is there. Contract, in the order the checks apply: - a non-finite coordinate propagates as NaN under IEEE rules rather than raising or collapsing to zero; - a zero-norm endpoint has no direction and the documented answer is 0.0; - an EXACTLY negatively-collinear pair is the cut locus, where the shortest geodesic is not unique, and raises. The predicate is the exact dyadic-rational one (eshkol_rm_sphere_antipodal), so a genuine near-antipode whose products underflow is still evaluated; - otherwise the cancellation-free atan2 form (eshkol_rm_sphere_angle).

**Returns**

NULL on success (angle written to `out),` else a reason.

### `eshkol_rm_dot_dd`

*Function* — line 685

```c
static double eshkol_rm_dot_dd(const double* a, const double* b, int n,
 double* lo) { ... }
```

<a,b> in double-double: returns the rounded sum and sets *`lo` to the residual, so that hi + lo is the dot product to about 32 digits. WHY THE BALL CHART NEEDS THIS. Every quantity in Poincare-ball geometry is built from 1 - B|x|^2 and 1 + B<x,y>, and those are exactly the quantities the chart destroys near the boundary: a point at Euclidean distance eps from the boundary has 1 - B|x|^2 ~ 2 eps, and computing it as one minus a rounded sum of squares knows it only to an ABSOLUTE 1e-16 -- eight significant digits at eps = 1e-9, and none at all at eps = 1e-16. Near-boundary points are not an edge case here; they are the entire reason to use the ball, because that is where hyperbolic embeddings put their leaves. Accumulating the sum exactly and closing with a fused multiply-add gives 1 - B|x|^2 to full RELATIVE precision instead, so the formulas downstream inherit a well-conditioned input.

### `eshkol_rm_one_minus_dot`

*Function* — line 702

```c
static double eshkol_rm_one_minus_dot(const double* a, const double* b, double B,
 int n) { ... }
```

1 - B<a,b>, to full relative precision. The fma performs the subtraction with a single rounding of the exact product, so a result of size 1e-12 is accurate to 1e-12 * eps rather than to eps.

### `eshkol_rm_one_plus_dot`

*Function* — line 711

```c
static double eshkol_rm_one_plus_dot(const double* a, const double* b, double B,
 int n) { ... }
```

1 + B<a,b>, to full relative precision.

### `eshkol_rm_one_minus_factor_norm2`

*Function* — line 722

```c
static double eshkol_rm_one_minus_factor_norm2(const double* x, double f1,
 double f2, int n) { ... }
```

1 - (f1 f2)|x|^2 to full relative precision, with the factor f1 f2 carried through the exponent-separated product (never formed as one rounded double).

### `eshkol_rm_one_minus_bnorm2`

*Function* — line 776

```c
static double eshkol_rm_one_minus_bnorm2(const double* x, double B, int n) { ... }
```

1 - B|x|^2, the chart's conformal denominator, to full relative precision.

### `eshkol_rm_chart_den`

*Function* — line 783

```c
static double eshkol_rm_chart_den(const double* x, double K, int n) { ... }
```

The conformal denominator 1 + K alpha^2 |x|^2 / 4 (= 1 - B|x|^2) from K directly, so it is exact in its parameter for EVERY finite K: B = -K alpha^2/4 itself loses its low bits below |K| = 2^-1020.

### `eshkol_rm_axpby_exact`

*Function* — line 808

```c
static void eshkol_rm_axpby_exact(double p, const double* a, double q,
 const double* b, int n, double* out) { ... }
```

p*a + q*b, componentwise, with the two products formed exactly and summed with their residuals. The Mobius numerator is p x + q y with p and q BOTH of size 1 - B|x|^2, and for two nearly-antipodal near-boundary points the two terms cancel down to the square of that. Rounding each product to a double first throws the answer away even when p and q are themselves exact: at B = 1, x = 0.999999999, y = -0.999999998 the terms are 2e-9 and the numerator is 3e-18, so a 1e-25 rounding in each product is a 1e-7 relative error in the result. Two-product plus two-sum keeps it.

### `eshkol_rm_ball_param`

*Function* — line 824

```c
static double eshkol_rm_ball_param(double c) { ... }
```

The chart parameter B for the curvature K = -c: B = c alpha^2/4. Signed: B > 0 for K < 0 (the ball |x|^2 < 1/B), B = 0 for K = 0, B < 0 for K > 0 (the stereographic sphere chart). Mobius addition is (+)_B for every sign. THE ONE PLACE the gauge enters the chart parameter.

### `eshkol_rm_lambda`

*Function* — line 832

```c
static double eshkol_rm_lambda(const double* x, double K, int n) { ... }
```

The conformal factor lambda_x = alpha/(1 - B|x|^2) of the metric of curvature `K` at `x,` for every sign of K. THE ONE PLACE this is computed. At K = 0 it is exactly alpha. Infinite or zero when the denominator is not representable (the pole of a positive chart).

### `eshkol_rm_metric_norm`

*Function* — line 879

```c
static double eshkol_rm_metric_norm(const double* v, const double* x, double K,
 int n) { ... }
```

The Riemannian norm of tangent vector `v` at `x.`

### `eshkol_rm_psi`

*Function* — line 913

```c
static double eshkol_rm_psi(double w, double* d1, double* d2) { ... }
```

psi(w) = asinh(sqrt w)/sqrt w for w >= 0 and asin(sqrt(-w))/sqrt(-w) for -1 <= w < 0, with psi'(w) and psi''(w). The geodesic distance is alpha sqrt(E/P) psi(B E/P) on the chart (see eshkol_rm_distance). Both branches share one Taylor series, psi(w) = sum_n a_n w^n, a_n = (-1)^n (2n)!/(4^n (n!)^2 (2n+1)), so K -> 0 from either side and K = 0 itself are one expression. Outside the series window the derivatives use psi' = (g - psi)/(2w), psi'' = (g' - 3 psi')/(2w) with g = (1+w)^(-1/2), which hold on both branches.

**Parameters**

- `d1` — psi'(w), may be NULL.
- `d2` — psi''(w), may be NULL.

**Returns**

NaN for w < -1 (beyond the antipode; no chart pair produces it).

### `eshkol_rm_tan_ratio`

*Function* — line 986

```c
static double eshkol_rm_tan_ratio(double s, double* d1) { ... }
```

tan_ratio(s) = tanh(sqrt s)/sqrt s (s >= 0), tan(sqrt -s)/sqrt -s (s < 0), analytic at 0 with value 1, and its derivative. The exponential map's step factor: exp_x(v) = x (+)_B [(lambda_x/alpha) tan_ratio(s) v] with s = B (lambda_x |v| / alpha)^2. On the positive branch the ratio has a pole at sqrt(-s) = pi/2: the geodesic reaches the pole of the chart there, and the caller reports it.

**Parameters**

- `d1` — tan_ratio'(s), may be NULL. Off the series window it is (1 - s T^2 - T)/(2s) on both branches.

### `eshkol_rm_artan_ratio`

*Function* — line 1020

```c
static double eshkol_rm_artan_ratio(double s) { ... }
```

artan_ratio(s) = artanh(sqrt s)/sqrt s (0 <= s < 1), atan(sqrt -s)/sqrt -s (s < 0); analytic at 0 with value 1. NaN for s >= 1 (on or beyond the ball boundary).

### `eshkol_rm_mobius_den`

*Function* — line 1052

```c
static double eshkol_rm_mobius_den(const double* x, const double* y, double B,
 int n) { ... }
```

The Mobius denominator 1 + 2B<x,y> + B^2 |x|^2 |y|^2, evaluated as (1 + B<x,y>)^2 + B^2 (|x|^2|y|^2 - <x,y>^2) which is the same number and is accurate where the direct sum is not. WHY. For two interior points the denominator is bounded below by (1 - sqrt(B)|x| sqrt(B)|y|)^2 > 0, so it never vanishes -- but it can be ARBITRARILY SMALL, and the direct sum computes it as a difference of terms of size 1. At B = 1, x = 0.999999999, y = -0.999999998 the true denominator is about 9e-18 and every digit of it is lost; the old code then floored the result at 1e-15 without scaling the numerator, turning an exact quotient of 1/3 into 0.003. The grouping above is exact instead: 1 + B<x,y> is a subtraction of two nearby numbers, hence exact in f64, and the Gram term is non-negative and computed without cancellation.

### `eshkol_rm_mobius_den_negx`

*Function* — line 1070

```c
static double eshkol_rm_mobius_den_negx(const double* x, const double* y,
 double B, int n) { ... }
```

The Mobius denominator of the pair (-x, y), i.e. (1 - B<x,y>)^2 + B^2(|x|^2|y|^2 - <x,y>^2), without materialising -x. Negating x flips the sign of <x,y> and leaves the Gram term alone.

### `eshkol_rm_mobius_add`

*Function* — line 1104

```c
static void eshkol_rm_mobius_add(const double* x, const double* y, double B,
 int n, double* out) { ... }
```

Mobius addition on the ball of parameter `B` > 0: x (+)_B y = ((1 + 2B<x,y> + B|y|^2) x + (1 - B|x|^2) y) / (1 + 2B<x,y> + B^2 |x|^2 |y|^2) At B = 0 this is x + y, which is why the Euclidean branch of every caller is the same code with B = 0 rather than a separate special case. THE DENOMINATOR IS NOT FLOORED. For two points strictly inside the ball, writing p = sqrt(B)|x| < 1 and q = sqrt(B)|y| < 1, the denominator is 1 + 2pq cos(theta) + p^2 q^2 >= (1 - pq)^2 > 0, so it cannot vanish and there is nothing to guard. It used to be floored at 1e-15 without scaling the numerator, which for x = 0.999999999, y = -0.999999998 at B = 1 turned an exact quotient of 1/3 into 0.003: a valid interior sum, off by two orders of magnitude, with no diagnostic. Callers that could hand this arbitrary vectors -- parallel transport was the only one -- no longer do; transport uses the linear gyration form below.

### `eshkol_rm_check_point`

*Function* — line 1123

```c
static const char* eshkol_rm_check_point(const double* x, double K, int n);
```

Mobius addition on the chart of curvature `K` (every sign), with the chart's domain checks: x (+)_B y, B = -K alpha^2/4. A sum that is the pole of the positive chart, or that leaves the ball, is refused.

### `eshkol_rm_gyration`

*Function* — line 1169

```c
static void eshkol_rm_gyration(const double* a, const double* b, const double* w,
 double B, int n, double* out) { ... }
```

The gyration gyr[a,b]w on the ball of parameter `B,` in CLOSED LINEAR FORM: D = 1 + 2B<a,b> + B^2 |a|^2 |b|^2 A = -B^2 <a,w> |b|^2 + B <b,w> + 2 B^2 <a,b><b,w> C = -B^2 <b,w> |a|^2 - B <a,w> gyr[a,b]w = w + 2 (A a + C b) / D WHY NOT THE MOBIUS COMPOSITION. gyr[a,b]w is also -(a (+) b) (+) (a (+) (b (+) w)), and that is how transport used to compute it -- by feeding the TANGENT VECTOR w into Mobius POINT addition. A tangent vector is not constrained to the ball, so an intermediate denominator can vanish even though the gyration is a globally defined linear isometry. Counterexample at B = 1, x = y = (0.5), v = (2): every gyration in one dimension is the identity, so transport from a point to itself must return (2); the intermediate (-0.5) (+) 2 has denominator zero, the old floor turned its exact 0/0 cancellation into zero, and the routine returned (0.5). The form above is linear in w and its only denominator D is the Mobius denominator of the two POINTS a and b, which are validated ball points, so it is bounded below by (1 - sqrt(B)|a| sqrt(B)|b|)^2 > 0.

### `eshkol_rm_check_point`

*Function* — line 1193

```c
static const char* eshkol_rm_check_point(const double* x, double K, int n) { ... }
```

Check that `x` is a legal point of the chart of curvature `K.` K < 0: strictly inside the ball of radius 2/sqrt(-K) (alpha = 1). K = 0: any finite point. K > 0: any finite point whose conformal denominator 1 + K|x|^2/4 is representable; a point where it overflows is numerically the POLE of the chart, which is not a point of the chart.

**Returns**

NULL when it is, else a reason naming what is wrong.

### `eshkol_rm_require_interior`

*Function* — line 1221

```c
static const char* eshkol_rm_require_interior(const double* out, double K, int n) { ... }
```

Check that a computed chart point is a point of the chart. tanh rounds to exactly 1.0 for arguments beyond about 19, so the exponential map and Mobius scalar multiplication can land on the ball boundary, which is not a point of the open ball; on the positive chart a geodesic that reaches the pole produces a non-representable conformal denominator. Refusing names the representability limit.

**Returns**

NULL when the point is in the chart, else a reason.

### `eshkol_rm_check_tangent`

*Function* — line 1236

```c
static const char* eshkol_rm_check_tangent(const double* x, const double* v,
 double K, int n) { ... }
```

Check a tangent vector of the chart: every finite vector is tangent.

**Returns**

NULL when every component is finite, else a reason.

### `eshkol_rm_sphere_check_point`

*Function* — line 1250

```c
static const char* eshkol_rm_sphere_check_point(const double* x, double K, int n) { ... }
```

Check that `x` lies ON the embedded sphere of radius 1/sqrt(K) in ambient coordinates (the model behind the `eshkol_rm_sphere_*` ops).

**Returns**

NULL when it does, else a reason.

### `eshkol_rm_sphere_check_tangent`

*Function* — line 1269

```c
static const char* eshkol_rm_sphere_check_tangent(const double* x,
 const double* v,
 double K, int n) { ... }
```

Check that `v` is tangent to the embedded sphere of curvature `K` at `x:` <x,v> must vanish, tested relative to |x| and |v|.

**Returns**

NULL when tangent, else a reason.

### `eshkol_rm_distance`

*Function* — line 1340

```c
static const char* eshkol_rm_distance(const double* x, const double* y, double K,
 int n, double* out) { ... }
```

Geodesic distance between `x` and `y` on the chart of curvature

**Returns**

NULL on success, else a reason.

### `eshkol_rm_sq_distance`

*Function* — line 1394

```c
static const char* eshkol_rm_sq_distance(const double* x, const double* y,
 double K, int n, double* out) { ... }
```

Squared geodesic distance d(x,y)^2 on the chart of curvature `K.` The value only. Its gradient is never formed from d (see eshkol_rm_sq_distance_grad): d^2 is smooth at coincidence while d is not.

### `eshkol_rm_exp_map`

*Function* — line 1417

```c
static const char* eshkol_rm_exp_map(const double* x, const double* v, double K,
 int n, double* out, double* scratch) { ... }
```

Exponential map exp_x(v) on the chart of curvature `K.` exp_x(v) = x (+)_B [ (lambda_x/alpha) tan_ratio(s) v ], s = B (lambda_x |v| / alpha)^2, for every sign: tanh on the ball, tan on the sphere chart, the series near s = 0, and x + v at K = 0. On the positive chart a geodesic that reaches the pole (sqrt(-s) = pi/2) is refused; past it, the chart formula continues onto the far hemisphere.

**Parameters**

- `scratch` — n doubles.

**Returns**

NULL on success, else a reason.

### `eshkol_rm_log_direction`

*Function* — line 1472

```c
static const char* eshkol_rm_log_direction(const double* x, const double* y,
 double K, int n, double* u,
 double* scratch, double* d_out) { ... }
```

The unit direction `u` of log_x(y) on the chart of curvature

**Parameters**

- `scratch` — n doubles.

### `eshkol_rm_log_map`

*Function* — line 1529

```c
static const char* eshkol_rm_log_map(const double* x, const double* y, double K,
 int n, double* out, double* scratch) { ... }
```

Logarithmic map log_x(y) on the chart of curvature `K,` the inverse of eshkol_rm_exp_map(). The DIRECTION comes from the Mobius numerator of (-x) (+)_B y and the MAGNITUDE from the geodesic distance, |log_x(y)| = d(x,y)/lambda_x: - the direction is taken from the numerator vector alone, never divided by the Mobius denominator, which is the quantity that goes to zero for two nearly-antipodal points; - the magnitude is d/lambda_x from eshkol_rm_distance's stable form, never (2/(sqrt(B) lambda_x)) artanh(sqrt(B)|u|), which loses the answer when sqrt(B)|u| rounds to 1 for a valid interior pair. On the positive chart the antipode (Mobius denominator zero) is the cut locus and is refused.

**Parameters**

- `scratch` — n doubles.

**Returns**

NULL on success, else a reason.

### `eshkol_rm_sq_distance_grad`

*Function* — line 1575

```c
static const char* eshkol_rm_sq_distance_grad(const double* x, const double* y,
 double K, int n, double* gx,
 double* gy, double* scratch) { ... }
```

Coordinate gradients of the SQUARED distance:

**Parameters**

- `scratch` — 2n doubles.

### `eshkol_rm_transport`

*Function* — line 1645

```c
static const char* eshkol_rm_transport(const double* x, const double* y,
 const double* v, double K, int n,
 double* out, double* scratch) { ... }
```

Parallel transport of tangent vector `v` from `x` to `y` along the connecting geodesic of the chart of curvature `K.` P_{x->y}(v) = (lambda_x / lambda_y) gyr[y, -x] v, with the gyration in its LINEAR closed form (eshkol_rm_gyration), whose only denominator is the Mobius denominator of the two points. On the positive chart that denominator vanishes at the antipode, where the geodesic and hence the transport is not unique, and the op refuses.

**Parameters**

- `scratch` — n doubles.

**Returns**

NULL on success, else a reason.

### `eshkol_rm_mobius_scalar`

*Function* — line 1692

```c
static const char* eshkol_rm_mobius_scalar(double r, const double* x, double K,
 int n, double* out) { ... }
```

Mobius scalar multiplication r (x)_B x = tan_B(r artan_B(|x|)) x/|x| on the chart of curvature `K,` every sign: a = r |x| artan_ratio(B|x|^2) (r artan_B |x|, length units) out = r artan_ratio(B|x|^2) tan_ratio(B a^2) x r x at K = 0. Refuses when the result is not a point of the chart (the ball boundary for K < 0, the pole for K > 0).

### `eshkol_rm_project`

*Function* — line 1733

```c
static const char* eshkol_rm_project(const double* x, double K, int n, double* out) { ... }
```

Project `x` onto the chart of curvature `K.` K < 0: rescale onto the open ball when `x` is on or outside it, leaving interior points untouched. K >= 0: every finite point is a chart point, so a copy (a positive-chart point at the pole is refused).

**Returns**

NULL on success, else a reason.

### `eshkol_rm_egrad_to_rgrad`

*Function* — line 1768

```c
static const char* eshkol_rm_egrad_to_rgrad(const double* g, const double* x,
 double K, int n, double* out) { ... }
```

Convert a Euclidean (coordinate) gradient `g` at `x` into the Riemannian gradient g / lambda_x^2, every sign of K. At K = 0 it is g / alpha^2 = g.

**Returns**

NULL on success, else a reason.

### `eshkol_rm_sphere_distance`

*Function* — line 1789

```c
static const char* eshkol_rm_sphere_distance(const double* x, const double* y,
 double K, int n, double* out) { ... }
```

Geodesic distance R theta on the embedded sphere, theta from the cancellation-free atan2 form; the antipode is refused.

### `eshkol_rm_sphere_exp_map`

*Function* — line 1805

```c
static const char* eshkol_rm_sphere_exp_map(const double* x, const double* v,
 double K, int n, double* out) { ... }
```

exp_x(v) = cos(|v|/R) x + R sin(|v|/R) v/|v| on the embedded sphere, with the tangency of v checked rather than assumed.

### `eshkol_rm_sphere_log_map`

*Function* — line 1826

```c
static const char* eshkol_rm_sphere_log_map(const double* x, const double* y,
 double K, int n, double* out) { ... }
```

log_x(y) on the embedded sphere; the antipode is refused.

### `eshkol_rm_sphere_project`

*Function* — line 1847

```c
static const char* eshkol_rm_sphere_project(const double* x, double K, int n,
 double* out) { ... }
```

Rescale `x` onto the embedded sphere of radius 1/sqrt(K); the origin has no projection and is refused.

### `eshkol_rm_sphere_to_chart`

*Function* — line 1889

```c
static const char* eshkol_rm_sphere_to_chart(const double* p, double K, int n,
 double* x) { ... }
```

The isometry from the embedded sphere of radius R = 1/sqrt(K) in R^{n+1} (`p,` n+1 coordinates, last one the polar axis) to the alpha = 1 chart (`x,` n coordinates): x = 2 R p' / (R + p_n), the stereographic projection from the pole p = -R e_n. Where p_n < 0 the denominator is formed as |p'|^2/(R - p_n) (the same number on the sphere) so the far hemisphere keeps its relative precision. The pole is refused.

### `eshkol_rm_chart_to_sphere`

*Function* — line 1916

```c
static const char* eshkol_rm_chart_to_sphere(const double* x, double K, int n,
 double* p) { ... }
```

The inverse isometry, chart -> embedded sphere: p' = lambda_x x, p_n = R (2 lambda_x - 1) = R (1 - K|x|^2/4) lambda_x, lambda_x = 1/(1 + K|x|^2/4).

### `eshkol_rm_distance_dK`

*Function* — line 2068

```c
static const char* eshkol_rm_distance_dK(const double* x, const double* y,
 double K, int n, double* d_out,
 double* d1_out, double* d2_out) { ... }
```

Geodesic distance between `x` and `y` AND its first two derivatives with respect to the sectional curvature K, at fixed chart points: the curvature JET (value, d/dK, d^2/dK^2) of ADR-0018. Exact closed-form derivatives, not difference quotients. One formula for every sign: with B = -K alpha^2/4 (dB/dK = -alpha^2/4), A = B|x|^2, C = B|y|^2, w = B|x-y|^2 / ((1-A)(1-C)) and D = B d/dB, L = D log w = 1 + A/(1-A) + C/(1-C) h = sqrt|w| psi(w) (asinh on the ball, asin on the sphere chart) d = alpha |B|^(-1/2) h Dh = q L, D^2 h = q (DL + L^2/(2(1+w))), q = sqrt|w| / (2 sqrt(1+w)), evaluated in the cancellation-free t1/t2 form below. For |w| <= ESHKOL_RM_Q_SERIES the degree-10 series is differentiated termwise instead (eshkol_rm_distance_dK_small_q), which is what carries the jet THROUGH K = 0: at B = 0 the series is evaluated at w = 0 and the left and right jets are the same expression. Coincident points give an identically zero d, so every derivative is exactly zero. A non-finite requested output is refused (for example d/dK at the subnormal negative-curvature floor with radius-scaled points, whose analytic value is beyond DBL_MAX); there is no blanket binade refusal.

**Parameters**

- `d_out` — the distance itself (may be NULL).
- `d1_out` — d(distance)/dK.
- `d2_out` — d^2(distance)/dK^2.

**Returns**

NULL on success, else a reason.

### `eshkol_rm_sq_distance_dK`

*Function* — line 2189

```c
static const char* eshkol_rm_sq_distance_dK(const double* x, const double* y,
 double K, int n, double* s_out,
 double* s1_out, double* s2_out) { ... }
```

The curvature jet of the SQUARED distance, from the distance jet: (d^2)' = 2 d d', (d^2)'' = 2 (d'^2 + d d'').

### `eshkol_rm_default_temperature`

*Function* — line 2217

```c
static double eshkol_rm_default_temperature(int head_dim) { ... }
```

The default attention temperature sqrt(head_dim).

### `eshkol_rm_attention_row`

*Function* — line 2230

```c
static const char* eshkol_rm_attention_row(const double* q, const double* keys,
 long key_stride, int limit, int dim,
 double K, double temperature,
 double* sqd, double* weights) { ... }
```

One attention row: weights_j = softmax_j(-d^2(q, k_j)/(2t)) over the first `limit` keys (the causal mask is the caller's limit).

**Parameters**

- `keys` — first key row; key j starts at keys + j*key_stride
- `sqd` — limit doubles: receives d^2(q, k_j)
- `weights` — limit doubles: receives the normalised weights

**Returns**

NULL on success, else a reason (a refused pair names the domain).

### `eshkol_geometry_header_t`

*Struct* — line 2262

```c
typedef struct {
 uint32_t convention;
 uint32_t reserved;
 double alpha;
 double curvature;
} eshkol_geometry_header_t;
```

The header every serialized geometry record carries.

### `eshkol_rm_geometry_header`

*Function* — line 2270

```c
static void eshkol_rm_geometry_header(double K, eshkol_geometry_header_t* h) { ... }
```

Fill `h` for the current convention at curvature `K.`

### `eshkol_rm_check_geometry_header`

*Function* — line 2282

```c
static const char* eshkol_rm_check_geometry_header(const eshkol_geometry_header_t* h) { ... }
```

Accept a serialized header only when it names the current convention and gauge. An unversioned or legacy record is refused, never guessed: legacy data goes through the explicit migration helpers below.

### `eshkol_rm_migrate_legacy_point`

*Function* — line 2305

```c
static const char* eshkol_rm_migrate_legacy_point(const double* x_v1, double K,
 int n, double* x_v2) { ... }
```

Migrate a legacy ball point (alpha = 2, radius 1/sqrt(-K)) to the current chart: x_v2 = 2 x_v1. K = 0 legacy coordinates were already canonical Euclidean and are copied. Positive legacy data was ambient (use eshkol_rm_sphere_to_chart).

### `eshkol_rm_migrate_legacy_tangent`

*Function* — line 2327

```c
static const char* eshkol_rm_migrate_legacy_tangent(const double* v_v1, double K,
 int n, double* v_v2) { ... }
```

Migrate a legacy tangent vector (or first-moment optimizer state) by the differential of x_v2 = 2 x_v1. Squared Riemannian norms (second-moment state) are invariant and are not touched.

### `eshkol_rm_migrate_legacy_temperature`

*Function* — line 2351

```c
static const char* eshkol_rm_migrate_legacy_temperature(double legacy_scale,
 double reference_distance,
 double* temperature) { ... }
```

Migrate a legacy attention score scale s (logit = -s d) to a squared-distance temperature t (logit = -d^2/(2t)). No temperature reproduces a linear-distance score exactly. The migration matches the two logits to first order at the caller's reference distance d_ref: softmax rows are invariant to a constant offset, and the slopes agree when t = d_ref / s. Callers record d_ref (for example the median attention distance of the checkpoint) next to the migrated temperature.

## Undocumented

| Symbol | Kind | Line |
|---|---|---:|
| `ESHKOL_RM_ALPHA` | Macro | 123 |
| `ESHKOL_GEOMETRY_CONVENTION_UNVERSIONED` | Macro | 128 |
| `ESHKOL_GEOMETRY_CONVENTION_LEGACY_BALL_V1` | Macro | 129 |
| `ESHKOL_GEOMETRY_CONVENTION_STEREOGRAPHIC_V2_ALPHA1` | Macro | 130 |
| `ESHKOL_RM_SPHERE_TOL` | Macro | 136 |
| `ESHKOL_RM_TANGENT_TOL` | Macro | 141 |
| `ESHKOL_RM_Q_SERIES` | Macro | 149 |
| `eshkol_rm_two_sum` | Function | 151 |
| `eshkol_rm_two_product` | Function | 152 |
| `eshkol_rm_dd_add` | Function | 154 |
| `eshkol_rm_dot` | Function | 158 |
| `eshkol_rm_two_sum` | Function | 172 |
| `eshkol_rm_two_product` | Function | 181 |
| `eshkol_rm_fma` | Function | 193 |
| `eshkol_rm_dd_add` | Function | 201 |
| `eshkol_rm_dd_mul` | Function | 222 |
| `eshkol_rm_norm` | Function | 240 |
| `eshkol_rm_scaled_product4` | Function | 266 |
| `eshkol_rm_sqrt_nonnegative` | Function | 271 |
| `eshkol_rm_scaled_norm2_times` | Function | 282 |
| `eshkol_rm_scaled_product4` | Function | 307 |
| `eshkol_rm_scaled_product_n` | Function | 323 |
| `eshkol_rm_scaled_product4_dd` | Function | 342 |
| `eshkol_rm_scaled_square` | Function | 360 |
| `eshkol_rm_difference_norm2_times2` | Function | 368 |
| `eshkol_rm_difference_norm2_times` | Function | 397 |
| `eshkol_rm_scaled_dot_factor` | Function | 405 |
| `eshkol_rm_points_equal` | Function | 430 |
| `eshkol_rm_dyadic_t` | Struct | 443 |
| `eshkol_rm_dyadic_from_double` | Function | 448 |
| `eshkol_rm_all_finite` | Function | 474 |
| `eshkol_rm_gcd_u64` | Function | 481 |
| `eshkol_rm_cross_ratio_equal` | Function | 494 |
| `eshkol_rm_sphere_antipodal` | Function | 523 |
| `eshkol_rm_sphere_log_map_stable` | Function | 547 |
| `eshkol_rm_sphere_angle` | Function | 594 |
| `eshkol_rm_check_output` | Function | 789 |
| `eshkol_rm_needs_rescale` | Function | 852 |
| `eshkol_rm_halved` | Function | 866 |
| `eshkol_rm_require_interior` | Function | 1124 |
| `eshkol_rm_check_output` | Function | 1125 |
| `eshkol_rm_chart_mobius_add` | Function | 1127 |
| `eshkol_rm_inverse_power_mul` | Function | 1303 |
| `eshkol_rm_distance_dK_small_q` | Function | 1943 |
| `eshkol_rm_directional` | Struct | 2377 |
| `eshkol_rm_directional::value` | Variable | 2378 |
| `eshkol_rm_directional::tangent` | Variable | 2379 |
| `eshkol_rm_dadd` | Function | 2382 |
| `eshkol_rm_dsub` | Function | 2387 |
| `eshkol_rm_dneg` | Function | 2392 |
| `eshkol_rm_dmul` | Function | 2396 |
| `eshkol_rm_ddiv` | Function | 2402 |
| `eshkol_rm_dsqrt` | Function | 2408 |
| `eshkol_rm_dtan_ratio` | Function | 2416 |
| `eshkol_rm_dnorm` | Function | 2422 |
| `eshkol_rm_dscaled_norm2_times` | Function | 2429 |
| `eshkol_rm_done_minus_bnorm2` | Function | 2435 |
| `eshkol_rm_done_plus_dot` | Function | 2441 |
| `eshkol_rm_done_minus_dot` | Function | 2449 |
| `eshkol_rm_dmobius_den` | Function | 2457 |
| `eshkol_rm_daxpby_exact` | Function | 2482 |
| `eshkol_rm_dmobius_add` | Function | 2497 |
| `eshkol_rm_distance_directional` | Function | 2519 |
| `eshkol_rm_exp_directional` | Function | 2574 |
| `eshkol_rm_log_directional` | Function | 2617 |
