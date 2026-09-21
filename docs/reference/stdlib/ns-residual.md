# `core.pde.ns-residual` — a residual oracle for incompressible Navier-Stokes

**Source**: [`lib/core/pde/ns-residual.esk`](../../../lib/core/pde/ns-residual.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually
`(require core.pde.ns-residual)`

Turns a candidate flow into its residual force — how far it is from solving
the incompressible Navier-Stokes equations
`R = d_t u + (u . grad) u - nu Lap u + grad p`, `div u = 0` — and, for
similarity-coordinate constructions, its singular structure near a chosen
time. Every derivative is an AD partial of the flow's own procedures, never
a hand-differentiated formula; this lifts and generalizes the residual
operator built ad hoc in
[`examples/mathematics_navier_stokes_viscosity_scaling.esk`](../../../examples/mathematics_navier_stokes_viscosity_scaling.esk)
and
[`examples/mathematics_navier_stokes_similarity_scales.esk`](../../../examples/mathematics_navier_stokes_similarity_scales.esk).

## Representation

A flow is **not** a record type — like `core.exact_linalg`'s matrices, it is
a plain vector built with the `vector` procedure (never a `#(...)` literal,
which the reader auto-promotes to an f64 tensor):

```
cylindrical (axisymmetric) flow = (ns-flow u_r u_theta u_z p nu)
Cartesian flow                  = (ns-cflow u p nu)
```

`u_r`/`u_theta`/`u_z`/`p` are procedures `(lambda (r z t) ...)`; the
Cartesian `u` is `(lambda (x y z t) ...)` returning a 3-vector
`(vector ux uy uz)`, its `p` is `(lambda (x y z t) ...)`; `nu` is the
(scalar, possibly exact-rational) kinematic viscosity. "Axisymmetric" means
none of the cylindrical procedures depend on the azimuthal angle theta —
that is what licenses the closed-form cylindrical Laplacian/divergence
below (no d/d-theta terms). Build and read flows only through the
accessors; never index the underlying vector from outside the module.

| Constructor | Accessors |
|---|---|
| `(ns-flow u_r u_theta u_z p nu)` | `ns-flow-ur` `ns-flow-utheta` `ns-flow-uz` `ns-flow-p` `ns-flow-nu` |
| `(ns-cflow u p nu)` | `ns-cflow-u` `ns-cflow-p` `ns-cflow-nu` |

## Exactness contract

Every residual/divergence formula is built only from `+ - * / expt` and the
AD operators — it introduces no floating-point literal of its own. So if a
flow's procedures are built entirely from exact-preserving operations
(`+ - * /`, non-negative integer `expt`) and evaluated at an exact rational
point, the residual comes back exact — the same R7RS contagion contract as
`core.exact_linalg`. `ns-similarity-field`'s A/D/h exponents are commonly
non-integer rationals (e.g. `1/2 + h`); `expt` at a non-integer exponent is
**not** exact-preserving even on an exact rational base (no closed exact
rational nth root in general), so fields built by `ns-similarity-field` are
exact only up to that unavoidable point.

## AD-nesting limit (read before raising `order` above 1)

`ns-residual` already performs an order-2 `derivative-n` pass internally
for the Laplacian. Wrapping *that* in a second order-2-or-higher
differentiation pass — `taylor`/`derivative-n f x k` with `k >= 2` as an
outer pass over a call to `ns-residual` — fails outright:

```
ERROR: unsupported nested differentiation: an order-2 `derivative-n`/
`taylor` pass inside another differentiation of order 2 or higher.
```

even on the nested-AD-and-exact-rational-seed fix this module is built on
and otherwise relies on throughout. Nor does repeating the outer pass as
several separately-nested order-1 calls recover it: that specific shape
returns `0` silently instead of raising an error — worse than the loud
failure. A **single** order-1 outer pass directly around the
order-2-internal residual body, however, is exact and verified correct.
Minimal repros for all three cases live under `.scratch/` in the branch
that added this module (not committed to the library — a compiler
front-end limitation to route around, not a defect in this file).
Consequently `ns-residual-tau-series` and `ns-force-smoothness-probe`
accept `order` **0 or 1 only** and raise `(error ...)` for anything higher.
`ns-residual`/`ns-residual-cartesian`/`ns-divergence`/`ns-divergence-cartesian`
themselves are unaffected — they perform exactly one level of AD internally
and support arbitrary flows.

## Engine support

`ns-flow`/`ns-cflow` and their accessors, `ns-simpson-nodes`/`-weights`/
`-step`, `ns-grid-integral` and `ns-energy` on an AD-free field are
supported on all three engines (native JIT, AOT, and the bytecode VM) --
see `tests/vm_parity/corpus/78_ns_residual.esk`. Every other function here that touches `ns-residual`/`ns-divergence`
internally (`ns-residual`, `ns-divergence`, the Cartesian variants,
`ns-dissipation`, `ns-residual-tau-series`, `ns-singular-orders`,
`ns-force-smoothness-probe`) is **native-only** (JIT and AOT, not the VM):
every one of them calls `derivative-n` internally, and `docs/VM_PARITY.md`
already documents "higher-order
nesting (gradient-of-derivative / Taylor tower, `op:DERIVATIVE_N`) stays
native-only" as a pre-existing, deliberate VM engine boundary. `derivative`
(order 1) is VM-supported, but that alone cannot close the gap: the
Laplacian is unavoidably an order-2 derivative, and the VM has no order-2
opcode. This module inherits that boundary rather than working around it.
(`ns-similarity-field` itself builds a flow using only `expt`/`+ - * /`, no
AD -- it is the *residual/divergence evaluation* of the flow it returns
that is native-only, not the construction.)

## Cylindrical residual and divergence

Axisymmetric; curvature enters through the r-only terms:

```
R_r     = d_t u_r + u_r d_r u_r + u_z d_z u_r - u_theta^2/r + d_r p
          - nu (d_rr u_r + d_r u_r / r - u_r / r^2 + d_zz u_r)
R_theta = d_t u_theta + u_r d_r u_theta + u_z d_z u_theta + u_r u_theta / r
          - nu (d_rr u_theta + d_r u_theta / r - u_theta / r^2 + d_zz u_theta)
R_z     = d_t u_z + u_r d_r u_z + u_z d_z u_z + d_z p
          - nu (d_rr u_z + d_r u_z / r + d_zz u_z)

div u   = d_r u_r + u_r/r + d_z u_z
```

(`R_theta` and `R_z` match the `R-theta`/`R-zed` formulas already verified
in `examples/mathematics_navier_stokes_similarity_scales.esk`; `R_r` and
`div u` are new here. The `(1/r) d_theta u_theta` term of the general
cylindrical divergence formula vanishes under axisymmetry.)

### `(ns-residual flow)`
`(vector Rr Rtheta Rz)`, each `(lambda (r z t) ...)`.

### `(ns-divergence flow)`
`(lambda (r z t) ...)` computing `div u`.

```scheme
(require stdlib)
;; Rigid rotation: u_theta = Omega r, p = (1/2) Omega^2 r^2 -- an exact
;; steady solution at any viscosity (no strain, so the viscous term
;; vanishes identically).
(define Omega 1/3)
(define flow (ns-flow (lambda (r z t) 0) (lambda (r z t) (* Omega r))
                       (lambda (r z t) 0) (lambda (r z t) (* 1/2 Omega Omega r r))
                       1/5))
(define R (ns-residual flow))
(display ((vector-ref R 0) 2 1/3 1/2)) (newline) ; 0, exact
(display (exact? ((vector-ref R 0) 2 1/3 1/2))) (newline) ; #t
```

## Cartesian residual and divergence

No curvature terms; every derivative is a straightforward partial of the
3-vector-valued `u`.

### `(ns-residual-cartesian cflow)`
`(lambda (x y z t) ...)` returning `(vector Rx Ry Rz)`.

### `(ns-divergence-cartesian cflow)`
`(lambda (x y z t) ...)`.

The two representations agree exactly wherever both make sense: a
cylindrical residual `(Rr, Rtheta, Rz)` rotates to Cartesian via
`Rx = Rr (x/r) - Rtheta (y/r)`, `Ry = Rr (y/r) + Rtheta (x/r)`, at any
`r = sqrt(x^2+y^2) > 0`. At a Pythagorean rational point (e.g.
`(x,y) = (3/5,4/5)`, so `r = 1` exactly) that agreement is checkable with
`=`, not a tolerance — see the "shear field" fixture in
[`tests/stdlib/ns_residual_test.esk`](../../../tests/stdlib/ns_residual_test.esk).

## Rational Simpson quadrature — energy and dissipation

Nodes/weights/step are built from exact `+ - * /` only, so an
all-exact-rational domain produces an exact-rational result. Simpson's rule
is exact for any integrand that is a polynomial of degree `<= 3` in each
axis *separately* (tensor-product exactness, not joint degree-3
exactness); the moment the flow, the domain bounds, or the field itself is
inexact, contagion demotes the whole sum to an ordinary floating-point
quadrature automatically — that demotion **is** the inexact fallback,
there is no separate code path.

### `(ns-simpson-nodes a b n)` / `(ns-simpson-weights n)` / `(ns-simpson-step a b n)`
The `n+1` nodes, the `n+1` unscaled weights `(1 4 2 4 ... 4 1)`, and the
step `(b-a)/n`, for a Simpson rule on `[a,b]` with `n` even subintervals.

### `(ns-grid-integral integrand xa xb xn ya yb yn za zb zn t)`
The Simpson triple integral of `(lambda (x y z t) ...)` over
`[xa,xb] x [ya,yb] x [za,zb]` (each of `xn`/`yn`/`zn` even), at fixed `t`.

### `(ns-energy cflow xa xb xn ya yb yn za zb zn t)`
`integral of |u|^2` over the box.

### `(ns-dissipation cflow xa xb xn ya yb yn za zb zn t)`
`nu * integral of |grad u|^2` over the box — the sum of squared component
gradients (`|grad ux|^2 + |grad uy|^2 + |grad uz|^2`), the quantity behind
the dissipation identity in `mathematics_navier_stokes_viscosity_scaling.esk`.
Not the full symmetric strain-rate contraction `2 e_ij e_ij`.

```scheme
(require stdlib)
(define cflow (ns-cflow (lambda (x y z t) (vector x 0 0)) (lambda (x y z t) 0) 1/6))
;; speed^2 = x^2 over the unit cube, Simpson n=2 (exact for degree <= 3):
(display (ns-energy cflow 0 1 2 0 1 2 0 1 2 0)) (newline)       ; 1/3, exact
(display (ns-dissipation cflow 0 1 2 0 1 2 0 1 2 0)) (newline)  ; 1/6, exact
```

## Similarity coordinates

`tau = 1 - t`, `A = 1/2 + h`, `D = 1/2 - h`; `q(z,tau)` solves the implicit
relation `q - z^2 q^{2h} = tau` by fixed-point iteration, then
`X = r^2/(2q)`, `eta = z q^{-D}`, and

```
u_theta = q^{-A} E(X,eta),   u_z = q^{-A} U(X,eta),
u_r = V(X,eta) / r,          p  = q^{-2A} Pi(X,eta).
```

### `(ns-similarity-field E U V Pi A D h nu)`
Builds a cylindrical flow (an `ns-flow`) from profile procedures
`E`/`U`/`V`/`Pi` of `(X, eta)` and exponents `A`/`D`/`h`. Unlike
`examples/mathematics_navier_stokes_similarity_scales.esk`, `V` is taken as
an **independent given profile**, not derived from `U` via the
incompressibility identity — that identity is exactly what `ns-divergence`
is for checking on the result, so a caller's `(E,U,V,Pi)` choice and its
divergence-freeness are decoupled. `nu` is not part of the similarity
ansatz itself but is required by the flow representation contract so the
result can be fed straight to `ns-residual`. `A`/`D`/`h` may be exact
rationals, but `expt` at that non-integer exponent will still make the
resulting field inexact (see "Exactness contract" above) — unavoidable, not
a defect in this constructor.

## The tau-series — the object a search compares against zero

### `(ns-residual-tau-series flow r z order)`
`(vector cs-r cs-theta cs-z)`, each a list of the Taylor coefficients
`c[0..order]` (`order` 0 or 1 — see the AD-nesting limit above) of the
corresponding residual component as a function of `tau = 1 - t`, expanded
about `tau = 0` (i.e. about `t = 1`), at the fixed point `(r,z)`. `c[0]` is
exact whenever `flow`'s procedures and `(r,z)` are; `c[1]` goes through one
AD pass and inherits its exactness contract.

### `(ns-singular-orders flow r z order)`
`(vector kr ktheta kz)`, the lowest tau-exponent at which each residual
component is nonzero at `(r,z)` — `0`, `1`, or `#f` (vanishes to at least
first order, i.e. every probed coefficient is zero). This is the
mechanical signal an ansatz search uses to reject a candidate: order `0`
means the candidate does not even instantaneously solve Navier-Stokes at
that point.

### `(ns-force-smoothness-probe flow force r z order)`
`force` is a `(vector fr ftheta fz)` of procedures `(lambda (r z t) ...)`,
a proposed external force/cutoff. Returns `#t` iff every Taylor
coefficient up to `order` (0 or 1) of every component of the forced
residual `(R - force)` stays finite (no NaN, no +-infinity) at the fixed
positive radius `r`. The finite-difference-free AD analogue of a
smoothness check: no stepping, no truncation error, just the tower — to
the order the tower supports on this compiler (see the AD-nesting limit).

```scheme
(require stdlib)
(define flow (ns-similarity-field E U V Pi A D h nu))    ; caller-supplied profiles
(define orders (ns-singular-orders flow 1.2 0.3 1))
(display orders) (newline)   ; e.g. #(0 #f 0) -- some components fail at tau=0
```

## Known defects worked around here (not fixed by this module)

- **`derivative` at an exact rational seed** returns 0 when the
  differentiand closes over a non-integer exact-rational captured
  argument. Every single-variable partial in this module goes through
  `derivative-n`, never bare `derivative` — see `nsr-d1`/`nsr-d2` in the
  source.
- **`derivative-n` on a seed-independent differentiand** can come back as
  an inexact zero even at an exact rational seed. `nsr-dn` snaps that zero
  back to exact.
- **Nested AD beyond one order-2 level** (this module's own discovery —
  see "AD-nesting limit" above): `ns-residual-tau-series` and
  `ns-force-smoothness-probe` cap `order` at 1 rather than risk it.
- `(expt 1/3 50)` returning `0` (repeated exact multiplication) and
  `#(...)`-literal auto-promotion to f64 tensors are general Eshkol
  pitfalls, not specific to this module — see
  `docs/breakdown/EXACT_ARITHMETIC.md` and `core.exact_linalg`'s own
  documentation of the latter.
