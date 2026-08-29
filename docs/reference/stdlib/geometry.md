# Geometric builtins — Riemannian manifolds, Lie groups, forms and geodesic attention

**Name table**: [`lib/backend/eshkol_vm.c`](../../../lib/backend/eshkol_vm.c) `BUILTINS[]`, native ids 804-861.
**Implementation**: [`lib/backend/vm_geometric.c`](../../../lib/backend/vm_geometric.c).
**Constant-curvature core**: [`inc/eshkol/backend/riemannian_core.h`](../../../inc/eshkol/backend/riemannian_core.h).
**Differential-form core**: [`inc/eshkol/backend/differential_form_core.h`](../../../inc/eshkol/backend/differential_form_core.h).
**Weighted Fréchet mean core**: [`inc/eshkol/backend/frechet_mean_core.h`](../../../inc/eshkol/backend/frechet_mean_core.h).
**Surface record**: `tests/coverage/language_surface.json`, category `geometry`.
**Regressions**: [`tests/vm/geometric_surface_regression.esk`](../../../tests/vm/geometric_surface_regression.esk) (arity/resolution) and [`tests/vm/geometric_riemannian_surface_regression.esk`](../../../tests/vm/geometric_riemannian_surface_regression.esk) (numeric, every assertion chosen so the pre-`SW-73` answer fails it), both run by `scripts/run_vm_surface_tests.sh`.

This page documents the **62 geometric builtin names** the bytecode VM registers
under native ids 804-861. They are a different surface from the pure-Scheme
[`core.manifold`](manifold.md) module, which shares six of their spellings — see
[Name collisions with `core.manifold`](#name-collisions-with-coremanifold) before
using either.

## Provenance of this page

Every signature, arity, argument order, return type and closed form below is read
directly from `lib/backend/eshkol_vm.c` (the name/id/arity table) and
`lib/backend/vm_geometric.c` (the dispatch bodies), at the revision this page was
written against. Unlike the other pages in this directory, **the examples carry no
stamped output**: these ops execute only on the bytecode VM, and no output was
captured for this revision. What is asserted instead is each op's *closed form*,
which is a source fact and can be checked by reading the cited case label. The call
shapes are the ones the committed regression exercises, so they are known to resolve
and to be stack-balanced.

## Engine availability

| Engine | Available |
|---|---|
| Bytecode VM (`--profile hosted-vm`, `.eskb` modules, the WASM playground's VM) | Yes — all 62 names |
| Native AOT (`eshkol-run file.esk`) | **No** |
| Native REPL/JIT (`eshkol-run -r`) | **No** |

These names appear in the VM builtin table only. They are absent from the native
LLVM dispatch (`lib/backend/llvm_codegen.cpp`) and from the native builtin closure
table (`lib/backend/eshkol_compiler.c`), so calling one from a natively compiled
program is an unknown-function error, not a slow path. `tests/coverage/language_surface.json`
records `"backends": ["vm"]` for every one of them.

To run a program that uses them, compile it to a bytecode module and execute that
module on the VM:

```sh
eshkol-run --profile hosted-vm --emit-eskb geo.eskb geo.esk
# then execute geo.eskb on the VM (build tree: ./build/eshkol-vm-standalone-test geo.eskb)
```

See [`../runtime/eshkol-run.md`](../runtime/eshkol-run.md) for `--profile` and
`--emit-eskb`.

## What the shipped build computes

`vm_geometric.c` has **one** dispatch body, and it computes the **closed forms
of constant-curvature geometry in f64** (`inc/eshkol/backend/riemannian_core.h`)
together with the **differential-form jet calculus**
(`inc/eshkol/backend/differential_form_core.h`).

It used to have two, selected by a legacy compile-time qLLM branch, the
second dispatching through the out-of-tree `semiclassical_qllm` library. **That
body has been deleted.** No target ever defined the macro, so it was unreachable
from every configuration of this repository; and it was not a working
implementation waiting for a switch. Measured against a current
`libsemiclassical_qllm`, a syntax-only pass reported **19 compile errors inside
`vm_geometric.c` alone** — moved arities on `qllm_hyperbolic_exp_map` /
`_log_map` / `_distance` / `_parallel_transport` / `_mobius_add` /
`_mobius_scalar`, `qllm_hyperbolic_project` taking a `qllm_tensor_t*` where the
call passed a `float*`, and SO(3)/SE(3) arms naming types the library no longer
declares — and it was fp32 throughout, which
`inc/eshkol/backend/frechet_mean_core.h` documents as unable to satisfy the
exact-derivative gate. An unreachable body that does not compile is not a second
implementation held in reserve; it is a second set of claims no gate can check.
The `ESHKOL_GEOMETRIC_QLLM` CMake option that briefly selected it is gone with
it.

The **differentiable** route to this geometry is a different, live integration
and is untouched: `lib/bridge/qllm_bridge.cpp` registers exact backward rules on
the AD tape. What the VM opcodes compute is the same mathematics that bridge
computes (`ad_hyperbolic_distance`, `ad_poincare_exp_map`, `ad_poincare_log_map`,
`ad_geodesic_attention`) — **for K < 0**, and only there.

The qualification is not a hedge. The bridge's entry points take a **ball
parameter c > 0**, not a sectional curvature, and convert with
`c = (curvature == 0) ? 1 : |curvature|`. For K < 0 that is c = −K and the two
agree exactly. For K = 0 the bridge silently selects the c = 1 ball — it returns
`2 artanh(0.5) = 1.0986` where this page's `geodesic-distance` returns `0.5` —
and for K > 0 it selects a ball where the VM selects a sphere. The bridge has to
branch on the sign of K and refuse K ≥ 0 on its Poincaré-only producers; that is
a change to `lib/bridge/qllm_bridge.cpp`, tracked on its own lane, and until it
lands **do not differentiate a geometric computation at K ≥ 0 and expect the
tape to agree with the opcode**.

This used not to be true, and the difference is worth stating because a reader of an
older build's output needs to know. Before `SW-73` these ops computed their **flat
(Euclidean) counterparts and discarded the curvature argument they accept**:
`hyperbolic-exp-map` was vector addition, `hyperbolic-log-map` subtraction,
`geodesic-distance` and `poincare-distance` the L2 distance, `mobius-add` addition,
`parallel-transport` and `riemannian-grad` the identity. At K = 0 those forms are
what this page's K = 0 branch still computes (see
[The K = 0 branch](#the-k--0-branch) for the convention question that raises), and
at every other curvature they were wrong without bound: two points of the unit
ball a tenth of a unit inside the boundary are 1.8 apart in L2 and
5.8888779583328814 apart in the metric the name promises. `tests/vm/geometric_riemannian_surface_regression.esk`
pins each of these against its closed form, with every assertion chosen so that the
flat answer fails it.

**All 62 names execute on the VM engine.** Two of them — `exterior-derivative`
and `hodge-star` — used to raise `not implemented on the VM engine`, because a
form's coefficient *values* at a point determine neither its exterior derivative
nor which duality its star is. They now take the input that does determine those:
see [Differential forms](#differential-forms) for the representation.

Three more — `curvature-gradient`, `curvature-hessian` and
`adaptive-curvature-step` — returned a sum, a constant `0.0` and a fixed-rate
update, none of which differentiated anything. They now measure the exact first
and second derivatives of a **named objective** with respect to the curvature
parameter: see [Curvature derivatives](#curvature-derivatives).

## The K = 0 branch

**There is an open convention question here, and it is stated rather than
hidden.** The three promises "sectional curvature K", "ball of radius 1/√−K" and
"canonical Euclidean geometry at K = 0, continuously" cannot all hold in one
coordinate family, and this page used to make all three.

The ball chart's conformal factor is `λ_x = λ₀/(1 − B‖x‖²)` with `λ₀ = 2` and
`B = c·λ₀²/4 = c` — the convention of Ganea et al., Nickel–Kiela and `geoopt`,
and the one every in-tree call site passes. Because `λ₀ = 2` for **every** `c`,
including in the limit, the K → 0⁻ limit of the ball branch is flat space with
the metric `4⟨·,·⟩`: `d(x,y) → 2‖x−y‖` and `riemannian-grad → g/4`. The K = 0
branch instead uses `λ = 1`, the canonical Euclidean metric, which is what
"K = 0 is Euclidean" means to a caller and what every flat-reduction example on
this page asserts.

So **the family is discontinuous at K = 0**: the geodesic distance jumps by a
factor of 2 and the Riemannian gradient by a factor of 4 as K crosses zero. Two
points a tenth of a unit inside the unit ball are `1.8` apart at K = 0 and
`3.6000000012` apart at K = −1e−9.

Which of the two to keep is a **ruling**, not a bug fix, because it changes
published numbers and the AD bridge's contract with them. Both resolutions are
one line in `inc/eshkol/backend/riemannian_core.h`, which is why the model is
centralized in exactly four named places there — `ESHKOL_RM_LAMBDA0`,
`ESHKOL_RM_FLAT_LAMBDA`, `eshkol_rm_ball_param` and `eshkol_rm_lambda`:

- setting `ESHKOL_RM_FLAT_LAMBDA` to `ESHKOL_RM_LAMBDA0` keeps the standard ball
  and makes the family real-analytic in K on K ≤ 0. `geodesic-distance` at K = 0
  becomes `2‖x−y‖` and `riemannian-grad` becomes `g/4`, and the curvature
  derivatives stop refusing at K = 0;
- setting `ESHKOL_RM_LAMBDA0` to `1` keeps canonical Euclidean at K = 0 by
  re-charting the ball to radius `2/√c`, and moves every published hyperbolic
  constant.

Until it is ruled on, the shipped behaviour is the second column of the table
above — `λ₀ = 2`, `flat λ = 1` — and the three curvature-derivative ops
(`curvature-gradient`, `curvature-hessian`, `adaptive-curvature-step`) **raise at
K = 0**, naming the discontinuity, rather than returning a one-sided derivative
as if it were two-sided.

Nothing else moves with the constant. `exp`, `log`, `parallel-transport`,
`manifold-project` and Möbius addition are the **same maps** under either choice:
a constant conformal rescale of a metric leaves the Levi-Civita connection
unchanged, so only distances, norms and the gradient conversion carry the factor.

## Error convention

**Shape and type failures return `()`.** An argument of the wrong type, a shape
mismatch, a non-positive dimension or an allocation failure pushes the empty list
`()` and execution continues; an unhandled id falls through to a `default:` arm that
pops `vm_geometric_arity(fid)` arguments and pushes `()`, so the stack stays balanced
either way. Check for `(null? result)` if you need to distinguish a failure from a
value.

**Domain failures raise a catchable Scheme condition** (via `vm_raise_error_msg`, so
`guard` can catch it), naming the builtin, the reason and the curvature. They raise
rather than return a number because the alternative is the case this whole surface
exists to exclude — a plausible value the caller cannot tell from a real one. The
conditions a caller can hit:

- a point that is not **strictly inside** the Poincaré ball of radius `1/√−K`, for
  any op taking K < 0;
- any non-finite supplied coordinate, curvature, tangent/gradient component or
  metric entry, and any formula whose computed result is not finite;
- a point that is not **on** the sphere of radius `1/√K`, for any op taking K > 0
  (`manifold-project` is the op that moves it there);
- a **tangent vector that is not tangent**, for any spherical op that takes one
  (`spherical-exp`, `parallel-transport`/`vector-transport` at K > 0, the Adam
  step): `⟨x,v⟩` must vanish, tested relative to `‖v‖`;
- **antipodal endpoints** for spherical `log` and `parallel-transport`: neither
  the connecting geodesic nor the transport along it is unique;
- a result that would land **on** the ball boundary rather than strictly inside
  it, for `hyperbolic-exp-map`, `mobius-add` and `mobius-scalar-mul`. `tanh`
  rounds to exactly `1.0` beyond about 19, so `exp_0(20)` at K = −1 used to
  return exactly `1.0` — not a point of the open ball, and the one place these
  ops could return a value outside their own codomain;
- `mobius-add` and `mobius-scalar-mul` with K > 0: they are the gyrogroup operations
  of the ball and are not defined on the sphere;
- `frechet-mean`, when the Karcher iteration has not reached stationarity. A mean
  that has not is exactly the input whose implicit derivative is a plausible wrong
  gradient;
- `spherical-project`, on the origin: it has no projection onto the sphere, and
  the zero vector this op used to return is not a point of the sphere either;
- `exterior-derivative`, on a **0-jet** form: `d` is a derivative, so the
  coefficients' first partials have to be supplied, and substituting zeros for
  them is precisely the assertion of closedness this op no longer makes;
- `hodge-star`, on a metric that is not symmetric positive definite: the star is
  defined by a Riemannian inner product on k-forms;
- `curvature-gradient`, `curvature-hessian` and `adaptive-curvature-step`, at
  **K = 0**, and whenever a supplied point is not a point of the manifold at the
  curvature being differentiated;
- `adaptive-curvature-step`, when no backtracked Newton step is admissible.

## Engine availability per builtin

**All sixty-two names execute on the bytecode VM.** None of them has a native
AOT or REPL/JIT implementation — see [Engine availability](#engine-availability)
above, which applies to all 62 uniformly.

Five of them changed what they compute after `SW-73` and `SW-75`, and a program
written against an older build needs to know which:

| Name | Id | Then | Now |
|---|---:|---|---|
| `exterior-derivative` | 835 | zeros of the input's shape, then (SW-73) a refusal | the exact `d` of a form given as a jet — see [Differential forms](#differential-forms) |
| `hodge-star` | 836 | its argument unchanged, then (SW-73) a refusal | the star with respect to the supplied metric |
| `curvature-gradient` | 852 | the sum of its tensor's elements | `dL/dK` of the geodesic-distance objective |
| `curvature-hessian` | 855 | the constant `0.0` | `d²L/dK²` of the same objective |
| `adaptive-curvature-step` | 856 | `K ← K − 0.01·Σgrad` | a backtracked damped Newton step on the same objective |

`spherical-project` (823) changed too, in one case only: a zero-norm argument
now raises instead of coming back unchanged.

## Data model

- **Points, tangent vectors, forms, quaternions, twists, poses, Q/K/V matrices** are
  all **tensors** (`(make-tensor '(shape) fill)`, or any op that returns one). They
  are *not* Scheme vectors — `vm_get_tensor` returns NULL for a non-tensor, and the
  op then pushes `()`.
- **A differential form** — for `exterior-derivative` (835) and `hodge-star`
  (836) only — is a tensor with the specific layout
  `[k, n, r, coefficient jets...]` described under
  [Differential forms](#differential-forms). The three other form-shaped ops
  (`wedge-product` 834, `interior-product` 837, `pullback` 838) take **flat
  coefficient arrays** and are unchanged; they are algebraic and need neither
  the degree nor a derivative.
- **A batch of point pairs** — for `curvature-gradient` (852),
  `curvature-hessian` (855) and `adaptive-curvature-step` (856) — is a tensor
  holding `(x₀, y₀, x₁, y₁, …)` consecutively, each point of the manifold's
  dimension. Any shape works; only the element count matters, and it must be a
  positive multiple of `2·dim`.
- **A manifold** is an opaque heap handle (`VAL_MANIFOLD`, heap type
  `HEAP_MANIFOLD`) wrapping `{int type; int dim; double curvature;}`. Construct it
  only with the four constructors; inspect it with `manifold-type`,
  `manifold-dim` and `get-curvature`.
- **Manifold type codes** are integers, not symbols: `0` euclidean, `1` hyperbolic,
  `2` spherical, `3` product.
- **A Riemannian-Adam state** is an opaque handle (`VAL_RIEMANNIAN_ADAM_STATE`)
  holding the first- and second-moment buffers plus a step counter.
- Manifold and optimizer-state memory is VM-arena memory owned by the VM region
  stack. `manifold-destroy!` invalidates the handle logically (it nulls the heap
  object's payload pointer); it does not free arena memory.

## The 62 names

Aliases share an id and are therefore the same op. `→` gives the result type:
`tensor`, `float`, `int`, `manifold`, `state`, or `()`.

| Name | Id | Arity | → | Shipped (portable) behaviour |
|---|---:|---:|---|---|
| `make-euclidean-manifold` | 804 | 1 | manifold | type 0, K = 0.0 |
| `make-hyperbolic-manifold` | 805 | 2 | manifold | type 1, K as given |
| `make-spherical-manifold` | 806 | 1 | manifold | type 2, K = 1.0 |
| `make-product-manifold` | 807 | 2 | manifold | type 3, dim = d1+d2, K = mean of the two |
| `manifold-curvature` | 808 | 1 | float | stored K |
| `hyperbolic-exp-map` / `manifold-exp-map` | 809 | 3 | tensor | exp_base(tangent): Möbius (K<0) / great-circle (K>0) / `base + tangent` (K=0) |
| `hyperbolic-log-map` / `manifold-log-map` | 810 | 3 | tensor | log_base(point), the inverse of 809; `point - base` at K=0 |
| `geodesic-distance` / `manifold-distance` | 811 | 3 | float | `arccosh(…)/√c` (K<0) / `R·acos` (K>0) / L2 (K=0) |
| `parallel-transport` / `manifold-parallel-transport` | 812 | 4 | tensor | `(λ_x/λ_y)·gyr[y,−x]v` (K<0) / geodesic rotation (K>0) / identity (K=0) |
| `manifold-project` | 813 | 2 | tensor | rescales onto the ball (K<0) or the sphere (K>0); a copy at K=0 |
| `mobius-add` | 814 | 3 | tensor | the gyrogroup sum `x ⊕_c y`; `x + y` at K=0; **raises for K>0** |
| `mobius-scalar-mul` | 815 | 3 | tensor | `(1/√c)·tanh(r·artanh(√c‖x‖))·x/‖x‖`; `r·x` at K=0; **raises for K>0** |
| `poincare-distance` | 816 | 3 | float | the geodesic distance (same op as 811) |
| `frechet-mean` | 817 | 3 | tensor | **real** weighted Karcher mean, gated; raises |
| `great-circle-distance` | 819 | 2 | float | `acos` of the clamped normalised dot |
| `slerp` | 820 | 3 | tensor | normalised `(1-t)x + t y` |
| `spherical-exp` / `spherical-exp-map` | 821 | 2 | tensor | `cos‖v‖·base + sin‖v‖·v/‖v‖` on the unit sphere |
| `spherical-log` / `spherical-log-map` | 822 | 2 | tensor | `θ·u/‖u‖`, `u = point − cosθ·base`, on the unit sphere |
| `spherical-project` | 823 | 1 | tensor | L2-normalised copy; **raises** on the origin |
| `so3-exp` | 824 | 1 | tensor | axis-angle → unit quaternion, shape `(4)` |
| `so3-log` | 825 | 1 | tensor | quaternion → axis-angle, shape `(3)` |
| `se3-exp` | 826 | 1 | tensor | twist → pose, shape `(7)` |
| `se3-log` | 827 | 1 | tensor | pose → twist, shape `(6)` |
| `quaternion-mul` | 828 | 2 | tensor | Hamilton product, shape `(4)` |
| `metric-tensor` | 829 | 1 | tensor | `dim × dim` identity |
| `christoffel` | 830 | 2 | tensor | `dim³` closed form, see entry |
| `riemann-curvature` | 831 | 1 | float | stored K (same case as 808) |
| `ricci-scalar` | 832 | 1 | float | `dim·(dim-1)·K` |
| `sectional-curvature` | 833 | 3 | float | stored K (u, v discarded) |
| `wedge-product` | 834 | 2 | tensor | the `n(n-1)/2` 2×2 minors |
| `exterior-derivative` | 835 | 1 | form | exact `d`, one jet order consumed |
| `hodge-star` | 836 | 2 | form | `sgn(I,J)·√det g·Σ det((g⁻¹)_{I I'})·ω_{I'}`, a 0-jet result |
| `interior-product` | 837 | 2 | tensor | shape `(1)` holding the dot product |
| `pullback` | 838 | 2 | tensor | `formᵀ · jacobian`, shape `(cols)` |
| `riemannian-sgd-step` | 839 | 4 | tensor | `exp_point(−lr·grad)`; `point − lr·grad` at K=0 |
| `riemannian-adam-step` | 840 | 6 | tensor | **raises** — an implicit pool cannot identify a parameter; use 861 |
| `riemannian-grad` | 841 | 3 | tensor | `((1−c‖x‖²)²/4)·grad` (K<0) / tangent projection (K>0) / a copy (K=0) |
| `retraction` | 842 | 3 | tensor | the exponential map (same op as 809) |
| `vector-transport` | 843 | 4 | tensor | parallel transport (same op as 812) |
| `geodesic-attention-scores` | 844 | 3 | tensor | `−d_K(q_i, k_j)`, shape `(nq nk)` |
| `geodesic-attention-values` | 845 | 3 | tensor | score-weighted average of `V`'s rows |
| `curvature-softmax` | 846 | 2 | tensor | softmax scaled by `1/√|K|` |
| `geodesic-attention-forward` | 847 | 4 | tensor | softmax over `−d_K/(√c·√dim)` then a weighted sum of `V`, shape `(nq vdim)` |
| `set-curvature!` | 850 | 2 | manifold | mutates K, returns the manifold |
| `get-curvature` | 851 | 1 | float | stored K (same case as 808) |
| `curvature-gradient` | 852 | 2 | float | `dL/dK`, `L(K) = Σ d_K(xᵢ, yᵢ)` over the pair batch |
| `transition-geometry!` | 853 | 3 | float | K ← K + rate·(target − K); returns new K |
| `manifold-interpolate` | 854 | 3 | float | `(1-t)·K₁ + t·K₂` — a **curvature**, not a manifold |
| `curvature-hessian` | 855 | 2 | float | `d²L/dK²` of the same `L` |
| `adaptive-curvature-step` | 856 | 2 | manifold | backtracked damped Newton step on `L`; returns the manifold |
| `manifold-type` | 857 | 1 | int | 0/1/2/3 |
| `manifold-dim` / `manifold-dimension` | 858 | 1 | int | stored dim |
| `manifold-destroy!` | 859 | 1 | `()` | invalidates the handle |
| `make-riemannian-adam-state` | 860 | 1 | state | zeroed moments shaped like the point |
| `riemannian-adam-step!` | 861 | 7 | tensor | intrinsic Riemannian Adam: scalar RMS second moment, geodesic retraction, moment transported; refuses non-finite norms |

Ids 818, 848 and 849 have no name bound to them.

`vector-transport` (843) is a geometric op, but
`tests/coverage/language_surface.json` files it under the `vector` category rather
than `geometry`, because `scripts/gen_language_surface.py` categorises by name
prefix. The `geometry` category therefore reports 61 while the family is 62. The
count is not affected — the builtin is registered and covered either way.

---

## Manifold handles

### `(make-euclidean-manifold dim)` — id 804

`dim` integer. Returns a manifold of type 0 with K = 0.0, or `()` if `dim <= 0` or
the arena allocation fails.

```scheme
(define m (make-euclidean-manifold 2))
(display (manifold-type m)) (newline)   ; 0
```

### `(make-hyperbolic-manifold dim curvature)` — id 805

`dim` integer, `curvature` real. Returns a manifold of type 1 carrying the curvature
you pass — it is *not* forced to −1. Note the arity: unlike its
[`core.manifold`](manifold.md) namesake, this constructor takes two arguments.

```scheme
(define h (make-hyperbolic-manifold 2 -1.0))
```

### `(make-spherical-manifold dim)` — id 806

`dim` integer. Returns a manifold of type 2 with K = 1.0.

```scheme
(define s (make-spherical-manifold 2))
```

### `(make-product-manifold m1 m2)` — id 807

Two manifolds. Returns a manifold of type 3 whose dimension is `d1 + d2` and whose
curvature is the arithmetic mean `0.5·(K₁ + K₂)`. `()` if either argument is not a
manifold.

```scheme
(display (manifold-dim (make-product-manifold (make-euclidean-manifold 2)
                                              (make-hyperbolic-manifold 3 -1.0))))
(newline)                                ; 5
```

### `(manifold-type m)` — id 857

Returns the integer type code: `0` euclidean, `1` hyperbolic, `2` spherical, `3`
product. `()` if `m` is not a manifold. This is an **integer**, where
`core.manifold`'s `manifold-type` returns a symbol.

```scheme
(display (manifold-type (make-spherical-manifold 2))) (newline)   ; 2
```

### `(manifold-dim m)` / `(manifold-dimension m)` — id 858

Returns the stored integer dimension, or `()` if `m` is not a manifold.

```scheme
(display (manifold-dim (make-euclidean-manifold 7))) (newline)    ; 7
```

### `(manifold-curvature m)` / `(get-curvature m)` / `(riemann-curvature m)` — id 808 / 851 / 831

Three distinct names dispatching to one case. All return the manifold's stored
constant curvature as a float, `()` if `m` is not a manifold. `riemann-curvature`
returns a **scalar**, not a rank-4 tensor, despite the name.

```scheme
(display (get-curvature (make-hyperbolic-manifold 2 -1.0))) (newline)
```

### `(set-curvature! m k)` — id 850

Mutates the manifold's stored curvature to `k` and returns the manifold itself
(`()` if `m` is not a manifold).

```scheme
(define h (make-hyperbolic-manifold 2 -1.0))
(set-curvature! h -0.5)
(display (get-curvature h)) (newline)
```

### `(manifold-destroy! m)` — id 859

Nulls the heap object's manifold pointer, so every later op on that handle sees "not
a manifold" and returns `()`. Returns `()`. Arena memory is **not** freed here — it
belongs to the VM region stack and is reclaimed when the region is popped.

```scheme
(define h (make-hyperbolic-manifold 2 -1.0))
(manifold-destroy! h)
(display (null? (manifold-dim h))) (newline)   ; #t
```

---

## Maps, distances and gyrovector operations

Every op in this group takes a trailing `curvature` argument, read as the
**sectional curvature K**: `K < 0` is the Poincaré ball of radius `1/√−K`, `K = 0` is
Euclidean space, `K > 0` is the sphere of radius `1/√K`. Each op reduces **exactly**
to its flat form at `K = 0`. Arguments must lie on the manifold the curvature names —
strictly inside the ball for `K < 0`, on the sphere for `K > 0` — or the op raises;
see [Error convention](#error-convention).

### `(hyperbolic-exp-map base tangent curvature)` / `(manifold-exp-map …)` / `(retraction base tangent curvature)` — id 809 / 842

`base`, `tangent` tensors of equal total size; `curvature` real. Returns
`exp_base(tangent)`, the point reached by following the geodesic from `base` in the
direction `tangent` for the Riemannian length of `tangent`; `()` on a shape mismatch.
`retraction` is a second name on the same op — and it is a true exponential map, not
merely a retraction.

- `K < 0`: `base ⊕_c tanh(√c·λ·‖v‖/2)·v/(√c‖v‖)` with `λ = 2/(1−c‖base‖²)`, `c = −K`.
- `K = 0`: `base + tangent`, exactly.
- `K > 0`: `cos(‖v‖/R)·base + R·sin(‖v‖/R)·v/‖v‖` with `R = 1/√K`; `v` must be
  tangent (`⟨base, v⟩ = 0`).

```scheme
(define base (make-tensor '(2) 0.0))
(tensor-set! base 0 0.3)
(define v (make-tensor '(2) 0.2))
(hyperbolic-exp-map base v -1.0)
```

### `(hyperbolic-log-map base point curvature)` / `(manifold-log-map …)` — id 810

Returns `log_base(point)`, the inverse of id 809: the tangent vector at `base`
whose exponential map is `point`. Its **Riemannian** length `λ_base·‖log‖` is the
geodesic distance; its ambient Euclidean length is not.

- `K < 0`: direction from the **Möbius numerator** of `(−base) ⊕_c point`,
  magnitude `d(base, point)/λ_base`.
- `K = 0`: `point − base`, exactly.
- `K > 0`: `θR·u/‖u‖` with `u = point − cos θ·base` and `θ` from `atan2`.

**Every pair of strictly interior ball points has a finite log, and this op
returns it.** It used to compute `(2/(√c·λ))·artanh(√c‖u‖)·u/‖u‖` and *refuse*
whenever `√c‖u‖` rounded to 1, on the stated grounds that no finite log existed
there. That was wrong: the rounding was a numerical failure of the `artanh`
route, not a fact about the manifold. For `x = 0.999999999`, `y = −0.999999999`
at `c = 1` the true distance is `4·artanh(x) ≈ 42.83`; this op refused, and the
AD bridge on the same input clamped to `1 − 1e−12` and returned a fabricated
`28.32`. Splitting the computation — direction from the numerator alone, never
divided by the near-zero Möbius denominator, magnitude from the stable distance
below — removes both failures.

The spherical branch raises for **antipodal** points, where the log is genuinely
not single-valued.

```scheme
(define base (make-tensor '(2) 0.0))
(tensor-set! base 0 0.3)
(define p (make-tensor '(2) 0.0))
(tensor-set! p 0 -0.5)
(hyperbolic-log-map base p -1.0)
```

### `(geodesic-distance x y curvature)` / `(manifold-distance …)` / `(poincare-distance x y curvature)` — id 811 / 816

Returns the geodesic distance as a float; `()` unless both arguments are tensors of
the same total size. `poincare-distance` is a second name on the same op, and it IS
the Poincaré metric.

- `K < 0`: `(2/√c)·asinh(√c‖x−y‖/√((1−c‖x‖²)(1−c‖y‖²)))`.
- `K = 0`: `‖x − y‖` times the flat conformal factor — see
  [The K = 0 branch](#the-k--0-branch).
- `K > 0`: `R·θ` with `θ = atan2(‖y − cos θ·x‖/R, cos θ)`, the great-circle
  distance.

The hyperbolic distance is not bounded by the ball diameter the way the chord is: it
diverges as either point approaches the boundary.

Both curved branches are written in the **stable** form. The `asinh` expression
is the same number as the `arccosh(1 + 2c‖x−y‖²/P)/√c` it replaces — the identity
is `arccosh(1 + 2z²) = 2·asinh(z)` — but `arccosh(1 + ε)` throws away every digit
of a small separation: at `c = 1`, `x = 0`, `y = 1e−9` the argument `1 + 2e−18`
rounds to `1` and the old form returned **exactly zero** for a distance of
`2e−9`. `atan2` does the same job for the sphere, where `acos(⟨x,y⟩/R²)` is
ill-conditioned near both `0` and `π`.

```scheme
(define a (make-tensor '(2) 0.0))
(tensor-set! a 0 0.9)
(define b (make-tensor '(2) 0.0))
(tensor-set! b 0 -0.9)
(display (geodesic-distance a b -1.0)) (newline)   ; 5.888877958332881
(display (geodesic-distance a b 0.0)) (newline)    ; 1.8
```

### `(parallel-transport x y v curvature)` / `(manifold-parallel-transport …)` / `(vector-transport x y v curvature)` — id 812 / 843

Transports the tangent vector `v` from `x` to `y` along the connecting geodesic.

- `K < 0`: `(λ_x/λ_y)·gyr[y, −x]v`, with the gyration in its **linear closed
  form** `gyr[a,b]w = w + 2(A·a + C·b)/D`.
- `K = 0`: the identity, exactly.
- `K > 0`: `v − (⟨y,v⟩/(R² + ⟨x,y⟩))·(x + y)`, the closed form, which is exactly
  tangent at `y` by construction. `v` must be tangent at `x` and the endpoints
  must not be antipodal, or the op raises.

Transport is an isometry of the Riemannian metric — `λ_y‖P v‖ = λ_x‖v‖` — even
though it changes the ambient components. Transport from a point to itself is the
identity, **for every tangent vector**, including one whose Euclidean norm is
larger than the ball's radius.

That last clause is the reason for the closed form. The gyration is also
`gyr[a,b]w = −(a ⊕ b) ⊕ (a ⊕ (b ⊕ w))`, and this op used to compute it that way —
which feeds a **tangent vector** into Möbius **point** addition. A tangent vector
is not constrained to the ball, so an intermediate denominator can vanish even
though the gyration is a globally defined linear isometry. At K = −1 with
`x = y = (0.5)` and `v = (2)`, every gyration in one dimension is the identity, so
the answer must be `(2)`; the intermediate `(−0.5) ⊕ 2` has denominator zero, the
old denominator floor turned its exact `0/0` cancellation into zero, and the op
returned `(0.5)`. The linear form is linear in `v` and its only denominator is
the Möbius denominator of the two validated **points**.

```scheme
(define x (make-tensor '(2) 0.0)) (tensor-set! x 0 0.3)
(define y (make-tensor '(2) 0.0)) (tensor-set! y 0 -0.5)
(define v (make-tensor '(2) 0.2))
(parallel-transport x y v -1.0)
```

### `(manifold-project x curvature)` — id 813

Moves `x` onto the manifold of curvature `K`.

- `K < 0`: rescales onto the open ball of radius `1/√−K` when `x` is on or outside
  it; interior points are returned unchanged. The result is **strictly** inside, so
  that `λ` stays finite and the log map does not degenerate.
- `K = 0`: a copy.
- `K > 0`: rescales to radius `1/√K`. Raises for the origin, which has no projection.

This is the op to call when a point may have drifted off the manifold: every other op
in this group refuses an off-manifold argument rather than projecting silently.

```scheme
(manifold-project (make-tensor '(2) 3.0) -1.0)
```

### `(mobius-add x y curvature)` — id 814

The gyrovector addition of the Poincaré ball model, `c = −K`:

```
x ⊕_c y = ((1 + 2c⟨x,y⟩ + c‖y‖²)x + (1 − c‖x‖²)y) / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²)
```

It is **neither commutative nor associative** — `gyr[x,y]` is exactly the failure of
commutativity. At `K = 0` it is `x + y`, exactly. It **raises for `K > 0`**: Möbius
addition is the gyrogroup operation of the ball and is not defined on the sphere,
and it raises when the result is not strictly interior.

**Near the boundary the denominator is small but never zero**, and it is
evaluated as `(1 + c⟨x,y⟩)² + c²(‖x‖²‖y‖² − ⟨x,y⟩²)` rather than as the sum
above — the same number, but the first term is a subtraction of two nearby
values (hence exact in f64) and the second is a sum of squares by Lagrange's
identity (hence free of cancellation). The direct sum computes a denominator of
`9e−18` as a difference of terms of size 1 and loses every digit of it, which is
what made the old `1e−15` *floor* look necessary; flooring the denominator
without scaling the numerator returned `0.003` for a sum whose value is about
`1/3`. The floor is gone.

The quantities `1 − c‖x‖²` and `1 + c⟨x,y⟩` are computed by a double-double dot
product closed with a fused multiply-add, to full **relative** precision. That
matters because the ball chart destroys them near the boundary: a point at
Euclidean distance `ε` from the boundary has `1 − c‖x‖² ≈ 2ε`, and computing it
as one minus a rounded sum of squares knows it only to an *absolute* `1e−16` —
eight significant digits at `ε = 1e−9`, none at `ε = 1e−16`. Near-boundary points
are not an edge case on the Poincaré ball; they are the reason to use it.

```scheme
(define x (make-tensor '(2) 0.1))
(define y (make-tensor '(2) 0.2))
(mobius-add x y -1.0)     ; not equal to (mobius-add y x -1.0)
```

### `(mobius-scalar-mul r x curvature)` — id 815

Note the argument order: the **scalar comes first**. Returns the gyrovector scalar
multiple `(1/√c)·tanh(r·artanh(√c‖x‖))·x/‖x‖`, which agrees with `mobius-add` — for
example `2 ⊗ x = x ⊕ x`. Because `tanh` is bounded, the result cannot leave the ball
for any `r`, which `r · x` does. At `K = 0` it is `r · x`, exactly. It **raises for
`K > 0`**, for the same reason `mobius-add` does, and it **raises when `tanh`
saturates**: `tanh` rounds to exactly `1.0` beyond about 19, so
`(mobius-scalar-mul 100 0.5 -1.0)` used to return exactly `1.0`, which is on the
boundary and therefore not a point of the ball.

```scheme
(mobius-scalar-mul 0.5 (make-tensor '(2) 0.4) -1.0)
```

### `(frechet-mean points weights curvature)` — id 817

The only op in this group whose result is defined by an iteration rather than a
closed form, which is why it is also the only one that can fail to converge.

- `points` — a tensor. Rank ≥ 2 is read as `(n_points dim)`; a rank-1 tensor is read
  as a single point of dimension `total`.
- `weights` — a tensor of `n_points` non-negative weights. They need not be
  normalised; only their positive sum matters. `()` weights are treated as absent.
- `curvature` — real, and **must be ≤ 0**: the op is the hyperbolic/Euclidean
  Karcher mean, and a positive curvature is rejected.
- Returns a tensor of shape `(dim)`.

The forward pass is `eshkol_frechet_mean_compute` in
[`inc/eshkol/backend/frechet_mean_core.h`](../../../inc/eshkol/backend/frechet_mean_core.h),
shared with the AD bridge producer so that the opcode and the derivative can never
compute different means. It iterates at most `ESHKOL_FRECHET_MAX_ITERS` = 256 times
and accepts the result only when the relative stationarity residual of
`Σᵢ wᵢ log_μ(xᵢ) = 0` falls to `ESHKOL_FRECHET_RESID_TOL` = `1e-9` or below.

If it cannot, the op raises a catchable error naming the cause, the point count, the
dimension, the curvature, the achieved residual and the tolerance. The refusals are
explicit in the core header: fewer than one point or a non-positive dimension, a
positive curvature, a NaN weight, a negative weight, a non-positive total weight, a
NaN coordinate, a point outside the Poincaré ball, an iterate that reached the ball
boundary, a `log_μ(xᵢ)` with no finite f64 value, and failure to reach stationarity.

```scheme
(define pts (make-tensor '(2 2) 0.1))
(define w   (make-tensor '(2) 1.0))
(frechet-mean pts w -1.0)
```

Catch the refusal rather than let it propagate if the inputs are user data:

```scheme
(guard (e (#t (display "frechet-mean refused") (newline)))
  (frechet-mean pts (make-tensor '(2) 0.0) -1.0))   ; total weight 0 → raises
```

---

## Sphere, rotation and rigid-motion groups

### `(great-circle-distance x y)` — id 819

Two tensors of equal total size. Returns `acos(clamp(⟨x,y⟩ / (‖x‖‖y‖), −1, 1))`, and
`0.0` if either norm is zero. `()` on a size mismatch.

```scheme
(define x (make-tensor '(3) 1.0))
(display (great-circle-distance x x)) (newline)
```

### `(slerp x y t)` — id 820

Returns the L2-normalised linear blend `normalize((1−t)·x + t·y)`. This is normalised
lerp ("nlerp"), not the constant-angular-velocity spherical interpolation the name
usually denotes; the two agree in direction but not in parameterisation.

```scheme
(slerp (make-tensor '(3) 1.0) (make-tensor '(3) 2.0) 0.5)
```

### `(spherical-exp base tangent)` / `(spherical-exp-map …)` — id 821

Two tensors. Returns the exponential map on the **unit sphere** (K = +1):
`cos‖v‖·base + sin‖v‖·v/‖v‖`. `base` must lie on the unit sphere and `v` must be
tangent to it (`⟨base, v⟩ = 0`), or the op raises.

It used to return `normalize(base + tangent)`, a retraction: the right geodesic at
the wrong arc length (`atan‖v‖` instead of `‖v‖`), so it agreed to first order and
disagreed at every order after.

The tangency test is `|⟨base,v⟩| ≤ 1e−9·R·‖v‖` — **relative to `‖v‖`**. It used to
be `|⟨base,v⟩|/R ≤ 1e−9·(‖v‖ + R)`, whose additive `R` made the threshold
absolute, so a *wholly radial* `v` of norm `1e−10` passed at `R = 1`: 100% radial
and accepted, because the threshold did not shrink with the vector.

```scheme
(define e1 (make-tensor '(3) 0.0)) (tensor-set! e1 0 1.0)
(define v  (make-tensor '(3) 0.0)) (tensor-set! v  1 1.0)
(spherical-exp e1 v)   ; (0.5403023058681398 0.8414709848078965 0)
```

### `(spherical-log base point)` / `(spherical-log-map …)` — id 822

Returns the logarithmic map on the **unit sphere**, the inverse of id 821:
`θ·u/‖u‖` with `u = point − cos θ·base` and `θ = acos⟨base, point⟩`, so its length is
the great-circle angle. Shares a case with id 810 but takes **two** arguments, not
three — the curvature is fixed at K = +1. Both arguments must lie on the unit sphere.
Antipodal points raise: the log is not single-valued there.

```scheme
(define e1 (make-tensor '(3) 0.0)) (tensor-set! e1 0 1.0)
(define e2 (make-tensor '(3) 0.0)) (tensor-set! e2 1 1.0)
(spherical-log e1 e2)   ; length pi/2
```

### `(spherical-project x)` — id 823

Returns an L2-normalised copy of `x`: the projection of `x` onto the unit sphere,
`K = +1`. **The origin raises**, because it has no projection — every point of
the sphere is equidistant from it, so no answer is determined. This op used to
return the zero vector unchanged in that case, which is not a point of the sphere
and therefore not a value of this op's own codomain; nothing in the returned
tensor showed that.

```scheme
(spherical-project (make-tensor '(3) 2.0))

(guard (e (#t 'no-projection))
  (spherical-project (make-tensor '(3) 0.0)))
```

### `(so3-exp omega)` — id 824

`omega` a tensor with at least 3 elements, read as an axis-angle rotation vector.
Returns a unit quaternion of shape `(4)` in `(w x y z)` order:
`w = cos(θ/2)`, `(x y z) = ω·sin(θ/2)/θ` where `θ = ‖ω‖`. For `θ ≤ 1e-12` it returns
the identity quaternion `(1 0 0 0)`. `()` if the input has fewer than 3 elements.

```scheme
(so3-exp (make-tensor '(3) 0.0))    ; → (1 0 0 0)
```

### `(so3-log q)` — id 825

`q` a tensor with at least 4 elements. Normalises it, then returns the axis-angle
vector of shape `(3)`: `θ = 2·atan2(‖v‖, w)`, direction `v/‖v‖`. Returns zeros for a
zero-norm input or a vector part below `1e-12`. `()` if the input has fewer than 4
elements.

```scheme
(so3-log (so3-exp (make-tensor '(3) 0.0)))
```

### `(se3-exp twist)` — id 826

`twist` a tensor with at least 6 elements: rotation part in `0..2`, translation part
in `3..5`. Returns a pose of shape `(7)` — the quaternion from the rotation part
followed by the translation copied through unchanged. `()` if fewer than 6 elements.

```scheme
(se3-exp (make-tensor '(6) 0.0))
```

### `(se3-log pose)` — id 827

`pose` a tensor with at least 7 elements: quaternion in `0..3`, translation in
`4..6`. Returns a twist of shape `(6)`, the inverse of `se3-exp`. `()` if fewer than
7 elements.

```scheme
(se3-log (se3-exp (make-tensor '(6) 0.0)))
```

### `(quaternion-mul q1 q2)` — id 828

Two tensors with at least 4 elements each, in `(w x y z)` order. Returns their
Hamilton product as a tensor of shape `(4)`. `()` if either has fewer than 4
elements.

```scheme
(quaternion-mul (so3-exp (make-tensor '(3) 0.0))
                (so3-exp (make-tensor '(3) 0.0)))
```

---

## Metric, connection and curvature queries

### `(metric-tensor m)` — id 829

Returns the `dim × dim` identity tensor — the metric of a constant-curvature space
at this level of approximation. `()` unless `0 < dim ≤ 256`.

```scheme
(metric-tensor (make-hyperbolic-manifold 2 -1.0))
```

### `(christoffel m point)` — id 830

`m` a manifold, `point` a tensor. Returns the rank-3 connection tensor of shape
`(dim dim dim)`, indexed `[k][i][j]`, with entries

`Γ = K · ( δᵢⱼ·xₖ − δⱼₖ·xᵢ − δᵢₖ·xⱼ )`

The working dimension is the manifold's `dim`, clamped down to `point`'s element
count. `()` if `point` is not a tensor, if the manifold lookup fails, or if the
working dimension is not in `1..64`.

```scheme
(christoffel (make-hyperbolic-manifold 2 -1.0) (make-tensor '(2) 0.1))
```

### `(ricci-scalar m)` — id 832

Returns the scalar curvature `dim·(dim−1)·K` as a float. `()` if the manifold lookup
fails or `dim <= 0`.

```scheme
(display (ricci-scalar (make-spherical-manifold 3))) (newline)   ; 3*2*1.0
```

### `(sectional-curvature m u v)` — id 833

Pops and discards the two tangent vectors and returns the manifold's constant
curvature. `()` if `m` is not a manifold.

```scheme
(define t (make-tensor '(2) 1.0))
(display (sectional-curvature (make-hyperbolic-manifold 2 -1.0) t t)) (newline)
```

---

## Differential forms

`exterior-derivative` (835) and `hodge-star` (836) take a **form value** — the
jet representation below. The other three ops in this section
(`wedge-product` 834, `interior-product` 837, `pullback` 838) take **flat
coefficient arrays** and are unchanged: they are algebraic and need neither a
degree nor a derivative.

### Why the two encodings differ

A form's coefficient **values at a point** determine neither `d` nor `★`, and
this is the reason both ops used to be wrong and then had to refuse:

- `d` is a derivative. It cannot be computed from values, only from the
  coefficients as differentiable functions of position. Returning zeros — what
  this op did before `SW-73` — is not "no information available"; it is the
  assertion that every form handed to it is closed, made without examining one.
- `★` depends on the degree `k` and the dimension `n`. A flat array records
  neither, and `C(n,k) = C(n,n−k)` leaves `k` ambiguous even when `n` and the
  array length are both known: three coefficients on `R³` is a 1-form or a
  2-form, and the star differs. Returning the argument unchanged is the star only
  for a self-dual middle-degree form in a Euclidean metric — the identity map
  under the name of a duality.

`d` is however a **first-order** operator, so it does not need the whole function
either: the 1-jet of the coefficients at the point is exactly enough, and is the
minimal input that determines the answer. That is what the representation
carries.

### The form representation

A **k-form on Rⁿ, known to jet order r at a point**, is a tensor whose flat data
is

    [ k, n, r, jet(ω_{I₀}), jet(ω_{I₁}), …, jet(ω_{I_{m−1}}) ]

- `k`, `n`, `r` are the first three elements, and must be integral;
- `m = C(n,k)`, and the basis `I_t` runs over the **increasing k-multi-indices**
  of `{0,…,n−1}` in **lexicographic order**, so `ω = Σ_t ω_{I_t} dx^{I_t}`;
- each jet block holds the coefficient's Taylor data at the point, by order,
  row-major inside each order:

      jet(ω_I) = [ ω_I | ∂_j ω_I  (n) | ∂_j ∂_l ω_I  (n²) | … ]

  The order-`s` block is the **full** `nˢ` array of `s`-th partials, stored
  redundantly (it is symmetric). The validator requires every payload entry
  to be finite and every derivative block to be exactly symmetric under
  permutation of its partial indices. The redundancy buys a contiguous slice,
  which is what makes `d` an addition over slices;
- the stride of one coefficient is `S(n,r) = 1 + n + n² + … + nʳ`, and a
  well-formed form holds **exactly** `3 + m·S(n,r)` elements.

A tensor of any other length is **not a form**: the op reports a shape failure
(`()`) rather than guessing a degree. Bounds: `1 ≤ n ≤ 8` and `0 ≤ r ≤ 3`, and
`0 ≤ k ≤ n` — with `k = n+1` denoting the zero top-degree form, which `d` produces
from an `n`-form and which has no coefficients. A form outside those bounds is
refused by name, never truncated.

The tensor's **shape** is irrelevant; only its element count and contents matter.
`(make-tensor '(10) 0.0)` and `(make-tensor '(2 5) 0.0)` are equally good
containers for a 2-jet 0-form on `R²`.

### Worked example: `d(df) = 0`

For `f(x,y) = x²y + 5xy²` at `(1,2)`: `f = 22`, `f_x = 24`, `f_y = 21`,
`f_xx = 4`, `f_xy = f_yx = 22`, `f_yy = 10`. As a 2-jet 0-form on `R²`
(`m = C(2,0) = 1`, `S(2,2) = 1+2+4 = 7`, total `3 + 7 = 10`):

    [0 2 2 | 22 | 24 21 | 4 22 22 10]

`d` of it is a **1-jet 1-form** (`m = 2`, `S(2,1) = 3`, total `9`) — the two
coefficients are `f_x` and `f_y`, each carrying its own gradient, which are the
rows of the Hessian:

    [1 2 1 | 24  4 22 | 21 22 10]

`d` of *that* is a **0-jet 2-form** (`m = 1`, total `4`):

    [2 2 0 | 0]

and the `0` is bit-equal zero, because the two mixed partials that cancel are the
same stored double rather than two separately-computed approximations of it. A
form that is *not* closed reports so: `ω = y dx` on `R²` as a 1-jet is
`[1 2 1 | 2 0 1 | 0 0 0]`, and `d(ω) = −dx∧dy`, coefficient `−1`.

### `(wedge-product a b)` — id 834

Two tensors. With `n = min(total_a, total_b)`, returns a tensor of shape
`(n(n−1)/2)` holding the 2×2 minors `aᵢbⱼ − aⱼbᵢ` for `i < j`, in row-major `(i, j)`
order. For `n ≤ 1` the result has shape `(1)` and is zero. `()` if either argument
is not a tensor.

```scheme
(wedge-product (make-tensor '(3) 1.0) (make-tensor '(3) 2.0))
```

### `(exterior-derivative form)` — id 835

`form` is a **differential form in the jet representation** below. Returns the
exterior derivative: a `(k+1)`-form of jet order `r−1`, on the same `n`.

    (dω)_J = Σ_{p=0..k} (−1)^p ∂_{J_p} ω_{J∖J_p}

over increasing `(k+1)`-multi-indices `J`, and the order-`s` block of each output
coefficient is the order-`(s+1)` block of the corresponding input coefficient,
sliced at its leading index. **The result is exact** — no difference quotient, no
step size, no truncation. The supplied payload must be finite and its redundant
derivative blocks must be exactly symmetric; a non-symmetric jet is refused
rather than accepted as a representation of a differentiable coefficient. For
every accepted jet, `d` is exact.

`d` consumes one jet order, so `d(d(ω))` is computable from an `r ≥ 2` form and
comes out **bit-equal zero**: the two mixed partials that cancel are the same
stored double, not two separately-computed approximations of it. Applying `d` to
the canonical zero `(n+1)`-form produced from an `n`-form returns that canonical
zero form again.

**Raises** on a 0-jet form (`r = 0`) — the partials it would need were not
supplied, and substituting zeros for them is the assertion of closedness this op
used to make for every input. Returns `()` for a tensor that is not a
well-formed form at all.

```scheme
;; f(x, y) = x²y + 5xy² at (1, 2), as a 2-jet 0-form on R²:
;;   f = 22, f_x = 24, f_y = 21, f_xx = 4, f_xy = f_yx = 22, f_yy = 10
(define jet-f (make-tensor '(10) 0.0))
;; [k n r | f | f_x f_y | f_xx f_xy | f_yx f_yy]
;;  0 2 2   22   24  21    4   22     22   10

(define df (exterior-derivative jet-f))   ; a 1-jet 1-form: 24 dx + 21 dy
(exterior-derivative df)                  ; a 0-jet 2-form, coefficient exactly 0.0
```

### `(hodge-star form metric)` — id 836

`form` is a differential form in the jet representation; `metric` is the `n × n`
metric `g` at the point, as a tensor of `n²` elements. Returns the `(n−k)`-form

    (★ω)_J = sgn(I, J) · √det g · Σ_{I'} det( (g⁻¹)_{I I'} ) · ω_{I'}

where `I` is the complement of `J` and `sgn(I, J)` is the sign of the permutation
`(I then J)` of `(0 … n−1)`. This is the star defined by `α ∧ ★ω = ⟨α, ω⟩ vol`
with `vol = √det g · dx⁰∧…∧dx^{n−1}`; the inner sum is the induced inner product
on k-forms, so a non-diagonal metric costs one `k × k` minor of `g⁻¹` per basis
pair.

**The result is a 0-jet form, always, and its header says so.** The star's
coefficients are functions of `g`, so the jet of `★ω` depends on the jet of the
metric — and a metric sampled at one point carries nothing about how `g` varies.
Propagating the input's derivative blocks through it would assert that `g` is
constant: true in the flat case, false on any curved manifold, and
indistinguishable from the truth in the returned tensor. Apply `d` **before** the
star, not after it, when both are wanted.

`g` must be **symmetric positive definite** — the star is defined by a Riemannian
inner product — and a metric that is not raises. The check is the Cholesky
factorisation itself, which also supplies `det g`, so "is this a metric" and "what
is its determinant" cannot disagree.

Involution: `★★ω = (−1)^{k(n−k)} ω`. On `R³` with `k = 1` that is `+1`, which is
what the old identity-map body also returned — the `n = 4`, `k = 1` case, where it
is `−1`, is the one that distinguishes them.

```scheme
;; *(dx ∧ dy) = dz in flat R³
(define dxdy (make-tensor '(6) 0.0))    ; [2 3 0 | 1 0 0]
(define id3 (make-tensor '(3 3) 0.0))   ; the identity metric
(hodge-star dxdy id3)                   ; [1 3 0 | 0 0 1], i.e. dz

;; On a conformal metric g = λ²·δ, the star of a k-form scales by λ^(n−2k):
;; with the Poincaré factor λ = 2/(1 − c|x|²) = 8/3 at |x|² = 1/4, c = 1,
;; *(dx) = (8/3)·(dy ∧ dz).
```

### `(interior-product vector form)` — id 837

Contraction `ι_v ω`. Both arguments must be tensors of the same total size. Returns
a tensor of shape `(1)` holding their dot product — a rank-0 result boxed as a
one-element tensor, not a float. `()` on a size mismatch.

```scheme
(interior-product (make-tensor '(3) 1.0) (make-tensor '(3) 2.0))
```

### `(pullback form jacobian)` — id 838

Returns `formᵀ · jacobian` as a tensor of shape `(cols)`. When `jacobian` has rank ≥
2 its shape supplies `(rows, cols)`; otherwise `rows` is taken from `form`'s element
count and `cols` inferred by division. `()` if the shapes cannot be reconciled
(`rows·cols` exceeding the jacobian's element count, or `rows` exceeding the form's).

```scheme
(pullback (make-tensor '(2) 1.0) (make-tensor '(2 3) 1.0))
```

---

## Riemannian optimizers

All three step ops read their trailing curvature argument and **retract onto the
manifold with the exponential map**, so the iterate cannot leave it. `gradient` is
the **Riemannian** gradient — a tangent vector; `riemannian-grad` is the op that
converts a Euclidean one, which is why they are separate names. At `K = 0` every step
reduces exactly to the ambient update.

### `(riemannian-sgd-step point gradient lr curvature)` — id 839

Returns `exp_point(−lr · gradient)` as a fresh tensor; `point − lr · gradient` at
`K = 0`. `()` on a shape mismatch, and raises if `point` is not on the manifold.

```scheme
(define p (make-tensor '(2) 0.25))
(define g (make-tensor '(2) 0.5))
(riemannian-sgd-step p g 0.25 -1.0)
```

### `(riemannian-adam-step point gradient lr beta1 beta2 curvature)` — id 840

**Raises.** Use `riemannian-adam-step!` (861) with an explicit state.

This op drew its optimizer state from an **implicit pool keyed by the point
tensor's shape**: 16 VM-lifetime slots, the first whose shape matched `point`,
overwriting slot 0 when all 16 were taken. Adam state belongs to a *parameter*,
not to a shape, so two independent optimisation loops over same-shaped
parameters shared one set of moments and one step counter — each one's update
depended on the other's history, with nothing in the returned tensor to show it.
At K = 0, lr = 0.1, β₁ = 0.9, β₂ = 0.999, a first parameter with g = +1 steps by
−0.1 and a second, independent parameter with g = −1 should step by +0.1; the
shared pool gave it **+0.00526316**.

There is no repair inside this arity, which is why the op refuses rather than
being fixed in place. Keying by the point tensor's identity instead of its shape
does not work either: this op **returns a new tensor** each call, so a
per-iteration parameter would never match its own state and Adam would silently
degrade to a bias-corrected SGD. The state has to be named by the caller.

```scheme
(guard (e (#t 'use-the-explicit-state))
  (riemannian-adam-step p g 0.01 0.9 0.999 -1.0))
```

### `(make-riemannian-adam-state point)` — id 860

Allocates a zeroed state — a first-moment buffer shaped like `point`, one scalar
second moment for this manifold factor, and a step counter — and returns the
handle. The state is region-scoped and identity-keyed to `point`; it cannot be
used with a different same-shaped tensor. `()` if `point` is not a usable tensor.

```scheme
(define st (make-riemannian-adam-state (make-tensor '(4) 1.0)))
```

### `(riemannian-adam-step! state point gradient lr beta1 beta2 curvature)` — id 861

Bias-corrected **intrinsic** Riemannian Adam with `ε = 1e-8`. `beta1` outside
`[0, 1)` falls back to `0.9`, `beta2` outside `[0, 1)` to `0.999`, and a negative
`lr` is negated. The delta is formed in the tangent space at `point`, retracted
with `exp_point`, and the **first moment is parallel-transported** to the new
point — it is a tangent vector at the old one and is meaningless at the new one
until it is moved. Despite the `!`, the returned value is a **new** tensor;
`point` is not mutated, `state` is. `()` if the state is not identity-keyed to
`point` (a shape match alone is not enough).

**The second moment is one scalar per manifold factor**, not a per-coordinate
array:

    s_t = β₂ s_{t−1} + (1 − β₂) ‖g_t‖²_{x_t}

measured in the factor's own metric, so the delta stays **parallel to the first
moment** and is therefore tangent wherever the gradient is. The per-coordinate
form this replaces is not an operation on a manifold — it does not commute with a
change of chart and it does not preserve a tangent space. On the unit sphere at
`x = (1,1,1)/√3` with the perfectly valid tangent gradient `g = (1,1,−2)` (note
`⟨x,g⟩ = 0`), dividing coordinate by coordinate gives a delta proportional to
`(1,1,−1)`, whose inner product with `x` is `−lr/√3`, and the exponential map
**refused a valid optimizer input**. The scalar construction is the one
Bécigneul–Ganea give and the one `geoopt`'s `RiemannianAdam` implements; this
page previously cited geoopt for the coordinate-wise version, which was wrong.

A consequence worth testing against: on the first step the bias corrections
cancel exactly, so the delta is `−lr·g/‖g‖_x` and **the step's Riemannian length
is exactly `lr`**, on every branch.

**A refused step changes nothing.** The moments, step counter, and parameter
identity are held in temporaries and committed only after the exponential map
and moment transport have both succeeded. They used to be written first, so a
call that ended in a raise still advanced the optimizer and the retry answered a
different question.

```scheme
(define p  (make-tensor '(4) 1.0))
(define g  (make-tensor '(4) 0.5))
(define st (make-riemannian-adam-state p))
(riemannian-adam-step! st p g 0.01 0.9 0.999 -1.0)
```

### `(riemannian-grad euclidean-grad point curvature)` — id 841

Converts a Euclidean gradient at `point` into the Riemannian one.

- `K < 0`: `((1 − c‖x‖²)² / 4) · grad`. The ball's metric is conformal with factor
  `λ_x = 2/(1 − c‖x‖²)`, so the Riemannian gradient is `grad / λ_x²`; the factor goes
  to zero at the boundary.
- `K = 0`: a copy, exactly.
- `K > 0`: the ambient gradient projected onto the tangent space at `point`.

```scheme
(define p (make-tensor '(2) 0.0)) (tensor-set! p 0 0.3) (tensor-set! p 1 -0.1)
(define g (make-tensor '(2) 0.0)) (tensor-set! g 0 1.0)
(riemannian-grad g p -1.0)   ; (0.2025 0)
```

---

## Geodesic attention

### `(geodesic-attention-scores Q K curvature)` — id 844

`Q` and `K` tensors. A rank-≥2 tensor is read as `(n, d)`; a rank-1 tensor is read as
a single row of dimension `total`. The two feature dimensions must agree. Returns a
tensor of shape `(nq nk)` whose entries are the **negated geodesic distances**
`−d_K(qᵢ, kⱼ)` — closer keys score higher — which is the convention
`ad_geodesic_attention` uses on the AD tape. `()` if either argument is not a tensor
or the dimensions disagree; raises if a row is off the manifold.

```scheme
(geodesic-attention-scores (make-tensor '(2 3) 0.1) (make-tensor '(2 3) 0.2) -1.0)
```

### `(geodesic-attention-values scores V curvature)` — id 845

`V` must have rank ≥ 2, read as `(n, dim)`. Returns a tensor of shape `(dim)`: the
average of `V`'s rows weighted by the leading `n` entries of `scores`, divided by
their sum (or left unnormalised if that sum is exactly zero). The scores are used
**raw** — pass them through `curvature-softmax` first if you want a probability
weighting.

```scheme
(define v (make-tensor '(2 3) 1.0))
(define s (make-tensor '(2) 0.5))
(geodesic-attention-values s v -1.0)
```

### `(curvature-softmax scores curvature)` — id 846

Numerically stable softmax (maximum subtracted before exponentiating) with the
logits scaled by `1/√|K|`, or by `1.0` when `K` is exactly zero. Returns a tensor of
the input's shape. `()` if `scores` is not a non-empty tensor.

```scheme
(curvature-softmax (make-tensor '(4) 0.25) -1.0)
```

### `(geodesic-attention-forward Q K V curvature)` — id 847

The fused path. `Q`, `K` and `V` must all have rank ≥ 2; `K`'s feature dimension must
equal `Q`'s and `V` must have at least as many rows as `K`. Returns a tensor of shape
`(nq vdim)` where row `i` is the average of `V`'s rows weighted by the softmax of
`−d_K(qᵢ, kⱼ)/(√c·√dim)`, the curvature-adaptive scaling `ad_geodesic_attention`
applies. The finite distances are shifted by their row minimum before scaling,
which is the same maximum shift without ever forming an overflowing absolute
score. The aggregation of `V` is a
Euclidean weighted sum, which is what the AD bridge also computes — only the SCORES
carry the metric. Note this is still **not** the composition of ids 844/846/845,
whose scalings differ.

```scheme
(geodesic-attention-forward (make-tensor '(2 3) 0.1)
                            (make-tensor '(2 3) 0.2)
                            (make-tensor '(2 3) 0.3) -1.0)
```

---

## Curvature control

These four ops treat curvature as a learnable scalar on the manifold handle.

### Curvature derivatives

`curvature-gradient` (852), `curvature-hessian` (855) and
`adaptive-curvature-step` (856) share one objective, and it is named rather than
implied:

    L(K) = Σ_p d_K(x_p, y_p)

the total geodesic distance over the batch of point pairs their second argument
packs, as a function of the **sectional curvature K** with the points held fixed.
`curvature-gradient` returns `L'(K)`, `curvature-hessian` returns `L''(K)`, and
`adaptive-curvature-step` takes a damped Newton step on `L`. All three read `K`
from the manifold, so `set-curvature!` moves the point of evaluation.

Both derivatives are the **exact closed forms** of the distance in `K`
(`eshkol_rm_distance_dK` in `inc/eshkol/backend/riemannian_core.h`), not central
differences: there is no step size to choose and no truncation error to bound. On
the ball, with `c = −K`, `a = |x|²`, `b = |y|²`, `D = |x−y|²`,

    P(c) = (1−ca)(1−cb),   Q = c/P,   A = 1 + 2DQ,   d = arccosh(A)/√c

and `L'`, `L''` are the chain rule on that composition. Coincident points are
handled exactly rather than in the limit: `D = 0` makes `d` identically zero in
`c`, so both derivatives are exactly `0`.

**The batch.** Points are packed consecutively — `(x₀, y₀, x₁, y₁, …)`, each of
the manifold's `dim` coordinates. Only the element count matters, so any tensor
shape will do, but it must be a **positive multiple of `2·dim`**: a length that is
not a whole number of pairs is a shape failure (`()`), not a batch with a partial
pair silently dropped.

**Why `K > 0` is reparametrised.** On the sphere, a point of radius `1/√K` is
*not* a point of a sphere of a different radius, so "hold the points fixed and
vary `K`" is not a curve in any single manifold. The family these ops
differentiate on the spherical branch holds each pair at **fixed angular
position** and lets the radius follow `K`: with `θ = arccos(⟨x,y⟩/R²)` fixed,
`d = θ·K^(−1/2)`, so `d' = −θK^(−3/2)/2` and `d'' = 3θK^(−5/2)/4`.

The spherical derivative is returned only when both analytic values are
representable in f64. At the positive subnormal curvature floor, a nonzero
fixed angle can make either derivative exceed f64 even though the distance is
finite; that case raises rather than returning a successful infinity. The zero
angle limit is `d' = d'' = 0`.

**Why `K = 0` refuses.** The Poincaré ball of curvature `K` has conformal factor
`λ_x = 2/(1 − c|x|²)`, so its distance tends to `2|x−y|` as `K → 0⁻` — twice what
the Euclidean branch of `geodesic-distance` returns at `K = 0` exactly — while the
spherical branch diverges as `K → 0⁺`. The family is genuinely discontinuous
there, so no number is the derivative, and returning one would be exactly the
plausible-wrong-number case this surface exists to exclude.

### `(curvature-gradient m pairs)` — id 852

Returns `dL/dK` at `m`'s stored curvature, where `L` is the
[curvature objective](#curvature-derivatives)

    L(K) = Σ_p d_K(x_p, y_p)

over the point pairs `pairs` packs. Exact closed form, not a difference quotient.
`()` if `m` is not a manifold, `pairs` is not a tensor, or `pairs`'s element count
is not a positive multiple of `2·dim`. **Raises** at `K = 0` and whenever a
supplied point is not a point of the manifold at `K`.

This used to return the plain **sum** of its second argument's elements — a
number with the type and the magnitude a gradient would have, and the derivative
of nothing.

```scheme
;; the origin and (1/2, 0) on the ball of curvature K = −1:
;;   dL/dK = −(2/3 − artanh(1/2)) = −0.1173605223326118
(define pairs (make-tensor '(4) 0.0))   ; [0 0 | 0.5 0]
(display (curvature-gradient (make-hyperbolic-manifold 2 -1.0) pairs))
(newline)
```

### `(transition-geometry! m target rate)` — id 853

Moves the stored curvature one exponential step toward `target`:
`K ← K + rate·(target − K)`. Mutates `m` and returns the **new curvature** as a
float, not the manifold. `()` if `m` is not a manifold.

```scheme
(define h (make-hyperbolic-manifold 2 -1.0))
(display (transition-geometry! h 0.0 0.1)) (newline)
```

### `(manifold-interpolate m1 m2 t)` — id 854

Returns the interpolated **curvature** `(1−t)·K₁ + t·K₂` as a float. It does not
build a manifold, and it ignores both dimensions. `()` if either argument is not a
manifold.

```scheme
(display (manifold-interpolate (make-hyperbolic-manifold 2 -1.0)
                               (make-spherical-manifold 2) 0.5))
(newline)
```

### `(curvature-hessian m pairs)` — id 855

Returns `d²L/dK²` of the same objective, on the same argument shapes, with the
same failure and refusal conditions as `curvature-gradient`. Exact closed form.

This used to return the constant `0.0` for every manifold and every argument —
which is the assertion that the objective is **affine in K**, made without
examining an objective, and false for the geodesic distance at every curvature.

```scheme
;; the same pair: d²L/dK² = 0.0461814387233047
(display (curvature-hessian (make-hyperbolic-manifold 2 -1.0) pairs))
(newline)
```

### `(adaptive-curvature-step m pairs)` — id 856

Applies one **backtracked damped Newton step** on the curvature objective,
mutating `m`, and returns the manifold:

    Δ = L'(K) / max(L''(K), 1e-8)
    K ← K − t·Δ,  t = 1, ½, ¼, … (at most 32 halvings)

`t` is halved until the candidate `K` satisfies all three of:

1. it has the **same sign** as the current `K` — crossing `K = 0` would change
   which geometry the manifold is, not merely how curved it is;
2. every supplied point is still a point of the manifold at it — on the ball,
   `c·|x|² < 1` can fail as `|K|` grows;
3. it does **not increase** `L`.

On the spherical branch, each trial first rescales every supplied point by
`sqrt(K / K_trial)`. This keeps its angular position fixed while the sphere
radius changes, which is the documented objective family; evaluating the old
coordinates directly would test points from the old sphere and reject the
correct trial.

The floor on `L''` is Levenberg damping: where the objective is flat or concave
in `K` the Newton quotient is meaningless or points uphill, and flooring the
denominator turns the step back into a descent step without inventing a
direction. **Raises** when no admissible `t` exists after 32 halvings — that is
not a near-converged state (the step is down by 2⁻³²), it is the absence of a
step, and the op says so rather than moving `K` somewhere it cannot justify. Also
raises at `K = 0`, and returns `()` on the same shape failures as the two ops
above.

This used to be `K ← K − 0.01·Σgrad`: a fixed rate applied to a sum that was not
a gradient, so nothing about the step adapted to anything.

```scheme
(define h (make-hyperbolic-manifold 2 -1.0))
;; the full Newton step lands at K = +1.54, which would change the geometry, so
;; the first two trials are rejected and t = 1/4 is taken:
(adaptive-curvature-step h pairs)
(display (get-curvature h)) (newline)   ; -0.36467699157352285
```

---

## Name collisions with `core.manifold`

Six spellings exist in both surfaces, and they are **not** the same functions:

| Spelling | This page (VM builtin) | [`core.manifold`](manifold.md) (Scheme) |
|---|---|---|
| `make-euclidean-manifold` | arity 1, opaque handle | arity 1, `#(type dim)` vector |
| `make-hyperbolic-manifold` | **arity 2** — `(dim curvature)` | **arity 1** — `(dim)`, K fixed at −1 |
| `make-spherical-manifold` | arity 1, opaque handle | arity 1, `#(type dim)` vector |
| `manifold-exp-map` / `-log-map` | tensors; closed-form Möbius / great-circle | Scheme vectors; closed-form Möbius / great-circle |
| `manifold-distance` | tensors; Poincaré `arccosh` / spherical `acos` | Scheme vectors; Poincaré `arccosh` / spherical `acos` |
| `manifold-type` / `manifold-dimension` | integer type code | symbol type, integer dimension |

Which one a call reaches depends on whether `core.manifold` has been loaded.
`core.manifold` is auto-loaded by `(require stdlib)` (it is in `lib/stdlib.esk`'s
`require` chain), and the VM reaches builtins through ordinary global-variable
lookup rather than a compiler intercept — `lib/backend/vm_compiler.c` says so
explicitly: *"Any function in the `BUILTINS[]` array in `eshkol_vm.c` is accessible
via normal global variable lookup and should NOT appear in this compiler"*, precisely
so that *"user-defined functions with the same name"* are not silently bypassed. A
`define` of one of these six names therefore rebinds the global, and a module that
pulls in the stdlib gets the Scheme implementations. **Do not rely on that to resolve
the ambiguity for you** — pick one surface per program deliberately, and note that
the two disagree on arity for `make-hyperbolic-manifold`, so a program that reaches
the wrong one fails at the call rather than quietly.

`core.manifold` is the better default when portability across engines is what you
need: it runs on native AOT and REPL/JIT as well as the VM, and its points are
ordinary Scheme vectors. Both surfaces now compute the same closed forms, so the
choice is about engine reach and data representation rather than about correctness —
note that `core.manifold` fixes K at −1 for the hyperbolic case while these builtins
take K as an argument. Reach for these builtins when you need the tensor-shaped
surface, the
Lie-group ops (`so3-*`, `se3-*`, `quaternion-mul`), the differential forms, the
optimizers, geodesic attention, or the gated `frechet-mean` — none of which
`core.manifold` provides.

## Automatic differentiation

None of these builtins is differentiable through Eshkol's AD operators: they are VM
natives, and AD is a native-engine facility. The qLLM bridge
(`lib/bridge/qllm_bridge.cpp`) is the differentiable route to the same geometry — and
it is now the same geometry in the strict sense: the closed forms these opcodes
compute live in `inc/eshkol/backend/riemannian_core.h` and are the ones the bridge's
`ad_hyperbolic_distance`, `ad_poincare_exp_map`, `ad_poincare_log_map` and
`ad_geodesic_attention` evaluate, so a value computed on the VM and a value computed
on the tape agree **at K < 0** — see
[What the shipped build computes](#what-the-shipped-build-computes) for why the
bridge's `c = |curvature|` conversion makes K ≥ 0 disagree, and note that
`ad_poincare_log_map` additionally **clamps** `√c|u|` to `1 − 1e−12` where this
page's `log` now computes the finite value that exists there. The
the exact backwards it registers are documented in
[`../ad/architecture.md`](../ad/architecture.md) and
[`../ad/support-matrix.md`](../ad/support-matrix.md). `frechet-mean`'s forward pass is
literally shared between the two — the same
`inc/eshkol/backend/frechet_mean_core.h` — so that the opcode and the derivative can
never disagree about what the mean is.

## See also

- [`manifold.md`](manifold.md) — the pure-Scheme `core.manifold` module.
- [`../ad/INDEX.md`](../ad/INDEX.md) — automatic differentiation reference.
- [`../ad/tape.md`](../ad/tape.md) — the explicit reverse-mode tape builtins.
- [`INDEX.md`](INDEX.md) — the stdlib module map.
