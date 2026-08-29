# Geometric builtins — Riemannian manifolds, Lie groups, forms and geodesic attention

**Name table**: [`lib/backend/eshkol_vm.c`](../../../lib/backend/eshkol_vm.c) `BUILTINS[]`, native ids 804-861.
**Implementation**: [`lib/backend/vm_geometric.c`](../../../lib/backend/vm_geometric.c).
**Weighted Fréchet mean core**: [`inc/eshkol/backend/frechet_mean_core.h`](../../../inc/eshkol/backend/frechet_mean_core.h).
**Surface record**: `tests/coverage/language_surface.json`, category `geometry`.
**Regression**: [`tests/vm/geometric_surface_regression.esk`](../../../tests/vm/geometric_surface_regression.esk), run by `scripts/run_vm_surface_tests.sh`.

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

`vm_geometric.c` has two dispatch bodies, selected at compile time by
`ESHKOL_GEOMETRIC_ENABLED`:

- **`#if !defined(ESHKOL_GEOMETRIC_ENABLED)`** — a self-contained,
  constant-curvature implementation that allocates its manifolds in the VM arena.
- **`#if defined(ESHKOL_GEOMETRIC_ENABLED)`** — calls out to the
  `semiclassical_qllm` library (`<semiclassical_qllm/manifold.h>` and friends).

**No target in this repository defines `ESHKOL_GEOMETRIC_ENABLED`.** `grep -rn
ESHKOL_GEOMETRIC_ENABLED` finds hits only inside `vm_geometric.c` itself, so every
shipped build — every CI lane, every release binary, the WASM playground — takes the
portable path, and that is the path documented here. The linked-library path is an
out-of-tree opt-in and is *not* covered by this page or by any gate in this
repository.

The consequence is worth stating plainly, because the names promise more than the
default build delivers. In the portable path several ops are **flat (Euclidean)
approximations that ignore the curvature argument they accept**: `hyperbolic-exp-map`
is vector addition, `hyperbolic-log-map` is subtraction, `geodesic-distance` and
`poincare-distance` are the L2 distance, `mobius-add` is addition,
`parallel-transport` is the identity, `manifold-project` and `riemannian-grad` are
copies, `exterior-derivative` returns zeros and `hodge-star` returns its input. Each
entry below says exactly what its op computes. The one op that is genuinely
Riemannian in the portable path is [`frechet-mean`](#frechet-mean-points-weights-curvature--id-817), which runs
a real Karcher iteration in f64 behind a stationarity gate.

## Error convention

With one exception, these ops **do not raise**. An argument of the wrong type, a
shape mismatch, a non-positive dimension or an allocation failure pushes the empty
list `()` and execution continues; an unhandled id falls through to a `default:` arm
that pops `vm_geometric_arity(fid)` arguments and pushes `()`, so the stack stays
balanced either way. Check for `(null? result)` if you need to distinguish a failure
from a value.

The exceptions are `frechet-mean`, and the portable VM's spherical
`great-circle-distance` and `spherical-log` operations at their cut locus. They
raise a catchable Scheme condition (via `vm_raise_error_msg`, so `guard` can
catch it) rather than return a non-stationary mean or a finite-looking answer
where the spherical logarithm is undefined. The reasoning is in the source: a mean that has not reached stationarity is
exactly the input whose implicit derivative is a plausible wrong gradient.

## Data model

- **Points, tangent vectors, forms, quaternions, twists, poses, Q/K/V matrices** are
  all **tensors** (`(make-tensor '(shape) fill)`, or any op that returns one). They
  are *not* Scheme vectors — `vm_get_tensor` returns NULL for a non-tensor, and the
  op then pushes `()`.
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
| `hyperbolic-exp-map` / `manifold-exp-map` | 809 | 3 | tensor | `base + tangent` (K discarded) |
| `hyperbolic-log-map` / `manifold-log-map` | 810 | 3 | tensor | `point - base` (K discarded) |
| `geodesic-distance` / `manifold-distance` | 811 | 3 | float | L2 distance (K discarded) |
| `parallel-transport` / `manifold-parallel-transport` | 812 | 4 | tensor | copy of `v` (identity transport) |
| `manifold-project` | 813 | 2 | tensor | copy of `x` |
| `mobius-add` | 814 | 3 | tensor | `x + y` (K discarded) |
| `mobius-scalar-mul` | 815 | 3 | tensor | `r * x` (K discarded) |
| `poincare-distance` | 816 | 3 | float | L2 distance (same case as 811) |
| `frechet-mean` | 817 | 3 | tensor | **real** weighted Karcher mean, gated; raises |
| `great-circle-distance` | 819 | 2 | float | `acos` of the clamped normalised dot; raises at antipodes |
| `slerp` | 820 | 3 | tensor | normalised `(1-t)x + t y` |
| `spherical-exp` / `spherical-exp-map` | 821 | 2 | tensor | normalised `base + tangent` |
| `spherical-log` / `spherical-log-map` | 822 | 2 | tensor | `point - base`; raises at antipodes |
| `spherical-project` | 823 | 1 | tensor | L2-normalised copy |
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
| `exterior-derivative` | 835 | 1 | tensor | zeros of the input's shape |
| `hodge-star` | 836 | 2 | tensor | copy of the form (metric discarded) |
| `interior-product` | 837 | 2 | tensor | shape `(1)` holding the dot product |
| `pullback` | 838 | 2 | tensor | `formᵀ · jacobian`, shape `(cols)` |
| `riemannian-sgd-step` | 839 | 4 | tensor | `point - lr · grad` (K discarded) |
| `riemannian-adam-step` | 840 | 6 | tensor | Euclidean Adam, **implicit pooled state** |
| `riemannian-grad` | 841 | 3 | tensor | copy of the Euclidean gradient |
| `retraction` | 842 | 3 | tensor | `base + tangent` (same case as 809) |
| `vector-transport` | 843 | 4 | tensor | copy of `v` (same case as 812) |
| `geodesic-attention-scores` | 844 | 3 | tensor | `-‖q_i - k_j‖`, shape `(nq nk)` |
| `geodesic-attention-values` | 845 | 3 | tensor | score-weighted average of `V`'s rows |
| `curvature-softmax` | 846 | 2 | tensor | softmax scaled by `1/√|K|` |
| `geodesic-attention-forward` | 847 | 4 | tensor | `exp(-distance)` attention, shape `(nq vdim)` |
| `set-curvature!` | 850 | 2 | manifold | mutates K, returns the manifold |
| `get-curvature` | 851 | 1 | float | stored K (same case as 808) |
| `curvature-gradient` | 852 | 2 | float | sum of the gradient's elements |
| `transition-geometry!` | 853 | 3 | float | K ← K + rate·(target − K); returns new K |
| `manifold-interpolate` | 854 | 3 | float | `(1-t)·K₁ + t·K₂` — a **curvature**, not a manifold |
| `curvature-hessian` | 855 | 2 | float | `0.0` |
| `adaptive-curvature-step` | 856 | 2 | manifold | K ← K − 0.01·Σgrad; returns the manifold |
| `manifold-type` | 857 | 1 | int | 0/1/2/3 |
| `manifold-dim` / `manifold-dimension` | 858 | 1 | int | stored dim |
| `manifold-destroy!` | 859 | 1 | `()` | invalidates the handle |
| `make-riemannian-adam-state` | 860 | 1 | state | zeroed moments shaped like the point |
| `riemannian-adam-step!` | 861 | 7 | tensor | Euclidean Adam with an **explicit** state |

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

Every op in this group accepts a trailing `curvature` argument and, in the portable
build, **discards it**.

### `(hyperbolic-exp-map base tangent curvature)` / `(manifold-exp-map …)` / `(retraction base tangent curvature)` — id 809 / 842

`base`, `tangent` tensors of equal total size; `curvature` real. Returns a fresh
tensor `base + tangent`, or `()` on a shape mismatch. `retraction` is a separate
name on the same case.

```scheme
(define t (make-tensor '(2) 0.0))
(hyperbolic-exp-map t t -1.0)
```

### `(hyperbolic-log-map base point curvature)` / `(manifold-log-map …)` — id 810

Returns `point - base`, the inverse of id 809's flat exp map.

```scheme
(define t (make-tensor '(2) 0.0))
(hyperbolic-log-map t t -1.0)
```

### `(geodesic-distance x y curvature)` / `(manifold-distance …)` / `(poincare-distance x y curvature)` — id 811 / 816

Returns the Euclidean L2 distance `‖x − y‖` as a float. `()` unless both arguments
are tensors of the same total size. `poincare-distance` is a second name on the same
case, so in the portable build it is **not** the Poincaré metric.

```scheme
(define a (make-tensor '(2) 0.0))
(define b (make-tensor '(2) 1.0))
(display (geodesic-distance a b -1.0)) (newline)
```

### `(parallel-transport x y v curvature)` / `(manifold-parallel-transport …)` / `(vector-transport x y v curvature)` — id 812 / 843

Pops and discards `x`, `y` and `curvature` and returns a copy of the tangent vector
`v` — the identity transport.

```scheme
(define t (make-tensor '(2) 1.0))
(parallel-transport t t t -1.0)
```

### `(manifold-project x curvature)` — id 813

Returns a copy of `x`. No projection is performed in the portable build.

```scheme
(manifold-project (make-tensor '(2) 3.0) -1.0)
```

### `(mobius-add x y curvature)` — id 814

The gyrovector addition of the Poincaré ball model. In the portable build it returns
`x + y`; the curvature is discarded, so it degenerates to the Euclidean group
operation.

```scheme
(define x (make-tensor '(2) 0.1))
(define y (make-tensor '(2) 0.2))
(mobius-add x y -1.0)
```

### `(mobius-scalar-mul r x curvature)` — id 815

Note the argument order: the **scalar comes first**. Returns `r · x`.

```scheme
(mobius-scalar-mul 0.5 (make-tensor '(2) 0.4) -1.0)
```

### `(frechet-mean points weights curvature)` — id 817

The one genuinely Riemannian op in the portable path, and the only one that raises.

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
`0.0` if either norm is zero. An antipodal pair raises a named condition because
the shortest geodesic is not unique. `()` on a size mismatch.

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

Two tensors. Returns `normalize(base + tangent)`.

```scheme
(spherical-exp (make-tensor '(3) 1.0) (make-tensor '(3) 0.1))
```

### `(spherical-log base point)` / `(spherical-log-map …)` — id 822

Returns `point − base`. An antipodal pair raises a named condition because the
spherical logarithm is not single-valued at the cut locus. Shares a case with id
810 but takes **two** arguments, not three — there is no curvature argument.

```scheme
(spherical-log (make-tensor '(3) 1.0) (make-tensor '(3) 2.0))
```

### `(spherical-project x)` — id 823

Returns an L2-normalised copy of `x` (unchanged if its norm is zero).

```scheme
(spherical-project (make-tensor '(3) 2.0))
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

### `(wedge-product a b)` — id 834

Two tensors. With `n = min(total_a, total_b)`, returns a tensor of shape
`(n(n−1)/2)` holding the 2×2 minors `aᵢbⱼ − aⱼbᵢ` for `i < j`, in row-major `(i, j)`
order. For `n ≤ 1` the result has shape `(1)` and is zero. `()` if either argument
is not a tensor.

```scheme
(wedge-product (make-tensor '(3) 1.0) (make-tensor '(3) 2.0))
```

### `(exterior-derivative form)` — id 835

Returns a **zero** tensor of the input's shape. The portable build carries no
coordinate information from which to differentiate a form, so `d` is the zero map
here — not an identity, and not an error.

```scheme
(exterior-derivative (make-tensor '(3) 1.0))
```

### `(hodge-star form metric)` — id 836

Pops and discards `metric` and returns a copy of `form`.

```scheme
(hodge-star (make-tensor '(3) 1.0) (make-tensor '(3 3) 0.0))
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

All three step ops accept a trailing curvature argument and discard it: the update is
Euclidean, with no retraction back onto the manifold.

### `(riemannian-sgd-step point gradient lr curvature)` — id 839

Returns `point − lr · gradient` as a fresh tensor. `()` on a shape mismatch.

```scheme
(define p (make-tensor '(4) 1.0))
(define g (make-tensor '(4) 0.5))
(riemannian-sgd-step p g 0.01 -1.0)
```

### `(riemannian-adam-step point gradient lr beta1 beta2 curvature)` — id 840

Bias-corrected Adam with `ε = 1e-8`. `beta1` outside `[0, 1)` falls back to `0.9`,
`beta2` outside `[0, 1)` to `0.999`, and a negative `lr` is negated.

The optimizer state is **implicit and pooled**. `vm_default_riemannian_adam_state`
keeps 16 VM-lifetime slots and hands back the first whose shape matches `point`,
creating one in the first empty slot otherwise — and **overwriting slot 0 when all 16
are occupied by other shapes**. Two independent optimisation loops over
same-shaped parameters therefore share one moment estimate. Use
`make-riemannian-adam-state` and `riemannian-adam-step!` when you need the state to
belong to one loop.

```scheme
(define p (make-tensor '(4) 1.0))
(define g (make-tensor '(4) 0.5))
(riemannian-adam-step p g 0.01 0.9 0.999 -1.0)
```

### `(make-riemannian-adam-state point)` — id 860

Allocates a zeroed state (first and second moment buffers shaped like `point`, step
counter 0) and returns the handle. This state is **region-scoped**, unlike the pooled
states id 840 creates, which are VM-lifetime. `()` if `point` is not a usable tensor.

```scheme
(define st (make-riemannian-adam-state (make-tensor '(4) 1.0)))
```

### `(riemannian-adam-step! state point gradient lr beta1 beta2 curvature)` — id 861

The same update as id 840 against an explicit `state`. Despite the `!`, the returned
value is a **new** tensor — `point` is not mutated; what is mutated is `state`'s
moment buffers and step counter. `()` if the state's shape does not match `point`.

```scheme
(define p  (make-tensor '(4) 1.0))
(define g  (make-tensor '(4) 0.5))
(define st (make-riemannian-adam-state p))
(riemannian-adam-step! st p g 0.01 0.9 0.999 -1.0)
```

### `(riemannian-grad euclidean-grad point curvature)` — id 841

Pops and discards `point` and `curvature` and returns a copy of `euclidean-grad`. No
tangent-space projection is applied in the portable build.

```scheme
(riemannian-grad (make-tensor '(4) 0.5) (make-tensor '(4) 1.0) -1.0)
```

---

## Geodesic attention

### `(geodesic-attention-scores Q K curvature)` — id 844

`Q` and `K` tensors. A rank-≥2 tensor is read as `(n, d)`; a rank-1 tensor is read as
a single row of dimension `total`. The two feature dimensions must agree. Returns a
tensor of shape `(nq nk)` whose entries are the **negated** Euclidean distances
`−‖qᵢ − kⱼ‖`. `()` if either argument is not a tensor or the dimensions disagree.

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
`(nq vdim)` where row `i` is the average of `V`'s rows weighted by
`exp(−‖qᵢ − kⱼ‖)`, normalised by the weight sum. Note this is **not** the composition
of ids 844/846/845: the weights are `exp(−d)` with no curvature scaling.

```scheme
(geodesic-attention-forward (make-tensor '(2 3) 0.1)
                            (make-tensor '(2 3) 0.2)
                            (make-tensor '(2 3) 0.3) -1.0)
```

---

## Curvature control

These four ops treat curvature as a learnable scalar on the manifold handle.

### `(curvature-gradient m loss-grad)` — id 852

Returns the plain sum of `loss-grad`'s elements as a float. `()` if `m` is not a
manifold or `loss-grad` is not a tensor. This is a placeholder reduction, not a
derivative of anything with respect to `K`.

```scheme
(display (curvature-gradient (make-hyperbolic-manifold 2 -1.0)
                             (make-tensor '(4) 0.25)))
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

### `(curvature-hessian m grad)` — id 855

Pops and discards `grad` and returns `0.0` for any manifold, `()` otherwise. A
constant, not a measurement.

```scheme
(display (curvature-hessian (make-hyperbolic-manifold 2 -1.0)
                            (make-tensor '(4) 0.25)))
(newline)
```

### `(adaptive-curvature-step m grad)` — id 856

Applies one fixed-rate curvature update `K ← K − 0.01·Σgrad`, mutating `m`, and
returns the manifold. `()` if `m` is not a manifold or `grad` is not a tensor.

```scheme
(define h (make-hyperbolic-manifold 2 -1.0))
(adaptive-curvature-step h (make-tensor '(4) 0.25))
(display (get-curvature h)) (newline)
```

---

## Name collisions with `core.manifold`

Six spellings exist in both surfaces, and they are **not** the same functions:

| Spelling | This page (VM builtin) | [`core.manifold`](manifold.md) (Scheme) |
|---|---|---|
| `make-euclidean-manifold` | arity 1, opaque handle | arity 1, `#(type dim)` vector |
| `make-hyperbolic-manifold` | **arity 2** — `(dim curvature)` | **arity 1** — `(dim)`, K fixed at −1 |
| `make-spherical-manifold` | arity 1, opaque handle | arity 1, `#(type dim)` vector |
| `manifold-exp-map` / `-log-map` | tensors; flat in the portable build | Scheme vectors; closed-form Möbius / great-circle |
| `manifold-distance` | tensors; L2 in the portable build | Scheme vectors; Poincaré `arccosh` / spherical `acos` |
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

`core.manifold` is the better default: it is engine-independent (it runs on native
AOT and REPL/JIT as well as the VM), its points are ordinary Scheme vectors, and its
maps and distances are genuine closed forms rather than the flat approximations
above. Reach for these builtins when you need the tensor-shaped surface, the
Lie-group ops (`so3-*`, `se3-*`, `quaternion-mul`), the differential forms, the
optimizers, geodesic attention, or the gated `frechet-mean` — none of which
`core.manifold` provides.

## Automatic differentiation

None of these builtins is differentiable through Eshkol's AD operators: they are VM
natives, and AD is a native-engine facility. The qLLM bridge
(`lib/bridge/qllm_bridge.cpp`) is the differentiable route to the same geometry, and
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
