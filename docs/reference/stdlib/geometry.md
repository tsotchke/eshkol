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

**The portable path is the shipped implementation.** `ESHKOL_GEOMETRIC_ENABLED` is
defined only when the `ESHKOL_GEOMETRIC_QLLM` CMake option is turned ON, which
requires `ESHKOL_QLLM_ROOT` to point at an installed `libsemiclassical_qllm` and
`FATAL_ERROR`s at configure time if the headers are not there. It is OFF by default,
so every CI lane, every release binary and the WASM playground take the portable
path, and that is the path documented here. The linked-library path is an
out-of-tree opt-in and is *not* covered by this page or by any gate in this
repository — and, measured against a current `libsemiclassical_qllm`, it does not
currently compile: the arities of `qllm_hyperbolic_exp_map` and its siblings have
moved and the SO(3)/SE(3) arms name types the library no longer declares. Bringing
it back is its own change against the current qLLM ABI.

The portable path computes the **closed forms of constant-curvature geometry in
f64**, in `inc/eshkol/backend/riemannian_core.h`. It is not an approximation of the
linked-library path: it is the same mathematics the AD bridge computes
(`lib/bridge/qllm_bridge.cpp`: `ad_hyperbolic_distance`, `ad_poincare_exp_map`,
`ad_poincare_log_map`, `ad_geodesic_attention`), so the VM engine and the AD tape
agree on what these operations mean.

This used not to be true, and the difference is worth stating because a reader of an
older build's output needs to know. Before `SW-73` these ops computed their **flat
(Euclidean) counterparts and discarded the curvature argument they accept**:
`hyperbolic-exp-map` was vector addition, `hyperbolic-log-map` subtraction,
`geodesic-distance` and `poincare-distance` the L2 distance, `mobius-add` addition,
`parallel-transport` and `riemannian-grad` the identity. At K = 0 those forms are
exactly right — every op below reduces to its flat form there — and at every other
curvature they were wrong without bound: two points of the unit ball a tenth of a
unit inside the boundary are 1.8 apart in L2 and 5.8888779583328814 apart in the
metric the name promises. `tests/vm/geometric_riemannian_surface_regression.esk`
pins each of these against its closed form, with every assertion chosen so that the
flat answer fails it.

Two ops are **not implemented on the VM engine** and raise rather than return a
plausible value: `exterior-derivative` and `hodge-star`. See
[Engine availability per builtin](#engine-availability-per-builtin).

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
- a point that is not **on** the sphere of radius `1/√K`, for any op taking K > 0
  (`manifold-project` is the op that moves it there);
- `log`, when the two points are too far apart in hyperbolic distance for the
  ambient ball coordinates to separate them — roughly 19 units, reachable from two
  points each strictly inside the ball. No finite log exists there, and clamping the
  argument would return a fabricated magnitude;
- `mobius-add` and `mobius-scalar-mul` with K > 0: they are the gyrogroup operations
  of the ball and are not defined on the sphere;
- `frechet-mean`, when the Karcher iteration has not reached stationarity. A mean
  that has not is exactly the input whose implicit derivative is a plausible wrong
  gradient;
- `exterior-derivative` and `hodge-star`, always: they are not implemented on the VM
  engine.

## Engine availability per builtin

Sixty of the sixty-two names execute on the bytecode VM. Two do not, and say so:

| Name | Id | Status on the VM engine |
|---|---:|---|
| `exterior-derivative` | 835 | **Raises `not implemented on the VM engine`.** The argument is a form's coefficient array at a single point, which carries no derivative information; computing `d` needs the coefficients as differentiable functions of position. It previously returned zeros of the input's shape, which asserts that every form handed to it is closed. |
| `hodge-star` | 836 | **Raises `not implemented on the VM engine`.** The star of a k-form depends on k and on the dimension, and a flat coefficient array records neither, so the op cannot tell which duality is being asked for. It previously returned its argument unchanged, which asserts that the star is trivial. |

Neither has a native AOT or REPL/JIT implementation either — see
[Engine availability](#engine-availability) above, which applies to all 62 names.

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
| `exterior-derivative` | 835 | 1 | tensor | **not implemented on the VM engine — raises** |
| `hodge-star` | 836 | 2 | tensor | **not implemented on the VM engine — raises** |
| `interior-product` | 837 | 2 | tensor | shape `(1)` holding the dot product |
| `pullback` | 838 | 2 | tensor | `formᵀ · jacobian`, shape `(cols)` |
| `riemannian-sgd-step` | 839 | 4 | tensor | `exp_point(−lr·grad)`; `point − lr·grad` at K=0 |
| `riemannian-adam-step` | 840 | 6 | tensor | Adam delta retracted with `exp`, moment transported, **implicit pooled state** |
| `riemannian-grad` | 841 | 3 | tensor | `((1−c‖x‖²)²/4)·grad` (K<0) / tangent projection (K>0) / a copy (K=0) |
| `retraction` | 842 | 3 | tensor | the exponential map (same op as 809) |
| `vector-transport` | 843 | 4 | tensor | parallel transport (same op as 812) |
| `geodesic-attention-scores` | 844 | 3 | tensor | `−d_K(q_i, k_j)`, shape `(nq nk)` |
| `geodesic-attention-values` | 845 | 3 | tensor | score-weighted average of `V`'s rows |
| `curvature-softmax` | 846 | 2 | tensor | softmax scaled by `1/√|K|` |
| `geodesic-attention-forward` | 847 | 4 | tensor | softmax over `−d_K/(√c·√dim)` then a weighted sum of `V`, shape `(nq vdim)` |
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
| `riemannian-adam-step!` | 861 | 7 | tensor | same as 840 with an **explicit** state |

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

- `K < 0`: `(2/(√c·λ))·artanh(√c‖u‖)·u/‖u‖` with `u = (−base) ⊕_c point`.
- `K = 0`: `point − base`, exactly.
- `K > 0`: `θR·u/‖u‖` with `u = point − cos θ·base`, `θ = acos(⟨base,point⟩/R²)`.

Raises when no finite log exists — see [Error convention](#error-convention).

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

- `K < 0`: `arccosh(1 + 2c‖x−y‖²/((1−c‖x‖²)(1−c‖y‖²)))/√c`, the same closed form
  `ad_hyperbolic_distance` computes on the AD tape.
- `K = 0`: `‖x − y‖`, exactly.
- `K > 0`: `R·acos(⟨x,y⟩/R²)`, the great-circle distance.

The hyperbolic distance is not bounded by the ball diameter the way the chord is: it
diverges as either point approaches the boundary.

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

- `K < 0`: `(λ_x/λ_y)·gyr[y, −x]v`, the gyration expanded through Möbius addition as
  `gyr[u,w]z = (−(u ⊕ w)) ⊕ (u ⊕ (w ⊕ z))`.
- `K = 0`: the identity, exactly.
- `K > 0`: the component of `v` along the geodesic direction rotated by the arc
  angle.

Transport is an isometry of the Riemannian metric — `λ_y‖P v‖ = λ_x‖v‖` — even
though it changes the ambient components. Transport from a point to itself is the
identity.

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
commutativity, and it is what `parallel-transport` is built from. At `K = 0` it is
`x + y`, exactly. It **raises for `K > 0`**: Möbius addition is the gyrogroup
operation of the ball and is not defined on the sphere.

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
`K > 0`**, for the same reason `mobius-add` does.

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

**Not implemented on the VM engine.** Raises a catchable condition naming itself.

`form` is a coefficient array evaluated at a single point, and `d` is a derivative:
it cannot be computed from the values of the coefficients, only from the coefficients
as differentiable functions of position. Nothing in this op's arguments carries that.

It used to return a zero tensor of the input's shape, which is not "no information
available" — it is the assertion that `d(form) = 0`, i.e. that every form handed to
it is closed, made without examining one.

```scheme
(guard (e (#t 'not-implemented))
  (exterior-derivative (make-tensor '(3) 1.0)))
```

### `(hodge-star form metric)` — id 836

**Not implemented on the VM engine.** Raises a catchable condition naming itself.

The star of a k-form depends on `k` and on the ambient dimension. A flat coefficient
array records neither, so the op has no way to tell which duality is being asked for.

It used to return its argument unchanged, which is the Hodge star only for a
self-dual middle-degree form in a Euclidean metric — the identity map presented under
the name of a duality.

```scheme
(guard (e (#t 'not-implemented))
  (hodge-star (make-tensor '(3) 1.0) (make-tensor '(3 3) 0.0)))
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

Bias-corrected Adam with `ε = 1e-8`. `beta1` outside `[0, 1)` falls back to `0.9`,
`beta2` outside `[0, 1)` to `0.999`, and a negative `lr` is negated. The Adam delta is
formed in the tangent space, retracted with `exp_point`, and the **first moment is
parallel-transported** to the new point — it is a tangent vector at the old one and
is meaningless at the new one until it is moved. The second moment is a
per-coordinate scale rather than a tangent vector and is left as is, which is the
same choice `geoopt`'s `RiemannianAdam` makes.

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
applies. The maximum is subtracted before exponentiating. The aggregation of `V` is a
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
on the tape agree. The
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
