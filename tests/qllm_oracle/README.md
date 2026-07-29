# qLLM geometric gradient oracle

Exact reference gradients for qLLM's geometric primitives, computed with
Eshkol's reverse-mode automatic differentiation and exported as JSON golden
vectors for qLLM's fp32 C / torch / Metal tests to assert against.

## Why this exists

qLLM's geometric kernels live on the Poincaré ball, and finite differences
fail worst exactly where hyperbolic numerics are delicate. A central
difference carries a truncation term proportional to `h² · f'''` and a
round-off term proportional to `eps/h`. For `artanh`, `f' = 1/(1-t²)` and
`f''' ~ (1-t²)⁻³`, so as a point approaches the ball boundary there is no `h`
that makes both terms small. Worse, once the point sits within one FD step of
the domain bound the stencil straddles qLLM's `artanh` clamp and finite
differences return a confidently wrong number.

Exact reverse-mode AD has no step size, so it has no such regime. That makes
it the only admissible reference for the fp32 kernels — which is the whole
point of the instrument.

## What is exported

| JSON file | primitive | qLLM reference |
|---|---|---|
| `golden/poincare_project.json` | `poincare_project` — Riemannian gradient rescaling, plus the conformal factor and its gradient | `python/qllm/torch_geometric.py::poincare_project` |
| `golden/poincare_project_exact.json` | the same Jacobian entry recomputed in **exact rational arithmetic** | as above |
| `golden/poincare_retract.json` | `poincare_retract` — radial clip back inside the ball | `python/qllm/torch_geometric.py::poincare_retract` |
| `golden/sphere_project.json` | `sphere_project` — tangent projection `g − ⟨g,x⟩x` | `python/qllm/torch_geometric.py::sphere_project` |
| `golden/sphere_retract.json` | `sphere_retract` — `(x+step)/‖x+step‖` with the eps guard | `python/qllm/torch_geometric.py::sphere_retract` |
| `golden/poincare_log_map_origin.json` | `log_0` — the FD-hostile primitive | `src/model/grr.c::fast_log_map_origin` |
| `golden/poincare_exp_map_origin.json` | `exp_0` | `src/model/grr.c::fast_exp_map_origin` |
| `golden/poincare_exp_log_basepoint.json` | `exp_x` / `log_x` — standard Möbius (Ganea) formulas | Möbius/Ganea, qLLM-clamped `artanh` |
| `golden/sheaf_ee_step.json` | one explicit-Euler sheaf diffusion step, `S=3`, `d=2`, FIXED_ORIENT | `python/qllm/torch_geometric.py::sheaf_attention_ee` |

Every Jacobian is a full `m × d` matrix, assembled row by row as
`gradient(⟨e_k, F⟩)`. A consumer can contract it with any cotangent to get
the VJP its own backward produces, so the artifact serves both Jacobian and
VJP tests.

## Schema

```json
{
  "schema_version": 1,
  "primitive": "poincare_project",
  "eshkol_version": "1.3.3",
  "generator": "tests/qllm_oracle/poincare_project.esk",
  "reference": "python/qllm/torch_geometric.py::poincare_project",
  "formula": "conf = clamp_min(1 - c*|x|^2, eps); out = 0.25 * conf^2 * grad",
  "dtype": "float64",
  "ad_mode": "reverse",
  "cases": [
    {
      "id": "poincare_project.d2.c1.u0p999",
      "dim": 2,
      "curvature": 1.0,
      "eps": 1e-06,
      "radius_sqrt_c": 0.999,
      "conf_clamped": false,
      "inputs":         { "x": [...], "grad": [...] },
      "primal_outputs": { "out": [...], "conformal_factor": 0.001999 },
      "gradients": {
        "d_out_d_x":              [[...], [...]],
        "d_out_d_grad":           [[...], [...]],
        "d_conformal_factor_d_x": [...]
      },
      "fd_cross_check": { "best_step": 1e-07, "best_rel_diff": 4.6e-13, "fd_usable": true }
    }
  ]
}
```

Per-primitive case objects add their own fields (`clipped`, `artanh_clamped`,
`tanh_arg`, `active_edge_entries`, …). All floating point is binary64 printed
to 17 significant digits, so values round-trip exactly.

**JSON has no NaN literal.** A non-finite gradient entry is written as `null`
and flagged by a sibling boolean (e.g. `d_out_d_x_all_finite: false`). A
consumer must check the flag; see finding 3 below for a case where `null` is
the mathematically correct answer.

## Running and regenerating

```bash
bash scripts/run_qllm_oracle_tests.sh              # JIT + AOT lanes
bash scripts/run_qllm_oracle_tests.sh --no-aot     # JIT only (faster)
```

The suite is registered in `scripts/run_all_tests.sh`, so a plain
`bash scripts/run_all_tests.sh` runs it. It needs `build/eshkol-run`
(`cmake --build build --target eshkol-run stdlib`); override the build tree
with `BUILD_DIR=...`.

The JSON under `golden/` is a **committed build product**: qLLM consumes it as
a reference, so a change in the numbers must appear as a reviewable diff.
There is no RNG anywhere in the exporters — every seed point is hardcoded and
rescaled to an exact requested radius — so regeneration is deterministic and
`git diff tests/qllm_oracle/golden` after a run is the drift check.

The JIT (`-r`) and AOT lanes produce **byte-identical** golden vectors; this
was verified by diffing the two output trees. The JIT lane is the one that
writes `golden/`, so the two lanes cannot race on the same files.

### Version pinning for qLLM consumers

Pin on `eshkol_version` + `schema_version` in the JSON, not on a git SHA.
`schema_version` is bumped when field names or nesting change; `eshkol_version`
tracks `ESHKOL_VERSION` in the root `CMakeLists.txt`. If a consumer needs to
know exactly which compiler produced a vector, record the Eshkol commit
alongside its own test fixture — the exporters do not embed one, because the
values are a property of the mathematics and the f64 format, not of the build.

## The FD-vs-exact comparison

`poincare_maps.esk` prints a table sweeping `h` over
`{1e-2 … 1e-8}` for `d(log_0)_0/dy_0` at `c = 1`, `y = (u, 0)`. Reproduced
from a run on this tree:

| `u` = √c·‖y‖ | exact (reverse-mode AD) | best FD rel. err | worst FD rel. err | usable steps |
|---|---|---|---|---|
| 0.5 | 1.3333333333333333 | 1.5e-11 | 5.9e-05 | 6 / 7 |
| 0.9 | 5.2631578947368443 | 5.2e-11 | 2.7e-03 | 5 / 7 |
| 0.99 | 50.251256281406995 | 4.9e-10 | 4.98 | 4 / 7 |
| 0.999 | 500.25012506253097 | 2.8e-09 | 3.94 | 3 / 7 |
| 0.9999 | 5000.2500125011738 | 8.4e-09 | 2.80 | 2 / 7 |
| 0.999999 | 500000.24998574721 | **3.3e-05** | 0.999 | **0 / 7** |
| 0.99999999 | **0.0** (above the clamp) | 0.0 | **5.8e+05** | 1 / 7 |

Two distinct failure modes, both fatal for a test oracle:

- At `u = 0.999999` **no step size works at all**. The best entry over the
  whole decade grid is still `3.3e-05` relative — six orders of magnitude
  worse than the `1e-11` FD achieves in the interior.
- At `u = 0.99999999` the point is above qLLM's `artanh` clamp
  (`1 − 1e-7`), so the shipped operator is locally constant and its exact
  gradient is **identically zero**. Finite differences report up to
  `5.8e+05` — they *invent* a gradient the operator does not have. Only
  `h = 1e-8`, small enough to keep both stencil points above the clamp,
  agrees. A test built on FD here would chase a nonexistent bug.

"Best over the `h` grid" flatters FD: it is the error you get if you already
know the answer. The honest column is `usable steps`, and it degrades
monotonically toward the boundary.

By contrast, `poincare_project` (a polynomial in `x`) is FD-friendly
everywhere — its worst best-case relative error is `1.4e-11`. That contrast is
deliberate: it shows the divergence is a property of the hyperbolic
transcendentals, not an artifact of the harness.

## Exact-rational cross-check

`poincare_project.esk` differentiates the primitive restricted to the ray
`x(t) = t·e₀` at `t = 999/1000` (i.e. `√c‖x‖ = 0.999`, one ulp-hostile step
from the boundary) using **`derivative-n` over exact rationals**. Because the
ray is along `e₀`, `d/dt out_k(x(t))` *is* the Jacobian entry
`d out_k/d x_0`. Nothing is hand-derived on either leg — both numbers come out
of an AD engine:

```
conf exact               = 1999/1000000
conf exact->f64          = 0.0019989999999999999
conf f64 (1 - x.x)       = 0.001998999999999973      <-- 1.3e-14 relative error
d out_0/d x_0 exact      = -1997001/2500000000        exact? = #t
d out_0/d x_0 exact->f64 = -0.00079880040000000004
d out_0/d x_0 reverse f64= -0.0007988003999999893
rel|f64_ad - exact|      = 1.3437150892271078e-14
```

The f64 leg loses ~2 decimal digits, and it loses them in the *forward* pass:
`conf = 1 − c‖x‖²` is a catastrophic cancellation of two nearly equal
quantities near the boundary. The gradient is then accurate only to the
conditioning of `conf`. An fp32 kernel loses ~9 digits the same way, so an
fp32 implementation cannot do better than `~1e-5` relative on the conformal
factor at `u = 0.999` no matter how correct its algebra is. **Budget test
tolerances against `conf`, not against the output magnitude.**

## Findings

Three properties of the qLLM formulas that the exact oracle surfaced. All
three are now asserted by the exporters, so a change in behaviour fails the
suite rather than silently changing a golden vector.

### 1. `exp_x` is not invertible near the boundary at coordinate-scaled steps

The conformal factor `λ_x = 2/(1−c‖x‖²)` multiplies the tangent *before*
`tanh` sees it, so a tangent that looks small in coordinates is not small
geodesically. At `√c‖x‖ = 0.999`, `λ_x = 1000.5`, and `‖v‖ = 0.05` gives
`tanh` argument `25.01`, which is exactly `1.0` in binary64. `exp_x` then
lands on the boundary and `log_x(exp_x(v)) ≠ v`:

| case | `λ_x` | `tanh` arg | roundtrip rel. err |
|---|---|---|---|
| `d2.c1.base0p999.tanh_saturated` | 1000.5 | 25.0125 | **2.55e-02** |
| `d8.c1.base0p999.tanh_saturated` | 1000.5 | 25.0125 | **1.57e-02** |
| `d2.c1.base0p999.geodesic_scaled` | 1000.5 | 0.5003 | 7.82e-14 |
| `d8.c1.base0p999.geodesic_scaled` | 1000.5 | 0.5003 | 4.86e-14 |

The geodesic-scaled companions (`‖v‖ ≈ 2/λ_x`) round-trip to 1e-14 at the
same base radius. So this is not a defect in the formulas — it is a
**step-size constraint on manifold residual updates**: A3 must scale tangents
by `1/λ_x`, or equivalently cap `√c·λ_x·‖v‖/2`, or the residual stream leaves
the manifold and the backward through `log_x` is meaningless. `exp_0` shows
the same wall directly: at `√c‖v‖ = 20` the image has `√c‖out‖ = 1.0` exactly,
so the point has numerically left the ball.

### 2. `edge_threshold` is inoperative in `sheaf_attention_ee`

The edge selector is, in both `sheaf_attention_ee` and
`_build_sheaf_edge_weights`:

```python
selected = tri & (w_raw > edge_threshold) & (w_raw > epsilon)
```

The second conjunct dominates the first for every `edge_threshold ≤ epsilon`.
The documented default is `edge_threshold = -1.0`, which suggests
negative-cosine edges are kept — they never are; the `epsilon` guard drops
every one of them. Consequences:

- The sheaf Laplacian **cannot become indefinite** through this path, so any
  claim resting on negative edge weights is currently unreachable.
- `edge_threshold` only does anything once raised above `epsilon`.

Asserted by a pair of cases whose Jacobians are bit-identical at
`edge_threshold = -1.0` and `0.0` (4 active entries either way), alongside an
`edge_threshold = 0.9` case that prunes a weak-but-positive edge (6 → 4
entries), proving the knob works where it can bite.

The same exporter records a structural fact worth having in a golden vector:
the upper-triangular builder reads `Q_i` only for `i < j` and `K_j` only for
`j > i`, so **`Q`'s last row and `K`'s first row receive exactly zero
gradient**. A C backward that writes anything there is wrong.

### 3. `sphere_retract`'s eps guard protects the forward pass, not the backward

At `z = x + step = 0` the guard makes the forward value finite (`out = x`),
but the point is a genuine singularity: perturbing `x` by `h` pushes `‖z‖`
above `eps`, and the other branch returns `h/‖h‖`, which has no limit.
`d out/d x` is therefore legitimately non-finite, while `d out/d step` stays
finite (identically zero, since `out = x` never reads `step`).

The torch reference carries the same hazard — `z.norm()` at `z = 0` yields a
NaN gradient through `torch.where` regardless of the `clamp_min` on the
denominator. A central difference at `h = 0.01` reports a clean `0.0` and
declares success.

Recorded as `d_out_d_x_all_finite: false` with `null` entries, and asserted:
the guard branch **must** be non-finite in `x` and finite in `step`. A qLLM
optimizer that can reach `x + step = 0` needs an explicit guard on the
backward, not just the forward.

## Notes on Eshkol's AD surface for a consumer

Things that shaped these exporters and that a qLLM-side reader should know:

- **`gradient` is reverse-mode over a flat vector point and returns a
  vector.** Vector-valued maps need one call per output component; that is
  what `jac-rows-ad` in `qllm_oracle_lib.esk` does, using a global row index.
- **`gradient` requires inexact points.** Handed a vector of exact rationals
  it returns pointer garbage rather than an exact gradient or an error. The
  exact-capable operators are the forward Taylor tower — `derivative-n` and
  `taylor` — which carry exact rationals end to end (verified: the exact leg
  above returns `-1997001/2500000000` with `exact? = #t`). This is why the
  exact cross-check uses `derivative-n` along a ray rather than `gradient`.
  Every literal on that path must be exact (`(/ 1 4)`, not `0.25`) or the
  value silently demotes to f64.
- **Do not capture a local parameter inside an AD lambda over a vector
  point.** That is tracked open bug ESH-0097 (see `tests/ad_oracle/README.md`)
  and fails the LLVM verifier on both `-r` and AOT. Every non-differentiated
  operand in these exporters travels as a global; passing the *projection
  function itself* as a parameter is fine, since the lambda captures nothing
  local.
- **Nested `gradient` needs an inline lambda** (ESH-0078/ESH-0096). Nothing
  here nests, but a second-order extension would hit it.
- `display` and `number->string` print binary64 to 17 significant digits, so
  values written by an exporter round-trip exactly.
- File output is `open-output-file` / `display … port` / `close-output-port`.
  `call-with-output-file` warns that `open-output-file` takes exactly one
  argument and writes nothing.

## Status of the bridge backwards (the oracle's work queue)

The qLLM campaign treats the unsupported-op error list in
`lib/backend/tensor_backward.cpp` as this instrument's work queue. Assessed on
this tree, all three remaining entries are blocked **upstream of the
backward**, so registering a backward today would be dead code no test can
reach:

**None of the three AD node types has a producer.** `AD_NODE_FRECHET_MEAN`
occurs in exactly two places in the whole tree — the enum at
`inc/eshkol/eshkol.h:951` and the `eshkol_fatal` at
`lib/backend/tensor_backward.cpp:1365`. `AD_NODE_TENSOR_ATTENTION` and
`AD_NODE_TENSOR_EMBEDDING` add only the dispatch-table entries in
`lib/bridge/tensor_backward.cpp`. Nothing anywhere assigns `ad_node_t.type`
any of these values, so the errors are unreachable and the dispatch table
36c2c761 built is, for these ops, a seam waiting for a forward.

- **Fréchet mean** — hardest, and not just plumbing. The forward
  (`lib/backend/vm_geometric.c:1239`, VM opcode 817) is a forward-only FFI
  shim: it downcasts to fp32 via `vm_tensor_to_float`, calls
  `qllm_hyperbolic_frechet_mean(..., 100, 1e-6f)`, copies out `dim` floats and
  calls `qllm_tensor_destroy(result)`. No tape node, no retained operands, no
  saved iterate — and not even f64. Beyond the plumbing there is a real
  mathematical decision: the weighted Fréchet mean `μ` is defined implicitly by
  `Σ_i w_i log_μ(x_i) = 0`, so the exact `∂μ/∂x_i` comes from implicitly
  differentiating that stationarity condition, which needs the Hessian of the
  weighted variance functional at `μ`. qLLM's forward is an iterative fixed
  point, so *implicit differentiation* (mathematically exact,
  iteration-count-independent) and *unrolling the 100 iterations* (bit-matches
  the shipped forward, but is not the derivative of the mathematical map) give
  different answers. Which one qLLM's C backward should match is a call to make
  before writing code — and the `exp`/`log` golden vectors in this directory
  are the prerequisite for building either.

- **Embedding** — the smallest and most tractable. The blocker is already named
  in the code: "the lookup-index tensor is not threaded through the AD node —
  ESH-0230" (tracked in `.swarm/tasks/ESH-0230.json`). The backward itself is
  an indexed scatter-add, straightforward *given* the indices; the work is
  forward-side, wiring `node->input2` to the index tensor.

- **Attention** — needs the Q/K/V split and the softmax intermediate retained
  on the node; the 5-step chain through `softmax(QKᵀ/√d)V` is then standard.
  Low marginal value for now, because the path users actually differentiate
  (`scaled-dot-attention`) decomposes to scalar AD nodes in
  `tensor_transformer_codegen.cpp` and is already exact — which is precisely
  why the bridge node has no producer.

Suggested order: ESH-0230 embedding (forward plumbing + scatter backward) →
attention → Fréchet mean.

## Files

- `qllm_oracle_lib.esk` — shared vector math, exact/FD Jacobian assembly,
  finiteness checks, JSON emission, verdict counters. Loaded by every
  exporter via `(load "qllm_oracle_lib.esk")`; not a probe itself.
- `poincare_project.esk`, `poincare_retract.esk`, `sphere_ops.esk`,
  `poincare_maps.esk`, `sheaf_ee_step.esk` — one exporter per primitive
  family. Each self-checks, prints `PASS:`/`FAIL:` lines and a
  `Passed:`/`Failed:` summary, exits nonzero on failure, and writes its JSON
  to `$QLLM_ORACLE_OUT` (default `tests/qllm_oracle/golden`).
- `golden/*.json` — the committed golden vectors.

Current corpus: **77 in-language checks across 5 exporters / 9 JSON files**,
green under both the JIT and AOT lanes.
