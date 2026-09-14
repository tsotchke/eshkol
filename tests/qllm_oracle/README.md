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
| `golden/squared_distance.json` | `d²` on `H^n` / `S^n` / `R^n` and a product of them, and what the `sqrt` route does at the diagonal | `inc/eshkol/bridge/space_form.h` (`AD_NODE_SQUARED_DISTANCE`) |

Every Jacobian is a full `m × d` matrix, assembled row by row as
`gradient(⟨e_k, F⟩)`. A consumer can contract it with any cotangent to get
the VJP its own backward produces, so the artifact serves both Jacobian and
VJP tests.

## Schema

```json
{
  "schema_version": 1,
  "primitive": "poincare_project",
  "eshkol_version": "1.3.4",
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

The JSON under `golden/` is a **committed independent reference**: qLLM
consumes it as a reference, and the runner never overwrites it. The runner
generates candidates under `.scratch/qllm-oracle` (or `ORACLE_SCRATCH_DIR`),
validates every candidate's schema, and byte-compares `squared_distance.json`
with its committed reference. A changed implementation therefore cannot
refresh its own squared-distance oracle into green. Two older geometric
goldens are schema-validated only because their host/compiler-sensitive text
has pre-existing byte drift.

### Version pinning for qLLM consumers

Pin on `eshkol_version` + `schema_version` in the JSON, not on a git SHA.
`schema_version` is bumped when field names or nesting change; `eshkol_version`
tracks `ESHKOL_VERSION` in the root `CMakeLists.txt`. If a consumer needs to
know exactly which compiler produced a vector, record the Eshkol commit
alongside its own test fixture — the exporters do not embed one, because the
values are a property of the mathematics and the f64 format, not of the build.

### Regeneration provenance (2026-07-30)

The golden vectors were regenerated once on this branch, after a pre-existing
reverse-mode AD regression that this instrument had detected and refused to
paper over was fixed upstream.

**What was wrong, and why regeneration was withheld earlier.** `(gradient f
x)` mis-attributed derivatives when `f` selected a component of a vector that
was freshly allocated and filled by `vector-set!` inside a loop — the write
barrier that promotes escaping values out of a loop's nursery arena did not
recognize a tagged AD dual number as carrying an arena pointer, so the nursery
reset on the loop's back edge recycled the dual's storage while the tangent
was still live. Two of the five exporters here (`sphere_ops.esk`,
`sheaf_ee_step.esk`) hit exactly this pattern through `jac-rows-ad`'s
row-by-row Jacobian assembly, so on the broken build their self-check against
finite differences failed. Regenerating anyway would have overwritten a known
correct golden with a silently wrong one, so the suite was deliberately left
red (4/10) and the previously committed vectors — generated before the
regression, and hand-verified against the closed-form Jacobian for
`sphere_project.d2` — were kept as the reference.

**The fix.** Root-caused and merged as PR #396 (`fix(memory): the iter-scope
write barrier promotes AD dual numbers before nursery reset`), landing on
master together with PR #393 (`fix(ad): exact-input derivative/gradient/
hessian route through the Taylor tower`). Neither touches this branch's own
files; both change AD behavior generally.

**The regeneration.** `origin/master` at `640faa7d` (which includes both
fixes) was merged into `feat/qllm-oracle-backwards` at merge commit `3ca121f9`,
the tree was rebuilt, and `scripts/run_qllm_oracle_tests.sh` was run to
regenerate all nine `golden/*.json` files.

- Gate: **10/10** (both JIT and AOT lanes of all five exporters), up from the
  4/10 recorded against the pre-fix build.
- `schema_version` is unchanged (`1`) in every file.
- Where the previous (pre-regression) golden was already correct, the
  regenerated values are the **same IEEE-754 doubles**, just not always the
  same printed digit string (e.g. `-0.95999999999999996` and `-0.96` are the
  same double; `float(a) == float(b)` was checked directly, not inferred from
  the text). `sphere_project.d2`'s `d_out_d_x` regenerated to
  `[[-0.96, 0.36], [0.32000000000000006, -1.2]]`, matching the closed-form
  `d out_i/d x_j = -(g_j x_i) - <g,x> delta_ij` used to hand-verify the
  original golden — the fix restores the pre-regression answer, it does not
  produce a new one.
- `fd_cross_check.best_rel_diff` for the polynomial `sphere_project` cases —
  the ones with a closed-form check and no hyperbolic transcendentals to
  confound the comparison — came back `2.691449756667046e-15` (d2),
  `3.970679993953501e-15` (d4), `2.425866844623763e-15` (d8): the same order
  as the `2.7e-15` recorded when the golden was first generated, before the
  regression existed. `fd_usable` is `true` everywhere finite-difference
  agreement is meaningful, and correctly `false` only at the boundary-hostile
  cases this README's FD-vs-exact section documents by design (near-clamp
  `log_0`/`exp_x` points, and the `sphere_retract` eps-guard singularity).
- **Determinism.** The regeneration was run twice against the identical build
  (`build/eshkol-run`, unchanged between runs). SHA-256 of all nine
  `golden/*.json` files was identical byte-for-byte across both runs.

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
`lib/backend/tensor_backward.cpp` as this instrument's work queue. Two of the
three entries are now implemented; the third is deliberately deferred.

**Embedding — DONE (ESH-0230).** `tensor_embedding_backward` is the exact
indexed scatter-add `dW[idx[i],:] += dy[i,:]`. The blocker named in the ticket —
the lookup-index tensor absent from the AD node — is closed by making
`node->input2` the index operand, with `params` carrying
`[num_indices, d_model, vocab_size]`. Duplicate indices accumulate (a row looked
up *k* times receives the sum of all *k* upstream rows) and rows never looked up
stay bitwise zero. `ctest -R tensor_embedding_backward_gradcheck`: 9 checks
including central finite differences, the duplicate-index case, and three
refusals (missing index operand, fractional index, out-of-range index).

**Fréchet mean — DONE, by implicit differentiation.** The mathematical decision
recorded above has been taken: `tensor_frechet_mean_backward` differentiates the
stationarity condition `Σ_i w_i log_μ(x_i) = 0` at the converged fixed point, not
the iteration that solves it. Those are different functions — the unrolled
derivative carries the iteration's transient and depends on the starting point
and the iteration count, neither of which is a property of the Fréchet mean. The
gradcheck measures the gap rather than asserting it: on its fixture the
one-step-unrolled derivative differs from the implicit one by up to `7.8e-2`.

Reverse mode needs one linear solve regardless of the number of points: solve
`Aᵀz = dL/dμ` with `A = Σ_i w_i ∂log_μ(x_i)/∂μ`, then
`dL/dx_j = -w_j (∂log_μ(x_j)/∂x_j)ᵀ z` and `dL/dw_j = -⟨log_μ(x_j), z⟩`.

**The residual gate is the companion requirement, not a nicety.** Every step of
that derivation assumes `F(μ*) = 0`. At a non-converged point the implicit
function theorem does not apply and the formulas still return a smooth,
plausible, *wrong* vector — which is strictly worse than an error, because
nothing downstream can distinguish it from a gradient. The rule recomputes the
residual from the retained `μ*`, points and weights (recomputed, not stored: a
stored residual can be stale relative to the operands actually on the node) and
refuses when it is not stationary, including for a displacement of only `1e-6`.

**The bar, exactly, because a consumer matching this has to use the same one:**

```
λ_μ‖F‖₂ ≤ tol · Σ_i w_i · (1 + λ_μ · max_i ‖log_μ(x_i)‖_∞),
λ_μ = 2/(1 − c‖μ‖²),   tol = 1e-9
```

Both sides are in **Riemannian** units, and that is load-bearing rather than
pedantic. The logs are stored in ambient ball coordinates; the tangent space at
`μ` carries the conformal metric `λ_μ²⟨·,·⟩`, so a tangent vector's invariant
length is `λ_μ‖v‖`. The factor cancels out of the relative term but not out of the
absolute floor — and the floor is why the bar is not purely relative (with every
point coincident with the mean, each log is zero to rounding, and a purely
relative bar would divide an exact residual by that noise). Scaled in ambient
coordinates the floor swamps the relative term as `μ` approaches the boundary,
`λ_μ` diverges, every `‖log‖` collapses, and the bar degenerates to
`‖F‖_ambient ≤ tol·Σw_i` — which a mean wrong by a whole unit of hyperbolic
distance satisfies. With the ambient scale the forward accepted means wrong by
`8.8e-8` and `7.6e-6` as converged. A consumer that reproduces this gate in
ambient units will believe it has a converged mean when it does not.

**The forward's honest range, which is narrower than the ball.** Ambient f64
coordinates cannot resolve hyperbolic position near the boundary: `u = (−μ) ⊕_c x`
is formed by cancellation, so a `μ` and an `x` more than roughly 19 units of
hyperbolic distance apart drive `‖u‖` to `1` even though both are strictly
interior, and `artanh` has no value. Nor is the residual resolvable there — it has
an evaluation noise floor of its own, so acceptance requires two consecutive
sub-tolerance iterates and a run whose residual stops improving is refused as
stagnant. Measured on points at `±x0` with weights `2:1`, closed form
`tanh(artanh(x0)/3)`: accepted out to `x0 = 1 − 1e-6` with relative error `8e-16`
to `2.3e-12`, refused from `x0 = 1 − 1e-7` inward. Consumers should treat a
refusal in that regime as correct and reformulate (recentre the chart, or carry
more precision), not retry.

The forward changed too, and consumers should note it: VM opcode 817 previously
returned the **Euclidean weighted average** and discarded its curvature argument
entirely. On the Poincaré ball that is not the Riemannian center of mass and not
an approximation of one — it agreed only at the origin or at zero curvature, with
nothing in the output revealing the curvature had been dropped. It is now a real
f64 Karcher iteration that gates its own convergence and input domain and raises
catchable errors. **f64, not the fp32 `qllm_hyperbolic_frechet_mean` entry
point**, and that is forced rather than preferred: an fp32 mean carries
`|μ − μ*| ~ 1e-7` and therefore a relative stationarity residual around `1e-7`,
two orders above the `1e-9` gate, so an fp32 forward makes the exact derivative
unavailable by construction. A qLLM-side backward that wants to match this must
compute its mean in double precision too, or accept that its own residual gate
cannot be satisfied.

For a 1-D cross-check independent of any implementation: on a diameter of the
disc the arc-length coordinate is `t = 2·artanh(x)` and the weighted Fréchet mean
is the weighted average in `t`, so points at `±0.8` with weights `3:1` give
exactly `0.5`, against a Euclidean average of `0.4`.

**Attention — DONE (SW-12).** `ad_tensor_attention` retains the dense softmax
weights `A` and the causal flag on the node, and `tensor_attention_backward` runs
the exact 5-step chain through `softmax(QKᵀ/√d)V` from them. Recomputing `A` in
the backward was the alternative and was rejected: it would have to re-derive the
softmax max-shift and the mask, and any drift between the two copies is precisely
the silently-wrong-gradient class SW-12 exists to close. The causal case is
checked by scanning the retained weights for a non-zero above the diagonal —
exact equality, not a tolerance.

**Producers, and what remains.** All three node types now have a forward that
records them: `ad_tensor_attention`, `ad_tensor_embedding` and `ad_frechet_mean`
in `lib/bridge/qllm_bridge.cpp`. Each is gradchecked *through the producer* —
`ctest -R qllm_bridge_producer_gradcheck` for embedding and the Fréchet mean,
`ctest -R qllm_bridge_gradcheck` for attention — which is a different claim from
the hand-built-node gradchecks that came first. A fixture assembled by hand
agrees with the backward by construction, because it is written from the same
contract; the one defect class it structurally cannot see is a producer that
fills that contract wrongly.

**The remaining gap is Eshkol-language reachability, and it is NOT a one-site
change.** The producers are reachable from C, which is the external-tensor bridge
path. They are not reachable from `(gradient (lambda (W) … (embedding idx W)))`,
because no *compiled* Eshkol program can create an `AD_NODE_TENSOR_*` node at
all. `lib/backend/llvm_codegen.cpp` (see the block comment above
`kDenseTensorADNodesEnabled`) enumerates three independent unfinished pieces, and
flipping the flag SIGSEGVs rather than producing a slower-but-correct gradient:

1. `recordADNodeTensor` leaves `tensor_gradient` NULL, while the reverse pass
   *selects* the tensor backward by testing that field non-null — constructor and
   consumer each wait for the other;
2. the node it builds is dropped on the floor (the function returns a plain
   tensor, so nothing downstream can find it);
3. under AD the scalarizing path leaves AD-node *pointers* in the result tensor's
   elements, which is what `tensor-sum` and friends consume, so a dense node
   would sever the chain at the next tensor op.

That is ADR-0002 Position A, the dense resident tape, scheduled for v1.6. The VM
is a separate matter again: `lib/backend/vm_autodiff.c` has its own scalar
`AdNode` representation and no VM file references `ad_node_t`, so `frechet-mean`
(opcode 817) cannot record one of these nodes without the shared-node-model work
described in `docs/reference/ad/architecture.md`. Its forward is nonetheless the
*same code* the bridge producer runs — `inc/eshkol/backend/frechet_mean_core.h` —
so the two cannot drift apart while that work is pending.

### The geometric bridge ops had no backward at all (SW-65)

Separate from the three above, and worse, because it was silent rather than
merely missing. `ad_hyperbolic_distance`, `ad_poincare_exp_map`,
`ad_poincare_log_map` and `ad_geodesic_attention` record **tensor-valued** AD
nodes — types 33, 34, 35 and 37 — and none of them had a backward. They did not
refuse and they did not warn: those type numbers sit in the band
`eshkol_tensor_backward_dispatch` treated as "scalar ops differentiated by
codegen", so they fell into its `default:` and the reverse sweep propagated
nothing. Every input gradient came back exactly `0.0`, which a caller cannot
tell from a genuine zero.

All four now have exact rules. The `exp`/`log` rules reuse the Möbius and
log-map Jacobians the Fréchet rule already carries rather than re-deriving them
— the log map *is* the function that routine differentiates, and a second
derivation could only introduce a disagreement.

**These golden vectors are what validate them.** The `exp_0` and `log_0`
Jacobians in `golden/` are computed by Eshkol's reverse-mode AD over an
independently written Eshkol transcription of the same formulas, so asserting
the C rules against them is a genuine two-implementation, two-language check:
`poincare_log_map_origin.d2.c1.u0p5` agrees to `3.7e-16` and
`poincare_exp_map_origin.d2.c1.tv0p1` to `1.1e-14`. Two further exact
references back them up — the conformal-factor identity
`|∇_x d| = λ_x = 2/(1−c‖x‖²)`, which holds at every interior pair, and the
inverse-Jacobian identity `J_log · J_exp = I`, which couples the two rules to
each other with no appeal to either derivation. Finite differences are the last
line, not the first. `ctest -R qllm_bridge_geometric_gradcheck`.

**Two facts the rules make explicit that the silent zero had hidden.** The
Riemannian distance behaves like `|x − y|` near coincidence: it has no
derivative at `x = y`, only a subgradient set, so both distance-based rules
refuse there. For geodesic attention that means the op is **not differentiable
whenever a query row equals a key row exactly** — the ordinary case when `Q` and
`K` are the same tensor. That is a property of scoring by distance rather than
by inner product, and a consumer needs to know it. Second,
`ad_poincare_log_map`'s forward clamps `artanh`'s argument at `t ≥ 1` and
returns a value where no finite log exists; the backward refuses there rather
than differentiate the clamp, matching what the Fréchet machinery already does
on the same condition.

The structural half of the fix is the AD-node registry
(`inc/eshkol/ad_node_registry.def`): the dispatcher no longer has a `default:`
to infer scalar-ness from a numeric band at all. Every node type declares its
backward disposition in one registry row, the dispatch arms and the backward
table are generated from those rows under `-Werror=switch-enum`, and these four
ops are declared `BRIDGE`, each row naming its backward function — a name that
must resolve or `lib/bridge/tensor_backward.cpp` does not compile. A
tensor-valued node type with no rule is an explicit `UNREGISTERED` row that
aborts naming itself instead of returning zero, while `LEAF` rows keep
`AD_NODE_VARIABLE` and `AD_NODE_CONSTANT`, which legitimately carry
`tensor_value`, out of the refusal. Both halves are red-proofed.

## Files

- `qllm_oracle_lib.esk` — shared vector math, exact/FD Jacobian assembly,
  finiteness checks, JSON emission, verdict counters. Loaded by every
  exporter via `(load "qllm_oracle_lib.esk")`; not a probe itself.
- `poincare_project.esk`, `poincare_retract.esk`, `sphere_ops.esk`,
  `poincare_maps.esk`, `sheaf_ee_step.esk`, `squared_distance.esk` — one
  exporter per primitive family. Each self-checks, prints `PASS:`/`FAIL:` lines and a
  `Passed:`/`Failed:` summary, exits nonzero on failure, and writes its JSON
  to `$QLLM_ORACLE_OUT` (the runner supplies a lane-local scratch directory).
- `golden/*.json` — the committed golden vectors.

Current corpus: **82 in-language checks across 6 exporters / 10 JSON files**,
green under both the JIT and AOT lanes.

## `squared_distance.json` is a different kind of artifact

Every other file here exports Eshkol's answer so that qLLM's fp32 kernels have
something exact to assert against. `squared_distance.json` runs the other way:
it exports a SECOND opinion on a primitive Eshkol itself now ships.

`AD_NODE_SQUARED_DISTANCE` (`lib/bridge/space_form_ad.cpp`) computes `d²` in the
log-map form and returns `grad_x d² = -2 log_x(y)`. The exporter computes `d`
in each chart's textbook closed form — `arcosh` on the ball, `arccos` on the
sphere — squares it, and lets Eshkol's generic reverse-mode AD differentiate
the whole composition. Two routes, two languages, no shared line of code; away
from the diagonal they agree to `1.5e-16` or better, and
`tests/bridge/squared_distance_gradcheck_test.cpp` asserts exactly that, citing
each case by id.

The native bridge's public convention is signed sectional curvature (`K < 0`
for the ball, `K = 0` for Euclidean, `K > 0` for the sphere), with spherical
points supplied on-manifold. The exporter keeps its own closed-form route and
uses only on-manifold spherical fixtures, so it remains an independent oracle
rather than reproducing the bridge's projection or stable-core code.

The last case is the interesting one. `squared_distance.ball.coincident.d3.c1`
differentiates the `arcosh` route at `x == y`, where `arcosh'(1)` is infinite
and `d` is zero, so `2·d·d'` is `0·∞`. The value comes back a clean `0.0` and
the gradient comes back **non-finite** — recorded as `null` entries with
`all_finite: false`, beside an `expected_from_log_map_route` block holding the
exact zeros the shipped node returns at the same point.

That contrast is the artifact's whole reason for existing. `d²` is smooth
across the diagonal and `d` is not, `d²` cannot be obtained from `d` by
squaring, and this file is the executable record of why — sitting in the same
directory as the golden vectors that would otherwise be the only evidence
anyone read.
