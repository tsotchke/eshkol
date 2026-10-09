# Depth-parametric AD oracle (adversarial campaign, pillar P6a)

Where the P3 AD oracle (`tests/ad_oracle`) tests a WIDE matrix at SHALLOW,
fixed nesting (nesting depth <= 2), this pillar sweeps the **nesting depth
itself**. Depth-dependent AD divergences — a composition correct at depth 1 or
2 but not at depth 3+ — slipped through every earlier harness, which is the
lesson this pillar encodes. Here every composable AD construct is
generated PARAMETRICALLY at depth `d = 1..8` and checked against a ground-truth
oracle that scales with depth, so we record the **max-correct-depth** of each
construct and whether it FAILS (silent wrong value) or hits a clean LIMIT.

## Compositions swept

| composition | meaning | sweep |
|---|---|---|
| `deriv`  | `derivative^d` of a scalar function | 1 |
| `gradn`  | `gradient^d` nested-reverse on a scalar | 3 |
| `gofd`   | `gradient` (reverse) OVER `derivative^d`, vector param via `vector-ref` | 2 |
| `jacod`  | `jacobian` OVER `derivative^d`, vector field | 4 |
| `hessod` | `hessian`  OVER `derivative^d`, scalar field | 4 |

Axes: shapes `mono`/`poly`/`expc`/`sinc`; points scalar / 2-vector / 3-vector;
bindings inline / named / lamvar; captures capnone / global / localparam /
vecref. Each `(composition, shape, point, binding, capture)` cell is swept d=1..8
on BOTH `-r` and AOT.

## Ground truth (no hand computation, no reliance on the AD path)

Every shape has a CLOSED-FORM n-th derivative, so the ground truth at ANY depth
is an analytic literal computed in Python (`mono` `t^K`→`K!/(K-n)! t^(K-n)`,
`expc`→`A^n e^{At}`, `sinc`→`A^n sin(At+nπ/2)`). As a second, AD-independent
anchor for the viable low-depth range, each `deriv` probe also computes an
in-language n-th central-difference stencil for `d <= 4` and checks it agrees.

### Tolerance schedule
- analytic vs AD: `d<=2` rtol/atol 1e-6; `d<=4` 1e-5; `d>=5` 1e-4 (nested fp).
- fd stencil: emitted only for `d <= 4` (order-n central difference is
  numerically dead beyond that: round-off ~ `eps/h^n`); `h=1e-2`, diagnostic.
- Failures return an exact `0`, an unrelated value, or stop on a signal — far
  outside any band.

## Running

```
scripts/run_ad_depth.sh              # full sweep, JIT + AOT
scripts/run_ad_depth.sh --no-aot     # JIT lane only
scripts/run_ad_depth.sh --regen      # regenerate the corpus first
scripts/run_ad_depth.sh --quick      # CI smoke subset
scripts/run_ad_depth.sh --max-depth 12
```

Products: `docs/reports/AD_DEPTH_REPORT.md` (per-cell depth tables + max-correct-depth),
`scripts/icc_traces/ad_depth.jsonl` (`kind:"ad_depth"` events), gated by
`.icc/completion-oracles.yaml::ad-depth`. The gate is PASS when no construct
REGRESSES below its tracked baseline max-depth
(`scripts/ad_depth_report.py::BASELINE`); a fix that raises a boundary shows up
as an "improvement" and stays green.

## Files

- `../../scripts/gen_ad_depth.py` — deterministic generator (byte-for-byte
  reproducible). Emits `generated/ad_depth_<comp>_NN.esk`, one-cell-per-file
  `generated/ad_depth_hessod_xc_NN.esk` for the hessian cells that once
  stopped on a signal, and
  `generated/cells.tsv` (cell registry consumed by the reporter).
- `../../scripts/ad_depth_report.py` — parses the run log into the report +
  ICC trace; holds the tracked baseline (`BASELINE`) and tracking map
  (`TRACK`).
- `../../scripts/run_ad_depth.sh` — JIT+AOT runner.
- `found/` — hand-shrunk minimal repros for the divergences this oracle
  discovered (acceptance tests of their tracking entries; not run by the
  gate).

## Findings (max-correct-depth)

The tracked baseline in `scripts/ad_depth_report.py::BASELINE` is the gate
contract; `docs/reports/AD_DEPTH_REPORT.md` carries the per-cell tables from
the last committed run.

| composition | capture | max-correct-depth (baseline) | history |
|---|---|---|---|
| deriv | capnone / global / localparam / vecref | **8** (the full ladder) | was 2 (capnone/global) and 1 (localparam/vecref) when this pillar landed: nested `derivative` chains of depth ≥ 3 now route through the arbitrary-order Taylor tower, and captures flow through the tower call unchanged |
| gradn | capnone | **3** | was 2 |
| gradn | vecref | **3** | was 1; the higher-order `derivative` closure is now dual-transparent (it seeds and extracts its own perturbation level), so the capture form is no longer the limit; depth 4+ is bounded by the 8-jet's three perturbation slots |
| gofd | vecref | **8** | was 1 |
| jacod | vecref | **8** | was 0 (forward-over-reverse) |
| hessod | vecref | **1** | was 0; the four `hessod.*.vecref` cells measure **8** at v1.3.6-evolve, above this baseline |

The hand-shrunk repros in `found/` all print their expected values at
v1.3.6-evolve (`d3 10752`, `jac-outer 1024`, `hess-outer 1024`,
`d2-local 6092.800000000001`, i.e. 1.7 × 3584).
