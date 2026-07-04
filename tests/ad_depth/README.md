# Depth-parametric AD oracle (adversarial campaign, pillar P6a)

Where the P3 AD oracle (`tests/ad_oracle`) tests a WIDE matrix at SHALLOW,
fixed nesting (nesting depth <= 2), this pillar sweeps the **nesting depth
itself**. Depth-dependent AD bugs — a composition correct at depth 1 or 2 but
broken at depth 3+ — slipped through every earlier harness (see the meta-lesson
in `.swarm/DEPTH_PARAMETRIC_TESTING.md`). Here every composable AD construct is
generated PARAMETRICALLY at depth `d = 1..8` and checked against a ground-truth
oracle that scales with depth, so we record the **max-correct-depth** of each
construct and whether it FAILS (silent wrong value) or hits a clean LIMIT.

## Compositions swept

| composition | meaning | sweep |
|---|---|---|
| `deriv`  | `derivative^d` of a scalar function | 1 |
| `gradn`  | `gradient^d` nested-reverse on a scalar | 3 |
| `gofd`   | `gradient` (reverse) OVER `derivative^d`, vector param via `vector-ref` | 2 (ESH-0117 family) |
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
- Failures return an exact `0` (or garbage / SIGSEGV), far outside any band.

## Running

```
scripts/run_ad_depth.sh              # full sweep, JIT + AOT
scripts/run_ad_depth.sh --no-aot     # JIT lane only
scripts/run_ad_depth.sh --regen      # regenerate the corpus first
scripts/run_ad_depth.sh --quick      # CI smoke subset
scripts/run_ad_depth.sh --max-depth 12
```

Products: `AD_DEPTH_REPORT.md` (per-cell depth tables + max-correct-depth),
`scripts/icc_traces/ad_depth.jsonl` (`kind:"ad_depth"` events), gated by
`.icc/completion-oracles.yaml::ad-depth`. The gate is PASS when no construct
REGRESSES below its tracked baseline max-depth
(`scripts/ad_depth_report.py::BASELINE`); a fix that raises a boundary shows up
as an "improvement" and stays green.

## Files

- `../../scripts/gen_ad_depth.py` — deterministic generator (byte-for-byte
  reproducible). Emits `generated/ad_depth_<comp>_NN.esk`, one-cell-per-file
  `generated/ad_depth_hessod_xc_NN.esk` for the crashing hessian cells, and
  `generated/cells.tsv` (cell registry consumed by the reporter).
- `../../scripts/ad_depth_report.py` — parses the run log into the report +
  ICC trace; holds the tracked baseline and ESH-task map.
- `../../scripts/run_ad_depth.sh` — JIT+AOT runner.
- `found/` — hand-shrunk minimal repros for the bugs this oracle discovered
  (acceptance tests of the referenced ESH tasks; not run by the gate).

## Findings (max-correct-depth on master, -r and AOT identical)

| composition | capture | max-correct-depth | tracked |
|---|---|---|---|
| deriv | capnone / global | **2** (d>=3 → exact 0) | ESH-0118 |
| deriv | localparam / vecref | **1** (d2 → garbage ~2.2e13) | ESH-0119 |
| gradn | capnone | **2** (d>=3 → 0) | ESH-0118 |
| gradn | vecref | **1** (d2 → garbage) | ESH-0119 |
| gofd  | vecref | **1** (d>=2 → 0) | ESH-0117 |
| jacod | vecref | **0** (d1 already → 0) | ESH-0120 |
| hessod | vecref | **0** (d1 → SIGSEGV) | ESH-0121 |
