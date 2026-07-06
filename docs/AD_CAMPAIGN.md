# Arbitrary-order AD Taylor-tower Campaign

This track integrates the arbitrary-order automatic differentiation
Taylor-tower campaign into the canonical release roadmap. It is not a
parallel version scheme: the Taylor tower is the enabling substrate that
lets the existing roadmap move from neuro-symbolic intelligence through
reasoning and into the v2.0 quantum/formal-verification arc.

`ROADMAP.md` remains the source of truth for release themes. This file
maps each AD phase to that roadmap, the ICC completion evidence expected
for the phase, and the ledger tasks from the Taylor-tower design branch.
`docs/design/AD_TAYLOR_TOWER.md` is not present on current `master`; it
lands with PR #147 (`design/ad-taylor-tower`) along with ESH-0185..0197.
Until that PR merges, the version and ICC gate mapping below is the
canonical tag source for those ledger tasks.

## Phase-to-roadmap Alignment

| Phase / ESH | Capability | Target version | ICC oracle criterion | Dependencies | Roadmap theme it feeds |
|---|---|---|---|---|---|
| P0 / ESH-0185 | Standalone recurrence POC for univariate Taylor arithmetic to d=8 | v1.3.0 design input | `ad-depth` design evidence: recurrence POC exits cleanly and matches analytic derivatives | Existing forward AD and numeric tower | v1.3-evolve language maturity and AD hardening |
| P1 / ESH-0186 | Runtime heap tower, `derivative^n`, epoch-tag scaffold; closes ESH-0118 | v1.3.1 | `ad-depth`: `derivative^d` PASS to d=8, nested-confusion suite PASS | P0, current AD runtime, arena heap subtype dispatch | v1.3-evolve AD correctness for day-to-day language use |
| P2 / ESH-0187 | Compile-time-K monomorphization, unrolled stack towers, shared FP-contraction policy | v1.3.2 | `mono-equiv`: monomorphized tower equals runtime tower bit-exactly; no AD hot-loop heap allocation | P1 | v1.3-evolve optimization and compiler ergonomics |
| P3 / ESH-0188 | Subsume JET8 into tagged Taylor towers while retaining JET4 hot path | v1.3.2 | `mono-equiv`: JET8 regression tests PASS through the tower route; perturbation model unified | P1, P2, PR #138 tests | v1.3-evolve AD surface consolidation |
| P4 / ESH-0189 | GUW multivariate mixed partials through `taylor_propagate` and interpolation | v1.4.x | `ad-exact`: multivariate `gradient^d`/mixed partials PASS to d=8 against analytic references | P1, P2 | v1.4-connection infrastructure track alongside networking |
| P5 / ESH-0190 | Reverse-over-Taylor and tower-aware high-order tape | v1.5 | `ad-reverse-highorder`: high-order reverse oracle PASS, AOT-verified | P1-P4, existing reverse tape | v1.5-intelligence differentiable logic and neuro-symbolic ML |
| P6 / ESH-0191 | Exact-coefficient towers with rational/bignum coefficients | v1.4.x | `ad-exact`: rational/bignum derivatives bit-exact vs symbolic oracle; exactness contagion PASS | P1, numeric tower | v1.4-connection infrastructure track alongside networking |
| P7 / ESH-0192 | Tensor-valued towers for conv2d/attention/batchnorm/layernorm high-order AD | v1.5 | `ad-tensor-highorder`: ML AD smoke PASS; order-1 agrees with current tensor AD | P1-P5, tensor AD path | v1.5-intelligence neural workloads and differentiable ML |
| P8 / ESH-0193 | Taylor models with rigorous interval remainders and validated AD | v2.0 | `ad-validated-bounds`: true range contained and enclosure tightens as order increases | P1-P7, interval/remainder arithmetic | v2.0-starlight formal verification of AD |
| P9 / ESH-0194 | Differentiable control flow through loops, branches, recursion, closures, named-let, map/fold | v1.5 | `ad-control-flow`: control-flow AD oracle PASS to d=8; kink policy enforced | P1-P5 | v1.5-intelligence differentiable programs, not just differentiable kernels |
| P10 / ESH-0195 | Checkpointed high-order reverse with rematerialization for tower-valued tapes | v1.5->v1.7 | `ad-reverse-highorder`: deep high-order reverse PASS within memory budget | P5, P9 | v1.5-intelligence now; v1.7-synthesis scalability for learned search |
| P11 / ESH-0196 | Tower-based user numerics: series ODEs, root finding, inversion, analytic continuation | v1.4.x | `ad-numerics`: ODE/root/inversion examples converge to analytic references | P1, P2, P6 | v1.4-connection infrastructure track as public numerical substrate |
| P12 / ESH-0197 | Sparse high-order tensors via sparse GUW recovery and star-coloring seed directions | v1.6 | `ad-sparse`: sparse recovery matches dense recovery on tractable cases; savings recorded | P4, P9 | v1.6-reasoning knowledge graphs and sparse relational structure |

## ICC Gate Names

The campaign uses these ICC criterion/event families so release targets
can compose AD evidence without duplicating implementation details:

- `ad-depth`: arbitrary-order univariate derivatives and nested AD to d=8.
- `ad-exact`: exact coefficients and multivariate analytic checks.
- `ad-control-flow`: tower propagation through control-flow and higher-order calls.
- `ad-tensor-highorder`: tensor-valued tower correctness for ML operators.
- `ad-reverse-highorder`: reverse-over-Taylor and checkpointed reverse.
- `ad-sparse`: sparse high-order mixed-partial recovery.
- `ad-validated-bounds`: Taylor-model enclosure soundness and tightening.
- `ad-numerics`: public Taylor-series numerical APIs.
- `mono-equiv`: compile-time monomorphized tower/runtime tower equivalence.

## Ledger Tagging

The current `master` branch has ledger tasks only through ESH-0184, so
there are no `.swarm/tasks/ESH-0185..0197.json` files to edit in this
branch. When PR #147 merges those tasks, tag each JSON file with:

```json
"version": "<target version from the table>",
"icc_gate": "<ICC oracle criterion from the table>"
```

For P10, use `version: "v1.5->v1.7"` and
`icc_gate: "ad-reverse-highorder"` to reflect the current release
bridge: it starts as v1.5 intelligence infrastructure and carries the
scaling work forward into synthesis.
