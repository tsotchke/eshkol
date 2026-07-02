# AD composition oracle (adversarial testing campaign, pillar P3)

Permanent bug-finding infrastructure for Eshkol's automatic-differentiation
system. Every AD bug found in the v1.3 campaign lived at a COMPOSITION point
that no unit test covered (nested gradient-in-named-let, named-function inner
gradient, mixed-mode vector-gradient-over-derivative). This oracle enumerates
the whole AD surface as a matrix and checks **every cell against central
finite differences computed in-language** — ground truth with no hand
computation, so the corpus can grow mechanically.

## The matrix

| axis      | values |
|-----------|--------|
| operator  | `derivative` `gradient` `jacobian` `hessian` `divergence` `curl` `laplacian` |
| point     | scalar, 2-vector, 3-vector, 2/3-tensor (`(tensor …)`), multi-param + `(list …)` |
| shape     | polynomial, product-of-linears, with-subtraction, rational `1/(1+x²)`, exp/sin composite, let-bound intermediate reused twice, named-let accumulation loop |
| binding   | inline lambda, named `define`, lambda-in-variable |
| capture   | none, global scalar, LOCAL param scalar, `vector-ref` of outer param |
| nesting   | none, derivative-of-derivative (pure 2nd order + perturbation confusion), gradient-of-derivative (scalar + vector param), gradient-of-gradient (scalar + vector param), AD-in-loop reuse |

For each valid cell the generator emits a probe that computes the AD value AND
a central finite-difference approximation of the same quantity, then checks

```
|ad - fd| <= atol + rtol*|fd|        rtol = 1e-4
```

First-order stencils use `h = 1e-5` (atol 1e-6); second-order stencils
(hessian entries, laplacian) use `h = 1e-4` (atol 1e-5) because the `eps/h²`
round-off term dominates at 1e-5. For nested cells the FD baseline is the
central difference of the (separately validated) inner AD computation.

Current corpus: **214 probes / 440 checks in 40 files**, run under BOTH the
JIT (`-r`) and AOT.

## Files

- `gen_ad_oracle.py` — deterministic generator (no RNG; rerunning reproduces
  the corpus byte-for-byte). Regenerate with
  `python3 tests/ad_oracle/gen_ad_oracle.py`.
- `generated/ad_oracle_<section>_<NN>.esk` — probe files (~15 probes each),
  sections: `deriv grad hess jac div curl lap nest loop`.
- `generated/ad_oracle_xc_<ESH-task>_<NN>.esk` — expected-crash /
  expected-compile-fail cells, ONE probe per file so a crash masks nothing
  else. The runner classifies these as XKNOWN while the referenced task is
  open, and they flip to PASS automatically once fixed.
- `generated/MANIFEST.txt` — file → probe/check counts.
- `found/` — hand-shrunk minimal repros for real compiler bugs discovered by
  this oracle (not executed by the runner; they are the acceptance tests of
  their ESH tasks).

## Running

```
scripts/run_ad_oracle.sh            # full sweep, JIT + AOT
scripts/run_ad_oracle.sh --quick    # CI subset (*_01.esk of each section/task)
scripts/run_ad_oracle.sh --no-aot   # JIT lane only
scripts/run_ad_oracle.sh --regen    # re-run the generator first
```

Verdicts per file+mode: `PASS` / `FAIL` (an untracked cell diverged from
finite differences) / `XKNOWN` (tracked open bug) / `CRASH` / `HANG`. The
gate is green iff there are no FAIL/CRASH/HANG. The runner emits
pytest-style `PASSED/FAILED/XFAIL` lines and ICC JSON-L events
(`kind:"ad_oracle"`) into `scripts/icc_traces/ad_oracle.jsonl`; the gate
event is consumed by `.icc/completion-oracles.yaml::ad-oracle`.

## Known-open cells (XKNOWN)

| task | cell | symptom |
|------|------|---------|
| ESH-0078 | `nest.gofg.*.s.named/lamvar` | 2nd-order gradient through a NAMED inner function returns 0 (inline lambda works). NOTE: the ledger marks this merged via #95, but the acceptance shape still returns 0 on master — regressed or never fully fixed. |
| ESH-0093 | `nest.gofd.*.v2` | vector-param gradient over inner derivative returns zeros (mixed forward/reverse). Fix in flight at time of writing. |
| ESH-0095 | `hess/lap poly.t2/t3` (xc files) | hessian/laplacian SIGSEGV on tensor points. **Found by this oracle.** |
| ESH-0096 | `nest.gofg.*.v1/v2` | vector-param gradient-of-gradient returns zeros (even the 1-d form documented in AUTODIFF.md). **Found by this oracle.** |
| ESH-0097 | `*.caplocal / *.capvrefout` for vector-param ops (xc files) | LLVM verifier: `PtrToInt source must be pointer (ptrtoint %eshkol_tagged_value %a to i64)` on both -r and AOT when the AD lambda captures a LOCAL function parameter. **Found by this oracle.** |

When a task is fixed, its probes print `PASS: … (fixed: ESH-NNNN)` and the
xknown count drops — no oracle change needed. Then delete the task id from
the tables above and (for xc cells) remove the `xc=` marks in
`gen_ad_oracle.py` so future regressions FAIL loudly instead of XKNOWN.

## Extending the matrix

Add a shape/operator/axis value in `gen_ad_oracle.py` (tables at the top,
`gen_*` methods per section), rerun the generator, run the sweep, triage any
FAIL/CRASH: generator error vs real AD bug. Real bugs get (1) a shrunk repro
in `found/`, (2) an ESH task in `.swarm/tasks/`, (3) an `xc=`/`chk-x` mark
referencing the task.
