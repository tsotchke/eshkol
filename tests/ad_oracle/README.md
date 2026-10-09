# AD composition oracle (adversarial testing campaign, pillar P3)

Permanent divergence-finding infrastructure for Eshkol's
automatic-differentiation system. Every AD divergence found in the v1.3
campaign lived at a COMPOSITION point that no unit test covered (nested gradient-in-named-let, named-function inner
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
| carrier   | compositions that cross between the two forward carriers — the 8-jet (`derivative`) and the heap Taylor tower (`derivative-n`/`taylor`) — in both directions, each value checked against FD **and** against the equivalent single-carrier spelling |

For each valid cell the generator emits a probe that computes the AD value AND
a central finite-difference approximation of the same quantity, then checks

```
|ad - fd| <= atol + rtol*|fd|        rtol = 1e-4
```

First-order stencils use `h = 1e-5` (atol 1e-6); second-order stencils
(hessian entries, laplacian) use `h = 1e-4` (atol 1e-5) because the `eps/h²`
round-off term dominates at 1e-5. For nested cells the FD baseline is the
central difference of the (separately validated) inner AD computation.

Current corpus (v1.3.6-evolve, `generated/MANIFEST.txt`): **249 probes / 644
checks in 44 files**, run under BOTH the JIT (`-r`) and AOT. A full sweep grades
88 file×mode cells and reports `ad_oracle summary: total=88 passed=88 xknown=0
failed=0 crashed=0 hung=0`.

## Files

- `gen_ad_oracle.py` — deterministic generator (no RNG; rerunning reproduces
  the corpus byte-for-byte). Regenerate with
  `python3 tests/ad_oracle/gen_ad_oracle.py`.
- `generated/ad_oracle_<section>_<NN>.esk` — probe files (~15 probes each),
  sections: `deriv grad hess jac div curl lap nest loop carrier`.
- `generated/ad_oracle_xc_<task>_<NN>.esk` — expected-fatal-signal /
  expected-compile-fail cells, ONE probe per file so one such cell masks
  nothing else. The runner classifies these as XKNOWN while the referenced
  entry is open, and they flip to PASS automatically once fixed. None exist at
  v1.3.6-evolve (`CRASH_TASKS` is empty).
- `generated/MANIFEST.txt` — file → probe/check counts.
- `found/` — hand-shrunk minimal repros for real compiler divergences
  discovered by this oracle (not executed by the runner; they are the
  acceptance tests of their tracking entries).

## Running

```
scripts/run_ad_oracle.sh            # full sweep, JIT + AOT
scripts/run_ad_oracle.sh --quick    # CI subset (*_01.esk of each section/task)
scripts/run_ad_oracle.sh --no-aot   # JIT lane only
scripts/run_ad_oracle.sh --regen    # re-run the generator first
```

Verdicts per file+mode: `PASS` / `FAIL` (an untracked cell diverged from
finite differences) / `XKNOWN` (tracked open divergence) / `CRASH` / `HANG`. The
gate is green iff there are no FAIL/CRASH/HANG. The runner emits
pytest-style `PASSED/FAILED/XFAIL` lines and ICC JSON-L events
(`kind:"ad_oracle"`) into `scripts/icc_traces/ad_oracle.jsonl`; the gate
event is consumed by `.icc/completion-oracles.yaml::ad-oracle`.

## Known-open cells (XKNOWN)

Nothing is XKNOWN at v1.3.6-evolve: every generated cell is expected to PASS,
and a regression FAILs loudly rather than being absorbed. `CRASH_TASKS` in
`gen_ad_oracle.py` is empty.

One family of marks remains in the generator: the four `nest.gofd.*.v2`
checks in `ad_oracle_nest_04.esk` (vector-parameter gradient over an inner
`derivative`, mixed forward/reverse) still pass through `chk-x`. They print
`PASS: … (fixed: <task>)` — the value matches finite differences — so the
xknown count is zero; removing those `xc=` arguments turns them into ordinary
`chk` cells.

History of the cells this oracle once tracked as open, all now PASS under
`-r` and AOT:

| cell | what it showed when filed |
|------|---------------------------|
| `*.caplocal` / `*.capvrefout` for the vector-param operators | an AD lambda that captured a LOCAL binding failed the LLVM verifier (`PtrToInt source must be pointer`). It no longer reproduces on `-r`, AOT or the bytecode VM: `found/esh0097_local_capture_vector_ad_ptrtoint.esk` prints the `#(4.42 0)` its own header names as the expected value, and the shapes are gated by `tests/ad/captured_local_vector_param_test.esk` (CTest, JIT + AOT), `tests/ad/sweep_c_regressions_test.esk` and the captured-local section of `tests/vm_parity/corpus/32_gradient_reverse.esk`. Ledger entry LE-06. |
| `nest.gofg.*.s.named/lamvar` | a 2nd-order gradient through a NAMED inner function returned 0 while the inline lambda worked. |
| `nest.gofd.*.v2` | a vector-param gradient over an inner derivative returned zeros (mixed forward/reverse). |
| `nest.gofg.*.v1/v2` | a vector-param gradient-of-gradient returned zeros, even the 1-d form documented in AUTODIFF.md. **Found by this oracle.** |
| Hessian / Laplacian at a `(tensor …)` point | the `found/esh0095_*` repros; the point forms are now ordinary hess/lap cells. |

When a marked cell is fixed, its probes print `PASS: … (fixed: <task>)` and
the xknown count drops — no oracle change needed. Then remove the `xc=` marks
in `gen_ad_oracle.py` (and the row above, if it is still listed as open) so
future regressions FAIL loudly instead of XKNOWN.

## Extending the matrix

Add a shape/operator/axis value in `gen_ad_oracle.py` (tables at the top,
`gen_*` methods per section), rerun the generator, run the sweep, triage any
FAIL/CRASH: generator error vs real AD divergence. Real divergences get (1) a
shrunk repro in `found/`, (2) a tracking entry, (3) an `xc=`/`chk-x` mark
referencing it.
