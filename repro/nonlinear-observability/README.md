# Certify polynomial observation fibers and all-order ambiguity symmetries

Exact polynomial ODE Lie jets, task-relative observation-fiber proofs, and invariant-family symmetry witnesses. Proof replay uses supplied multipliers and re-expansion; Groebner membership is a separate discovery hint.

Base: `755b29fe4f3510d6f446ec1410af03581c143d41` (`feat/v14-groebner`). This is one independently reviewable cut.

The base is #880. Also place the unchanged checker from #900 commit
`4e867ab5e7457a298561ce10d3bd83c20067c904`,
`tools/certcheck/eshkol_certcheck.py`, at that same path. Its SHA-256 is
`c227afa063e4e754202a584730859385a663a0d004b5dacb2e3df981b3ec80dd`.
The gate requires this explicit dependency; it does not merge incompatible ring APIs.

## Run

```sh
python3 repro/nonlinear-observability/self_test.py
python3 repro/nonlinear-observability/gate.py --eshkol-bin /path/to/eshkol-run --certcheck tools/certcheck/eshkol_certcheck.py --results repro/nonlinear-observability/results.json
```

The gate forces `ESHKOL_JIT_CACHE=0` in every native child, even when its caller
sets it to 1. It requires complete expected output and uncached in-process JIT /
AOT-O0 equality, freezes the tested sources for the duration of the run, and
executes producer-free replay with disabled producers exercised as negative
controls. Boundary self-tests launch no native compiler or workload.

The four-state nonlinear ring has local rank four while a period-two swap preserves every observation trajectory. One asymmetric sensor admits task/coordinate certificates; the same sparse proof construction scales to 128 states.

## Check received witnesses

Run `python3 repro/nonlinear-observability/oracle.py repro/nonlinear-observability/results.json` for source replay, then `python3 tools/certcheck/eshkol_certcheck.py --strict repro/nonlinear-observability/certificates.json` for the standard polynomial-identity claims. The standard checker establishes the identities; source-to-fiber binding is checked separately by the source oracle.

## Limits

Polynomial ODEs over exact Q; no restricted-domain, real-radical completeness, global-existence, or automatic Nullstellensatz-search claim. A positive task-difference power proves vanishing. A failed hint or witness yields unknown. Symmetry means indistinguishability for intervals on which the trajectories exist.

Qualification uses the published v1.3.6-evolve Linux x64 lite compiler/runtime in
Ubuntu 22.04/gcc-11. These are native library-level qualifications against pinned
feature-branch source interfaces. No new compiler build, full upstream CI/CTest,
VM/WASM, Windows, GPU or hardware qualification is implied. The small standalone
CTest project checks this cut's registration and boundary readiness separately.
