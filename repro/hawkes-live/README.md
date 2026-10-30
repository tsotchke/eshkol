# Maintain signed Hawkes forecasts and sensitivities through DBSP

Status: JIT/AOT qualified. Selected MIT contribution by Richard Hoekstra.

Dependencies: PR 1; observable. Prerequisite implementations are outside this patch. `SOURCE.toml` records exact identities.

```sh
python3 repro/hawkes-live/run_live_gate.py --eshkol-bin /opt/eshkol/bin/eshkol-run --compile-linalg-source --results /tmp/hawkes-live.json
```


`live.esk` uses the standing aggregate in `core.dbsp.circuit`. Each immutable
row `(key x1 x2 n1 n2)` is lifted once per consolidated changed row; its weight
is a signed integer multiplicity. Admission rejects over-retraction. The
additive group carries total observation mass and all three coefficients of
each requested moment. Normalization and covariance happen after aggregation,
so mixtures include between-observation covariance. Averaging covariances of
individual rows would give a different answer.

Twelve ticks cover insertion, retraction, replacement, cancellation and empty
updates, and retain predecessor snapshots and independent forks. The exact
output deltas integrate to each new output. Cancelled/empty ticks incur zero
reported DBSP work. Sixteen output snapshots, including parameter and query
revisions, match full-source ensemble forecasts.

A parameter or query revision explicitly reconstructs from retained raw
observations and publishes a replacement delta. It is a charged global
refresh, not a claim of free incremental parameter changes. A mean-only
point-specialized model has dimension 3; a family in which w activates an
additional coupling needs 4 even at w=0. The point certificate cannot authorize
the family. The gate reconstructs at w=1/2 and checks the activated direction
against the full model.

This is a fixed-horizon forecast for a changing bag of observations. It does
not advance physical time or revise a discarded historical trajectory.
Owned live snapshots and callbacks must remain immutable in process. Their
canonical identity detects accidental mutation; it is not a signature or an
admission mechanism for hostile deserialized vectors. Import untrusted sources
through full `forecast-plan` admission, never by forging a live state vector.

The additive-group zero predicate calls `zero?` through a lambda. Passing the
builtin directly as a first-class value (`every zero?`) made the published
1.3.6 JIT crash during forecast admission when these modules were combined.
The wrapper preserves the predicate while avoiding that consumer crash; it
does not change the compiler. The full consumer now passes in-process JIT
with `ESHKOL_JIT_CACHE=0` and AOT-O0. The gate requires both routes by default,
compares their complete output, and checks all sixteen ensemble snapshots
against the full-model oracle. `--jit` remains a compatibility alias.

`results.json` records this fresh qualification, including the executable and
source identities. Earlier archived AOT-only receipts remain historical.
Full-consumer AOT-O2 exceeded the earlier 240-second budget; no success at
that optimization level is claimed.
