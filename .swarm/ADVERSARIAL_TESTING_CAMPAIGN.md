# Adversarial Testing Campaign — permanent bug-finding infrastructure (2026-07-02)

Premise: every bug found in the v1.3 campaign lived at a COMPOSITION point
(closure×set!, letrec×instances, first-class×predicate, quote×let-tail,
gradient×derivative×vector-param, define×libc-symbol). Unit tests structurally
miss these. Users must never be the discovery mechanism again.

Four permanent pillars, each a repo-committed, rerunnable harness wired into ICC
as oracles (traces → `icc readiness`/`completion-oracle`; findings → ESH tasks):

## P1 — Differential execution testing (`scripts/run_differential.sh`, tests/differential/)
Eshkol has THREE execution paths (JIT `-r`, AOT binary, VM/ESKB) plus O0/O2 and
run-cache on/off. Identical program + identical input ⇒ identical output, across
every axis. ANY divergence is a bug by definition — no external oracle needed.
Includes a seeded random-program generator (bounded depth, all core forms) with
automatic shrinking of divergent cases to minimal repros.

## P2 — Feature-pair composition matrix (`scripts/run_edge_matrix.sh`, tests/edge_matrix/)
Enumerate the language surface as ~30 feature axes (let-family, lambda/closures,
set!, quote/quasiquote/vector-literals, TCO, named-let, call/cc, dynamic-wind,
guard/raise, define/internal-define, first-class builtins, apply/map/for-each,
strings, chars, numeric tower (int/rational/bignum/double/complex), vectors,
hash-tables, streams/delay, macros, modules/require, keyword args, let-values,
match, FFI/extern, parallel-map, tensors, AD ops, regions, error objects,
tail position, shadowing). Systematically generate PAIRWISE probes (A used inside/
around B) with hand-checkable expectations. Bugs live at intersections; cover
all ~900 ordered pairs over time, committed as a growing corpus.

## P3 — AD composition oracle (`scripts/run_ad_oracle.sh`, tests/ad_oracle/)
The AD surface is a matrix: {derivative, gradient, jacobian, hessian, divergence,
curl, laplacian} × {scalar, vector-param, tensor} × {inline lambda, named fn,
closure captures: none/local/global/vector-ref} × {nesting: none, self, mixed
forward/reverse} × {reuse: single, loop-iterated}. EVERY cell is auto-checked
against central finite differences (the ground truth needs no hand computation).
ESH-0093 (mixed-mode) would have been caught by this in minutes.

## P4 — Extreme stress harness (`scripts/run_stress.sh`, tests/stress/)
Scale + resource dimensions with hard budgets (RSS ceilings, wall-time ceilings,
exit-code assertions): deep recursion (TCO and non-TCO), deep nesting (parser),
wide data (100k-element vectors/strings/lists, 7168-dim tensors), iteration
endurance (50k+ AD/alloc loops), arena/region pressure, parallel-map races
(N workers × M serialized mutations), numeric extremes (2^63 boundaries, bignum
growth, inf/nan propagation, exactness edges), pathological inputs (NUL bytes,
huge symbols, unicode, empty everything).

## ICC wiring (all pillars)
- Each runner emits `PASSED/FAILED nodeid::name` + `{"kind":"..."}` trace lines
  into scripts/icc_traces/ (same pattern as run_sicp_smoke.sh).
- New completion-oracle targets in .icc/completion-oracles.yaml:
  `differential-clean`, `edge-matrix`, `ad-oracle`, `stress-budget`.
- `icc audit-to-tasks` on findings; every divergence/crash becomes an ESH task
  with its shrunken minimal repro.
- Addresses ICC weakness-map #1 (adversarial scenario evals).

## Operating cadence
- Every runner is deterministic-seeded and CI-able; a nightly scheduled lane
  runs all four with a fixed time budget (follow-up: .github/workflows/nightly-adversarial.yml).
- Findings ledger: .swarm/tasks/ESH-NNNN.json per confirmed bug, minimal repro mandatory.
