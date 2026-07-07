# Eshkol v1.3.0-evolve

The "evolve" release: full SICP coverage, exact nested/mixed-mode automatic
differentiation, closure-semantics completion, and every CI platform green —
backed by a new permanent adversarial-testing infrastructure.

## Release gates (all green on the release SHA)
- ICC readiness `v1.3-evolve`: ready, score 100/100 (trace-verified oracle)
- CI: 14/14 lanes including windows-arm64 lite/cuda/xla (first all-green matrix)
- SICP full-book gate: 88/88 probes across all 5 chapters, `-r` AND AOT

## Highlights
- **All of SICP** — ch1–5 corpora incl. metacircular/analyzing/lazy/amb evaluators,
  query system, register machines (stack/recursive), stop-and-copy GC model,
  explicit-control evaluator, and the SICP compiler (#85, #87, #102, #105–#109).
- **Automatic differentiation** — exact iterated/nested higher-order AD (#84, #95),
  mixed-mode vector-gradient-over-derivative (#113), AD-visible min/max (#82),
  centralized tensor type-guards (#79), conv2d unified across codegen/VM (#80).
- **Closures & semantics** — shared set!-mutated captures (#83), per-activation
  letrec instances (#89), first-class/apply'd predicate booleans (#86), quote-token
  dispatch across all clause parsers (#110, #117), single-evaluation of constructor
  operands (#116), R7RS numeric tower exactness/promotion (#82).
- **Compiler & types** — precompiled-module external defines no longer have their
  bodies scanned by the type checker or nested-function declaration pass (#101).
- **Platforms** — windows-arm64 unbroken across 3 layers (sincos, thunk convergence,
  SEH-safe dead-strip; #77); with-region memory routing 6.6GB→41MB (#81).
- **Native agent/ML substrate** — EAGLE linear-head FFI training (#104), 7168-dim
  vector gradients exact, durable hash-chained memory store.
- **Adversarial infrastructure** (new, permanent): multi-path differential harness +
  fuzzer (#114), feature-pair edge matrix (#112), AD finite-difference oracle (#111),
  stress harness with RSS/time budgets (#115), VM parity ratchet + manifest (#118) —
  all wired into ICC readiness.

## Known issues (tracked, non-blocking)
The adversarial harnesses shipped in this release found and now track a backlog of
edge-case findings (.swarm/tasks/): libc-symbol global naming (ESH-0092/0103),
char degradation via apply/write (ESH-0099), vector gradient-of-gradient (ESH-0096),
hessian on tensor points, stdlib non-tail depth limits, rational→double contagion
at bignum boundaries, and 27 bytecode-VM divergences + 351 VM parity gaps
(tests/vm_parity/PARITY.tsv). Each has a minimal repro and a ledger entry.
The stdlib verification batch (ESH-0110) additionally tracks csv-parse row order,
a partial-application SIGBUS, and an fft chain issue, and ESH-0111 tracks findings
from the AD/tensor reference verification — both documented with repros in the
reference pages (docs/reference/stdlib, docs/reference/ad).
