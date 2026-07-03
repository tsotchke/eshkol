# Eshkol Stress Tests

Two suites live here:

1. **P4 extreme stress harness** (adversarial testing campaign, pillar P4 —
   `.swarm/ADVERSARIAL_TESTING_CAMPAIGN.md`): budget-asserted scale/resource
   probes driven by `scripts/run_stress.sh` + `budgets.tsv`.
2. **Legacy soak harnesses** (`stress_alloc_loop.esk`,
   `stress_fd_exhaustion.esk`, `stress_parallel_at_scale.esk`): long-running
   exhaustion loops driven by `scripts/run_stress_tests.sh` (minutes–hours,
   CI-separate). Unchanged; see the bottom section.

## P4 harness — budgets asserted by the runner, not the program

```
bash scripts/run_stress.sh            # full sweep, JIT (-r) + AOT
bash scripts/run_stress.sh --quick    # CI subset (quick=1 rows, 5 jitcache runs)
bash scripts/run_stress.sh --no-aot   # JIT only
bash scripts/run_stress.sh --only sort  # substring filter
```

Every row of `budgets.tsv` (file, mode, class, wall-time ceiling, per-mode
max-RSS ceiling, expected stdout, XKNOWN ledger id) is executed under the JIT
and/or AOT and classified as `PASS / FAIL / CRASH / HANG / OVER-RSS /
OVER-TIME`. RSS comes from `/usr/bin/time -l` (max resident set size);
timeouts use `perl alarm` (macOS has no `timeout(1)`). Reference baselines on
macOS arm64: a trivial `-r` run is ~222MB RSS (stdlib object + LLVM), a
trivial AOT binary ~28MB — ceilings are sized above those floors, and every
loosened ceiling carries a comment with the measured number that justified it.

Special classes:

- `jitcache` — 50 sequential `-r` invocations of the same file against one
  fresh shared JIT cache dir; run 1 gets the cold ceiling, runs 2..50 must
  each finish inside `STRESS_WARM_CEILING_S` (default 5s).
- `rep3` — the program runs 3× per mode; all three stdouts must be
  byte-identical (spawn/join flake detector).

ICC wiring mirrors `run_sicp_smoke.sh`: `PASSED/FAILED/XFAIL/XPASS
tests/stress/<file>::<mode>` lines plus `kind:"stress_smoke"` JSON-L events in
`scripts/icc_traces/stress_smoke.jsonl`, consumed by the `stress-budget`
oracle in `.icc/completion-oracles.yaml` (summary event: `stress_suite_green`).

### Corpus layout

- `rec_*` — recursion: TCO 10⁸ in O(1) stack, non-TCO at the documented safe
  depth (250k; first failure ~270k), mutual recursion, 10k nested
  dynamic-wind, 20k CPS chain (ESH-0080 class).
- `parser_*` + `generated/parser_*` — 10k-deep parens, 10k quoted list, 1MB+
  of defines, 999-deep quasiquote template, 9.5k-char escape-mix literal.
- `data_*` — 1M list build/reverse/count, 100k vector map, 50k sort, 200k-key
  hash (compound keys), 16MB string-append, 1024×1024 matmul.
- `endur_*` — 100k gradient loop and 100k with-region loop (PR #81 class);
  the tight RSS ceilings ARE the plateau assertions.
- `par_*` — 8×heavy-closure parallel-map, mutex-serialized shared counter,
  100×spawn/join under rep3.
- `num_*` — int64→bignum boundary, 10^1000 expt round-trip, int64-range
  rationals, inf/nan propagation, 2^53 exactness edges, division-by-zero
  forms.
- `path_*` — empty program, comments-only, 1000 nested lets, 10k-char
  identifier, unicode identifiers/strings, 10k index-capturing closures.

Large mechanical sources are regenerated deterministically into `generated/`
(gitignored) by `gen_stress_sources.sh`; the runner invokes it automatically.

### found/ — minimal repros for bugs this harness discovered

Each file's header records the measured numbers and thresholds; each has a
`.swarm/tasks/ESH-NNNN.json` ledger entry and an XKNOWN row in `budgets.tsv`.
XKNOWN failures don't gate; an XKNOWN row that starts PASSING is reported as
XPASS and FAILS the gate so stale entries get promoted.

| Repro | Ledger | One-liner |
|---|---|---|
| `closure_loop_global_set.esk` | ESH-0094 | lambda in named-let loop that `set!`s a global drops every write (prints 0, expected 3) |
| `serialized_counter_10k.esk` | ESH-0094 | …so a mutex-serialized worker counter stays 0 instead of 10000 |
| `quote_sugar_in_guard.esk` | ESH-0106 | `'sym` anywhere inside `(guard …)` compiles as a variable reference; `(quote sym)` works |
| `nested_quasiquote.esk` | ESH-0107 | level≥2 quasiquote collapses to `()` |
| `list_length_1m.esk` | ESH-0108 | stdlib `length`/`filter` non-tail: SIGILL, no diagnostic, ~500k+ lists |
| `sort_100k.esk` | ESH-0098 | `sort` depth is O(n): 99999 SIGILLs, ≥100001 hits the depth guard |
| `string_nul_long_literal.esk` | ESH-0099 | NUL-bearing literal >512 source bytes decodes to wrong length/content |
| `parallel_worker_loop_20k.esk` | ESH-0100 | named-let loop in a parallel-map worker eats stack/iter: SIGBUS at ~8k iters |
| `deep_recursion_270k_no_diagnostic.esk` | ESH-0101 | ~270k-frame recursion dies SIGILL with no message |
| `mutual_tail_1e7.esk` | ESH-0102 | mutual tail calls not TCO'd; crash between 300k and 500k hops |
| `jit_deep_expr_compile_blowup.esk` | ESH-0103 | 10k-deep expr: JIT compile 35.8s/6.7GB + macro-depth spam; AOT 0.73s/93MB (doc file; enforced by the split `parser_nested_parens_10k` rows) |
| `quasiquote_long_form.esk` | ESH-0104 | `(quasiquote x)`/`(unquote x)` long forms are inert; only `` ` ``/`,` sugar works |
| `rational_bignum_exactness.esk` | ESH-0105 | exact rationals silently become doubles once a bignum appears (`(/ 1 (expt 10 19))` → `1e-19`) |

### Adding a probe

1. Drop a self-checking `.esk` in `tests/stress/` (print a unique `OK …`
   token) or a generator stanza in `gen_stress_sources.sh`.
2. Add a `budgets.tsv` row; measure first (`/usr/bin/time -l build/eshkol-run
   -r file.esk`), then set ceilings just above the measurement with a comment
   if they deviate from the defaults (384MB r / 128–160MB aot / 60s).
3. If it pins an open bug: put the repro in `found/`, add the measured
   numbers to its header, create the ledger task, and set the `xknown`
   column.

## Legacy soak harnesses

| Suite | Duration | Purpose |
|---|---|---|
| `stress_parallel_at_scale.esk` | 1-5 min | `parallel-map` / `parallel-fold` at N=1M |
| `stress_alloc_loop.esk` | 10 min | Arena alloc/free cycling — leak detection |
| `stress_fd_exhaustion.esk` | 30 s | Subprocess spawn/destroy loop — fd cleanup |

```
# Individual:
./build/eshkol-run -r tests/stress/stress_parallel_at_scale.esk

# Under ASan (rebuild sanitizer tree first):
bash scripts/build-sanitizer.sh asan
(cd build-asan && ./eshkol-run -r ../tests/stress/stress_parallel_at_scale.esk)

# Whole suite (long — intended for CI):
bash scripts/run_stress_tests.sh

# 24h soak (manual):
bash scripts/run_stress_tests.sh --hours 24 | tee 24h.log
```

The soak harness exits 0 only if every iteration succeeds and the RSS stays
within 2x the initial baseline.
