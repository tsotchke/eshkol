# Eshkol Stress Tests

Two suites live here:

1. **P4 extreme stress harness** (adversarial testing campaign, pillar P4):
   budget-asserted scale/resource probes driven by `scripts/run_stress.sh` +
   `budgets.tsv`.
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
max-RSS ceiling, expected stdout, XKNOWN tracking id) is executed under the JIT
and/or AOT and classified as `PASS / FAIL / CRASH / HANG / OVER-RSS /
OVER-TIME` (plus `XKNOWN` / `XPASS` for rows that carry a tracking id). RSS comes from `/usr/bin/time -l` (max resident set size);
timeouts use `scripts/lib/guarded_exec.pl` (macOS has no `timeout(1)`), which
stops the program together with every process it started. Reference baselines on
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

- `rec_*` — recursion: TCO 10⁸ in O(1) stack, non-TCO at 200k (the passing
  margin row) and 250k, mutual recursion, 10k nested dynamic-wind, 20k CPS
  chain.
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

### found/ — minimal repros for divergences this harness discovered

Each file's header records the numbers measured when it was filed; the
`budgets.tsv` row of each tracked one carries its tracking id in the `xknown`
column. XKNOWN failures don't gate; an XKNOWN row that starts PASSING is
reported as XPASS and FAILS the gate, so a fixed row is promoted to an
ordinary row (its `xknown` column cleared) rather than left tolerated.

At v1.3.6-evolve every repro below produces its expected output under both
the JIT (`-r`) and AOT, within its budget:

| Repro | What it showed when filed | Status at v1.3.6-evolve |
|---|---|---|
| `closure_loop_global_set.esk` | lambda in named-let loop that `set!`s a global dropped every write (printed 0, expected 3) | `OK 3`, JIT and AOT |
| `serialized_counter_10k.esk` | …so a mutex-serialized worker counter stayed 0 instead of 10000 | `OK 10000`, JIT and AOT |
| `quote_sugar_in_guard.esk` | `'sym` anywhere inside `(guard …)` compiled as a variable reference; `(quote sym)` worked | ordinary gated row, PASS |
| `nested_quasiquote.esk` | level≥2 quasiquote collapsed to `()` | ordinary gated row, PASS |
| `list_length_1m.esk` | stdlib `length`/`filter` non-tail: stopped on a signal with no diagnostic at ~500k+ element lists | `OK 1000000`, JIT and AOT |
| `sort_100k.esk` | `sort` depth was O(n): 99999 stopped on a signal, ≥100001 hit the depth guard | `OK 0`, JIT and AOT |
| `string_nul_long_literal.esk` | NUL-bearing literal >512 source bytes decoded to a different length/content | ordinary gated row, PASS |
| `parallel_worker_loop_20k.esk` | named-let loop in a parallel-map worker used stack per iteration: signal at ~8k iterations | `OK`, JIT and AOT |
| `mutual_tail_1e7.esk` | 10⁷ mutual tail calls grew the stack; resolved 2026-07-04 by emitting mutual tail calls as LLVM `musttail` | ordinary gated row, PASS (O(1) stack) |
| `deep_recursion_270k_no_diagnostic.esk` | hard gate: the default stack gives a stack-overflow diagnostic; `ESHKOL_STACK_SIZE=1G` completes 2M frames; JIT/AOT and worker variants are driven by `scripts/run_stack_overflow_diagnostic.sh` | gated by that script |
| `jit_deep_expr_compile_growth.esk` | 10k-deep expr: JIT compile 35.8s/6.7GB + macro-depth messages; AOT 0.73s/93MB (doc file; enforced by the split `parser_nested_parens_10k` rows) | the `-r` row now completes inside its budget; AOT 0.22s / 10MB |
| `quasiquote_long_form.esk` | `(quasiquote x)`/`(unquote x)` long forms were inert; only `` ` ``/`,` sugar worked | ordinary gated row, PASS |
| `rational_bignum_exactness.esk` | exact rationals became doubles once a bignum appeared (`(/ 1 (expt 10 19))` → `1e-19`) | `OK rational-bignum-exact`, JIT and AOT |

The non-TCO 250k row (`rec_deep_nontco_250k.esk`) likewise completes in both
modes at v1.3.6-evolve.

### Adding a probe

1. Drop a self-checking `.esk` in `tests/stress/` (print a unique `OK …`
   token) or a generator stanza in `gen_stress_sources.sh`.
2. Add a `budgets.tsv` row; measure first (`/usr/bin/time -l build/eshkol-run
   -r file.esk`), then set ceilings just above the measurement with a comment
   if they deviate from the defaults (384MB r / 128–160MB aot / 60s).
3. If it pins an open divergence: put the repro in `found/`, add the
   measured numbers to its header, create the tracking entry, and set the
   `xknown` column.

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
