# Sweep E — Re-verification + Full Gate Matrix Report

- Date: 2026-07-03
- Base: current `master` = `0733dc9b` (contains #110, #112–#118)
- Branch: `test/sweep-e-verification`
- Platform: macOS (Darwin 24.1.0) arm64; timeouts via `perl -e 'alarm N; exec'`
- Build: fresh `cmake --build build --target eshkol-run stdlib -j` (exit 0),
  plus `eshkol-vm-standalone-test` for the VM-parity gate.
- Scope: verification, triage, ledger + harness-expectation updates only.
  **No compiler source was modified.**

---

## TASK 1 — Re-verification of open findings

Every repro was run under JIT (`-r`) **and** AOT on the fresh build. Verdicts:

### FIXED (verified -r AND AOT)

| Finding | Ledger | Repro | Evidence | Fixed by |
|---|---|---|---|---|
| Quote sugar `'sym` in `guard` clause/raise | ESH-0106 | `stress/found/quote_sugar_in_guard.esk` | `OK quote-in-guard` both paths | #117 quote-dispatch-family |
| Nested quasiquote (level ≥ 2) | ESH-0107 | `stress/found/nested_quasiquote.esk` | `OK nested-quasiquote` both paths | #117 |
| `(quasiquote …)`/`(unquote …)` long form | ESH-0104 | `stress/found/quasiquote_long_form.esk` | `OK quasiquote-long-form` both paths | #117 |
| Differential 001 — guard-caught value garbage/divergent | (guard family) | `differential/found/001…` | all axes → `boom` | #117 |
| Differential 002 — 2nd guard SIGSEGV at -O1+ | (guard family) | `differential/found/002…` | all axes rc 0 → `x` | #117 |
| EM-1 quote in let-body tail truncates | edge/EM1 | pair007/008 | pairs PASS | #117 / #229 family |
| EM-2 quote in guard clause = undefined var | edge/EM2 | pair096/098-101 | pairs PASS | #117 |
| EM-3 match quote-clause SEGV | edge/EM3 | pair144-149 | pairs PASS | #117 |
| EM-4 `(quasiquote…)`/`(unquote…)` long form + nested | edge/EM4 | pair005/006/009/010 | pairs PASS | #117 |
| EM-6 top-level constructor arg re-evaluated | edge/EM6 | pair070/082 | pairs PASS | #116 eval-once |
| EM-7 write char from list | edge/EM7 | — | `#\a / (1 #\b) / #\b` | (already fixed) |

### STILL OPEN (re-verified against current master)

| Finding | Ledger | Current behaviour |
|---|---|---|
| Global `set!` from a function lost on cached-JIT & AOT | ESH-0094 | `closure_loop_global_set` → `OK 0` (want 3); `serialized_counter_10k` → `OK 0`; **differential 003** diverges: uncached-JIT `(a)` vs cached-JIT/AOT `()`. Also the sole divergent corpus program (`34_boolean_shortcircuit`). Release-blocking. |
| NUL-string literal >512 src bytes wrong length + path divergence | ESH-0099 | `-r` → `OK 284`, AOT → `OK 276` (want 300); JIT≠AOT. |
| Exact rational degrades to double once bignum involved | ESH-0105 | `FAIL` both paths. |
| stdlib `length`/`filter` non-tail SIGILL on big lists | ESH-0108 | `list_length_1m` SIGILL (rc 132) after `OK `. |
| Deep non-tail recursion SIGILL, no diagnostic | ESH-0101 | `deep_recursion_270k` SIGILL (rc 132). Safe depth **shrank** (see ESH-0112). |
| Mutual tail calls not TCO'd | ESH-0102 | `mutual_tail_1e7` SIGILL (rc 132). |
| stdlib `sort` O(n) recursion depth | ESH-0098 | `sort_100k` → `maximum recursion depth (100000) exceeded` (graceful diagnostic). |
| parallel-map worker loops eat stack | ESH-0100 | `parallel_worker_loop_20k` CRASH (rc 132), no output. |
| EM-5 `apply` degrades char args to raw ints | ESH-0113 (new) | `EM5…` → `97 #f` (want `a #t`); pair060/065 fail both modes. |
| JIT deep-nested-expr compile blowup | ESH-0103 | Not re-run (35 s / 6.7 GB repro); remains pinned XKNOWN via `budgets.tsv` (AOT row passes). |

### Ledger / corpus / expectation updates

- `.swarm/tasks/ESH-0104/0106/0107.json` → **status `done`** + sweep-E evidence.
- `.swarm/tasks/ESH-0094/0098/0099/0100/0101/0102/0105/0108.json` → sweep-E
  `still-open` verification note appended.
- Fixed differential repros 001/002 **moved into the corpus** as
  `corpus/41_guard_value_and_double.esk` (all axes agree). 003 kept in `found/`
  (still divergent).
- Fixed stress rows `quote_sugar_in_guard`, `nested_quasiquote`,
  `quasiquote_long_form` **promoted** from XKNOWN → normal gating rows.
- Edge-matrix `KNOWN_FAILURES.txt` pruned: the EM-1/2/3/4/6 families (26 rows)
  now PASS and were removed; only EM-5 (and new EM-8) remain allowlisted.

---

## TASK 2 — Full gate matrix (current master)

| Gate | Result | Exit | Notes |
|---|---|---|---|
| `run_v1_3_readiness.sh` | **PASS** | 0 | verifyModule clean, JIT REPL clean exit, stdlib rebuilds clean |
| `run_sicp_smoke.sh` | **PASS** | 0 | 88/88 probes, 0 xfail, 0 XPASS |
| `run_differential.sh` (corpus) | **RED (known-open)** | 1 | 243/246 axis pairs agree across 41 programs; the **1** divergent program is `34_boolean_shortcircuit` = open **ESH-0094**. New `corpus/41` agrees on all 6 axis pairs. No compiler fix in scope. |
| `run_differential_fuzz.sh --seed 43 --count 100` | **PASS** | 0 | 100 agree, 0 diverge (new seed) |
| `run_edge_matrix.sh` (300 pairs) | **PASS** | 0 | 576 file×mode: 560 PASS, 16 allowlisted (EM-5 ×4 + EM-8 ×12), **0 unexpected** |
| `run_ad_oracle.sh` | **PASS** | 0 | total 80: 46 pass, 34 XKNOWN (ESH-0095 nested-AD), 0 fail/crash |
| `run_stress.sh --quick` | **PASS** | 0 | 58/58 PASS, 8 XKNOWN, **0 XPASS** (after re-calibration) |
| `run_vm_parity.sh` | **PASS** | 0 | audit: 912 codegen symbols all VM-supported/waived; 56 parity probes pass, 0 fail |
| tests/parser (`run_parser_tests.sh`) | **PASS** | 0 | 30/30 |
| tests/codegen (`run_codegen_tests.sh`) | **PASS** | 0 | 3/3 |
| tests/ad (`run_autodiff_tests.sh`) | **PASS** | 0 | 54/54 |
| tests/ml (`run_ml_tests.sh`) | **PASS** | 0 | 41/41 |
| tests/closures (direct -r + AOT) | **PASS** | 0 | 40/40 + 24/24, both paths |
| tests/predicates (direct -r + AOT) | **PASS** | 0 | 33/33, both paths |

The only non-green gate is `run_differential.sh`, and it is red **by design**: it
has no allowlist and correctly reports the open ESH-0094 global-`set!` divergence.

---

## TASK 3 — Edge-matrix tranche 2 (next 150 pairs)

- Regenerated to 300 priority pairs (`gen_matrix.py --max-pairs 300`): first 150
  byte-identical, pairs 150–299 are new (288 files after type-incompatible skips).
- Ran all 300 under JIT + AOT.
- Triage of the new tranche: **1 real bug**, everything else PASS.
  - `numeric_tower` pairs 156–161 (both modes, 12 rows) fail with
    `expected=1/3 actual=1/3` — the `edge-chk` uses `(equal? actual expected)`,
    exposing a genuine compiler bug (below). Not a generator defect: the
    generator's use of `equal?` is correct R7RS. Allowlisted to ESH-0114.

---

## New bugs found this sweep

| ID | Severity | Title |
|---|---|---|
| **ESH-0114** | high | `eqv?`/`equal?` on **rational** and **complex** numbers return `#f` for numerically identical operands (fall through to pointer compare; `=` and bignums are fine). `(eqv? 1/3 1/3)` → `#f`. JIT+AOT identical. Repro `edge_matrix/found/EM8_eqv_equal_rational_complex.esk`. |
| **ESH-0112** | medium | Non-tail recursion safe depth **regressed**: `rec_deep_nontco_250k` (a normal margin row) now SIGILLs. Current build: JIT 240k ok / 250k SIGILL; AOT 200k ok / 250k SIGILL (was 270k/280k first-fail). Margin row re-calibrated 250k→200k; 250k retained as ESH-0112 crash pin. |
| **ESH-0113** | high | `apply` degrades char args to raw ints (EM-5 promoted to ledger; was tracked only in edge-matrix allowlist). |

## Fuzzing

`run_differential_fuzz.sh --seed 43 --count 100`: 100/100 programs agree across
all native axes, 0 divergences — no new bugs from the new seed.
