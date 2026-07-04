# Recursion / control depth-parametric sweep (pillar P6b)

Date: 2026-07-03 · master `1ee11b06` · macOS arm64 (M-series), default stacks ·
JIT (`-r`) and AOT.

Harness: `scripts/gen_recursion_depth.py` (generator) +
`scripts/run_recursion_depth.sh` (runner) + `tests/recursion_depth/generated/`
(54 probes). ICC trace: `scripts/icc_traces/recursion_depth.jsonl`
(kind `recursion_depth`). Oracle target `recursion-depth` in
`.icc/completion-oracles.yaml`.

For each recursion/control KIND a depth ladder is swept to find and gate the
maximum safe depth under BOTH lanes. Every probe computes a closed-form value
(triangular sum `N(N+1)/2`, or list length `N`) and self-checks it in-language,
so a WRONG value is caught, not just a crash.

## Verdict taxonomy

| verdict | meaning | gate |
|---|---|---|
| PASS | value correct, clean exit | ok |
| CLEAN-LIMIT | nonzero/fatal exit WITH a diagnostic (recursion-depth guard, `[Eshkol] fatal signal` handler) | ok (documented boundary) |
| XKNOWN | tracked SILENT crash / wrong value tied to an ESH task | tolerated |
| SILENT-CRASH | fatal signal (SIGILL rc132) with NO diagnostic | **BUG (gate fail)** |
| WRONG-VALUE | ran, exit 0, wrong answer | **BUG (gate fail)** |

Gate: PASS. `total=108 pass=74 clean_limit=8 xknown=26 fail=0` (108 = 54 probes ×
{-r, AOT}). No silent crash and no wrong value on any non-XKNOWN cell.

## Max safe depth per KIND (largest PASS depth, both lanes)

| KIND | max safe depth | boundary behavior above | tracked bug |
|---|---:|---|---|
| self_tail | **100,000,000+** | O(1) stack — 1e8 correct, no ceiling found | — |
| mutual_tail2 (2-cycle) | **200,000** | SILENT SIGILL by 300k (not TCO'd) | ESH-0102 |
| mutual_tail3 (3-cycle) | **200,000** | SILENT SIGILL by 300k (not TCO'd) | ESH-0102 |
| non_tail | **200,000** | SILENT SIGILL by 250k | ESH-0112 |
| cps / continuation chain | **50,000** | CLEAN — `maximum recursion depth (100000) exceeded`, rc1 | ESH-0080 (fixed #93) |
| through_map (higher-order) | **100,000** | CLEAN — 100000 guard, rc1 | — |
| metacircular (interpreted eval) | **10,000** | SILENT SIGILL by 30k | ESH-0119 (new) |
| dynamic_wind nesting | **90,000** | ~100k overflow: SIGBUS diagnostic OR silent SIGILL (nondeterministic) | ESH-0119 (new) |
| callcc nesting | **100,000** | CLEAN — 100000 guard, rc1 | — |
| guard nesting | **150,000** | SILENT SIGILL by 200k | ESH-0119 (new) |
| stdlib length (long list) | **300,000** | SILENT SIGILL by 500k (non-tail `length`) | ESH-0108 |

Self-tail recursion is properly O(1): 10^8 iterations return the exact
triangular sum on both lanes. Everything that keeps a frame per level (non-tail,
mutual-tail hops, nested control) has a finite ceiling; the CPS / map / call-cc
paths route through the runtime recursion guard and degrade CLEANLY at 100000,
but mutual-tail, non-tail, guard, metacircular and stdlib-length overflow the
raw native stack.

## Findings

### NEW — filed ESH-0119 (SIGILL-from-stack-overflow is not caught)

The fatal-signal handler catches SIGBUS/SIGSEGV and prints
`[Eshkol] fatal signal: … — terminating` (nonzero exit = an acceptable
diagnostic boundary), but **SIGILL slips through**, so a stack-exhaustion crash
that manifests as SIGILL dies SILENTLY (rc132, no message). Which manifestation
occurs at the boundary is nondeterministic. New silent instances surfaced and
gated by this sweep:

- **deep guard nesting** — safe to 150,000; SILENT SIGILL by 200,000 (both lanes).
- **metacircular-evaluator recursion** — interpreted recursion safe to 10,000;
  SILENT SIGILL by 30,000 (the eval/apply loop is host-non-tail).
- **deep dynamic-wind nesting** at ~100,000 — flaps between a caught SIGBUS
  diagnostic and a silent SIGILL across runs.

This is the same architectural root that makes ESH-0102 / ESH-0108 / ESH-0112
silent. Proposed fix: `sigaltstack` + register SIGILL alongside SIGBUS/SIGSEGV
so every stack-exhaustion crash is diagnosed; ideally extend the graceful
recursion-depth guard (already covering the CPS/map/call-cc path) to plain user
recursion so these become catchable errors.

### Re-confirmed existing (tracked, XKNOWN)

- **ESH-0102** mutual tail calls not TCO'd — 2- and 3-cycle both silent-SIGILL
  by 300k (safe 200k). Confirms the "state machine as mutually-tail-calling
  functions" ceiling.
- **ESH-0112** non-tail recursion — silent-SIGILL by 250k (safe 200k).
- **ESH-0108** stdlib `length` non-tail — silent-SIGILL by 500k (safe 300k).

No WRONG-VALUE was observed at any depth on any kind: every value that came back
was the exact closed form.

## Running

```
cmake --build build --target eshkol-run stdlib -j
scripts/run_recursion_depth.sh              # full sweep, -r + AOT
scripts/run_recursion_depth.sh --quick      # skip slowest self_tail/length cells
scripts/run_recursion_depth.sh --no-aot     # JIT only
scripts/run_recursion_depth.sh --regen      # regenerate corpus first
```
