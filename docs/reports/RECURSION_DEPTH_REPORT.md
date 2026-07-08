# Recursion / control depth-parametric sweep (pillar P6b)

Date: 2026-07-04 · branch `fix/mutual-tail-tco` (base master `aa0e71a0`) · macOS
arm64 (M-series), default stacks · JIT (`-r`) and AOT.

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

Gate: PASS. `total=108 pass=80 clean_limit=14 xknown=14 fail=0` (108 = 54 probes ×
{-r, AOT}). No silent crash and no wrong value on any non-XKNOWN cell.

Update 2026-07-04 (ESH-0102 fixed + non_tail recalibrated): mutual tail calls are
now proper R7RS tail calls (emitted as LLVM `musttail`), so `mutual_tail2` and
`mutual_tail3` PASS to 5,000,000 hops on both lanes (O(1) stack, previously a
~200k ceiling that became a FAIL after the P7 SIGILL-altstack fix turned the
overflow into a caught CLEAN-LIMIT on a `pass` cell). `non_tail` deep cells are
recalibrated from `pass`/`xknown` to `limit`: non-tail recursion is legitimately
bounded, so a caught diagnostic ceiling is CORRECT degradation, not a bug.

## Max safe depth per KIND (largest PASS depth, both lanes)

| KIND | max safe depth | boundary behavior above | tracked bug |
|---|---:|---|---|
| self_tail | **100,000,000+** | O(1) stack — 1e8 correct, no ceiling found | — |
| mutual_tail2 (2-cycle) | **5,000,000+** | O(1) stack — proper tail call via `musttail`, no ceiling found | ESH-0102 (fixed) |
| mutual_tail3 (3-cycle) | **5,000,000+** | O(1) stack — proper tail call via `musttail`, no ceiling found | ESH-0102 (fixed) |
| non_tail | **100,000** | CLEAN diagnostic ceiling above (SIGBUS caught, rc≠0) — bounded by design | ESH-0112 (depth only) |
| cps / continuation chain | **50,000** | CLEAN — `maximum recursion depth (100000) exceeded`, rc1 | ESH-0080 (fixed #93) |
| through_map (higher-order) | **100,000** | CLEAN — 100000 guard, rc1 | — |
| metacircular (interpreted eval) | **10,000** | SILENT SIGILL by 30k | ESH-0119 (new) |
| dynamic_wind nesting | **90,000** | ~100k overflow: SIGBUS diagnostic OR silent SIGILL (nondeterministic) | ESH-0119 (new) |
| callcc nesting | **100,000** | CLEAN — 100000 guard, rc1 | — |
| guard nesting | **150,000** | SILENT SIGILL by 200k | ESH-0119 (new) |
| stdlib length (long list) | **300,000** | SILENT SIGILL by 500k (non-tail `length`) | ESH-0108 |

Self-tail AND mutual-tail recursion are properly O(1): both return the exact
triangular sum at 10^8 / 5×10^6 hops on both lanes (mutual recursion is a proper
R7RS tail call as of the ESH-0102 fix — LLVM `musttail`). The constructs that
genuinely keep a frame per level (non-tail recursion, nested control) have a
finite native-stack ceiling by design; the CPS / map / call-cc paths route
through the runtime recursion guard and degrade CLEANLY at 100000, while
non-tail, guard, metacircular and stdlib-length overflow the raw native stack at
their respective depths.

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

### FIXED — ESH-0102 (mutual tail calls now proper R7RS tail calls)

Non-self calls in tail position are now emitted as LLVM `musttail` (see the
mutual-TCO block in `codegenCall`, `lib/backend/llvm_codegen.cpp`) instead of the
old `TCK_Tail` hint that the backend ignored. The `musttail` is guarded by a
matching signature/calling-convention and the absence of pointer-into-frame
arguments — the latter being exactly what made LLVM's X86 backend fatally reject
closure-forwarding stdlib wrappers, so those fall back to an ordinary bounded
call. 2- and 3-cycle mutual recursion now run in O(1) stack to 5,000,000 hops on
both lanes; `even?`/`odd?` to 10^7 ok; stdlib builds clean. The "state machine as
mutually-tail-calling functions" idiom no longer has a hidden ceiling. Remaining
limitation: a higher-order tail call forwarding a stack-allocated closure
argument is not `musttail`'d (bounded, use the trampoline).

### Re-confirmed existing (tracked, XKNOWN)

- **ESH-0112** non-tail recursion — bounded by design; caught SIGBUS diagnostic
  (CLEAN-LIMIT) above ~100k. Recalibrated to `limit` cells (a clean ceiling is
  accepted; silent crash / wrong value still fails the gate).
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
