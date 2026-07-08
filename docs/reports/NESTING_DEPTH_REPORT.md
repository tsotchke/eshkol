# Syntax / Data Nesting-Depth-Parametric Report (P6c)

Pillar P6c of the depth-parametric adversarial campaign
(`.swarm/DEPTH_PARAMETRIC_TESTING.md`). Every composable syntactic / data
construct is nested **parametrically** at nesting depth `d = 1,2,4,8,16,32,64,128,256`
and each cell is verified two ways at once:

1. **Closed-form oracle** — the generator bakes the exact expected integer into
   the probe's directive (`; VALUE: n`); the accumulate-by-level constructs sum
   to `d(d+1)/2`, the application chain to `d`.
2. **Differential across three execution axes** — JIT (`-r`), AOT `-O0`, AOT
   `-O2`. The runner compares the value each axis produces both against the
   closed form and against the other axes.

Each cell is classified:

| class | meaning |
|---|---|
| **PASS** | every axis returned the closed-form value |
| **WRONG** | all three axes agree with each other but the value is wrong (a silent miscompile only the closed form catches) |
| **AXIS-DIVERGENCE** | the axes disagree, or one axis returns a value while another hits a boundary/crash (`-r` vs `-O0` vs `-O2`) |
| **LIMIT** | a clean, consistent boundary on every axis (diagnosed crash or timeout) — a documented capability boundary |
| **SILENT-CRASH** | an undiagnosed failure (clean exit with no output, or fatal signal with no diagnostic) on some axis |

MCD = max-correct nesting depth = largest tested depth whose cell is PASS.

## Result — gate PASS

```
total=162  pass=160  limit=2  wrong=0  diverge=0  silent_crash=0
```

**No silent wrong value, no `-r`/`-O0`/`-O2` divergence, and no silent crash at
any tested nesting depth.** The only non-PASS cells are the two deepest
`lambda_chain` cells (`d=128`, `d=256`), which hit a *clean, diagnosed* capture
limit (see findings). Curried chains were previously correct only to depth 16
(with `d=32,64,128,256` all failing); the ESH-0210 fix raised that to **64**, so
`d=32` and `d=64` are now PASS.

## Per-construct max-correct nesting depth

| construct | family exercised | oracle | MCD | first-LIMIT | max-d tested |
|---|---|---|---|---|---|
| quote_nest | nested `'(...)` literal, walker-summed | tri | **256** | - | 256 |
| quasiquote_nest | nested `` `(,x ...) `` unquote at every level | tri | **256** | - | 256 |
| unquote_splice | nested `` `(,x ,@`(...)) `` quasiquote + splicing | tri | **256** | - | 256 |
| let_nest | nested `let` | tri | **256** | - | 256 |
| let_star_nest | nested `let*` | tri | **256** | - | 256 |
| letrec_nest | nested `letrec` | tri | **256** | - | 256 |
| letrec_star_nest | nested `letrec*` | tri | **256** | - | 256 |
| lambda_chain | curried `((((lambda…)1)2)…)` chain | tri | **64** | 128 | 256 |
| closure_capture | nested lets each closing over the previous closure | tri | **256** | - | 256 |
| vector_nest | nested `(vector …)`, walker-summed | tri | **256** | - | 256 |
| list_nest | nested `(list …)`, walker-summed | tri | **256** | - | 256 |
| if_nest | nested `if` | tri | **256** | - | 256 |
| cond_nest | nested `cond` | tri | **256** | - | 256 |
| case_nest | nested `case` | tri | **256** | - | 256 |
| begin_nest | nested `begin` (`set!` accumulator) | tri | **256** | - | 256 |
| guard_nest | nested `guard`, each handler re-raises accumulating | tri | **256** | - | 256 |
| dynamic_wind_nest | nested `dynamic-wind`, before-thunks accumulate | tri | **256** | - | 256 |
| app_chain | deep application `(f (f (f … x)))` | dep | **256** | - | 256 |

17 of the 18 constructs are correct on all three axes to the maximum tested
depth (256). The MCD of 256 is the ladder ceiling, not an observed failure — ad
hoc probing shows `let_nest` correct past 1024 and `app_chain` past 2048; the
ladder simply stops at 256.

## Findings

There are **no silent-wrong / miscompile / axis-divergence findings** — the goal
class of this pillar. Every axis agrees with every other axis and with the
closed form at every depth that produced a value.

One capability boundary was found here and has since been **fixed** (tracked as
**ESH-0210**; originally mis-filed as ESH-0185, which is reserved for the AD
campaign P0). It was a *clean, diagnosed* boundary — never a silent bug.

### ESH-0210 — deeply curried lambda-application chains crashed from depth 18 (FIXED)

Minimal repro (`tests/nesting_depth/generated/nest_lambda_chain_d18.esk`,
regression-guarded by `tests/closures/curried_chain_depth_test.esk`):

```scheme
; before the fix: correct through 17, SIGSEGV from 18
; after  the fix: correct through 65, diagnosed limit from 66
(display
 ((((((((((((((((((
   (lambda (x1)(lambda (x2) ... (lambda (x18)
     (+ x1 x2 x3 ... x18)) ...))
   1) 2) 3) ... 18))
(newline)
```

**Actual root cause (the original `O(d²)`-codegen-stack-overflow hypothesis was
wrong — the compiler never overflowed; it was a *runtime* miscompile).** The
innermost body `(+ x1 … xd)` forces the innermost lambda to capture all `d-1`
enclosing parameters, so its closure carries `d-1` captures. Every closure call
site lowers to an inline `(arg_count × capture_count)` dispatch matrix in
`codegenClosureCall`, and the capture dimension was capped at
`MAX_CLOSURE_DISPATCH_CAPTURES = 16`. The matched dispatch index is
`arg_count·(MAX+1) + num_captures`; when `num_captures` exceeded `16` the raw
index **silently aliased into a valid but wrong `(arg,cap)` case** (e.g. 17
captures with 1 arg → index 34 → the `(2 args, 0 captures)` case). The target
lambda was then called through a *mismatched* LLVM signature — capture pointers
landed in the wrong registers/stack slots and the callee dereferenced a
garbage/null capture → `SIGSEGV` (the fatal-signal handler guessed “stack
overflow”, hence the original misdiagnosis; it was actually a null/garbage
capture-pointer load). Depth 18 ⇒ 17 captures ⇒ first count past 16.

`closure_capture`, `app_chain` and the `let*/letrec*` families stay correct to
256 because their closures capture few variables — they never cross the 16-capture
dispatch ceiling.

**The fix** (`lib/backend/llvm_codegen.cpp`, `codegenClosureCall`):

1. Replace the per-`(arg,cap)` dispatch matrix with **over-provisioning** of
   captures. The call site now switches only on the *argument* count; each arm
   passes *all* `MAX_CLOSURE_DISPATCH_CAPTURES` capture pointers (contiguous env
   slots) and the callee — compiled with exactly its own `N ≤ MAX` capture-
   pointer parameters — reads only the first `N`. Captures follow the fixed args
   positionally, so as long as the arg count matches, the callee’s slots line up
   with the first `N` pointers passed; the extra trailing pointers are ignored,
   ABI-safe on caller-cleanup ABIs (AArch64 AAPCS, x86-64 SysV) — the same trick
   the existing *argument* padding already relied on. The over-count capture GEPs
   are pure address arithmetic, never dereferenced.
2. Raise `MAX_CLOSURE_DISPATCH_CAPTURES` from 16 to **64** → curried chains
   correct to depth **65** (innermost captures 64). Because captures are
   over-provisioned rather than *switched on*, this costs **zero** extra dispatch
   cases: exactly one call per arg arm regardless of the ceiling.
3. Add a **capture-overflow guard**: `num_captures > 64` branches to a diagnosed
   runtime error (`eshkol_raise`, “closure capture limit exceeded …”, non-zero
   exit) instead of the callee reading an unpassed capture pointer — so beyond 65
   is a *clean bounded limit*, never a crash.

The intermediate design (keep the matrix, just raise the ceiling to 64) was
rejected: a `(arg × cap)` matrix emits `O(MAX_CAP)` distinct-signature calls per
call site, which `-O2` cannot merge — it made `-O2` compilation of closure-heavy
code (256 nested closures) **~18× slower** (13 s → 237 s, enough to time out and
show as an axis divergence) and blew the `-O0` frame (regressing the deeply
recursive SICP `ch4_amb_evaluator`). Over-provisioning removes the capture
dimension entirely, so `-O2` of that same 256-closure probe actually got a touch
*faster* than baseline (13 s → 8 s).

Residual limit / deeper fix: 64 captures is now a *documented, diagnosed* ceiling
rather than a crash. Removing it entirely means changing the closure ABI to pass
the environment base pointer once and have the callee index captures from it
(instead of N individual capture-pointer parameters), dropping the per-call
argument-padding fan-out too — scoped as a follow-up.

## Reproduce

```sh
cmake --build build --target eshkol-run stdlib -j8
scripts/run_nesting_depth.sh          # full sweep + gate (regenerates corpus)
scripts/run_nesting_depth.sh --quick  # drops the d=128,256 cells
```

The runner emits `kind:"nesting_depth"` ICC events (per-cell verdicts,
`max_correct_<construct>`, and the final `nesting_depth_gate`) to
`scripts/icc_traces/nesting_depth.jsonl`, consumed by the additive
`nesting-depth` oracle in `.icc/completion-oracles.yaml` (also folded into the
`v1.3-evolve` release-readiness depth-campaign block).

## Disk budget

Per the mandatory budget: each cell compiles to a single reused temp binary
(`$TMPDIR/nesting-depth.XXXX/probe.bin`) deleted after every run; only an
on-disk `artifacts/nesting-depth/` (git-ignored, hard-capped at 1 GB with a
`du` check that aborts) is retained. Observed peak this run: **12 KB** artifacts
+ 768 KB generated corpus (git-ignored). An on-exit trap removes the temp
workdir.
