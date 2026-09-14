# Native compilation on an actual 8 MiB stack

The explicit parser continuations in `cf7fe34a` exposed two downstream native
stack dependencies. On a compiler built from candidate `1d201d21`, 16,000
nested additions overflowed first in `TypeChecker::synthesize` /
`synthesizeApplication`, then (after fixing synthesis) in
`codegenAST -> codegenOperation -> codegenCall -> codegenArithmetic ->
codegenTypedAST`. Both JIT and AOT traverse these paths.

Type synthesis/checking and the central LLVM expression/call/arithmetic
lowering chain now suspend child computations on the parser's explicit
continuation driver, extracted unchanged into
`inc/eshkol/util/continuation_task.h`. The parser keeps its `ParserTask` alias
and grammar implementation. Synchronous compiler APIs and backend callbacks
still return ordinary values. The recursive edges inside the converted
lowering chain await child tasks; suspended locals retain operand values,
source-provenance guards, scope state and IR insertion order. Specialized
backend helpers outside this chain retain their existing synchronous entry
points; this is not a claim that every compiler pass accepts arbitrary nesting.

No production stack size, parser safety margin, ESH-0103 depth/time/RSS limit,
or CI skip policy changed.

## Regression enforcement

`parser_stack_compile` now sets `ESHKOL_JIT_CACHE=0`: a private cache directory
alone allowed `-r` to compile and execute an AOT cache entry rather than exercise
ORC in process. Both soft and hard `RLIMIT_STACK` remain exactly 8 MiB.

On Darwin, `RLIMIT_STACK` cannot shrink the stack mapping specified by
`LC_MAIN`. The test copies the compiler, changes only that load command's stack
reservation to 8 MiB and renews the ad-hoc signature. It does the same for the
AOT executable before running it. Production executables and linker settings
remain unchanged. Linux runs the original executable under the hard limit.

The gate compiles the shipped stdlib to nonempty object and bitcode files,
compiles/runs 16,000 additions through in-process JIT and AOT, and repeats
with 16,000 alternating left/right addition/subtraction forms surrounding
observable calls. Exact output `9`, `4`, `5` checks evaluation order and
preservation of suspended operands.

## Evidence: macOS ARM64 / LLVM 21, September 8, 2026

- Unchanged candidate type checker/codegen rebuilt from `1d201d21`: strengthened
  gate fails with `jit failed on 8 MiB stack: 139` after the stdlib case passes.
- Patched compiler: all seven stdlib/JIT/AOT/operand-order stages pass with an
  actual 8 MiB executable mapping and hard resource limit.
- Parser explicit-stack CTest passes on its actual 8 MiB pthread stack.
- Unchanged ESH-0103 checker, inheriting a hard 8 MiB limit and disabling the
  JIT cache, passes every 1k/4k/16k sample. JIT times: 0.409/0.356/1.386 s;
  RSS: 83.5/154.9/412.2 MiB. AOT times: 0.147/0.332/1.344 s;
  RSS: 80.0/127.0/381.4 MiB. Maximum ratios: time 4.046, RSS 3.004 (limit 8).
- AD carrier/curried gradients, capture-shadowing, native tail-position JIT/AOT
  and ESH-0103 checker self-test: 8/8 CTests pass with the JIT cache disabled.
- Source-span type-error test fails for a missing runtime location prefix in
  both the untouched candidate and patched compiler; recorded separately from
  the passing stack regression.
- Mechanical rule-body audit: after removing task/await/return syntax, all
  fourteen converted type/codegen method bodies match their candidate originals
  (ignoring comments/whitespace). No typing or lowering rules were replaced.
- Strict type-system suite: 56 passed, 0 failed. Parser suite: 31 passed,
  0 failed. HoTT types suite: 12/13 passed; `mixed_types_stress_test.esk`
  crashes at its composed-closure invocation (address `0x18`) identically in
  the untouched candidate and patched compiler. It is a separate candidate
  regression, not hidden as a passing suite.

A Linux runner was not available locally (Docker daemon unavailable), so these
are measured Darwin results with Linux-equivalent stack enforcement, not a
claim of Linux execution. The existing Linux x64 lite CI gate runs this same
strengthened test.

ICC was queried first: `eshkol_lang` refused context as BLIND (missing stores),
and `eshkol-astra-v135` refused it as STALE (56 commits behind). Diagnosis uses
live source, a crash-frame trace, exact-candidate baseline builds and executable
results rather than a stale ICC readiness verdict.
