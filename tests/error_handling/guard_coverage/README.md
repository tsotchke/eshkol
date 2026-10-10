# `guard` coverage

R7RS `guard` had two kinds of coverage before this directory, and neither was
complete.

**One engine.** `tests/error_handling/*.esk` are the `guard` tests, and
`scripts/run_error_handling_tests.sh` compiles each one AOT and runs the
binary. No `guard` program in the repo was ever executed under `-r` or on the
bytecode VM, so a form that worked in one engine and not another had nothing
watching it.

**Differential, not golden.** The harnesses that *do* run several engines — the
P1 differential (`tests/differential/`) and the P5 parity ratchet
(`tests/vm_parity/`) — compare the engines to *each other*. A clause form both
backends handle the same non-R7RS way compares byte-equal and passes: a
shared divergence is invisible to a differential.

This directory closes both at once: every case is graded against a
**hand-authored golden** derived from R7RS 4.2.7, on **five axes**.

| axis | command |
|---|---|
| `jit` | `eshkol-run -r f.esk` |
| `aot-o0` | `eshkol-run -O0 f.esk -o bin && ./bin` |
| `aot-o2` | `eshkol-run -O2 f.esk -o bin && ./bin` |
| `vm-src` | `eshkol-vm-standalone-test f.esk` |
| `vm-eskb` | `eshkol-run --profile hosted-vm --emit-eskb …` then the VM |

Two AOT optimization levels are separate axes deliberately: the differential
finding this corpus inherits was an **optimization-level-dependent** fault that
`-O0` alone could never have reproduced.

## What the first runs found

Ten divergences from R7RS. The first four were **fixed in the same change** and
are green on all five axes; the other six were filed under `found/` (below) and
have since been fixed as well.

Every one of the first four differed from R7RS on native **and** on the VM —
which is exactly why no differential had ever seen them:

* **Arrow clause.** `(test => receiver)`, the R7RS arrow clause, was not
  recognised by any of the three clause readers. Native code-generated the
  literal identifier `=>` as a variable reference; both bytecode compilers
  compiled it as an ordinary body expression. (The same gap existed in the VM's
  `cond`, whose native counterpart already supported `=>`.)
* **Test-only clause.** The clause `(test)`, whose value R7RS defines as the
  test's own value, left the result unset: native substituted `'()` and the VM
  returned whatever happened to be on the stack.
* **Implicit re-raise payload.** `eshkol_raise()` keeps a caller-supplied value
  only while `g_raised_value_set_by_user` is set and clears that flag on every
  raise, so the guard's own fall-through re-raise took the fallback branch and
  replaced the payload with the exception struct. An enclosing guard inspecting
  its variable got the opaque `#<exception>` instead of what was raised —
  `(guard (o (#t o)) (guard (i ((number? i) "n")) (raise "payload")))` answered
  `#<exception>`, exit 0. It now answers `"payload"`.
* **Closure built inside a guard clause.** The handler is inlined into its
  parent chunk, but only the handler's own entry pc was relocated; a nested
  `lambda`'s pc constant was copied verbatim, so the closure ran whatever sat at
  that unrelocated offset. `(define f (guard (e (#t (lambda (x) (+ x 100))))
  (raise 1)))` then `(f 1)` answered `1`; it now answers `101`. `=>` is what made
  it loud: an arrow clause calls the closure immediately, so a lambda receiver
  re-entered the top of the program until `STACK OVERFLOW`.

## `found/` — the six filed repros, now all converged

Same convention as `tests/vm_parity/found/`: one minimal repro per divergence,
with the expected and originally observed values in its header. These are
**not** run by the gate. At v1.3.6-evolve every one of them prints its expected
value on all five axes (`jit`, `aot-o0`, `aot-o2`, `vm-src`, `vm-eskb`):

| repro | axes that differed when filed | expected value, now printed on all five axes |
|---|---|---|
| `weh_handler_return_swallows_condition.esk` | **all five** | `guard-fired` |
| `vm_setbang_global_from_guard_clause_lost.esk` | vm-src, vm-eskb | `caught 1` |
| `vm_internal_defines_in_guard_body.esk` | vm-src, vm-eskb | `42` |
| `native_guard_variable_not_a_binding.esk` | jit, aot-o0, aot-o2 | `(mutated orig) captured-value` |
| `guard_wind_multi_module_dominance.esk` | jit, aot-o0, aot-o2 | three `PASS:` wind checks |
| `vm_eskb_wind_order_vs_clause.esk` | vm-eskb only | `(clause after before)` |

The first one is the notable one: it differed from R7RS **identically on every
axis**, so no amount of native-vs-VM differencing could ever have surfaced it.
That is the whole argument for grading against a golden. The numbered cases
that now cover the same shapes are `16_internal_defines_in_body`,
`17_wind_unwind_order` and `18_guard_variable_mutation`; the
`with-exception-handler` shape is also kept in `tests/vm_parity/resolved/`.

## Layout

| Path | What it is |
|---|---|
| `NN_*.esk` + `NN_*.expected` | a case and its golden. Each case prints `PASS:<name>` / `FAIL:<name>` tokens; the golden is the exact token sequence, newline-stripped and blank-collapsed. |
| `fatal/*.esk` | fail-closed probes. An unhandled condition must exit nonzero, print a diagnostic on stderr, and never reach the trailing `MUST-NOT-PRINT` sentinel. |
| `ENGINES.tsv` | the only way to exempt a `(case, axis)` pair, justification mandatory. The gate fails on a stale row (naming a case or axis that does not exist) and on a row with no justification. |

Run it with `BUILD_DIR=build scripts/run_guard_coverage.sh`. CI runs the same
script in `.github/workflows/ci.yml`. At v1.3.6-evolve it reports
`guard-coverage: 107 passed, 0 failed, 0 infra, 0 waived across:
jit,aot-o0,aot-o2,vm-src,vm-eskb`.

## Out of scope, on purpose

`guard` in **tail position** (a self tail call through `guard` must keep every
activation's handler live) is pinned by `tests/tco/guard_tail_context/`.
Two lanes pinning the same behaviour would mean two answers to maintain, so
nothing here tests a tail call in a guard body.

## Adding a case

1. Write `NN_name.esk` so that every check prints `PASS:<check-name> ` on
   success and `FAIL:<check-name> ` on failure, using a local `chk` helper —
   do **not** `(require stdlib)`, which the VM axes cannot resolve.
2. Write `NN_name.expected` **by hand**, from R7RS, before running anything.
   Generating it from a run would certify the implementation against itself.
3. Run the gate. If an axis disagrees, that is the engine to fix, not a golden
   to adjust — the only alternative is an `ENGINES.tsv` row with a real reason.
