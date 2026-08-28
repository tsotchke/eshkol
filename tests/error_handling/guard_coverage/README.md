# `guard` coverage (ESH-0101)

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
backends get wrong the same way compares byte-equal and passes. The ledger
already had a name for that blind spot (SW-06, "shared-defect blindness").

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
finding this corpus inherits was an **optimization-level-dependent** crash that
`-O0` alone could never have reproduced.

## What the first run found

Both defects were wrong on native **and** on the VM, which is exactly why no
differential had ever seen them:

* **SW-78** — `(test => receiver)`, the R7RS arrow clause, was not recognised by
  either clause reader. Native code-generated the literal identifier `=>` as a
  variable reference; the VM compiled it as an ordinary body expression. (The
  same gap existed in the VM's `cond`, whose native counterpart has had `=>`
  since ESH-0109 closed.)
* **SW-79** — the test-only clause `(test)`, whose value R7RS defines as the
  test's own value, left the result unset: native substituted `'()` and the VM
  returned whatever happened to be on the stack.

## Layout

| Path | What it is |
|---|---|
| `NN_*.esk` + `NN_*.expected` | a case and its golden. Each case prints `PASS:<name>` / `FAIL:<name>` tokens; the golden is the exact token sequence, newline-stripped and blank-collapsed. |
| `fatal/*.esk` | fail-closed probes. An unhandled condition must exit nonzero, print a diagnostic on stderr, and never reach the trailing `MUST-NOT-PRINT` sentinel. |
| `ENGINES.tsv` | the only way to exempt a `(case, axis)` pair, justification mandatory. The gate fails on a stale row (naming a case or axis that does not exist) and on a row with no justification. |

Run it with `BUILD_DIR=build scripts/run_guard_coverage.sh`.

## Out of scope, on purpose

`guard` in **tail position** is `SW-58` (a self tail call through `guard`
collapses the handler chain) and is pinned by `tests/tco/guard_tail_context/`.
Two lanes pinning the same behaviour would mean two answers to maintain, so
nothing here tests a tail call in a guard body.

## Adding a case

1. Write `NN_name.esk` so that every check prints `PASS:<check-name> ` on
   success and `FAIL:<check-name> ` on failure, using a local `chk` helper —
   do **not** `(require stdlib)`, which the VM axes cannot resolve.
2. Write `NN_name.expected` **by hand**, from R7RS, before running anything.
   Generating it from a run would certify the implementation against itself.
3. Run the gate. If an axis disagrees, that is a defect to fix, not a golden to
   adjust — the only alternative is an `ENGINES.tsv` row with a real reason.
