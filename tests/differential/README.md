# Differential execution testing (adversarial campaign, pillar P1)

Eshkol has several execution paths that MUST agree on every deterministic
program. Identical program + identical input => identical `(exit code,
normalized stdout)` on every axis. **Any divergence is a compiler bug by
definition** — no external oracle or hand-computed expectation is needed.

## Axes

| axis          | command                                                        |
|---------------|----------------------------------------------------------------|
| `jit`         | `./build/eshkol-run -r f.esk` (run-cache default ON)           |
| `jit-nocache` | `ESHKOL_JIT_CACHE=0 ./build/eshkol-run -r f.esk`               |
| `aot-o0`      | `./build/eshkol-run -O0 f.esk -o bin && ./bin`                 |
| `aot-o2`      | `./build/eshkol-run -O2 f.esk -o bin && ./bin`                 |
| `vm-src`*     | `./build/eshkol-vm-standalone-test f.esk` (VM's mini-compiler) |
| `vm-eskb`*    | `eshkol-run --profile hosted-vm --emit-eskb f.eskb f.esk` then `eshkol-vm-standalone-test f.eskb` |

\* VM axes are opt-in via `--with-vm` and are compared **only against each
other** (`vm-src-vs-vm-eskb`), not against the native axes. Two reasons,
verified 2026-07-02:

1. **Print semantics**: the VM's `display` appends a trailing newline per
   call; native paths do not. `(display 42)(display 43)` prints `4243` on
   native paths but `42\n43\n` in the VM, so byte-comparison against native
   axes fails on essentially every program.
2. **Language subset**: `eshkol-vm-standalone-test` embeds its own mini
   compiler (`lib/backend/eshkol_vm.c`) covering a core-Scheme subset; the
   full corpus (tensors, AD, keyword args, ...) is not expected to compile
   there yet.

The `vm-src-vs-vm-eskb` pair is still a real differential: the VM's internal
compiler versus `eshkol-run`'s ESKB emitter targeting the same interpreter.
When the VM's print semantics converge with the native runtime, drop the
special-casing in `run_axis`/pairing and let the VM axes join the native
pairwise matrix.

## Running

```sh
# corpus across all native axes (default corpus dir shown)
scripts/run_differential.sh tests/differential/corpus/

# include the VM pair
scripts/run_differential.sh --with-vm

# seeded random-program fuzzing with auto-shrinking
scripts/run_differential_fuzz.sh --seed 42 --count 200
```

Prerequisite: `cmake --build build --target eshkol-run stdlib -j` (plus
`eshkol-vm-standalone-test` for `--with-vm`). Point `BUILD_DIR` elsewhere to
use another build tree. Per-run timeout: `DIFFERENTIAL_TIMEOUT` (default 90s).

Both runners emit pytest-style `PASSED/FAILED <nodeid>::<axisA-vs-axisB>`
lines plus `{"kind":"differential_smoke",...}` JSON-L records into
`scripts/icc_traces/differential_smoke.jsonl` (corpus) and
`scripts/icc_traces/differential_fuzz.jsonl` (fuzz), consumed by the
`differential-clean` oracle in `.icc/completion-oracles.yaml`.

## Interpreting a divergence

A FAIL names the axis pair, e.g.
`FAILED tests/differential/20_guard_raise.esk::aot-o0-vs-aot-o2`, and prints
the first differing lines of each side. Divergence categories:

- **stdout differs, both rc=0** — silent miscompilation on at least one path
  (the worst kind: users get wrong answers, no error).
- **exit codes differ** — crash / uncaught error on one path only
  (rc 139 = SIGSEGV, rc 124 = timeout, rc 125 = compile failed).
- **all axes exit nonzero identically** — not a divergence, but corpus
  programs must be green, so the corpus runner still FAILs the file.

Divergence does not tell you which path is *correct* — only that at least
one is wrong. Triage by checking the R7RS-expected output by hand, then file
an ESH task (`.swarm/tasks/`) carrying the minimal repro.

## Corpus (`corpus/`)

~40 deterministic programs, one feature cluster each: numeric tower
(int/rational/bignum/double/complex, inf/nan printing), strings (unicode,
embedded NUL), chars, radix `string->number`/`number->string`, lists,
vectors, hash tables, closures + `set!`, TCO loops, named let, mutual
recursion, `call/cc`, `dynamic-wind`, `guard`/`raise`, quasiquote, macros,
streams/`delay`, keyword args, `let-values`, `match`, `apply`/`map` with
first-class builtins, higher-order functions, AD (`derivative`, `gradient`),
tensors (`matmul`, element ops), equality, `write` vs `display`, sorting.

Rules for adding a corpus entry:

1. Deterministic output only — no timestamps, no randomness, no pointer
   values, no hash-table iteration order (query keys directly instead).
2. Every program must print something (`display`/`write` + `newline`) —
   an output-free program can't diverge.
3. Prefer one feature *cluster* per file, and exercise it through data flow
   (a value computed by the feature reaches stdout).
4. If a file exposes a real divergence, LEAVE IT IN THE CORPUS (it is the
   regression test for the eventual fix), add the hand- or auto-shrunk
   minimal repro to `found/`, and file an ESH task. `differential-clean`
   stays red until the compiler is fixed — that is the point of the gate.

## Findings (`found/`)

Each `NNN_*.esk` is a minimal repro of a confirmed divergence, with a header
comment giving per-axis behavior and provenance (corpus file or fuzz seed +
program index). These files are evidence, not gate inputs: the corpus runner
does not execute `found/`.

## Fuzzing + shrinking flow

`scripts/gen_differential.py` generates bounded-depth, well-typed, always
terminating programs (mixed exact/inexact arithmetic, `let`/`let*`/`letrec`,
lambda + application, `if`/`cond`-style branching, quote, list/vector/string
ops, `set!` on locals, bounded named-let loops) from `--seed`; program `i`
of a seed is fully reproducible.

When a generated program diverges, the shrinker greedily (1) drops top-level
forms, (2) replaces subtrees with same-type children or literals — re-running
all axes after each candidate edit and keeping it only while the divergence
persists (budget: `--shrink-budget` axis-evaluations). The minimal repro is
written to `found/NNN_shrunk.esk`; duplicate shrunken programs (same bytes)
are not re-saved. Re-run any finding directly:

```sh
scripts/run_differential.sh tests/differential/found/   # expect FAILs: these are bugs
```
