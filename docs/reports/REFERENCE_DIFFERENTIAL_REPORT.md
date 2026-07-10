# Reference-Implementation Differential Report (P7a)

> STALE SNAPSHOT — the "27/34 AGREE" results table below is a pre-fix
> historical snapshot. On current master the harness reports **34/34 AGREE
> (100%)**; the 7 divergences listed here (ESH-0150..0156) and ESH-0225 are all
> fixed (see `CHANGELOG.md`). Re-run `scripts/run_reference_differential.sh` to
> confirm. The fixed 34-program corpus no longer exposes anything, which is why
> the differential is now *generative* — see
> `docs/reports/GENERATIVE_DIFFERENTIAL_REPORT.md` (pillar P7c).

External R7RS ground-truth conformance oracle for Eshkol. Unlike every other
differential harness in this repo (which diffs Eshkol against *itself* across
its JIT/AOT axes — self-consistency only), this pillar runs the **same portable
R7RS-small program on Eshkol and on a real reference R7RS Scheme** and diffs the
result. A divergence on portable code is an Eshkol conformance bug measured
against external truth, not an internal inconsistency.

## Reference implementation

| | |
|---|---|
| Reference | **chibi-scheme 0.12.0 "magnesium"** (strict R7RS; `r7rs ratios complex mini-float uvector threads full-unicode modules`) |
| Install | `brew install chibi-scheme` |
| File-run invocation | `chibi-scheme <prog.scm>` where `<prog.scm>` is a fixed import prologue followed by the program body |
| Import prologue (chibi-only) | `(import (scheme base) (scheme write) (scheme char) (scheme inexact) (scheme cxr))` |

chibi was chosen over guile/chez/racket because it is the strictest mainstream
R7RS-small implementation and rejects R5RS-only names (e.g. `exact->inexact`),
which keeps the corpus honestly portable. The runner
(`scripts/run_reference_differential.sh`) auto-falls back to guile → chez if
chibi is absent.

## How it works

For each program in `tests/reference-diff/corpus/` the runner executes three
engines on the **byte-identical program body** (chibi additionally gets the
import prologue prepended; Eshkol needs and forbids imports):

* `ref`     — `chibi-scheme prog.scm`
* `esh-jit` — `build/eshkol-run -r prog.esk`
* `esh-aot` — `build/eshkol-run -O0 prog.esk -o BIN && BIN`

All three stdouts pass through `scripts/lib/normalize_scheme_output.py` and are
compared. Classification: **AGREE** / **ESHKOL-DIVERGES** (ref ok, Eshkol
errors/crashes or differs) / **REFERENCE-ERROR-ESHKOL-OK** (corpus program not
portable) / **BOTH-ERROR**. A `kind:"reference_diff"` ICC trace is written to
`scripts/icc_traces/reference_diff.jsonl`; the `reference_diff_gate` event is
PASS iff every program AGREES. The `reference-diff` oracle in
`.icc/completion-oracles.yaml` gates on it.

### Output normalization (so only *semantic* diffs flag)

Applied identically to both engines — each rule can only ever collapse an
implementation-defined rendering difference, never manufacture a false
agreement between genuinely different values:

1. **ANSI escapes** stripped.
2. **Boolean spelling** `#true`/`#false` → `#t`/`#f` (R7RS permits both).
3. **Exact/inexact + float precision.** Rationals (`a/b`) and integers are left
   verbatim; any float token is reformatted with `%.6g` (6 significant digits).
   Consequences, all intended by the task spec ("exact/inexact printing, float
   precision"): inexact integers collapse (`1.0`→`1`, `3.0`→`3`, matching
   Eshkol's default printing) and precision beyond 6 sig-figs is dropped
   (`1.4142135623730951`→`1.41421`). A *wrong* number still differs and flags.
4. **Nested string/char rendering under `display`.** R7RS `display` renders
   strings unquoted and chars as glyphs *recursively*; Guile, Racket and Eshkol
   do so, but chibi emits the `write` form (`"b"`, `#\c`) for members of a
   display'd datum. To avoid false-flagging Eshkol for chibi's quirk we strip
   the `#\` prefix from single printable chars and strip all `"`. Crucially
   this does **not** hide `write`-escaping bugs: finding #7 below (write emits a
   raw newline instead of `\n`) survives, because the difference is in the
   content (a literal newline vs the two characters backslash-n), not the
   quotes.
5. Trailing whitespace / blank lines trimmed.

The full rationale lives in the docstring of
`scripts/lib/normalize_scheme_output.py`; it is the single source of truth and
must be updated in lockstep with any change here.

## Results

| Metric | Value |
|---|---|
| Programs | **34** (categories: numeric 7, control 8, list 5, string 4, vector 3, binding 3, equality 2, char 1, io 1) |
| AGREE | **27** |
| ESHKOL-DIVERGES | **7** |
| REFERENCE-ERROR-ESHKOL-OK | 0 |
| BOTH-ERROR | 0 |
| **Agreement rate** | **79.4 %** |
| Peak disk under `artifacts/reference-diff/` | **~0.2 MB** (cap 1024 MB) |

Gate is (correctly) **RED** while the 7 conformance gaps below remain — mirroring
the `differential-clean` oracle philosophy: the divergences *are* the open bugs.

## Semantic divergences (the treasure)

Every one is a valid R7RS-small program that chibi runs correctly and Eshkol
gets wrong. Each is filed as an ESH task; the minimal repro and both outputs are
below.

### 1. `apply` with leading args before the final list → SIGSEGV — ESH-0150
R7RS: `(apply proc a1 a2 … args-list)`. Eshkol supports only `(apply proc list)`.
```
(display (apply + 1 2 '(3 4 5)))
```
| chibi | eshkol -r / AOT |
|---|---|
| `15` | crash, exit 139 (no output) |
Also `(apply + 1 (list 3 4 5))`→13 and `(apply max 1 '(9 2))`→9 crash. The
single-list form `(apply + '(1 2 3 4))` works.

### 2. Multi-argument `vector-map` / `vector-for-each` ignore extra vectors — ESH-0151
Returns the **first** input vector unchanged (silent wrong answer, no error).
```
(display (vector-map + #(1 2 3) #(10 20 30)))
```
| chibi | eshkol -r / AOT |
|---|---|
| `#(11 22 33)` | `#(1 2 3)` |
Single-vector `vector-map` is correct.

### 3. `cond` / `case` `=>` (arrow) clauses unsupported — ESH-0152
`=>` is treated as an undefined variable; the proc is printed instead of applied.
```
(display (cond ((assv 2 '((1 . "a") (2 . "b"))) => cdr) (else "none")))
```
| chibi | eshkol -r / AOT |
|---|---|
| `b` | `#<procedure>` (plus `Undefined variable: =>` on stderr) |
Same for `(case 5 ((4 5 6) => (lambda (x) (* x 10))) (else 'no))`: chibi `50`.

### 4. `vector-copy` unimplemented (all arities) — ESH-0153
`Unknown function: vector-copy` aborts the whole compilation.
```
(display (vector-copy #(1 2 3 4 5) 1 4))
```
| chibi | eshkol -r / AOT |
|---|---|
| `#(2 3 4)` | codegen error, exit 1 |
`(vector-copy v)` and `(vector-copy v 1)` also error.

### 5. Quasiquoted vector `` `#(… ,x …) `` produces no output — ESH-0154
The form is silently dropped.
```
(display `#(1 ,(+ 2 2) 3))
```
| chibi | eshkol -r / AOT |
|---|---|
| `#(1 4 3)` | *(empty)* |
`(list->vector (list 1 (+ 2 2) 3))` and list quasiquote both work.

### 6. `error-object?` / `error-object-message` unimplemented — ESH-0155
`Unknown function: error-object?` aborts the file.
```
(display (guard (e ((error-object? e) (error-object-message e))) (error "something failed")))
```
| chibi | eshkol -r / AOT |
|---|---|
| `something failed` | codegen error, exit 1 |
`guard` itself works — `(guard (e (#t 'recovered)) (car '()))` returns
`recovered`; only the error-object accessors are missing.

### 7. `write` does not escape control chars in strings — ESH-0156
`write` must emit a re-readable representation; Eshkol emits raw control chars.
```
(write "a\nb")
```
| chibi | eshkol -r / AOT |
|---|---|
| `"a\nb"` (one line) | `"a`⏎`b"` (literal newline splits the line) |
`(write "tab\there")` emits a raw tab. `display` is correct; only `write` is
non-conformant.

## Reproduce

```sh
brew install chibi-scheme
cmake --build build --target eshkol-run stdlib -j
python3 scripts/gen_reference_corpus.py            # regenerate corpus (deterministic)
scripts/run_reference_differential.sh              # run the differential + write ICC trace
```

Per-run compile artifacts use a **single reused temp binary** deleted after each
program (never accumulated); everything lives under `artifacts/reference-diff/`
(gitignored, hard-capped at 1 GB with an abort-on-exceed check and an on-exit
cleanup trap). Divergence logs (program + all three stdouts + AOT compile log)
are kept only for ESHKOL-DIVERGES programs under
`artifacts/reference-diff/divergences/`.
