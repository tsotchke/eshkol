# core.transcript cross-check report

- Date: 2026-10-30
- Branch: `feat/v14-math-transcripts`
- Build: `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-13
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DESHKOL_BUILD_TESTS=ON
  -DESHKOL_REQUIRED_LLVM_MAJOR=21`, Linux x64 (gcc-13, LLVM 21), built from
  source rather than run against an installed release binary, since
  `lib/core/json.esk` ships pre-linked into that binary's precompiled
  stdlib cache and a source-level change to it is visible only in a build
  made from this source tree
- Scope: `lib/core/transcript.esk`, `lib/core/json.esk`, the four wired
  programs, `scripts/math_crosscheck/{replay_check,exact_math,crosscheck_pari_sympy}.py`

## Summary

Every claim recorded by `core.transcript` in the four wired programs was
independently recomputed and agreed, by three separate paths: Eshkol's own
output (unchanged by recording), a pure-Python replay
(`replay_check.py`), and PARI/GP + SymPy
(`crosscheck_pari_sympy.py`).

## 1. Recording does not change program output

For each wired program, stdout was captured with `ESHKOL_TRANSCRIPT` unset and
then set, and diffed:

| Program | `diff` result |
|---|---|
| `mathematics_group_cohomology_order_eight.esk` | byte-identical |
| `mathematics_cohomology_rings_cup_products.esk` | byte-identical |
| `mathematics_fermat_hodge_classes_sweep.esk` | differs only in the `elapsed_ms` field the program itself prints (wall-clock jitter, unrelated to recording) |
| `tests/stdlib/multimodular_test.esk` | byte-identical |

## 2. `core.json` now emits an empty JSON array distinctly from `null`

`core.json`'s `json-stringify` maps Eshkol's `'()` to JSON `null`, which is
also what `json-parse` produces for `null` — the two share the same Eshkol
value, with nothing else to distinguish them. `json-stringify` now treats a
**vector** as an array unconditionally, including the empty vector, which is
the one Eshkol value that is never ambiguous with `null`.
`core.transcript`'s `tv-list`/`tv-vector`/`tv-matrix-rows`/`tv-poly` build
their `"data"`/`"coeffs"` payload as `(list->vector ...)` rather than a plain
list, so an empty array round-trips as `[]`. The first transcript generated
(`mathematics_group_cohomology_order_eight.esk`, whose degree-1 normalized
boundary is the zero map for every one of the six groups recorded) is the
case this covers: every empty Smith-normal-form claim in that program's
fixture now carries `[]`, verified by `replay_check.py` below.

## 3. The cohomology-rings transcript block passes `rank-mod-p` a list of vectors

`rank-mod-p` (defined in `mathematics_cohomology_rings_cup_products.esk`)
calls `vector-copy` on each row, so its input is a list of vectors.
`coboundary-image`'s own output is already in that shape; the transcript
block computes the rank directly from it and converts to lists only for
`tv-matrix-rows`, which takes list-of-lists.

## 4. The PARI/GP harness writes every matrix with `Mat(...)`

GP's bracket-with-semicolons literal syntax needs a semicolon to mark a row
boundary, so a single-row matrix (no semicolon) reads as a plain vector
(`t_VEC`) rather than a matrix (`t_MAT`), and `matrank`/`matsnf` require the
matrix type. `crosscheck_pari_sympy.py` wraps every literal in `Mat(...)`,
which is unambiguously the matrix type whether the matrix has one row or
many. This covers every rank-0 coboundary-image claim and every empty-SNF
boundary claim in the fixtures — the same cases §2 covers on the Eshkol
side, now from the PARI/GP side.

## 5. Final verified results

`scripts/math_crosscheck/fixtures/*.json` are the four transcripts generated
from the build above, each with an `ESHKOL_TRANSCRIPT`-unset/-set diff
already confirmed clean (§1).

### `replay_check.py` (pure Python stdlib; the CTest-gated check)

```
python3 scripts/math_crosscheck/replay_check.py --dir scripts/math_crosscheck/fixtures
```

| File | Claims | AGREE | DISAGREE | SKIPPED |
|---|---|---|---|---|
| `mathematics_cohomology_rings_cup_products.json` | 12 | 12 | 0 | 0 |
| `mathematics_fermat_hodge_classes_sweep.json` | 2 | 2 | 0 | 0 |
| `mathematics_group_cohomology_order_eight.json` | 12 | 12 | 0 | 0 |
| `multimodular_test.json` | 2 | 2 | 0 | 0 |
| **Total** | **28** | **28** | **0** | **0** |

Every hash tamper check also passed (recomputed FNV-1a 64 over the canonical
JSON text matches the recorded `"hash"` on all 28 claims). `RESULT: ALL PASS`.

### `crosscheck_pari_sympy.py` (for a host with PARI/GP and SymPy installed; PARI/GP 2.17.4, SymPy 1.14.0)

```
python3 scripts/math_crosscheck/crosscheck_pari_sympy.py --dir scripts/math_crosscheck/fixtures --receipt <path>
```

| Tool | Verdicts | AGREE | DISAGREE | ERROR |
|---|---|---|---|---|
| PARI/GP | 28 | 28 | 0 | 0 |
| SymPy | 18 | 18 | 0 | 0 |
| **Total** | **46** | **46** | **0** | **0** |

(1 claim — `character_count` — SKIPPED by design: griffiths/shioda are
already two independently-derived Eshkol-side counts compared for equality,
not a linear-algebra primitive either tool exposes directly.) `RESULT: ALL
PASS`.

### Full CTest gate (`scripts/run_transcript_gate.sh`)

```
BUILD_DIR=build JOBS=4 ./scripts/run_transcript_gate.sh
```

```
core.transcript gate: PASS (ctest 11/11 passed (ran 11, exit 0))
```

`transcript_jit`/`transcript_aot`, the four wired programs' own CTest entries
(JIT and AOT), and `transcript_replay_check` all passed. Emitted
`{"kind":"v1_4_connection","name":"v14_math_transcripts","value":"PASS"}`,
which the `v14_math_transcripts` completion-oracle criterion in
`.icc/completion-oracles.yaml` consumes.

### Full project CTest (Linux x64, gcc-13, LLVM 21)

PENDING-FILL-IN
