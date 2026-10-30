# core.transcript cross-check report

- Date: 2026-10-30
- Branch: `feat/v14-math-transcripts` @ `6a972bb8`
- Build: `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-13
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DESHKOL_BUILD_TESTS=ON
  -DESHKOL_REQUIRED_LLVM_MAJOR=21`, old-donkey (Linux x64, gcc-13/clang-21,
  `/usr/lib/llvm-21`) — a real build, not atlas's installed v1.3.6 release
  binary (see below for why that distinction matters)
- Scope: `lib/core/transcript.esk`, `lib/core/json.esk`, the four wired
  programs, `scripts/math_crosscheck/{replay_check,exact_math,crosscheck_pari_sympy}.py`

## Summary

Every claim recorded by `core.transcript` in the four wired programs was
independently recomputed and agreed, by three separate paths: Eshkol's own
output (unchanged by recording), a pure-Python replay
(`replay_check.py`), and PARI/GP + SymPy
(`crosscheck_pari_sympy.py`). One real bug was found and fixed in each of
the three pieces along the way (`core.json`, the cohomology-rings wiring,
and the PARI/GP crosscheck script itself) — this report records the
verified end state, with each bug noted where it was caught.

## 1. Recording does not change program output

For each wired program, stdout was captured with `ESHKOL_TRANSCRIPT` unset and
then set, and diffed:

| Program | `diff` result |
|---|---|
| `mathematics_group_cohomology_order_eight.esk` | byte-identical |
| `mathematics_cohomology_rings_cup_products.esk` | byte-identical |
| `mathematics_fermat_hodge_classes_sweep.esk` | differs only in the `elapsed_ms` field the program itself prints (wall-clock jitter, unrelated to recording) |
| `tests/stdlib/multimodular_test.esk` | byte-identical |

## 2. Bug found: `core.json` could not emit an empty JSON array

`core.json`'s `json-stringify` dispatches `'()` to `"null"` before any other
check, because `'()` is also what `json-parse` produces for JSON `null` — the
two are unavoidably the same Eshkol value. The first transcript generated
(`mathematics_group_cohomology_order_eight.esk`, whose degree-1 normalized
boundary is the zero map for every one of the six groups recorded) produced
`{"t":"list","n":0,"data":null}` for every empty Smith-normal-form claim: a
real, silent loss of information, not a hypothetical edge case.

**Fix** (`lib/core/json.esk`): `json-stringify` now treats a **vector** as an
array unconditionally, including the empty vector — the one Eshkol value
that is never ambiguous with `null`. **Fix** (`lib/core/transcript.esk`):
`tv-list`/`tv-vector`/`tv-matrix-rows`/`tv-poly` build their `"data"`/
`"coeffs"` payload as `(list->vector ...)` rather than a plain list, so the
empty case always round-trips as `[]`.

This fix lives in `lib/core/json.esk`, which ships pre-linked into the
installed compiler's `stdlib.bc`/`stdlib.o`. **Atlas's installed v1.3.6
release binary cannot see this fix** — `require`-ing `core.json` there
resolves to the symbols already baked into that cache, not to the edited
source — so every transcript this report cites was generated from a real
`cmake --build` on old-donkey, not atlas's `eshkol-run -r`. Verified
directly: before the fix, every `"data"`/`"coeffs"` field was `null`
(empty payloads silently, non-empty payloads too, since the stale binary
also could not see `tv-list`'s new vector-based encoding); after the build,
`replay_check.py` sees genuine `[]` and `[1,1,2,...]` arrays as appropriate.

## 3. Bug found: the cohomology-rings wiring passed the wrong shape to `rank-mod-p`

`rank-mod-p` (defined in `mathematics_cohomology_rings_cup_products.esk`)
calls `vector-copy` on each row, so it needs a list of **vectors**. The first
version of the new transcript block converted `coboundary-image`'s output to
a list of **lists** for the JSON encoder and then fed that same converted
value to `rank-mod-p`, crashing with `Type error in vector-copy: expected
vector or tensor, got pair` immediately after the program's last existing
check. **Fix:** compute the rank from `coboundary-image`'s own vector shape,
and convert to lists only for `tv-matrix-rows`. Caught by running the
program, not by reading it — a reminder that `ESHKOL_TRANSCRIPT=<path>
eshkol-run -r <program>` with the exit code checked is the actual
acceptance test for any wiring, not just a diff of stdout.

## 4. Bug found: PARI/GP reads a single-row matrix literal as a vector

`crosscheck_pari_sympy.py`'s first version built GP matrix literals as
`[a,b;c,d]` (bracket-with-semicolons). For a matrix with exactly one row
there is no semicolon, so GP parses the literal as a plain vector (`t_VEC`),
and `matrank`/`matsnf` both reject it outright (`incorrect type in rank/matsnf
(t_VEC)`) rather than silently misanswering. This hit every rank-0
coboundary-image claim and every empty-SNF boundary claim in the real
fixtures — i.e., the same real cases the `core.json` fix targets, from the
PARI/GP side. **Fix:** wrap every literal in `Mat(...)`, which forces the
matrix type unconditionally, single row or not.

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

### `crosscheck_pari_sympy.py` (atlas-only; PARI/GP 2.17.4 + SymPy 1.14.0)

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
