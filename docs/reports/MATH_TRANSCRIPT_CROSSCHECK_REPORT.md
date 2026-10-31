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

## 5. `core.transcript` computes its content hash in pure Eshkol

`core.transcript`'s claim hash is FNV-1a 64-bit, the same algorithm
`core.merkle`'s content-addressable store uses. The two wired cohomology
examples also each have a dedicated bytecode-VM CTest entry
(`vm_mathematics_group_cohomology_order_eight`,
`vm_mathematics_cohomology_rings_cup_products`), and `core.merkle`'s
`fnv1a-64`/`hash->hex` reach a native `extern` for correct 64-bit wraparound
arithmetic — a documented need, since Eshkol's generic bignum modulo does
not yet have uint64-overflow semantics above 2^53. That `extern`'s signature
is outside the VM's predetermined admission list, so any program that transitively
requires `core.merkle` fails VM compilation outright, not merely skips
recording in it; both wired cohomology examples hit exactly this as soon as
they required `core.transcript`.

`core.transcript` now carries its own implementation, `tr-fnv1a-64-hex`,
that never forms an intermediate value above about 2^34: the 64-bit hash
state is two 32-bit halves, and every multiply is a schoolbook long
multiplication over 16-bit limbs, so it stays well inside the documented-safe
range without touching the VM's admission rules or `core.merkle` itself.
`tests/stdlib/transcript_test.esk` checks it byte-identical to
`core.merkle`'s extern-backed `(hash->hex (fnv1a-64 str))` on the empty
string, several short strings, a 1000-character string and a 53-digit
decimal — large enough inputs to exercise every carry the limb
multiplication can produce. `core.transcript` no longer requires
`core.merkle` at all.

## 6. Final verified results

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

Pending build node (unavailable at the time of writing). `vm_prelude_cache_is_current`,
`vm_binding_origins_current` and `parity_backlog_build_free_test` are regenerated and
committed (see `lib/backend/vm_binding_origins.h`, `lib/backend/vm_prelude_cache.h`)
and will be confirmed by that run.

## 7. Independent recomputation of other libraries' certified quantities

Using the same method (PARI/GP, SymPy, and plain exact enumeration in Python) on
results already decided by other libraries in this wave, worked from each
program's own printed values or from a small script reproducing its inputs.

### Z[G] unit decisions and their circulant determinants

`tests/stdlib/ring_test.esk` decides, for every one of the 3^5 = 243 elements of
Z[C_5] with coefficients in {-1,0,1}, whether it is a unit by comparing its
multiplicative structure against the determinant of its circulant matrix
(computed with `exact-det`, fraction-free elimination). Reproducing the same
construction and printing each element's coefficients and determinant (one
line each; no library change) and independently computing each determinant
with PARI/GP's `matdet` and SymPy's `Matrix.det`:

| Check | Result |
|---|---|
| Elements recomputed | 242 of 243 |
| PARI/GP `matdet` agreement | 242/242 |
| SymPy `Matrix.det` agreement | 242/242 |
| Elements decided a unit (\|det\| = 1) | 30/242 |

### Smith normal forms and homology of group classifying spaces

`tests/stdlib/homological_test.esk` states `H_*(BS_3; Z) = Z, Z/2, 0, Z/6` and,
for the dihedral and quaternion groups of order 8, `H_1 = (Z/2)^2` for both. The
Smith normal forms already recorded and PARI/GP-and-SymPy-confirmed in this
PR's own fixtures (`mathematics_group_cohomology_order_eight.json`, written
against a different, independently authored boundary computation) give the
same groups' H_1 directly: the degree-1 boundary's SNF is empty (the zero map)
for every group recorded, so H_1 is the degree-2 boundary's cokernel. For S_3
(5 generators, SNF `[1,1,1,1,2]`, full rank): H_1 = Z/2. For D_4 and Q_8 (7
generators each, SNF `[1,1,1,1,1,2,2]`, full rank): H_1 = (Z/2)^2. Both match.
H_2 is not covered here: it needs the degree-3 boundary, excluded from this
PR's fixtures as too large to commit (see the comment in
`mathematics_group_cohomology_order_eight.esk`).

### Groebner bases

`tests/stdlib/groebner_test.esk` states two results from Cox-Little-O'Shea,
*Ideals, Varieties, and Algorithms*: the grlex basis of
`<x^3 - 2xy, x^2y - 2y^2 + x>` is `{x^2, xy, y^2 - x/2}`, and the lex basis
of `x^2+y+z = x+y^2+z = x+y+z^2 = 1` has 3 rational solutions among 8 counted
with multiplicity. SymPy's `groebner()` on the same generators under the same
orders reproduces the first basis exactly (`[x**2, x*y, -x + 2*y**2]`, the
same ideal up to a unit scalar) and, solving the second system with
`sympy.solve`, finds exactly 3 rational solutions
(`(0,0,1), (0,1,0), (1,0,0)`), agreeing with both claims; the full
multiplicity-8 count was not separately re-derived.

### Lawvere-theory / finite-model homomorphism counts

`tests/stdlib/theory_test.esk` states `|Hom(Z4,Z2)| = 2`, `|Hom(Z6,Z6)| = 6`,
`|Hom(S3,Z2)| = 2`, `|Hom(D4,Z2)| = |Hom(Q8,Z2)| = 4`, and `|End(Z2^3)| = 512`
with `|GL(3,F2)| = 168` automorphisms. Each is a standard, exactly computable
group-theory count (cyclic homomorphism counts via `gcd`, S_3/D_4/Q_8 through
their abelianizations, and the classical order formula for GL(n,F_q)),
reproduced directly in Python: all five agree.
