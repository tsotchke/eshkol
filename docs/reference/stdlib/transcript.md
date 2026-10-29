---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/transcript.esk
  - lib/core/merkle.esk
  - lib/core/json.esk
  - scripts/math_crosscheck/exact_math.py
  - scripts/math_crosscheck/replay_check.py
  - tests/stdlib/transcript_test.esk
---

# `core.transcript` — exact, content-addressed transcripts of certified quantities

`(require core.transcript)` records, for one program run, every certified
mathematical quantity a program computes (a Smith-normal-form invariant-factor
list, a rank mod p, a determinant, a polynomial gcd, ...) together with the
exact inputs that produced it, as a single JSON document. The point is
independent recomputation: a second, differently-written program -- the
pure-Python replay checker in `scripts/math_crosscheck/`, or PARI/GP and SymPy
on a second host via `scripts/math_crosscheck/crosscheck_pari_sympy.py` -- can
recompute each claim from its own copy of the inputs and compare against the
recorded output. A mismatch means Eshkol and the independent implementation
disagree about the mathematics; a hash mismatch means the transcript file was
altered after Eshkol wrote it.

## Recording is off by default and never changes a program's output

```scheme
(require core.transcript)
(define T (transcript-open "my-program"))   ;; INACTIVE unless ESHKOL_TRANSCRIPT is set
(transcript-record! T "determinant" "example"
  (list (cons "matrix" (tv-matrix-rows '((1 2) (3 4)))))
  (tv-int -2))
(transcript-close! T)                       ;; no-op, no file touched, when inactive
```

`(transcript-open name)` returns an **inactive** handle unless the
`ESHKOL_TRANSCRIPT` environment variable names a (non-empty) path, in which
case it returns an **active** handle bound to that path.
`transcript-record!`/`transcript-close!` on an inactive handle are silent
no-ops: no file is touched and nothing beyond the handle check is computed.
`transcript-record!` itself never prints anything, so turning recording on
never changes a program's displayed output or exit code -- every
`mathematics_*.esk` example wired to `core.transcript` captures its
stdout before and after setting `ESHKOL_TRANSCRIPT` and diffs them to prove
this.

```text
ESHKOL_TRANSCRIPT=/tmp/t.json eshkol-run -r examples/mathematics_group_cohomology_order_eight.esk
```

## Canonical value encoding

Every encoded value is a JSON object tagged by `"t"`:

| Encoder | JSON shape |
|---|---|
| `(tv-int z)` | `{"t":"z","v":"<decimal string>"}` |
| `(tv-rat q)` | `{"t":"q","n":"<num string>","d":"<den string>"}` (integers encode as `tv-int`) |
| `(tv-bool b)` | `{"t":"bool","v":true\|false}` |
| `(tv-string s)` | `{"t":"str","v":"<string>"}` |
| `(tv-opaque label repr)` | `{"t":"opaque","label":"<label>","repr":"<string>"}` |
| `(tv-list xs)` `(tv-int-list xs)` `(tv-rat-list xs)` | `{"t":"list","n":N,"data":[<value>...]}` |
| `(tv-vector xs)` | `{"t":"vec","n":N,"data":[<value>...]}` |
| `(tv-matrix-rows rows)` `(tv-matrix-vec V)` `(tv-matrix V)` | `{"t":"mat","rows":R,"cols":C,"data":[[<value>...]...]}` |
| `(tv-poly var coeffs)` | `{"t":"poly","var":"<name>","deg":D,"coeffs":[<value>...]}` (lowest degree first) |

Integers and rational numerators/denominators are **always** decimal
**strings**, never bare JSON numbers: Eshkol's exact integers are unbounded
bignums, and a JSON number is not guaranteed beyond float64 precision by every
reader, so a bare big integer would silently lose precision in some consumer.
Small structural counts (`n`, `rows`, `cols`, `deg`) are plain JSON numbers --
they are always small int64s by construction.

A **claim** document is

```text
{"kind":"<claim kind, e.g. smith_normal_form>",
 "label":"<human label identifying the instance>",
 "inputs":{"<name>":<value>, ...},
 "output":<value>,
 "hash":"<16-hex fnv1a-64 of the canonical JSON text of
          {"kind":...,"label":...,"inputs":...,"output":...} in that
          field order, with the hash field itself excluded>"}
```

and a **transcript** document is
`{"schema":"eshkol.transcript.v1","program":"<name>","claims":[<claim>...]}`.
The hash reuses `core.merkle`'s `fnv1a-64` directly (not a second hash
function): a transcript file is tamper-evident the same way `core.merkle`'s
content-addressable store is. `scripts/math_crosscheck/exact_math.py` carries
an independent, matching implementation of `fnv1a-64` so an external reader
can verify the hash without Eshkol at all.

### Why `"data"`/`"coeffs"` are vectors, not lists, internally

`core.json`'s `json-stringify` cannot tell an empty Eshkol **list** apart from
JSON `null` -- both are `'()`, and `null` parses back to `'()` too, so there is
no third state to distinguish them. `tv-list`, `tv-vector`, `tv-matrix-rows`
and `tv-poly` therefore build their array payload as a **vector**
(`list->vector`): `json-stringify` treats a vector as an array unconditionally,
including the empty vector, which is the one Eshkol value that is never
ambiguous with `null`. This matters in practice, not just in principle: the
degree-1 normalized boundary of every group of order eight in
`mathematics_group_cohomology_order_eight.esk` is the zero map, so its Smith
normal form has **no** nonzero invariant factor, and that empty-output claim
is exactly the case this guards.

## Writing and replaying a claim

```scheme
(require core.transcript)
(define T (transcript-open "my-program"))
(transcript-record! T "smith_normal_form" "Z8_normalized_boundary_1"
  (list (cons "boundary" (tv-matrix-vec some-matrix)))
  (tv-int-list (snf! some-matrix)))
(transcript-close! T)
```

```text
python3 scripts/math_crosscheck/replay_check.py /tmp/t.json
```

`replay_check.py` is pure Python 3 standard library (no numpy, no sympy, no
PARI) -- the piece CTest can run in ordinary CI. For every claim it (1)
recomputes the FNV-1a 64 hash over the canonical JSON text and compares it
against the recorded `"hash"` (a **tamper check**: catches a transcript edited
after the fact, but does not by itself prove the claim correct), then (2) for
every claim `"kind"` it knows how to recompute, replays the mathematics from
`"inputs"` with `exact_math.py` -- a second, independently written
implementation of determinant (fraction-free Bareiss elimination), rank mod p,
Smith normal form (diagonal invariant factors, canonically sorted), and
polynomial gcd over Z[x] -- and compares against `"output"` (the
**correctness check**). Known kinds: `determinant`, `smith_normal_form`,
`rank_mod_p`, `gram_matrix_rank_mod_p`, `gcd`, `character_count`. An unknown
kind is reported `SKIPPED`, not `AGREE`, and does not by itself fail the run.

`scripts/math_crosscheck/crosscheck_pari_sympy.py` is the atlas-only sibling:
it additionally checks `determinant`/`smith_normal_form`/`rank_mod_p`/`gcd`
claims against PARI/GP's `matdet`/`matsnf`/`matrank`/`gcd` and SymPy's
`Matrix.det`/`Matrix.rank`/`Poly.gcd`, and writes a JSON receipt recording the
tool versions used (PARI/GP and SymPy are not installed on every host, unlike
the Python-stdlib-only `replay_check.py`).

## Which examples carry a transcript

| Program | Claim kinds recorded |
|---|---|
| `examples/mathematics_group_cohomology_order_eight.esk` | `smith_normal_form` — the degree-1/2 normalized boundary of S_3 and the five groups of order eight (the data behind the printed H_1 abelianization) |
| `examples/mathematics_cohomology_rings_cup_products.esk` | `rank_mod_p` — every coboundary-image matrix behind a printed dim H^n value, including the 49x343 image on (Z/2)^3 that decides whether the ten cubic monomials are independent |
| `examples/mathematics_fermat_hodge_classes_sweep.esk` | `gram_matrix_rank_mod_p` — the linear-cycle Gram matrix and its rank mod p; `character_count` — the Griffiths/Shioda Hodge-number cross-check |
| `tests/stdlib/multimodular_test.esk` | `determinant` — the 6x6 13-digit case; `gcd` — a large-coefficient polynomial pair |

## Tests

```text
ctest --test-dir build --output-on-failure -R '^transcript_(jit|aot)$'
python3 scripts/math_crosscheck/replay_check.py --dir scripts/math_crosscheck/fixtures
```

`tests/stdlib/transcript_test.esk` covers the inactive-by-default no-op, every
canonical encoder's shape (including the empty-list case against
`json-stringify` directly), hash determinism and content-sensitivity, and a
real active write/read round trip through `core.json`.
`scripts/math_crosscheck/fixtures/` holds committed transcripts from the four
programs above; CTest runs `replay_check.py` against them so the cross-check
runs in ordinary CI, with no Eshkol build or PARI/SymPy required.
