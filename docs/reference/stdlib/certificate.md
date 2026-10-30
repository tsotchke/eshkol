---
kind: reference
status: current
owner-area: stdlib
since: v1.7.0-synthesis
sources:
  - lib/core/certificate.esk
  - lib/core/transcript.esk
  - tools/certcheck/eshkol_certcheck.py
  - tools/certcheck/lean_export.py
  - tools/certcheck/tamper_suite.py
  - tests/stdlib/certificate_test.esk
  - tests/certificates/run_certificate_roundtrip.py
---

# `core.certificate` — certificates checked outside Eshkol

`(require core.certificate)` turns a certified result into a
**certificate**: the result, its exact inputs, and a **witness** that lets a
small program that does not trust the compiler confirm the result by cheap
recomputation. The checker is `tools/certcheck/eshkol_certcheck.py`; the
decidable finite facts also export to Lean 4. Design record:
[ADR-0047](../../design/adr/0047-certificates-and-independent-checker.md).

```scheme
(require core.certificate)
(define c (certify-smith-normal-form (vector (vector 2 4 4) (vector -6 6 12) (vector 10 -4 -16))))
(certificate-value c)        ;; => ((2 6 12) U V), as smith-normal-form returns
(certificate-kind c)         ;; => "smith_normal_form"
(certificate-id c)           ;; => "sha256:..."
(certificate-write-file "snf.json" "my-program" (list c))
```

```text
python3 tools/certcheck/eshkol_certcheck.py snf.json
snf.json: program='my-program' claims=1
  ACCEPTED kind=smith_normal_form label='smith_normal_form': invariant factors [2, 6, 12]
RESULT: ALL ACCEPTED
```

## Kinds

| Producer | Kind | Witness | The checker confirms |
|---|---|---|---|
| `(certify-rank A)` | `rank` (over Q) | independent rows and pivot columns; a kernel basis and its free columns | the named minor is nonsingular; A k = 0 for each kernel vector; the kernel vectors are independent |
| `(certify-determinant A)` | `determinant` | `perm`, `L`, `U` from `exact-plu` | P A = L U, L unit lower, U upper; det = sign(P) * prod diag U |
| `(certify-smith-normal-form A)` | `smith_normal_form` (integer A) | `U`, `V`, `U_inv`, `V_inv` | U A V = D; U U^-1 = I; V V^-1 = I; d_1 \| d_2 \| ... |
| `(certify-linear-solution A b)` | `linear_solution` | `inverse` | A A^-1 = I; A x = b |
| `(certify-solve-enclosure A b x R [w dA db])` | `linear_solve_enclosure` | the norm (weighted max) | recomputes residual, I - R A, theta < 1 and the radii of `core.certified_solve`; each box contains x +- radius |
| `(certify-psd M)` | `psd` | `B`, `D` with D > 0, or `v` | M = B^T diag(D) B, or v^T M v < 0 |
| `(certify-integer-gcd a b)` | `integer_gcd` | `s`, `t`, `a_over_g`, `b_over_g` | s a + t b = g; g divides a and b |
| `(certify-poly-gcd a b)` | `poly_gcd` (Q[x], monic) | `s`, `t`, cofactors | the same identities in Q[x] |
| `(certify-polynomial-identity lhs rhs)` | `polynomial_identity` | variables and the normal form of lhs - rhs | its own expansion of both sides equals the stated difference; the output is whether it is empty |
| `(certify-primality n)` | `primality` | a Pratt tree; or a Miller-Rabin base, a factor, or n < 2 | the Pratt conditions at every node; the base fails Miller-Rabin; the factor divides |
| `(certify-polynomial-range coeffs lo hi)` | `polynomial_range` | the centre and the re-expanded coefficients | p(x) = sum q_i (x - c)^i; the range contains q_0 +- sum \|q_i\| r^i |
| `(certify-taylor-model-range tm)` | `taylor_model_range` | rounding `outward`, the order | the range contains P([-r, r]) + R in exact arithmetic; CONDITIONAL on the premise `remainder_bound` |
| `(certify-cocycle D z)` | `cocycle` | the product D z | D z, and whether it is zero |
| `(certify-cohomology-class D z)` | `cohomology_class` (over Z) | `w` with D w = z; or a Smith form of D and an index | D w = z; or U D V = diag(d), U, V invertible over Z and (U z)_i not in d_i Z |

Each producer takes an optional label as its last argument. Every value in
a witness must be exact: a value that has left the exact tower raises
instead of being recorded. Doubles (Taylor coefficients, interval
endpoints) are recorded exactly with `tv-float` as m * 2^e.

`pratt-certificate p` returns the Pratt tree of a prime on its own.

## Claim and document format

A claim is the ordered JSON object
`{"kind","label","inputs","output","witness","premises","id","hash"}`:
a [`core.transcript`](transcript.md) claim with the witness, the premises
the checker takes on trust (empty for every kind but
`taylor_model_range`), the sha256 content address `id`, and the
`core.transcript` fnv1a-64 `hash`. A document is
`{"schema":"eshkol.certificate.v1","program":...,"claims":[...]}`.
`scripts/math_crosscheck/replay_check.py` reads certificate documents too
and replays the kinds it knows.

The identifiers name a claim; they do not prove it. The checker rejects a
claim whose `id` or `hash` does not match, and accepts one only when the
mathematics checks.

## Sinks and the `(certify ...)` form

| Procedure | Meaning |
|---|---|
| `(certificate-open program)` | a sink, active iff `ESHKOL_CERTIFICATES` names a file |
| `(certificate-open-path program path)` | an active sink bound to `path` |
| `(certificate-record! h c)` / `(certificate-close! h)` | add a claim / write the document (the sink stays usable) |
| `(certificate-write-file path program list)` | write a list of certificates in one step |
| `(certificate->json c)` | the sealed claim as JSON text |
| `(make-certificate kind label inputs output witness premises value)` | build a certificate for a new kind |

```scheme
(require core.certificate)
(define d (certify (exact-det M)))      ;; d is (exact-det M)
```

`(certify (f arg ...))` evaluates to `(f arg ...)`. When
`ESHKOL_CERTIFICATES` is set it also appends a certificate for the call to
that file (the program name comes from `ESHKOL_CERTIFICATES_PROGRAM`). The
compiler rewrites the form to `(certify-call "f" f (list arg ...))` on the
native and VM paths. Registered names: `exact-rank`, `exact-det`,
`smith-normal-form`, `exact-solve`, `certify-linear-solve`,
`exact-psd-factor`, `gcd`, `exact-gcdext`, `exact-poly-gcdext`,
`exact-prime?`, `tm-range`; `(register-certifier! name producer)` adds one,
`(certifier-names)` lists them. `certify` is a reserved form name.

```text
ESHKOL_CERTIFICATES=out.json eshkol-run -r program.esk
python3 tools/certcheck/eshkol_certcheck.py out.json
```

## The checker and its trusted base

`tools/certcheck/eshkol_certcheck.py` is one file that imports only `json`,
`fractions`, `hashlib` and `argparse` and nothing from the Eshkol tree.
`--trusted-base` prints what it relies on: the interpreter, those modules,
the file, and one named theorem per kind. It runs no elimination,
factorization or search. Exit status 0 means every claim was ACCEPTED (or
CONDITIONAL; `--strict` counts those as failures).

`tools/certcheck/tamper_suite.py` is the test of the checker: for every
claim it builds an unsealed edit, a resealed false output and a resealed
broken witness, and requires each to be rejected.

## Lean export

```text
python3 tools/certcheck/lean_export.py out.json -o Out.lean
lean Out.lean
```

`determinant` (n <= 6), `rank` (integral matrices), `smith_normal_form`,
`integer_gcd`, `primality`, `cocycle` and `cohomology_class` claims become
Lean 4 theorems whose statement is the witness identity and whose proof is
`by decide`. The file depends on Lean core only. For a gcd it also proves,
through the lemma `bezout_gcd`, that every common divisor divides g. That a
Pratt tree implies primality (Lehmer's theorem, Mathlib's
`lucas_primality`) and rank-nullity are named in the file, not re-proved.
`tests/certificates/lean/` holds the files exported from the committed
fixtures; [CERTIFICATE_LEAN_CHECK.md](../../reports/CERTIFICATE_LEAN_CHECK.md)
records the Lean version that checked them.

## Tests

```text
ctest --test-dir build -R '^certificate_'
python3 tools/certcheck/eshkol_certcheck.py --dir tests/certificates/fixtures
python3 tools/certcheck/tamper_suite.py --dir tests/certificates/fixtures
./scripts/run_certificate_gate.sh
```
