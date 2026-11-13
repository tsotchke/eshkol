---
kind: explanation
status: current
owner-area: stdlib
since: v1.7.0-synthesis
sources:
  - lib/core/certificate.esk
  - lib/core/transcript.esk
  - lib/core/exact_linalg.esk
  - lib/core/polynomial_exact.esk
  - lib/core/ring.esk
  - lib/frontend/parser.cpp
  - lib/backend/vm_compiler.c
  - tools/certcheck/eshkol_certcheck.py
  - tools/certcheck/lean_export.py
  - tools/certcheck/tamper_suite.py
  - tests/stdlib/certificate_test.esk
  - tests/certificates/run_certificate_roundtrip.py
---
# ADR-0047: Certificates, an independent checker and a Lean export

**Status:** Proposed
**Date:** 2026-10-30
**Scope:** the format in which Eshkol states a certified result together
with the data that proves it; the program outside Eshkol that checks it;
the export of the decidable finite facts to Lean 4; and the `(certify ...)`
form that makes a certified primitive emit a certificate at run time.
**Depends on:** `core.transcript` (content-addressed exact transcripts),
`core.exact_linalg`, `core.integer_linalg`, `core.exact_psd`,
`core.certified_solve`, `core.polynomial_exact`, `core.ring`,
`core.symbolic`, `core.ad.taylor_models`; ADR-0034 (exact computation over
ring carriers); ADR-0004 (the type-system trajectory, whose Lean kernel
export this record starts on the data side).

## Context

Eshkol computes many results exactly: ranks, determinants, Smith normal
forms, rational solves, positive-semidefinite factorizations, gcds,
primality, polynomial identities, enclosures, cocycles. `core.transcript`
already records each such quantity with its exact inputs, and a pure-Python
replay recomputes it. A replay trusts nothing in Eshkol, but it has to redo
the computation: it needs its own elimination, its own Smith form, its own
factorization, and its cost and its code size grow with the problem.

The roadmap item for v1.7.0 asks for more: "a certified step emits its
witnesses, interval endpoints, rounding direction and derivation chain, and
a checker outside Eshkol re-verifies the certificate without trusting the
compiler." Formal Verification asks for integration with Lean. The
mathematics programme needs the same thing for its exact data: arguments
that cite a computed rank or invariant factor should rest on something a
reader can check.

For most certified quantities there is a witness that makes checking far
cheaper than computing:

| Fact | Witness | Check |
|---|---|---|
| rank A = r | a nonsingular r x r minor; n - r kernel vectors | one small determinant per side, A k = 0 |
| det A = d | P A = L U | one product, a diagonal product |
| Smith form of A | U, V and their integer inverses | U A V = D, U U^-1 = I, divisibility |
| x solves A x = b uniquely | A^-1 | A A^-1 = I, A x = b |
| M is (not) PSD | M = B^T diag(D) B, D > 0 (or v with v^T M v < 0) | one product (one quadratic form) |
| g = gcd(a, b) | s, t with s a + t b = g, cofactors | three products |
| n is prime | a Pratt tree | modular powers |
| n is composite | a Miller-Rabin base or a factor | modular powers or one division |
| p(x) in [L, H] on a box | the re-expansion about the centre | a polynomial identity, one sum |
| z is not a coboundary | a Smith form of D and an index | products and one remainder |

## Decision

### 1. Format: a certificate is a transcript claim with a witness

A certificate claim is the ordered JSON object

```text
{"kind", "label", "inputs", "output", "witness", "premises", "id", "hash"}
```

- `kind`, `label`, `inputs`, `output` are exactly a `core.transcript`
  claim, with the same canonical value encoding (exact integers and
  rationals as decimal strings; matrices, vectors, lists, polynomials).
  One encoding is added to `core.transcript` for every user of it:
  `tv-float` writes a finite double exactly as `{"t":"dyadic","m","e"}`,
  the value m * 2^e, so interval endpoints and Taylor coefficients are
  recorded with no decimal rounding.
- `witness` holds the data in the table above, per kind.
- `premises` lists every statement the checker takes on trust for the
  claim. The checker sets, per kind, which premises are allowed; a claim
  cannot add its own. Every kind has none except `taylor_model_range`,
  whose remainder bound comes from a sampled derivative bound
  (`core.ad.taylor_models`) and is listed as the premise
  `remainder_bound`. Claims with premises are reported CONDITIONAL.
- `id` is `sha256:` and the sha256 of the canonical JSON text of the
  fields before it (`agent.crypto`'s sha256): the content address used to
  name a certificate, for example in a Lean theorem name.
- `hash` is `core.transcript`'s fnv1a-64 over every field before it, so a
  certificate claim is also a valid transcript claim. `core.transcript`'s
  hash rule is generalized to "every field except `hash`, in order", which
  is unchanged for four-field claims, and `replay_check.py` reads both
  schemas.

A document is `{"schema":"eshkol.certificate.v1","program","claims"}`. The
version is in the schema name; a later version adds kinds or fields under a
new name, and the checker refuses schemas it does not know.

Identifiers are names, not proofs. A mismatch rejects a claim; a match
proves nothing, because anyone can recompute a hash. The verdict rests only
on recomputation.

### 2. Producers: `core.certificate`

`core.certificate` computes each value through the library that owns it and
records the witness that library already has, or adds the missing piece to
that library:

- `exact-plu` (P A = L U) and `exact-pivot-columns` join
  `core.exact_linalg`;
- `exact-poly-gcdext` (Bezout in Q[x]) joins `core.polynomial_exact`, and
  `exact-poly-inverse-mod` is now expressed through it;
- `exact-gcdext` and `exact-expt-mod` join `core.ring`, next to
  `exact-prime?`;
- Smith forms come from `core.integer_linalg`, PSD factors from
  `core.exact_psd`, solve enclosures from `core.certified_solve`,
  polynomial normal forms from `core.symbolic`, Taylor models from
  `core.ad.taylor_models`.

The producers are `certify-rank`, `certify-determinant`,
`certify-smith-normal-form`, `certify-linear-solution`,
`certify-solve-enclosure`, `certify-psd`, `certify-integer-gcd`,
`certify-poly-gcd`, `certify-polynomial-identity`, `certify-primality`
(with `pratt-certificate`), `certify-polynomial-range`,
`certify-taylor-model-range`, `certify-cocycle` and
`certify-cohomology-class`. Each returns a certificate object whose value
is the computed result. Every witness entry must be exact: a value that
left the exact tower (for instance a rational whose numerator or
denominator outgrew 64 bits and became a double) raises instead of being
recorded.

Sinks are `core.transcript` handles with the certificate schema:
`certificate-open` is active iff `ESHKOL_CERTIFICATES` names a file, and
`certificate-open-path` / `certificate-write-file` write to an explicit
path. Emission is off by default and never prints.

### 3. The independent checker: `tools/certcheck/eshkol_certcheck.py`

The checker is one Python 3 file that imports only `json`, `fractions`,
`hashlib` and `argparse`. It imports nothing from the Eshkol tree and runs
no elimination, factorization or search: each kind is checked by products,
comparisons, modular powers and polynomial expansion. Its trusted base,
printed by `--trusted-base`, is the interpreter, those four modules, the
file, and one named theorem per kind (rank-nullity, multiplicativity of
the determinant, uniqueness of the Smith form, the Banach fixed point for
the solve enclosure, Lehmer's theorem for Pratt certificates, the centred
form for polynomial ranges). The compiler, the runtime, LLVM and the
producing library are outside it.

Python rather than C: exact rationals over unbounded integers are in the
Python standard library, so the checker contains no hand-written bignum
arithmetic, which would be the largest and least reviewed part of a C
checker.

Malformed input is always rejected cleanly: duplicate JSON keys, JSON
floats where decimal strings are required, fields out of order, unknown
kinds and unknown schemas are all REJECTED.

`tools/certcheck/tamper_suite.py` is the test of the checker. For every
claim it builds an unsealed edit, a resealed false output and a resealed
tampered witness (ids and hashes recomputed, as a forger would) and requires
every one to be rejected.

### 4. Lean export: `tools/certcheck/lean_export.py`

For the kinds whose statements are decidable finite facts over the
integers, the exporter writes Lean 4 theorems whose statement is the
witness identity and whose proof is `by decide`, so the Lean kernel
evaluates both sides:

- `determinant` (n <= 6): `det n A = d`, with `det` the cofactor
  expansion;
- `rank` (integral A): `A K = 0`, a nonzero r x r minor, a nonzero
  (n - r) x (n - r) minor of the kernel;
- `smith_normal_form`: `U A V = D`, `U U^-1 = I`, `V V^-1 = I`, the
  divisibility chain;
- `integer_gcd`: `s a + t b = g`, `a = g qa`, `b = g qb`, and from these,
  through the lemma `bezout_gcd` proved once in the file, that g divides a
  and b and every common divisor divides g;
- `primality`: the Pratt conditions for every node of the tree;
- `cocycle` and `cohomology_class`: the product identities and the
  remainder that rules out a coboundary.

The files depend on Lean core only. Mathlib is not installed on the
checking host, so the statements that need it are not stated as theorems:
that a Pratt tree implies primality (Mathlib's `lucas_primality`) and
rank-nullity are named in the file and in the trusted base. The exporter
is deterministic; `tests/certificates/lean/` holds the files exported from
the committed fixtures and a CTest entry requires them to be current.

### 5. The `(certify ...)` form

`(certify (f arg ...))` evaluates to `(f arg ...)`. When
`ESHKOL_CERTIFICATES` is set it also appends a certificate for that call to
the file. The native parser and the VM compiler both rewrite it to

```scheme
(certify-call "f" f (list arg ...))
```

which `core.certificate` defines. The operator must be a name, because the
producer is looked up by name in a registry: `exact-rank`, `exact-det`,
`smith-normal-form`, `exact-solve`, `certify-linear-solve`,
`exact-psd-factor`, `gcd`, `exact-gcdext`, `exact-poly-gcdext`,
`exact-prime?` and `tm-range` are registered, and `register-certifier!`
adds more. The compiler change is this rewrite and nothing else: no new
AST node, no new opcode, no runtime support. Without a sink the call costs
one environment check. The process sink is rewritten after every record,
so no claim is lost if the program exits without closing it. Like `delay`,
`certify` is a reserved form name.

## Alternatives rejected

- **Replay only (the transcript as it was).** A replay must re-implement
  every algorithm and grows with the problem; a witness check stays a
  product. The replay remains, and still runs on certificate documents,
  as a second independent recomputation.
- **A checker in C.** It would need its own bignum and rational layer,
  enlarging the trusted base with exactly the code most likely to hide an
  error.
- **Lean as the only checker.** Lean core without Mathlib cannot state
  primality or rank-nullity as theorems, and not every user has Lean. The
  Python checker covers every kind; Lean re-checks the finite facts with
  its kernel.
- **`native_decide` in the Lean files.** It trusts the Lean compiler and
  the code it generates; `decide` keeps the check in the kernel.
- **A hash as the proof of integrity.** fnv1a-64 is not collision
  resistant and sha256 only names content; neither proves a claim.
- **A compile-time flag that certifies every certified primitive.** It
  would certify calls the programmer did not ask about and would need
  codegen support for every primitive. The form marks exactly the steps
  whose results the program publishes.

## Consequences

- Every certified quantity Eshkol computes now has a certificate kind, a
  checker that does not trust Eshkol, and a tamper test.
- The decidable facts reach the Lean kernel today, on Lean core alone.
- Rational witnesses inherit the 64-bit numerator and denominator bound of
  Eshkol's rationals: a witness entry that overflows raises at emission,
  and the checker rejects any inexact value. Integer witnesses (Smith
  forms, Bezout, Pratt, cochains) are bignum-exact.
- The Taylor-model enclosure is checked in exact arithmetic but stays
  CONDITIONAL on its remainder premise. Planned for v1.7.0: a remainder
  computed from a validated derivative enclosure, which removes the premise,
  and the Lean statement of the enclosure that the Formal Verification item
  asks for.
- `certify` is reserved as a form name in native and VM programs.

## Staging

| Release | Stage | Gate |
|---|---|---|
| v1.7.0-synthesis | This record: the format, fourteen kinds, the checker, the tamper suite, the Lean export of seven kinds, the `(certify ...)` form. | `certificate_jit/aot`, `certificate_roundtrip_jit/aot`, `certificate_check_fixtures`, `certificate_tamper_fixtures`, `certificate_lean_export_current`; the exported Lean files checked by `lean` on a host that has it. |
| v1.7.0-synthesis | Planned for v1.7.0: a validated remainder for Taylor models (no premise); certificates for interval Newton steps of `core.certified_nonlinear` and continuation; derivation chains linking certificates by id. | Every certificate of the examples accepted without premises. |
| v2.0-starlight | Planned for v2.0: Lean statements with Mathlib where it is available (primality as `Nat.Prime`, rank as `Matrix.rank`), and the Lean kernel export of the compiler's normative corpus (ADR-0004). | The exported theorems proved against Mathlib definitions. |

## Evidence

- `tests/stdlib/certificate_test.esk`: every kind and its value, the
  claim format, `tv-float`, `exact-plu`, the `(certify ...)` form and the
  registry.
- `tests/certificates/run_certificate_roundtrip.py`: emission, the
  checker, the tamper suite and the Lean export, in JIT and AOT.
- `tests/certificates/fixtures/` and `tests/certificates/lean/`: committed
  certificate documents and the Lean files exported from them;
  [CERTIFICATE_LEAN_CHECK.md](../../reports/CERTIFICATE_LEAN_CHECK.md)
  records the Lean version and the check.

## References

- V. R. Pratt, "Every prime has a succinct certificate", SIAM J. Comput. 4
  (1975).
- D. H. Lehmer, "Tests for primality by the converse of Fermat's theorem",
  Bull. AMS 33 (1927).
- M. O. Rabin, "Probabilistic algorithm for testing primality", J. Number
  Theory 12 (1980).
- H. Cohen, *A Course in Computational Algebraic Number Theory*, GTM 138,
  2.4.14 (Smith normal form).
- S. M. Rump, "Verification methods: rigorous results using floating-point
  arithmetic", Acta Numerica 19 (2010).
- R. M. McConnell, K. Mehlhorn, S. Naher, P. Schweitzer, "Certifying
  algorithms", Computer Science Review 5 (2011).
