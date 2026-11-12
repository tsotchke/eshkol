---
kind: reference
status: current
owner-area: stdlib
since: v1.7.0-synthesis
sources:
  - tools/certcheck/lean_export.py
  - tests/certificates/fixtures/certificate_test.json
  - tests/certificates/fixtures/certificate_test_certify_forms.json
  - tests/certificates/lean/CertCertificateTest.lean
  - tests/certificates/lean/CertCertificateTestCertifyForms.lean
---

# Lean check of the exported certificate facts (ADR-0047)

The certificate documents in `tests/certificates/fixtures/` were emitted by
`tests/stdlib/certificate_test.esk` (explicit certificates and the
`(certify ...)` forms) on Linux x86-64. `tools/certcheck/lean_export.py`
exported their decidable finite facts to the two files in
`tests/certificates/lean/`; CTest `certificate_lean_export_current`
requires those files to equal a fresh export.

| File | Claims exported | Theorems (`by decide`) | Not exported |
|---|---|---|---|
| `CertCertificateTest.lean` | 13 | 15 (+ `bezout_gcd`) | 13 (kinds checked by the Python checker only, and compositeness claims) |
| `CertCertificateTestCertifyForms.lean` | 5 | 6 (+ `bezout_gcd`) | 0 |

Exported kinds: determinant, rank, Smith normal form, integer gcd (Bezout
identity and the greatest-common-divisor theorem through `bezout_gcd`),
primality (Pratt trees for 2, 1000003 and 2^61 - 1), cocycle and both
cohomology-class verdicts.

## Result

```text
$ lean --version
Lean (version 4.12.0, arm64-apple-darwin23.6.0, commit dc2533473114, Release)
$ lean tests/certificates/lean/CertCertificateTest.lean            # no output: accepted
$ lean tests/certificates/lean/CertCertificateTestCertifyForms.lean # no output: accepted
```

Both files are also accepted by Lean 4.30.0-rc2. Each check takes under one
second; the kernel evaluates the products and modular powers with its
built-in natural-number arithmetic. A control file with one Pratt exponent
and one base changed is rejected (`decide` reports the proposition false).

The files depend on Lean core only. Two theorems they rely on are named,
not re-proved, because Mathlib is not installed where they were checked:
Lehmer's theorem (a Pratt tree implies primality; Mathlib
`lucas_primality`) and rank-nullity.

Reproduce:

```text
python3 tools/certcheck/lean_export.py --dir tests/certificates/fixtures --out-dir /path/to/out
lean /path/to/out/CertCertificateTest.lean
./scripts/run_certificate_gate.sh      # also runs lean on tests/certificates/lean when lean is on PATH
```
