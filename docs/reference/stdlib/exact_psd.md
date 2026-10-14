# `core.exact_psd` — exact rational PSD certificates

Require `(require core.exact_psd)`. Matrices use generic vectors of row vectors,
constructed with `vector`, preserving exact rational cells.

`(exact-psd-factor M)` returns an opaque immutable result. Malformed, asymmetric
or inexact inputs raise; loss of exactness during arithmetic raises too.

- `(psd-factor? result)` decides whether the matrix is positive semidefinite.
- On success, `(psd-factor-rows result)` returns fresh rows B and
  `(psd-factor-weights result)` returns fresh strictly positive weights D, with
  `M = transpose(B) diag(D) B`. The number of weights is the exact rank.
- On symmetric indefiniteness, `(psd-negative-direction result)` returns a
  fresh rational vector v satisfying `transpose(v) M v < 0`; on success it
  returns `#f`. Factor observers refuse indefinite results.

Zero and singular matrices succeed, including the empty matrix. Inputs are
preserved and public observations do not expose retained mutable vectors.
This is dense exact Schur elimination, without numerical tolerances or sparse
cost guarantees. Rational coefficient growth remains subject to runtime limits.

Run `ctest --test-dir build --output-on-failure -R exact_psd`. Tests exhaust
all 729 ternary symmetric 3x3 matrices against an independent principal-minor
oracle, checking each reconstruction or negative-energy certificate.
