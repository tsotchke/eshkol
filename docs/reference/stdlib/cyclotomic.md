# `core.cyclotomic` — exact cyclotomic number fields

**Source:** [cyclotomic.esk](../../../lib/core/cyclotomic.esk).
**Require:** `(require core.cyclotomic)`; depends on `core.polynomial_exact` and `stdlib`, no compiler changes.

The module owns arithmetic in `Q(zeta_n) = Q[x]/(Phi_n)`. It promotes the
coefficient arithmetic previously repeated in the
[Turaev–Viro](../../../examples/mathematics_turaev_viro_cyclotomic.esk) and
[Reidemeister-torsion](../../../examples/mathematics_lens_spaces_reidemeister_torsion.esk)
examples. The standard cyclotomic-field construction is also described in the
[Sage number-field reference](https://doc.sagemath.org/html/en/reference/number_fields/sage/rings/number_field/number_field.html#sage.rings.number_field.number_field.CyclotomicFieldFactory).

```scheme
(require core.cyclotomic)
(define F (make-cyclotomic-field 4))
(define i (cyclotomic-root F 1))
(cyclotomic-rational (cyclotomic-mul i i)) ; => -1, exactly
(cyclotomic-coefficients
  (cyclotomic-inverse (cyclotomic-element F '(1 1)))) ; => #(1/2 -1/2)
```

| Procedure | Contract |
|---|---|
| `(make-cyclotomic-field n [order-budget])` | Construct the canonical order-`n` field. `n` is a positive exact integer; the default order budget is 256. |
| `(cyclotomic-order field-or-element)` | Inspect the defining order. |
| `(cyclotomic-degree field)` | Dimension of the rational power basis. |
| `(cyclotomic-polynomial field)` | Fresh coefficient vector of `Phi_n`, constant coefficient first. |
| `(cyclotomic-element field coefficients)` | Reduce an exact rational scalar, list or vector of power-basis coefficients. An empty list means zero. |
| `(cyclotomic-coefficients a)` | Fresh reduced coefficient vector of length `phi(n)`. |
| `(cyclotomic-root field k)` | `zeta_n^k`, for any signed exact integer `k`. |
| `(cyclotomic-add a b)` / `(cyclotomic-sub a b)` | Exact sum/difference within one canonical order. |
| `(cyclotomic-scale a q)` | Multiply by an exact rational. |
| `(cyclotomic-mul a b)` | Exact convolution followed by reduction modulo `Phi_n`. |
| `(cyclotomic-inverse a)` | Exact polynomial Euclid inverse; zero raises. |
| `(cyclotomic-zero? a)` / `(cyclotomic-equal? a b)` | Decide by reduced exact coefficients. Different orders raise. |
| `(cyclotomic-rational a)` | Extract an exact rational constant, or return `#f`. Exact zero returns `0`. |
| `(cyclotomic-galois a r)` | Substitute `zeta_n -> zeta_n^r`; requires `gcd(r,n)=1`. `r=-1` is complex conjugation. |
| `(cyclotomic-norm a)` | Product of all unit-indexed Galois conjugates, returned as an exact rational. |

Fields and elements are opaque immutable Scheme values. Constructor inputs,
coefficient observations and modulus observations do not share mutable public
vectors. Independently constructed fields of the same order are compatible:
their generator and polynomial are canonical. Different orders refuse implicit
transport, including isomorphic presentations such as orders 3 and 6. There
is no mutable global interning table or hidden field cache.

Every scalar arithmetic result is checked for exact rational type. Floating
inputs refuse, and a numeric runtime that silently demotes an overflowing
rational causes this module to raise instead of returning rounded field data.
Scalar arithmetic limits and OALR allocation limits still apply. Use ordinary
lists or the `vector` procedure for exact coefficient data; a runtime that
reads numeric vector literals as double tensors will have those inputs refused.

Cyclotomic polynomials are constructed by exact divisor-product division,
with a construction-local memo table and checked zero remainders. This is a
specialized field constructor using the cyclotomic polynomial theorem; it does
not admit arbitrary polynomial quotients as fields. General algebraic numbers,
real-root isolation, arbitrary composita, cross-order embeddings, native tensor
storage, automatic differentiation dispatch and minimal-polynomial inference
remain outside this module. Tagged field elements use the explicit operations
above rather than overloading Eshkol's scalar `+`, `*` or `exact?` predicates.

## Verification

```sh
ctest --test-dir build --output-on-failure \
  -R 'cyclotomic|torsion_lens_spaces_reidemeister'
python3 scripts/build_example_catalogue.py --check
```

[The arithmetic suite](../../../tests/stdlib/cyclotomic_test.esk) checks known
prime/composite polynomials, primitive orders, field laws, inverses, conjugation,
exact finite-update identities, copied public data and admission failures.
Euclid inverses are compared with `core.exact_linalg` Gaussian solves, and
Galois norms with independent Bareiss determinants of multiplication matrices.
Both migrated examples retain their mathematical verdicts and JIT/AOT gates.
The release catalogue keeps historical source receipts separate from the
refactored sources; new release measurements must be obtained for those bytes.
