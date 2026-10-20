---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/algebraic_field.esk
  - tests/stdlib/algebraic_field_test.esk
---

# `core.algebraic_field` — selected exact real number fields

`(make-algebraic-field coefficients prime root-index [split-budget])` admits a
monic integer polynomial of degree 2 through 16. It verifies irreducibility
modulo the supplied prime (at most 65521), then uses exact Sturm isolation to
select the zero-based root in ascending real order. Modular admission is
sufficient but incomplete for irreducibility over Q. The default isolation
budget is 256; failure never admits an uncertified field.

`algebraic-field-degree`, `algebraic-field-polynomial`, `algebraic-generator`,
`algebraic-element` and `algebraic-coefficients` construct/inspect values.
Coefficients stay exact rationals, reduced modulo the defining polynomial.
`algebraic-add`, `algebraic-sub`, `algebraic-scale`, `algebraic-mul`,
`algebraic-inverse`, `algebraic-zero?`, `algebraic-equal?` and
`algebraic-rational` supply explicit field operations. Zero inversion refuses;
constant extraction returns `#f` for nonrational elements. Public vectors are
owned copies. Foreign field objects require an explicit transport.

`(algebraic-enclosure value [refinements])` returns rational lower/upper bounds,
with zero refinements by default. `(algebraic-sign value [budget])` returns
-1, 0 or 1; exact reduction decides zero and interval separation certifies
nonzero signs. Default budget is 256; exhaustion and arithmetic demotion raise.

`(make-algebraic-embedding source target generator-image)` checks the source
polynomial and the selected real interval, returning a checked transport
procedure. Wrong conjugates and foreign source values refuse.

No implicit field coercion, general compositum, complex embedding, native
tensor cell type or AD dispatch is added. Run CTest with
`-R 'polynomial_exact|algebraic_field|cyclotomic'`; inverse coefficients are
cross-checked against independent exact Gaussian solves.
