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

## Certified coprime composita

`(make-algebraic-compositum left right [degree-budget [candidate-budget
[refinement-budget]]])` returns `(vector field embed-left embed-right)`.
The two procedures explicitly transport elements of the selected source fields
into the resulting field. Composing the same field object returns that object
and checked copying transports. Distinct sources require coprime degrees;
noncoprime composition refuses rather than guessing an intersection.

For distinct sources the constructor forms their exact tensor-product basis
and searches primitive elements `gamma = alpha + c*beta`, starting at `c=1`.
It verifies a full power basis, guards every inverse coefficient for exactness,
and checks the entire matrix inverse residual. The resulting monic polynomial
must have integer coefficients. Coprime certified source degrees and the full
power basis certify the combined extension; admission does not pretend to have
found another modular irreducibility prime. Both source polynomial relations
and selected real embeddings must pass before either transport is returned.
Interval refinement selects the target root and checks the images against the
original source intervals.

Defaults are combined degree 16, eight primitive candidates and 128 interval
refinements; maxima are 16, 16 and 256 respectively. Budgets must be nonnegative
exact integers. Exhaustion, arithmetic demotion, singular candidate bases or
uncertified real embeddings never return an unchecked field. The same-object
identity case still observes the degree budget and needs no search/refinement.

`(algebraic-field-certificate field)` returns copied public provenance:
`(modular-prime prime polynomial interval)` for modular admission, or
`(coprime-compositum degree multiplier left-description right-description
polynomial interval)` for composition. Nested descriptions contain copies;
this observer is not a serialized-certificate loader or an unchecked factory.

The [joint example](../../../examples/nested_algebraic_fields.esk) constructs
`Q(sqrt(2),cuberoot(2))` and uses both checked transports with
[nested exact AD](ad_algebraic.md). Composition remains explicit; general
noncoprime composita, complex embeddings and native tensor cells are outside
this module's admitted surface.

Run CTest with `-R 'polynomial_exact|algebraic_field|algebraic_compositum|cyclotomic'`.
Inverse coefficients and source images are checked against independently
derived exact formulas as well as full defining residuals.
