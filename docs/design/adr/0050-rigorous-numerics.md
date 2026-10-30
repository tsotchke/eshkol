---
kind: explanation
status: current
owner-area: numerics
since: v1.5.0
sources:
  - lib/core/ad/rigorous_interval.esk
  - lib/core/ad/interval.esk
  - lib/core/ad/rigorous_taylor_models.esk
  - lib/core/ad/taylor_models.esk
  - lib/core/ad/interval_jet.esk
  - lib/core/ad/compact_bounds.esk
  - lib/core/ad/interval_tensor.esk
  - lib/core/torus.esk
  - tests/stdlib/rigorous_numerics_test.esk
  - tests/vm_parity/corpus/102_rigorous_numerics.esk
---
# ADR-0050: Rigorous numerics — directed rounding by default, compact-set bounds, torus averaging, interval and Taylor-model element tensors

**Status:** Accepted (for root review; lands in rung order)
**Date:** 2026-10-30
**Rungs:** v1.5.0-intelligence (directed-rounding default, compact-set bounds, torus
averaging); v1.6.0-reasoning (interval and Taylor-model tensor element types)
**Depends on:** `fl-next-up` / `fl-next-down` (v1.3.5), `core.ad.rigorous_interval`,
`core.ad.rigorous_taylor_models`, the native AD operators, exact bignum rationals

## Context

v1.3.5 shipped proof-backed interval primitives (`core.ad.rigorous_interval`) behind an
opt-in `rigorous?` flag, while `core.ad.interval` kept an outward relative-epsilon widening
whose soundness rested on a libm error budget, and `taylor-model` padded a sampled remainder
by the same epsilon. The Navier–Stokes mechanization
(`docs/design/NAVIER_STOKES_RESIDUAL_MECHANIZATION.md`, rows 19, 35, 37, 50–52, 58, 67)
needs three facilities on top of that: certified sup/inf over a compact parameter box
(a compactness constant produced rather than asserted), periodic averaging on `T¹`/`T²` with
exact Haar and angular means and the inverse of a directional derivative on the zero-mean
subspace, and enclosures that carry derivatives.

## Decision

### 1. Directed rounding is the default of `core.ad.interval`

Every `interval-*` operation evaluates through `core.ad.rigorous_interval` with no flag.
Rounding is directed by stepping each round-to-nearest endpoint one ulp outward with
`nextafter` (IEEE 754-2019 §5.3.1 nextUp/nextDown): a single correctly rounded operation errs
by at most half an ulp, so one step brackets the exact result. The FPU rounding mode
(`fesetround`) is not used: it is per-thread ambient state that does not survive a JIT call
boundary, the bytecode VM or WebAssembly. The trailing `rigorous?` argument remains accepted
and has no further effect; `iv-eps`, `iv-rel-pad`, `iv-abs-pad` stay exported as named
constants that no operation uses. `taylor-model` rounds its remainder magnitude upward from
exact arithmetic instead of padding it.

Elementary functions are not libm calls with an error budget: exp, log, atan and sin/cos are
range-reduced Taylor polynomials with exact rational coefficients plus proven Lagrange,
integral or alternating-series remainders; sqrt is bracketed by certified squaring. This ADR
tightens that layer:

- remainder magnitudes are computed exactly over the rationals and rounded upward once
  (`ia-exact-up`), never through `exact->inexact` followed by a single nudge;
- exp, log and atan (monotone) and sqrt are evaluated per endpoint, so an interval of any
  width is accepted and the dependency problem disappears; sin and cos take the hull of the
  endpoint values and every extremum whose rigorous location `c + 2kπ` may lie in the
  interval, intersected with `[-1, 1]`;
- even powers square through `ia-sqr` (`[-2,3]² = [0,9]`), by binary powering;
- an exact zero operand gives an exact product and an exact sum (no `[-tiny, tiny]`);
- outward-rounded coefficient enclosures are cached per series order.

### 2. Interval automatic differentiation (`core.ad.interval_jet`)

Two element types: the interval Taylor jet (coefficient enclosures `Cₙ ⊇ f⁽ⁿ⁾(X)/n!`,
computed by the Taylor-coefficient recurrences for ×, ÷, exp, log, sqrt, sin/cos, atan in
directed-rounding interval arithmetic) and the interval gradient (forward mode, first
order). Generic operators `rn+ rn- rn* rn/ rn-neg rn-sqr rn-expt rn-exp rn-log rn-sqrt
rn-sin rn-cos rn-atan` dispatch on their arguments: plain numbers go to the native
operators, so the native `derivative`, `gradient`, `derivative-n` and `taylor`
differentiate a function written with them exactly as before (exactly at exact points);
intervals, jets, gradients and rigorous Taylor models go to their rigorous rules. One
function therefore runs on every carrier, and the tests compare the carriers against each
other. `taylor-model-certified` builds a rigorous Taylor model (Berz–Makino) whose
remainder is the jet enclosure of `f⁽ᵏ⁺¹⁾` over the whole domain — no sampling.

`tm-deriv` now differentiates only models whose remainder is exactly `[0,0]`: a real-domain
remainder bounds values, not slopes, so the earlier Cauchy-estimate rule (which assumed a
complex-disk bound the real interval arithmetic does not provide) is retired, and the error
message points to `interval-derivative-n`. Rigorous Taylor-model coefficients are held
exactly, so coefficient arithmetic never rounds.

### 3. Compact-set bounds (`core.ad.compact_bounds`)

`box-sup`, `box-inf` and `box-certify-bound` run branch and bound over a box (a vector of
intervals). Each box is enclosed by the natural extension intersected with the mean-value
form `f(c) + Σ Dᵢ (Xᵢ − cᵢ)`, both from one interval-gradient pass. The lower bound of the
supremum is the largest certified lower endpoint of `f` at an evaluated midpoint; boxes
below it are discarded; a coordinate on which `Dᵢ ≥ 0` (or `≤ 0`) collapses to the face that
holds the supremum (monotonicity pruning); the box with the largest upper bound is bisected
along the coordinate maximizing `width(Xᵢ)·|Dᵢ|`, at the exact midpoint when its endpoints are
exact. The result is a certificate `[L, U] ∋ sup f` with a witness point, box counts and
whether the tolerance was reached; the enclosure is sound in either case.

### 4. Torus fields (`core.torus`)

Coordinates are angles on `Tᵈ = ℝᵈ/(2πℤ)ᵈ`; the Haar measure is normalized. Trigonometric
polynomials are exact data (canonical frequency vectors, exact coefficients): sums, products
by the product-to-sum identities, partial and directional derivatives, the Haar mean, partial
angular means, and the inverse of `v·∇` on the zero-mean subspace (an error names any
resonant frequency `v·k = 0`) are exact on exact input. Evaluation and evaluation at a phase
map go through the generic operators, so the native AD operators propagate the chain rule
through the phase, and interval angles give enclosures. Supported fields carry a procedure
and a support (boxes of closed arcs in turns, exact rationals); products intersect supports
exactly and a product of disjointly supported fields is the certified zero field. Their Haar
mean is an enclosure: a cell of width `h` contributes `f(c) + C₂(c) h²/12 + C₄(X) h⁴/80` on
`T¹` (Taylor with Lagrange remainder; odd moments of a centered cell vanish), and on `Tᵈ`
`f(c) + Σ C₂,ᵢᵢ(X) h²/12 ± Σ |Hᵢⱼ(X)| h²/16`, refined until narrower than the tolerance.

### 5. Interval and Taylor-model element tensors (`core.ad.interval_tensor`, v1.6.0 item)

An element tensor is a shape plus a row-major Scheme vector of carrier values. Elementwise
arithmetic, elementary functions, `itensor-matmul`, `itensor-sum` and `itensor-transpose`
apply the generic operators entry by entry, so one tensor program evaluates to a tensor, an
interval tensor, a tensor of derivative enclosures (jets) or a tensor-valued Taylor model,
and the native AD operators differentiate it on numbers. The AD rules are the element rules
of §2 composed with the bilinear structure of matmul. Interval tensors convert to and from
dense f64 `(lower, upper)` pairs with outward rounding, which is the boundary with the
existing tensor runtime and the tensor-valued Taylor towers of `core.ad.tensor_tower`. Boxed
entries are deliberate: dense tensors store raw binary64 bit patterns.

## Soundness argument and references

Every enclosure reduces to: (a) the half-ulp bound of one correctly rounded IEEE operation
and one `nextafter` step per endpoint; (b) exact rational arithmetic, which never rounds;
(c) a remainder theorem stated at the call site (Lagrange, integral form, alternating
series); (d) the inclusion property of interval extensions (Moore's fundamental theorem).
Branch and bound only discards a box whose certified upper bound is below a certified lower
bound of the supremum.

R. E. Moore, *Interval Analysis* (1966); R. E. Moore, R. B. Kearfott, M. J. Cloud,
*Introduction to Interval Analysis* (SIAM 2009); R. B. Kearfott, *Rigorous Global Search:
Continuous Problems* (Kluwer 1996); E. Hansen, G. W. Walster, *Global Optimization Using
Interval Analysis* (2nd ed., 2004); A. Neumaier, *Interval Methods for Systems of Equations*
(1990); W. Tucker, *Validated Numerics* (Princeton 2011); K. Makino, M. Berz, *Taylor models
and other validated functional inclusion methods* (Int. J. Pure Appl. Math. 4, 2003) and
M. Berz, K. Makino, *Verified integration of ODEs and flows using differential algebraic
methods on high-order Taylor models* (Reliable Computing 4, 1998); A. Griewank, A. Walther,
*Evaluating Derivatives* (2nd ed., 2008), ch. 13; IEEE 754-2019 §5.3.1.

## Verification

`tests/stdlib/rigorous_numerics_test.esk` (JIT and AOT CTest entries
`rigorous_numerics_runtime_smoke`, `rigorous_numerics_aot_smoke`) checks against exact
rational ground truth computed inside the test from series with proven tails: randomized
containment of exp, log, sin, cos, atan, sqrt and a polynomial on exact and binary64
intervals (100% required), jet derivatives equal to the native exact AD at exact points,
sup/inf against closed forms, a produced compactness threshold, Parseval identities, the
inverse directional derivative, chain rule through a phase map, support bookkeeping, the
enclosure of `⟨exp cos θ⟩ = I₀(1)`, and element-tensor programs on every carrier.
`tests/vm_parity/corpus/102_rigorous_numerics.esk` is the VM-parity program;
`tests/stdlib/certified_enclosures_test.esk` checks the new default.

## Consequences

Every caller of `core.ad.interval` gets proven enclosures without changing code; results
that were exact stay exact and binary64 results move by at most a few ulps outward or
inward (tighter where endpoint evaluation removes the dependency problem). `tm-deriv` raises
on a model with a nonzero remainder. Graded coefficient classes (the second half of the
v1.6.0 item) and the certificate export of v1.7.0 are separate work.
