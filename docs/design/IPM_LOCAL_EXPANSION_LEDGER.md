# Incompressible porous media: local velocity expansion, proof ledger

Machine-readable twin: `.icc/ipm-proof-ledger.yaml`. Program:
`examples/mathematics_ipm_velocity_expansion.esk` (CTest
`ipm_velocity_expansion_closes_jit` / `_aot`; ICC oracle `ipm-local-expansion`,
trace produced by `scripts/run_ipm_velocity_expansion_gate.sh`).

Source: D. Cordoba, L. Martinez-Zoroa, *Finite time singularities of smooth
solutions for the 2D incompressible porous media (IPM) equation with a smooth
source*, arXiv:2410.22920v3, Section 2. The section constructs the local
approximation `V^K` of the velocity operator `V(w)(x) = ∫ h1/|h|² w(x+h) dh`
on oscillatory densities `w = f(x) sin(N(b x1 + a x2) + Θ)` and the
truncated velocity `u^K` built from it (Corollary 2.6.1).

Status vocabulary is exactly one of **EXACT** (checked with exact rational
arithmetic by a passing program), **VALIDATED** (checked numerically to a
stated tolerance), **ANALYTIC-ONLY** (no computable content in the tree; the
missing capability is named). Every EXACT row below is exercised by the one
program, which exits nonzero on any failed check and carries a negative
control per claim.

**What makes the section computable.** `V` is `−2π ∂1 Δ⁻¹`: the kernel
`h1/|h|²` is the gradient of the logarithm, the fundamental solution of the
Laplacian. The expansion constants of Definition 2.5, which the paper defines
as `N → ∞` limits of oscillatory integrals (Lemma 2.4), are therefore the
Fourier-multiplier expansion coefficients

```text
c_{i,j} = C_0 · (i(−i)^i) · ∂1^{i−j} ∂2^{j} σ(b, a) / ((i−j)! j!),   σ(ξ) = ξ1/|ξ|²,   C_0 = 2π,
```

with the real part on `sin` and the imaginary part on `cos`. Every identity
of the section is homogeneous in `C_0`, so it divides out and the whole
section is exact rational arithmetic at a rational direction `(b, a)`. The
derivatives of `σ` are computed by `taylor` on line restrictions and
assembled into mixed partials by an exact Vandermonde solve
(`core.exact_linalg`); nothing is transcribed by hand. The unit-circle
normalization `a² + b² = 1` is not load-bearing: `σ` carries `|k|²` itself
and the program verifies closure at a direction off the circle.

**Counts:** EXACT 7, VALIDATED 0, ANALYTIC-ONLY 4, total 11.

| id | row | paper ref | operation | status | program(s) / missing capability |
|---|---|---|---|---|---|
| ipm-proof-001 | 1 | Lemma 2.6 | `c^s_{0,0} = 0`, `c^c_{0,0} = b C_0` with `C_0 > 0`; the value follows the phase direction | EXACT | ipm_velocity_expansion_closes |
| ipm-proof-002 | 2 | Remark 4 | `c^s_{i,j} = 0` for even `i`, `c^c_{i,j} = 0` for odd `i`, `i ≤ 4`; complementary constants nonzero | EXACT | ipm_velocity_expansion_closes |
| ipm-proof-003 | 3 | Definition 2.5; Lemma 2.4 | every `c_{i,j}`, `i ≤ 4`, in closed form; an order-by-order solve of the closure relation reproduces the 15-term `V⁴(ρ)` term by term at two directions | EXACT | ipm_velocity_expansion_closes |
| ipm-proof-004 | 4 | (5)-(8); Definition 2.5 | `Δ V^K(w) + 2π ∂1 w` vanishes exactly through order `N^{−(K−1)}`, survives at `N^{−K}`, `K = 0..4`; failing coefficient printed | EXACT | ipm_velocity_expansion_closes |
| ipm-proof-005 | 5 | Corollary 2.6.1 | `u⁰ = (a b C_0 ρ, −b² C_0 ρ)` | EXACT | ipm_velocity_expansion_closes |
| ipm-proof-006 | 6 | Corollary 2.6.1 | `div u^K` and `curl u^K + ∂1 ρ` vanish exactly through `N^{−(K−1)}`, survive at `N^{−K}`, `K = 0..4` | EXACT | ipm_velocity_expansion_closes |
| ipm-proof-007 | 7 | Remark 5; Definition 2.1 | odd density gives odd `u^K` (checked at `K = 3`; fails with an odd part added to `f`) | EXACT | ipm_velocity_expansion_closes |
| ipm-proof-008 | 8 | Lemma 2.2; (13) | far-field remainder bound | ANALYTIC-ONLY | certified enclosures with a rigorous bound for an oscillatory integral over an unbounded domain |
| ipm-proof-009 | 9 | Lemma 2.3 | near remainder `|R_{K,N}| ≤ C N^{−(K+2)ε} ‖f‖_{C^{K+1}}` | ANALYTIC-ONLY | certified enclosures (proved Taylor-model remainder) |
| ipm-proof-010 | 10 | Lemma 2.4 | convergence rate `N^{−A}` of the truncated oscillatory integrals (their limits are row 3) | ANALYTIC-ONLY | rigorous oscillatory quadrature with a tail enclosure |
| ipm-proof-011 | 11 | (19); (22) | `C^J` bounds on `V − V^K` and `u − u^K` | ANALYTIC-ONLY | certified enclosures; the formal series (rows 4, 6) closes one order beyond the analytic bound, which carries rows 8-10 |

Negative controls in the program: a perturbed constant drops the closure
order from 2 to 1; the symbol of `∂2 Δ⁻¹` in place of `∂1 Δ⁻¹` fails at the
lowest order and disagrees with the order-by-order route; dropping the
`V^{K−1}(∂_j f sin)` terms of Corollary 2.6.1 drops the divergence order from
3 to 0; stopping at `V^{K−1}` leaves the order-`(K−1)` term; an odd part added
to `f` breaks oddness.

Routed-around defects (not fixed here): `expt` on an exact rational returns
inexact `0` (integer powers are repeated multiplication); `#(...)` literals
promote to f64 tensors (`vector`/`list->vector` throughout); no loop variable
is captured by a differentiand; every derived constant and evaluated
coefficient is asserted `exact?` because int64 rationals demote silently on
overflow.
