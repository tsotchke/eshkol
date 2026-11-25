---
kind: explanation
status: current
owner-area: ad
since: v1.4.0-connection
sources:
  - lib/core/ml/diffopt.esk
  - lib/core/ml/gradient_estimators.esk
  - lib/core/ad/tape.esk
  - lib/core/exact_linalg.esk
  - tests/ml/diffopt_test.esk
---
# ADR-0031: Differentiating through optimization is one tape pullback per rule

**Status:** Accepted — implemented in v1.4.0-connection (`core.ml.diffopt`)
**Date:** 2026-10-23
**Scope:** The Scheme-level reverse-mode tape (`core.ad.tape`) on native JIT
and AOT; the bytecode VM runs the module except seeded noise generation
(Planned for v1.4.1).

## Context

Many layers worth training through return the *solution* of an optimization
problem rather than a composition of arithmetic: a ranking, a shortest path, a
top-k selection, a ridge fit, a fixed point, a projection onto the simplex.
The forward computation is a solver; reverse-mode AD of the solver's
instructions is either zero almost everywhere (combinatorial solvers),
undefined (an `argmax`), or wasteful and unstable (unrolling hundreds of
iterations of a fixed-point loop). The Elements of Differentiable Programming
(Blondel and Roulet, chapters 11, 13 and 14) organizes the answers into four
rules, each of which replaces "differentiate the solver" by a pullback defined
by the problem:

1. **Blackbox combinatorial solvers** (Vlastelica et al., ICLR 2020). For
   `y(w) = argmin_{y in Y} <w, y>`, the gradient of the piecewise-affine
   interpolation of `L(y(w))` is `-(y(w) - y(w + lambda g)) / lambda` with
   `g = dL/dy`: one extra solver call per backward pass.
2. **Perturbed optimizers** (Berthet et al., NeurIPS 2020). The smoothed
   maximizer `y_eps(theta) = E[argmax_{y in C} <y, theta + eps Z>]` is
   differentiable with `J y_eps = (1/eps) E[y*(theta + eps Z) grad nu(Z)^T]`,
   `nu = -log density of Z`, and the Fenchel-Young loss
   `F_eps(theta) - <theta, y>` has gradient `y_eps(theta) - y`.
3. **Implicit differentiation.** If `F(x*, theta) = 0` and `d_x F` is
   invertible, the pullback of `g` is `-(d_theta F)^T u` where
   `(d_x F)^T u = g`. Fixed points (`F = T(x, theta) - x`) and argmin layers
   (`F = grad_x f`) are the same rule.
4. **Smooth relaxations with closed-form Jacobians:** softmax (the entropic
   projection onto the simplex) and sparsemax (the Euclidean projection), with
   their Fenchel-Young losses.

Eshkol already had every substrate these rules need, each in exactly one
place: the custom-op tape with an explicit backward closure (`record-op!` in
`lib/core/ad/tape.esk`), argmax and softmax and Gumbel noise
(`core.ml.gradient_estimators`), exact linear algebra on the scalar exact tower
(`exact-solve`, `exact-matrix-transpose` in `core.exact_linalg`), the f64
`linear-solve`, the compiled `jacobian`, `gradient` and `hessian` operators, and
an isolated seeded generator (`make-prng`).

## Decision

Add one module, `core.ml.diffopt`, in which **each rule is one pure
vector-Jacobian-product function plus one recorder** that installs it on the
tape with `record-op!`:

| Rule | Pure pullback | Tape recorder(s) |
|---|---|---|
| Blackbox solver | `blackbox-vjp` | `blackbox-solve` |
| Perturbed optimizer | `perturbed-estimate`, `perturbed-vjp` | `perturbed-solve` (with `perturbed-argmax`, `perturbed-top-k`), `perturbed-max` |
| Fenchel-Young loss | prediction minus target | one recorder shared by `perturbed-fenchel-young`, `fenchel-young-softmax`, `fenchel-young-sparsemax` |
| Implicit function theorem | `implicit-root-vjp` | `implicit-root`; `implicit-fixed-point` and `implicit-argmin` build their `F` and call it |
| Simplex projections | `softmax-vjp`, `sparsemax-vjp` | `tape-softmax`, `tape-sparsemax` |

The following choices are part of the decision:

- **No new AD engine.** Every layer is a `record-op!` node, so it composes with
  every other tape op in one reverse pass. No pullback in the module evaluates
  a finite difference; the test asserts `(= (ad-finite-difference-evals) 0)`.
- **One linear solve for the adjoint system.** `diffopt-linear-solve` routes a
  system whose entries are all exact to `exact-solve` (an exact rational
  answer) and any other system to the f64 `linear-solve`. Implicit
  differentiation is therefore exact on exact data whenever the caller's
  partial Jacobians are exact.
- **Partial Jacobians come from the compiled operators by default**
  (`jacobian` for `d_x F` and `d_theta F`, `hessian` and the Jacobian of
  `gradient` for an argmin layer), and may be passed explicitly. The compiled
  vector-point operators answer in f64, so the exact path is the explicit one.
- **Fixed points and argmin layers are not separate engines**: both construct
  their residual and partial Jacobians and call `implicit-root`.
- **One softmax pullback.** `softmax-vjp` moves into
  `core.ml.gradient_estimators`, and Gumbel-Softmax now calls it, as do
  `tape-softmax`, `softmax-jacobian` and the Fenchel-Young softmax loss.
- **One noise source.** `sample-gumbel-noise` gains an optional generator
  argument and `sample-normal-noise` joins it; with a generator from
  `(make-prng seed)` the draws are reproducible and leave the shared generator
  untouched. A perturbation is the explicit value `#(perturbation kind samples)`,
  so the forward pass and the backward pass use the same draws (common random
  numbers): the Monte Carlo value and the Monte Carlo Jacobian are estimated
  from one sample.
- **Solver conventions are stated, not inferred.** Blackbox solvers minimize
  `<w, y>`; perturbed solvers maximize `<y, theta>`, as in the two papers.
  `rank-solver` (normalized descending rank) and `grid-path-solver` (monotone
  minimum-cost grid path by dynamic programming) are provided as reference
  solvers and used by the tests.
- **The perturbed Fenchel-Young loss omits `eps Omega(y)`**, which does not
  depend on `theta` and has no closed form for a general polytope. The softmax
  and sparsemax Fenchel-Young losses are complete, nonnegative, and zero
  exactly when the target equals the prediction; the sparsemax loss is exact on
  exact input.

## Verification

`tests/ml/diffopt_test.esk` (CTest `diffopt_runtime_smoke` and
`diffopt_aot_smoke`; ICC probe `diffopt_pullbacks_exact`) compares every
pullback with an independent reference:

- blackbox ranking and shortest-path pullbacks equal hand-computed exact
  rationals, and gradient descent through them reaches the target ranking and
  the target path;
- Gumbel-perturbed argmax equals `softmax(theta/eps)` in expectation, its Monte
  Carlo Jacobian equals the softmax Jacobian divided by `eps`, and the
  perturbed maximum equals `eps logsumexp(theta/eps) + eps gamma`; Normal noise
  in two dimensions matches `Phi(d / (eps sqrt 2))` and its derivative;
- the ridge least-squares sensitivity `dw*/dlambda` from `implicit-root-vjp` is
  the exact rational closed form `-(X^T X + lambda I)^{-1} w*`, and the
  compiled-AD default agrees with it to `1e-12`; an exact linear fixed point
  gives `(I - A)^{-T} g` exactly; a nonlinear fixed point matches
  `1 / (1 + sin(x)/2)` and a central difference of the iterated solver;
- sparsemax and its Jacobian are exact on rationals and agree with central
  differences in f64.

## Consequences

- The four families from the literature are available on the tape with their
  published semantics, and exact data stays exact through the pure pullbacks.
- The numeric tape seeds with `1.0` and accumulates in f64, so a cotangent read
  back from `tape-gradient` is f64 even when the pullback computed it exactly;
  exact cotangents are read from the pure pullback functions.
- Reaching these pullbacks from the compiled `gradient` operator, through the
  native custom-VJP node, is Planned for v1.4.1.
- On the bytecode VM everything except `perturbation-noise` runs (the VM has no
  binding for `make-prng`; an explicit `make-perturbation` works). Seeded noise
  on the VM is Planned for v1.4.1 and is recorded in `tests/vm_parity/PARITY.tsv`.

## References

- M. Blondel and V. Roulet, *The Elements of Differentiable Programming*,
  2024, chapters 11, 13 and 14.
- M. Vlastelica, A. Paulus, V. Musil, G. Martius and M. Rolinek,
  *Differentiation of Blackbox Combinatorial Solvers*, ICLR 2020.
- Q. Berthet, M. Blondel, O. Teboul, M. Cuturi, J.-P. Vert and F. Bach,
  *Learning with Differentiable Perturbed Optimizers*, NeurIPS 2020.
- M. Blondel, A. F. T. Martins and V. Niculae, *Learning with Fenchel-Young
  Losses*, JMLR 2020.
- A. F. T. Martins and R. F. Astudillo, *From Softmax to Sparsemax*, ICML 2016.
