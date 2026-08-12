# Automatic Differentiation — Reference

Complete, machine-verified reference for Eshkol's automatic-differentiation
operators (v1.3.4-evolve). Every signature, example, and output on these pages
was produced by running the current compiler; open bugs are documented against
their ledger id, never hidden.

For conceptual background (the three AD modes, node opcodes, tensor backward
pass, numeric-type boundary) see the breakdown:
[../../breakdown/AUTODIFF.md](../../breakdown/AUTODIFF.md).

## Pages

| Page | Contents |
|------|----------|
| [operators.md](operators.md) | Every operator — `derivative`, `gradient`, `jacobian`, `hessian`, `laplacian`, `directional-derivative`, `divergence`, `curl`, `diff` — with signature, accepted point types, binding forms, capture rules, and composition/nesting. |
| [architecture.md](architecture.md) | Forward 4-component Taylor jet, reverse tape, the `__ad_pert_level` runtime perturbation counter, mixed reverse-over-forward mode (v1.3), numeric boundary, measured performance. |
| [support-matrix.md](support-matrix.md) | The AD-oracle support matrix (235 probes / 490 checks), the PASS cells, the five cells that were open through v1.3.3 and are now closed (ESH-0072/0078/0095/0096/0097), and how to run `scripts/run_ad_oracle.sh`. |

## At a glance

| Operator | Field | Signature | Result on this build |
|----------|-------|-----------|----------------------|
| `derivative` | ℝ→ℝ | `(derivative f x)` | `(derivative (lambda (x) (* x x)) 3.0)` → `6` |
| `gradient` | ℝⁿ→ℝ | `(gradient f pt)` | `#(6 8)` for x²+y² @(3,4) |
| `jacobian` | ℝⁿ→ℝᵐ | `(jacobian f pt)` | `#((6 0) (0 8))` |
| `hessian` | ℝⁿ→ℝ | `(hessian f pt)` | `#((48))` for x⁴ @2 |
| `laplacian` | ℝⁿ→ℝ | `(laplacian f pt)` | `4` for x²+y² @(3,4) |
| `directional-derivative` | ℝⁿ→ℝ | `(directional-derivative f pt dir)` | `6` |
| `divergence` | ℝⁿ→ℝⁿ | `(divergence f pt)` | `14` |
| `curl` | ℝ³→ℝ³ | `(curl f pt)` | `#(0 0 0)` |
| `diff` | AST→AST | `(diff f 'x)` | symbolic, compile-time |

## Status summary

- Oracle gate **PASS** — 60 PASS / 0 XKNOWN / 0 FAIL/CRASH/HANG (JIT + AOT).
- **New in v1.3:** mixed reverse-over-forward AD (outer vector `gradient` over
  inner `derivative`, ESH-0093 / #113) — 15/15 in the mixed-mode test.
- **Closed since v1.3.3:** local-scalar/param captures under reverse mode
  (ESH-0072/0097), second-order operators on `tensor`/`#(…)` points
  (ESH-0095), vector-param gradient-of-gradient (ESH-0096), named
  inner-function gradient (ESH-0078). No cell is XKNOWN on this build. See
  [support-matrix.md](support-matrix.md).
- **Exact at an exact point:** `derivative`, `gradient` and `hessian` return an
  exact integer or rational when the point is exact and the body is pure tower
  arithmetic — `(derivative (lambda (x) (* x x)) 1/3)` → `2/3`.

## See also

- [../../guide/AUTOMATIC_DIFFERENTIATION.md](../../guide/AUTOMATIC_DIFFERENTIATION.md) — the example-driven AD user guide (v1.3 Taylor-tower matrix: arbitrary order, exact, validated, tensor, sparse, checkpointed, control flow, tower numerics)
- [../tensors/INDEX.md](../tensors/INDEX.md) — tensors, ML ops, and the modules AD flows through
- [../../breakdown/AUTODIFF.md](../../breakdown/AUTODIFF.md) — AD internals breakdown
