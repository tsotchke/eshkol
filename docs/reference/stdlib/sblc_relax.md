---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/sblc/relax.esk
  - tests/stdlib/sblc_relax_test.esk
  - scripts/run_sblc_relax_gate.sh
  - proofs/lean/EshkolSBLCRelaxation.lean
  - docs/design/adr/0017-stochastic-binary-lambda-calculus.md
---

# `core.sblc.relax` — Differentiable Proposals for SBLC

**Source**: [`lib/core/sblc/relax.esk`](../../../lib/core/sblc/relax.esk)
**Require**: `(require core.sblc.relax)` (must be required individually; depends on `core.blc`, `core.sblc`, and `core.ml.gradient_estimators`).
**Design record**: [ADR-0017](../../design/adr/0017-stochastic-binary-lambda-calculus.md) (item 5).
**Formal proofs**: [`proofs/lean/EshkolSBLCRelaxation.lean`](../../../proofs/lean/EshkolSBLCRelaxation.lean) (Lean 4.12.0, 0 `sorry`, 0 custom axioms).

`core.sblc.relax` layers differentiable proposal distributions over the exact
`core.sblc` verifier without altering exact prefix-code priors (`2^-|M|`), fair
binary choice semantics (`M ⊕ N`), bounded completeness, or verifier outcomes.

## Strict Receipt Segregation

Every entry point returns a tagged `sblc-relax-receipt-v1` record with explicit
method, exactness, status, gradient, variance diagnostics, projection residual,
and exact-verified candidates:

| Method symbol | `sblc-relax-receipt-exact?` | Contract |
|---|---|---|
| `'exact-oracle` | `#t` | Exact softmax expectation and analytical gradient over an enumerated or explicit closed-program domain; verified against native `(gradient ...)` AD. Refuses non-enumerated grammar domains loudly (`'domain-error`) with zero implicit fallback. |
| `'reinforce-rb` | `#f` | Seeded outer-program REINFORCE (`make-prng`) with exact rational Rao–Blackwellized inner coins via `sblc-likelihood` / `sblc-distribution` and leave-one-out / mean / zero baseline. Reports `baseline-variance`, `post-rb-variance`, `variance-reduction-ratio`, `variance-bound`, and `confidence` (`'exact-zero-variance`, `'high`, or `'low-confidence`). |
| `'gumbel-softmax` | `#f` | Temperature-annealed (`tau > 0`) Concrete / Gumbel-Softmax relaxation over depth-masked `sblc-v1` grammar logits with native AD straight-through surrogate gradients, exact L1 `projection-residual`, strict closed-term projection (`sblc-relax-project-strict`), and loud `'projection-error` refusal on closedness or caller hard-constraint failure (never silent clipping). |

## API Surface

### Domains & Unified Entry Point
- `(sblc-relax-enumerated-domain max-length examples step-bound)` — closed SBLC programs up to `max-length <= 16`.
- `(sblc-relax-program-domain programs examples step-bound)` — explicit non-empty closed-program catalogue.
- `(sblc-relax-grammar-domain max-depth examples step-bound)` — non-enumerated recursive grammar domain.
- `(sblc-relax-sample domain method logits opts)` — unified proposal entry point enforcing strict exact-vs-approximate segregation.

### Proposal Estimators, Projection & Optimization
- `(sblc-relax-exact-oracle domain logits opts)` / `(sblc-relax-exact-expectation ...)` / `(sblc-relax-exact-gradient ...)` / `(sblc-relax-exact-ad-gradient ...)`
- `(sblc-relax-reinforce-rb domain logits opts)`
- `(sblc-relax-gumbel-grammar domain logits opts)` / `(sblc-relax-grammar-step logits env depth tau prng)` / `(sblc-relax-project-strict term constraint residual)`
- `(sblc-relax-verify-invariance domain logits1 logits2)` — verifies byte-identical exact `core.sblc` weights, likelihoods, and conditioning receipts across proposal reweightings and catalogue orderings.
- `(sblc-relax-optimize-logits domain method init-logits lr steps opts)` — seeded gradient-ascent loop across `'exact-oracle`, `'reinforce-rb`, or `'gumbel-softmax`.

### Receipt & Candidate Accessors
- `sblc-relax-receipt?`, `sblc-relax-receipt-version`, `sblc-relax-receipt-method`, `sblc-relax-receipt-exact?`, `sblc-relax-receipt-status`, `sblc-relax-receipt-expected-reward`, `sblc-relax-receipt-gradient`, `sblc-relax-receipt-variance-stats`, `sblc-relax-receipt-baseline-variance`, `sblc-relax-receipt-post-rb-variance`, `sblc-relax-receipt-variance-reduction-ratio`, `sblc-relax-receipt-variance-bound`, `sblc-relax-receipt-confidence`, `sblc-relax-receipt-sample-count`, `sblc-relax-receipt-projection-residual`, `sblc-relax-receipt-error-detail`, `sblc-relax-receipt-candidates`, `sblc-relax-receipt-best`, `sblc-relax-receipt->string`
- `sblc-relax-candidate-term`, `sblc-relax-candidate-bits`, `sblc-relax-candidate-prior`, `sblc-relax-candidate-likelihood`, `sblc-relax-candidate-weight`, `sblc-relax-candidate-probability`, `sblc-relax-candidate-verified?`

## Verification Gate

Run `scripts/run_sblc_relax_gate.sh` (CTest target `sblc_relax_gate`). It executes
all 126 assertions in `tests/stdlib/sblc_relax_test.esk` under both JIT
(`eshkol-run -r`) and AOT (`eshkol-run -o`) and requires byte-identical stdout.
