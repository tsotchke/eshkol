# ADR-0032: Exact-forward control fine-tuning of the constructed interpreter

- **Status:** Accepted — implemented for v1.4.0-connection
- **Date:** 2026-10-23
- **Decision owners:** SDNC and AD maintainers
- **Applies to:** `lib/core/sdnc_core.h`, `lib/core/sdnc_finetune.{h,c}`,
  `inc/eshkol/core/saturating_vjp.h`, `lib/backend/qllm_backward.c`,
  `lib/core/sdnc_api.c`, `lib/core/sdnc.esk`, the `--control-finetune` mode of
  `lib/backend/weight_matrices.c`
- **Relates to:** ADR-0005 (resident programs-as-weights: `theta_sem`,
  `psi_program`, `phi_plastic`), [`docs/SDNC.md`](../../SDNC.md) §14

## Context

The SDNC interpreter is a six-layer transformer whose weights are constructed
analytically, not trained (`generate_weights` in
`lib/backend/weight_matrices.c`). Every gate pre-activation and every Layer-0
attention logit is an integer multiple of the construction unit `SCALE = 300`.
That margin is what makes the forward pass exact: `sdnc_sigmoidf` returns
literal `0.0f` or `1.0f` for |x| > 20, and the attention softmax underflows to
a one-hot vector (`docs/SDNC.md` §9). The interpreter reproduces the reference
VM on every byte of every step of the verifier battery.

The same margin makes the exact derivative of every **control** weight
identically zero: the gate thresholds (`ff_gate_b`), gate columns (`ff_gate`),
`W_Q`, `W_K` and `b_Q`. A saturated gate has `s (1 - s) = 0`; a one-hot softmax
has `a_p (da_p - <a, da>) = 0` for every `p`. The value weights (`ff_up`,
`ff_down`, `W_V`, `W_O`) keep exact, sparse gradients. So gradient descent on
the constructed weights cannot change which gate fires, which is the part of
the machine that dispatches opcodes.

Measured alternatives, all rejected:

1. **Lowering the forward scale.** The interpreter stays exact down to a scale
   of about 60, is off by 0.3% of transitions at 20 and does not run at 10.
   Any scale low enough to give useful gradients gives up exactness.
2. **Annealing / continuation** on a softened forward pass, and **untied
   per-neuron SGD** on thresholds. Neurons come in `±M` pairs that share one
   gate; their per-neuron gradients cancel, and an SGD step that is not a
   whole construction unit leaves gates half-open.
3. **A squared loss.** Address and pointer slots carry errors of hundreds of
   units; they dominate the squared loss and hide the dispatch errors.
4. **A line search along the full gradient direction.** It moves many gates by
   fractions of a unit at once and is accepted by no exact criterion.

## Decision

Keep the forward pass exact and change only what the backward pass reports
about the saturating nonlinearities, then let the exact forward pass decide.

**1. One reverse-mode rule with an explicit inverse temperature.**
`inc/eshkol/core/saturating_vjp.h` defines the vector-Jacobian products of the
logistic gate and the softmax once, generic over the scalar type:

```text
s = sigma(beta * pre)       d pre = upstream * beta * s * (1 - s)
a = softmax(beta * z)       d z_p = beta * a_p * (da_p - <a, da>)
```

`lib/core/sdnc_core.h` (float) and `lib/backend/qllm_backward.c` (`qllm_real`,
also built in double by its gradient check) both use it; before, each carried
its own copy. At `beta = 1` the rule is the exact derivative, evaluated in the
original operation order.

**2. Backward smoothing is an argument, not state.**
`sdnc_backward_smoothed(..., SdncBackwardSmoothing sm, ...)` takes
`{gate_beta, attn_beta}`. The gate value and attention weights used by the rule
are re-evaluated at `beta`; every other chain-rule factor (gate values in the
up branch, attention weights in the value branch, up projections) is the exact
forward value. This is the Jacobian of the logistic-perturbed gate: the
expectation of a hard threshold under logistic noise of scale `1/beta` is
`sigma(beta * pre)`. `sdnc_backward` is defined as the `beta = 1` case.
`sdnc_forward_cached` is the forward half that both use; `sdnc_forward` and
`sdnc_forward_tail` share the layer code, so a caller can resume a forward pass
at any layer.

**3. Parameterize thresholds per construction gate.**
`sdnc_gate_groups_build` ties every neuron whose gate column and bias are
identical (value equality, so `-0` and `+0` tie). A tied gate moves as one
threshold. The constructed interpreter has 1102 tied gates over its gated
layers.

**4. Rank with the smoothed backward, accept with the exact forward.**
`sdnc_control_finetune` repeats, first over tied gates and then over single
neurons:

- accumulate the smoothed gradient of the Huber objective (delta 1) over the
  transitions that differ from the record;
- rank tied gates (neurons) by the magnitude of their summed threshold
  gradient;
- for each of the top `K` candidates, move its threshold by one construction
  unit against the gradient sign and evaluate the exact objective;
- accept the candidate with the fewest transitions that are not byte-identical
  to the record (ties: lower Huber loss), only if it improves on the current
  weights; stop when no candidate does or nothing differs.

The exact evaluation is incremental and still exact: a threshold move in layer
`L` cannot change anything before FFN `L`, and it cannot change a transition in
which every moved gate is clamped to the same value on both sides of the move
(checked with a margin that covers any rounding of the pre-activation). Only
the remaining transitions re-run the forward tail from `L`, from cached layer
inputs. The final report re-evaluates every transition from scratch.

**5. Teacher data are recorded transitions.** An `SdncTransitionSet` holds
`(state, program embeddings, next state)` triples; consecutive steps of one
program share their embeddings. `weight_matrices --control-finetune` records
every forward transition of the verifier battery through `run_with_weights`,
and writes them as an `SDTR` v1 file next to the weights (`QLMW` v3).

**6. Eshkol surface.** `core.sdnc` adds `(sdnc-load-weights! θ path)`,
`(sdnc-mismatches θ transitions)`, `(sdnc-repair! θ transitions [beta])` and a
smoothing argument to `(sdnc-weight-grad θ input target [smoothing])`, where
`smoothing` is a positive real or a pair `(gate-beta . attention-beta)`. The
three new procedures are `extern` wrappers over C entry points in
`lib/core/sdnc_api.c`, so no new AST operation is introduced; the smoothing
argument is the optional fourth argument of the existing operation.

### Relation to ADR-0005

ADR-0005 freezes `theta_sem` by default and treats its derivative as a
diagnostic. This decision does not make `theta_sem` plastic. A repair is an
explicit operation whose acceptance criterion is exact reproduction of a
recorded transition set, and every move it makes is a whole construction unit,
so the result is again a set of analytic gate thresholds.

## Measured result

`weight_matrices --control-finetune` (ctest `sdnc_control_finetune`),
release build:

| Quantity | Value |
|---|---|
| Verifier battery on the constructed weights | 127 / 127 |
| Recorded transitions / program runs | MEASURED_TRANSITIONS / MEASURED_PROGRAMS |
| ADD-dispatch gate neurons switched off (Layer 3) | MEASURED_DAMAGED |
| Battery on the damaged weights | MEASURED_DAMAGED_BATTERY |
| Transitions that differ after the damage | MEASURED_BEFORE |
| Exact derivative of every control weight | identically zero |
| Backward inverse temperature | `beta = 10 / 300` for gates and attention |
| Tied-gate moves / neuron moves | MEASURED_GATE_MOVES / MEASURED_NEURON_MOVES |
| Transitions that differ after the repair | 0 |
| Battery on the repaired weights | 127 / 127 |
| Recorded transitions byte-identical (`forward_with_weights`) | MEASURED_TRANSITIONS / MEASURED_TRANSITIONS |
| Wall time | MEASURED_TIME |

The same run checks that `sdnc_forward` reproduces every recorded transition
of `forward_with_weights` byte for byte, and that `sdnc_backward` is
bit-identical to `backward_through_weights` on sampled transitions of the
constructed weights and on unsaturated random weights.

## Consequences

- The control weights of the constructed interpreter are trainable against an
  exact criterion. A damaged dispatch is repaired without ever running a
  softened forward pass.
- `sdnc_backward` and `qllm_backward.c` share one logistic/softmax rule; the
  `qllm_backward_gradcheck` test and the beta = 1 bit-identity checks pin it.
- Thresholds that no recorded transition exercises are not constrained by the
  repair; a repaired weight set can differ from the construction there while
  behaving identically on the record.
- Planned for v1.4.1: control moves for gate columns and the attention query
  and key weights (`W_Q`, `W_K`, `b_Q`), which this release ranks but does not
  move; and a repair objective over a full rollout rather than single
  transitions.
