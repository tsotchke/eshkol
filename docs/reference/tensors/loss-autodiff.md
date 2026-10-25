# Native loss differentiation

`cross-entropy-loss` supports first-order native reverse differentiation with
respect to its **logits**. It accepts either normalized probability rows with
the same shape as the logits or constant class-index labels with the final
class dimension removed. The loss is averaged over rows. For logits `x`,
probability targets `y`, and `R` rows, the logit derivative is
`(softmax(x) - y) / R`; indexed labels use their corresponding one-hot targets.

```scheme
(gradient (lambda (logits)
            (cross-entropy-loss logits #(1.0 0.0)))
          #(0.0 0.0))
;; => #(-0.5 0.5)
```

The loss records a tensor tape node, so its derivatives propagate through
supported operations that produce the logits, including matrix multiplication,
GELU, and broadcast addition. Scalar arithmetic after the loss also propagates
its upstream derivative. The same rule applies to direct, captured, and
indirect calls. It uses analytic reverse differentiation, not numerical
finite differences.

Targets are treated as constant labels. An AD-carried target is rejected with
`cross-entropy-loss: differentiation of targets is not supported; supply constant labels`.
This includes class indices, which are discrete, and differentiable probability
labels, whose normalization constraint needs its own API contract. It can also
reject a label constructed through tape operations even when that computation
would have a zero derivative; supplying the constant label tensor avoids this
unsupported path. This contract does not add higher-order or forward tensor
rules, or imply equivalent coverage in the bytecode VM.

The remaining native loss builtins currently implement numeric evaluation but
do not record a reverse rule: `mse-loss`, `mae-loss`, `bce-loss`,
`binary-cross-entropy-loss`, `huber-loss`, `kl-div-loss`, `hinge-loss`,
`smooth-l1-loss`, `focal-loss`, `triplet-loss`, `contrastive-loss`,
`label-smoothing-loss`, and `cosine-embedding-loss`. Calling one while a reverse
tape is active now raises a named "reverse-mode differentiation is not
implemented" error instead of returning a silently disconnected value. The
check uses runtime tape state and therefore also covers indirect calls. Normal
numeric calls outside that state retain their existing behavior.

To differentiate such a loss, express it using operations with derivative
rules. For example, equal-shaped prediction and target tensors can use
`(tensor-mean (tensor-mul (tensor-sub prediction target)
                         (tensor-sub prediction target)))`
for mean squared error. This does not change separately implemented library
loss functions or manually supplied gradients.

The native regression harness is `tests/ad/cross_entropy_gradient_test.py`.
It checks JIT and AOT execution under both tensor tape lowerings, analytical
logit derivatives, finite-difference slopes for all weights of a two-layer
network, existing numeric loss cases, and explicit refusals. The finite
differences are test oracles only. The C++ backward differential test separately
checks that a tensor cotangent is honored when the scalar mirror is zero.
