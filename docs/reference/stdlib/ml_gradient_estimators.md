# `core.ml.gradient_estimators` — gradient estimators for discrete operations

**Source**: [`lib/core/ml/gradient_estimators.esk`](../../../lib/core/ml/gradient_estimators.esk)
**Require**: `(require core.ml.gradient_estimators)` — **NOT** auto-loaded via `(require stdlib)`. This is a pure-Scheme module; using its functions without the require fails with *"called undefined function"*. It is built on the stateful tape, so also `(require core.ad.tape)`. **Name-collision caveat:** `softmax` here shadows a native tensor `softmax` builtin — without the require, `(softmax v)` resolves to the *builtin* tensor op (same result for a plain vector); with the require you get this module's numerically-stable Scheme `softmax`.

Discrete operations (argmax, categorical sampling, rounding) have zero derivative almost everywhere, so you cannot train through them directly. This module provides **biased-but-useful estimators** that substitute a surrogate backward pass, all recorded on the [`core.ad.tape`](ad_tape.md) via `record-op!` so a loss can route through them alongside builtin tape ops:

- **Gumbel-Softmax (Concrete)** — replace a hard categorical sample with a temperature-controlled soft one-hot `softmax((logits+g)/tau)`. As `tau → 0` it approaches a one-hot; gradients flow to the logits.
- **Straight-Through** — forward emits the *hard* value (one-hot / round), backward pretends the op was the identity so the upstream gradient passes straight through.

## Functions

### `(softmax v)`
Numerically-stable softmax of a list **or** vector of numbers; returns a fresh vector summing to 1. Empty input returns a length-0 vector.

### `(categorical-pick probs)`
The **hard** decision: return the integer index of the largest entry (argmax). Accepts a list or vector.

### `(argmax-onehot probs)`
Return a hard one-hot vector with `1.0` at the argmax position (length = input length).

```scheme
;; ge-basic.esk
(require core.ad.tape)
(require core.ml.gradient_estimators)
(display "softmax ")(display (softmax (vector 1.0 2.0 0.5)))(newline)
(display "pick ")(display (categorical-pick (vector 0.1 0.9 0.3)))(newline)
(display "onehot ")(display (argmax-onehot (vector 0.1 0.9 0.3)))(newline)
```
```
softmax #(0.231224 0.628532 0.140244)
pick 1
onehot #(0 1 0)
```

### `(sample-gumbel-noise n)`
Draw a length-`n` vector of Gumbel noise `g = -log(-log(u))`, `u ~ Uniform(0,1)`. **Nondeterministic** (uses `(random)`); tests pass explicit noise for reproducibility.

### `(gumbel-softmax-det t logits-node tau noise)`
**Deterministic** Gumbel-Softmax recorded as one custom op on tape `t`. `logits-node` is a tape node whose value is a logit vector, `tau > 0` the temperature, `noise` an explicit Gumbel-noise vector. Forward output is the soft one-hot `softmax((logits+noise)/tau)`; the backward is the softmax Jacobian-vector product scaled by `1/tau`, so gradients flow to `logits-node`. Returns the output node.

### `(gumbel-softmax t logits-node tau)`
Stochastic variant of the above — draws fresh Gumbel noise internally (matching the logits length), then delegates to `gumbel-softmax-det`.

```scheme
;; ge-gumbel.esk — output is a valid distribution; grad flows to logits
(require core.ad.tape)
(require core.ml.gradient_estimators)
(define noise (vector 0.10 -0.20 0.30 0.05))
(define p
  (with-tape "g"
    (lambda ()
      (let* ((t  (current-tape))
             (lg (tape-input t (vector 1.0 2.0 0.5 0.0)))
             (gs (gumbel-softmax-det t lg 0.5 noise)))
        (node-value gs)))))
(display p)(newline)
(display "sum=")
(display (let ((n (vector-length p)))
  (let loop ((i 0)(s 0.0)) (if (< i n) (loop (+ i 1) (+ s (vector-ref p i))) s))))(newline)
```
```
#(0.174628 0.70815 0.0958377 0.0213843)
sum=1
```

### `(straight-through t est-node hard-value)`
Generic straight-through op. Forward emits `hard-value` (precomputed, same shape as the node — scalar or vector); backward is the identity, routing the gradient straight to `est-node`. Returns the output node.

### `(straight-through-round t x-node)`
Straight-through on scalar rounding: forward `= (round (node-value x-node))`, backward `= identity`.

### `(straight-through-onehot t probs-node)`
Straight-through on categorical argmax: forward `= argmax-onehot` of the (soft) probability node's value, backward routes the gradient back to the soft probs unchanged. This is how you train through a discrete sample while keeping a usable gradient.

```scheme
;; ge-st.esk — forward is the hard value; backward passes gradient through
(require core.ad.tape)
(require core.ml.gradient_estimators)
(define oh
  (with-tape "st"
    (lambda ()
      (let* ((t (current-tape))
             (x (tape-input t (vector 0.1 0.7 0.2))))
        (node-value (straight-through-onehot t x))))))
(display "onehot ")(display oh)(newline)
(define r
  (with-tape "str"
    (lambda ()
      (let* ((t (current-tape)) (x (tape-input t 2.3)))
        (node-value (straight-through-round t x))))))
(display "round(2.3) ")(display r)(newline)
```
```
onehot #(0 1 0)
round(2.3) 2
```

## Edge cases
- `softmax` of an empty vector returns a length-0 vector (no error).
- These estimators are **biased by design** — that is the point; the gumbel-softmax gradient matches finite differences of the *soft* loss (rel err < 1e-2, per the acceptance test), not of the hard discrete op.
- `gumbel-softmax` / `sample-gumbel-noise` are nondeterministic; use the `-det` variants with explicit noise for reproducible output.

## Verification note
`tests/ml/gradient_estimators_test.esk` passes 16/16 under `eshkol-run -r`: gumbel-softmax is a valid distribution, its gradient matches central finite differences (rel<1e-2), straight-through forward equals the hard value, and both ST and gumbel-softmax training loops drive the argmax to the target class (`p[target] > 0.8` end-to-end). No `.swarm` ledger issues reference these functions.
