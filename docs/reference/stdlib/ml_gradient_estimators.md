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
softmax #(0.23122389762214907 0.6285317192117624 0.14024438316608848)
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
#(0.17462771023134466 0.7081502847403103 0.09583771935941783 0.021384285668927173)
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

## Native-AD estimators

The functions below are written over generic arithmetic and the
[`stop-gradient`](../ad/operators.md#stop-gradient--a-value-every-enclosing-pass-treats-as-a-constant)
builtin, so they compose with any loss differentiated by `gradient` (reverse
mode) or `derivative` (forward mode), with no tape object. Logits may be a
list or a vector; results are vectors. Design:
[ADR-0049](../../design/adr/0049-neuro-symbolic-bridge.md).

The straight-through identity: for a soft surrogate `s` and a hard value `h`,
`s + stop-gradient(h - s)` has the **value** `h` and the **derivative** of `s`.

| Function | Forward value | Derivative | Bias |
|---|---|---|---|
| `(st-value s h)` | `h` | `ds` | the identity itself |
| `(st-round x)` | `round(x)` | `1` | biased |
| `(st-softmax logits)` | one-hot of `argmax(logits)` | softmax Jacobian | biased |
| `(gumbel-softmax-relaxed logits noise tau)` | `softmax((logits + noise) / tau)` | exact | unbiased for the relaxed objective; biased for the discrete one, bias falling as `tau -> 0` while the variance grows |
| `(gumbel-softmax-st logits noise tau)` | the Gumbel-max one-hot | the relaxed sample's | biased |
| `(st-bernoulli a u)` | `1` when `u < sigmoid(a)`, else `0` (an exact Bernoulli sample) | `sigmoid'(a)` | biased |

`noise` is a vector of Gumbel(0,1) draws, from `sample-gumbel-noise` or given
explicitly for reproducible runs; `u` is a uniform draw.

### Samplers and exact references

- `(gumbel-max-sample logits noise)` — the index of `argmax(logits + noise)`. With Gumbel noise this is an **exact** sample of `Categorical(softmax(logits))`.
- `(categorical-log-prob logits k)` — `log softmax(logits)[k]`, carrier-generic.
- `(score-function-gradient logits k fk)` — the REINFORCE estimate `f(k) (e_k - p)` of `d/dlogits E_{k~Cat(p)}[f(k)]`. **Unbiased**; its variance grows with the spread of `f`.
- `(categorical-expectation-gradient logits fvals)` — the exact gradient `p (f - E_p f)` the estimators are measured against.

```scheme
;; ge-native.esk
(require core.ml.gradient_estimators)
(define logits (vector 1.0 0.0 -1.0 0.5))
(define w (vector 0.3 -1.0 2.0 0.7))
(define noise (vector 0.13 -0.42 0.88 0.05))
(define (dot a b) (let loop ((i 0) (s 0.0)) (if (< i (vector-length a)) (loop (+ i 1) (+ s (* (vector-ref a i) (vector-ref b i)))) s)))
(display (st-softmax logits)) (newline)
(display (gumbel-softmax-relaxed logits noise 0.5)) (newline)
(display (gumbel-softmax-st logits noise 0.5)) (newline)
(display (gradient (lambda (l) (dot (gumbel-softmax-st l noise 0.5) w)) logits)) (newline)
(display (gumbel-max-sample logits noise)) (newline)
(display (st-round 2.7)) (display " ") (display (derivative (lambda (x) (* 5.0 (st-round x))) 2.7)) (newline)
(display (categorical-expectation-gradient logits (vector 1.0 3.0 -2.0 0.5))) (newline)
```
```
GE_NATIVE_OUTPUT
```

The properties in the table are checked on known distributions by
[`tests/ml/gradient_estimators_native_test.esk`](../../../tests/ml/gradient_estimators_native_test.esk):
Gumbel-max frequencies match `softmax(logits)` within 4.5 standard deviations
over 20000 draws; the score-function mean matches the exact gradient within
4.5 standard errors; for the relaxed estimator, the bias against the discrete
gradient is smaller at `tau = 0.25` than at `tau = 2` and the variance is
larger; the straight-through Bernoulli frequency matches `sigmoid(a)`; and a
categorical choice trained through `gumbel-softmax-st` samples its target
class at least 95% of the time.

## Edge cases
- `softmax` of an empty vector returns a length-0 vector (no error).
- These estimators are **biased by design** — that is the point; the gumbel-softmax gradient matches finite differences of the *soft* loss (rel err < 1e-2, per the acceptance test), not of the hard discrete op.
- `gumbel-softmax` / `sample-gumbel-noise` are nondeterministic; use the `-det` variants with explicit noise for reproducible output.

## Verification note
`tests/ml/gradient_estimators_test.esk` passes 16/16 under `eshkol-run -r`: gumbel-softmax is a valid distribution, its gradient matches central finite differences (rel<1e-2), straight-through forward equals the hard value, and both ST and gumbel-softmax training loops drive the argmax to the target class (`p[target] > 0.8` end-to-end). No `.swarm` ledger issues reference these functions.
