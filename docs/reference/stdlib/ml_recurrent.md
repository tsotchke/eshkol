# `core.ml.recurrent` — LSTM and GRU cells

**Source**: [`lib/core/ml/recurrent.esk`](../../../lib/core/ml/recurrent.esk)
**Require**: `(require core.ml.recurrent)` — not loaded by `(require stdlib)`.
**Design**: [ADR-0049](../../design/adr/0049-neuro-symbolic-bridge.md)

The two standard gated recurrent cells over a flat parameter vector `theta`.
Every function is written over generic arithmetic, so the built-in AD
operators differentiate a whole unrolled sequence exactly:
`(gradient loss theta)` is backpropagation through time with no hand-written
backward pass, and `derivative` along any coordinate gives the same numbers in
forward mode.

## Equations

LSTM (Hochreiter & Schmidhuber 1997, with a forget gate), `[x; h]` the input
concatenated with the previous hidden state:

```
i = sigma(W_i [x; h] + b_i)      f = sigma(W_f [x; h] + b_f)
g = tanh (W_g [x; h] + b_g)      o = sigma(W_o [x; h] + b_o)
c' = f * c + i * g               h' = o * tanh(c')
```

GRU (Cho et al. 2014, the convention of PyTorch's `nn.GRU`):

```
r = sigma(W_ir x + W_hr h + b_r)
z = sigma(W_iz x + W_hz h + b_z)
n = tanh (W_in x + b_in + r * (W_hn h + b_hn))
h' = (1 - z) * n + z * h
```

## Parameter layout

Blocks start at an offset `off` into `theta`, so several layers share one
vector. Rows are stored row-major.

| Block | Layout | Size |
|---|---|---|
| LSTM | gates `i, f, g, o`; each `H` rows of `[W_x (I) \| W_h (H) \| b]` | `4H(I+H+1)` |
| GRU | gates `r, z`; each `H` rows of `[W_x \| W_h \| b]`; then `n` as `H` rows of `[W_in (I) \| b_in \| W_hn (H) \| b_hn]` | `3H(I+H+1) + H` |
| readout | `O` rows of `[W (H) \| b]`, `y = W h + b` | `O(H+1)` |

## Functions

### `(lstm-param-count input hidden)`, `(gru-param-count input hidden)`, `(readout-param-count hidden out)`
Block sizes from the table above.

### `(make-params n)`
A zero vector of length `n`.

### `(lstm-init-params! theta off input hidden seed)`, `(gru-init-params! theta off input hidden seed)`, `(readout-init-params! theta off hidden out seed)`
Fill a block with deterministic uniform values of Glorot scale `1/sqrt(fan-in)` from an LCG `seed`. The LSTM forget-gate biases are set to 1 (Jozefowicz et al. 2015). Return `theta`.

### `(lstm-cell theta off input hidden x h c)`
One LSTM step; returns `(h' . c')`, both vectors of length `hidden`.

### `(gru-cell theta off input hidden x h)`
One GRU step; returns `h'`.

### `(lstm-run theta off input hidden xs)`, `(gru-run theta off input hidden xs)`
Unroll over a list of input vectors from the zero state. `lstm-run` returns `(hs . c-final)` and `gru-run` returns `hs`, with `hs` the list of hidden states in time order.

### `(readout theta off hidden out h)`
The dense layer `y = W h + b`, a vector of length `out`.

### `(rnn-sigmoid z)`
`1 / (1 + exp(-z))`.

```scheme
;; rnn-basic.esk
(require core.ml.recurrent)
(define I 2) (define H 3)
(define theta (make-params (+ (lstm-param-count I H) (readout-param-count H 1))))
(lstm-init-params! theta 0 I H 11)
(readout-init-params! theta (lstm-param-count I H) H 1 12)
(display (vector-length theta)) (newline)
(define hc (lstm-cell theta 0 I H (vector 0.5 -1.0) (make-vector H 0.0) (make-vector H 0.0)))
(display (car hc)) (newline)
(define seq (list (vector 0.5 -1.0) (vector 0.2 0.3) (vector -0.7 0.1)))
(define (loss th)
  (let* ((hs (car (lstm-run th 0 I H seq)))
         (y (vector-ref (readout th (lstm-param-count I H) H 1 (caddr hs)) 0)))
    (* y y)))
(display (loss theta)) (newline)
(define g (gradient loss theta))
(display (vector-ref g 0)) (display " ") (display (derivative (lambda (s) (let ((t2 (vector-map (lambda (x) x) theta))) (vector-set! t2 0 (+ (vector-ref theta 0) s)) (loss t2))) 0.0)) (newline)
(define gt (make-params (gru-param-count 1 2)))
(gru-init-params! gt 0 1 2 5)
(display (gru-run gt 0 1 2 (list (vector 1.0) (vector 0.0)))) (newline)
```
```
76
#(-0.06120933019521913 0.13322488330777119 0.005358489828566206)
0.13199686990779308
0.002039899769473234 0.0020398997694732334
(#(-0.1395756706564083 -0.01807066785165838) #(-0.33417036994253796 -0.014788338160847132))
```

The fourth line is one coordinate of the BPTT gradient computed twice: by
reverse mode over the whole parameter vector and by forward mode along that
coordinate.

## Verification
[`tests/ml/recurrent_test.esk`](../../../tests/ml/recurrent_test.esk) checks
one-step derivatives of both cells against their closed forms (1e-14),
reverse mode against forward mode through four-step sequences (1e-12), an
LSTM that learns to recall the first input of a sequence, and a GRU that
learns to count the ones in a binary sequence and answers a held-out
sequence correctly.
