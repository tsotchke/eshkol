# `ml.activations` — activation functions and normalization

**Source**: [`lib/ml/activations.esk`](../../../lib/ml/activations.esk)
**Require**: `(require ml.activations)` — **must be required individually**. It is **NOT** auto-loaded by `(require stdlib)` (only `ml.optimization` is). (The header comment in the source says `(require ml)`, but the actual require name derived from the path `lib/ml/activations.esk` is `ml.activations`.)

Two families: **scalar** activations that operate on a single number (`relu-scalar`, `sigmoid-scalar`, `tanh-scalar`, `softplus-scalar`), and **tensor** activations/normalizers that operate on a tensor written `#(...)` (`silu`, `swish`, `mish`, `normalize-minmax`, `normalize-zscore`).

## Scalar activations

### `(relu-scalar x)`
`max(x, 0)`.

```scheme
(require ml.activations)
(display (relu-scalar -2.0)) (newline)
(display (relu-scalar 3.0)) (newline)
```
```
0
3
```

### `(sigmoid-scalar x)`
`1 / (1 + e^{−x})`.

```scheme
(display (sigmoid-scalar 0.0)) (newline)
```
```
0.5
```

### `(tanh-scalar x)`
Hyperbolic tangent, computed as `(e^{2x} − 1)/(e^{2x} + 1)`.

```scheme
(display (tanh-scalar 1.0)) (newline)
```
```
0.761594
```

### `(softplus-scalar x)`
`log(1 + e^x)`, with a numerically stable pass-through (`x` returned directly for `x > 20`).

```scheme
(display (softplus-scalar 0.0)) (newline)
```
```
0.693147
```

## Tensor activations

These take a tensor `#(...)` and return a tensor, using SIMD tensor ops internally.

### `(silu tensor)`
Sigmoid Linear Unit, `x · sigmoid(x)` (elementwise).

```scheme
(display (silu #(1.0 2.0 3.0))) (newline)
```
```
#(0.731059 1.76159 2.85772)
```

### `(swish tensor beta)`
Swish, `x · sigmoid(beta·x)` (elementwise). With `beta = 1` it equals `silu`.

```scheme
(display (swish #(1.0 2.0 3.0) 1.0)) (newline)
```
```
#(0.731059 1.76159 2.85772)
```

### `(mish tensor)`
Intended: Mish, `x · tanh(softplus(x))`. **Currently broken** — see [Known issues](#known-issues).

## Normalization

### `(normalize-minmax tensor)`
Min-max normalization to `[0, 1]`; if all elements are equal the tensor is returned unchanged.

```scheme
(display (normalize-minmax #(1.0 2.0 3.0 4.0))) (newline)
```
```
#(0 0.333333 0.666667 1)
```

### `(normalize-zscore tensor)`
Z-score standardization (subtract mean, divide by std); if std is 0 only the mean is subtracted.

```scheme
(display (normalize-zscore #(1.0 2.0 3.0 4.0))) (newline)
```
```
#(-1.34164 -0.447214 0.447214 1.34164)
```

## Internal helpers (not in `provide`)

`tensor-from-scalar`, `clip`, and `dropout` are defined in the module but are not exported.

## Known issues

### `mish` fails — `tensor-apply` rejects user-defined functions

`mish` is implemented with `(tensor-apply tensor softplus-scalar)`, but the `tensor-apply` builtin only accepts a **named builtin** function (e.g. `sin`, `cos`, `+`), not a user/library-defined Scheme procedure like `softplus-scalar`. Calling `mish` errors; because the failure surfaces at module compile time, the error banner prints even when `mish` is never called.

```scheme
;; repro.esk
(require ml.activations)
(display (mish #(1.0 2.0 3.0))) (newline)
```
```
ERROR: tensor-apply: function argument must be a named function (e.g., sin, cos, +)
Unhandled exception: Type error in tensor-apply: expected tensor, got integer
```
No workaround within this module — compute `x · tanh(softplus(x))` manually per element instead.

### Tensor activations require tensor input

Like the rest of the tensor stack, `silu`/`swish`/`normalize-*` assume a `HEAP_SUBTYPE_TENSOR` operand; passing a heterogeneous vector or non-tensor can misread memory (tracked as ESH-0069). Use `#(...)` literals.
