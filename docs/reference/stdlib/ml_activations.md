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
0.7615941559557649
```

### `(softplus-scalar x)`
`log(1 + e^x)`, with a numerically stable pass-through (`x` returned directly for `x > 20`).

```scheme
(display (softplus-scalar 0.0)) (newline)
```
```
0.6931471805599453
```

## Tensor activations

These take a tensor `#(...)` and return a tensor, using SIMD tensor ops internally.

### `(silu tensor)`
Sigmoid Linear Unit, `x · sigmoid(x)` (elementwise).

```scheme
(display (silu #(1.0 2.0 3.0))) (newline)
```
```
#(0.7310585786300049 1.7615941559557646 2.8577223804673)
```

### `(swish tensor beta)`
Swish, `x · sigmoid(beta·x)` (elementwise). With `beta = 1` it equals `silu`.

```scheme
(display (swish #(1.0 2.0 3.0) 1.0)) (newline)
```
```
#(0.7310585786300049 1.7615941559557646 2.8577223804673)
```

### `(mish tensor)`
Mish, `x · tanh(softplus(x))` (elementwise).

```scheme
(display (mish #(1.0 2.0 3.0))) (newline)
```
```
#(0.8650983882673103 1.9439589595339946 2.9865350049679575)
```

## Normalization

### `(normalize-minmax tensor)`
Min-max normalization to `[0, 1]`; if all elements are equal the tensor is returned unchanged.

```scheme
(display (normalize-minmax #(1.0 2.0 3.0 4.0))) (newline)
```
```
#(0 0.3333333333333333 0.6666666666666666 1)
```

### `(normalize-zscore tensor)`
Z-score standardization (subtract mean, divide by std); if std is 0 only the mean is subtracted.

```scheme
(display (normalize-zscore #(1.0 2.0 3.0 4.0))) (newline)
```
```
#(-1.3416407864998738 -0.4472135954999579 0.4472135954999579 1.3416407864998738)
```

> **Engine note.** `normalize-zscore` is native-only: it reaches `tensor-std`,
> which the bytecode VM does not provide, so the VM reports
> `undefined variable 'tensor-std'` and the call raises there. Every other
> procedure on this page runs on both engines.

## Internal helpers (not in `provide`)

`tensor-from-scalar`, `clip`, and `dropout` are defined in the module but are not exported.

## Known issues

### Closed — `mish` and `tensor-apply` with a library procedure

Through v1.3.4, `mish` — implemented as `(tensor-apply tensor softplus-scalar)`
— could not run: `tensor-apply` resolved its second operand through a
function-**name** table that held only builtins such as `sin`, `cos` and `+`,
so a user- or library-defined Scheme procedure was refused, and because the
failure surfaced at module compile time the banner printed even when `mish` was
never called.

`(tensor-apply tensor callable)` now evaluates both operands once and invokes
the **resolved callable** on each scalar in row-major order. No function-name
table and no identity substitution participates, so a lexical binding or a user
definition that shadows a historical builtin name determines what runs, and a
non-callable raises even for an empty tensor. `mish` works on both engines —
see [its entry above](#mish-tensor).

### Tensor activations require tensor input

Like the rest of the tensor stack, `silu`/`swish`/`normalize-*` assume a `HEAP_SUBTYPE_TENSOR` operand; passing a heterogeneous vector or non-tensor can misread memory (tracked as ESH-0069). Use `#(...)` literals.
