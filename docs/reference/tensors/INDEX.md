# Tensors & ML — Reference

Machine-verified reference for Eshkol's tensor system and the ML/numerical
library modules (v1.3.5-evolve). Every signature and output on these pages was
produced by running the current compiler; broken operations are documented
against their behavior, not hidden.

## Pages

| Page | Contents |
|------|----------|
| [creation.md](creation.md) | `vector` vs `tensor` (heterogeneous 16-byte tagged values vs homogeneous 8-byte doubles), literals `#(…)`, `make-tensor`/`make-vector`, and dtypes (f16/bf16/f32/f64/i8). |
| [operations.md](operations.md) | Shape ops, elementwise & unary math, linear algebra + decompositions, reductions, conv1d/2d/3d, pooling, normalization, attention, embedding, activations, the PR-#79 type guards, pixel fills, and save/load — with known-broken ops flagged. |
| [eskm-v1.md](eskm-v1.md) | Normative ESKM v1 checkpoint wire format, decoder requirements, historical provenance, and executable compatibility fixtures. |
| [gpu.md](gpu.md) | Honest GPU-dispatch status: the `gpu-*` builtins, the cost-model threshold, what actually runs on Metal in `-r` vs AOT, and how that squares with ESH-0022/0023. |
| [ml-modules.md](ml-modules.md) | `ml.optimization`, `core.manifold`, `signal.fft` — every `provide` with signature and a run example. |
| [eskt-engine-parity.md](eskt-engine-parity.md) | Public ESKM tensor-file 4 × 4 producer/consumer test, exact byte/CRC oracle, and scope (historical page filename). |

## Two containers at a glance

| | `vector` / `make-vector` | `tensor` / `#(…)` / `make-tensor` |
|---|---|---|
| Storage | heterogeneous 16-byte tagged values | homogeneous 8-byte doubles |
| Mixed types | yes | no (numeric only) |
| dtype / shape | no | yes (`f64` default) |
| Displays as | `#(…)` | `#(…)` (same — distinguish via `tensor-dtype`) |

## Binary element-wise arithmetic takes two operands of matching shape

`+ - * /` over a vector or tensor is a **binary** contract: both operands are
classified before either is dereferenced, so the Scheme-vector kernel runs only
when both sides are Scheme vectors of equal length and every other combination
goes to the tensor path, where the shared operand check validates each side
independently. Scalar broadcast is a **separate operator**, `tensor-scale`.

- A scalar in either position raises, at the call site's own location:
  `Type error in tensor-mul: expected tensor, got integer`.
- A pair the broadcast computation refuses raises
  `Shape mismatch in tensor-mul: shapes (3) and (2) are not broadcast-compatible`.
- A length-1 operand still broadcasts: `(* #(2.0) #(1.0 2.0 3.0))` is `#(2 4 6)`.
- A vector and a rank-1 tensor are two spellings of one value, so a mixed pair
  is the element-wise result: `(* (vector 1.0 2.0) (tensor 3.0 4.0))` is
  `#(3 8)`.

Every one of those was previously a silent wrong answer or an uncatchable fatal
signal (ledger LE-18, LE-19, LE-22). Element-wise arithmetic over
vectors/tensors is a **native-engine** capability: the bytecode VM raises
`*: expected numeric operands` for `(* vector vector)`.

## Reading a printed tensor back in

A rank-2 tensor **prints** as `#((1 2) (3 4))`, but that text is not how you
write one: read back, it is a vector of two *lists*, and handing it to a tensor
operation is a compile-time refusal rather than a rank-2 tensor. Build one with
nested vector literals, `reshape`, or nested lists:

```scheme
#(#(1.0 2.0) #(3.0 4.0))                          ; nested vector literals
(reshape (tensor 1.0 2.0 3.0 4.0) (list 2 2))     ; from a flat tensor
(tensor (list (list 1.0 2.0) (list 3.0 4.0)))     ; from nested lists
```

All three are the same 2x2 tensor. Closing the print/read asymmetry is a build
item; so is turning the current refusal into a plain diagnostic rather than an
internal codegen error.

## What works, what to avoid

**Solid:** all shape ops, elementwise/unary math, linear algebra incl.
LU/QR/SVD/Cholesky/solve/inverse/det plus `linear-solve` (full-f64
mixed-precision iterative refinement — see [operations.md](operations.md#linear-solve--full-f64-solve-with-a-mixed-precision-fast-path)),
reductions, `conv1d`/`conv2d`/`conv3d`,
`max-pool2d`/`avg-pool2d`, `multi-head-attention` (forward), `embedding`,
`positional-encoding`, `dropout`, `softmax`, tensor-only activations, dtype
casts, rect/disk pixel fills, and all three library modules.

**Known limitations on this build:**

| Operation | Issue |
|-----------|-------|
| tensor creation | no element type-check for non-numeric args (`(tensor 1.0 "x")` → garbage) |
| type-guard text | reports raw doubles as `integer` |

(The release sweep-B fixes landed: `batch-norm`/`layer-norm` handle scalar
**and** per-feature tensor gamma/beta plus the 5-arg axis form; `tensor-load`
round-trips the shape; `gpu-reduce` returns a scalar; scalar math like `tanh`
maps elementwise over a tensor; `tensor-pow` accepts a scalar exponent; and
`(tensor 1 2.5 3)` builds the obvious 1-D tensor.)

## GPU status in one line

The `gpu-*` builtins resolve and run in both `-r` and AOT; plain `tensor-matmul`
auto-dispatches to Metal (both `-r` and AOT) when the size threshold is met, so
AOT is **not** CPU-only on Metal. ESH-0022/0023 are therefore partly stale —
see [gpu.md](gpu.md).

## See also

- [../ad/INDEX.md](../ad/INDEX.md) — automatic differentiation (flows through tensors)
- [../../breakdown/AUTODIFF.md](../../breakdown/AUTODIFF.md) — tensor backward pass, GPU gradient flow
