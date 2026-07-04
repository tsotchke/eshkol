# Tensors & ML — Reference

Machine-verified reference for Eshkol's tensor system and the ML/numerical
library modules (v1.3.0-evolve). Every signature and output on these pages was
produced by running the current compiler; broken operations are documented
against their behavior, not hidden.

## Pages

| Page | Contents |
|------|----------|
| [creation.md](creation.md) | `vector` vs `tensor` (heterogeneous 16-byte tagged values vs homogeneous 8-byte doubles), literals `#(…)`, `make-tensor`/`make-vector`, and dtypes (f16/bf16/f32/f64/i8). |
| [operations.md](operations.md) | Shape ops, elementwise & unary math, linear algebra + decompositions, reductions, conv1d/2d/3d, pooling, normalization, attention, embedding, activations, the PR-#79 type guards, pixel fills, and save/load — with known-broken ops flagged. |
| [gpu.md](gpu.md) | Honest GPU-dispatch status: the `gpu-*` builtins, the cost-model threshold, what actually runs on Metal in `-r` vs AOT, and how that squares with ESH-0022/0023. |
| [ml-modules.md](ml-modules.md) | `ml.optimization`, `core.manifold`, `signal.fft` — every `provide` with signature and a run example. |

## Two containers at a glance

| | `vector` / `make-vector` | `tensor` / `#(…)` / `make-tensor` |
|---|---|---|
| Storage | heterogeneous 16-byte tagged values | homogeneous 8-byte doubles |
| Mixed types | yes | no (numeric only) |
| dtype / shape | no | yes (`f64` default) |
| Displays as | `#(…)` | `#(…)` (same — distinguish via `tensor-dtype`) |

## What works, what to avoid

**Solid:** all shape ops, elementwise/unary math, linear algebra incl.
LU/QR/SVD/Cholesky/solve/inverse/det, reductions, `conv1d`/`conv2d`/`conv3d`,
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
