# Tensor Operations Reference

Signatures and outputs below are verified on the v1.3.0 compiler. Every tensor
operation is a **codegen builtin** — no `(require …)` is needed for anything on
this page (the library modules are separate; see
[ml-modules.md](ml-modules.md)). Outputs are pasted as printed by `display`
(whole doubles print without a decimal point, e.g. `3.0` → `3`).

Known-broken operations are called out inline with a **⚠ BUG** marker and
summarized in [Known limitations](#known-limitations).

---

## Shape operations

| Op | Signature | Example → output |
|----|-----------|------------------|
| `tensor-shape` | `(tensor-shape t)` | 2×3 → `(2 3)` |
| `tensor-reshape` / `reshape` | `(tensor-reshape t shape-list)` | `(reshape (tensor 1.0 2.0 3.0 4.0) (list 2 2))` → `#((1 2) (3 4))` |
| `tensor-transpose` / `transpose` | `(tensor-transpose t)` | 2×3 → `#((1 4) (2 5) (3 6))` |
| `tensor-length` | `(tensor-length t)` | total element count → `4` |
| `tensor-ref` | `(tensor-ref t i)` | flat index → `20` |
| `tensor-get` | `(tensor-get t i j …)` | multi-index → `3` |
| `tensor-set!` / `tensor-set` | `(tensor-set! t i v)` | mutates in place → `#(1 99 3)` |
| `tensor-data` | `(tensor-data t)` | underlying data → `#(1 2 3)` |
| `tensor-dtype` | `(tensor-dtype t)` | → `f64` |

`tensor-reshape` and `reshape` take the shape as a **list** (`(list 2 2)` or
`'(2 2)`).

---

## Elementwise & unary math

All binary elementwise ops require two tensors of matching shape and return a
new tensor:

```scheme
(tensor-add (tensor 1.0 2.0) (tensor 3.0 4.0))   ;; => #(4 6)
(tensor-sub (tensor 5.0 5.0) (tensor 1.0 2.0))   ;; => #(4 3)
(tensor-mul (tensor 2.0 3.0) (tensor 4.0 5.0))   ;; => #(8 15)
(tensor-div (tensor 8.0 6.0) (tensor 2.0 2.0))   ;; => #(4 3)
(tensor-maximum (tensor 1.0 5.0) (tensor 3.0 2.0)) ;; => #(3 5)
(tensor-minimum (tensor 1.0 5.0) (tensor 3.0 2.0)) ;; => #(1 2)
```

Scalar-broadcast and unary:

```scheme
(tensor-scale (tensor 1.0 2.0 3.0) 10.0)   ;; => #(10 20 30)
(tensor-neg (tensor 1.0 -2.0))             ;; => #(-1 2)
(tensor-abs (tensor -1.0 2.0 -3.0))        ;; => #(1 2 3)
(tensor-exp (tensor 0.0 1.0))              ;; => #(1 2.71828)
(tensor-log (tensor 1.0 2.71828))          ;; => #(0 1)
(tensor-sqrt (tensor 4.0 9.0))             ;; => #(2 3)
(tensor-sin (tensor 0.0))                  ;; => #(0)
(tensor-cos (tensor 0.0))                  ;; => #(1)
```

> **`tensor-pow` takes a tensor exponent, not a scalar** — it is fully
> element-wise:
>
> ```scheme
> (tensor-pow (tensor 2.0 3.0) (tensor 2.0 2.0))  ;; => #(4 9)
> (tensor-pow (tensor 2.0 3.0) 2.0)               ;; ERROR: expected tensor, got integer
> ```

---

## Linear algebra

| Op | Signature | Example → output |
|----|-----------|------------------|
| `tensor-dot` | `(tensor-dot a b)` | `#(1 2 3)·#(4 5 6)` → `32` |
| `tensor-matmul` / `matmul` | `(tensor-matmul A B)` | 2×2·2×2 → `#((19 22) (43 50))` |
| `batch-matmul` | `(batch-matmul A B)` | batched (2 args) → batched product |
| `tensor-norm` | `(tensor-norm v)` | `#(3 4)` → `5` |
| `tensor-inverse` | `(tensor-inverse A)` | 2×2 inverse |
| `tensor-det` | `(tensor-det A)` | → `10` |
| `tensor-solve` | `(tensor-solve A b)` | solves `Ax=b` → `#(2 3)` |
| `tensor-cholesky` | `(tensor-cholesky A)` | lower-triangular `L` |
| `tensor-lu` | `(tensor-lu A)` | **list** `(LU pivot sign)` |
| `tensor-qr` | `(tensor-qr A)` | **list** `(Q R)` |
| `tensor-svd` | `(tensor-svd A)` | **list** `(U S V)` |
| `tensor-cast` | `(tensor-cast t 'f32)` | dtype conversion (see [creation.md](creation.md#data-types-dtypes)) |

The three decompositions return **lists of tensors**, e.g.
`(tensor-svd A)` → `(U S V)`. `matmul` (and `gpu-matmul`) route through the
cost-model dispatch and may run on the GPU (see [gpu.md](gpu.md));
`batch-matmul` is a separate CPU path.

```scheme
(tensor-matmul (reshape (tensor 1.0 2.0 3.0 4.0) (list 2 2))
               (reshape (tensor 5.0 6.0 7.0 8.0) (list 2 2)))
;; => #((19 22) (43 50))
```

---

## Reductions

| Op | Signature | Example → output |
|----|-----------|------------------|
| `tensor-sum` | `(tensor-sum t)` | `#(1 2 3 4)` → `10` |
| `tensor-mean` | `(tensor-mean t)` | → `2.5` |
| `tensor-max` / `tensor-min` | `(tensor-max t)` | `#(1 5 3)` → `5` / `1` |
| `tensor-argmax` / `tensor-argmin` | `(tensor-argmax t)` | → `1` / `0` |
| `tensor-std` | `(tensor-std t)` | population (÷N) → `1.11803` |
| `tensor-var` | `(tensor-var t)` | population (÷N) → `1.25` |
| `tensor-reduce` | `(tensor-reduce t proc init)` | `(tensor-reduce t + 0.0)` → `10` |
| `tensor-reduce-all` | `(tensor-reduce-all t proc init)` | → `10` |
| `tensor-cov` | `(tensor-cov a b)` | two 1-D tensors → `0.666667` |
| `tensor-corrcoef` | `(tensor-corrcoef a b)` | two 1-D tensors → `1` |

`tensor-std`/`tensor-var` use the **population** normalization (divide by N).
`tensor-cov`/`tensor-corrcoef` take **two 1-D tensors** (not one 2-D matrix).

---

## Convolution & pooling

| Op | Signature | Status |
|----|-----------|--------|
| `conv1d` | `(conv1d input kernel stride)` | ✅ `(conv1d #(1 2 3 4 5) #(1 1) 1)` → `#(3 5 7 9)` |
| `conv2d` | `(conv2d input kernel stride)` | ✅ single scalar `stride`, VALID padding, batch-preserving (PR #80 unified VM/codegen) |
| `conv3d` | `(conv3d input kernel)` | ✅ (no stride argument) |
| `max-pool2d` | `(max-pool2d input kernel-size stride)` | ✅ 4×4 `2 2` → `#((6 8) (14 16))` |
| `avg-pool2d` | `(avg-pool2d input kernel-size stride)` | ✅ 4×4 `2 2` → `#((3.5 5.5) (11.5 13.5))` |

```scheme
;; 3×3 ⊛ 2×2 diagonal kernel, stride 1, VALID padding
(conv2d (reshape (make-tensor (list 3 3) 1.0) (list 3 3))
        (reshape (tensor 1.0 0.0 0.0 1.0) (list 2 2)) 1)
;; => #((2 2) (2 2))
```

The pooling ops are named `max-pool2d` / `avg-pool2d` (with the `2d` suffix).

---

## Normalization, attention, embedding

| Op | Signature | Status |
|----|-----------|--------|
| `batch-norm` | `(batch-norm input gamma beta epsilon [axis])` | ✅ scalar **or** per-feature tensor gamma/beta; 4-arg and 5-arg (axis) forms |
| `layer-norm` | `(layer-norm input gamma beta epsilon [axis])` | ✅ scalar **or** per-feature tensor gamma/beta; 4-arg and 5-arg (axis) forms |
| `multi-head-attention` | `(multi-head-attention Q K V num-heads W_Q W_K W_V W_O [mask])` | ✅ forward pass runs (8–9 args) |
| `embedding` | `(embedding indices weights)` | ✅ `idx #(0 2)`, 3×2 weights → the selected rows |
| `positional-encoding` | `(positional-encoding max-len d-model)` | ✅ sinusoidal, exactly 2 args |
| `dropout` | `(dropout tensor drop-prob)` | ✅ 2 args; `(dropout #(1 2 3) 0.0)` → `#(1 2 3)` |
| `softmax` | `(softmax tensor [axis])` | ✅ `(softmax #(1 2 3))` → `#(0.0900306 0.244728 0.665241)` |

> **`batch-norm` and `layer-norm` accept both a scalar and a per-feature
> tensor for `gamma`/`beta`** and support the optional 5th `axis` argument.
> `(layer-norm #(1.0 2.0 3.0 4.0) #(1.0 …) #(0.0 …) 1e-5)` normalizes to
> `#(-1.3416 -0.4472 0.4472 1.3416)`. (Earlier builds mis-read a tensor
> gamma/beta as a scalar — garbage denormals — and SIGSEGV'd on the axis form;
> both are fixed.) **Attention backward** is still a passthrough stub in the
> tensor backward pass — see
> [../../breakdown/AUTODIFF.md](../../breakdown/AUTODIFF.md) "v1.1 AD
> Extensions".

---

## Activations

Most activations are **tensor-only** and raise a catchable type error on a
scalar argument:

```scheme
(relu (tensor -1.0 2.0 -3.0))    ;; => #(0 2 0)
(sigmoid (tensor 0.0))           ;; => #(0.5)
(gelu (tensor 0.0 1.0))          ;; => #(0 0.841192)
(leaky-relu (tensor -2.0 3.0))   ;; => #(-0.02 3)   (α = 0.01)
(silu (tensor 0.0 1.0))          ;; => #(0 0.731059)
(relu -5.0)                      ;; ERROR: Type error in relu: expected tensor
```

> **`tanh` and the other scalar math builtins now map elementwise over a
> tensor.** `(tanh 0.0)` → `0` and `(tanh (tensor 0.0 1.0))` →
> `#(0 0.761594)`. The same holds for `sin`/`cos`/`tan`/`exp`/`log`/`sqrt`/…
> when handed a tensor. (Earlier builds reinterpreted the tensor pointer as a
> scalar double and returned a silently-wrong scalar.)

---

## Type guards (PR #79)

Every tensor operation routes its operands through a single runtime choke point
(`eshkol_tensor_operand_checked`, `lib/backend/tensor_codegen.cpp`). A
wrong-typed operand raises a **catchable** exception (not a crash), naming the
op and the offending type:

```scheme
(tensor-add (tensor 1.0 2.0) "not-a-tensor")
;; ERROR: Type error in tensor-add: expected tensor, got string   (and raises)

(guard (e (#t (display "caught") (newline)))
  (tensor-add (tensor 1.0 2.0) "not-a-tensor"))
;; prints: caught
```

The exception is catchable via `guard` and `with-exception-handler`. One
cosmetic caveat: a raw **double** operand is reported as `got integer` (the
guard tags any untagged scalar as INT64).

> **Creation is not guarded.** `(tensor 1.0 "x" 3.0)` does *not* raise — it
> reinterprets the string pointer as a double and yields garbage. Build tensors
> from numeric literals only. Mixing integer and float *numeric* arguments is
> fine: `(tensor 1 2.5 3)` builds the obvious 1-D tensor `#(1 2.5 3)`. Leading
> integers are treated as a shape prefix only when their product exactly
> matches the number of remaining elements (e.g. `(tensor 2 2 1.0 2.0 3.0 4.0)`
> → a 2×2 tensor); otherwise every argument is a 1-D element. (Earlier builds
> raised a confusing "insufficient elements" error on `(tensor 1 2.5 3)`.)

---

## Pixel fills

Native C-runtime fills for 2-D/3-D framebuffers (~7 µs/frame for compose-heavy
rendering):

```scheme
;; half-open rectangle [row0,row1) × [col0,col1)
(tensor-rect-fill! fb 1 1 3 3 9.0)   ;; fills the 2×2 block at (1,1)

;; filled disk of given radius centered at (cy,cx)
(tensor-disk-fill! fb 2 2 1 7.0)     ;; radius-1 disk centered at (2,2)
```

Note the rectangle's **exclusive** upper bounds: to fill a 2×2 block starting
at (1,1) use `1 1 3 3`.

---

## Save / load

```scheme
(tensor-save "path.bin" t)   ;; arg order is (PATH TENSOR); returns #t
(tensor-load "path.bin")     ;; round-trips shape + data
```

`tensor-save` writes a correct binary file (magic `TKSE`, IEEE-754 element
bit-patterns) and returns `#t`. **The argument order is `(path, tensor)`.**

> **`tensor-load` round-trips the shape.** After a save/load the shape, element
> data, count and dtype all survive (`(tensor-shape (tensor-load …))` on a 2×2
> save returns `(2 2)`). (Earlier builds allocated the `dimensions[]` array but
> never copied the loaded dimensions into it, so `tensor-shape` reported `(0 …)`
> and the tensor displayed as `#()`.)

---

## GPU-labeled builtins

`gpu-matmul`, `gpu-elementwise`, `gpu-softmax`, `gpu-transpose`, `gpu-reduce`
all **resolve** and run (`gpu-reduce` returns a scalar). See [gpu.md](gpu.md)
for signatures and honest Metal/CUDA dispatch status.

---

## Known limitations

| Operation | Issue |
|-----------|-------|
| tensor creation | no element type-check for non-numeric args (`(tensor 1.0 "x")` → garbage) — build tensors from numeric literals only |
| type-guard text | reports raw doubles as `integer` |
| attention/embedding backward | passthrough stubs (see AUTODIFF.md) |

These are current-build facts, reported for honesty — none are fixed by this
documentation.

---

## See also

- [creation.md](creation.md) — vector vs tensor, dtypes
- [gpu.md](gpu.md) — GPU dispatch honest status
- [ml-modules.md](ml-modules.md) — `ml.optimization`, `core.manifold`, `signal.fft`
- [../ad/INDEX.md](../ad/INDEX.md) — autodiff through tensor operations
