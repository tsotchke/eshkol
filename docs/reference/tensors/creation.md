# Tensors — Creation, dtypes, and the vector/tensor distinction

Everything below is verified by running it on the v1.3.4 compiler; outputs are
pasted as printed.

---

## Two containers: `vector` vs `tensor`

Eshkol has two superficially similar 1-D containers with different memory
layouts and purposes.

| | `vector` / `make-vector` | `tensor` / `#(…)` literal / `make-tensor` |
|---|---|---|
| Heap subtype | `HEAP_SUBTYPE_VECTOR` | `HEAP_SUBTYPE_TENSOR` |
| Element storage | heterogeneous **16-byte tagged values** | homogeneous **8-byte doubles** |
| Holds mixed types? | yes (int, string, double, …) | no — numeric only |
| Has `dtype`/`shape`? | no | yes (`f64` default) |
| Multi-dimensional? | nested vectors only | native n-D via shape |
| AD point type | all operators incl. `hessian`/`laplacian` | all operators incl. `hessian`/`laplacian` (ESH-0095 closed by #343) |

```scheme
;; vector holds heterogeneous values
(vector 1 "two" 3.0)            ;; => #(1 two 3)   (int, string, double)

;; tensor is homogeneous doubles with dtype + shape
(define t (tensor 1.0 2.0 3.0))
(tensor-dtype t)                ;; => f64
(tensor-shape t)                ;; => (3)
t                               ;; => #(1 2 3)
```

> **Display note.** Both containers print with the `#(…)` reader syntax, so
> `display` alone does not tell them apart — use `tensor-dtype`/`tensor-shape`
> (defined only for tensors) or the surrounding context. Integers and whole
> doubles both print without a decimal point (`3.0` → `3`), and strings print
> without quotes.

---

## Creating tensors

```scheme
(tensor 1.0 2.0 3.0)                       ;; 1-D tensor from elements  => #(1 2 3)
#(1.0 2.0 3.0)                             ;; literal tensor            => #(1 2 3)
(make-tensor (list 2 2) 0.0)               ;; shape (2 2) filled with 0.0
(tensor-reshape (tensor 1.0 2.0 3.0 4.0) (list 2 2))  ;; reshape flat -> 2x2
```

`(make-tensor shape fill)` takes a shape *list* and a fill value; the result is
`shape`-dimensional. Reshape a flat tensor into higher rank with
`tensor-reshape` (see [operations.md](operations.md)).

### From a nested collection

`(tensor X)` on a single collection argument takes its shape from the value's
own nesting, whatever built it — lists, vectors, or a mix, to any rank up to 8:

```scheme
(tensor (list 1.0 2.0 3.0))                        ;; => #(1 2 3)        shape (3)
(tensor (list (list 1.0 2.0) (list 3.0 4.0)))      ;; => #((1 2) (3 4))  shape (2 2)
(tensor #(#(1 2) #(3 4)))                          ;; => #((1 2) (3 4))  shape (2 2)
(tensor (list (vector 1.0 2.0) (vector 3.0 4.0)))  ;; => #((1 2) (3 4))  shape (2 2)
(tensor (list (list (list 1.0 2.0) (list 3.0 4.0)) ;; rank 3
              (list (list 5.0 6.0) (list 7.0 8.0)))) ;; shape (2 2 2)
```

The nest must be **rectangular** — every sub-collection at the same depth needs
the same length, and a given depth is either all numbers or all sub-collections.
A ragged nest is not a tensor and raises a catchable error rather than producing
a wrong shape:

```scheme
(guard (e (#t 'not-rectangular))
  (tensor (list (list 1.0 2.0) (list 3.0 4.0 5.0))))   ;; => not-rectangular
```

## Creating vectors

```scheme
(vector 1.0 2.0)                           ;; from elements
(make-vector 3 0.0)                         ;; length 3, each element 0.0
```

Use `vector` when you need a heterogeneous or symbolic container; use `tensor`
for numeric array math, GPU dispatch, and ML ops.

---

## Data types (dtypes)

Tensors carry a dtype tag. The default is `f64`. `tensor-cast` converts between
`f16`, `bf16`, `f32`, and `f64`; the dtype round-trips through
`tensor-dtype`:

```scheme
(define t (tensor 1.0 2.0 3.0))
(tensor-dtype t)                        ;; => f64   (default)
(tensor-dtype (tensor-cast t 'f32))     ;; => f32
(tensor-dtype (tensor-cast t 'f16))     ;; => f16
(tensor-dtype (tensor-cast t 'bf16))    ;; => bf16
(tensor-dtype (tensor-cast t 'f64))     ;; => f64
```

**Status.** The dtype metadata and `tensor-cast` tagging are wired for all four
types and round-trip correctly. The reduced-precision **software float paths**
(sf64 kernels) exercised by the GPU pipeline are covered by the
[`tests/gpu/sf64_*`](../../../tests/gpu/) suite. Full end-to-end f16/bf16
*storage and compute* across every operation is still being completed — see the
GPU-campaign tasks ESH-0022/ESH-0023 and [gpu.md](gpu.md) for the honest
current dispatch status.

---

## See also

- [operations.md](operations.md) — shape ops, elementwise, matmul, reductions, activations, conv/pool/norm/attention, fills, save/load, type guards
- [gpu.md](gpu.md) — GPU dispatch status (Metal/CUDA/XLA) and the `gpu-*` builtins
- [../ad/operators.md](../ad/operators.md#accepted-point-types) — which point type each AD operator accepts
