# `core.ml.sparse` — compressed sparse row matrices

**Source**: [`lib/core/ml/sparse.esk`](../../../lib/core/ml/sparse.esk)
**Require**: `(require core.ml.sparse)` — not loaded by `(require stdlib)`.
**Design**: [ADR-0049](../../design/adr/0049-neuro-symbolic-bridge.md)

A sparse matrix stores only its structural nonzeros: row offsets, the column of each stored entry (ascending within a row) and the values. Every operation is written over generic arithmetic, so stored values may be AD carriers: `sparse-with-values` swaps a pattern's values for a parameter vector and `gradient` differentiates through products with respect to the nonzeros only.

## Functions

- `(sparse-from-triplets rows cols triplets)` — from `(row col value)` triplets, duplicates summed; `(sparse-from-dense m)` from a vector of row vectors, zeros not stored; `(sparse-identity n)`.
- `(sparse->dense s)`, `(sparse->triplets s)`.
- `(sparse? x)`, `(sparse-rows s)`, `(sparse-cols s)`, `(sparse-nnz s)`, `(sparse-values s)`, `(sparse-ref s i j)` (0 for an entry not stored; indices are range-checked).
- `(sparse-with-values s vals)`, `(sparse-map-values f s)`, `(sparse-scale c s)`.
- `(sparse-matvec s x)`, `(sparse-transpose s)`, `(sparse-matmul a b)` (Gustavson's row-by-row product), `(sparse-add a b)`.

```scheme
;; sparse-ex.esk
(require core.ml.sparse)
(define s (sparse-from-triplets 3 3 '((0 0 2.0) (1 2 -1.0) (2 1 4.0) (1 2 0.5))))
(display (sparse-nnz s)) (display " ") (display (sparse-ref s 1 2)) (newline)
(display (sparse-matvec s (vector 1.0 2.0 3.0))) (newline)
(display (sparse->triplets (sparse-transpose s))) (newline)
(display (gradient (lambda (vals) (vector-ref (sparse-matvec (sparse-with-values s vals) (vector 1.0 2.0 3.0)) 1)) (sparse-values s))) (newline)
```
```
3 -0.5
#(2 -1.5 8)
((0 0 2) (1 2 4) (2 1 -0.5))
#(0 3 0)
```

## Verification
[`tests/ml/sparse_test.esk`](../../../tests/ml/sparse_test.esk) checks construction, lookup, products, sums and transposes against dense arithmetic, and gradients with respect to the stored values against closed forms and forward mode.
