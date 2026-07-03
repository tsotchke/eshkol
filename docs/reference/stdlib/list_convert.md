# `core.list.convert` — list ↔ vector conversion

**Source**: [`lib/core/list/convert.esk`](../../../lib/core/list/convert.esk)
**Require**: `(require core.list.convert)` — auto-loaded by `(require stdlib)`.

Two conversions between proper lists and Eshkol vectors. Note that `#(...)`
literals are Eshkol *tensors* (homogeneous doubles) while `(vector ...)`
produces heterogeneous tagged-value vectors; `vector->list` works on both via
`vector-ref`/`vector-length`.

## Functions

### `(list->vector lst)`
Allocates a fresh vector of `(length lst)` slots (initialised to `#f`) and fills
it left-to-right from the list. Returns the vector.

```scheme
;; convert.esk
(require core.list.convert)
(display (list->vector '(1 2 3))) (newline)
(display (list->vector '())) (newline)
```
```
#(1 2 3)
#()
```

Edge cases: empty list returns an empty vector `#()`.

### `(vector->list vec)`
Builds a list from vector elements in index order by folding from the last
index down (so the result order matches the vector). Returns a proper list.

```scheme
;; v2l.esk
(require core.list.convert)
(display (vector->list #(4 5 6))) (newline)
(display (vector->list #())) (newline)
```
```
(4 5 6)
()
```

Edge cases: empty vector returns `'()`.
