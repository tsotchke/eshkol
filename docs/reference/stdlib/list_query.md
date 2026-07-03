# `core.list.query` — list length and predicate queries

**Source**: [`lib/core/list/query.esk`](../../../lib/core/list/query.esk)
**Require**: `(require core.list.query)` — auto-loaded by `(require stdlib)`.

Small query helpers over proper lists. `last` and `last-pair` are **not** here —
they are kept as builtins for performance.

## Functions

### `(count-if pred lst)`
Returns the number of elements for which `(pred elt)` is truthy.

### `(find pred lst)`
Returns the first element satisfying `pred`, or `#f` if none match.

### `(length lst)`
Returns the number of elements in a proper list.

```scheme
;; query.esk
(require core.list.query)
(display (count-if even? '(1 2 3 4 5 6))) (newline)
(display (find even? '(1 3 5 6 7))) (newline)
(display (find even? '(1 3 5))) (newline)
(display (length '(a b c d))) (newline)
(display (length '())) (newline)
```
```
3
6
#f
4
0
```

Edge cases: on `'()`, `count-if` → `0`, `find` → `#f`, `length` → `0`.

### Known issues — depth ceiling (ESH-0108)

**`length` is non-tail-recursive** (`(+ 1 (length (cdr lst)))`), so it consumes
one native stack frame per element. On large lists it SIGILLs (exit code 132)
with **no diagnostic** — the runtime's `ESHKOL_MAX_RECURSION_DEPTH=100000` guard
only covers C runtime list helpers, not stdlib user functions, so this failure
mode is a silent trap rather than a "maximum recursion depth exceeded" message.

Observed thresholds (JIT, `-r`, this machine):

| `length` input | result |
| --- | --- |
| 400,000 | `400000` (ok) |
| 500,000 | SIGILL, rc 132, no output |

Repro:
```scheme
(require core.list.query)
(require core.list.generate)
(display (length (iota 500000))) (newline)   ; => SIGILL (rc 132)
```

`count-if` shares the same non-tail shape (plus a predicate call per frame, so it
fails earlier — around ~250–300k). Tracked as **ESH-0108** ("stdlib
length/filter are non-tail-recursive: SIGILL without diagnostic on ~500k+
lists"). This is a docs-only note; no code changed. If you must measure very
long lists, fold a counter with a tail-recursive `fold`/`fold-left`
(see [`list_higher_order.md`](list_higher_order.md)) instead:
`(fold (lambda (acc _) (+ acc 1)) 0 lst)`.
