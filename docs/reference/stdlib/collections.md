# `core.collections` — priority queue, hash set, and deque

**Source**: [`lib/core/collections.esk`](../../../lib/core/collections.esk)
**Require**: `(require core.collections)` — **NOT** auto-loaded via `stdlib`; require it explicitly.

Three pure-Scheme containers not otherwise in stdlib: a binary **min-heap** priority queue (`#(data size compare)`), a **hash set** (thin wrapper over `make-hash-table`), and a two-list **deque** (`#(front back)`).

## Priority queue (min-heap)

### `(make-pq)` / `(make-pq compare)`
Create an empty priority queue. With no argument it is a min-heap ordered by
numeric key ascending (smallest key pops first). With a `compare` argument — a
2-arg procedure returning `<0` / `0` / `>0` — the heap is ordered by that
comparator; e.g. `(make-pq (lambda (a b) (- b a)))` gives a max-heap.

### `(pq-push! pq key item)`
Insert `item` with priority `key`. Lower key = higher priority.

### `(pq-pop! pq)`
Remove and return the min-key entry as a cons `(key . item)`. Errors if empty. (Note: the source header comment says `(values key item)`, but the actual return is a **cons pair**.)

### `(pq-peek pq)`
Return the min-key entry `(key . item)` without removing it. Errors if empty.

### `(pq-size pq)`
Current element count.

### `(pq-empty? pq)`
`#t` if empty.

```scheme
;; collections.esk
(require core.collections)
(define pq (make-pq))
(pq-push! pq 5 'five)
(pq-push! pq 1 'one)
(pq-push! pq 3 'three)
(display (pq-size pq)) (newline)   ; 3
(display (pq-peek pq)) (newline)   ; (1 . one)
(display (pq-pop! pq)) (newline)   ; (1 . one)
(display (pq-pop! pq)) (newline)   ; (3 . three)
(display (pq-empty? pq)) (newline) ; #f
```
```
3
(1 . one)
(1 . one)
(3 . three)
#f
```

Edge cases: `pq-pop!`/`pq-peek` on an empty queue → `Unhandled exception: pq-pop!: priority queue is empty` (resp. `pq-peek: ...`).

## Hash set

A set of `equal?`-keyed elements (backed by `make-hash-table`). Add is idempotent.

### `(make-set)` — empty set.
### `(set-add! s x)` — insert `x`.
### `(set-remove! s x)` — remove; returns `#t` if present, `#f` otherwise.
### `(set-contains? s x)` — membership `#t`/`#f`.
### `(set-size s)` — element count.
### `(set->list s)` — elements as a list (unspecified order).
### `(set-clear! s)` — empty the set.
### `(set-union a b)` / `(set-intersect a b)` / `(set-difference a b)` — return a **fresh** set; inputs unmodified. `set-difference` is `a \ b`.

```scheme
;; collections.esk
(require core.collections)
(define a (make-set)) (set-add! a 1) (set-add! a 2)
(define b (make-set)) (set-add! b 2) (set-add! b 3)
(set-add! a 1)                                   ; idempotent
(display (set-size a)) (newline)                 ; 2
(display (set-contains? a 1)) (newline)          ; #t
(display (sort (set->list (set-union a b)) <)) (newline)
(display (sort (set->list (set-intersect a b)) <)) (newline)
(display (sort (set->list (set-difference a b)) <)) (newline)
```
```
2
#t
(1 2 3)
(2)
(1)
```

## Deque (double-ended queue)

O(1) amortised at both ends. Represented as `#(front back)` — pops from the empty side pay an O(n) rebalance.

### `(make-deque)` — empty deque.
### `(deque-push-front! dq x)` / `(deque-push-back! dq x)` — add to either end.
### `(deque-pop-front! dq)` / `(deque-pop-back! dq)` — remove and return from either end; error if empty.
### `(deque-peek-front dq)` / `(deque-peek-back dq)` — inspect without removing; error if empty. (No `!` — these do not mutate.)
### `(deque-size dq)` / `(deque-empty? dq)` — count / emptiness.
### `(deque->list dq)` — front-to-back list.

```scheme
;; collections.esk
(require core.collections)
(define dq (make-deque))
(deque-push-back! dq 1) (deque-push-back! dq 2) (deque-push-front! dq 0)
(display (deque->list dq)) (newline)       ; (0 1 2)
(display (deque-size dq)) (newline)        ; 3
(display (deque-peek-front dq)) (newline)  ; 0
(display (deque-peek-back dq)) (newline)   ; 2
(display (deque-pop-front! dq)) (newline)  ; 0
(display (deque-pop-back! dq)) (newline)   ; 2
(display (deque->list dq)) (newline)       ; (1)
```
```
(0 1 2)
3
0
2
0
2
(1)
```

Edge cases: popping/peeking an empty deque → `Unhandled exception: deque-pop-front!: deque is empty` (resp. the matching message per op).

## Known issues

None. (Historically `make-pq` ignored a comparator argument and always used the
built-in `(- a b)` min-heap. `make-pq` now accepts an optional comparator:

```scheme
(require core.collections)
(define pq (make-pq (lambda (a b) (- b a))))  ; max-heap
(pq-push! pq 1 'one)
(pq-push! pq 9 'nine)
(display (pq-pop! pq)) (newline)   ; (9 . nine)
```
```
(9 . nine)
```
)
