# `core.distributed` — logical clocks and CRDTs

**Source**: [`lib/core/distributed.esk`](../../../lib/core/distributed.esk)
**Require**: `(require core.distributed)` — must be required individually (not auto-loaded by `(require stdlib)`).

Pure value-level distributed-systems primitives: Lamport clocks, vector clocks, and
a family of state-based CRDTs (G/PN counters, observed-remove set, last-writer-wins
register and map, and an RGA sequence). They are intentionally independent of
sockets/async I/O so they can back local simulations now and networked replication
later. Every CRDT `*-merge` is commutative, associative, and idempotent, and value
tie-breaks use `sexp->canonical-string` so equal-timestamp merges converge
deterministically.

All examples below were run with `(require core.distributed)`.

## Lamport clocks

A Lamport clock is just an integer.

### `(lamport-zero)`
Returns `0`, the initial clock.

### `(lamport-tick clock [amount])`
Advance `clock` by `amount` (default `1`). `amount` must be non-negative or it
`error`s (`lamport-tick: expected a non-negative amount`).

### `(lamport-merge a b)`
Returns `max(a, b)`.

### `(lamport-recv local remote)`
Receive rule: `max(local, remote) + 1`.

### `(lamport-before? a b)`
Returns `#t` iff `a < b`.

```scheme
;; distributed.esk
(require core.distributed)
(display (lamport-zero))          (newline)
(display (lamport-tick 4))        (newline)
(display (lamport-tick 4 3))      (newline)
(display (lamport-merge 2 7))     (newline)
(display (lamport-recv 2 7))      (newline)
(display (lamport-before? 3 4))   (newline)
```
```
0
5
7
7
8
#t
```

## Vector clocks

Representation: an alist of `(node counter)`. Nodes are compared with `equal?`, so
symbols, strings, and small numeric IDs all work.

### `(vector-clock-empty)`
Returns `'()`.

### `(vector-clock-ref clock node)`
Counter for `node`, or `0` if absent.

### `(vector-clock-set clock node counter)`
Return a clock with `node` set to `counter`.

### `(vector-clock-tick clock node [amount])`
Increment `node`'s counter by `amount` (default `1`). `amount` must be non-negative
or it `error`s.

### `(vector-clock-merge a b)`
Pairwise max of the two clocks.

### `(vector-clock-nodes clock)`
List of node ids in the clock.

### `(vector-clock-compare a b)`
Returns one of `'before`, `'after`, `'equal`, `'concurrent`.

### `(vector-clock-before? a b)` / `(vector-clock-after? a b)` / `(vector-clock-concurrent? a b)` / `(vector-clock-equal? a b)`
Boolean shortcuts over `vector-clock-compare`.

```scheme
;; distributed.esk
(require core.distributed)
(define vc0 (vector-clock-empty))
(define vc2 (vector-clock-tick (vector-clock-tick vc0 'a) 'a 2))
(define vcb (vector-clock-tick vc0 'b))
(define vcm (vector-clock-merge vc2 vcb))
(display vc2)                                  (newline)
(display (vector-clock-ref vc2 'a))           (newline)
(display (vector-clock-ref vc2 'z))           (newline)
(display vcm)                                  (newline)
(display (vector-clock-nodes vcm))            (newline)
(display (vector-clock-compare vc2 vcm))      (newline)
(display (vector-clock-before? vc2 vcm))      (newline)
(display (vector-clock-concurrent? vc2 vcb))  (newline)
```
```
((a 3))
3
0
((a 3) (b 1))
(a b)
before
#t
#t
```

## G-counter (grow-only)

Representation: `(vector 'g-counter <vector-clock>)`.

### `(make-g-counter)`
Fresh zero counter.

### `(g-counter-counts counter)`
The underlying per-node vector clock.

### `(g-counter-inc counter node [amount])`
Return a counter with `node`'s increment raised by `amount` (default `1`,
non-negative or `error`).

### `(g-counter-value counter)`
Sum of all per-node counts.

### `(g-counter-merge a b)`
Pairwise-max merge.

```scheme
;; distributed.esk
(require core.distributed)
(define g  (g-counter-inc (make-g-counter) 'a 2))
(define gb (g-counter-inc (make-g-counter) 'b 5))
(display (g-counter-counts g))                    (newline)
(display (g-counter-value g))                     (newline)
(display (g-counter-value (g-counter-merge g gb))) (newline)
```
```
((a 2))
2
7
```

## PN-counter (increment/decrement)

Representation: `(vector 'pn-counter <positive g-counter> <negative g-counter>)`.

### `(make-pn-counter)`
Fresh counter (both halves zero).

### `(pn-counter-positive counter)` / `(pn-counter-negative counter)`
The increment / decrement G-counter halves.

### `(pn-counter-inc counter node [amount])` / `(pn-counter-dec counter node [amount])`
Bump the positive / negative half by `amount` (default `1`, non-negative or `error`).

### `(pn-counter-value counter)`
`value(positive) - value(negative)`.

### `(pn-counter-merge a b)`
Merge each half independently.

```scheme
;; distributed.esk
(require core.distributed)
(define pn  (pn-counter-dec (pn-counter-inc (make-pn-counter) 'a 5) 'a 1))
(define pnb (pn-counter-inc (make-pn-counter) 'b 3))
(display (pn-counter-value pn))                       (newline)
(display (pn-counter-value (pn-counter-merge pn pnb))) (newline)
```
```
4
3
```

## OR-set (observed-remove set)

Representation: `#(or-set adds removals clock)`. Adds are `(value dot)` entries;
removals are dots. A value is visible while at least one of its add-dots has not
been removed — so a concurrent add that a remove never observed survives.

### `(make-or-set)`
Fresh empty set.

### `(or-set-adds set)` / `(or-set-removes set)` / `(or-set-clock set)`
Field accessors (add entries, removed dots, and the writer vector clock).

### `(or-set-add set node value)`
Add `value` tagged with a fresh dot from `node`; ticks the clock.

### `(or-set-remove set value)`
Remove all currently-observed dots of `value`.

### `(or-set-member? set value)`
Is `value` currently visible?

### `(or-set-elements set)`
List of visible values.

### `(or-set-merge a b)`
Union of adds and removals, merged clocks.

```scheme
;; distributed.esk
(require core.distributed)
(define os  (or-set-add (make-or-set) 'a 'task))
(define osb (or-set-add (make-or-set) 'b 'task))
(display (or-set-adds os))                          (newline)
(display (or-set-clock os))                         (newline)
(display (or-set-member? os 'task))                 (newline)
(display (or-set-elements os))                      (newline)
(display (or-set-member? (or-set-remove os 'task) 'task)) (newline)
(display (or-set-member? (or-set-merge os osb) 'task))    (newline)
```
```
((task (a 1)))
((a 1))
#t
(task)
#f
#t
```

## LWW-register (last-writer-wins)

Representation: `(vector 'lww-register value timestamp writer deleted?)`. Later
timestamp wins; ties break on writer (canonical-string order), then delete-over-live,
then value.

### `(make-lww-register value timestamp writer)`
Construct a live register.

### `(lww-register-value r)` / `(lww-register-timestamp r)` / `(lww-register-writer r)` / `(lww-register-deleted? r)`
Field accessors.

### `(lww-register-present? r)`
`#t` unless the register is a tombstone.

### `(lww-register-set r value timestamp writer)`
Return a new register with the given value/timestamp/writer (a plain write, not a
merge — the caller supplies a monotone timestamp).

### `(lww-register-delete r timestamp writer)`
Return a tombstone register at the given timestamp.

### `(lww-register-merge a b)`
Return whichever of `a`/`b` is newer by the tie-break rules.

```scheme
;; distributed.esk
(require core.distributed)
(define r  (make-lww-register 'old 1 'a))
(define r2 (lww-register-set r 'new 2 'a))
(display (lww-register-value r))                      (newline)
(display (lww-register-present? r))                   (newline)
(display (lww-register-value (lww-register-merge r r2))) (newline)
(define rd (lww-register-delete r2 3 'a))
(display (lww-register-deleted? rd))                 (newline)
(display (lww-register-present? rd))                 (newline)
```
```
old
#t
new
#t
#f
```

## LWW-map

Representation: `(vector 'lww-map <alist of (key register)>)`. Each key's value is
an LWW-register, so per-key writes/removes/merges follow LWW rules.

### `(make-lww-map)`
Fresh empty map.

### `(lww-map-entries map)`
Raw `(key register)` alist (registers included, tombstones and all).

### `(lww-map-set map key value timestamp writer)`
Set `key` to `value` at `timestamp`/`writer`.

### `(lww-map-remove map key timestamp writer)`
Tombstone `key` at `timestamp`/`writer`.

### `(lww-map-ref map key [default])`
Visible value for `key`, else `default` (else `#f`). Tombstoned keys return
`default`.

### `(lww-map-contains? map key)`
Is `key` present (not tombstoned)?

### `(lww-map-visible-entries map)`
List of `(key value)` for visible keys only.

### `(lww-map-merge a b)`
Per-key register merge.

```scheme
;; distributed.esk
(require core.distributed)
(define m  (lww-map-set (make-lww-map) 'mode 'draft 1 'a))
(define m2 (lww-map-set m 'mode 'final 2 'b))
(display (lww-map-entries m))                         (newline)
(display (lww-map-ref m 'mode))                       (newline)
(display (lww-map-ref m 'missing 'DEF))               (newline)
(display (lww-map-contains? m 'mode))                 (newline)
(display (lww-map-ref (lww-map-merge m m2) 'mode))    (newline)
(display (lww-map-visible-entries m2))                (newline)
```
```
((mode #(lww-register draft 1 a #f)))
draft
DEF
#t
final
((mode final))
```

## RGA (replicated growable array — ordered sequence CRDT)

Representation: `#(rga entries clock)`. Entries are `(id prev-id value deleted?)`.
Inserts reference the previous entry's id; deletes tombstone so stale replicas can
never reintroduce a removed element during merge. Concurrent inserts at the same
position order deterministically by `(counter, node)`.

### `(rga-root-id)`
The sentinel root id `(rga-root 0)` — use as `prev-id` to insert at the head.

### `(make-rga)`
Fresh empty sequence.

### `(rga-entries rga)` / `(rga-clock rga)`
Field accessors (raw entry list incl. tombstones; the writer vector clock).

### `(rga-entry-id e)` / `(rga-entry-prev e)` / `(rga-entry-value e)` / `(rga-entry-deleted? e)`
Accessors for a single entry.

### `(rga-insert-after rga node prev-id value)`
Insert `value` from `node` after the entry with id `prev-id`; ticks the clock.

### `(rga-append rga node value)`
Insert `value` after the current visible tail (`rga-last-id`).

### `(rga-delete rga id)`
Tombstone the entry with `id`.

### `(rga-values rga)`
Visible values in sequence order.

### `(rga-last-id rga)`
Id of the last visible entry (or `(rga-root 0)` if empty).

### `(rga-merge a b)`
Union entries (tombstones win) and merge clocks.

```scheme
;; distributed.esk
(require core.distributed)
(define r1 (rga-append (make-rga) 'a 'alpha))
(define aid (rga-last-id r1))
(define r2 (rga-append r1 'a 'beta))
(display (rga-root-id))       (newline)
(display (rga-entries r1))    (newline)
(display (rga-values r2))     (newline)
(display (rga-last-id r2))    (newline)
(display (rga-values (rga-delete r2 aid))) (newline)
;; concurrent head inserts converge deterministically:
(define rl (rga-insert-after (make-rga) 'a (rga-root-id) 'left))
(define rr (rga-insert-after (make-rga) 'b (rga-root-id) 'right))
(display (rga-values (rga-merge rr rl))) (newline)
```
```
(rga-root 0)
(((a 1) (rga-root 0) alpha #f))
(alpha beta)
(a 2)
(beta)
(left right)
```

Notes: `rga-values`/`rga-last-id` are O(n) tree walks over the entry list; deletes
persist as tombstones, so `rga-entries` grows even as `rga-values` shrinks.
