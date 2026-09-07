# `core.memory` — append-only, content-addressed, CRDT-merged event log

**Source**: [`lib/core/memory.esk`](../../../lib/core/memory.esk)
**Require**: `(require core.memory)` — must be required individually (not auto-loaded by `(require stdlib)`). Pulls in `agent.crypto` (for `sha256`), `core.distributed`, and `core.merkle`.

An in-memory event-log faculty for a distributed intelligence, composed from verified
core primitives. Each event is immutable and **content-addressed**: its id is the
`sha256` of the `sexp->canonical-string` of its body (everything except the id). The
body embeds the previous event's hash, so the log is also a **hash chain** — tampering
any field breaks the recomputed id. The log itself is an RGA (a convergent
append-only sequence CRDT from `core.distributed`), so two nodes can append
independently and `memory-merge` is conflict-free (commutative, idempotent,
associative) with dedup-by-id inherent to the RGA.

**Event shape**: a list `(mem-ev-v1 id prev-hash vclock node type payload)`.
**Log shape**: a mutable vector `#(mem-log node-id rga vclock prev-hash)`.

This module works under both `eshkol-run -r` (JIT) and AOT.

## Functions

### `(make-memory-log node-id)`
Create an empty log owned by `node-id` (any `equal?`-comparable value; symbols are
typical). Takes exactly **one** argument.

### `(memory-append! log type payload)`
Append an event of `type` carrying `payload`. Ticks the log's vector clock,
computes the content hash, chains it to the previous id, and RGA-appends. **Mutates**
`log` and **returns the new event**. Argument order is `(log type payload)`.

### `(memory-events log)`
Ordered list of events (RGA convergent order).

### `(memory-event? ev)`
Predicate: is `ev` a `mem-ev-v1` event?

### `(memory-event-id ev)` / `(memory-event-prev ev)` / `(memory-event-vclock ev)` / `(memory-event-node ev)` / `(memory-event-type ev)` / `(memory-event-payload ev)`
Event field accessors. `-id` is the sha256 hex string, `-prev` is the predecessor id
(or `#f` for the first event), `-vclock` is the vector clock at append time.

```scheme
;; memory.esk
(require core.memory)
(define la (make-memory-log 'node-A))
(define e1 (memory-append! la 'episodic '(saw cat)))
(memory-append! la 'fact  '(cat is-a animal))
(memory-append! la 'value (cons 'mood 0.7))
(display (memory-event? e1))          (newline)
(display (memory-event-id e1))        (newline)
(display (memory-event-prev e1))      (newline)
(display (memory-event-vclock e1))    (newline)
(display (memory-event-node e1))      (newline)
(display (memory-event-type e1))      (newline)
(display (memory-event-payload e1))   (newline)
(display (length (memory-events la))) (newline)
```
```
#t
e118a18ab8d6942308f0c9a221511a46a4fdb22566446351825217156fa39bf5
#f
((node-A 1))
node-A
episodic
(saw cat)
3
```

### `(memory-verify-events events)`
Recompute each event's id from its body and check integrity. Returns `#t` if all
events verify, else `(offending-event . reason)` where `reason` is one of
`not-an-event`, `no-id`, `hash-mismatch`. Works on single-node and merged logs
(content-integrity is independent of interleave order).

### `(memory-verify-chain log)`
Convenience: `(memory-verify-events (memory-events log))`.

```scheme
;; memory.esk
(require core.memory)
(define la (make-memory-log 'node-A))
(define good (memory-append! la 'episodic '(saw cat)))
(display (memory-verify-chain la)) (newline)
;; forge an event that keeps the id but mutates the payload:
(define tampered (list 'mem-ev-v1 (memory-event-id good) (memory-event-prev good)
                       (memory-event-vclock good) (memory-event-node good)
                       (memory-event-type good) 'TAMPERED))
(display (memory-verify-events (list tampered))) (newline)
```
```
#t
((mem-ev-v1 e118a18ab8d6942308f0c9a221511a46a4fdb22566446351825217156fa39bf5 #f ((node-A 1)) node-A episodic TAMPERED) . hash-mismatch)
```

### `(memory-merge a b)`
Conflict-free merge of two logs: RGA union (dedup-by-id inherent) plus vector-clock
merge. Commutative, idempotent, associative, and lossless. Returns a new log; the
result still verifies via `memory-verify-chain`.

```scheme
;; memory.esk
(require core.memory)
(define la (make-memory-log 'node-A))
(define lb (make-memory-log 'node-B))
(memory-append! la 'episodic '(saw cat))
(memory-append! la 'fact     '(cat is-a animal))
(memory-append! lb 'episodic '(heard dog))
(define m (memory-merge la lb))
(display (length (memory-events m))) (newline)   ; lossless: 2 + 1
(display (memory-verify-chain m))    (newline)
```
```
3
#t
```

### `(memory-fold-lww log key)`
Fold `value`-typed events whose payload is `(key . value)` into an LWW register,
using append order as the timestamp, and return the last-written value for `key`
(or `#f` if none). Since events are totally ordered (including post-merge), this is
"last write wins" over the log.

```scheme
;; memory.esk
(require core.memory)
(define la (make-memory-log 'node-A))
(memory-append! la 'value (cons 'mood 0.7))
(memory-append! la 'value (cons 'mood 0.9))
(display (memory-fold-lww la 'mood))          (newline)
(display (memory-fold-lww la 'nonexistent))   (newline)
```
```
0.9
#f
```

Edge cases: `memory-verify-events` on `'()` returns `#t`. `memory-fold-lww` on a
missing key returns `#f`.

### `(event-content-hash node prev vclock type payload)`
Re-derives an event's content hash from its body fields — the same computation
`memory-append!` uses to mint an id. Exported so a cross-module consumer can verify
integrity without rebuilding an RGA; `core.memory_store`'s streaming
[`memory-store-audit`](memory_store.md) is the reason it is public. Returns the
lowercase hex sha256 string.

```scheme
;; memory.esk
(require core.memory)
(define la (make-memory-log 'node-A))
(define e1 (memory-append! la 'episodic '(saw cat)))
(display (string=? (memory-event-id e1)
                   (event-content-hash (memory-event-node e1)
                                       (memory-event-prev e1)
                                       (memory-event-vclock e1)
                                       (memory-event-type e1)
                                       (memory-event-payload e1))))
(newline)
```
```
#t
```

## Known issues

None open.

**Closed:** through v1.3.4 this module emitted one gradual-typing warning from
`memory-fold-lww`'s inner loop, and was rejected outright under `--strict-types`
with `Type error: argument 3 of 'loop': expected Boolean, got Vector` — a named-let
accumulator seeded `#f` and reassigned to an `lww-register` vector on the first
match, which is a genuine Boolean/Vector unification conflict rather than a checker
false positive. Ledger **SW-63**, closed by **#494**, which split the fold into
`memory-lww-key-present?` (Boolean-only) and `memory-fold-lww-vector` (called only
once a match is known to exist, so its accumulator is register-shaped for its whole
lifetime). Both loops now carry one fixed accumulator shape, and the module compiles
clean with and without `--strict-types`.
