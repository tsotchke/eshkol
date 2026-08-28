# `core.memory_store` — durable, hash-chained event store on disk

**Source**: [`lib/core/memory_store.esk`](../../../lib/core/memory_store.esk)
**Require**: `(require core.memory_store)` — must be required individually (not auto-loaded by `(require stdlib)`). Pulls in `core.memory`, `core.sexp`, and `core.files`.

Step 2 of `core.memory`: makes the in-memory event log **permanent**. Every append is
written to an append-only log file and `fsync`'d **before the append returns** (T0
durability), using fixed-arity libc externs (`fopen`/`fwrite`/`fflush`/`fileno`/
`fsync`/`fclose`) — variadic `open` is unsafe across the FFI on ARM64. The
persistence format is **one canonical s-expression per line** — the same rendering
the event ids are hashed over — so the file *is* the chain: replay re-parses each
line and re-derives every hash; a flipped bit anywhere breaks verification. No
external DB.

**Compatibility store shape**: `#(mem-store log path)`, where `log` is a `core.memory`
log vector. Durable constructors return an opaque eight-field handle that owns the
journal lock and retains only the authoritative tail and counters for compact opens.

An O(1)-open **head sidecar** (`<path>.head`, holding
`(ms-head-v1 <prev-hash> <vclock>)`) is written atomically after each append and used
by `memory-store-open-fast`; it is pure optimization — the log file remains the single
source of truth.

**v1 payload contract**: payloads are alists of `(symbol . string|integer)`. Strings
must be printable ASCII with no double-quotes or newlines — `core.sexp`'s canonical
rendering does not escape them. `memory-store-append!` sanitizes string values
automatically (defense in depth); `memory-store-sanitize` is exposed for ingest.

## Functions

### `(make-memory-store log path)`
Wrap an existing `core.memory` `log` and a file `path` into a store vector. Low-level
constructor; most callers use `memory-store-open`.

### `(memory-store? s)`
Predicate: is `s` a `mem-store` vector?

### `(memory-store-log s)` / `(memory-store-path s)`
Accessors: the underlying `core.memory` log, and the file path.

### `(memory-store-open node-id path)`
Open (or create) the durable chain at `path` for `node-id`, **replaying** all existing
events into an in-memory log (full RGA rebuild). Existing files are strict: a malformed,
torn, hash-invalid, or broken-link row fails closed instead of being skipped. If the
file does not exist, returns a fresh compatibility store. Argument order is
`(node-id path)`.

### `(memory-store-open-durable node-id path)` /
`(memory-store-open-fast-durable node-id path)`
Acquire exclusive ownership of `path` for the handle lifetime using a journal lock.
The full durable open rebuilds the RGA; the fast durable open validates the whole
canonical journal while retaining only the vector-clock, count, head, and complete
tail event. The sidecar is only a cache and is healed from journal truth. Both return
`#f` when ownership cannot be acquired or strict replay fails.

### `(memory-store-append! store type payload)`
Append an event of `type` with `payload` (a `(symbol . string|integer)` alist):
sanitize string values, append to the in-memory chain, then persist + `fsync` before
returning, and refresh the head sidecar. Durable handles advance only after the
complete row is flushed, synced, and closed; a failed sidecar publication does not
roll back an already committed journal row. Returns the new event, or `#f` if
persistence failed. An ambiguous first-file parent-directory sync poisons the
durable handle and raises, requiring restart from journal truth. Argument order is
`(store type payload)`.

### `(memory-store-close! store)`
Release the durable journal lock. A closed durable handle refuses further appends.

### `(memory-store-count store)`
Number of committed events. Compact durable handles report the journal count without
rebuilding the historic RGA.

### `(memory-store-head store)`
Content-id (hex string) of the last committed event, or `#f` if empty.

### `(memory-store-tail store)`
Return the complete last committed event, or `#f` if empty. Compact durable handles
retain this one event as their authoritative tail.

### `(memory-store-verify store)`
Full two-layer audit of the loaded chain: (1) `core.memory` content hashes catch a
**modified** event (`hash-mismatch`); (2) strict linear linkage catches a **deleted**
event (`linkage-broken`) — the first event's `prev` must be `#f` and each event's
`prev` must equal its predecessor's id. Returns `#t` or a failure descriptor. Valid
for single-node chains; merged multi-node logs have legitimate forks.

The log is **append-only**, so these two examples assume a fresh
`.scratch/events.log` — re-running them without deleting the file first replays the
previous run's events too and the counts grow accordingly.

```scheme
;; memory_store.esk  (writes to .scratch/events.log)
(require core.memory_store)
(define P ".scratch/events.log")
(define st (memory-store-open 'node-A P))
(display (memory-store? st))                              (newline)
(memory-store-append! st 'episodic (list (cons 'note "hello world")))
(memory-store-append! st 'fact     (list (cons 'k 42)))
(display (memory-store-count st))                         (newline)
(display (memory-store-verify st))                        (newline)
;; reopen from disk (replay) sees the same events and verifies:
(define st2 (memory-store-open 'node-A P))
(display (memory-store-count st2))                        (newline)
(display (memory-store-verify st2))                       (newline)
```
```
#t
2
#t
2
#t
```

The resulting file is one canonical event per line (ids are content-derived):
```
(mem-ev-v1 "909f8d2c…" #f ((node-A 1)) node-A episodic ((note . "hello world")))
(mem-ev-v1 "882b173f…" "909f8d2c…" ((node-A 2)) node-A fact ((k . 42)))
```
and the sidecar `events.log.head`:
```
(ms-head-v1 "882b173f…" ((node-A 2)))
```

### `(memory-store-open-fast node-id path)`
Fast open for **append-only** use (the per-tick weave): restore `prev-hash` + vclock
from the sidecar, then validate against the file tail (a plain line scan, no hashing);
if the sidecar is stale it heals from the tail with a warning. **The returned store's
in-memory event list holds only NEW events appended in this session** — use
`memory-store-open` (full replay) for reads. Falls back to full open if there is no
sidecar or no file.

```scheme
;; memory_store.esk
(require core.memory_store)
(define P ".scratch/events.log")        ; from the example above (2 events on disk)
(define st (memory-store-open-fast 'node-A P))
(display (memory-store? st))            (newline)
(memory-store-append! st 'value (list (cons 'x "y")))
(display (memory-store-count st))      (newline)   ; only NEW events this session
(define full (memory-store-open 'node-A P))
(display (memory-store-count full))    (newline)   ; full replay sees all
```
```
#t
1
3
```

### `(memory-store-sanitize str)`
Return `str` with every double-quote, control char, and non-printable/non-ASCII byte
replaced by a space. Load-bearing: `fwrite`'s length argument is `string-length`
(characters), but the file gets UTF-8 **bytes**, so one multibyte char (e.g. an
em-dash) truncated a write and glued two events onto one line. Until byte-length
crosses the FFI, payload strings are printable ASCII only.

```scheme
;; memory_store.esk
(require core.memory_store)
(display (memory-store-sanitize "a\"b"))            (newline)
(display (string-length (memory-store-sanitize "clean"))) (newline)
```
```
a b
5
```

### `(memory-store-audit path)`
A **streaming** O(n)-flat integrity audit that never builds the RGA: stream each
line, re-derive its content hash, check its parent exists among previously-seen
ids, discard. Returns `(ms-audit-v1 ok links forks)` on success, or
`(line-number . reason)` with `reason` in `unparseable | hash-mismatch |
orphan-parent` on the first problem.

```scheme
(require core.memory_store)
(memory-store-audit "/path/to/events.log")
;; => (ms-audit-v1 #t <links> <forks>)
```

### `(memory-store-audit-linear path)` / `(memory-store-linear-evidence store)`

`memory-store-audit-linear` is the strict canonical-journal evidence scanner:
it retains only the prior id while it streams, rejects a non-`#f` first parent,
any fork/link discontinuity, torn/trailing row, or content-hash mismatch, and
returns `(ms-audit-linear-v1 #t count head bytes)`. A missing path is valid
empty evidence. It compares file size before and after scanning and returns
`(0 . source-changed)` if it changed; that is diagnostic only, not ownership.

`memory-store-linear-evidence` requires an open durable writer handle and also
cross-checks the scanner's count/head against that handle. Use it when evidence
will authorize a rebuild or other mutable follow-up.

## Known issues

None. (Historically `memory-store-audit` was uncallable: its body references
`event-content-hash`, which was defined in `core.memory` but omitted from that
module's `(provide …)` list, so the symbol did not resolve across the module
boundary and `memory-store-audit` failed to generate — reported as an unknown
function at every call site. `event-content-hash` is now exported from
`core.memory`, so the audit both generates and runs.)
  No matching ledger id found in `.swarm/tasks/` (`ESH-0072` and `ESH-0085` mention
  adjacent areas — AD closure capture and `sha256` symbol hygiene — but not this).

- **Durability externs / NULL convention.** `fopen` failure returns `'()` from the
  FFI (not `0`); the module's `null-ptr?` checks `null?` accordingly. This is by
  design, noted here because it affects anyone extending the module.
