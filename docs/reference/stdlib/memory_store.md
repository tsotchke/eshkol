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

**Store shape**: `#(mem-store log path)`, where `log` is a `core.memory` log vector.

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
events into an in-memory log (full RGA rebuild). If the file does not exist, returns
a fresh store. Unparseable lines are skipped with a warning. Argument order is
`(node-id path)`.

### `(memory-store-append! store type payload)`
Append an event of `type` with `payload` (a `(symbol . string|integer)` alist):
sanitize string values, append to the in-memory chain, then persist + `fsync` before
returning, and refresh the head sidecar. Returns the new event, or `#f` (with a
`DURABILITY FAILURE` message) if the disk write failed. Argument order is
`(store type payload)`.

### `(memory-store-count store)`
Number of events currently in the store's in-memory log.

### `(memory-store-head store)`
Content-id (hex string) of the last event in the in-memory log, or `#f` if empty.

### `(memory-store-verify store)`
Full two-layer audit of the loaded chain: (1) `core.memory` content hashes catch a
**modified** event (`hash-mismatch`); (2) strict linear linkage catches a **deleted**
event (`linkage-broken`) — the first event's `prev` must be `#f` and each event's
`prev` must equal its predecessor's id. Returns `#t` or a failure descriptor. Valid
for single-node chains; merged multi-node logs have legitimate forks.

```scheme
;; memory_store.esk  (writes to /tmp/events.log)
(require core.memory_store)
(define P "/tmp/events.log")
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
(define P "/tmp/events.log")            ; from the example above (2 events on disk)
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

### `(memory-store-audit path)` — currently unusable, see Known issues
Intended to be a **streaming** O(n)-flat integrity audit that never builds the RGA:
stream each line, re-derive its content hash, check its parent exists among
previously-seen ids, discard. Would return `(ms-audit-v1 ok links forks)` or
`(line-number . reason)` with `reason` in `unparseable | hash-mismatch |
orphan-parent`. **It cannot currently be called** — see below.

## Known issues

- **`memory-store-audit` is uncallable (unresolved cross-module symbol).**
  `memory-store-audit`'s body references `event-content-hash`, which is a private
  helper defined in `core.memory` but **not in that module's `(provide …)` list**.
  Across the module boundary the symbol does not resolve, so `memory-store-audit`
  itself fails to generate and is reported as an unknown function at every call site.
  All the *other* provided store functions work. Requiring `core.memory` explicitly
  first does **not** help (the failure is at `memory_store`'s own compile time).
  Repro:
  ```scheme
  (require core.memory_store)
  (memory-store-audit "/tmp/events.log")
  ```
  Observed:
  ```
  error: Unknown function: memory-store-audit
  ERROR: Failed to generate LLVM IR due to earlier code generation errors
  Unhandled exception: called undefined function 'memory-store-audit'
    (forward-referenced but never defined …)
  ```
  Likely fix (code, out of scope for docs): add `event-content-hash` to
  `core.memory`'s `provide` block, or re-derive it inside `core.memory_store`.
  No matching ledger id found in `.swarm/tasks/` (`ESH-0072` and `ESH-0085` mention
  adjacent areas — AD closure capture and `sha256` symbol hygiene — but not this).

- **Durability externs / NULL convention.** `fopen` failure returns `'()` from the
  FFI (not `0`); the module's `null-ptr?` checks `null?` accordingly. This is by
  design, noted here because it affects anyone extending the module.
