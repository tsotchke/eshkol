# Memory Faculty — `core.memory` and `core.memory-store`

An append-only, content-addressed, CRDT-merged **event log** for a distributed
agent. It is composed entirely from verified core primitives — `sha256`
(content-addressing, from [`agent.crypto`](crypto.md)), canonical s-expression
rendering (`core.sexp`), and the RGA / LWW-register / vector-clock CRDTs from
`core.distributed`.

Two layers ship:

- `core.memory` — the in-memory event-log faculty (v0).
- `core.memory-store` — a durable, fsync'd, replayable event log with a hash
  chain and head file.

Both run under `eshkol-run -r` (JIT) **and** AOT. Because the faculty calls
`sha256` internally, the transitive [agent-FFI link scan](ffi.md) pulls
`eshkol_sha256` into AOT binaries automatically.

---

## `core.memory` — In-memory event log

```scheme
(require core.memory)
```

Source: `lib/core/memory.esk`.

An **event** is immutable and content-addressed: its id is the `sha256` of the
canonical rendering of its body (everything except the id). The body embeds the
`prev` hash, so the log is also a hash chain — tampering with any field breaks
the id. The log itself is an RGA (a convergent append-only ordered-sequence
CRDT), so two nodes can append independently and `memory-merge` is conflict-free
(commutative, idempotent, associative) with dedup-by-id inherent to the RGA.

| Procedure | Signature | Description |
|-----------|-----------|-------------|
| `make-memory-log` | `(make-memory-log node-id)` | New empty log for a node |
| `memory-append!` | `(memory-append! log type payload)` | Append an event; returns the event |
| `memory-events` | `(memory-events log)` | Events in RGA order |
| `memory-merge` | `(memory-merge a b)` | Conflict-free merge of two logs |
| `memory-verify-chain` | `(memory-verify-chain log)` | Verify the hash chain |
| `memory-verify-events` | `(memory-verify-events events)` | Verify a list of events |
| `memory-fold-lww` | `(memory-fold-lww log key)` | Last-writer-wins fold over a key |

### Event accessors

`memory-event?`, `memory-event-id`, `memory-event-prev`, `memory-event-vclock`,
`memory-event-node`, `memory-event-type`, `memory-event-payload`.

Identity ("soul") and episodic ("anamnesis") memory are simply event *types* in
one log.

---

## `core.memory-store` — Durable event store

```scheme
(require core.memory-store)
```

Source: `lib/core/memory-store.esk`. Adds fsync'd append-to-file, a `.head`
pointer file, replay-on-open, and linkage verification. Uses fixed-arity libc
externs (`fopen`/`fwrite`/`fflush`/`fileno`/`fsync`/`fclose`) for ARM64-safe
FFI.

| Procedure | Signature |
|-----------|-----------|
| `make-memory-store` | `(make-memory-store log path)` |
| `memory-store-open` | `(memory-store-open node-id path)` — open + replay |
| `memory-store-open-fast` | `(memory-store-open-fast node-id path)` — open via head file |
| `memory-store-append!` | `(memory-store-append! store type payload)` |
| `memory-store-verify` | `(memory-store-verify store)` — verify chain linkage |
| `memory-store-count` | `(memory-store-count store)` |
| `memory-store-head` | `(memory-store-head store)` |
| `memory-store?` / `memory-store-log` / `memory-store-path` | accessors |
| `memory-store-sanitize` | payload sanitizer |

Durable content-addressed storage (`core.merkle` CAS) and network fan-out
(git / Syncthing) are the planned step-2 layers on top of this store.
