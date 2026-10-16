# Transactional SQLite DBSP changefeeds

`(require core.dbsp.sqlite_changefeed)` exports:

```scheme
(dbsp-sqlite-create! db source template)
(dbsp-sqlite-enqueue! db source sequence changes)
(dbsp-sqlite-consume! db source template [on-advance])
(dbsp-sqlite-state db source template)
```

The borrowed handle is an existing `agent.sqlite` connection. One database owns
two versioned tables: `eshkol_dbsp_stream_v1` stores produced/consumed cursors
and the circuit checkpoint; `eshkol_dbsp_batch_v1` stores canonical source
batches. Source identity is a 1..256-byte string, bound as data in prepared
statements. Sequences are positive exact signed-word integers. There is no SQL
interpolation of caller data, background polling, connection ownership or
external transport acknowledgment.

Create requires a clock-zero resumable template and returns `created` or
`existing`. Reopening validates the stored checkpoint against that template
and checks that its logical clock equals the consumed cursor. Enqueue accepts
at most 64 named Z-set changes, consolidates repeated input names, and admits
rows through the existing typed checkpoint codec. Arbitrary strings/symbols/
characters are encoded by that codec; inexact, cyclic or over-budget data
refuses. Queue payloads reuse a versioned identity-circuit checkpoint, rather
than defining a second serialization format. A batch must have the next
produced sequence. An identical old sequence returns `duplicate`; changed
payloads and sequence gaps refuse. Batches remain in the database so old
replays can be checked. There is no retention/compaction policy yet.

Consume runs one `BEGIN IMMEDIATE` transaction, restores the authoritative
checkpoint, reads exactly the next queued batch, runs the existing circuit,
and writes the next checkpoint and consumed cursor together. Only after
successful `COMMIT` does it return `#(advanced state output-delta)`. With no
pending batch it returns `#(idle state empty-delta)`. It keeps no in-memory
cursor between calls. A failure rolls back and publishes no successor. Invalid
input names or callback/model errors during consumption leave the queued batch
pending; producer/model compatibility remains the application's responsibility.

The optional `on-advance` callback receives the staged sequence after the
checkpoint/cursor write and before COMMIT. A same-database transactional outbox
can be written there. It can read owned staged state with `dbsp-sqlite-state`
on the same handle. It must check its own SQLite return codes, must not start
nested transactions, and may run again after rollback. External effects are
not made atomic by this callback. No callback runs on idle. The caller must
not share the same connection across concurrent transactions; independent
connections serialize through SQLite's write lock. BUSY and other step errors
are explicit failures, not an empty queue.

`agent.sqlite` adds `sqlite-step-code` (the raw integer status) and
`sqlite-step-checked` (true for ROW, false for DONE, raises for all other codes).
The old `sqlite-step` remains compatible. Every adapter statement checks bind,
step, prepare and transaction results. The checkpoint's private renderer now
uses the existing canonical S-expression renderer over its typed wire data,
avoiding stale output-string-port bytes on an older release runtime. This
preserves the existing canonical bytes and integrity/template checks.

The atomicity boundary is one SQLite database. SQLite's configured journal,
synchronous mode, filesystem and storage determine power-loss durability; this
module does not change them or promise fsync/power-failure qualification. An
ambiguous commit error requires rereading the persisted cursor before deciding
whether to retry. Output deltas are returned after commit; if the process dies
before an external consumer observes them, use a transactional outbox. This is
not exactly-once external delivery. Checkpoints rebuild and verify logical
operator state, and can be expensive; no indexed/warm restore or flat RSS claim
is made. Batch/checkpoint byte and node budgets are inherited from the codec.

The primary suite checks signed hidden distinct state, source isolation, replay
and gap admission, quoted/Unicode data, schema mismatch, checked versus legacy
stepping, trigger-induced failure and companion-write rollback/commit. The
process fixture kills execution after both checkpoint/cursor fields are staged
but before COMMIT, verifies recovery to the old prefix, then kills after COMMIT
and verifies the new prefix plus idempotent replay. JIT and AOT crash tests are
registered with agent FFI on POSIX when Python 3 is available. They test process death, not power loss.
