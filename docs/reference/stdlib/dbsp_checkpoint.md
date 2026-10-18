# Versioned DBSP checkpoints and process restart

`(require core.dbsp.checkpoint)` wraps standing relational circuits in an explicit
versioned template and logical clock. The native hosted implementation uses the
existing SHA-256 and atomic file-output facilities.

`(make-dbsp-resumable inputs declarations output registry schema)` accepts the
standing circuit's named inputs and topological declarations. Unary declarations
are `(map|filter|project output input operator-name)`, joins are
`(join output left right operator-name)`, and union/distinct retain their
existing data-only forms. Registry entries are `(name version-string kind
procedure...)`: one callback for unary operators, three for join. Callbacks must
be pure and immutable. Explicit versions are trusted semantic declarations,
not proofs that closures are equivalent. Schema, plan and version metadata are
captured by value; no closure or pointer identity is persisted.

`(dbsp-resumable-step snapshot next-tick changes)` requires the next exact integer
clock and returns `#(next-snapshot output-delta)`. Input row data is frozen before
admission. `dbsp-resumable-output` returns an owned Z-set;
`dbsp-resumable-tick` observes the clock. This clock rejects duplicate/skipped
logical ticks; it is not a transport deduplication or external-effects protocol.

`dbsp-resumable-checkpoint` returns canonical bytes containing template, clock,
current input Z-sets and a digest of every logical operator state. The shared
circuit's `dbsp-circuit-checkpoint-state` includes hidden signed distinct counts
and join arrangements, not only visible output. A typed wire codec preserves
exact rational trees, Unicode strings/symbols, characters, booleans, lists,
improper pairs and vectors; inexact/opaque/cyclic/deep data refuse. Bounds are
100,000 value/traversal units, nesting 64 and 4 MiB of checkpoint bytes.

`(dbsp-resumable-restore template bytes)` checks integrity, canonical encoding
and the full template/version contract before invoking any operator. It rebuilds
indexes from saved current inputs and requires the complete logical operator
state digest to agree. Restore therefore includes an explicit batch scan of
current inputs; it does not replay prior history. Subsequent steps consume only
the suffix. Checksums detect corruption; they do not authenticate a writer.

`(dbsp-checkpoint-save path snapshot)` publishes through existing same-directory
temporary-file/rename output; `(dbsp-checkpoint-load path template)` performs a
bounded read and verified restore. This provides atomic process-restart files.
The existing atomic writer does not promise fsync/power-loss durability, journal
transactions, concurrent-writer arbitration or exactly-once external effects.
There is no VM/browser persistence, recursive circuit or tensor checkpoint claim.

Run CTest with `-R dbsp_checkpoint`. The library suite checks every streamed
prefix, corruption/schema/version refusals, hidden negative distinct counts,
Unicode/escaped data and owned snapshots. The process fixture launches separate
native save/resume processes at all 13 cuts of a 12-tick stream, comparing the
final canonical state with uninterrupted execution; a truncated file must fail.
