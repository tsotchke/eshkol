---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/dbsp/circuit.esk
  - tests/stdlib/dbsp_circuit_test.esk
---

# `core.dbsp.circuit` — standing relational circuits

`(make-dbsp-circuit inputs declarations output)` builds a pure immutable
snapshot. Inputs and output are symbols. Declarations name nodes in topological
order: `(map out in f)`, `(project out in f)`, `(filter out in predicate)`,
`(union out left right)`, `(distinct out in)`, or
`(join out left right left-key right-key emit)`. Cycles, duplicate names and
forward references refuse. Construction admits at most 65536 nodes.

`(dbsp-circuit-step snapshot changes)` accepts `((input-name Z-set) ...)` and
returns `(new-snapshot . output-delta)`. Repeated inputs consolidate before any
node runs. Only changed consumers enter the topological worklist. Every retained
row trace and join arrangement uses the shared persistent radix owner. Joins
include all three finite product terms; distinct retains exact signed counts.

`dbsp-circuit-output` and `(dbsp-circuit-input snapshot name)` explicitly
materialize full canonical Z-sets. `(dbsp-circuit-work snapshot)` returns a
fresh vector: visited nodes, join probes/matches, trace-row updates and
dictionary index writes. These are structural counters, not allocation bounds.
Hot-key buckets and affected outputs can be large; scheduling a large dirty
cone uses an ordered list. No whole-update constant-time claim is made.

Rows must be immutable and callbacks pure. A failed step publishes no new
snapshot and leaves historical/forked snapshots usable; external callback
effects are outside this guarantee. This initial slice has no recursive
feedback, durable checkpoint format, transport deduplication or compiler pass.

`(zset-row-key row)` exposes core.dbsp's existing canonical key encoding so
index owners share one serialization contract. Hash collisions still compare
keys with `equal?`. Run CTest with `-R dbsp_circuit` for prefix-by-prefix batch
equivalence, shared operands, collisions, rollback and fixed touched-key work.
