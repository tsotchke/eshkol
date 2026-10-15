# `core.dbsp.circuit` — standing relational circuits

`(make-dbsp-circuit inputs declarations output)` builds a pure immutable
snapshot. Inputs and output are symbols. Declarations name nodes in topological
order: `(map out in f)`, `(project out in f)`, `(filter out in predicate)`,
`(union out left right)`, `(distinct out in)`,
`(join out left right left-key right-key emit)`, or
`(aggregate out in key lift group emit)`. Cycles, duplicate names and
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

`(make-dbsp-group zero add negate zero?)` supplies explicit additive-group
operations. `zero` constructs an identity; `add` and `negate` return immutable
values; `zero?` recognizes identity exactly. The constructor checks the supplied
identity and its sum/negation, but arbitrary group laws remain the caller's
obligation. Group values and operations must be immutable/pure.

An aggregate sums `weight * lift(row)` per `key(row)`, using binary doubling
for signed integer weights. The whole tick is consolidated per key before
emitting replacements: retract `emit(key,old)` and insert `emit(key,new)`.
Identity-valued groups are absent, including nonempty groups whose values cancel.
Zero total row multiplicity does not erase a nonidentity sum. Only touched keys
are visited; emitted-row collisions consolidate normally. Group state advances
even when the emitted output delta cancels. Declarations and the group operation
vector are copied at construction. `#f` can itself be a nonidentity group value.

`core.dbsp.ad` supplies a numeric vector group and an existing-tape AD lift;
see [dbsp_ad.md](dbsp_ad.md). Run CTest with `-R 'dbsp_aggregate|dbsp_ad'` for
signed-prefix batch equivalence, large weights, key/emission collisions,
callback failure, captured declarations and cold-state locality checks.

Rows must be immutable and callbacks pure. A failed step publishes no new
snapshot and leaves historical/forked snapshots usable; external callback
effects are outside this guarantee. This initial slice has no recursive
feedback, durable checkpoint format, transport deduplication or compiler pass.

`(zset-row-key row)` exposes core.dbsp's existing canonical key encoding so
index owners share one serialization contract. Hash collisions still compare
keys with `equal?`. Run CTest with `-R dbsp_circuit` for prefix-by-prefix batch
equivalence, shared operands, collisions, rollback and fixed touched-key work.
