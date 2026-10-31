# `core.dbsp.recursive`

`(require core.dbsp.recursive)` adds nested logical clocks over the existing
persistent standing-circuit executor.

```scheme
(make-dbsp-recursive declarations output [round-budget])
(dbsp-recursive-step snapshot signed-base-delta)
(dbsp-recursive-output snapshot)
(dbsp-recursive-work snapshot)
```

Body inputs are `base` and `recur`. The ordinary builder admits
map/filter/project/union/difference/join/distinct nodes; a body must be monotone
in `recur` (use `difference` only on relations derived from `base`) and pure immutable-row callbacks.
Starting at empty selects the least fixed point when monotone approximants
converge. The round budget defaults to 256, maximum 1024. Nonconvergent bag
recurrences and insufficient budgets refuse without publishing a successor.

Each retained inner round owns an incremental circuit. Outer ticks install the
signed base change and change to the preceding approximant in each column.
Additional rounds reuse the prior steady column. Deletions use the same signed
operator path as insertions. Integrated base weights must remain nonnegative;
`dbsp-circuit-input-weight` checks touched rows through the existing input index.

Step returns `#(successor output-delta work)`. Empty consolidated updates execute
zero inner rounds. Work is `#(rounds checked-base-rows executed-nodes join-probes
trace-updates index-writes)`. Historical snapshots and forks remain valid.
Observers copy outer containers; row payloads retain the immutable-row contract.

Approximation consolidation/nonnegativity checks and final output differencing
can scan the corresponding approximant/output. Costs depend on recursive depth,
affected results, key skew and retained columns. This is not a constant-work or
flat-RSS claim. Stratified negation and a Datalog front end are provided by
[`core.reasoning.datalog`](reasoning.md), which compiles each stratum to one
recursive circuit and negates only relations of lower strata through the
`difference` operator. Nonmonotone recursive aggregates, a planner and a
durable recursive checkpoint are Planned for v1.6.1.

Tests compare every signed prefix, including cycles, alternate paths, weights
and deletions, with independent Floyd-Warshall closure. They also check delta
integration, retained forks, no-op cancellation, callback failures, excess
retraction and convergence refusals.
