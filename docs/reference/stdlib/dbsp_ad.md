---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/dbsp/ad.esk
  - tests/stdlib/dbsp_ad_test.esk
  - examples/streaming_least_squares.esk
---

# `core.dbsp.ad` — maintained fixed-parameter AD objectives

`(dbsp-ad-lift point build-loss)` returns a pure row-to-vector callback. The
generic vector `point` is copied once. For each row, `build-loss` receives
`(tape parameter-nodes row)` and returns a scalar node constructed on that tape
using the existing `core.ad.tape` operations. The lift runs the existing reverse
pass and returns an owned vector `#(loss dL/dp0 dL/dp1 ...)`. Each call has a
fresh tape. Point, loss and gradients must be finite reals. This is the numeric
tape interpretation, including its floating seeds, not exact algebraic AD.

`(dbsp-vector-group dimension)` builds the fixed-width numeric additive group
used to sum those contributions. Operations allocate fresh result vectors,
refuse dimension/nonfinite errors, and recognize zero exactly without a
tolerance. Floating accumulation can round; no exact-real group law is claimed.

Use the lift in an `aggregate` node from `core.dbsp.circuit`. A signed row
replacement retracts the old row contribution and inserts the new one. The
runtime retains one sum per key, so unchanged rows need no AD reevaluation.
Builders must be pure, use the provided tape correctly, and retain no nodes.
Rows and group values must be treated as immutable.

The [streaming example](../../../examples/streaming_least_squares.esk) maintains
loss and two derivatives for training/test cohorts and checks replacements,
deletions, historical snapshots and the absence of finite differences.

This first bridge fixes the parameter point for its lifetime. Changing the
point requires a new lift/circuit and replay. It does not incrementalize an
arbitrary reverse graph or promise a general training speedup. The benchmark
[`bench/dbsp_streaming_gradient.esk`](../../../bench/dbsp_streaming_gradient.esk)
compares identical row AD kernels under maintained and full recomputation,
excluding initialization and the closed-form oracle. It reports callback
counts, elapsed jiffies and process-global arena used bytes before/after ticks.
Those arena samples exclude other regions and are not total allocation or RSS
bounds. Run fresh processes with external peak-RSS measurement when comparing
memory behavior.
