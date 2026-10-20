---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/ad/sparsity.esk
  - lib/core/ad/sparse_guw.esk
  - tests/stdlib/hessian_sparsity_test.esk
---

# Shared Hessian graph coloring

`core.ad.sparsity` provides `(hessian-color-graph adjacency)`, a pure
distance-two coloring function used by both numeric `core.ad.sparse_guw` and
exact `core.ad.algebraic_sparse`. The adjacency vector must contain sorted,
distinct, symmetric neighbor lists with no self edges. It returns
`(colors . color-count)` with the established degree ordering and deterministic
tie rules preserved. This is distance-two coloring rather than a minimal star
coloring: a four-vertex path uses three colors, while a star can use one color
per vertex.

The numeric suite checks retained coloring order, empty graphs, the star case,
malformed graph refusal, and numeric sparse-Hessian behavior. It is registered
as `hessian_sparsity` alongside the exact sparse-curvature suite.
