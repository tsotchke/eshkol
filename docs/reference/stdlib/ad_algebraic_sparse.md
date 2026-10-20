---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/ad/algebraic_sparse.esk
  - tests/stdlib/algebraic_sparse_test.esk
  - examples/algebraic_sparse.esk
---

# Exact sparse curvature

`core.ad.sparsity` owns the existing deterministic distance-two graph coloring,
now shared with numeric `core.ad.sparse_guw`. `(hessian-color-graph adjacency)`
accepts a vector of sorted distinct symmetric neighbor lists and returns
`(colors . color-count)`. Degree order and tie rules are preserved. This is
distance-two coloring, not minimal star coloring: a path uses three colors,
while a star can require one color per vertex.

`core.ad.algebraic_sparse` adds
`(algebraic-sparse-hessian owner function point pattern [hvp])`. `owner` is an
exact coefficient domain, `point` a vector of at most 128 values and `pattern`
an explicit list of structural `(i j)` pairs. Pairs are symmetrized,
canonicalized and deduplicated. The pattern must contain every structural
nonzero; there is no numerical sparsity detector or proof of that obligation.

Without `hvp`, `function` receives hyperdual seeds and returns a value in their
exact context. A scalar probe reads one compressed-column row. Actual calls
are counted: the 12-variable path witness uses 23 scalar probes, not three.
This mode needs one spare perturbation level. With `hvp`, the callback receives
an owned point and a fresh integer 0/1 direction and returns the entire vector
`H*direction` in the declared domain. Columns are copied and cached; the
120-variable path witness uses three actual HVP calls. Callback semantics and
pattern completeness remain explicit caller obligations.

`algebraic-sparse-hessian-ref`, `algebraic-sparse-hessian-dense` and
`algebraic-sparse-hessian-summary` inspect results. Entry/dense observers copy
coefficient storage. The summary is `#(dimension colors calls upper-entries
mode)`, with mode `scalar` or `hvp`. Empty patterns make no probes. Cached
whole columns use `O(n*used-colors)` storage, and requested dense observations
cost quadratic storage. There is no wall-clock speed claim from call counts.

Run CTest with `-R 'algebraic_sparse|hessian_sparsity|sparse_tensors'`.
Exact nonlinear tridiagonal Hessians are compared with independent formulas,
and the existing numeric sparse suite exercises the shared coloring seam.


The suite's analytic reference is defined in
[`algebraic_sparse_test.esk`](../../../tests/stdlib/algebraic_sparse_test.esk#L28):
its diagonal is `12*x_i^2 + 2*x_(i-1)` (omitting the predecessor at `i=0`),
its adjacent mixed entries are `2*x_max(i,j)`, and other entries are zero.
The scalar recovery is compared against that formula for all 144 dense entries
by the assertion loop at lines 47–51. The HVP path also checks selected entries
against the same formula at lines 56–58. These are in-suite analytic formula
checks; no separate Python/Fraction reference executable was run.
