# `core.dbsp.tensor_snapshot` — shared-block tensor versions

Require `(require core.dbsp.tensor_snapshot)`, on top of `core.dbsp.tensor`.

- `(tensor-snapshot tensor)` owns a baseline copy in blocks of at most 256 cells.
- `(tensor-snapshot-shape snapshot)` returns a fresh shape list.
- `(tensor-snapshot-ref snapshot coordinates)` reads one zero-based cell.
- `(tensor-snapshot-apply snapshot delta)` returns a new version, copying each
  touched block once and sharing untouched blocks and the baseline table.
- `(tensor-snapshot->tensor snapshot)` explicitly materializes an owned native tensor.
- `(tensor-snapshot->reader snapshot)` returns a read-only operand accepted by
  `tensor-matmul-delta`, reading affected cells directly from shared storage.
- `(tensor-snapshot-work snapshot)` returns a copied vector containing touched
  blocks, copied payload cells, copied internal index nodes and copied internal
  index slots. Leaf wrappers are excluded from the latter two counters.

Initialization and full observation scan the dense payload. Sparse updates
never initialize a tensor-sized delta or dirty array. Native cells remain
floating point. Shapes stay fixed; old snapshots and independent forks remain
valid. The counters describe structural work, not total OALR/heap allocation.

Run `ctest --test-dir build --output-on-failure -R tensor_snapshot`. The fixed
one-block edit is checked at 1024, 65536 and 1048576 cells, including actual
baseline identity, fresh changed blocks, cancellation and historical values.
