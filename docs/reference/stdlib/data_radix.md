---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/data/radix.esk
  - tests/stdlib/tensor_snapshot_test.esk
---

# `core.data.radix` — persistent integer index

`(radix-empty)` returns an empty index. Keys are exact integers from zero
through 2^63-1. `(radix-ref root key fallback)` observes a value;
`(radix-set root key value)` and `(radix-delete root key)` return new roots
without changing old versions. Values are caller-owned and must be immutable.
A stored `#f` is distinct from a missing leaf.

`(radix-fold root procedure initial)` calls `procedure(key,value,accumulator)`
in deterministic radix order, which is not numeric order. Every set path-copies
sixteen 16-slot nodes and one leaf. Deletion prunes empty paths. Index nodes
are private; there is no mutable global table or background process.
