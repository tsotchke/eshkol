---
kind: explanation
status: current
owner-area: ad
since: v1.3.5
sources:
  - inc/eshkol/ad_node_registry.def
  - lib/core/runtime_autodiff.cpp
  - lib/core/runtime_tensor_alloc.cpp
  - lib/backend/tensor_backward.cpp
  - lib/backend/tagged_value_codegen.cpp
  - .icc/ledger/entries/SW-181.yaml
---
# ADR-0023: A dense tensor AD node is read as a tensor through one resolver

**Status:** Accepted
**Ledger:** SW-181; follow-ups SW-186, SW-187, SW-188
**Scope:** Native LLVM code generation, JIT and AOT, and the C runtime the
emitted code calls. The bytecode VM has no dense tensor carrier (SW-186).

## Context

ADR-0002 Position A gave the reverse tape a dense representation: `matmul`,
the dense elementwise operators and `batch-matmul` publish their result as one
`ad_node_t` whose `tensor_value` is an f64 buffer, tagged `CALLABLE`, so a chain
of dense operations records one node per operation. Operators with a dense
rule of their own recognise that node before reading it. Every other consumer
of a tensor value did not: the tensor boundary `eshkol_tensor_operand_checked`
classified HEAP_PTR tensors and numeric collections and raised for anything
else, so `tensor-dot`, `reshape`, `tensor-scale`, `relu`, `tensor-exp`,
`tensor-sqrt`, `softmax` and `tensor-get` refused a matmul result with
"expected tensor, got ad-node" (SW-181); the two-argument `tensor-ref`
treated every CALLABLE as a scalar node and returned the dense node whole;
`vector-length`, `vector-ref` and `vector->list` classified by object header
and read the node as a Scheme vector, which is how the stdlib's `tensor-norm`,
written over `vref` and `vector-length`, answered 0 through a matmul. The
sweep also found `eshkol_ad_node_probe` bounding a candidate node's type by
the literal 63 while the registry had grown to 95 rows.

Each of these is a value crossing from the dense representation into a
consumer that knows only the scalarised one. Adding a case per consumer would
be the closed-list pattern that ADR-0022 retired for cons cells.

## Decision

One resolver, `eshkol_ad_dense_node_elements` in `lib/core/runtime_autodiff.cpp`,
turns a dense node into the tensor of its shape whose element `i` is an
`AD_NODE_DENSE_ELEM` node (registry row 95, `TENSOR` payload, `INLINE`
backward) projecting element `i`: `input1` is the dense node, `params[0]` the
flat index, `tensor_value` a one-element copy so the scalar-consumer bridge
applies, `value` the same number for the scalar rules. Its backward is the
identity scatter into `input1`'s gradient at `params[0]`, the inverse of
`TENSOR_PACK`, and like it performs no arithmetic, so it can change only the
representation of a gradient and never its value. With no tape recording, the
resolver returns the plain values.

Two thin adapters route to it and nothing else does: the tensor boundary
(`eshkol_tensor_operand_checked` and `eshkol_tensor_matrix_operand_checked`)
for every tensor operator, and `TaggedValueCodegen::resolveDenseTensorNode`
for the collection builtins and `tensor-ref`. Operators with dense rules are
untouched and keep their one node per operation. `eshkol_ad_node_probe`'s
bound is `AD_NODE_TYPE_COUNT`.

## Consequences

- Every scalarising tensor consumer accepts a dense node, in forward and
  reverse mode, JIT and AOT, with the analytic gradient
  (`tests/ad/dense_tensor_operand_boundary_test.esk`, 24 checks).
- A dense node read by a scalarising consumer costs one projection node per
  element, the same order as the scalarising path it feeds; dense chains are
  unaffected.
- What a dense node *is* (`tensor?`, `display`) is still answered by its
  CALLABLE tag (SW-188); jacobian's function-result slot and the forward-dual
  matmul used by hessian have not been given the resolver (SW-187); the VM
  carries no derivative through tensor operators at all (SW-186). The end
  state this ADR points at is a dense AD tensor that is a `HEAP_SUBTYPE_TENSOR`
  object carrying its node, after which no consumer needs a case and this
  resolver becomes internal to the tape.
