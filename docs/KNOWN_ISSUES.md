---
kind: project
status: current
owner-area: project
since: v1.3.5
sources:
  - tests/vm_parity/PARITY.tsv
  - lib/backend/vm_region_evac.c
  - lib/backend/vm_logic.c
  - tests/coverage/release_record.json
---
# Known Issues

These are current limitations of v1.3.5-evolve. Resolved behavior is recorded in
[CHANGELOG.md](../CHANGELOG.md); the manifest in
[VM_PARITY.md](VM_PARITY.md) classifies each VM operation.

## VM memory reclamation

`with-region` reclaims VM heap storage at its lexical boundary. The
user-managed `region-open` and `region-close` handles share the native handle
protocol and diagnostics, but a VM close does not reclaim heap storage. The VM
reports this once on stderr; `ESHKOL_VM_REGION_QUIET=1` suppresses the note.
The manifest marks those two names `native-only-justified` for reclamation.

Outside `with-region`, the VM has no general garbage collector or automatic
per-loop nursery. A resident program without regions can grow until it reaches
the host limit. `ESHKOL_VM_HEAP_BUDGET_MB` defaults to 1024 and reports growth;
`ESHKOL_VM_HEAP_BUDGET_FATAL=1` turns the budget into a nonzero exit.
Escaping out-of-line payloads and continuations may retain a region even when
its lexical body finishes. Use an explicit `with-region` around transient VM
workloads and monitor the heap budget for long-running processes.

## VM parity and library calls

The parity manifest has **962** rows: 620 `vm-supported`, 46
`native-only-justified`, and 296 `gap`. The **388/388** differential count in
[the release record](../tests/coverage/release_record.json) covers its test
corpus, not every manifest row. Consult the row for a specific operation before
moving a native workload to the VM.

Some documented optional arguments still have no implementation on either
engine. For example, the two-argument form of `string-pad-left` does not supply
the documented default character. Use the fully specified form until the
lowering implements that default. The optional-arity generator
(`scripts/gen_builtin_min_arity.py`) identifies the affected signatures.

The `bytevector` constructor is a VM `gap` row; use `make-bytevector` or a
native engine for that constructor. The optional-argument comparison also
records value differences for `write-string` and `make-tensor` between engines.
Keep code that depends on those return or shape details on one engine until
its parity row and differential test establish the desired behavior.

## VM model loading

ESKM v1 is the default model format. Native loading can materialize rank-0 and
empty tensors; VM loading cannot yet materialize those shapes. Validate model
shapes before passing such checkpoints to the VM.

## Logic and types

The VM logic occurs-check does not recurse through fact internals
(`lib/backend/vm_logic.c`). A cyclic variable reference inside a fact can pass
that check. Avoid recursive fact terms when using VM unification.

The optional type system supports outermost `forall` quantification and tensor
shape dependent types. Higher-rank polymorphism and arbitrary value-level
computation in types are outside its implemented scope.

## Reporting an issue

Include a small `.esk` program, the engine and command used, the observed
output, and the expected output. See [CONTRIBUTING.md](../CONTRIBUTING.md).
