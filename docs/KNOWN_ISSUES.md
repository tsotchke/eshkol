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
  - .icc/ledger/entries/LE-20.yaml
  - .icc/ledger/entries/LE-30.yaml
  - .icc/ledger/entries/SW-87.yaml
  - .icc/ledger/entries/SW-88.yaml
  - .icc/ledger/entries/SW-85b.yaml
  - .icc/ledger/entries/SW-86b.yaml
  - .icc/ledger/entries/SW-89.yaml
  - tests/error_handling/guard_coverage/README.md
  - tests/error_handling/guard_coverage/ENGINES.tsv
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

## Inherited tracked limitations

The following open issues were recorded before the v1.3.6 release work. They
remain known limitations; a coverage waiver records an engine gap and does not
fix it. The v1.3.5 behavior and release-record facts above remain historical
until the v1.3.6 release record is cut.

| Ledger | Affected engine(s) and behavior | Workaround | Evidence |
|---|---|---|---|
| LE-20 | Native JIT and AOT reject a legal nested named-let when an inner loop reuses an outer binding name. | Rename the inner accumulator; removing the intermediate named let also avoids the failure. | `tests/math_acceptance/shadowed_named_let_scope_test.esk` (wired, expected value 176). |
| LE-30 | Native JIT and AOT reject `set!` of a procedure introduced with `(define (f ...) ...)`. | Define the procedure with a lambda-valued variable definition, `(define f (lambda (...) ...))`. | Direct measurement on the v1.3.5 cut, recorded in `.icc/ledger/entries/LE-30.yaml`. |
| SW-87 | Native JIT and AOT reject assignment to a guard variable; capturing it in a closure fails LLVM verification. VM routes run both forms correctly. | Use a VM route for mutable or captured guard variables. | `tests/error_handling/guard_coverage/found/native_guard_variable_not_a_binding.esk`; measured in `18_guard_variable_mutation.esk`. |
| SW-88 | Native JIT and AOT fail LLVM verification when several individually valid `guard`/`dynamic-wind` blocks share one module. VM routes run the combined repro. | Compile the blocks separately on native engines, or run the combined form on a VM route. | `tests/error_handling/guard_coverage/found/guard_wind_multi_module_dominance.esk`; isolated controls `12_wind_normal_exit.esk`, `13_guard_inside_wind.esk`, and `14_sequential_winds.esk`. |
| SW-85b (historical alias SW-85) | Both VM routes (`vm-src`, `vm-eskb`) can lose a top-level mutation made in a guard clause. | Use a native engine for code that relies on the clause's global side effect. | `tests/error_handling/guard_coverage/found/vm_setbang_global_from_guard_clause_lost.esk`; the ledger notes a recurring-loop case where the two VM routes differ in when the mutation is lost. |
| SW-86b | Both VM routes return the wrong value for internal definitions in a guard body. Native engines are correct. | Use a native engine for this form. | `tests/error_handling/guard_coverage/found/vm_internal_defines_in_guard_body.esk`; measured in `16_internal_defines_in_body.esk`. |
| SW-89 | `vm-eskb` runs a guard clause before the dynamic-wind after-thunk; `vm-src` and native engines follow the required order. | Use `vm-src` or a native engine when the unwind/clause order matters. | `tests/error_handling/guard_coverage/found/vm_eskb_wind_order_vs_clause.esk`; measured by `17_wind_unwind_order.esk`. |

These are ledgered findings, not newly introduced regressions. The
`guard_coverage/ENGINES.tsv` manifest defines the five-axis default and the
format for any per-case waiver; it contains no case-specific waiver rows at this
base. A waiver, when present, records coverage scope and does not change the
status or behavior described here.

## Reporting an issue

Include a small `.esk` program, the engine and command used, the observed
output, and the expected output. See [CONTRIBUTING.md](../CONTRIBUTING.md).
