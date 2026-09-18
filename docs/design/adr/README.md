---
kind: explanation
status: current
owner-area: docs
since: v1.3.5
sources:
  - docs/design/adr
  - .icc/adrs
  - ROADMAP.md
  - CHANGELOG.md
---
# Architecture Decision Records

An Architecture Decision Record (ADR) states one architectural decision: the
context that forces it, the decision itself, the alternatives rejected, and
the staged path that delivers it. A record is a blueprint. Its body is not
edited down when the tree lags behind it; the `Status:` line says how much of
the decision the tree carries, and the remaining stages stay in the record as
planned work. The release schedule for those stages is in
[`ROADMAP.md`](../../../ROADMAP.md).

This page is the index of every record in `docs/design/adr/`. The
documentation hub lists the same records in
[`docs/README.md`](../../README.md).

## Index

| Id | Title | Status | Date | Implemented in |
|---|---|---|---|---|
| [0000](0000-unified-trajectory.md) | Unified architectural trajectory | Accepted — partially implemented | 2026-07-09 | Stages 1 and 2: `inc/eshkol/frontend/node_identity.h` (`NodeId`), `inc/eshkol/frontend/semantic_identity.h` (`BindingId`, `NominalTypeId`, `TypeRef`), gates `scripts/run_adr0000_stage1_gate.sh` and `scripts/run_adr0000_stage2_gate.sh`; Stage 5 slice: `lib/core/dbsp.esk` |
| [0001](0001-oalr-concurrent-resident.md) | Concurrent, resident-grade OALR | Accepted — partially implemented | 2026-07-09 | Phase A accessor: `eshkol_memctx_current`, `eshkol_current_arena` in `lib/core/runtime_regions.cpp`; Phase C header definition: `eshkol_object_header_v2_t` in `inc/eshkol/memory_abi_v2.h` behind `ESHKOL_MEMORY_ABI_V2` |
| [0002](0002-ad-staged-dense-kernels.md) | Dense tensor AD nodes and a staged value-and-grad kernel | Accepted — partially implemented | 2026-07-09 | Phase A: `eshkol_value_and_grad` in `lib/core/runtime_autodiff.cpp`; Phase C dense nodes: `recordADNodeTensor` in `lib/backend/autodiff_codegen.cpp`, gate `scripts/run_dense_tensor_ad_gate.sh` |
| [0003](0003-codegen-vm-parity.md) | VM/LLVM parity conformance matrix and modularization notes | Superseded by `tests/vm_parity/PARITY.tsv` (op-by-op table) | 2026-07-09 | The live matrix is `tests/vm_parity/PARITY.tsv`, checked by `scripts/run_vm_parity.sh`; see [`docs/VM_PARITY.md`](../../VM_PARITY.md) |
| [0004](0004-type-system-trajectory.md) | One quantitative dependent type system for Eshkol | Accepted — partially implemented | 2026-07-09 | Type identity substrate: `inc/eshkol/frontend/semantic_identity.h`; the gradual relation of ADR 0013: `lib/types/type_relation.cpp` |
| [0005](0005-lambda-foundations-programs-to-weights.md) | Lambda foundations to resident programs-as-weights | Proposed | 2026-07-09 | — |
| [0006](0006-language-conformance-modules.md) | Binding-resolved libraries and proper tail invocation | Accepted — partially implemented | 2026-07-09 | `BindingId`, `ImportSet`, `ImportResolver`, `BindingResolver` in `inc/eshkol/frontend/semantic_identity.h` |
| [0007](0007-performance-pgo-wpo.md) | PGO, whole-program optimization, and staged training throughput | Accepted — partially implemented | 2026-07-09 | Phase 0 slice: `scripts/run_codegen_optlevel_tests.sh`, `bench/pgo_corpus/` with `scripts/run_pgo_corpus_smoke.sh` |
| [0008](0008-dev-experience-tooling.md) | One semantic tooling core for Eshkol developer experience | Accepted — partially implemented | 2026-07-09 | `WorkspaceResolver` in `inc/eshkol/frontend/workspace.h`, Diagnostic v1 in `inc/eshkol/frontend/diagnostic.h`, `eshkol check` and `eshkol doc modules` in `exe/eshkol.cpp` |
| [0009](0009-incremental-dataflow-dbsp.md) | Native DBSP-style incremental dataflow and unified differentiation | Accepted — partially implemented | 2026-07-09 | `core.dbsp` in `lib/core/dbsp.esk`, gate `scripts/run_dbsp_gate.sh`; reference page [`dbsp.md`](../../reference/stdlib/dbsp.md) |
| [0010](0010-closed-loop-assurance.md) | Closed-loop assurance architecture | Accepted — partially implemented | 2026-07-24 | `release-readiness-gate` in `.github/workflows/release.yml`, `.github/workflows/adversarial-nightly.yml`, `scripts/language_coverage.py`, `tests/diagnostics/` |
| [0011](0011-guest-collector-adapter.md) | Hosted guest collectors over OALR regions | Proposed | 2026-08-25 | — (the design's falsifier harness is [`0011-gc-adapter-falsifier/`](0011-gc-adapter-falsifier/build-and-run.sh)) |
| [0012](0012-object-abi-staged-migration.md) | Object ABI: discrimination, enforcement, and the staged migration | Accepted — partially implemented | 2026-08-25 | Stages 0-2: `.icc/abi-header-baseline.json` ratchet, `inc/eshkol/abi_fingerprint.h` (link-time guard, `ESHKOL_OBJECT_ABI_CACHE_TAG`), the WASM geometry guard |
| [0013](0013-gradual-type-relation.md) | One gradual type relation | Accepted — implemented in v1.3.5-evolve | 2026-09-14 | `TypeRelation` in `inc/eshkol/types/type_relation.h` and `lib/types/type_relation.cpp` |
| [0014](0014-release-invariant-contracts.md) | Release invariant contracts are measured before grading | Accepted — implemented in v1.3.5-evolve | 2026-09-14 | `scripts/lib/release_invariant_probes.sh`, `scripts/run_release_invariant_probes.sh`, `tests/toolchain/test_release_invariant_contracts.py`, `tests/toolchain/wasm_flat_ad_import_test.py` |
| [0015](0015-static-callee-binding-identity.md) | Static callee binding identity | Accepted — implemented in v1.3.5-evolve | 2026-09-14 | `inc/eshkol/backend/static_callee_binding.h` |
| [0016](0016-ad-alt-architect.md) | Staged dense-tensor reverse-mode AD and `value_and_grad` | Proposed | 2026-07-09 | — (ADR 0000 section 5 schedules four of its artifacts into the staged kernel ABI of ADR 0002 and keeps its typed static reverse schedule as the v2.0 endpoint) |
| [0017](0017-stochastic-binary-lambda-calculus.md) | Stochastic Binary Lambda Calculus | Proposed | 2026-08-30 | — (first slice targets v1.4.0) |
| [0018](0018-signed-curvature-stereographic-geometry.md) | Signed-curvature stereographic geometry and the K = 0 execution contract | Accepted (maintainer ruling 2026-08-30) | 2026-08-30 | — (targets v1.4.0) |
| [0019](0019-evergreen-documentation-architecture.md) | Evergreen documentation architecture | Accepted — partially implemented | 2026-09-17 | The v1.3.5 stage named in the record; contributor guide [`docs/DOCUMENTATION.md`](../../DOCUMENTATION.md) |
| [0020](0020-container-slot-store-boundary.md) | One store boundary for every container slot | Accepted — implemented (#701) | 2026-09-17 | v1.3.5 |
| [0021](0021-ast-string-owner.md) | One owner for AST string payloads, one spelling for recorded paths | Accepted — implemented in v1.3.5-evolve | 2026-09-18 | `inc/eshkol/frontend/ast_strings.h`, `inc/eshkol/frontend/source_paths.h`, gates `ast_strings_test`, `source_paths_test`, `scripts/check_ast_string_owner.py`, `scripts/check_artifact_paths.py` |
| [0022](0022-ad-value-boundaries.md) | One boundary for each place a value crosses a representation | Accepted — implemented (#706) | 2026-09-18 | v1.3.5 |

## Numbering

Each record has exactly one number, and each number names exactly one record.
A new record takes the next free number, which is one more than the highest
number in the index above. A number is never reused, including the number of a
superseded or withdrawn record: the old record stays in the directory with its
status changed, and the replacement gets a new number. The file name starts
with the four-digit number and a short slug (`0013-gradual-type-relation.md`),
and the H1 repeats the number (`# ADR-0013: One gradual type relation`). A
supporting directory carries the number of the record that owns it
(`0011-gc-adapter-falsifier/` belongs to ADR 0011).

Three records carry a `Renumbered:` line in their header. Citations written
before 2026-09-17, including the sections of `CHANGELOG.md` for releases
before 1.3.5-evolve, use their former numbers:

| Record | Former number | The record that keeps the former number |
|---|---|---|
| [0016](0016-ad-alt-architect.md) — Staged dense-tensor reverse-mode AD and `value_and_grad` | 0002 | [0002](0002-ad-staged-dense-kernels.md) — Dense tensor AD nodes and a staged value-and-grad kernel |
| [0017](0017-stochastic-binary-lambda-calculus.md) — Stochastic Binary Lambda Calculus | 0011 | [0011](0011-guest-collector-adapter.md) — Hosted guest collectors over OALR regions |
| [0018](0018-signed-curvature-stereographic-geometry.md) — Signed-curvature stereographic geometry | 0012 | [0012](0012-object-abi-staged-migration.md) — Object ABI staged migration |

## Status vocabulary

The `Status:` line of a record uses one of these forms, in the header style the
record already has:

| Form | Meaning |
|---|---|
| `Proposed` | The decision is written down and nothing of it is in the tree. |
| `Accepted` | The decision is ruled; the record names the release its implementation targets. |
| `Accepted — partially implemented: <slice>; remaining stages Proposed` | The named stage or slice is in the tree, and the line names the file or module that proves it. The other stages stay in the record as planned work. |
| `Accepted — implemented in <release>` | The decision is fully realised; the line names the file or symbol that carries it. |
| `Superseded by <record or artifact>` | Another record or a maintained artifact replaces this one. The superseded record stays in the directory. |

A status line claims an implementation only with a pointer that exists: a file
path and, where one applies, a symbol. The pointer is checked against the tree
when the line is written.

## Registering a record with ICC

The ICC registry under `.icc/adrs/` holds one JSON claim per record, which is
what `icc citation-binding` resolves an `ADR-NNNN` citation against. After
adding a record or changing a status:

1. Import the directory. Every `NNNN-*.md` file becomes (or refreshes) one
   `.icc/adrs/ADR-NNNN-<title-slug>.json` entry whose `source_path` is the
   markdown file; this page has no number and is skipped.

   ```sh
   icc adr import-markdown --repo <alias> --dir docs/design/adr --no-write --format markdown
   icc adr import-markdown --repo <alias> --dir docs/design/adr
   ```

2. Set the registry status. The importer recognises a status only when the
   whole value is one registry word (`proposed`, `accepted`, `rejected`,
   `superseded`, `deprecated`) on a `- Status:` list item, and records
   `proposed` otherwise, so every `Accepted — ...` and `Superseded ...` record
   is set explicitly after each import:

   ```sh
   icc adr update --repo <alias> --id ADR-0013 --status accepted
   icc adr update --repo <alias> --id ADR-0003 --status superseded
   ```

3. Remove an entry whose record was renamed or retitled, because the entry
   file name contains the title slug and the importer writes a second file
   instead of replacing the first:

   ```sh
   icc adr delete --repo <alias> --id <stale id>
   ```

4. Check the result. The list has one row per record, every id equals the
   record's number, and no id ends in `-dupN` (the importer's marker for two
   files sharing a number):

   ```sh
   icc adr list --repo <alias> --format markdown
   icc citation-binding --repo <alias> --no-memory --format markdown
   ```
