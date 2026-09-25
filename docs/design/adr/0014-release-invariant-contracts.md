# ADR 0014 — Release invariant contracts are measured before grading

- Status: Accepted — implemented in v1.3.5-evolve (`scripts/lib/release_invariant_probes.sh`, `scripts/run_release_invariant_probes.sh`, and the CTest contracts `tests/toolchain/test_release_invariant_contracts.py` and `tests/toolchain/wasm_flat_ad_import_test.py`)
- Date: 2026-09-14
- Decision owners: Eshkol compiler/runtime maintainers; release assurance maintainers
- Related: ADR-0000 (frontend identity substrate), ADR-0001 (OALR object model),
  ADR-0010 (closed-loop assurance), `.icc/architecture-model.yaml`

## Context

The architecture model checks compiler and runtime invariants from source, but
the release architecture grade also needs evidence that the corresponding
runtime and ABI checks actually ran on the release build. Several contracts
cross type, layout, generated glue, and shell-harness boundaries. A declaration
or a green process exit alone cannot establish those contracts.

## Decision

Release invariants use executable, fail-closed contracts at each boundary:

- The parser's `eshkol_ast_t::node_id`, the NodeId allocator, and semantic
  queries share the `eshkol_node_id_t` key type. The AST field remains a
  32-bit alias, preserving its public layout.
- Each heap subtype marked `[DEEPWALK]` has a corresponding native evacuation
  handler. Leaf evacuation cases do not count as deep walks.
- The flat automatic-differentiation WASM imports are generated from the
  checked-in import fragment and a single core-key manifest. The import test
  checks generated freshness, required keys, and runtime behavior.
- Smoke probes distinguish a measured `PASS` or `FAIL` from harness
  infrastructure failure. Only measured outcomes emit typed `test_result`
  receipts; infrastructure failure emits no result that could be mistaken
  for a verdict.
- The release workflow emits ABI/layout, closed-enum, live AD-counter, and VM-parity receipts
  before the ICC architecture grade. The VM parity gate supplies its own
  aggregate receipt. The AD counter gate measures both an exact positive case
  and a real finite-difference negative control before grading.
- Package verification is required on both archive-producing workflow paths.
- Bridge backward functions must match their canonical registry rows, including
  the squared-distance implementation in its separate translation unit.

CTest registers the source-contract, receipt-contract, and generated WASM
import tests so local and CI test runs exercise the same assertions.

## Consequences

An invariant grade is not treated as release evidence until its concrete probe
has run against the release build. Missing receipts remain missing, and a
tooling failure cannot be promoted to a passing or failing product result.
Changes to AST identity types, evacuation coverage, the flat-AD import surface,
or receipt order now fail at the closest contract test.

The ABI proof in this change is limited to the shared public C/C++ header and
the asserted AST field type and layout. It does not claim cross-architecture
binary equivalence for serialized or compiler-generated artifacts.
