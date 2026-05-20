# Eshkol Documentation Audit — 2026-05-20

## Overview

This report summarises a project-wide documentation audit and the resulting corrections, performed at the `v1.2.1-scale` release boundary against source commit `a5d9dea` (master). The audit consumed every Markdown artefact in the repository (`README.md`, `DESIGN.md`, the five top-level language documents, `docs/breakdown/` 35-doc subsystem set, the 30-doc tutorial collection, the 14-doc platform programme, the eight `docs/vision/` documents, the seven `docs/future/` documents, the two `docs/superpowers/` plans, and the `press/`, `examples/`, `tests/`, `tools/icc_extras/`, and `.icc/` peripheral READMEs).

## Method

The audit was driven by the Infinite Context Coder (ICC) project at `~/Desktop/infinite_context_coder`. The Eshkol repository is registered with ICC as `eshkol_lang` and was reindexed before the run (1,555 indexed files, 989,767 lines across all tracked languages). Two ICC artefacts framed the comparison:

1. `icc architecture-summary` (cached under `.icc/eshkol_arch_summary.md`) — public module roots, integration surfaces, dependency hubs, test roots, and per-module size statistics drawn directly from the source index.
2. `icc doc-intelligence` (cached under `.icc/eshkol_doc_intel.md`) — per-document, claim-level scoring against the source index, producing a triage table of *grounded*, *unsupported*, and *unresolved-reference* counts for every Markdown artefact in the repository (7,192 claims scored, 1,928 unresolved-reference fingerprints).

Nine parallel auditor agents (one Explore agent per documentation cluster) were dispatched against this evidence. Each agent received the doc-intelligence rows for its assigned files, the architecture summary, and the project memory snapshot at `~/.claude/projects/-Users-tyr-Desktop-eshkol/memory/MEMORY.md`. Findings were aggregated into a single corpus, two false-negative claims about file existence were caught on verification (`lib/backend/eshkol_vm.c` and `.gitlab-ci.yml` both exist), and the remaining corrections were applied directly to the working tree.

The audit philosophy was *correct, do not rewrite*. Documents whose content was accurate but whose register was informal were left intact; corrections were limited to factual errors, stale numerics, broken cross-references, and missing or extra content.

## Findings Summary

### Class A — Factual errors against current source (corrected)

| Document | Issue | Resolution |
|---|---|---|
| `ESHKOL_LANGUAGE_GUIDE.md` | Documented `--target wasm` CLI flag; `exe/eshkol-run.cpp` only accepts `--wasm` / `-w`. | Replaced with the correct flag. |
| `DESIGN.md` | Module table listed `tensor_codegen.cpp` as a 19,187-line monolith; the file was split into thirteen per-domain modules in v1.2 (now totalling ~20,500 lines). Total codegen claim of "~232,000 lines" was off by a factor of ~2.7. | Rebuilt the table from current `wc -l` output; rewrote the total-lines summary; added a paragraph documenting the tensor-codegen split and listing the new sibling modules. |
| `docs/ESHKOL_V1_ARCHITECTURE.md` | Main-codegen size given as 35,074 lines; actual 33,962. | Synced. |
| `docs/breakdown/COMPILER_ARCHITECTURE.md` | Module table mirrored the stale DESIGN figures and the "232,000-line" total. | Rebuilt against current source. |
| `docs/breakdown/PARALLEL_COMPUTING.md` | Source-file reference table understated `parallel_codegen.cpp` (705 vs 945), `parallel_llvm_codegen.cpp` (2,401 vs 2,601), `thread_pool.cpp` (1,132 vs 1,350). | Synced. |
| `docs/breakdown/AUTODIFF.md` | Claimed "16 AD node types total" with a 12-entry enum; the actual enum spans 0–18 plus 19–45 (activations, tensor primitives, scalar utilities). | Rewrote the enum block with the elementary, activation, tensor, and utility ranges and a note that the enum continues to grow. |
| `docs/breakdown/DEVELOPER_TOOLS.md` | "33 special forms and syntax keywords"; the listed array (and `eshkol_lsp.cpp` `keywords()`) contains 47. | Corrected to 47. |
| `docs/breakdown/TYPE_SYSTEM.md` | The Uω universe row implied a feature parity with U₀–U₂; the inference engine does not exercise Uω. | Marked Uω as "target: v1.3+" and added a clarifying sentence. |
| `docs/breakdown/COMMAND_LINE_REFERENCE.md` | Omitted `eshkol-repl --machine` / `-m` warm-worker mode (in source at `exe/eshkol-repl.cpp:786-793`) and the `:examples` REPL command (`exe/eshkol-repl.cpp:544`). | Added a dedicated paragraph for `--machine` (EREPL framing on stderr) and a row in the inspection-commands table for `:examples`. |
| `docs/breakdown/DOCKER.md` | Listed five images; `docker/Dockerfile.paper` (pinned for the SDNC paper reproducibility suite) was undocumented. | Added a sixth image section with build/run instructions and added the row to the image-comparison table. |
| `docs/breakdown/ROADMAP.md` | Framed v1.0-foundation as the current release; treated v1.1 and v1.2 as future work, contradicting `RELEASE_NOTES.md` and `RELEASE_READINESS_REPORT.md`. | Restructured into "Current Release", "Released", and "Forward Roadmap" sections; documented what shipped in v1.1-accelerate, v1.2-scale, and v1.2.1-scale; moved still-aspirational items to v1.3-evolve. |
| `docs/breakdown/MATH_STDLIB.md` | "See Also" table pointed at `TENSOR_SYSTEM.md` and `NUMERIC_TOWER.md` — neither file exists. | Repointed at `VECTOR_OPERATIONS.md` and `EXACT_ARITHMETIC.md`. |
| `docs/breakdown/FUNCTION_COMPOSITION.md` | Two distinct sections both titled "## Composition Patterns" produced a malformed TOC and duplicate anchors. | Renamed the second to "## Advanced Composition Patterns". |
| `docs/breakdown/VM_MEMORY_OPS_AS_WEIGHT_MATRICES.md` | Title claimed "26 delegated operations"; inventory table summed to 38 with the control-flow class understated (row claim 8, listed opcodes 10); current implementation encodes 82 opcodes per the document's own implementation note. | Rewrote the document title to drop the spurious count, added an inventory clarification, and corrected the control-flow row. |
| `docs/vision/FUTURE_ROADMAP.md` | Marked v1.2-scale "Planned" although the release has shipped (and v1.2.1-scale closed out on 2026-05-20). | Re-keyed the development-timeline table to record v1.2-scale and v1.2.1-scale as complete and to add v1.3-evolve and revised targets. |
| `docs/superpowers/plans/2026-04-04-llvm-21-toolchain-unification.md` | All task checkboxes unchecked; the core CMake enforcement (`cmake/LLVMToolchain.cmake`, `ESHKOL_REQUIRED_LLVM_MAJOR = 21`) has actually landed on master. | Added a "Status: partial" header note recording which portions shipped and which remain deferred. |
| `docs/KNOWN_ISSUES.md` | "Roadmap" table targeted multiple features at "v1.2" although v1.2 has shipped; ONNX / model-serialisation framing did not reflect the native `.eshkol-model` serialiser; FFI line claimed callbacks were missing while the HTTP/SQLite/subprocess surfaces shipped. | Rewrote the table for accurate target releases (most items bumped to v1.3) and clarified the alternatives. |
| `docs/API_REFERENCE.md` | 64 markdown source-citation links of the form `[name](path.cpp:NNN)` — GitHub's web view does not honour `:line` anchors on file links, so every one resolved to file root rather than the cited line. | Stripped the line suffixes from all 64 markdown link targets so each link resolves to the cited file; inline prose mentions of line numbers were preserved. |
| `docs/ESHKOL_V1_ARCHITECTURE.md`, `docs/breakdown/{AUTODIFF, FUNCTION_COMPOSITION, VECTOR_OPERATIONS, MEMORY_MANAGEMENT}.md` | Same broken-anchor pattern across breakdown documents (79 links across five files). | Stripped uniformly via the same transformation. |

### Class B — Version and date stamps

| Document | From | To |
|---|---|---|
| `DESIGN.md` | `## v1.2.0-scale` | `## v1.2.1-scale` |
| `ESHKOL_V1_LANGUAGE_REFERENCE.md` | `Version v1.2.0-scale` | `Version v1.2.1-scale` |
| `COMPLETE_LANGUAGE_SPECIFICATION.md` | `Version: v1.2.0-scale`, `Generated: 2026-05-01` | `Version: v1.2.1-scale`, `Generated: 2026-05-20` |

`README.md` already carried `v1.2.1-scale` in its version badge (the audit's initial reading was confused by the shields.io URL-encoded double hyphen).

### Class C — Missing content (added)

| Document | Added |
|---|---|
| `examples/README.md` | Rows for `consciousness_grr_inference.esk`, `selene_agent.esk`, `selene_tools.esk`, `milli_mag_bohrification.esk` (the v1.2.1 physical-constants demo); corrected the in-browser claim that conflated repository contents with the website snapshot. |
| `tests/stress/README.md` | Removed the row for `stress_long_subprocess.esk` — the harness does not exist in the tree. |
| `tools/icc_extras/README.md` | Added `codegen_audit.py` and `audit_baseline.json` to the layout table. |
| `.icc/README.md` | Added rows for the two ICC artefacts produced by this audit (`eshkol_doc_intel.md`, `eshkol_arch_summary.md`) and their regeneration commands. |

### Class D — Carried, not yet corrected

Two findings were verified but deferred:

- **`tests/fuzz/README.md`**: documents four harnesses, three of which are explicitly marked `TODO`. This is not an error — the table is structurally honest — but a future pass should either implement the three missing drivers or remove them from the table until they land.
- **`docs/breakdown/SCHEME_COMPATIBILITY.md`**: states "missing 12 procedures" without enumeration. A future revision should add the explicit list. The current text is not wrong, only undercited.
- **`docs/breakdown/OVERVIEW.md`**: still uses the figure "75+ ML builtins" in several paragraphs; the AD-node-count and codegen-module-count claims have been grounded, but a full count of the ML surface is left for a follow-up.

### Class E — False findings (no action)

Two audit-agent findings were rejected on verification:

- **`docs/breakdown/BYTECODE_VM.md`** was flagged as describing absent code; `lib/backend/eshkol_vm.c` and the family of `vm_*.c` / `weight_*.c` / `eskb_*.c` files exist and back the document.
- **`docs/breakdown/CI_CD.md`** was flagged for describing a non-existent GitLab CI pipeline; `.gitlab-ci.yml` exists at the repository root.

Both files were left intact.

## Audit deliverables

- This report (`docs/2026_DOCUMENTATION_AUDIT_REPORT.md`).
- `.icc/eshkol_doc_intel.md` — frozen snapshot of the doc-intelligence run that drove the audit.
- `.icc/eshkol_arch_summary.md` — frozen architecture snapshot used as the comparison anchor.
- Approximately 25 corrected documents across the top-level, `docs/`, `docs/breakdown/`, `docs/vision/`, `docs/superpowers/`, `examples/`, `tests/`, `tools/icc_extras/`, and `.icc/` trees.

## Suggested follow-up

1. Run `icc doc-intelligence --repo eshkol_lang` again after these corrections land and compare against the cached snapshot to confirm that the *unresolved-reference* count drops.
2. Schedule a register-level prose pass on `docs/breakdown/OVERVIEW.md` and `docs/vision/TECHNICAL_WHITE_PAPER.md`: both are well above the project's normal density of marketing prose, and the white paper carries 323 unsupported claims (it is a high-level synthesis rather than an implementation specification and would benefit from an explicit disclaimer pointing at `ADDENDUM_TECHNICAL_WHITE_PAPER_V1.md`).
3. Complete (or formally close) the LLVM 21 toolchain unification plan by either landing the deferred scripting / Docker / validation pieces or rotating the document into `docs/superpowers/completed/` with a closeout note.
