# LLVM verifier coverage — audit notes

`llvm::verifyModule(*module, &error_stream)` runs at three sites in the
codebase, covering the three code paths that produce LLVM IR:

| Site | Path | Always-on? | What it covers |
|------|------|------------|----------------|
| `lib/backend/llvm_codegen.cpp:2240` | `EshkolLLVMCodeGen::generateIR` | Yes | Every AOT/JIT/library IR-emit. The single canonical verifier — both `eshkol_generate_llvm_ir` (line 31611) and `eshkol_generate_llvm_ir_library` (line 31634) route through `generateIR()`, so this catches all IR before it leaves the codegen layer. |
| `lib/backend/llvm_codegen.cpp:31804` | `eshkol_compile_llvm_ir_to_object` | Debug only (`#ifndef NDEBUG`) | Belt-and-braces re-verify before object emission. Redundant with site #1 because the IR isn't mutated between them, hence the NDEBUG gate is fine for release-build performance. |
| `lib/repl/repl_jit.cpp:843` | REPL JIT path | Yes | Verifies modules generated for live-eval before they reach the LLJIT. Throws on failure (REPL doesn't want to silently mis-execute). |

## Coverage summary

- **AOT (eshkol-run -o foo / eshkol-run foo.esk → a.out)**: site #1
  (always) + site #2 (debug only).
- **`--shared-lib` (stdlib.o build)**: site #1 (always).
- **REPL JIT (`eshkol-run -r foo.esk` / `eshkol-repl`)**: site #1
  (always — JIT generates IR through the same `generateIR()`) + site #3
  (always, REPL-specific verify before JIT submission).
- **Browser/WASM target**: site #1 still applies (the WASM target shares
  the IR generation pipeline; only the final object emission differs).

## What we did NOT change

The audit found the existing coverage is complete. No new verifier sites
needed. The NDEBUG gate at site #2 is a deliberate optimisation — module
hasn't been mutated since site #1 ran, so re-verification adds latency
without catching anything new in release builds.

## Failure modes the verifier catches

For reference (so reviewers know what would land if site #1 were ever
disabled or moved):

- malformed PHI (predecessor block list doesn't match incoming-value list)
- `addIncoming(value, named_block)` where the block named is no longer
  the actual predecessor (the floor/ceil/round/truncate class — see
  `MEMORY.md`)
- type mismatches in `InsertValue` / `ExtractValue` (the
  tagged-value-data-field-{4} class)
- function definitions referencing values from other functions
- unreachable code with side effects
- undef poisoning a uniquely-typed value before SSA construction

## Recommendation for v1.3

Site #1 is the gatekeeper. Treat it as a load-bearing invariant:

- The `verify_module_drop` rule in `codegen_audit_rules.json` was removed
  (over-noisy — every reference was a finding, not a problem). Instead,
  any PR that *removes* a `verifyModule` call should be flagged for
  review by the cross-file checker.
- A future improvement: also call `verifyFunction(*func)` at the end of
  every `codegen<X>` method, optionally gated behind
  `ESHKOL_AGGRESSIVE_VERIFY=1` env var. This catches per-function bugs at
  emission rather than at end-of-module.
