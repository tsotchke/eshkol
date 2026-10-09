# LLVM verifier coverage — audit notes

`llvm::verifyModule` runs at three sites on the paths that produce program
IR, plus one for the opt-in TensorCore adapter module. Sites are named by
function rather than line number, because line numbers in `llvm_codegen.cpp`
move with every change:

| Site | Path | Always-on? | What it covers |
|------|------|------------|----------------|
| `lib/backend/llvm_codegen.cpp`, `EshkolLLVMCodeGen::generateIR` | every IR emit | Yes | Every AOT/JIT/library IR-emit. The single canonical verifier — both `eshkol_generate_llvm_ir` and `eshkol_generate_llvm_ir_library` route through `generateIR()`, so this catches all IR before it leaves the codegen layer. On failure it reports `LLVM module verification failed: …`; with `ESHKOL_DUMP_IR_ON_VERIFY_FAIL` set it prints the whole module first. |
| `lib/backend/llvm_codegen.cpp`, `eshkol_compile_llvm_ir_to_object` | object emission | Debug only (`#ifndef NDEBUG`) | Belt-and-braces re-verify before object emission. Redundant with site #1 because the IR isn't mutated between them, hence the NDEBUG gate is fine for release-build performance. |
| `lib/repl/repl_jit.cpp`, `ReplJITContext::addModule` | REPL JIT path | Yes | Verifies modules generated for live-eval before they reach the LLJIT. Prints the module and fails on error (REPL doesn't want to silently mis-execute). |
| `lib/backend/tensorcore_codegen.cpp`, `verifyTensorcoreAdapterModule` | TensorCore adapter (`ESHKOL_TENSORCORE_ENABLED=ON`) | Called by its test | Verifies the adapter module; exercised by `tests/backend/tensorcore_codegen_test.cpp`. |

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
  the actual predecessor (the floor/ceil/round/truncate class)
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
  every `codegen<X>` method, optionally gated behind an
  `ESHKOL_AGGRESSIVE_VERIFY=1` env var (Planned; not implemented at
  v1.3.6-evolve). This catches per-function faults at emission rather than at
  end-of-module.
