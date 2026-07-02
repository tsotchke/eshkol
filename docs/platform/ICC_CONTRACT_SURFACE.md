# Eshkol ICC Runtime Contract Surface

This file is an ICC-readable manifest for Eshkol's first-party runtime surface.
It does not replace Eshkol source or tests. It gives ICC stable contract handles
for the runtime substrate that Tsotchke-chan and Noesis depend on during
Geometric Recursive Self-Improvement.

Runtime backend selector values: "native" "cpu" "gpu" "metal" "cuda" "fallback".

Runtime backend provider values: "llvm" "vm" "aot" "xla" "metal" "gpu".

GRSI state projection:
{
  "eshkol_runtime_state": "producer",
  "eshkol_backend_status": "producer",
  "eshkol_compiler_step": "producer",
  "eshkol_checkpoint_status": "producer",
  "eshkol_test_result": "producer",
  "eshkol_completion_status": "producer"
}

Checkpoint artifact policy writes "state/eshkol-toolchain-checkpoint.ckpt" only
for bounded compiler/runtime snapshots; checkpoint promotion requires a matching
test result and an artifact hash.

Runtime artifact policy writes "reports/eshkol-runtime-smoke.json" after each
native runtime smoke, and writes "reports/eshkol-aot-smoke.json" after each AOT
runtime smoke.

Completion gate "eshkol-runtime-contract-complete" is done when the native VM,
AOT, GPU/Metal, and static-library paths each produce a passing runtime smoke.

Completion gate "eshkol-noesis-link-contract-complete" is done when Noesis links
against the current Eshkol compiler and libeshkol-static without ABI drift.

Completion gate "eshkol-grsi-substrate-complete" is done when ICC sees backend,
state, checkpoint, artifact, completion, and test contracts for the Eshkol repo.
