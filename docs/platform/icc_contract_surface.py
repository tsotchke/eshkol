"""ICC-readable runtime contract manifest for Eshkol.

This module is not imported by Eshkol. It exists so ICC can extract the
first-party runtime contract surface without crawling generated or vendored
trees. Keep values aligned with docs/platform/ICC_CONTRACT_SURFACE.md.
"""

runtime_backend_values = ["native", "cpu", "gpu", "metal", "cuda", "fallback"]
runtime_provider_values = ["llvm", "vm", "aot", "xla", "metal", "gpu"]

grsi_state = {
    "eshkol_runtime_state": "producer",
    "eshkol_backend_status": "producer",
    "eshkol_compiler_step": "producer",
    "eshkol_checkpoint_status": "producer",
    "eshkol_test_result": "producer",
    "eshkol_completion_status": "producer",
}

checkpoint_artifacts = ["state/eshkol-toolchain-checkpoint.ckpt"]

runtime_artifacts = [
    "reports/eshkol-runtime-smoke.json",
    "reports/eshkol-aot-smoke.json",
]

completion_gates = [
    "eshkol-runtime-contract-complete",
    "eshkol-noesis-link-contract-complete",
    "eshkol-grsi-substrate-complete",
]

completion_done_when = {
    "eshkol-runtime-contract-complete": "native VM, AOT, GPU/Metal, and static-library paths each produce a passing runtime smoke",
    "eshkol-noesis-link-contract-complete": "Noesis links against the current Eshkol compiler and libeshkol-static without ABI drift",
    "eshkol-grsi-substrate-complete": "ICC sees backend, state, checkpoint, artifact, completion, and test contracts for the Eshkol repo",
}
