# shellcheck shell=bash
# wasm_vm_sources.sh — the ONE list of translation units the bytecode VM needs
# when it is linked for WebAssembly, plus the leaf runtime symbols that are
# deliberately left unresolved.
#
# WHY THIS IS A SHARED FILE
# -------------------------
# The VM is a C unity build (lib/backend/eshkol_vm.c #includes the rest), but
# several runtime helpers it calls live in their own translation units under
# lib/core.  Under `-s ERROR_ON_UNDEFINED_SYMBOLS=0` — which the WASM links
# need, because a few genuine leaf dependencies have no WASM implementation —
# a TU that is left OUT of the link does not fail: emscripten substitutes an
# ABORTING STUB, the C header declaration still compiles, and the omission is
# invisible until a browser hits `(char-alphabetic? #\a)` and the whole module
# dies with "Aborted(missing function: ...)".
#
# That is exactly what happened once the VM's char/string-case predicates moved
# onto eshkol_unicode_*: the execute-and-diff lane had learned to link lib/core
# and kept working, while the browser bundle — whose build command lived only
# as prose in CONTRIBUTING.md — silently shipped the stubs.  Two build recipes
# for the same VM is the defect; this file is the single list both use.
#
# Consumers:
#   scripts/build-wasm-repl.sh        — site/static/eshkol-vm.{js,wasm}
#   scripts/run_wasm_differential.sh  — the CI execute-and-diff module
#
# Requires: REPO_ROOT set by the caller.

: "${REPO_ROOT:?wasm_vm_sources.sh: REPO_ROOT must be set before sourcing}"

ESHKOL_WASM_VM_SOURCES=(
    "$REPO_ROOT/lib/backend/vm_wasm_repl.c"
    "$REPO_ROOT/lib/core/unicode.cpp"
    "$REPO_ROOT/lib/core/platform_runtime.cpp"
    "$REPO_ROOT/lib/core/model_io_atomic.c"
    "$REPO_ROOT/lib/core/tensor_validation.cpp"
    "$REPO_ROOT/lib/core/tensor_cross_entropy.c"
    # The GPU seam for the wasm target (ADR-0029): the VM's tensor natives
    # reach eshkol_gpu_* exactly as on native; in the browser the WebGPU
    # bridge serves them, and without it every call takes the CPU path.
    "$REPO_ROOT/lib/backend/gpu/gpu_memory_webgpu.cpp"
    # eshkol_printf, which the GPU backend reports allocation failures through.
    "$REPO_ROOT/lib/core/logger.cpp"
)

# Compile/link flags every WASM VM build uses (both consumers above).
#   -fwasm-exceptions -sSUPPORT_LONGJMP=wasm: the VM's error recovery uses
#     setjmp/longjmp. Emscripten's default lowers that to invoke_* calls that
#     round-trip through JavaScript, and JSPI cannot suspend across a JS frame,
#     so a GPU readback inside a guarded evaluation would trap. Native wasm
#     exception handling keeps the whole evaluation in wasm frames (ADR-0029).
ESHKOL_WASM_VM_FLAGS=(
    -DESHKOL_VM_WASM
    -DESHKOL_VM_NO_DISASM
    -DESHKOL_GPU_ENABLED
    -fwasm-exceptions
    -sSUPPORT_LONGJMP=wasm
)

# Leaf runtime dependencies with no WASM implementation.  These become
# aborting stubs ON PURPOSE: a program that calls one fails loudly instead of
# mis-executing, which is the documented contract.  The list is an ALLOWLIST —
# scripts/build-wasm-repl.sh fails the build on anything not named here, so a
# newly introduced runtime dependency has to be either linked or consciously
# added, never silently turned into an abort.
ESHKOL_WASM_ALLOWED_UNDEFINED=(
    eshkol_capability_require
    eshkol_linear_solve
    eshkol_qrng_double
    eshkol_qrng_uint64
)
