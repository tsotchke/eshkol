#!/bin/bash
# Compile and execute deterministic extended-surface probes on the hosted VM.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
ESHKOL_RUN="$ROOT_DIR/$BUILD_DIR/eshkol-run"
VM="$ROOT_DIR/$BUILD_DIR/eshkol-vm-standalone-test"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-vm-surface.XXXXXX")"
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

TESTS=(
    tests/vm/geometric_surface_regression.esk
    tests/vm/geometric_fallback_numeric_regression.esk
    tests/vm/riemannian_adam_state_regression.esk
    tests/vm/kb_factor_graph_extensions_regression.esk
    tests/vm/workspace_introspection_regression.esk
    tests/vm/ad_tape_lowlevel_regression.esk
    tests/vm/vm_kb_tensor_test.esk
    tests/vm/numeric_alias_surface_regression.esk
    tests/vm/event_emitter_surface_regression.esk
    tests/vm/parameter_runtime_surface_regression.esk
)

echo "========================================="
echo "  Eshkol VM Extended Surface Tests"
echo "========================================="

if [ ! -x "$ESHKOL_RUN" ] || [ ! -x "$VM" ]; then
    echo "eshkol-run or eshkol-vm-standalone-test missing under $BUILD_DIR" >&2
    exit 2
fi

passed=0
for relative in "${TESTS[@]}"; do
    source_file="$ROOT_DIR/$relative"
    stem="$(basename "$relative" .esk)"
    module="$RUN_DIR/$stem.eskb"
    output="$RUN_DIR/$stem.out"
    printf "Testing %-54s " "$stem"
    "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$module" "$source_file" \
        >"$RUN_DIR/$stem.compile.out" 2>&1
    ESHKOL_VM_NO_DISASM=1 "$VM" "$module" >"$output" 2>&1
    if grep -Eq '(^|[[:space:]:])FAIL([[:space:]:]|$)|ERROR:|unhandled native call' "$output"; then
        echo "FAIL"
        tail -80 "$output"
        exit 1
    fi
    echo "PASS"
    passed=$((passed + 1))
done

echo ""
echo "Passed: $passed"
echo "Failed: 0"
