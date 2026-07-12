#!/bin/bash
# Execute deterministic high-risk extension suites that were previously
# committed but not wired into the complete CI harness.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
ESHKOL_RUN="$ROOT_DIR/$BUILD_DIR/eshkol-run"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-surface-extensions.XXXXXX")"
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

TESTS=(
    tests/dnc/dnc_test.esk
    tests/quant/dequant_test.esk
    tests/sdnc/sdnc_api_test.esk
    tests/v1_3_edge_cases/tensor_dtype_test.esk
    tests/ad/one_pass_gradient_test.esk
    tests/ad/sparse_tensors_test.esk
    tests/ad/taylor_tower_test.esk
)

echo "========================================="
echo "  Eshkol High-Risk Extension Tests"
echo "========================================="

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "eshkol-run not found at $ESHKOL_RUN - build first." >&2
    exit 2
fi

passed=0
for relative in "${TESTS[@]}"; do
    source_file="$ROOT_DIR/$relative"
    stem="$(basename "$relative" .esk)"
    binary="$RUN_DIR/$stem"
    output="$RUN_DIR/$stem.out"
    printf "Testing %-48s " "$stem"
    "$ESHKOL_RUN" -L"$ROOT_DIR/$BUILD_DIR" "$source_file" -o "$binary" \
        >"$RUN_DIR/$stem.compile.out" 2>&1
    "$binary" >"$output" 2>&1
    if grep -Eq '(^|[[:space:]:])FAIL([[:space:]:]|$)|Failed:[[:space:]]+[1-9]|ERROR:' "$output"; then
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
