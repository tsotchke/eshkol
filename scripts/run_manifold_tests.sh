#!/bin/bash
# Compile and run the differential-geometry regression suite.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
ESHKOL_RUN="$ROOT_DIR/$BUILD_DIR/eshkol-run"
TEST_FILE="$ROOT_DIR/tests/manifold/manifold_test.esk"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-manifold-tests.XXXXXX")"
TEST_BIN="$RUN_DIR/manifold_test"
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

echo "========================================="
echo "  Eshkol Differential Geometry Tests"
echo "========================================="

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "eshkol-run not found at $ESHKOL_RUN - build first." >&2
    exit 2
fi

"$ESHKOL_RUN" -L"$ROOT_DIR/$BUILD_DIR" "$TEST_FILE" -o "$TEST_BIN"
"$TEST_BIN"

echo ""
echo "Passed: 1"
echo "Failed: 0"
