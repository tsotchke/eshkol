#!/usr/bin/env bash
# Standalone hosted compiler regression for the lossless closure-count path.

set -euo pipefail

# FuncChunk retains large legacy compile-time tables for this compatibility
# host; allow those tables to live on the process stack during the regression.
ulimit -s unlimited

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPILER="${1:?usage: $0 <hosted-compiler> <work-dir>}"
WORK_DIR="${2:?usage: $0 <hosted-compiler> <work-dir>}"

mkdir -p "$WORK_DIR"
python3 "$ROOT_DIR/tests/vm_parity/generate_closure_capture_regressions.py" "$WORK_DIR"

SOURCE_FILE="$WORK_DIR/closure_capture_65537.esk"
OUTPUT_FILE="$WORK_DIR/hosted_65537.out"
ERROR_FILE="$WORK_DIR/hosted_65537.err"

"$COMPILER" "$SOURCE_FILE" >"$OUTPUT_FILE" 2>"$ERROR_FILE"
grep -Fqx "  [compiled: 986046 instructions, 328201 constants, 66004 locals]" "$OUTPUT_FILE"
test ! -s "$ERROR_FILE"

echo "PASS: hosted standalone compiler compiled 65537 captures (66004 locals)"
