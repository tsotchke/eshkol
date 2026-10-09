#!/usr/bin/env bash

# Deterministic R7RS conformance probes. Every source in tests/r7rs is
# compiled and executed, so language-surface coverage cannot be earned by a
# dormant test file.

set -euo pipefail
export LC_ALL=C
export LC_CTYPE=C
export LANG=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
ESHKOL_RUN="$ROOT_DIR/$BUILD_DIR/eshkol-run"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-r7rs.XXXXXX")"
cleanup() { rm -rf -- "$RUN_DIR"; }
trap cleanup EXIT

# Shared wall-clock guard (scripts/lib/harness_outcome.sh): exits 124 on
# timeout and stops the command together with every process it started,
# so none of them can keep the output pipe open after the deadline.
. "$ROOT_DIR/scripts/lib/harness_outcome.sh"
run_guarded() { eshkol_outcome_guarded "$@"; } # seconds cmd...

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "eshkol-run missing under $BUILD_DIR" >&2
    exit 2
fi

echo "========================================="
echo "  Eshkol R7RS Conformance Tests"
echo "========================================="

passed=0
for source_file in "$ROOT_DIR"/tests/r7rs/*.esk; do
    [ -f "$source_file" ] || continue
    stem="$(basename "$source_file" .esk)"
    binary="$RUN_DIR/$stem"
    compile_log="$RUN_DIR/$stem.compile.log"
    output="$RUN_DIR/$stem.out"
    printf "Testing %-50s " "$stem"

    if ! "$ESHKOL_RUN" -L"$ROOT_DIR/$BUILD_DIR" -o "$binary" "$source_file" >"$compile_log" 2>&1; then
        echo "COMPILE FAIL"
        tail -80 "$compile_log"
        exit 1
    fi
    if ! run_guarded 20 "$binary" >"$output" 2>&1; then
        echo "RUNTIME FAIL"
        tail -80 "$output"
        exit 1
    fi
    if grep -Eq '(^|[[:space:]:])FAIL([[:space:]:]|$)|ERROR:' "$output"; then
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
