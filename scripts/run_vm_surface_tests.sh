#!/bin/bash
# Compile and execute deterministic extended-surface probes on the hosted VM.

set -euo pipefail
export LC_ALL=C
export LC_CTYPE=C
export LANG=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
ESHKOL_RUN="$ROOT_DIR/$BUILD_DIR/eshkol-run"
VM="$ROOT_DIR/$BUILD_DIR/eshkol-vm-standalone-test"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-vm-surface.XXXXXX")"
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

TESTS=(
    tests/vm/geometric_fallback_numeric_regression.esk
    tests/vm/riemannian_adam_state_regression.esk
    tests/vm/kb_factor_graph_extensions_regression.esk
    tests/vm/workspace_introspection_regression.esk
    tests/vm/ad_tape_lowlevel_regression.esk
    tests/vm/vm_kb_tensor_test.esk
)

# Every *_surface_regression probe is deterministic and self-checking.  Keep
# this glob in the executable gate (and mirrored in language_coverage.py) so a
# newly added VM surface test cannot sit dormant and manufacture token-only
# coverage credit.
for source_file in "$ROOT_DIR"/tests/vm/*_surface_regression.esk; do
    [ -f "$source_file" ] || continue
    TESTS+=("${source_file#"$ROOT_DIR"/}")
done

run_guarded() { # seconds command...
    perl -e 'my $s=shift; eval { local $SIG{ALRM}=sub{ exit 124 }; alarm $s; exec @ARGV or exit 127; }' \
        "$1" "${@:2}"
}

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
    if ! ESHKOL_VM_NO_DISASM=1 run_guarded 20 "$VM" "$module" >"$output" 2>&1; then
        echo "FAIL"
        tail -80 "$output"
        exit 1
    fi
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
