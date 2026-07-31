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

# ICC evidence.  These probes are the SELF-CHECKING half of the VM surface:
# each asserts against R7RS (or against a closed form) inside one run, so
# unlike the native-vs-VM differential in scripts/run_vm_parity.sh a defect
# SHARED by both back ends cannot pass here by agreement.  That is why the
# release oracle reads this trace as well as the parity one — most visibly for
# the 549-check numeric tag-dispatch probe, whose whole point is that an
# integral-valued flonum must not come back exact.
TRACE_DIR="$ROOT_DIR/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/vm_surface.jsonl"
mkdir -p "$TRACE_DIR"
: "${TRACE_FILE:?}"
: > "$TRACE_FILE"

emit_event() { # name PASS|FAIL snippet
    python3 -c '
import json, sys
print(json.dumps({"kind": "vm_surface", "name": sys.argv[1], "value": sys.argv[2],
                  "snippet": sys.argv[3], "confidence": 0.95}, ensure_ascii=False))
' "$1" "$2" "$3" >> "${TRACE_FILE:?}"
}

echo "========================================="
echo "  Eshkol VM Extended Surface Tests"
echo "========================================="

if [ ! -x "$ESHKOL_RUN" ] || [ ! -x "$VM" ]; then
    echo "eshkol-run or eshkol-vm-standalone-test missing under $BUILD_DIR" >&2
    exit 2
fi

passed=0
failed=0
for relative in "${TESTS[@]}"; do
    source_file="$ROOT_DIR/$relative"
    stem="$(basename "$relative" .esk)"
    module="$RUN_DIR/$stem.eskb"
    output="$RUN_DIR/$stem.out"
    printf "Testing %-54s " "$stem"
    "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$module" "$source_file" \
        >"$RUN_DIR/$stem.compile.out" 2>&1 || true
    verdict=PASS
    detail="self-checking VM surface probe green"
    if ! ESHKOL_VM_NO_DISASM=1 run_guarded 20 "$VM" "$module" >"$output" 2>&1; then
        verdict=FAIL
        detail="$(tail -c 200 "$output")"
    elif grep -Eq '(^|[[:space:]:])FAIL([[:space:]:]|$)|ERROR:|unhandled native call' "$output"; then
        verdict=FAIL
        detail="$(grep -Em3 '(^|[[:space:]:])FAIL([[:space:]:]|$)|ERROR:|unhandled native call' "$output" | tr '\n' ' ')"
    fi
    emit_event "$stem" "$verdict" "$detail"
    if [ "$verdict" = "PASS" ]; then
        echo "PASS"
        echo "PASSED $relative::vm-surface"
        passed=$((passed + 1))
    else
        # Every probe runs: the trace is the evidence, and stopping at the
        # first failure would report one defect and hide the rest.
        echo "FAIL"
        echo "FAILED $relative::vm-surface — $detail"
        tail -80 "$output"
        failed=$((failed + 1))
    fi
done

echo ""
echo "Passed: $passed"
echo "Failed: $failed"

if [ "$failed" -eq 0 ]; then
    emit_event "vm_surface_regression_suite" PASS \
        "$passed/$passed VM extended-surface probes green (tests/vm/*_surface_regression.esk, compiled to .eskb and executed on the standalone VM)"
else
    emit_event "vm_surface_regression_suite" FAIL \
        "$failed of $((passed + failed)) VM extended-surface probes failed"
fi
echo "Trace written: $TRACE_FILE"
[ "$failed" -eq 0 ] || exit 1
