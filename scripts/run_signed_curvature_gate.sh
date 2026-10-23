#!/usr/bin/env bash
# run_signed_curvature_gate.sh — producer for the ADR-0018 criterion
# (v1.4-connection: signed-curvature stereographic geometry).
#
# Usage: scripts/run_signed_curvature_gate.sh            (BUILD_DIR, default build)
#
# PASS requires, in one run against the same build:
#   1. signed_curvature_reference_grid_test in binary128 mode: the f64 shared
#      core against the independent binary128 reference at every finite binade
#      of K of both signs (all subnormal binades, DBL_TRUE_MIN, DBL_MAX, 0),
#      plus the committed golden subset re-derived bit for bit, plus the
#      identity checks (jets through K = 0, exp/log, transport isometry,
#      squared-distance gradient at coincidence, attention continuity and flat
#      reduction, convention and migration). "Results: N passed, 0 failed".
#   2. tests/geometry/signed_curvature_test.esk (core.curvature) under the JIT
#      AND compiled AOT: "RESULT: ALL PASS" and "Failed: 0" in both.
#   3. tests/vm/signed_curvature_surface_regression.esk on the bytecode VM:
#      every assertion PASS (the VM ops reproduce the same reference values).
#
# A build without binary128 arithmetic runs the golden-subset mode; that is
# reported as FAIL with the reason, never as the all-binade PASS.
#
# Writes one ICC runtime_event to scripts/icc_traces/v14_signed_curvature.jsonl:
#   {"kind":"geometry_exact","name":"v14_signed_curvature_reference_grid","value":"PASS"|"FAIL",...}
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
. "$REPO_ROOT/scripts/lib/checked_write.sh"
TRACE_FILE="$TRACE_DIR/v14_signed_curvature.jsonl"
mkdir -p "$TRACE_DIR"
eshkol_require_output_file_path "$TRACE_FILE"
: > "$TRACE_FILE"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ABS_BUILD="$BUILD_DIR" ;;
    *)  ABS_BUILD="$REPO_ROOT/$BUILD_DIR" ;;
esac
ESHKOL_RUN="$ABS_BUILD/eshkol-run"
VM="$ABS_BUILD/eshkol-vm-standalone-test"
GRID="$ABS_BUILD/signed_curvature_reference_grid_test"
PROGRAM="$REPO_ROOT/tests/geometry/signed_curvature_test.esk"
VMPROG="$REPO_ROOT/tests/vm/signed_curvature_surface_regression.esk"
WORK="$ABS_BUILD/signed-curvature-gate"
mkdir -p "$WORK"
export ESHKOL_JIT_CACHE=0 ESHKOL_LIB_DIR="$ABS_BUILD"

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}
emit() { # PASS|FAIL snippet
    eshkol_require_output_file_path "$TRACE_FILE"
    printf '{"kind":"geometry_exact","name":"v14_signed_curvature_reference_grid","value":"%s","snippet":"%s","confidence":0.95}\n' \
        "$1" "$(json_escape "$2")" >> "$TRACE_FILE"
    echo "$1: v14_signed_curvature_reference_grid ($2)"
}

for f in "$GRID" "$ESHKOL_RUN" "$VM"; do
    if [ ! -x "$f" ]; then emit FAIL "missing $f (build first)"; exit 1; fi
done

status=0
notes=""

# 1. binary128 reference grid
gout="$("$GRID" 2>&1)"; grc=$?
gline="$(printf '%s\n' "$gout" | grep -m1 '^GRID ')"
gres="$(printf '%s\n' "$gout" | grep -m1 '^Results: ')"
if [ "$grc" -ne 0 ] || ! printf '%s' "$gres" | grep -q ', 0 failed$'; then
    status=1; notes="$notes grid: $gres $(printf '%s\n' "$gout" | grep -m3 '^FAIL' | tr '\n' ' ');"
elif ! printf '%s' "$gline" | grep -q '^GRID binary128 '; then
    status=1; notes="$notes grid ran in golden-subset mode (no binary128 arithmetic on this build): $gline;"
else
    notes="$notes $gres; $gline;"
fi

# 2. core.curvature on JIT and AOT
verdict_ok() {
    printf '%s' "$1" | grep -q '^RESULT: ALL PASS$' || return 1
    printf '%s' "$1" | grep -q '^Failed: 0$' || return 1
    printf '%s' "$1" | grep -qE '^FAIL:|fatal signal|LLVM module verification failed' && return 1
    return 0
}
jout="$("$ESHKOL_RUN" -r "$PROGRAM" -L"$ABS_BUILD" 2>&1)"; jrc=$?
if [ "$jrc" -ne 0 ] || ! verdict_ok "$jout"; then
    status=1; notes="$notes JIT: $(printf '%s\n' "$jout" | grep -m3 -E '^FAIL:|error|RESULT' | tr '\n' ' ');"
else
    notes="$notes JIT $(printf '%s\n' "$jout" | grep -m1 '^Passed:');"
fi
if "$ESHKOL_RUN" -O 1 -o "$WORK/signed_curvature_aot" "$PROGRAM" -L"$ABS_BUILD" > "$WORK/aot.compile" 2>&1; then
    aout="$("$WORK/signed_curvature_aot" 2>&1)"; arc=$?
    if [ "$arc" -ne 0 ] || ! verdict_ok "$aout"; then
        status=1; notes="$notes AOT: $(printf '%s\n' "$aout" | grep -m3 -E '^FAIL:|RESULT' | tr '\n' ' ');"
    else
        notes="$notes AOT $(printf '%s\n' "$aout" | grep -m1 '^Passed:');"
    fi
else
    status=1; notes="$notes AOT compile failed: $(grep -m3 -i error "$WORK/aot.compile" | tr '\n' ' ');"
fi

# 3. the VM ops
if "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$WORK/signed_curvature_vm.eskb" "$VMPROG" > "$WORK/vm.compile" 2>&1 &&
   vout="$(ESHKOL_VM_NO_DISASM=1 "$VM" "$WORK/signed_curvature_vm.eskb" 2>&1)"; then
    vpass="$(printf '%s\n' "$vout" | grep -c '^PASS: ')"
    if printf '%s\n' "$vout" | grep -qE '^FAIL|ERROR:|unhandled native call' || [ "$vpass" -eq 0 ]; then
        status=1; notes="$notes VM: $(printf '%s\n' "$vout" | grep -m3 -E '^FAIL|ERROR' | tr '\n' ' ');"
    else
        notes="$notes VM $vpass PASS;"
    fi
else
    status=1; notes="$notes VM run failed: $(grep -m2 -i error "$WORK/vm.compile" | tr '\n' ' ');"
fi

if [ "$status" -eq 0 ]; then emit PASS "${notes# }"; else emit FAIL "${notes# }"; fi
exit "$status"
