#!/usr/bin/env bash
# run_ipm_velocity_expansion_gate.sh — IPM local velocity expansion gate.
#
# Runs examples/mathematics_ipm_velocity_expansion.esk under BOTH the JIT (-r)
# and AOT and requires the program's own verdict line "RESULT: ALL PASS" with
# "Failed: 0" in each mode. The program checks, in exact rational arithmetic,
# that the constants of the local velocity expansion for the 2D incompressible
# porous media equation are the derivatives of the operator symbol, that the
# expansion closes its defining relation order by order, that the truncated
# velocity is divergence free and satisfies Darcy's law to the same order, and
# that an independent order-by-order solve reproduces every constant; it
# carries negative controls for each claim and exits nonzero on any failure.
#
# On success it writes an ICC runtime_event to
#   scripts/icc_traces/ipm_velocity_expansion.jsonl
# consumed by .icc/completion-oracles.yaml
#   (event_kinds: [ipm_local_expansion],
#    event_names: ["ipm_velocity_expansion_closes"], event_values: ["PASS"]).
#
# Usage: scripts/run_ipm_velocity_expansion_gate.sh [--no-aot]
set -u

export LC_ALL=C
export LC_CTYPE=C
export LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
TRACE_FILE="$TRACE_DIR/ipm_velocity_expansion.jsonl"
PROGRAM="$REPO_ROOT/examples/mathematics_ipm_velocity_expansion.esk"
mkdir -p "$TRACE_DIR"
: > "$TRACE_FILE"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run"; ABS_BUILD="$BUILD_DIR" ;;
    *)  ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run"; ABS_BUILD="$REPO_ROOT/$BUILD_DIR" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_ipm_velocity_expansion_gate.sh: $BUILD_DIR/eshkol-run not found — build first." >&2
    exit 2
fi

: "${ESHKOL_JIT_CACHE_DIR:=$ABS_BUILD/ipm-gate-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

DO_AOT=1
for arg in "$@"; do
    case "$arg" in
        --no-aot) DO_AOT=0 ;;
        *) echo "run_ipm_velocity_expansion_gate.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

json_escape() {
    printf '%s' "$1" | perl -0pe 's/\\/\\\\/g; s/"/\\"/g; s/\n/\\n/g; s/\r/\\r/g; s/\t/\\t/g; s/([\x00-\x08\x0b\x0c\x0e-\x1f])/sprintf("\\u%04x", ord($1))/ge'
}

verdict_ok() {
    printf '%s' "$1" | grep -q '^RESULT: ALL PASS$' || return 1
    printf '%s' "$1" | grep -q '^Failed: 0$' || return 1
    printf '%s' "$1" | grep -qE '^FAIL:|fatal signal|LLVM module verification failed' && return 1
    return 0
}

status=0
jout="$("$ESHKOL_RUN" -r "$PROGRAM" 2>&1)"; jrc=$?
if [ "$jrc" -ne 0 ] || ! verdict_ok "$jout"; then
    echo "FAIL: JIT run (exit $jrc)"; printf '%s\n' "$jout" | grep -E '^FAIL:|fatal signal|RESULT:' | head
    status=1
fi
jpass="$(printf '%s' "$jout" | grep -oE '^Passed: [0-9]+' | grep -oE '[0-9]+')"

apass=""
if [ "$DO_AOT" -eq 1 ] && [ "$status" -eq 0 ]; then
    AOT_BIN="$ABS_BUILD/mathematics_ipm_velocity_expansion_gate_aot"
    if "$ESHKOL_RUN" -o "$AOT_BIN" "$PROGRAM" >/dev/null 2>&1; then
        aout="$("$AOT_BIN" 2>&1)"; arc=$?
        if [ "$arc" -ne 0 ] || ! verdict_ok "$aout"; then
            echo "FAIL: AOT run (exit $arc)"; printf '%s\n' "$aout" | grep -E '^FAIL:|fatal signal|RESULT:' | head
            status=1
        fi
        apass="$(printf '%s' "$aout" | grep -oE '^Passed: [0-9]+' | grep -oE '[0-9]+')"
    else
        echo "FAIL: AOT compile"; status=1
    fi
fi

if [ "$status" -eq 0 ]; then
    snippet="JIT passed=${jpass:-?} AOT passed=${apass:-skipped}; RESULT: ALL PASS in every mode"
    printf '{"kind":"ipm_local_expansion","name":"ipm_velocity_expansion_closes","value":"PASS","snippet":"%s","confidence":0.95}\n' \
        "$(json_escape "$snippet")" >> "$TRACE_FILE"
    echo "PASS: ipm_velocity_expansion_closes ($snippet)"
else
    printf '{"kind":"ipm_local_expansion","name":"ipm_velocity_expansion_closes","value":"FAIL","snippet":"see gate output","confidence":0.95}\n' >> "$TRACE_FILE"
fi
exit "$status"
