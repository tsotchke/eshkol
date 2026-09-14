#!/usr/bin/env bash
# run_math_example_gate.sh — verdict gate for one mathematics example program.
#
# Usage: scripts/run_math_example_gate.sh <program-basename> <event-kind> <event-name> [--no-aot]
#
# Runs examples/<program-basename>.esk under BOTH the JIT (-r) and AOT and
# requires the program's own verdict line "RESULT: ALL PASS" with
# "Failed: 0" in each mode (every such program exits nonzero on any failed
# check and carries a negative control per claim). On success it writes an
# ICC runtime_event to scripts/icc_traces/<event-name>.jsonl:
#   {"kind":"<event-kind>","name":"<event-name>","value":"PASS",...}
# consumed by .icc/completion-oracles.yaml.
set -u

export LC_ALL=C
export LC_CTYPE=C
export LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <program-basename> <event-kind> <event-name> [--no-aot]" >&2
    exit 2
fi
PROGRAM_NAME="$1"; EVENT_KIND="$2"; EVENT_NAME="$3"; shift 3
PROGRAM="$REPO_ROOT/examples/$PROGRAM_NAME.esk"
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
TRACE_FILE="$TRACE_DIR/$EVENT_NAME.jsonl"
mkdir -p "$TRACE_DIR"
: > "$TRACE_FILE"
if [ ! -f "$PROGRAM" ]; then
    echo "run_math_example_gate.sh: $PROGRAM not found" >&2
    exit 2
fi

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run"; ABS_BUILD="$BUILD_DIR" ;;
    *)  ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run"; ABS_BUILD="$REPO_ROOT/$BUILD_DIR" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_math_example_gate.sh: $BUILD_DIR/eshkol-run not found — build first." >&2
    exit 2
fi

: "${ESHKOL_JIT_CACHE_DIR:=$ABS_BUILD/math-gate-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

DO_AOT=1
for arg in "$@"; do
    case "$arg" in
        --no-aot) DO_AOT=0 ;;
        *) echo "run_math_example_gate.sh: unknown argument: $arg" >&2; exit 2 ;;
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
    echo "FAIL: JIT run of $PROGRAM_NAME (exit $jrc)"; printf '%s\n' "$jout" | grep -E '^FAIL:|fatal signal|RESULT:' | head
    status=1
fi
jpass="$(printf '%s' "$jout" | grep -oE '^Passed: [0-9]+' | grep -oE '[0-9]+')"

apass=""
if [ "$DO_AOT" -eq 1 ] && [ "$status" -eq 0 ]; then
    AOT_BIN="$ABS_BUILD/${PROGRAM_NAME}_gate_aot"
    if "$ESHKOL_RUN" -o "$AOT_BIN" "$PROGRAM" >/dev/null 2>&1; then
        aout="$("$AOT_BIN" 2>&1)"; arc=$?
        if [ "$arc" -ne 0 ] || ! verdict_ok "$aout"; then
            echo "FAIL: AOT run of $PROGRAM_NAME (exit $arc)"; printf '%s\n' "$aout" | grep -E '^FAIL:|fatal signal|RESULT:' | head
            status=1
        fi
        apass="$(printf '%s' "$aout" | grep -oE '^Passed: [0-9]+' | grep -oE '[0-9]+')"
    else
        echo "FAIL: AOT compile of $PROGRAM_NAME"; status=1
    fi
fi

if [ "$status" -eq 0 ]; then
    snippet="$PROGRAM_NAME: JIT passed=${jpass:-?} AOT passed=${apass:-skipped}; RESULT: ALL PASS in every mode"
    printf '{"kind":"%s","name":"%s","value":"PASS","snippet":"%s","confidence":0.95}\n' \
        "$EVENT_KIND" "$EVENT_NAME" "$(json_escape "$snippet")" >> "$TRACE_FILE"
    echo "PASS: $EVENT_NAME ($snippet)"
else
    printf '{"kind":"%s","name":"%s","value":"FAIL","snippet":"see gate output","confidence":0.95}\n' \
        "$EVENT_KIND" "$EVENT_NAME" >> "$TRACE_FILE"
fi
exit "$status"
