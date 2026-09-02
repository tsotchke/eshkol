#!/usr/bin/env bash
# Run the ESKM v1 model-load oracle through native JIT/AOT and VM source/ESKB.

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
SELF_TEST=0
if [ "${1:-}" = "--self-test" ]; then SELF_TEST=1; shift; fi
ESHKOL_RUN="${1:-$BUILD_DIR/eshkol-run}"
VM="${2:-$BUILD_DIR/eshkol-vm-standalone-test}"
SOURCE="$ROOT_DIR/tests/core/eskm_v1_model_load_parity.esk"
FIXTURES="$ROOT_DIR/tests/core/fixtures/eskm-v1"
TIMEOUT_SECONDS="${ESKM_PARITY_TIMEOUT:-120}"

if [ ! -x "$ESHKOL_RUN" ] || [ ! -x "$VM" ]; then
    echo "INFRA: expected executable eshkol-run and VM paths" >&2
    exit 127
fi

if ! python3 "$ROOT_DIR/scripts/check_eskm_v1_fixtures.py"; then
    echo "FAIL: ESKM v1 fixture integrity check failed" >&2
    exit 1
fi

source "$ROOT_DIR/scripts/lib/harness_outcome.sh"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-eskm-v1.XXXXXX")" || exit 125
trap 'rm -rf -- "$WORK_DIR"' EXIT HUP INT TERM

report_failure() {
    local rc="$1" outcome
    shift
    outcome="$(eshkol_outcome_classify_exit "$rc")"
    echo "$outcome: $* (rc=$rc)" >&2
}

prepare_axis() {
    local axis_dir="$WORK_DIR/$1"
    mkdir "$axis_dir" || return 1
    cp "$FIXTURES"/*.eskm "$axis_dir/" || return 1
    if [ "$SELF_TEST" -eq 1 ]; then
        cp "$axis_dir/bad-magic.eskm" "$axis_dir/ordinary-2x3.eskm" || return 1
    fi
}

run_axis() {
    local axis="$1" axis_dir rc
    axis_dir="$WORK_DIR/$axis"
    shift
    prepare_axis "$axis" || { echo "INFRA: $axis setup failed" >&2; return 125; }
    if (cd "$axis_dir" && eshkol_outcome_guarded "$TIMEOUT_SECONDS" "$@") \
            >"$axis_dir/stdout" 2>"$axis_dir/stderr"; then
        rc=0
    else
        rc=$?
    fi

    if [ "$SELF_TEST" -eq 1 ]; then
        if [ "$rc" -eq 1 ] && grep -q '^ESKM-V1-MODEL-LOAD:FAIL$' "$axis_dir/stdout"; then
            echo "PASS: $axis negative control made the oracle fail"
            return 0
        fi
        report_failure "$rc" "$axis negative control was not detected"
    elif [ "$rc" -eq 0 ] &&
         [ "$(grep -c '^ESKM-V1-MODEL-LOAD:PASS$' "$axis_dir/stdout")" -eq 1 ] &&
         ! grep -q '^FAIL:' "$axis_dir/stdout" &&
         cmp -s "$axis_dir/ordinary-2x3.eskm" "$axis_dir/actual-ordinary-2x3.eskm" &&
         cmp -s "$axis_dir/rank8.eskm" "$axis_dir/actual-rank8.eskm" &&
         cmp -s "$axis_dir/multi-tensor.eskm" "$axis_dir/actual-multi-tensor.eskm"; then
        echo "PASS: $axis accepted, rejected, and exactly rewrote the expected fixtures"
        return 0
    else
        report_failure "$rc" "$axis compatibility oracle failed"
    fi
    sed -n '1,80p' "$axis_dir/stdout" >&2
    sed -n '1,80p' "$axis_dir/stderr" >&2
    return 1
}

FAILED=0
run_axis jit "$ESHKOL_RUN" --no-stdlib -r "$SOURCE" || FAILED=1

AOT_BIN="$WORK_DIR/eskm-v1-aot"
if eshkol_outcome_guarded "$TIMEOUT_SECONDS" "$ESHKOL_RUN" --no-stdlib \
        "$SOURCE" -o "$AOT_BIN" >"$WORK_DIR/aot-compile.out" 2>"$WORK_DIR/aot-compile.err"; then
    run_axis aot "$AOT_BIN" || FAILED=1
else
    rc=$?
    report_failure "$rc" "AOT compile failed"
    sed -n '1,80p' "$WORK_DIR/aot-compile.err" >&2
    FAILED=1
fi

export ESHKOL_VM_NO_DISASM=1
run_axis vm-source "$VM" "$SOURCE" || FAILED=1

ESKB="$WORK_DIR/eskm-v1.eskb"
if eshkol_outcome_guarded "$TIMEOUT_SECONDS" "$ESHKOL_RUN" --profile hosted-vm \
        --emit-eskb "$ESKB" "$SOURCE" >"$WORK_DIR/eskb-compile.out" 2>"$WORK_DIR/eskb-compile.err" &&
        [ -s "$ESKB" ]; then
    run_axis vm-bytecode "$VM" "$ESKB" || FAILED=1
else
    rc=$?
    report_failure "$rc" "ESKB compile failed or wrote no bytecode"
    sed -n '1,80p' "$WORK_DIR/eskb-compile.err" >&2
    FAILED=1
fi

if [ "$FAILED" -ne 0 ]; then exit 1; fi
if [ "$SELF_TEST" -eq 1 ]; then
    echo "PASS: ESKM v1 parity negative control"
else
    echo "PASS: ESKM v1 model-load parity (jit/aot/vm-source/vm-bytecode)"
fi
