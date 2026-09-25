#!/usr/bin/env bash
# Gate all six historical fixtures through the ESKM v1 4 x 4 engine matrix.

set -u

# Perl's timeout wrapper must not inherit a locale unavailable on macOS.
export LC_ALL=C LC_CTYPE=C LANG=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
SELF_TEST=0
CONTROL=""
if [ "${1:-}" = "--self-test" ]; then SELF_TEST=1; shift; fi
ESHKOL_RUN="${1:-$BUILD_DIR/eshkol-run}"
VM="${2:-$BUILD_DIR/eshkol-vm-standalone-test}"
SOURCE="$ROOT_DIR/tests/core/eskm_v1_model_load_parity.esk"
FIXTURES="$ROOT_DIR/tests/core/fixtures/eskm-v1"
TIMEOUT_SECONDS="${ESKM_PARITY_TIMEOUT:-120}"
ENGINES=(jit aot vm-source vm-bytecode)
VALID_FIXTURES=(scalar empty-0x3 ordinary-2x3 rank8 multi-tensor large-32x32)
SINGLE_FIXTURES=(scalar empty-0x3 ordinary-2x3 rank8 large-32x32)

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")" ;;
    esac
}

ESHKOL_RUN="$(absolute_path "$ESHKOL_RUN")"
VM="$(absolute_path "$VM")"

if [ ! -x "$ESHKOL_RUN" ] || [ ! -x "$VM" ]; then
    echo "INFRA: expected executable eshkol-run and VM paths" >&2
    exit 127
fi

if ! python3 "$ROOT_DIR/scripts/check_eskm_v1_fixtures.py"; then
    echo "FAIL: ESKM v1 fixture integrity check failed" >&2
    exit 1
fi

source "$ROOT_DIR/scripts/lib/harness_outcome.sh"
# shellcheck source=lib/durable_work_root.sh
source "$ROOT_DIR/scripts/lib/durable_work_root.sh"
if [ -n "${ESHKOL_TEST_TMPDIR:-}" ]; then
    WORK_DIR="$(mktemp -d "$ESHKOL_TEST_TMPDIR/eshkol-eskm-v1.XXXXXX")" || exit 125
elif eshkol_durable_enabled; then
    WORK_DIR="$(eshkol_durable_prepare_dir eskm-v1-model-io-parity)" || exit $?
else
    TMP_BASE="${ESHKOL_TEST_TMP_ROOT:-${TMPDIR:-/tmp}}"
    WORK_DIR="$(mktemp -d "$TMP_BASE/eshkol-eskm-v1.XXXXXX")" || exit 125
fi

cleanup() {
    if [ -z "${ESHKOL_TEST_KEEP_TMPDIR:-}" ] &&
       [ -z "${ESHKOL_DURABLE_WORK_ROOT:-}" ]; then
        rm -rf -- "$WORK_DIR"
    else
        echo "ESKM parity artifacts retained at $WORK_DIR" >&2
    fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

report_failure() {
    local rc="$1" outcome
    shift
    if [ "$rc" -eq 0 ]; then
        outcome=FAIL
    else
        outcome="$(eshkol_outcome_classify_exit "$rc")"
    fi
    echo "$outcome: $* (rc=$rc)" >&2
}

record_failure() {
    if [ "$(eshkol_outcome_classify_exit "$1")" = "INFRA" ]; then
        INFRA=1
    else
        FAILED=1
    fi
}

exit_if_incomplete() {
    if [ "$FAILED" -ne 0 ]; then exit 1; fi
    if [ "$INFRA" -ne 0 ]; then exit 125; fi
}

show_output() {
    local axis_dir="$1"
    sed -n '1,100p' "$axis_dir/stdout" >&2
    sed -n '1,100p' "$axis_dir/stderr" >&2
}

AOT_BIN="$WORK_DIR/eskm-v1-aot"
if eshkol_outcome_guarded "$TIMEOUT_SECONDS" "$ESHKOL_RUN" --no-stdlib \
        "$SOURCE" -o "$AOT_BIN" >"$WORK_DIR/aot-compile.out" \
        2>"$WORK_DIR/aot-compile.err"; then
    :
else
    rc=$?
    report_failure "$rc" "AOT compile failed"
    sed -n '1,100p' "$WORK_DIR/aot-compile.err" >&2
    if [ "$(eshkol_outcome_classify_exit "$rc")" = "INFRA" ]; then exit 125; fi
    exit 1
fi

ESKB="$WORK_DIR/eskm-v1.eskb"
if eshkol_outcome_guarded "$TIMEOUT_SECONDS" "$ESHKOL_RUN" \
        --profile hosted-vm --emit-eskb "$ESKB" "$SOURCE" \
        >"$WORK_DIR/eskb-compile.out" 2>"$WORK_DIR/eskb-compile.err"; then
    if [ ! -s "$ESKB" ]; then
        echo "FAIL: ESKB compile wrote no bytecode" >&2
        exit 1
    fi
else
    rc=$?
    report_failure "$rc" "ESKB compile failed"
    sed -n '1,100p' "$WORK_DIR/eskb-compile.err" >&2
    if [ "$(eshkol_outcome_classify_exit "$rc")" = "INFRA" ]; then exit 125; fi
    exit 1
fi

run_engine() {
    local engine="$1" axis_dir="$2" mode="$3"
    (
        cd "$axis_dir" || exit 125
        export ESKM_PARITY_MODE="$mode"
        export ESKM_PARITY_ENGINE="$engine"
        export ESKM_PARITY_CONTROL="${CONTROL:-}"
        export ESHKOL_VM_NO_DISASM=1
        case "$engine" in
            jit)
                eshkol_outcome_guarded "$TIMEOUT_SECONDS" \
                    "$ESHKOL_RUN" --no-stdlib -r "$SOURCE"
                ;;
            aot)
                eshkol_outcome_guarded "$TIMEOUT_SECONDS" "$AOT_BIN"
                ;;
            vm-source)
                eshkol_outcome_guarded "$TIMEOUT_SECONDS" "$VM" "$SOURCE"
                ;;
            vm-bytecode)
                eshkol_outcome_guarded "$TIMEOUT_SECONDS" "$VM" "$ESKB"
                ;;
            *)
                echo "INFRA: unknown ESKM parity engine: $engine" >&2
                exit 125
                ;;
        esac
    )
}

run_captured() {
    local engine="$1" axis_dir="$2" mode="$3"
    if run_engine "$engine" "$axis_dir" "$mode" \
            >"$axis_dir/stdout" 2>"$axis_dir/stderr"; then
        return 0
    else
        return $?
    fi
}

has_exact_success() {
    local axis_dir="$1" marker="$2"
    [ "$(grep -c "^${marker}$" "$axis_dir/stdout")" -eq 1 ] &&
        ! grep -q '^FAIL:' "$axis_dir/stdout"
}

PRODUCERS="$WORK_DIR/producers"
mkdir "$PRODUCERS" || exit 125
FAILED=0
INFRA=0

for producer in "${ENGINES[@]}"; do
    axis_dir="$WORK_DIR/produce-$producer"
    mkdir "$axis_dir" || exit 125
    cp "$FIXTURES"/*.eskm "$axis_dir/" || exit 125
    if run_captured "$producer" "$axis_dir" produce; then rc=0; else rc=$?; fi
    output="$axis_dir/producer.eskm"
    if [ "$rc" -eq 0 ] &&
       has_exact_success "$axis_dir" "ESKM-V1-PRODUCE:PASS" &&
       [ -s "$output" ] &&
       cmp -s "$FIXTURES/multi-tensor.eskm" "$output"; then
        cp "$output" "$PRODUCERS/producer-$producer.eskm" || exit 125
        for fixture in "${VALID_FIXTURES[@]}"; do
            output="$axis_dir/produce-$fixture.eskm"
            if cmp -s "$FIXTURES/$fixture.eskm" "$output"; then
                cp "$output" "$PRODUCERS/producer-$producer-$fixture.eskm" || exit 125
            else
                report_failure 0 "$producer $fixture producer byte oracle failed"
                record_failure 0
            fi
        done
        echo "PASS: $producer producer metadata checks completed"
    else
        report_failure "$rc" "$producer producer oracle failed"
        show_output "$axis_dir"
        record_failure "$rc"
    fi
done

exit_if_incomplete

prepare_consumer() {
    local axis_dir="$1"
    mkdir "$axis_dir" || return 1
    cp "$FIXTURES"/*.eskm "$axis_dir/" || return 1
    cp "$PRODUCERS"/*.eskm "$axis_dir/" || return 1
}

check_rewrites() {
    local axis_dir="$1" producer fixture scope
    for fixture in "${VALID_FIXTURES[@]}"; do
        cmp -s "$FIXTURES/$fixture.eskm" \
            "$axis_dir/rewrite-$fixture.eskm" || return 1
        for producer in "${ENGINES[@]}"; do
            cmp -s "$FIXTURES/$fixture.eskm" \
                "$axis_dir/rewrite-producer-$producer-$fixture.eskm" || return 1
        done
    done
    for fixture in "${SINGLE_FIXTURES[@]}"; do
        cmp -s "$FIXTURES/$fixture.eskm" \
            "$axis_dir/tensor-rewrite-$fixture.eskm" || return 1
    done
    for scope in function region; do
        for fixture in scalar empty-0x3; do
            cmp -s "$FIXTURES/$fixture.eskm" \
                "$axis_dir/$scope-rewrite-$fixture.eskm" || return 1
        done
    done
    for producer in "${ENGINES[@]}"; do
        cmp -s "$PRODUCERS/producer-$producer.eskm" \
            "$axis_dir/rewrite-producer-$producer.eskm" || return 1
    done
}

run_negative_control() {
    local consumer="$1" kind="$2" expected="$3" axis_dir rc CONTROL=""
    axis_dir="$WORK_DIR/self-test-$kind-$consumer"
    prepare_consumer "$axis_dir" || return 125
    case "$kind" in
        producer)
            cp "$FIXTURES/ordinary-2x3.eskm" \
                "$axis_dir/producer-vm-bytecode.eskm" || return 125
            ;;
        metadata|payload)
            # Change the expectation only: all input fixtures stay valid and unchanged.
            CONTROL="$kind"
            ;;
        malformed)
            cp "$FIXTURES/ordinary-2x3.eskm" \
                "$axis_dir/bad-magic.eskm" || return 125
            ;;
    esac
    if run_captured "$consumer" "$axis_dir" consume; then rc=0; else rc=$?; fi
    if [ "$rc" -eq 1 ] &&
       [ "$(grep -c '^ESKM-V1-CONSUME:FAIL$' "$axis_dir/stdout")" -eq 1 ] &&
       ! grep -q '^ESKM-V1-CONSUME:PASS$' "$axis_dir/stdout" &&
       grep -Fqx "FAIL: $expected" "$axis_dir/stdout"; then
        echo "PASS: $consumer detected the $kind negative control"
        return 0
    fi
    report_failure "$rc" "$consumer missed the $kind negative control"
    show_output "$axis_dir"
    if [ "$(eshkol_outcome_classify_exit "$rc")" = "INFRA" ]; then return 125; fi
    return 1
}

if [ "$SELF_TEST" -eq 1 ]; then
    for consumer in "${ENGINES[@]}"; do
        if run_negative_control "$consumer" producer \
                "producer vm-bytecode list structure"; then :; else
            rc=$?
            record_failure "$rc"
        fi
        if run_negative_control "$consumer" metadata "scalar rank/dimensions"; then :; else
            rc=$?
            record_failure "$rc"
        fi
        if run_negative_control "$consumer" payload "scalar payload values"; then :; else
            rc=$?
            record_failure "$rc"
        fi
        if run_negative_control "$consumer" malformed \
                "reject bad-magic.eskm"; then :; else
            rc=$?
            record_failure "$rc"
        fi
    done
    exit_if_incomplete
    echo "PASS: ESKM v1 matrix negative controls (16 checks: model structure, scalar metadata/payload, malformed rejection)"
    exit 0
fi

for consumer in "${ENGINES[@]}"; do
    axis_dir="$WORK_DIR/consume-$consumer"
    prepare_consumer "$axis_dir" || exit 125
    if run_captured "$consumer" "$axis_dir" consume; then rc=0; else rc=$?; fi
    if [ "$rc" -eq 0 ] &&
       has_exact_success "$axis_dir" "ESKM-V1-CONSUME:PASS" &&
       check_rewrites "$axis_dir"; then
        echo "PASS: $consumer rewrote six goldens and all 24 producer fixtures, five public tensors, four lifetime cases"
    else
        report_failure "$rc" "$consumer compatibility oracle failed"
        show_output "$axis_dir"
        record_failure "$rc"
    fi
done

exit_if_incomplete
echo "PASS: ESKM v1 model-load parity (6 fixtures x 4 producers x 4 consumers; 96 cells; 20 public tensor and 16 lifetime cases)"
