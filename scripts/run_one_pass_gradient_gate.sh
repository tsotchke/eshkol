#!/usr/bin/env bash
# run_one_pass_gradient_gate.sh — AD staged-kernel Phase A gate.
#
# Runs tests/ad/one_pass_gradient_test.esk under BOTH the JIT (-r) and AOT and
# checks that a scalar-loss-of-vector gradient is ONE-PASS:
#   * exact values (analytic 2w and central finite differences), and
#   * the AD counters prove primal_calls == 1, reverse_passes == 1,
#     tape_allocations == 1, finite_difference_evals == 0 (for N = 4 and N = 64) —
#     i.e. NOT the old per-component N-primal / N-reverse replay.
#
# Usage: scripts/run_one_pass_gradient_gate.sh [--no-aot]
set -u

export LC_ALL=C
export LC_CTYPE=C
export LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TEST_FILE="$REPO_ROOT/tests/ad/one_pass_gradient_test.esk"

: "${ESHKOL_JIT_CACHE_DIR:=${TMPDIR:-/tmp}/eshkol-one-pass-grad-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *)  ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_one_pass_gradient_gate.sh: $BUILD_DIR/eshkol-run not found — run" \
         "\`cmake --build $BUILD_DIR --target eshkol-run stdlib -j\` first." >&2
    exit 2
fi

DO_AOT=1
for arg in "$@"; do
    case "$arg" in
        --no-aot) DO_AOT=0 ;;
        *) echo "run_one_pass_gradient_gate.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

JIT_TIMEOUT="${JIT_TIMEOUT:-180}"
AOT_COMPILE_TIMEOUT="${AOT_COMPILE_TIMEOUT:-300}"
AOT_RUN_TIMEOUT="${AOT_RUN_TIMEOUT:-60}"

# macOS has no `timeout(1)`; emulate with perl alarm (exit 142 on SIGALRM).
run_guarded() {
    perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $ARGV[0]: $!\n"' \
        "$1" "${@:2}"
}

# ok (0) iff the final ALL PASS line and a "0 failed" summary are present and no
# [FAIL]/crash markers appear.
check_output() {
    local out="$1"
    printf '%s' "$out" | grep -q '^one_pass_gradient_test: ALL PASS' || return 1
    printf '%s' "$out" | grep -qE '^Failed: 0$' || return 1
    printf '%s' "$out" | grep -qE '\[FAIL\]|fatal signal|LLVM module verification failed' && return 1
    return 0
}

echo "== AD Phase A one-pass gradient gate =="
overall=PASS

# ---- JIT ----
jout="$(run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$TEST_FILE" -L"$REPO_ROOT/$BUILD_DIR" 2>&1)"
if check_output "$jout"; then
    jpass="$(printf '%s' "$jout" | grep -oE 'Passed: [0-9]+' | grep -oE '[0-9]+')"
    echo "  JIT  PASS ($jpass checks)"
else
    echo "  JIT  FAIL"
    printf '%s\n' "$jout" | grep -E '\[FAIL\]|fatal signal' | head
    overall=FAIL
fi

# ---- AOT ----
if [ "$DO_AOT" -eq 1 ]; then
    bin="$(mktemp "${TMPDIR:-/tmp}/one_pass_grad_gate_bin.XXXXXX")"
    run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$TEST_FILE" -o "$bin" -L"$REPO_ROOT/$BUILD_DIR" >/dev/null 2>&1; crc=$?
    if [ "$crc" -ne 0 ] || [ ! -x "$bin" ]; then
        echo "  AOT  COMPILE-FAIL rc=$crc"
        overall=FAIL
    else
        aout="$(run_guarded "$AOT_RUN_TIMEOUT" "$bin" 2>&1)"
        if check_output "$aout"; then
            apass="$(printf '%s' "$aout" | grep -oE 'Passed: [0-9]+' | grep -oE '[0-9]+')"
            echo "  AOT  PASS ($apass checks)"
        else
            echo "  AOT  FAIL"
            printf '%s\n' "$aout" | grep -E '\[FAIL\]|fatal signal' | head
            overall=FAIL
        fi
    fi
    rm -f "$bin"
fi

echo "AD Phase A one-pass gradient gate: $overall"
[ "$overall" = PASS ]
