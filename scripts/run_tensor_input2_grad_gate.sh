#!/usr/bin/env bash
# run_tensor_input2_grad_gate.sh — ESH-0212 tensor-AD second-operand gate.
#
# Runs tests/ad/tensor_input2_grad_test.esk under BOTH the JIT (-r) and AOT and
# checks that every second-operand gradient (matmul B, conv2d kernel,
# scaled-dot-attention K and V, and VECTOR/per-feature batch-norm & layer-norm
# gamma) matches a central finite-difference oracle in ALL THREE calling forms:
# a literal lambda, a first-class variable, and a higher-order wrapper.
#
# Guards two regressions that ESH-0212 fixed:
#   * first-class tensor loss silently returning #(0 0 0 ...) (defect 1), and
#   * a vector learnable gamma silently returning zero gradient (defect 2).
#
# Usage: scripts/run_tensor_input2_grad_gate.sh [--no-aot]
set -u

export LC_ALL=C
export LC_CTYPE=C
export LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TEST_FILE="$REPO_ROOT/tests/ad/tensor_input2_grad_test.esk"

: "${ESHKOL_JIT_CACHE_DIR:=${TMPDIR:-/tmp}/eshkol-tensor-input2-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *)  ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_tensor_input2_grad_gate.sh: $BUILD_DIR/eshkol-run not found — run" \
         "\`cmake --build $BUILD_DIR --target eshkol-run stdlib -j\` first." >&2
    exit 2
fi

DO_AOT=1
for arg in "$@"; do
    case "$arg" in
        --no-aot) DO_AOT=0 ;;
        *) echo "run_tensor_input2_grad_gate.sh: unknown argument: $arg" >&2; exit 2 ;;
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

# ok (0) iff the final PASS line, a "N passed, 0 failed" summary, and no
# FAIL:/crash markers are present.
check_output() {
    local out="$1"
    printf '%s' "$out" | grep -q '^PASS: tensor_input2_grad_test' || return 1
    printf '%s' "$out" | grep -qE 'Results: [0-9]+ passed, 0 failed' || return 1
    printf '%s' "$out" | grep -qE '^FAIL:|fatal signal|LLVM module verification failed' && return 1
    return 0
}

overall=PASS

echo "== ESH-0212 tensor-AD second-operand gate =="

# ---- JIT (-r) ----
jout="$(run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$TEST_FILE" -L"$REPO_ROOT/$BUILD_DIR" 2>&1)"
if check_output "$jout"; then
    jpass="$(printf '%s' "$jout" | grep -oE 'Results: [0-9]+ passed' | grep -oE '[0-9]+')"
    echo "  JIT  PASS ($jpass checks)"
else
    echo "  JIT  FAIL"
    printf '%s\n' "$jout" | grep -E '^FAIL:|fatal signal|failed$' | head
    overall=FAIL
fi

# ---- AOT ----
if [ "$DO_AOT" -eq 1 ]; then
    bin="$(mktemp "${TMPDIR:-/tmp}/tensor_input2_gate_bin.XXXXXX")"
    cout="$(run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$TEST_FILE" -o "$bin" -L"$REPO_ROOT/$BUILD_DIR" 2>&1)"; crc=$?
    if [ "$crc" -ne 0 ] || [ ! -x "$bin" ]; then
        echo "  AOT  COMPILE-FAIL rc=$crc"
        overall=FAIL
    else
        aout="$(run_guarded "$AOT_RUN_TIMEOUT" "$bin" 2>&1)"
        if check_output "$aout"; then
            apass="$(printf '%s' "$aout" | grep -oE 'Results: [0-9]+ passed' | grep -oE '[0-9]+')"
            echo "  AOT  PASS ($apass checks)"
        else
            echo "  AOT  FAIL"
            printf '%s\n' "$aout" | grep -E '^FAIL:|fatal signal|failed$' | head
            overall=FAIL
        fi
    fi
    rm -f "$bin"
fi

echo "ESH-0212 tensor-AD second-operand gate: $overall"
[ "$overall" = "PASS" ] && exit 0 || exit 1
