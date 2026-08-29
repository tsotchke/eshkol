#!/usr/bin/env bash
# run_ad_exactness_gate.sh — the gate behind "no finite-difference fallback
# anywhere in the gradient path", plus the tape-size ratchet ADR-0002 needs.
#
# WHY THIS SCRIPT EXISTS
#
# Eshkol asserted its exactness guarantee as `(= (ad-finite-difference-evals) 0)`
# in tests/ad/one_pass_gradient_test.esk. That assertion was VACUOUS: the
# counter's only writer, eshkol_ad_count_fd(), had zero callers on the native
# back end, so `finite_difference_evals` could not become nonzero and the check
# was true by construction — green even if a finite-difference fallback were
# introduced the next day. The counter is now wired (see the
# `(ad-note-finite-difference!)` builtin and lib/core/ad/tape.esk), and this
# gate runs the assertion together with the negative control that proves it can
# still go red.
#
# WHAT IT RUNS
#
#   POSITIVE   scripts/run_one_pass_gradient_gate.sh — an exact vector gradient
#              is one primal / one reverse / one tape / ZERO finite differences.
#              (That script had no callers of its own until this one; the Phase A
#              gate was written and then never wired to anything.)
#
#   NEGATIVE   tests/ad/fd_counter_negative_test.esk — a real finite-difference
#              backward drives the counter above zero, the count equals the
#              perturbations actually evaluated, and the shipped assertion form
#              evaluates #f under exactly those conditions. A gate that cannot
#              fail is not a gate; this is the half that proves it can.
#
#   RATCHET    tests/ad/matmul_tape_node_count_test.esk — matmul gradients stay
#              exact and their AD tape does not grow. This is the instrument for
#              ADR-0000's Stage-7 gate (`scalar_ad_nodes_from_matmul == 0`),
#              which was unmeasurable while nothing counted the nodes.
#
# The negative control runs on BOTH engines. That matters more than it looks:
# the native and bytecode back ends keep SEPARATE finite-difference counters
# (__eshkol_ad_counters.finite_difference_evals vs VM->ad_finite_difference_evals),
# so a wiring that works on one proves nothing about the other. This is exactly
# how the native counter stayed disconnected while the VM's was live.
#
# Usage: scripts/run_ad_exactness_gate.sh [--no-aot]
set -u

export LC_ALL=C
export LC_CTYPE=C
export LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) BUILD_PATH="$BUILD_DIR" ;;
    *)  BUILD_PATH="$REPO_ROOT/$BUILD_DIR" ;;
esac
ESHKOL_RUN="$BUILD_PATH/eshkol-run"
VM_BIN="$BUILD_PATH/eshkol-vm-standalone-test"

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_ad_exactness_gate.sh: $BUILD_DIR/eshkol-run not found — run" \
         "\`cmake --build $BUILD_DIR --target eshkol-run stdlib -j\` first." >&2
    exit 2
fi

DO_AOT=1
# `"$@"` with zero arguments trips `set -u` on bash 3.2 (the /bin/bash macOS
# still ships), so the loop is guarded rather than relying on bash >= 4.4.
for arg in ${@+"$@"}; do
    case "$arg" in
        --no-aot) DO_AOT=0 ;;
        *) echo "run_ad_exactness_gate.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# Keep JIT artefacts inside the build tree unless the caller placed them.
: "${ESHKOL_JIT_CACHE_DIR:=$BUILD_PATH/.ad-exactness-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

JIT_TIMEOUT="${JIT_TIMEOUT:-240}"
AOT_COMPILE_TIMEOUT="${AOT_COMPILE_TIMEOUT:-300}"
AOT_RUN_TIMEOUT="${AOT_RUN_TIMEOUT:-120}"

# macOS has no timeout(1); emulate with perl alarm (exit 142 on SIGALRM).
run_guarded() {
    perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $ARGV[0]: $!\n"' \
        "$1" "${@:2}"
}

overall=PASS
fail() { echo "  $1"; overall=FAIL; }

# ok (0) iff the test's own ALL PASS line and a "0 failed" summary are present
# and no failure/crash marker appears.
check_output() {
    local out="$1" marker="$2"
    printf '%s' "$out" | grep -q "^${marker}: ALL PASS" || return 1
    printf '%s' "$out" | grep -qE '^Failed: 0$' || return 1
    printf '%s' "$out" | grep -qE '\[FAIL\]|fatal signal|LLVM module verification failed' && return 1
    return 0
}

# Same, for the VM, whose display of the summary counters is formatted
# differently; the ALL PASS line and the absence of failure markers are the
# portable part.
check_output_vm() {
    local out="$1" marker="$2"
    printf '%s' "$out" | grep -q "^${marker}: ALL PASS" || return 1
    printf '%s' "$out" | grep -qE '\[FAIL\]|fatal signal|Runtime error' && return 1
    return 0
}

run_case() {
    local label="$1" file="$2" marker="$3"
    local out

    out="$(run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$REPO_ROOT/$file" -L"$BUILD_PATH" 2>&1)"
    if check_output "$out" "$marker"; then
        echo "  $label  JIT  PASS ($(printf '%s' "$out" | grep -oE 'Passed: [0-9]+' | grep -oE '[0-9]+') checks)"
    else
        fail "$label  JIT  FAIL"
        printf '%s\n' "$out" | grep -E '\[FAIL\]|fatal signal' | head
    fi

    if [ "$DO_AOT" -eq 1 ]; then
        local bin rc
        bin="$(mktemp "$BUILD_PATH/ad_exactness_gate_bin.XXXXXX")"
        run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$REPO_ROOT/$file" -o "$bin" \
            -L"$BUILD_PATH" >/dev/null 2>&1; rc=$?
        if [ "$rc" -ne 0 ] || [ ! -x "$bin" ]; then
            fail "$label  AOT  COMPILE-FAIL rc=$rc"
        else
            out="$(run_guarded "$AOT_RUN_TIMEOUT" "$bin" 2>&1)"
            if check_output "$out" "$marker"; then
                echo "  $label  AOT  PASS ($(printf '%s' "$out" | grep -oE 'Passed: [0-9]+' | grep -oE '[0-9]+') checks)"
            else
                fail "$label  AOT  FAIL"
                printf '%s\n' "$out" | grep -E '\[FAIL\]|fatal signal' | head
            fi
        fi
        rm -f "$bin"
    fi
}

echo "== AD exactness gate =="

# ---- POSITIVE: the exact path really is finite-difference free --------------
#
# Capture, then print. Piping the inner gate straight into `sed` for indentation
# would make `if` test SED's exit status, not the gate's — the gate could fail
# and this script would call it a pass. That is the same shape of defect this
# whole file exists to close, so it is spelled out rather than left to a reader
# to notice.
echo "  positive (one-pass exact gradient, finite_difference_evals == 0):"
if [ "$DO_AOT" -eq 1 ]; then
    pos_out="$(BUILD_DIR="$BUILD_DIR" bash "$REPO_ROOT/scripts/run_one_pass_gradient_gate.sh" 2>&1)"; pos_rc=$?
else
    pos_out="$(BUILD_DIR="$BUILD_DIR" bash "$REPO_ROOT/scripts/run_one_pass_gradient_gate.sh" --no-aot 2>&1)"; pos_rc=$?
fi
printf '%s\n' "$pos_out" | sed 's/^/    /'
if [ "$pos_rc" -ne 0 ]; then
    fail "one-pass gradient gate FAIL (rc=$pos_rc)"
fi

# ---- NEGATIVE: the counter is a live instrument, on BOTH engines ------------
echo "  negative (the FD counter fires, and the assertion goes red):"
run_case "fd-counter" tests/ad/fd_counter_negative_test.esk fd_counter_negative_test

if [ -x "$VM_BIN" ]; then
    vmout="$(ESHKOL_VM_NO_DISASM=1 ESHKOL_PATH="$REPO_ROOT/lib" \
             run_guarded "$JIT_TIMEOUT" "$VM_BIN" \
             "$REPO_ROOT/tests/ad/fd_counter_negative_test.esk" 2>&1)"
    if check_output_vm "$vmout" fd_counter_negative_test; then
        echo "  fd-counter  VM   PASS"
    else
        fail "fd-counter  VM   FAIL"
        printf '%s\n' "$vmout" | grep -E '\[FAIL\]|fatal signal|Runtime error' | head
    fi
else
    echo "  fd-counter  VM   SKIP ($VM_BIN not built)"
fi

# ---- RATCHET: matmul gradients stay exact and the tape does not grow --------
echo "  ratchet (matmul AD tape node count):"
run_case "matmul-nodes" tests/ad/matmul_tape_node_count_test.esk matmul_tape_node_count_test

# ---- DENSE PATH: one node per tensor op, and the SAME gradient --------------
#
# The ratchet above measures the tape. This measures the answer: the dense
# lowering (ADR-0002 Position A) and the scalarizing one are run over the same
# gradcheck program and their gradients compared byte-for-byte. Kept as its own
# script because it compiles the program twice, once per lowering.
echo "  dense tensor AD path (ADR-0002 Position A):"
if [ "$DO_AOT" -eq 1 ]; then
    dense_out="$(BUILD_DIR="$BUILD_DIR" bash "$REPO_ROOT/scripts/run_dense_tensor_ad_gate.sh" 2>&1)"; dense_rc=$?
else
    dense_out="$(BUILD_DIR="$BUILD_DIR" bash "$REPO_ROOT/scripts/run_dense_tensor_ad_gate.sh" --no-aot 2>&1)"; dense_rc=$?
fi
printf '%s\n' "$dense_out" | sed 's/^/    /'
if [ "$dense_rc" -ne 0 ]; then
    fail "dense tensor AD gate FAIL (rc=$dense_rc)"
fi

echo "AD exactness gate: $overall"
[ "$overall" = PASS ]
