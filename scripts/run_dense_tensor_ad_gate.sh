#!/usr/bin/env bash
# run_dense_tensor_ad_gate.sh — the gate behind ADR-0002 Position A: a dense
# tensor op records ONE AD tape node, and it computes the SAME gradient the
# scalarizing lowering did.
#
# WHY THIS SCRIPT EXISTS
#
# The dense tensor AD node had never executed. Its admitting guard was
# unsatisfiable — `autodiff_ && ad_mode && !after_matmul_compute`, where
# `after_matmul_compute` is assigned non-null under exactly `autodiff_ &&
# ad_mode` — and forcing it SIGSEGV'd on the first matmul gradient
# (.icc/silent-wrong-ledger.yaml SW-48). What ran instead scalarized every
# matmul into 2*M*N*K scalar tape nodes.
#
# Replacing an execution model that produces CORRECT answers carries exactly
# one risk worth gating: that the replacement produces different ones. So this
# gate runs the SAME program under both lowerings and parses the printed
# gradients as numeric vectors:
#
#   ESHKOL_DENSE_TENSOR_AD_NODES=1   one dense node per tensor op (shipped)
#   ESHKOL_DENSE_TENSOR_AD_NODES=0   the scalarizing lowering (the oracle)
#
# The variable is read at CODEGEN time, so the two runs are two different
# emitted programs, not one program taking two branches. Numeric agreement
# demonstrates that packing changes the REPRESENTATION of a gradient and never
# its value.
#
# The gate also requires the cost to have actually dropped and the dense count
# to be exactly the documented four nodes for the 6x6 case.
#
# Usage: scripts/run_dense_tensor_ad_gate.sh [--no-aot]
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

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_dense_tensor_ad_gate.sh: $BUILD_DIR/eshkol-run not found — run" \
         "\`cmake --build $BUILD_DIR --target eshkol-run stdlib -j\` first." >&2
    exit 2
fi

DO_AOT=1
for arg in ${@+"$@"}; do
    case "$arg" in
        --no-aot) DO_AOT=0 ;;
        *) echo "run_dense_tensor_ad_gate.sh: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

: "${ESHKOL_JIT_CACHE_DIR:=$BUILD_PATH/.dense-tensor-ad-jit-cache}"
export ESHKOL_JIT_CACHE_DIR
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

JIT_TIMEOUT="${JIT_TIMEOUT:-300}"
AOT_COMPILE_TIMEOUT="${AOT_COMPILE_TIMEOUT:-360}"
AOT_RUN_TIMEOUT="${AOT_RUN_TIMEOUT:-180}"

# macOS has no timeout(1); emulate with perl alarm (exit 142 on SIGALRM).
run_guarded() {
    perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $ARGV[0]: $!\n"' \
        "$1" "${@:2}"
}

TEST=tests/ad/dense_tensor_ad_gradcheck_test.esk
MARKER=dense_tensor_ad_gradcheck_test

overall=PASS
fail() { echo "  $1"; overall=FAIL; }

check_output() {
    local out="$1"
    printf '%s' "$out" | grep -q "^${MARKER}: ALL PASS" || return 1
    printf '%s' "$out" | grep -qE '^Failed: 0$' || return 1
    printf '%s' "$out" | grep -qE '\[FAIL\]|fatal signal|LLVM module verification failed' && return 1
    return 0
}

# The dense lowering emits a DIFFERENT program, and the JIT object cache is
# keyed on the source, not on this variable — so each lowering gets its own
# cache directory rather than reusing the other's compiled objects.
run_lowering() {
    local mode="$1"   # 1 = dense, 0 = scalarizing
    local out
    # Assignments before a shell function are not necessarily exported to the
    # function's exec'd child. Export explicitly: otherwise both invocations
    # can silently inherit the previous mode and the differential is invalid.
    export ESHKOL_DENSE_TENSOR_AD_NODES="$mode"
    export ESHKOL_JIT_CACHE_DIR="$ESHKOL_JIT_CACHE_DIR/mode$mode"
    out="$(run_guarded "$JIT_TIMEOUT" "$ESHKOL_RUN" -r "$REPO_ROOT/$TEST" -L"$BUILD_PATH" 2>&1)"
    printf '%s' "$out"
}

echo "== dense tensor AD gate (ADR-0002 Position A, SW-48) =="

mkdir -p "$ESHKOL_JIT_CACHE_DIR/mode1" "$ESHKOL_JIT_CACHE_DIR/mode0"

dense_out="$(run_lowering 1)"
scalar_out="$(run_lowering 0)"

if check_output "$dense_out"; then
    echo "  dense lowering       JIT  PASS ($(printf '%s' "$dense_out" | grep -oE 'Passed: [0-9]+' | grep -oE '[0-9]+') checks)"
else
    fail "dense lowering       JIT  FAIL"
    printf '%s\n' "$dense_out" | grep -E '\[FAIL\]|mismatch|fatal signal' | head -20
fi

if check_output "$scalar_out"; then
    echo "  scalarizing oracle   JIT  PASS ($(printf '%s' "$scalar_out" | grep -oE 'Passed: [0-9]+' | grep -oE '[0-9]+') checks)"
else
    fail "scalarizing oracle   JIT  FAIL"
    printf '%s\n' "$scalar_out" | grep -E '\[FAIL\]|mismatch|fatal signal' | head -20
fi

# ---- the two lowerings must agree numerically -------------------------------
dense_grads="$(printf '%s\n' "$dense_out" | grep '^GRAD ' | sort)"
scalar_grads="$(printf '%s\n' "$scalar_out" | grep '^GRAD ' | sort)"

if [ -z "$dense_grads" ] || [ -z "$scalar_grads" ]; then
    fail "no GRAD lines captured — the differential proved nothing"
elif python3 -c '
import math, sys
def parse(s):
    out = {}
    for line in s.splitlines():
        p = line.split()
        if len(p) < 3 or p[0] != "GRAD":
            continue
        out[p[1]] = [float(x) for x in p[2:]]
    return out
a, b = parse(sys.argv[1]), parse(sys.argv[2])
if set(a) != set(b):
    raise SystemExit("gradient labels differ")
for k in sorted(a):
    if len(a[k]) != len(b[k]):
        raise SystemExit(f"gradient lengths differ for {k}")
    for i, (x, y) in enumerate(zip(a[k], b[k])):
        if not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9):
            raise SystemExit(f"gradient mismatch {k}[{i}]: {x} != {y}")
' "$dense_grads" "$scalar_grads"; then
    echo "  differential         both lowerings agree numerically within 1e-9 ($(printf '%s\n' "$dense_grads" | wc -l | tr -d ' ') gradients)"
else
    fail "differential         THE TWO LOWERINGS DISAGREE"
    diff <(printf '%s\n' "$scalar_grads") <(printf '%s\n' "$dense_grads") | head -30
fi

# ---- and the dense tape must actually be smaller ---------------------------
# Read the 6x6 count, the shape where the difference is largest: 2*216 = 432
# scalar nodes for the multiply-accumulates alone.
dense_6x6="$(printf '%s\n' "$dense_out"  | sed -nE 's/.*tape_nodes 2x2=[0-9]+ 6x6=([0-9]+).*/\1/p' | head -n 1)"
scalar_6x6="$(printf '%s\n' "$scalar_out" | sed -nE 's/.*tape_nodes 2x2=[0-9]+ 6x6=([0-9]+).*/\1/p' | head -n 1)"

if [ -z "${dense_6x6:-}" ] || [ -z "${scalar_6x6:-}" ]; then
    fail "could not read the 6x6 tape_nodes counts"
elif [ "$dense_6x6" -ne 4 ]; then
    fail "tape size            dense 6x6 expected exactly 4 nodes, got $dense_6x6"
elif [ "$dense_6x6" -lt "$scalar_6x6" ]; then
    echo "  tape size            6x6 matmul: $scalar_6x6 nodes scalarized -> $dense_6x6 dense"
else
    fail "tape size            dense ($dense_6x6) is not smaller than scalarized ($scalar_6x6)"
fi

# ---- AOT agrees with JIT ---------------------------------------------------
if [ "$DO_AOT" -eq 1 ]; then
    bin="$(mktemp "$BUILD_PATH/dense_tensor_ad_gate_bin.XXXXXX")"
    export ESHKOL_DENSE_TENSOR_AD_NODES=1
    run_guarded "$AOT_COMPILE_TIMEOUT" "$ESHKOL_RUN" "$REPO_ROOT/$TEST" -o "$bin" \
        -L"$BUILD_PATH" >/dev/null 2>&1; rc=$?
    if [ "$rc" -ne 0 ] || [ ! -x "$bin" ]; then
        fail "dense lowering       AOT  COMPILE-FAIL rc=$rc"
    else
        aot_out="$(run_guarded "$AOT_RUN_TIMEOUT" "$bin" 2>&1)"
        if check_output "$aot_out"; then
            echo "  dense lowering       AOT  PASS ($(printf '%s' "$aot_out" | grep -oE 'Passed: [0-9]+' | grep -oE '[0-9]+') checks)"
        else
            fail "dense lowering       AOT  FAIL"
            printf '%s\n' "$aot_out" | grep -E '\[FAIL\]|mismatch|fatal signal' | head -20
        fi
        aot_grads="$(printf '%s\n' "$aot_out" | grep '^GRAD ' | sort)"
        if [ "$aot_grads" = "$dense_grads" ]; then
            echo "  AOT/JIT              identical gradients"
        else
            fail "AOT/JIT              AOT and JIT disagree"
            diff <(printf '%s\n' "$dense_grads") <(printf '%s\n' "$aot_grads") | head -20
        fi
    fi
    rm -f "$bin"
fi

echo "dense tensor AD gate: $overall"
[ "$overall" = PASS ]
