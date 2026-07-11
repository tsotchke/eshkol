#!/usr/bin/env bash
# ESH-0103: nested-expression compile time must scale ~linearly, not O(n^2).
#
# A deeply-nested arithmetic expression -- (+ 1 (+ 1 ... 0)) -- compiles to a
# SINGLE LLVM function whose size (basic blocks + allocas) is linear in the
# nesting depth, because arithmetic codegen inlines the full generic
# numeric-tower dispatch (AD tape / Taylor / bignum / rational / tensor /
# complex / double / int) per operator. LLVM's per-function optimization
# passes -- SROA above all, and the whole -O2 pipeline -- are super-linear in
# single-function size, so AOT compile time grew as ~O(depth^2).
#
# Locus (see the ESH-0103 report):
#   - quadratic phase : optimizeModule()  (lib/backend/llvm_codegen.cpp)
#   - dominant pass   : SROA  (confirmed by per-pass ablation)
#   - driver          : per-node alloca/basic-block bloat emitted by
#                       ArithmeticCodegen (lib/backend/arithmetic_codegen.cpp)
#
# This gate measures the AOT compile time of the nested expression at two
# depths and asserts the growth is sub-quadratic. For a chain of length n:
#   - perfectly linear   -> t(2n)/t(n) ~= 2.0
#   - perfectly quadratic -> t(2n)/t(n) ~= 4.0
# We fail if the ratio exceeds RATIO_MAX (i.e. the curve is closer to
# quadratic than to linear). The gate uses `-n -O0 -c` to isolate the
# nested-expression codegen from stdlib regeneration and to match the
# historically-documented ESH-0103 numbers (depth 500 ~= 8.3s pre-fix).
#
# Pass criteria:
#   - t(D2)/t(D1) < RATIO_MAX          (growth is sub-quadratic / ~linear)
#   - t(D2)       < ABS_CEIL_SECONDS   (absolute blow-up safety net)
set -euo pipefail

ESHKOL_RUN="${1:-${ESHKOL_RUN:-}}"
if [ -z "$ESHKOL_RUN" ]; then
    if [ -x "./build/eshkol-run" ]; then
        ESHKOL_RUN="./build/eshkol-run"
    elif [ -x "./build-verify/eshkol-run" ]; then
        ESHKOL_RUN="./build-verify/eshkol-run"
    else
        echo "FAIL: nested_expr_compile_time_test could not locate eshkol-run" >&2
        exit 1
    fi
fi

# Tunables (overridable via env for local experimentation).
D1="${ESH0103_D1:-250}"
D2="${ESH0103_D2:-500}"
# Linear=2.0, quadratic=4.0 for a 2x depth step. Threshold sits between the
# two with margin; the pre-fix curve measured ~3.2 (quadratic), a linearised
# codegen measures ~2.0-2.3.
RATIO_MAX="${ESH0103_RATIO_MAX:-2.7}"
# Generous absolute ceiling for the larger depth so a total blow-up (e.g. the
# -O2 default path, or a fresh regression) trips the gate even if the ratio
# math is confused by noise. Sized well above a healthy compile.
ABS_CEIL_SECONDS="${ESH0103_ABS_CEIL_SECONDS:-30}"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/esh0103.XXXXXX")"
cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

gen_nested() {
    # $1 = depth, $2 = output path. Emits (+ 1 (+ 1 ... 0)) of the given depth.
    python3 - "$1" "$2" <<'PY'
import sys
depth = int(sys.argv[1])
out = sys.argv[2]
with open(out, "w") as f:
    f.write("(+ 1 " * depth)
    f.write("0")
    f.write(")" * depth)
    f.write("\n")
PY
}

compile_time() {
    # $1 = depth. Prints wall-clock seconds for `-n -O0 -c` compile.
    local depth="$1"
    local src="$WORK_DIR/d${depth}.esk"
    local obj="$WORK_DIR/d${depth}.o"
    gen_nested "$depth" "$src"
    local start end
    start="$(python3 -c 'import time; print(time.time())')"
    if ! "$ESHKOL_RUN" -n -O0 -c "$src" -o "$obj" >"$WORK_DIR/d${depth}.log" 2>&1; then
        echo "FAIL: compile of depth-${depth} nested expression failed" >&2
        cat "$WORK_DIR/d${depth}.log" >&2
        exit 1
    fi
    end="$(python3 -c 'import time; print(time.time())')"
    python3 -c "print(f'{$end - $start:.3f}')"
}

echo "ESH-0103 nested-expr compile-time gate"
echo "  compiler : $ESHKOL_RUN"
echo "  depths   : $D1 -> $D2   (ratio must be < $RATIO_MAX)"

T1="$(compile_time "$D1")"
T2="$(compile_time "$D2")"

echo "  t($D1) = ${T1}s"
echo "  t($D2) = ${T2}s"

# Guard against degenerate (too-fast-to-measure) timings that make the ratio
# meaningless: require the smaller depth to take at least a few hundred ms.
RESULT="$(python3 - "$T1" "$T2" "$RATIO_MAX" "$ABS_CEIL_SECONDS" <<'PY'
import sys
t1, t2, ratio_max, abs_ceil = (float(x) for x in sys.argv[1:5])
if t1 < 0.20:
    # Too fast to form a reliable ratio; only enforce the absolute ceiling.
    print(f"SKIP-RATIO {t2 <= abs_ceil}")
else:
    ratio = t2 / t1
    ok = (ratio < ratio_max) and (t2 <= abs_ceil)
    print(f"RATIO {ratio:.3f} {ok}")
PY
)"

echo "  scaling  : $RESULT"

verdict="$(echo "$RESULT" | awk '{print $NF}')"
if [ "$verdict" != "True" ]; then
    echo "FAIL: nested-expression compile time scales super-linearly (ESH-0103)." >&2
    echo "      t($D2)/t($D1) is closer to quadratic (4x) than linear (2x)," >&2
    echo "      or t($D2)=${T2}s exceeds the ${ABS_CEIL_SECONDS}s absolute ceiling." >&2
    echo "      Root cause: optimizeModule()/SROA over the single giant function" >&2
    echo "      that ArithmeticCodegen emits for the nested expression." >&2
    exit 1
fi

echo "PASS: nested-expression compile time scales sub-quadratically."
exit 0
