#!/usr/bin/env bash
# tests/memory/define_loop_flat_rss_aot_test.sh — ESH-0214b AOT flat-RSS
# regression gate.
#
# This AOT gate preserves the original 1M define+guard control and adds
# 10k/50k/100k slope checks for the Eliot named-let list, discarded tensor
# literal, and narrow tensor-dot forms. The slope check compares high-water RSS
# at different pass counts so a short run cannot look flat by itself.
#
# Unlike scripts/run_rss_bounded_test.sh (which gates the JIT `-r` path),
# this gate is AOT-focused: it compiles each source ahead-of-time with
# `eshkol-run <src> -o <bin>`, runs the
# binary directly under `/usr/bin/time` (macOS: `-l`, Linux: `-v`), and fails
# if peak RSS exceeds a generous flat ceiling. The fixed behavior measures
# ~27MB; the broken (pre-fix) behavior measures ~2.6GB for the same
# 1,000,000-iteration program, so a 200MB ceiling cleanly separates the two
# with wide margin in both directions.
#
# A second, advisory-only half recompiles the same source with
# ESHKOL_NO_ITER_SCOPE=1 set at COMPILE time (disabling the fix globally)
# and reports its peak RSS, to demonstrate the gate would actually catch a
# regression. That half is informational (echo only) and never fails the
# gate, to keep CI time bounded.
#
# Usage: tests/memory/define_loop_flat_rss_aot_test.sh [--ceiling-mb N] [--timeout S]
#                                                         [--small N] [--middle N] [--large N]
#   BUILD_DIR env var selects the build directory (default: build).
#   ESHKOL_RUN env var overrides the eshkol-run binary path directly.
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
. "$REPO_ROOT/scripts/lib/durable_work_root.sh"
# shellcheck source=../../scripts/lib/checked_write.sh
. "$REPO_ROOT/scripts/lib/checked_write.sh"

BUILD_DIR="${BUILD_DIR:-build}"
if [ -z "${ESHKOL_RUN:-}" ]; then
    case "$BUILD_DIR" in
        /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
        *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
    esac
fi
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "define_loop_flat_rss_aot_test.sh: $ESHKOL_RUN not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

SRC="$REPO_ROOT/tests/memory/define_loop_flat_rss_aot_test.esk"
NAMED_SRC="$REPO_ROOT/tests/memory/named_let_flat_rss_aot_test.esk"
TENSOR_SRC="$REPO_ROOT/tests/memory/define_loop_discarded_tensor_flat_rss_aot_test.esk"
DOT_SRC="$REPO_ROOT/tests/memory/define_loop_tensor_dot_flat_rss_aot_test.esk"
EXIT_PROBE_SRC="$REPO_ROOT/tests/features/iter_scope_exit_arg_order_test.esk"
for source in "$SRC" "$NAMED_SRC" "$TENSOR_SRC" "$DOT_SRC" "$EXIT_PROBE_SRC"; do
    if [ ! -f "$source" ]; then
        echo "define_loop_flat_rss_aot_test.sh: $source not found." >&2
        exit 2
    fi
done

CEILING_MB=200
TIMEOUT_S=60
SMALL_N=10000
MIDDLE_N=50000
LARGE_N=100000
while [ $# -gt 0 ]; do
    case "$1" in
        --ceiling-mb) shift; CEILING_MB="${1:-$CEILING_MB}" ;;
        --timeout) shift; TIMEOUT_S="${1:-$TIMEOUT_S}" ;;
        --small) shift; SMALL_N="${1:-$SMALL_N}" ;;
        --middle) shift; MIDDLE_N="${1:-$MIDDLE_N}" ;;
        --large) shift; LARGE_N="${1:-$LARGE_N}" ;;
        *) echo "define_loop_flat_rss_aot_test.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# Detect which peak-RSS-reporting `time` flavor is available.
if eshkol_durable_enabled; then
    WORK="$(eshkol_durable_prepare_dir define-loop-flat-rss-aot)" || exit $?
    TIME_PROBE="$WORK/time-probe.log"
else
    TIME_PROBE="/tmp/.deflrat_probe.$$"
fi
# macOS (BSD time): `/usr/bin/time -l` reports "NNNN  maximum resident set size" in BYTES.
# Linux (GNU time):  `/usr/bin/time -v` reports "Maximum resident set size (kbytes): NNNN" in KB.
TIME_MODE=""
if /usr/bin/time -l true >/dev/null 2>"$TIME_PROBE"; then
    if grep -q "maximum resident set size" "$TIME_PROBE" 2>/dev/null; then
        TIME_MODE="bsd"
    fi
fi
eshkol_require_output_file_path "$TIME_PROBE"
if [ -z "$TIME_MODE" ] && /usr/bin/time -v true >"$TIME_PROBE" 2>&1; then
    if grep -qi "Maximum resident set size" "$TIME_PROBE" 2>/dev/null; then
        TIME_MODE="gnu"
    fi
fi
if ! eshkol_durable_enabled; then eshkol_checked_rm "$TIME_PROBE"; fi
if [ -z "$TIME_MODE" ]; then
    echo "define_loop_flat_rss_aot_test.sh: neither \`/usr/bin/time -l\` (macOS) nor \`/usr/bin/time -v\` (Linux) reports peak RSS on this host — cannot gate." >&2
    exit 2
fi

if ! eshkol_durable_enabled; then
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-flat-rss-aot.XXXXXX")"
    trap 'rm -rf "$WORK"' EXIT
fi

# run_aot <src> <bin> <compile_env...=> -> sets compile/run, RSS, arena report and output globals
run_aot() {
    local src="$1" bin="$2"; shift 2
    local compile_log="$WORK/compile_$(basename "$bin").log"
    local run_out="$WORK/run_$(basename "$bin").out"
    local time_log="$WORK/time_$(basename "$bin").log"

    ( cd "$WORK" && env ESHKOL_PATH="$REPO_ROOT/lib" "$@" \
        "$ESHKOL_RUN" "$src" -o "$bin" ) > "$compile_log" 2>&1
    FR_COMPILE_RC=$?
    if [ "$FR_COMPILE_RC" -ne 0 ]; then
        FR_RUN_RC=127
        FR_RSS_MB=0
        FR_ARENA_BYTES=0
        FR_OUT="$compile_log"
        FR_TIME_LOG="$compile_log"
        return
    fi
    chmod +x "$bin"

    if [ "$TIME_MODE" = "bsd" ]; then
        ( cd "$WORK" && ESHKOL_ARENA_REPORT=1 /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
            "$TIMEOUT_S" "$bin" ) > "$run_out" 2> "$time_log"
        FR_RUN_RC=$?
        FR_RSS_MB=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$time_log")
    else
        ( cd "$WORK" && ESHKOL_ARENA_REPORT=1 /usr/bin/time -v perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
            "$TIMEOUT_S" "$bin" ) > "$run_out" 2> "$time_log"
        FR_RUN_RC=$?
        FR_RSS_MB=$(awk -F: '/Maximum resident set size/{printf "%d", $2/1024}' "$time_log")
    fi
    [ -n "$FR_RSS_MB" ] || FR_RSS_MB=0
    FR_ARENA_BYTES=$(awk -F= '/global_total_allocated_bytes/{print $2; exit}' "$time_log")
    [ -n "$FR_ARENA_BYTES" ] || FR_ARENA_BYTES=0
    FR_OUT="$run_out"
    FR_TIME_LOG="$time_log"
}

# run_slope_gate <kind> <source> <counter-name> checks 10x pass-count scaling.
run_slope_gate() {
    local kind="$1" src="$2" counter="$3"
    local small_rss=0 middle_rss=0 large_rss=0 n expected expected_line variant bin output_ok ratio
    echo "--- AOT RSS slope: $kind ($SMALL_N/$MIDDLE_N/$LARGE_N iterations) ---"
    for n in "$SMALL_N" "$MIDDLE_N" "$LARGE_N"; do
        variant="$WORK/${kind}_${n}.esk"
        bin="$WORK/${kind}_${n}_bin"
        sed -E "s/\\(define ${counter} [0-9]+\\)/(define ${counter} ${n})/" "$src" > "$variant"
        if ! grep -Fqx "(define $counter $n)" "$variant"; then
            echo "FAIL: could not substitute $counter=$n in $src"
            return 1
        fi
        run_aot "$variant" "$bin"
        if [ "$FR_COMPILE_RC" -ne 0 ]; then
            echo "FAIL: $kind AOT compile at N=$n (exit=$FR_COMPILE_RC)"
            cat "$FR_OUT"
            return 1
        fi
        if [ "$FR_RUN_RC" -ne 0 ]; then
            echo "FAIL: $kind AOT run at N=$n (exit=$FR_RUN_RC)"
            cat "$FR_OUT"
            return 1
        fi

        expected=$((n * 12))
        output_ok=1
        case "$kind" in
            namedlet)
                expected_line=$(printf 'RESULT\tOK\t%s\t%s' "$n" "$expected")
                grep -Fqx "$expected_line" "$FR_OUT" || output_ok=0
                ;;
            tensor_literal)
                grep -Fqx "passes=$n result=$expected" "$FR_OUT" || output_ok=0
                grep -Fqx "PASS" "$FR_OUT" || output_ok=0
                grep -Fqx "$(printf 'CARRY\tOK')" "$FR_OUT" || output_ok=0
                grep -Fqx "$(printf 'SIDE_EFFECTS\tOK\t4')" "$FR_OUT" || output_ok=0
                ;;
            tensor_dot)
                grep -Fqx "passes=$n result=$expected" "$FR_OUT" || output_ok=0
                grep -Fqx "PASS" "$FR_OUT" || output_ok=0
                grep -Fqx "$(printf 'AD\tOK\t384')" "$FR_OUT" || output_ok=0
                grep -Fqx "$(printf 'DOT_SHADOW\tOK')" "$FR_OUT" || output_ok=0
                grep -Fqx "$(printf 'CALLBACK_EXIT\tOK')" "$FR_OUT" || output_ok=0
                grep -Fqx "$(printf 'CALLBACK_DOT\tOK')" "$FR_OUT" || output_ok=0
                ;;
        esac
        if [ "$output_ok" -ne 1 ]; then
            echo "FAIL: $kind semantic output at N=$n"
            cat "$FR_OUT"
            return 1
        fi
        if [ "$FR_ARENA_BYTES" -gt 10000000 ]; then
            echo "FAIL: $kind exit-time arena use ${FR_ARENA_BYTES}B exceeds 10MB at N=$n"
            return 1
        fi
        echo "  N=$n peak_rss=${FR_RSS_MB}MB arena=${FR_ARENA_BYTES}B checksum=$expected"
        case "$n" in
            "$SMALL_N") small_rss=$FR_RSS_MB ;;
            "$MIDDLE_N") middle_rss=$FR_RSS_MB ;;
            "$LARGE_N") large_rss=$FR_RSS_MB ;;
        esac
    done

    ratio=$(awk -v small="$small_rss" -v large="$large_rss" \
        'BEGIN { if (small < 1) small=1; printf "%.2f", large/small }')
    echo "  peak RSS ratio (large/small)=${ratio}x"
    if [ "$large_rss" -gt "$CEILING_MB" ]; then
        echo "FAIL: $kind N=$LARGE_N RSS ${large_rss}MB exceeds ${CEILING_MB}MB"
        return 1
    fi
    if ! awk -v ratio="$ratio" 'BEGIN { exit !(ratio <= 2.5) }'; then
        echo "FAIL: $kind RSS scales with iteration count (ratio ${ratio}x > 2.5x)"
        return 1
    fi
    echo "PASS: $kind AOT RSS stays flat across the bounded sweep."
}

run_debug_rejections() {
    local src="$1" out="$2"; shift 2
    local log="$WORK/${out}_debug_compile.log"
    ( cd "$WORK" && env ESHKOL_PATH="$REPO_ROOT/lib" \
        "$ESHKOL_RUN" -d "$src" -o "$WORK/${out}_debug_bin" ) > "$log" 2>&1
    if [ $? -ne 0 ]; then
        echo "FAIL: debug compile for $out"
        tail -40 "$log"
        return 1
    fi
    while [ $# -gt 0 ]; do
        if ! grep -Fq "$1" "$log"; then
            echo "FAIL: $out analyzer did not report '$1'"
            return 1
        fi
        shift
    done
    echo "PASS: $out analyzer rejects unsafe tensor-dot shapes and effects."
}

run_exit_arg_probe() {
    echo "--- bounded terminal-exit argument and shutdown probe ---"
    run_aot "$EXIT_PROBE_SRC" "$WORK/exit_arg_order_bin"
    if [ "$FR_COMPILE_RC" -ne 0 ] || [ "$FR_RUN_RC" -ne 0 ] ||
       ! grep -Fqx "$(printf 'SHADOW\tOK')" "$FR_OUT"; then
        echo "FAIL: exit argument or shadowed exit probe"
        cat "$FR_OUT"
        return 1
    fi
    if [ "$FR_ARENA_BYTES" -gt 10000000 ]; then
        echo "FAIL: exit callback left ${FR_ARENA_BYTES}B in the global arena at shutdown"
        return 1
    fi
    echo "  exit=$FR_RUN_RC peak_rss=${FR_RSS_MB}MB arena=${FR_ARENA_BYTES}B"
    echo "PASS: exit argument ran before scope finish; shutdown saw reclaimed arena memory."
}

echo "=========================================================="
echo "  ESH-0214b AOT flat-RSS gate: define_loop_flat_rss_aot_test.esk"
echo "  ceiling=${CEILING_MB}MB  time-mode=${TIME_MODE}"
echo "=========================================================="
echo

echo "--- gate: AOT compile + run (fix ON, default build) ---"
run_aot "$SRC" "$WORK/flat_rss_gate_bin"
gate_compile_rc=$FR_COMPILE_RC
gate_run_rc=$FR_RUN_RC
gate_rss=$FR_RSS_MB
gate_out="$FR_OUT"

fail=0
if [ "$gate_compile_rc" -ne 0 ]; then
    echo "FAIL: AOT compile failed (exit=$gate_compile_rc). Output:"
    cat "$gate_out"
    fail=1
elif [ "$gate_run_rc" -ne 0 ] || ! grep -q "^PASS$" "$gate_out"; then
    echo "FAIL: AOT binary did not complete cleanly (exit=$gate_run_rc). Output:"
    cat "$gate_out"
    fail=1
else
    echo "  exit=$gate_run_rc  peak_rss=${gate_rss}MB"
    if [ "$gate_rss" -gt "$CEILING_MB" ]; then
        echo "FAIL: peak RSS ${gate_rss}MB exceeds ceiling ${CEILING_MB}MB"
        echo "      -- looks like a reintroduced per-iteration leak in the"
        echo "      define-loop + catch-all-guard arena-reclamation path (ESH-0214b)."
        fail=1
    else
        echo "PASS: peak RSS ${gate_rss}MB is within the ${CEILING_MB}MB flat ceiling."
    fi
fi

echo
echo "--- bounded AOT slope gates: named-let, discarded tensor, tensor-dot ---"
if ! run_slope_gate namedlet "$NAMED_SRC" total-ticks; then fail=1; fi
tensor_slope_ok=0
if run_slope_gate tensor_literal "$TENSOR_SRC" total-passes; then
    tensor_slope_ok=1
else
    fail=1
fi
if ! run_slope_gate tensor_dot "$DOT_SRC" total-passes; then fail=1; fi

echo
echo "--- exact 1M discarded-tensor proof after bounded slope passes ---"
if [ "$tensor_slope_ok" -eq 1 ]; then
    tensor_1m_src="$WORK/tensor_literal_1000000.esk"
    sed -E 's/\(define total-passes [0-9]+\)/(define total-passes 1000000)/' \
        "$TENSOR_SRC" > "$tensor_1m_src"
    run_aot "$tensor_1m_src" "$WORK/tensor_literal_1000000_bin"
    if [ "$FR_COMPILE_RC" -ne 0 ] || [ "$FR_RUN_RC" -ne 0 ] ||
       ! grep -Fqx "passes=1000000 result=12000000" "$FR_OUT" ||
       ! grep -Fqx "PASS" "$FR_OUT" ||
       ! grep -Fqx "$(printf 'CARRY\tOK')" "$FR_OUT" ||
       ! grep -Fqx "$(printf 'SIDE_EFFECTS\tOK\t4')" "$FR_OUT" ||
       [ "$FR_RSS_MB" -gt "$CEILING_MB" ] || [ "$FR_ARENA_BYTES" -gt 10000000 ]; then
        echo "FAIL: tensor-literal AOT 1M proof exit=$FR_RUN_RC rss=${FR_RSS_MB}MB arena=${FR_ARENA_BYTES}B"
        cat "$FR_OUT"
        fail=1
    else
        echo "PASS: tensor-literal AOT 1M result=12000000 peak_rss=${FR_RSS_MB}MB arena=${FR_ARENA_BYTES}B"
    fi
else
    echo "SKIP: tensor-literal AOT 1M because its bounded slope gate failed."
fi

echo
echo "--- conservative analyzer and exit cleanup checks ---"
if ! run_debug_rejections "$TENSOR_SRC" tensor_literal \
    "loop 'unsafe-tensor-loop' iter-scope disabled (analysis)"; then fail=1; fi
if ! run_debug_rejections "$DOT_SRC" tensor_dot \
    "loop 'computed-dot-loop' iter-scope disabled (analysis)" \
    "loop 'unequal-dot-loop' iter-scope disabled (analysis)" \
    "loop 'rank2-dot-loop' iter-scope disabled (analysis)" \
    "loop 'mutating-dot-loop' iter-scope disabled (analysis)" \
    "loop 'callback-exit-loop' iter-scope disabled (analysis)" \
    "loop 'callback-dot-loop' iter-scope disabled (analysis)" \
    "loop 'parallel-dot-loop' iter-scope disabled (reachable from a parallel/future callback)"; then fail=1; fi
if ! run_debug_rejections "$EXIT_PROBE_SRC" exit_shadow \
    "loop 'shadow-exit-loop' iter-scope disabled (analysis)"; then fail=1; fi
if ! run_exit_arg_probe; then fail=1; fi

echo
echo "--- advisory (informational only, not gated): AOT compile + run with"
echo "    ESHKOL_NO_ITER_SCOPE=1 at COMPILE time (fix disabled) ---"
run_aot "$SRC" "$WORK/flat_rss_noiter_bin" ESHKOL_NO_ITER_SCOPE=1
if [ "$FR_COMPILE_RC" -ne 0 ]; then
    echo "  (advisory) compile failed with ESHKOL_NO_ITER_SCOPE=1 (exit=$FR_COMPILE_RC) -- skipping comparison."
elif [ "$FR_RUN_RC" -ne 0 ] || ! grep -q "^PASS$" "$FR_OUT"; then
    echo "  (advisory) run failed with ESHKOL_NO_ITER_SCOPE=1 (exit=$FR_RUN_RC) -- skipping comparison."
else
    echo "  (advisory) peak_rss=${FR_RSS_MB}MB with fix disabled (fix-on gate measured ${gate_rss}MB)."
    if [ "$FR_RSS_MB" -gt "$CEILING_MB" ]; then
        echo "  (advisory) confirms the gate WOULD catch this regression: ${FR_RSS_MB}MB > ${CEILING_MB}MB ceiling."
    else
        echo "  (advisory) NOTE: disabling the fix did not exceed the ceiling on this host/build --"
        echo "             the gate's discriminating power could not be confirmed this run."
    fi
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "define_loop_flat_rss_aot_test.sh: PASS"
else
    echo "define_loop_flat_rss_aot_test.sh: FAIL"
fi
exit "$fail"
