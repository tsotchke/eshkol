#!/usr/bin/env bash
# tests/memory/bignum_rational_flat_rss_test.sh — SW-164 flat-RSS + correctness
# gate for exact bignum-rational arithmetic, on both lowering paths.
#
# Exact rational arithmetic used to grow resident memory in proportion to the
# WORK an operation did rather than the VALUES it produced, for two reasons that
# compound:
#
#   * Reducing a result ran a Euclidean GCD over the full-width numerator and
#     denominator. The step count grows with the operands' digit count and every
#     step bump-allocated a quotient, a remainder and two limb buffers into the
#     caller's arena — which reclaims only at a scope or region boundary, and a
#     numeric primitive had no boundary inside it. Producing one reduced
#     rational therefore retained O(bit-length) dead bignums, permanently.
#
#   * The per-iteration loop scope (ESH-0214b) reclaimed an iteration only when
#     nothing flowing into the next one pointed into it. An accumulator is built
#     inside the iteration, so it always pointed into it, so the scope always
#     committed and the loop retained every iteration it had ever run.
#
# Bignum INTEGER arithmetic has no reduction step and hence neither problem,
# which is why integer loops stayed flat while rational ones did not — the
# asymmetry this gate exists to keep closed.
#
# The gate runs two fixtures on BOTH lowering paths (JIT via `eshkol-run -r`
# and AOT via `-o`), and fails if any of them
#   (1) does not print PASS — each fixture checks its own answer, including
#       closed-form values and a lowest-terms round trip, so a reclamation bug
#       that frees something live is caught here and not mistaken for a win, or
#   (2) exceeds its peak-RSS ceiling, or
#   (3) exceeds its wall-clock ceiling.
#
# Usage: tests/memory/bignum_rational_flat_rss_test.sh [--harmonic-ceiling-mb N]
#                                                      [--mul-ceiling-mb N]
#                                                      [--harmonic-seconds N]
#                                                      [--timeout S]
#   BUILD_DIR env var selects the build directory (default: build).
#   ESHKOL_RUN env var overrides the eshkol-run binary path directly.
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
if [ -z "${ESHKOL_RUN:-}" ]; then
    case "$BUILD_DIR" in
        /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
        *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
    esac
fi
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "bignum_rational_flat_rss_test.sh: $ESHKOL_RUN not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

# Ceilings. The fixed behavior measures far below each of these and the defect
# measured orders of magnitude above, so every ceiling separates the two with
# wide margin in both directions — it is a tripwire, not a benchmark.
#
# The AOT lane carries the strict budget, because there the process contains
# nothing but the computation. The JIT lane runs the SAME budget plus one named
# allowance for the compiler, which is resident in the same process and is not
# what this gate is measuring; keeping it as an explicit constant rather than
# folding it into the numbers keeps the budget legible. The JIT lane also runs
# with ESHKOL_JIT_CACHE=0 so it genuinely JIT-executes: with the cache on, a
# warm `-r` re-execs a cached native binary and would silently measure the AOT
# path twice.
HARMONIC_CEILING_MB=200
HARMONIC_SECONDS=2
MUL_CEILING_MB=300
JIT_ALLOWANCE_MB=300
JIT_ALLOWANCE_SECONDS=4
TIMEOUT_S=120
# math_acceptance_exact_rational_memory is gated as a RATIO against a control
# that runs the identical loop shapes over machine integers, plus a floor so a
# few MB of ordinary variation is never a failure. The defect it guards was a
# hundredfold, so a factor of two is a wide tripwire — and a ratio holds its
# meaning on a machine whose baseline is nothing like this one's.
ACCEPTANCE_RATIO_NUM=2
ACCEPTANCE_FLOOR_MB=64
while [ $# -gt 0 ]; do
    case "$1" in
        --harmonic-ceiling-mb) shift; HARMONIC_CEILING_MB="${1:-$HARMONIC_CEILING_MB}" ;;
        --mul-ceiling-mb) shift; MUL_CEILING_MB="${1:-$MUL_CEILING_MB}" ;;
        --harmonic-seconds) shift; HARMONIC_SECONDS="${1:-$HARMONIC_SECONDS}" ;;
        --timeout) shift; TIMEOUT_S="${1:-$TIMEOUT_S}" ;;
        *) echo "bignum_rational_flat_rss_test.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# Detect which peak-RSS-reporting `time` flavor is available.
PROBE="$(mktemp "${TMPDIR:-/tmp}/eshkol-bigrat-probe.XXXXXX")"
TIME_MODE=""
if /usr/bin/time -l true >/dev/null 2>"$PROBE"; then
    grep -q "maximum resident set size" "$PROBE" 2>/dev/null && TIME_MODE="bsd"
fi
if [ -z "$TIME_MODE" ] && /usr/bin/time -v true >"$PROBE" 2>&1; then
    grep -qi "Maximum resident set size" "$PROBE" 2>/dev/null && TIME_MODE="gnu"
fi
rm -f "$PROBE"
if [ -z "$TIME_MODE" ]; then
    echo "bignum_rational_flat_rss_test.sh: neither \`/usr/bin/time -l\` (macOS) nor \`/usr/bin/time -v\` (Linux) reports peak RSS on this host — cannot gate." >&2
    exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-bigrat-rss.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "=========================================================="
echo "  SW-164 flat-RSS + correctness gate:"
echo "  exact bignum-rational arithmetic (JIT and AOT)"
echo "  harmonic: <= ${HARMONIC_CEILING_MB}MB, <= ${HARMONIC_SECONDS}s"
echo "  multiply: <= ${MUL_CEILING_MB}MB"
echo "  time-mode=${TIME_MODE}"
echo "=========================================================="
echo

fail=0

# read_rss <time-log>
read_rss() {
    if [ "$TIME_MODE" = "bsd" ]; then
        awk '/maximum resident set size/{printf "%d", $1/1048576}' "$1"
    else
        awk -F: '/Maximum resident set size/{printf "%d", $2/1024}' "$1"
    fi
}

# read_centis <time-log> — wall clock in hundredths of a second. Whole-second
# rounding put a cliff right at the ceiling (anything over 1.00s failed a "2s"
# budget as 2s); hundredths compare the measurement against the budget in the
# same unit, so the margin is the margin and not an artifact of rounding.
read_centis() {
    if [ "$TIME_MODE" = "bsd" ]; then
        awk '/ real /{printf "%d", $1 * 100 + 0.5; exit}' "$1"
    else
        awk '/Elapsed \(wall clock\)/{ sub(/.*: */, "", $0);
                 n = split($0, p, ":"); s = p[n] + 0;
                 if (n > 1) s += p[n-1] * 60;
                 if (n > 2) s += p[n-2] * 3600;
                 printf "%d", s * 100 + 0.5; exit }' "$1"
    fi
}

# run_case <label> <fixture> <mode:jit|aot> <ceiling-mb> <seconds-or-0>
run_case() {
    label="$1"; fixture="$2"; mode="$3"; ceiling="$4"; secs="$5"
    src="$REPO_ROOT/tests/memory/$fixture"
    if [ ! -f "$src" ]; then
        echo "FAIL [$label/$mode]: fixture $src not found"
        fail=1
        return
    fi

    run_out="$WORK/$label.$mode.out"
    time_log="$WORK/$label.$mode.time"
    compile_log="$WORK/$label.$mode.compile"

    if [ "$mode" = "aot" ]; then
        bin="$WORK/$label.$mode.bin"
        ( cd "$WORK" && ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" "$src" -o "$bin" ) \
            > "$compile_log" 2>&1
        if [ $? -ne 0 ]; then
            echo "FAIL [$label/$mode]: AOT compile failed. Output:"
            cat "$compile_log"
            fail=1
            return
        fi
        chmod +x "$bin"
        set -- "$bin"
    else
        ceiling=$((ceiling + JIT_ALLOWANCE_MB))
        [ "$secs" -ne 0 ] && secs=$((secs + JIT_ALLOWANCE_SECONDS))
        export ESHKOL_JIT_CACHE=0
        set -- "$ESHKOL_RUN" -r "$src"
    fi

    # A plain `alarm` wrapper keeps the timeout portable (macOS has no
    # coreutils `timeout`), and keeps /usr/bin/time as the outermost process so
    # it still reports peak RSS for the whole subtree.
    ( cd "$WORK" && ESHKOL_PATH="$REPO_ROOT/lib" \
        /usr/bin/time $( [ "$TIME_MODE" = "bsd" ] && echo -l || echo -v ) \
        perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
        "$TIMEOUT_S" "$@" ) > "$run_out" 2> "$time_log"
    rc=$?

    rss_mb="$(read_rss "$time_log")"; [ -n "$rss_mb" ] || rss_mb=0
    centis="$(read_centis "$time_log")"; [ -n "$centis" ] || centis=0
    elapsed="$(awk -v c="$centis" 'BEGIN{printf "%.2f", c/100}')"

    if [ "$rc" -ne 0 ] || ! grep -q "^PASS$" "$run_out" || \
       { grep -q "PASS-COND\|FAIL-COND" "$run_out" && ! grep -q "^PASS-COND$" "$run_out"; }; then
        echo "FAIL [$label/$mode]: did not complete cleanly (exit=$rc). Output:"
        cat "$run_out"
        echo "      -- the fixtures assert their own answers (closed-form values"
        echo "      and a lowest-terms round trip), so a failure here is a WRONG"
        echo "      RESULT, not merely a memory regression: promotion out of a"
        echo "      reclaimed span has dropped or corrupted live data."
        fail=1
        return
    fi

    unset ESHKOL_JIT_CACHE
    line="  [$label/$mode] exit=$rc peak_rss=${rss_mb}MB/${ceiling}MB elapsed=${elapsed}s/${secs}s (answer: PASS)"
    case_failed=0
    if [ "$rss_mb" -gt "$ceiling" ]; then
        echo "$line"
        echo "FAIL [$label/$mode]: peak RSS ${rss_mb}MB exceeds ceiling ${ceiling}MB"
        echo "      -- exact-rational temporaries are being retained again: either"
        echo "      the normalization scratch is no longer bracketed by an arena"
        echo "      scope, or the loop's per-iteration scope has gone back to"
        echo "      committing instead of promoting its survivors."
        case_failed=1
    fi
    if [ "$secs" -ne 0 ] && [ "$centis" -gt "$((secs * 100))" ]; then
        [ "$case_failed" -eq 0 ] && echo "$line"
        echo "FAIL [$label/$mode]: elapsed ${elapsed}s exceeds ceiling ${secs}s"
        echo "      -- reduction has gone back to a full-width GCD over the"
        echo "      result instead of GCDs over the smaller operands."
        case_failed=1
    fi
    [ "$case_failed" -eq 0 ] && echo "$line"
    [ "$case_failed" -eq 1 ] && fail=1
    return 0
}

# run_ratio_case <label> <fixture> <control-fixture> <mode>
# The acceptance shape from the mathematics stream: exact rational arithmetic in
# a plain loop, with no region annotation, must be flat — measured against the
# same loop shape over machine integers rather than against a fixed number.
run_ratio_case() {
    label="$1"; fixture="$2"; control="$3"; mode="$4"

    ctl_log="$WORK/$label.control.$mode.time"
    ctl_out="$WORK/$label.control.$mode.out"
    ctl_src="$REPO_ROOT/tests/memory/$control"
    ctl_bin="$WORK/$label.control.$mode.bin"
    ( cd "$WORK" && ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" "$ctl_src" -o "$ctl_bin" ) \
        > "$WORK/$label.control.compile" 2>&1
    if [ $? -ne 0 ]; then
        echo "FAIL [$label/control]: AOT compile failed."
        cat "$WORK/$label.control.compile"
        fail=1
        return
    fi
    chmod +x "$ctl_bin"
    ( cd "$WORK" && /usr/bin/time $( [ "$TIME_MODE" = "bsd" ] && echo -l || echo -v ) \
        "$ctl_bin" ) > "$ctl_out" 2> "$ctl_log"
    ctl_rc=$?
    ctl_mb="$(read_rss "$ctl_log")"; [ -n "$ctl_mb" ] || ctl_mb=0
    if [ "$ctl_rc" -ne 0 ] || ! grep -q "^PASS$" "$ctl_out"; then
        echo "FAIL [$label/control]: the integer control did not pass (exit=$ctl_rc)."
        echo "      -- without a control that passes, the ratio below means nothing."
        cat "$ctl_out"
        fail=1
        return
    fi

    # Reuse run_case's machinery for the exact-rational run by giving it a
    # ceiling derived from the control.
    budget=$(( ctl_mb * ACCEPTANCE_RATIO_NUM ))
    [ "$budget" -lt "$ACCEPTANCE_FLOOR_MB" ] && budget="$ACCEPTANCE_FLOOR_MB"
    echo "  [$label/control] peak_rss=${ctl_mb}MB -> exact-rational budget ${budget}MB (${ACCEPTANCE_RATIO_NUM}x, floor ${ACCEPTANCE_FLOOR_MB}MB)"
    run_case "$label" "$fixture" "$mode" "$budget" 0
}

run_case harmonic bignum_rational_flat_rss_harmonic_test.esk jit "$HARMONIC_CEILING_MB" "$HARMONIC_SECONDS"
run_case harmonic bignum_rational_flat_rss_harmonic_test.esk aot "$HARMONIC_CEILING_MB" "$HARMONIC_SECONDS"
run_case multiply bignum_rational_flat_rss_mul_test.esk jit "$MUL_CEILING_MB" 0
run_case multiply bignum_rational_flat_rss_mul_test.esk aot "$MUL_CEILING_MB" 0
run_ratio_case math_acceptance_exact_rational_memory \
    math_acceptance_exact_rational_memory.esk \
    math_acceptance_exact_rational_memory_control.esk aot

echo
if [ "$fail" -eq 0 ]; then
    echo "bignum_rational_flat_rss_test.sh: PASS"
else
    echo "bignum_rational_flat_rss_test.sh: FAIL"
fi
exit "$fail"
