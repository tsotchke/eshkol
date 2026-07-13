#!/usr/bin/env bash
# tests/memory/define_loop_flat_rss_aot_test.sh — ESH-0214b AOT flat-RSS
# regression gate.
#
# ESH-0214b extended automatic per-iteration arena-scope reclamation from
# named-let loops to self-tail-recursive top-level `define` loops, and
# taught the escape analysis to accept a catch-all `guard` body instead of
# rejecting any guarded loop outright. This is the exact shape of a
# production daemon/resident loop: a top-level `define` loop wrapped in an
# error boundary. Without the fix, such a loop leaks roughly one iteration's
# worth of garbage forever; with it, peak RSS stays flat regardless of
# iteration count.
#
# Unlike scripts/run_rss_bounded_test.sh (which gates the JIT `-r` path),
# this gate is AOT-only: it compiles tests/memory/define_loop_flat_rss_aot_test.esk
# ahead-of-time to a native binary with `eshkol-run <src> -o <bin>`, runs the
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
    echo "define_loop_flat_rss_aot_test.sh: $ESHKOL_RUN not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

SRC="$REPO_ROOT/tests/memory/define_loop_flat_rss_aot_test.esk"
if [ ! -f "$SRC" ]; then
    echo "define_loop_flat_rss_aot_test.sh: $SRC not found." >&2
    exit 2
fi

CEILING_MB=200
TIMEOUT_S=60
while [ $# -gt 0 ]; do
    case "$1" in
        --ceiling-mb) shift; CEILING_MB="${1:-$CEILING_MB}" ;;
        --timeout) shift; TIMEOUT_S="${1:-$TIMEOUT_S}" ;;
        *) echo "define_loop_flat_rss_aot_test.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# Detect which peak-RSS-reporting `time` flavor is available.
# macOS (BSD time): `/usr/bin/time -l` reports "NNNN  maximum resident set size" in BYTES.
# Linux (GNU time):  `/usr/bin/time -v` reports "Maximum resident set size (kbytes): NNNN" in KB.
TIME_MODE=""
if /usr/bin/time -l true >/dev/null 2>/tmp/.deflrat_probe.$$; then
    if grep -q "maximum resident set size" /tmp/.deflrat_probe.$$ 2>/dev/null; then
        TIME_MODE="bsd"
    fi
fi
if [ -z "$TIME_MODE" ] && /usr/bin/time -v true >/tmp/.deflrat_probe.$$ 2>&1; then
    if grep -qi "Maximum resident set size" /tmp/.deflrat_probe.$$ 2>/dev/null; then
        TIME_MODE="gnu"
    fi
fi
rm -f /tmp/.deflrat_probe.$$
if [ -z "$TIME_MODE" ]; then
    echo "define_loop_flat_rss_aot_test.sh: neither \`/usr/bin/time -l\` (macOS) nor \`/usr/bin/time -v\` (Linux) reports peak RSS on this host — cannot gate." >&2
    exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-flat-rss-aot.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# run_aot <src> <bin> <compile_env...=> -> sets FR_COMPILE_RC FR_RUN_RC FR_RSS_MB FR_OUT
run_aot() {
    local src="$1" bin="$2"; shift 2
    local compile_log="$WORK/compile_$(basename "$bin").log"
    local run_out="$WORK/run_$(basename "$bin").out"
    local time_log="$WORK/time_$(basename "$bin").log"

    ( cd "$WORK" && env "$@" "$ESHKOL_RUN" "$src" -o "$bin" ) > "$compile_log" 2>&1
    FR_COMPILE_RC=$?
    if [ "$FR_COMPILE_RC" -ne 0 ]; then
        FR_RUN_RC=127
        FR_RSS_MB=0
        FR_OUT="$compile_log"
        return
    fi
    chmod +x "$bin"

    if [ "$TIME_MODE" = "bsd" ]; then
        ( cd "$WORK" && /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
            "$TIMEOUT_S" "$bin" ) > "$run_out" 2> "$time_log"
        FR_RUN_RC=$?
        FR_RSS_MB=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$time_log")
    else
        ( cd "$WORK" && /usr/bin/time -v perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
            "$TIMEOUT_S" "$bin" ) > "$run_out" 2> "$time_log"
        FR_RUN_RC=$?
        FR_RSS_MB=$(awk -F: '/Maximum resident set size/{printf "%d", $2/1024}' "$time_log")
    fi
    [ -n "$FR_RSS_MB" ] || FR_RSS_MB=0
    FR_OUT="$run_out"
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
