#!/usr/bin/env bash
# tests/memory/region_mutating_loop_flat_rss_test.sh — ESH-0214c AOT flat-RSS
# + correctness gate for the with-region mutating tick loop.
#
# ESH-0214c made with-region escape promotion DEEP (transitive evacuation with
# a forwarding map) and added a mutation write barrier, so a resident tick loop
# can allocate transient garbage inside a per-iteration region while mutating
# persistent outer state with freshly consed values. Before the fix the choice
# was: leak ~17.5KB/iteration without the region (~7GB at 400k iterations), or
# corrupt the outer state with dangling interior pointers with it.
#
# This gate compiles tests/memory/region_mutating_loop_flat_rss_test.esk AOT
# (same harness pattern as define_loop_flat_rss_aot_test.sh), runs it under
# /usr/bin/time (macOS: -l, Linux: -v), and fails if
#   (1) the program does not print PASS (correctness: every outer-vector slot
#       reads back intact after 400k region-wrapped mutations), or
#   (2) peak RSS exceeds a flat ceiling. The fixed behavior measures tens of
#       MB; the leak this guards against is ~7GB, so a 300MB ceiling separates
#       the two with wide margin in both directions.
#
# Usage: tests/memory/region_mutating_loop_flat_rss_test.sh [--ceiling-mb N] [--timeout S]
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
    echo "region_mutating_loop_flat_rss_test.sh: $ESHKOL_RUN not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

SRC="$REPO_ROOT/tests/memory/region_mutating_loop_flat_rss_test.esk"
if [ ! -f "$SRC" ]; then
    echo "region_mutating_loop_flat_rss_test.sh: $SRC not found." >&2
    exit 2
fi

CEILING_MB=300
TIMEOUT_S=120
while [ $# -gt 0 ]; do
    case "$1" in
        --ceiling-mb) shift; CEILING_MB="${1:-$CEILING_MB}" ;;
        --timeout) shift; TIMEOUT_S="${1:-$TIMEOUT_S}" ;;
        *) echo "region_mutating_loop_flat_rss_test.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# Detect which peak-RSS-reporting `time` flavor is available.
TIME_MODE=""
if /usr/bin/time -l true >/dev/null 2>/tmp/.regmut_probe.$$; then
    if grep -q "maximum resident set size" /tmp/.regmut_probe.$$ 2>/dev/null; then
        TIME_MODE="bsd"
    fi
fi
if [ -z "$TIME_MODE" ] && /usr/bin/time -v true >/tmp/.regmut_probe.$$ 2>&1; then
    if grep -qi "Maximum resident set size" /tmp/.regmut_probe.$$ 2>/dev/null; then
        TIME_MODE="gnu"
    fi
fi
rm -f /tmp/.regmut_probe.$$
if [ -z "$TIME_MODE" ]; then
    echo "region_mutating_loop_flat_rss_test.sh: neither \`/usr/bin/time -l\` (macOS) nor \`/usr/bin/time -v\` (Linux) reports peak RSS on this host — cannot gate." >&2
    exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-region-mut-rss.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "=========================================================="
echo "  ESH-0214c AOT flat-RSS + correctness gate:"
echo "  region_mutating_loop_flat_rss_test.esk"
echo "  ceiling=${CEILING_MB}MB  time-mode=${TIME_MODE}"
echo "=========================================================="
echo

: "${WORK:?mktemp work dir must be set}"
BIN="$WORK/region_mut_rss_bin"
COMPILE_LOG="$WORK/compile.log"
RUN_OUT="$WORK/run.out"
TIME_LOG="$WORK/time.log"
: "${COMPILE_LOG:?}" "${RUN_OUT:?}" "${TIME_LOG:?}"

( cd "$WORK" && "$ESHKOL_RUN" "$SRC" -o "$BIN" ) > "$COMPILE_LOG" 2>&1
COMPILE_RC=$?
if [ "$COMPILE_RC" -ne 0 ]; then
    echo "FAIL: AOT compile failed (exit=$COMPILE_RC). Output:"
    cat "$COMPILE_LOG"
    echo "region_mutating_loop_flat_rss_test.sh: FAIL"
    exit 1
fi
chmod +x "$BIN"

if [ "$TIME_MODE" = "bsd" ]; then
    ( cd "$WORK" && /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
        "$TIMEOUT_S" "$BIN" ) > "${RUN_OUT:?}" 2> "${TIME_LOG:?}"
    RUN_RC=$?
    RSS_MB=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$TIME_LOG")
else
    ( cd "$WORK" && /usr/bin/time -v perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
        "$TIMEOUT_S" "$BIN" ) > "${RUN_OUT:?}" 2> "${TIME_LOG:?}"
    RUN_RC=$?
    RSS_MB=$(awk -F: '/Maximum resident set size/{printf "%d", $2/1024}' "$TIME_LOG")
fi
[ -n "$RSS_MB" ] || RSS_MB=0

fail=0
if [ "$RUN_RC" -ne 0 ] || ! grep -q "^PASS$" "$RUN_OUT"; then
    echo "FAIL: AOT binary did not complete cleanly (exit=$RUN_RC). Output:"
    cat "$RUN_OUT"
    echo "      -- a correctness failure here means the outer vector's slots"
    echo "      dangle into freed region arenas: the deep escape promotion /"
    echo "      mutation write barrier (ESH-0214c) has regressed."
    fail=1
else
    echo "  exit=$RUN_RC  peak_rss=${RSS_MB}MB  (correctness: PASS)"
    if [ "$RSS_MB" -gt "$CEILING_MB" ]; then
        echo "FAIL: peak RSS ${RSS_MB}MB exceeds ceiling ${CEILING_MB}MB"
        echo "      -- looks like per-iteration region reclamation regressed, or"
        echo "      the write barrier is over-promoting (copying the transient"
        echo "      garbage instead of only the escaping subgraph)."
        fail=1
    else
        echo "PASS: peak RSS ${RSS_MB}MB is within the ${CEILING_MB}MB flat ceiling."
    fi
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "region_mutating_loop_flat_rss_test.sh: PASS"
else
    echo "region_mutating_loop_flat_rss_test.sh: FAIL"
fi
exit "$fail"
