#!/usr/bin/env bash
# run_rss_bounded_test.sh — ESH-0214 bounded-RSS regression gate.
#
# BUG #6 / Noesis PRIORITY 1: a long-lived `eshkol-run -r` process that
# repeatedly read-line's a file and appends into a structure must run at
# BOUNDED (flat) RSS, not grow monotonically with iteration count. Two
# independent interpreter-arena leaks (read-line buffers, dropped
# accumulator cons cells) plus a named-let TCO analysis bug (silently
# falling back to real recursion for `if`-guarded loops, capping any loop
# at a few hundred thousand iterations regardless of memory) combined to
# OOM a production memory-weaver daemon at 24GB / crash it via stack
# overflow. See tests/memory/long_loop_rss_test.esk and the ESH-0214
# commit for the full root-cause writeup.
#
# This harness runs that same test program at two very different pass
# counts (default 100k and 1,000,000) under `eshkol-run -r`, measures peak
# RSS via `/usr/bin/time -l` (macOS) for each, and asserts:
#   1. both runs complete (exit 0, print PASS)
#   2. peak RSS does NOT scale with iteration count (a real leak would grow
#      ~linearly: 10x the passes -> ~10x the RSS; a bounded-arena/with-region
#      fix gives near-identical RSS regardless of N)
#   3. the large-N run stays under an absolute sanity ceiling
#
# Usage: scripts/run_rss_bounded_test.sh [--small N] [--large N] [--ceiling-mb N]
#   BUILD_DIR env var selects the build directory (default: build).
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TEST_SRC="$REPO_ROOT/tests/memory/long_loop_rss_test.esk"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_rss_bounded_test.sh: $BUILD_DIR/eshkol-run not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi
if [ ! -f "$TEST_SRC" ]; then
    echo "run_rss_bounded_test.sh: $TEST_SRC not found." >&2
    exit 2
fi

SMALL_N=100000
LARGE_N=1000000
CEILING_MB=500
TIMEOUT_S=180
while [ $# -gt 0 ]; do
    case "$1" in
        --small) shift; SMALL_N="${1:-$SMALL_N}" ;;
        --large) shift; LARGE_N="${1:-$LARGE_N}" ;;
        --ceiling-mb) shift; CEILING_MB="${1:-$CEILING_MB}" ;;
        --timeout) shift; TIMEOUT_S="${1:-$TIMEOUT_S}" ;;
        *) echo "run_rss_bounded_test.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-rss-bounded.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# run_one <n> -> sets RB_RC RB_RSS_MB RB_WALL_S RB_OUT
run_one() {
    local n="$1"
    local variant="$WORK/rss_test_n${n}.esk"
    # Substitute the pass count; everything else about the program (the
    # per-pass with-region idiom, the witness file it writes/reads) is
    # unchanged. Run each variant from its own scratch dir so the witness
    # file one variant writes can't race with another.
    sed "s/(define total-passes 300000)/(define total-passes ${n})/" "$TEST_SRC" > "$variant"
    local rundir="$WORK/run_n${n}"
    mkdir -p "$rundir"
    local outfile="$WORK/out_n${n}.txt"
    local timefile="$WORK/time_n${n}.txt"
    ( cd "$rundir" && \
      /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
        "$TIMEOUT_S" "$ESHKOL_RUN" -r "$variant" ) > "$outfile" 2> "$timefile"
    RB_RC=$?
    RB_RSS_MB=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$timefile")
    [ -n "$RB_RSS_MB" ] || RB_RSS_MB=0
    RB_WALL_S=$(grep -E '^\s*[0-9.]+ real' "$timefile" | awk '{print $1}')
    RB_OUT="$outfile"
}

echo "=========================================================="
echo "  ESH-0214 bounded-RSS regression gate"
echo "  small N=$SMALL_N  large N=$LARGE_N  ceiling=${CEILING_MB}MB"
echo "=========================================================="

echo ""
echo "--- run 1/2: N=$SMALL_N ---"
run_one "$SMALL_N"
SMALL_RC=$RB_RC; SMALL_RSS=$RB_RSS_MB; SMALL_WALL=$RB_WALL_S
echo "  exit=$SMALL_RC  peak_rss=${SMALL_RSS}MB  wall=${SMALL_WALL}s"
if [ "$SMALL_RC" -ne 0 ] || ! grep -q "^PASS$" "$RB_OUT"; then
    echo "FAIL: N=$SMALL_N run did not complete cleanly (exit=$SMALL_RC). Output:"
    cat "$RB_OUT"
    exit 1
fi

echo ""
echo "--- run 2/2: N=$LARGE_N ---"
run_one "$LARGE_N"
LARGE_RC=$RB_RC; LARGE_RSS=$RB_RSS_MB; LARGE_WALL=$RB_WALL_S
echo "  exit=$LARGE_RC  peak_rss=${LARGE_RSS}MB  wall=${LARGE_WALL}s"
if [ "$LARGE_RC" -ne 0 ] || ! grep -q "^PASS$" "$RB_OUT"; then
    echo "FAIL: N=$LARGE_N run did not complete cleanly (exit=$LARGE_RC, e.g. OOM-killed"
    echo "or a stack-overflow crash — see tests/memory/long_loop_rss_test.esk header)."
    echo "Output:"
    cat "$RB_OUT"
    exit 1
fi

echo ""
echo "--- verdict ---"
SCALE=$(perl -e 'printf "%.2f", $ARGV[1]/$ARGV[0]' "$SMALL_N" "$LARGE_N")
RSS_RATIO=$(perl -e 'my ($a,$b)=@ARGV; $a=1 if $a<1; printf "%.2f", $b/$a' "$SMALL_RSS" "$LARGE_RSS")
echo "  pass-count scale factor:  ${SCALE}x"
echo "  peak-RSS ratio (large/small): ${RSS_RATIO}x"

FAIL=0
# A genuine per-iteration leak scales ~linearly with pass count (RSS_RATIO
# ~= SCALE). Bounded/flat memory gives RSS_RATIO close to 1 regardless of
# SCALE. Allow generous headroom (half the pass-count scale factor) before
# calling it a regression -- this is a coarse but decisive signal given how
# large SCALE typically is (10x-100x).
THRESHOLD=$(perl -e 'printf "%.2f", $ARGV[0]/2' "$SCALE")
if perl -e 'exit(($ARGV[0] > $ARGV[1]) ? 0 : 1)' "$RSS_RATIO" "$THRESHOLD"; then
    echo "  FAIL: peak RSS grew with iteration count (ratio $RSS_RATIO > threshold $THRESHOLD)"
    echo "        -- looks like a reintroduced per-iteration leak, not bounded/flat memory."
    FAIL=1
fi
if [ "$LARGE_RSS" -gt "$CEILING_MB" ]; then
    echo "  FAIL: N=$LARGE_N peak RSS ${LARGE_RSS}MB exceeds ceiling ${CEILING_MB}MB"
    FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    echo "  PASS: RSS plateau is flat (${SMALL_RSS}MB @ N=$SMALL_N -> ${LARGE_RSS}MB @ N=$LARGE_N)"
    echo "        despite a ${SCALE}x increase in loop iterations."
    exit 0
else
    exit 1
fi
