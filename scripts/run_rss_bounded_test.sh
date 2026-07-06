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
# Two bounded-RSS programs are gated:
#   * tests/memory/long_loop_rss_test.esk       — the explicit per-iteration
#     (with-region ...) idiom (ESH-0214)
#   * tests/memory/long_loop_rss_auto_test.esk  — NO annotations: the
#     automatic per-iteration arena scope reclamation (ESH-0214b) must
#     bound RSS entirely on its own
#
# For each program the harness runs two very different pass counts (default
# 100k and 1,000,000) under `eshkol-run -r`, measures peak RSS via
# `/usr/bin/time -l` (macOS), and asserts:
#   1. both runs complete (exit 0, print PASS)
#   2. peak RSS does NOT scale with iteration count (a real leak grows
#      ~linearly: 10x the passes -> ~10x the RSS; bounded memory gives
#      near-identical RSS regardless of N)
#   3. the large-N run stays under an absolute sanity ceiling
#
# Usage: scripts/run_rss_bounded_test.sh [--small N] [--large N] [--ceiling-mb N]
#                                        [--timeout S] [--test <file.esk>]
#                                        [--with-region-only | --auto-only]
#   Default gates BOTH programs. --test substitutes a custom program (it
#   must contain "(define total-passes 300000)" and print PASS on success).
#   BUILD_DIR env var selects the build directory (default: build).
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
    *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
esac
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_rss_bounded_test.sh: $BUILD_DIR/eshkol-run not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

SMALL_N=100000
LARGE_N=1000000
CEILING_MB=500
TIMEOUT_S=180
TESTS=(
    "$REPO_ROOT/tests/memory/long_loop_rss_test.esk"
    "$REPO_ROOT/tests/memory/long_loop_rss_auto_test.esk"
)
while [ $# -gt 0 ]; do
    case "$1" in
        --small) shift; SMALL_N="${1:-$SMALL_N}" ;;
        --large) shift; LARGE_N="${1:-$LARGE_N}" ;;
        --ceiling-mb) shift; CEILING_MB="${1:-$CEILING_MB}" ;;
        --timeout) shift; TIMEOUT_S="${1:-$TIMEOUT_S}" ;;
        --test) shift; TESTS=("${1:?--test requires a file}") ;;
        --with-region-only) TESTS=("$REPO_ROOT/tests/memory/long_loop_rss_test.esk") ;;
        --auto-only) TESTS=("$REPO_ROOT/tests/memory/long_loop_rss_auto_test.esk") ;;
        *) echo "run_rss_bounded_test.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-rss-bounded.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# run_one <test_src> <n> -> sets RB_RC RB_RSS_MB RB_WALL_S RB_OUT
run_one() {
    local src="$1" n="$2"
    local base; base=$(basename "$src" .esk)
    local variant="$WORK/${base}_n${n}.esk"
    # Substitute the pass count; everything else about the program (the
    # witness file it writes/reads, its reclamation strategy) is unchanged.
    # Run each variant from its own scratch dir so witness files can't race.
    sed "s/(define total-passes 300000)/(define total-passes ${n})/" "$src" > "$variant"
    local rundir="$WORK/run_${base}_n${n}"
    mkdir -p "$rundir"
    local outfile="$WORK/out_${base}_n${n}.txt"
    local timefile="$WORK/time_${base}_n${n}.txt"
    ( cd "$rundir" && \
      /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
        "$TIMEOUT_S" "$ESHKOL_RUN" -r "$variant" ) > "$outfile" 2> "$timefile"
    RB_RC=$?
    RB_RSS_MB=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$timefile")
    [ -n "$RB_RSS_MB" ] || RB_RSS_MB=0
    RB_WALL_S=$(grep -E '^\s*[0-9.]+ real' "$timefile" | awk '{print $1}')
    RB_OUT="$outfile"
}

# gate_one <test_src> -> returns 0 on PASS
gate_one() {
    local src="$1"
    local name; name=$(basename "$src")
    if [ ! -f "$src" ]; then
        echo "run_rss_bounded_test.sh: $src not found." >&2
        return 1
    fi

    echo "=========================================================="
    echo "  ESH-0214 bounded-RSS gate: $name"
    echo "  small N=$SMALL_N  large N=$LARGE_N  ceiling=${CEILING_MB}MB"
    echo "=========================================================="

    echo ""
    echo "--- run 1/2: N=$SMALL_N ---"
    run_one "$src" "$SMALL_N"
    local small_rc=$RB_RC small_rss=$RB_RSS_MB small_wall=$RB_WALL_S
    echo "  exit=$small_rc  peak_rss=${small_rss}MB  wall=${small_wall}s"
    if [ "$small_rc" -ne 0 ] || ! grep -q "^PASS$" "$RB_OUT"; then
        echo "FAIL: N=$SMALL_N run did not complete cleanly (exit=$small_rc). Output:"
        cat "$RB_OUT"
        return 1
    fi

    echo ""
    echo "--- run 2/2: N=$LARGE_N ---"
    run_one "$src" "$LARGE_N"
    local large_rc=$RB_RC large_rss=$RB_RSS_MB large_wall=$RB_WALL_S
    echo "  exit=$large_rc  peak_rss=${large_rss}MB  wall=${large_wall}s"
    if [ "$large_rc" -ne 0 ] || ! grep -q "^PASS$" "$RB_OUT"; then
        echo "FAIL: N=$LARGE_N run did not complete cleanly (exit=$large_rc, e.g. OOM-killed"
        echo "or a stack-overflow crash — see the test file's header)."
        echo "Output:"
        cat "$RB_OUT"
        return 1
    fi

    echo ""
    echo "--- verdict: $name ---"
    local scale ratio threshold
    scale=$(perl -e 'printf "%.2f", $ARGV[1]/$ARGV[0]' "$SMALL_N" "$LARGE_N")
    ratio=$(perl -e 'my ($a,$b)=@ARGV; $a=1 if $a<1; printf "%.2f", $b/$a' "$small_rss" "$large_rss")
    echo "  pass-count scale factor:  ${scale}x"
    echo "  peak-RSS ratio (large/small): ${ratio}x"

    local fail=0
    # A genuine per-iteration leak scales ~linearly with pass count (ratio
    # ~= scale). Bounded/flat memory gives ratio close to 1 regardless of
    # scale. Allow generous headroom (half the pass-count scale factor)
    # before calling it a regression -- coarse but decisive given how large
    # scale typically is (10x-100x).
    threshold=$(perl -e 'printf "%.2f", $ARGV[0]/2' "$scale")
    if perl -e 'exit(($ARGV[0] > $ARGV[1]) ? 0 : 1)' "$ratio" "$threshold"; then
        echo "  FAIL: peak RSS grew with iteration count (ratio $ratio > threshold $threshold)"
        echo "        -- looks like a reintroduced per-iteration leak, not bounded/flat memory."
        fail=1
    fi
    if [ "$large_rss" -gt "$CEILING_MB" ]; then
        echo "  FAIL: N=$LARGE_N peak RSS ${large_rss}MB exceeds ceiling ${CEILING_MB}MB"
        fail=1
    fi

    if [ "$fail" -eq 0 ]; then
        echo "  PASS: RSS plateau is flat (${small_rss}MB @ N=$SMALL_N -> ${large_rss}MB @ N=$LARGE_N)"
        echo "        despite a ${scale}x increase in loop iterations."
        return 0
    fi
    return 1
}

overall=0
for t in "${TESTS[@]}"; do
    gate_one "$t" || overall=1
    echo ""
done

if [ "$overall" -eq 0 ]; then
    echo "run_rss_bounded_test.sh: ALL bounded-RSS gates PASS"
else
    echo "run_rss_bounded_test.sh: bounded-RSS gate FAILED"
fi
exit "$overall"
