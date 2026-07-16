#!/usr/bin/env bash
# tests/memory/region_evac_subtype_coverage_test.sh — ESH-0214d AOT flat-RSS +
# correctness gate for the region escape-evacuator's neuro-symbolic / workspace
# subtype coverage.
#
# ESH-0214c (PR #210) made with-region escape promotion DEEP for CONS / VECTOR /
# MULTI_VALUE / HASH / TENSOR / EXCEPTION / CLOSURE, but its type dispatch
# dropped SUBSTITUTION / FACT / KNOWLEDGE_BASE / FACTOR_GRAPH / WORKSPACE to a
# SHALLOW leaf byte-copy that did not follow interior pointers (fact args, KB
# fact arrays, workspace content + per-module process_fn closures, factor-graph
# numeric buffers). ESH-0214d completes the transitive closure for those.
#
# This gate compiles tests/memory/region_evac_subtype_coverage_test.esk AOT
# (same harness pattern as region_mutating_loop_flat_rss_test.sh), runs it under
# /usr/bin/time (macOS: -l, Linux: -v) AND under ESHKOL_ARENA_POISON=1 — the
# 0xCB poison allocator that fills freed region bytes on region_pop — and fails
# if:
#   (1) the program does not print PASS (correctness: after the region-wrapped
#       loops every promoted subtype's interior — fact args, KB facts, workspace
#       module closure, factor-graph buffers, record/list slots — reads back
#       intact), or
#   (2) peak RSS exceeds a flat ceiling.
# Under poison a missed interior pointer dereferences 0xCB.. and crashes loudly
# (SIGSEGV at 0xcbcbcbcb..) instead of silently reading stale-but-valid data, so
# this gate fails hard on a real evacuation gap rather than passing by luck.
#
# The fixed behavior measures ~100MB (the 1M-iteration phase promotes ~56MB of
# small FACTs into the global arena, which has no GC, plus base runtime). The
# regression this guards against — per-iteration region reclamation failing so
# the heavy transient (make-list 40) garbage is not freed — is 600MB+ / OOM, so
# a 300MB ceiling separates the two with wide margin (matching the sibling
# region_mutating_loop_flat_rss_test.sh gate).
#
# Usage: tests/memory/region_evac_subtype_coverage_test.sh [--ceiling-mb N] [--timeout S]
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
    echo "region_evac_subtype_coverage_test.sh: $ESHKOL_RUN not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

SRC="$REPO_ROOT/tests/memory/region_evac_subtype_coverage_test.esk"
if [ ! -f "$SRC" ]; then
    echo "region_evac_subtype_coverage_test.sh: $SRC not found." >&2
    exit 2
fi

CEILING_MB=300
TIMEOUT_S=180
while [ $# -gt 0 ]; do
    case "$1" in
        --ceiling-mb) shift; CEILING_MB="${1:-$CEILING_MB}" ;;
        --timeout) shift; TIMEOUT_S="${1:-$TIMEOUT_S}" ;;
        *) echo "region_evac_subtype_coverage_test.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# Detect which peak-RSS-reporting `time` flavor is available.
TIME_MODE=""
if /usr/bin/time -l true >/dev/null 2>/tmp/.regevac_probe.$$; then
    if grep -q "maximum resident set size" /tmp/.regevac_probe.$$ 2>/dev/null; then
        TIME_MODE="bsd"
    fi
fi
if [ -z "$TIME_MODE" ] && /usr/bin/time -v true >/tmp/.regevac_probe.$$ 2>&1; then
    if grep -qi "Maximum resident set size" /tmp/.regevac_probe.$$ 2>/dev/null; then
        TIME_MODE="gnu"
    fi
fi
rm -f /tmp/.regevac_probe.$$
if [ -z "$TIME_MODE" ]; then
    echo "region_evac_subtype_coverage_test.sh: neither \`/usr/bin/time -l\` (macOS) nor \`/usr/bin/time -v\` (Linux) reports peak RSS on this host — cannot gate." >&2
    exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-region-evac.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "=========================================================="
echo "  ESH-0214d AOT flat-RSS + correctness gate (poisoned):"
echo "  region_evac_subtype_coverage_test.esk"
echo "  ceiling=${CEILING_MB}MB  time-mode=${TIME_MODE}  ESHKOL_ARENA_POISON=1"
echo "=========================================================="
echo

: "${WORK:?mktemp work dir must be set}"
BIN="$WORK/region_evac_bin"
COMPILE_LOG="$WORK/compile.log"
RUN_OUT="$WORK/run.out"
TIME_LOG="$WORK/time.log"
: "${COMPILE_LOG:?}" "${RUN_OUT:?}" "${TIME_LOG:?}"

( cd "$WORK" && ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" "$SRC" -o "$BIN" ) > "$COMPILE_LOG" 2>&1
COMPILE_RC=$?
if [ "$COMPILE_RC" -ne 0 ]; then
    echo "FAIL: AOT compile failed (exit=$COMPILE_RC). Output:"
    cat "$COMPILE_LOG"
    echo "region_evac_subtype_coverage_test.sh: FAIL"
    exit 1
fi
chmod +x "$BIN"

# Run the AOT binary with the region-arena poison allocator enabled so a missed
# interior pointer crashes at a 0xCB.. address rather than passing by luck.
if [ "$TIME_MODE" = "bsd" ]; then
    ( cd "$WORK" && ESHKOL_ARENA_POISON=1 /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
        "$TIMEOUT_S" "$BIN" ) > "${RUN_OUT:?}" 2> "${TIME_LOG:?}"
    RUN_RC=$?
    RSS_MB=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$TIME_LOG")
else
    ( cd "$WORK" && ESHKOL_ARENA_POISON=1 /usr/bin/time -v perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
        "$TIMEOUT_S" "$BIN" ) > "${RUN_OUT:?}" 2> "${TIME_LOG:?}"
    RUN_RC=$?
    RSS_MB=$(awk -F: '/Maximum resident set size/{printf "%d", $2/1024}' "$TIME_LOG")
fi
[ -n "$RSS_MB" ] || RSS_MB=0

fail=0
if [ "$RUN_RC" -ne 0 ] || ! grep -q "^PASS$" "$RUN_OUT"; then
    echo "FAIL: AOT binary did not complete cleanly under poison (exit=$RUN_RC). Output:"
    cat "$RUN_OUT"
    echo "      -- a crash at a 0xcbcbcbcb.. address (or a missing PASS) means a"
    echo "      promoted logic/workspace subtype kept an interior pointer aimed"
    echo "      into a freed region arena: the ESH-0214d evacuator subtype"
    echo "      coverage has regressed (dropped back to a shallow leaf copy)."
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
    echo "region_evac_subtype_coverage_test.sh: PASS"
else
    echo "region_evac_subtype_coverage_test.sh: FAIL"
fi
exit "$fail"
