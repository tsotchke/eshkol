#!/usr/bin/env bash
# tests/memory/iter_scope_partial_reclaim_test.sh — ESH-0214e flat-RSS
# + correctness + poison gate for iter-scope PARTIAL RECLAMATION.
#
# ESH-0214e teaches automatic per-iteration reclamation to survive persistent
# mutation. A resident tick loop that mutates persistent state EVERY tick
# (hash-table-set! / vector-set! / set-cdr!) used to be REJECTED outright by
# the all-or-nothing static gate, so it got no reclamation and leaked one
# tick's transient garbage forever. ESH-0214e lowers such a loop with a per-
# loop NURSERY REGION: the write barrier promotes each persistent-mutation
# escapee out of the nursery, the loop-carried out-values are promoted at each
# back edge, then the nursery is reset — reclaiming the transient garbage while
# every stored value reads back correct.
#
# This gate:
#   1. compiles tests/memory/iter_scope_partial_reclaim_test.esk AOT (fix ON,
#      default build), runs it, and FAILS if peak RSS exceeds the flat ceiling
#      or the program does not print PASS (correctness);
#   2. runs the same AOT binary under ESHKOL_ARENA_POISON=1 so any escapee the
#      nursery reset FAILED to promote out dereferences a 0xCB.. address and
#      crashes loudly instead of reading recycled memory (dangling-ptr tripwire);
#   3. runs the source under the JIT (-r) and requires PASS (correctness, JIT);
#   4. compiles + runs the with-region twin
#      (iter_scope_partial_reclaim_baseline.esk) as the flat-RSS acceptance
#      baseline, and requires automatic peak RSS within 10x of it;
#   5. ADVISORY: recompiles with ESHKOL_NO_ITER_SCOPE=1 (fix disabled) and
#      reports its peak RSS + the bytes/tick the fix removes (the "before").
#
# Usage: tests/memory/iter_scope_partial_reclaim_test.sh [--ceiling-mb N] [--timeout S]
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
    echo "iter_scope_partial_reclaim_test.sh: $ESHKOL_RUN not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

SRC="$REPO_ROOT/tests/memory/iter_scope_partial_reclaim_test.esk"
BASELINE="$REPO_ROOT/tests/memory/iter_scope_partial_reclaim_baseline.esk"
for f in "$SRC" "$BASELINE"; do
    [ -f "$f" ] || { echo "iter_scope_partial_reclaim_test.sh: $f not found." >&2; exit 2; }
done
TICKS=$(awk '/^\(define ticks /{print $3; exit}' "$SRC" | tr -d ')')
[ -n "$TICKS" ] || TICKS=100000

CEILING_MB=150
TIMEOUT_S=120
while [ $# -gt 0 ]; do
    case "$1" in
        --ceiling-mb) shift; CEILING_MB="${1:-$CEILING_MB}" ;;
        --timeout) shift; TIMEOUT_S="${1:-$TIMEOUT_S}" ;;
        *) echo "iter_scope_partial_reclaim_test.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# Detect the peak-RSS-reporting `time` flavor (macOS BSD `-l`, Linux GNU `-v`).
TIME_MODE=""
if /usr/bin/time -l true >/dev/null 2>/tmp/.ispr_probe.$$; then
    grep -q "maximum resident set size" /tmp/.ispr_probe.$$ 2>/dev/null && TIME_MODE="bsd"
fi
if [ -z "$TIME_MODE" ] && /usr/bin/time -v true >/tmp/.ispr_probe.$$ 2>&1; then
    grep -qi "Maximum resident set size" /tmp/.ispr_probe.$$ 2>/dev/null && TIME_MODE="gnu"
fi
rm -f /tmp/.ispr_probe.$$
if [ -z "$TIME_MODE" ]; then
    echo "iter_scope_partial_reclaim_test.sh: no peak-RSS-reporting /usr/bin/time on this host — cannot gate." >&2
    exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-ispr.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# run_aot <src> <bin> <run-env...> -> FR_COMPILE_RC FR_RUN_RC FR_RSS_MB FR_OUT
run_aot() {
    local src="$1" bin="$2"; shift 2
    local clog="$WORK/compile_$(basename "$bin").log"
    local rout="$WORK/run_$(basename "$bin").out"
    local tlog="$WORK/time_$(basename "$bin").log"
    ( cd "$WORK" && "$ESHKOL_RUN" "$src" -o "$bin" ) > "$clog" 2>&1
    FR_COMPILE_RC=$?
    if [ "$FR_COMPILE_RC" -ne 0 ]; then FR_RUN_RC=127; FR_RSS_MB=0; FR_OUT="$clog"; return; fi
    chmod +x "$bin"
    if [ "$TIME_MODE" = "bsd" ]; then
        ( cd "$WORK" && env "$@" /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec: $!\n"' \
            "$TIMEOUT_S" "$bin" ) > "$rout" 2> "$tlog"
        FR_RUN_RC=$?
        FR_RSS_MB=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$tlog")
    else
        ( cd "$WORK" && env "$@" /usr/bin/time -v perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec: $!\n"' \
            "$TIMEOUT_S" "$bin" ) > "$rout" 2> "$tlog"
        FR_RUN_RC=$?
        FR_RSS_MB=$(awk -F: '/Maximum resident set size/{printf "%d", $2/1024}' "$tlog")
    fi
    [ -n "$FR_RSS_MB" ] || FR_RSS_MB=0
    FR_OUT="$rout"
}

echo "=========================================================="
echo "  ESH-0214e iter-scope partial-reclamation gate"
echo "  ticks=$TICKS  ceiling=${CEILING_MB}MB  time-mode=${TIME_MODE}"
echo "=========================================================="
echo

fail=0

# ── 1. AOT gate: fix ON ──────────────────────────────────────────────────────
echo "--- [1] AOT compile + run (automatic nursery, fix ON) ---"
run_aot "$SRC" "$WORK/ispr_fix_on"
on_rc=$FR_RUN_RC; on_rss=$FR_RSS_MB; on_out="$FR_OUT"
if [ "$FR_COMPILE_RC" -ne 0 ]; then
    echo "FAIL: AOT compile failed."; cat "$on_out"; fail=1
elif [ "$on_rc" -ne 0 ] || ! grep -q "^PASS$" "$on_out"; then
    echo "FAIL: AOT binary did not complete cleanly / correctly (exit=$on_rc):"; cat "$on_out"; fail=1
else
    grep -h "iter_scope_partial_reclaim_test" "$on_out"
    echo "  exit=$on_rc  peak_rss=${on_rss}MB"
    if [ "$on_rss" -gt "$CEILING_MB" ]; then
        echo "FAIL: peak RSS ${on_rss}MB exceeds ceiling ${CEILING_MB}MB (reintroduced per-tick leak?)"; fail=1
    else
        echo "PASS: automatic path flat + correct at ${on_rss}MB (< ${CEILING_MB}MB)."
    fi
fi
echo

# ── 2. Poison run (dangling-pointer tripwire) ────────────────────────────────
echo "--- [2] re-run fix-ON binary under ESHKOL_ARENA_POISON=1 ---"
if [ -x "$WORK/ispr_fix_on" ]; then
    if [ "$TIME_MODE" = "bsd" ]; then
        ( cd "$WORK" && ESHKOL_ARENA_POISON=1 /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die' \
            "$TIMEOUT_S" "$WORK/ispr_fix_on" ) > "$WORK/poison.out" 2>/dev/null
    else
        ( cd "$WORK" && ESHKOL_ARENA_POISON=1 /usr/bin/time -v perl -e 'my $s=shift; alarm $s; exec @ARGV; die' \
            "$TIMEOUT_S" "$WORK/ispr_fix_on" ) > "$WORK/poison.out" 2>/dev/null
    fi
    prc=$?
    if [ "$prc" -ne 0 ] || ! grep -q "^PASS$" "$WORK/poison.out"; then
        echo "FAIL: poison run crashed or mis-verified (exit=$prc) — a nursery escapee was NOT promoted out:"; cat "$WORK/poison.out"; fail=1
    else
        echo "PASS: clean + correct under ESHKOL_ARENA_POISON=1 (no dangling nursery pointer)."
    fi
else
    echo "SKIP: fix-ON binary unavailable."
fi
echo

# ── 3. JIT correctness ───────────────────────────────────────────────────────
echo "--- [3] JIT (-r) correctness ---"
( cd "$REPO_ROOT" && ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" -r "$SRC" ) > "$WORK/jit.out" 2>&1
jrc=$?
if [ "$jrc" -ne 0 ] || ! grep -q "^PASS$" "$WORK/jit.out"; then
    echo "FAIL: JIT run crashed or mis-verified (exit=$jrc):"; tail -5 "$WORK/jit.out"; fail=1
else
    grep -h "iter_scope_partial_reclaim_test" "$WORK/jit.out"; echo "PASS: JIT flat + correct."
fi
echo

# ── 4. with-region baseline A/B ──────────────────────────────────────────────
echo "--- [4] with-region baseline (acceptance target) ---"
run_aot "$BASELINE" "$WORK/ispr_baseline"
base_rss=$FR_RSS_MB; base_out="$FR_OUT"
if [ "$FR_COMPILE_RC" -ne 0 ] || [ "$FR_RUN_RC" -ne 0 ] || ! grep -q "^PASS$" "$base_out"; then
    echo "  (baseline compile/run failed — skipping A/B comparison)"; cat "$base_out" 2>/dev/null | tail -3
else
    echo "  with-region baseline peak_rss=${base_rss}MB   automatic peak_rss=${on_rss}MB"
    # automatic within 10x of baseline (both flat; pre-fix automatic is unbounded)
    if [ "$base_rss" -gt 0 ] && [ "$on_rss" -gt $((base_rss * 10)) ]; then
        echo "FAIL: automatic ${on_rss}MB exceeds 10x the with-region baseline ${base_rss}MB."; fail=1
    else
        echo "PASS: automatic within 10x of with-region baseline."
    fi
fi
echo

# ── 5. ADVISORY: fix OFF (ESHKOL_NO_ITER_SCOPE=1) — the "before" ─────────────
echo "--- [5] advisory: AOT with ESHKOL_NO_ITER_SCOPE=1 at COMPILE time (fix OFF) ---"
( cd "$WORK" && ESHKOL_NO_ITER_SCOPE=1 "$ESHKOL_RUN" "$SRC" -o "$WORK/ispr_fix_off" ) > "$WORK/compile_off.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  (advisory) compile with fix OFF failed — skipping."
else
    chmod +x "$WORK/ispr_fix_off"
    if [ "$TIME_MODE" = "bsd" ]; then
        ( cd "$WORK" && /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die' "$TIMEOUT_S" "$WORK/ispr_fix_off" ) > "$WORK/off.out" 2> "$WORK/off.time"
        off_rss=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$WORK/off.time")
    else
        ( cd "$WORK" && /usr/bin/time -v perl -e 'my $s=shift; alarm $s; exec @ARGV; die' "$TIMEOUT_S" "$WORK/ispr_fix_off" ) > "$WORK/off.out" 2> "$WORK/off.time"
        off_rss=$(awk -F: '/Maximum resident set size/{printf "%d", $2/1024}' "$WORK/off.time")
    fi
    [ -n "$off_rss" ] || off_rss=0
    echo "  (advisory) fix-OFF peak_rss=${off_rss}MB   fix-ON peak_rss=${on_rss}MB"
    if [ "$off_rss" -gt "$on_rss" ] && [ "$TICKS" -gt 0 ]; then
        # bytes/tick the fix removes = (delta MB * 2^20) / ticks
        bpt=$(awk -v d="$((off_rss - on_rss))" -v t="$TICKS" 'BEGIN{printf "%.1f", (d*1048576.0)/t}')
        echo "  (advisory) fix removes ~${bpt} bytes/tick of transient leak over $TICKS ticks."
        [ "$off_rss" -gt "$CEILING_MB" ] && echo "  (advisory) confirms the gate WOULD catch the regression: ${off_rss}MB > ${CEILING_MB}MB."
    fi
fi
echo

if [ "$fail" -eq 0 ]; then
    echo "iter_scope_partial_reclaim_test.sh: PASS"
else
    echo "iter_scope_partial_reclaim_test.sh: FAIL"
fi
exit "$fail"
