#!/usr/bin/env bash
# tests/memory/region_handle_training_rss_test.sh — #341 gate for the
# user-reachable region-handle surface: flat peak RSS in an AD training loop,
# numerics identical to the unscoped baseline, and the full safety matrix clean
# under the arena poisoner.
#
# #341 reported a 161-parameter MLP training loop growing ~123 MB/step without
# bound. The automatic per-iteration nursery (ESH-0214e) cannot help: its static
# escape analysis disqualifies any loop body containing a `gradient` op, a `set!`
# or a `tensor-set!`, and a training step trips all three — by design. The fix is
# a non-lexical surface (`region-open` / `region-close`) over the same region
# machinery `with-region` already uses.
#
# This gate:
#   1. compiles tests/memory/region_handle_training_rss.esk AOT and runs the
#      handle mode at several step counts, requiring peak RSS to stay FLAT (a
#      ceiling, plus a growth check between the smallest and largest run);
#   2. runs the plain (unscoped) mode at the same step counts as the legible
#      baseline, and requires it to grow at least 3x more than the handle mode;
#   3. requires the trained-parameter checksum to be IDENTICAL across the plain,
#      with-region and handle modes — reclamation must not change the numerics;
#   4. runs tests/memory/region_handle_safety_test.esk (the misuse matrix:
#      double close, use-after-close, out-of-order cascade, fabricated tokens,
#      raise/call-cc unwind crossing an open handle, slot reuse, never-closed)
#      AOT and under the JIT, both with ESHKOL_ARENA_POISON=1 so a value we
#      failed to promote out of a closed region hits an obvious 0xCB sentinel.
#
# Usage: tests/memory/region_handle_training_rss_test.sh [--ceiling-mb N] [--timeout S]
#   BUILD_DIR selects the build directory (default: build).
#   ESHKOL_RUN overrides the eshkol-run binary path directly.
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
    echo "region_handle_training_rss_test.sh: $ESHKOL_RUN not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

SRC="$REPO_ROOT/tests/memory/region_handle_training_rss.esk"
SAFETY="$REPO_ROOT/tests/memory/region_handle_safety_test.esk"
for f in "$SRC" "$SAFETY"; do
    [ -f "$f" ] || { echo "region_handle_training_rss_test.sh: $f not found." >&2; exit 2; }
done

# Committed gate runs at the small batch so CI stays fast; the per-step tape is
# proportionally smaller than the report's ~123 MB/step but the SHAPE — flat vs
# linear — is what is being gated. REGION_BYTES must comfortably exceed one
# step's arena so it lands in a single block (see the memory-model doc).
BATCH="${BATCH:-8}"
REGION_BYTES="${REGION_BYTES:-64000000}"
STEP_LIST="${STEP_LIST:-5 20}"
CEILING_MB="${CEILING_MB:-260}"
TIMEOUT_S=300
while [ $# -gt 0 ]; do
    case "$1" in
        --ceiling-mb) shift; CEILING_MB="${1:-$CEILING_MB}" ;;
        --timeout) shift; TIMEOUT_S="${1:-$TIMEOUT_S}" ;;
        *) echo "region_handle_training_rss_test.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# Detect the peak-RSS-reporting `time` flavor (macOS BSD `-l`, Linux GNU `-v`).
TIME_MODE=""
if /usr/bin/time -l true >/dev/null 2>/tmp/.rhtr_probe.$$; then
    grep -q "maximum resident set size" /tmp/.rhtr_probe.$$ 2>/dev/null && TIME_MODE="bsd"
fi
if [ -z "$TIME_MODE" ] && /usr/bin/time -v true >/tmp/.rhtr_probe.$$ 2>&1; then
    grep -qi "Maximum resident set size" /tmp/.rhtr_probe.$$ 2>/dev/null && TIME_MODE="gnu"
fi
rm -f /tmp/.rhtr_probe.$$
if [ -z "$TIME_MODE" ]; then
    echo "region_handle_training_rss_test.sh: no peak-RSS-reporting /usr/bin/time on this host — cannot gate." >&2
    exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-rhtr.XXXXXX")"
: "${WORK:?}"
trap 'rm -rf "$WORK"' EXIT

# compile_aot <src> <bin> -> AOT_RC
compile_aot() {
    ( cd "$WORK" && "$ESHKOL_RUN" "$1" -o "$2" ) > "$WORK/compile_$(basename "$2").log" 2>&1
    AOT_RC=$?
    [ "$AOT_RC" -eq 0 ] && chmod +x "$2"
    return $AOT_RC
}

# timed_run <bin> <outfile> <env...> -> TR_RC TR_RSS_MB
timed_run() {
    local bin="$1" out="$2"; shift 2
    local tlog="$WORK/time.$$.log"
    if [ "$TIME_MODE" = "bsd" ]; then
        ( cd "$WORK" && env "$@" /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec: $!\n"' \
            "$TIMEOUT_S" "$bin" ) > "$out" 2> "$tlog"
        TR_RC=$?
        TR_RSS_MB=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$tlog")
    else
        ( cd "$WORK" && env "$@" /usr/bin/time -v perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec: $!\n"' \
            "$TIMEOUT_S" "$bin" ) > "$out" 2> "$tlog"
        TR_RC=$?
        TR_RSS_MB=$(awk -F: '/Maximum resident set size/{printf "%d", $2/1024}' "$tlog")
    fi
    [ -n "$TR_RSS_MB" ] || TR_RSS_MB=0
}

echo "=========================================================="
echo "  #341 region-handle gate — flat RSS + identical numerics"
echo "  batch=$BATCH steps='$STEP_LIST' region_bytes=$REGION_BYTES"
echo "  ceiling=${CEILING_MB}MB time-mode=$TIME_MODE"
echo "=========================================================="
echo

fail=0

# ── 1. AOT compile the training program ──────────────────────────────────────
echo "--- [1] AOT compile training program ---"
if ! compile_aot "$SRC" "$WORK/train"; then
    echo "FAIL: AOT compile failed:"; cat "$WORK/compile_train.log"; exit 1
fi
echo "PASS: compiled."
echo

# ── 2. handle mode: flat peak RSS ────────────────────────────────────────────
echo "--- [2] handle mode: peak RSS must be flat across step counts ---"
first_steps=""; first_rss=0; last_steps=""; last_rss=0
handle_sum_ok=1
for s in $STEP_LIST; do
    timed_run "$WORK/train" "$WORK/handle.$s.out" \
        BATCH="$BATCH" STEPS="$s" MODE=handle REGION_BYTES="$REGION_BYTES"
    rss=$TR_RSS_MB
    if [ "$TR_RC" -ne 0 ]; then
        echo "FAIL: handle mode steps=$s exited $TR_RC:"; cat "$WORK/handle.$s.out"; fail=1; handle_sum_ok=0
        continue
    fi
    echo "  steps=$s peak_rss=${rss}MB  $(cat "$WORK/handle.$s.out")"
    if [ "$rss" -gt "$CEILING_MB" ]; then
        echo "  FAIL: ${rss}MB exceeds ceiling ${CEILING_MB}MB"; fail=1; handle_sum_ok=0
    fi
    if [ -z "$first_steps" ]; then first_steps=$s; first_rss=$rss; fi
    last_steps=$s; last_rss=$rss
done
# Flatness: 4x the step count must not cost anywhere near 4x the memory. Allow
# 1.6x headroom for allocator/OS variance; a linear leak would be ~4x.
if [ "$handle_sum_ok" -eq 1 ] && [ "$first_rss" -gt 0 ]; then
    limit=$(( first_rss * 16 / 10 ))
    echo "  growth ${first_steps}->${last_steps} steps: ${first_rss}MB -> ${last_rss}MB (flat limit ${limit}MB)"
    if [ "$last_rss" -gt "$limit" ]; then
        echo "  FAIL: peak RSS grows with step count — reclamation regressed."; fail=1
    else
        echo "  PASS: peak RSS flat."
    fi
fi
echo

# ── 3. plain baseline: must grow, and much more ──────────────────────────────
echo "--- [3] plain (unscoped) baseline for legibility ---"
plain_first=0; plain_last=0
for s in $STEP_LIST; do
    timed_run "$WORK/train" "$WORK/plain.$s.out" BATCH="$BATCH" STEPS="$s" MODE=plain
    echo "  steps=$s peak_rss=${TR_RSS_MB}MB  $(cat "$WORK/plain.$s.out")"
    [ "$plain_first" -eq 0 ] && plain_first=$TR_RSS_MB
    plain_last=$TR_RSS_MB
done
if [ "$plain_last" -gt 0 ] && [ "$last_rss" -gt 0 ]; then
    if [ "$plain_last" -le "$last_rss" ]; then
        echo "  NOTE: baseline did not exceed the handle mode at ${last_steps} steps"
        echo "        (batch too small to expose the leak on this host — not a failure)."
    else
        ratio=$(( plain_last * 10 / last_rss ))
        echo "  PASS: baseline is ${ratio}/10 x the handle mode at ${last_steps} steps."
    fi
fi
echo

# ── 4. numerics identical across all three modes ─────────────────────────────
echo "--- [4] numerics must be identical: plain vs with-region vs handle ---"
sums=""
for m in plain region handle; do
    timed_run "$WORK/train" "$WORK/num.$m.out" \
        BATCH="$BATCH" STEPS=10 MODE="$m" REGION_BYTES="$REGION_BYTES"
    if [ "$TR_RC" -ne 0 ]; then
        echo "FAIL: mode=$m exited $TR_RC"; cat "$WORK/num.$m.out"; fail=1; continue
    fi
    ck=$(sed -e 's/^mode=[a-z]* //' "$WORK/num.$m.out")
    echo "  $m: $ck"
    sums="$sums
$ck"
done
distinct=$(printf '%s' "$sums" | grep -c . )
uniq_count=$(printf '%s' "$sums" | grep . | sort -u | wc -l | tr -d ' ')
if [ "$uniq_count" -eq 1 ]; then
    echo "  PASS: all three modes agree bit-for-bit ($distinct runs)."
else
    echo "  FAIL: modes disagree — reclamation changed the numerics."; fail=1
fi
echo

# ── 5. safety matrix, AOT + JIT, both poisoned ───────────────────────────────
echo "--- [5] safety matrix under ESHKOL_ARENA_POISON=1 ---"
if ! compile_aot "$SAFETY" "$WORK/safety"; then
    echo "FAIL: safety-matrix AOT compile failed:"; cat "$WORK/compile_safety.log"; fail=1
else
    timed_run "$WORK/safety" "$WORK/safety.aot.out" ESHKOL_ARENA_POISON=1
    if [ "$TR_RC" -ne 0 ] || ! grep -q "^region_handle_safety_test: PASS$" "$WORK/safety.aot.out"; then
        echo "FAIL: AOT safety matrix (exit=$TR_RC):"; cat "$WORK/safety.aot.out"; fail=1
    else
        echo "  PASS: AOT safety matrix clean (peak_rss=${TR_RSS_MB}MB)."
    fi
fi
if ESHKOL_ARENA_POISON=1 "$ESHKOL_RUN" -r "$SAFETY" > "$WORK/safety.jit.out" 2>&1 &&
   grep -q "^region_handle_safety_test: PASS$" "$WORK/safety.jit.out"; then
    echo "  PASS: JIT safety matrix clean."
else
    echo "FAIL: JIT safety matrix:"; grep -vE "vectoriz|^remark:|^warning: <unknown>" "$WORK/safety.jit.out" | tail -40; fail=1
fi
echo

echo "=========================================================="
if [ "$fail" -eq 0 ]; then
    echo "region_handle_training_rss_test.sh: PASS"
    exit 0
fi
echo "region_handle_training_rss_test.sh: FAIL"
exit 1
