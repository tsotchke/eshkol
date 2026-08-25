#!/usr/bin/env bash
# tests/memory/resident_longrun_flat_gate.sh — SW-53 long-horizon residency gate.
#
# WHY THIS EXISTS, GIVEN THE FOUR FLAT-RSS GATES THAT ALREADY DO
#
# Every pre-existing flat-memory gate in this directory stops at 100 000 ticks
# and asserts a peak-RSS CEILING (iter_scope_partial_reclaim_test.sh: 150 MB at
# 100k; define_loop_flat_rss_aot_test.sh; region_mutating_loop_flat_rss_test.sh;
# vm_region_flat_rss_test.sh). Both of those choices hide a linear leak:
#
#   1. ONE POINT IS NOT A CURVE. A ceiling at a single tick count cannot tell
#      "flat" from "linear with a small slope". SW-53 leaked 48 bytes/tick — 4.6
#      MB over 100k ticks, invisible under a 150 MB ceiling, but 3.1 GB over a
#      week-long daemon run. The public benchmark suite found it only by sweeping
#      to 400k. This gate therefore measures at TWO horizons and gates on the
#      DIFFERENCE — the slope — not on either absolute value.
#
#   2. PEAK RSS IS THE WRONG INSTRUMENT TO GATE ON. `maximum resident set size`
#      is a high-water mark of INSTANTANEOUS residency, so on a loaded host the
#      memory compressor evicts pages and the recorded maximum comes back LOWER
#      than what the process actually retains. Measuring the SW-53 repro on a
#      24-core box at load average ~200 produced 97 MB and 193 MB for the same
#      binary on consecutive runs. A leak gate built on that number is quietest
#      exactly when CI is busiest. This gate reads the arena's own byte counter
#      (ESHKOL_ARENA_REPORT=1 -> global_total_allocated_bytes) instead: exact,
#      deterministic to the byte, and unaffected by system load. Peak RSS is
#      still reported per point, as the number the public claim is phrased in,
#      but it is ADVISORY here.
#
# WHAT IS GATED
#
#   A. ZERO-GROWTH CHANNELS (hard gate, exact equality). A resident loop whose
#      per-tick allocation is transient, and whose persistent stores publish
#      values that are not freshly allocated each tick, must retain EXACTLY the
#      same number of arena bytes at the long horizon as at the short one — not
#      "within a factor", byte-identical. One row per barriered mutation channel
#      (vector-set! / hash-table-set! / set-cdr! / set! of a global), each inside
#      the catch-all `guard` error boundary that makes a daemon loop resident,
#      because SW-53's leak was one 48-byte handler frame per guard ENTRY and so
#      appeared only in the guarded shape.
#
#   B. PUBLISHED-BYTES RATE (hard gate, ceiling on the SLOPE). A loop that
#      allocates a fresh heap object every tick and publishes it into persistent
#      state legitimately retains that object: its predecessor is dead the moment
#      it is overwritten, and without a tracing collector those bytes cannot come
#      back. That is a documented design limit (docs/reference/runtime/memory-
#      model.md, "What is flat and what is not"), not a leak — but its RATE must
#      stay pinned to what the program actually publishes, or a future
#      per-iteration leak would hide inside it. The ESH-0214e fixture shape
#      publishes 5 cons cells per tick and measures 240.0 bytes/tick exactly
#      (48 bytes/cons); the ceiling below allows headroom for allocator layout
#      changes while still failing on one extra retained word per tick.
#
# Usage: resident_longrun_flat_gate.sh [--short N] [--long N] [--timeout S]
#                                      [--publish-ceiling-bpt N]
#   BUILD_DIR selects the build directory (default: build).
#   ESHKOL_RUN overrides the eshkol-run binary path.
#   --long 6400000 is the nightly horizon (see .github/workflows/adversarial-nightly.yml).
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
. "$REPO_ROOT/scripts/lib/durable_work_root.sh"

BUILD_DIR="${BUILD_DIR:-build}"
if [ -z "${ESHKOL_RUN:-}" ]; then
    case "$BUILD_DIR" in
        /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
        *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
    esac
fi
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "resident_longrun_flat_gate.sh: $ESHKOL_RUN not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

SHORT_TICKS=200000
LONG_TICKS=1600000
TIMEOUT_S=300
PUBLISH_CEILING_BPT=320     # measured 240.0 exactly; see block comment B
while [ $# -gt 0 ]; do
    case "$1" in
        --short) shift; SHORT_TICKS="${1:?}" ;;
        --long) shift; LONG_TICKS="${1:?}" ;;
        --timeout) shift; TIMEOUT_S="${1:?}" ;;
        --publish-ceiling-bpt) shift; PUBLISH_CEILING_BPT="${1:?}" ;;
        *) echo "resident_longrun_flat_gate.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
if [ "$LONG_TICKS" -le "$SHORT_TICKS" ]; then
    echo "resident_longrun_flat_gate.sh: --long ($LONG_TICKS) must exceed --short ($SHORT_TICKS)." >&2
    exit 2
fi

if eshkol_durable_enabled; then
    WORK="$(eshkol_durable_prepare_dir resident-longrun-flat)" || exit $?
else
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-rlf.XXXXXX")"
    trap 'rm -rf "$WORK"' EXIT
fi

# Disk cap: fixtures + binaries only, but a horizon knob invites accidents.
DISK_CAP_MB="${ESHKOL_GATE_DISK_CAP_MB:-1024}"
disk_cap_check() {
    local used
    used=$(du -sm "$WORK" 2>/dev/null | awk '{print $1}')
    if [ "${used:-0}" -gt "$DISK_CAP_MB" ]; then
        echo "resident_longrun_flat_gate.sh: work dir exceeded ${DISK_CAP_MB}MB (${used}MB) — aborting." >&2
        exit 3
    fi
}

# Peak-RSS-reporting `time` flavor (advisory only — see block comment 2).
TIME_MODE=""
if /usr/bin/time -l true >/dev/null 2>"$WORK/.probe"; then
    grep -q "maximum resident set size" "$WORK/.probe" 2>/dev/null && TIME_MODE="bsd"
fi
if [ -z "$TIME_MODE" ] && /usr/bin/time -v true >"$WORK/.probe" 2>&1; then
    grep -qi "Maximum resident set size" "$WORK/.probe" 2>/dev/null && TIME_MODE="gnu"
fi

# ── fixtures ────────────────────────────────────────────────────────────────
# Every fixture is the resident daemon shape this repo documents: a self-tail
# `define` loop inside a catch-all `guard`. Only the per-tick body differs, so
# the difference between two rows isolates one channel.
emit_fixture() { # <name> <ticks> -> path
    local name="$1" ticks="$2" path="$WORK/${1}_${2}.esk" prelude body
    prelude=""
    case "$name" in
      transient)
        body='(let ((s (make-vector 200 i))) (vector-length s))' ;;
      vector_set_imm)
        prelude='(define ws (make-vector 64 0))'
        body='(begin (vector-set! ws (modulo i 64) i) i)' ;;
      hash_set_imm)
        prelude='(define kb (make-hash-table))'
        body='(begin (hash-table-set! kb (modulo i 256) i) i)' ;;
      set_cdr_imm)
        prelude='(define cell (cons 0 0))'
        body='(begin (set-cdr! cell i) i)' ;;
      set_bang_imm)
        prelude='(define slot 0)'
        body='(begin (set! slot i) i)' ;;
      publish_five_cons)
        prelude='(define kb (make-hash-table))
(define ws (make-vector 64 0))'
        body='(let ((s (make-vector 200 i)))
             (hash-table-set! kb (modulo i 256) (list i (* i 2) (vector-length s)))
             (vector-set! ws (modulo i 64) (list i (* i 3)))
             i)' ;;
      *) echo "resident_longrun_flat_gate.sh: unknown fixture $name" >&2; exit 2 ;;
    esac
    cat > "$path" <<EOF
;; generated by tests/memory/resident_longrun_flat_gate.sh — $name @ $ticks ticks
(define ticks $ticks)
$prelude
(define (tick i)
  (guard (e (#t i))
    (if (>= i ticks)
        i
        (begin
          $body
          (tick (+ i 1))))))
(define result (tick 0))
(if (= result ticks)
    (begin (display "PASS") (newline))
    (begin (display "FAIL result=") (display result) (newline) (exit 1)))
EOF
    printf '%s\n' "$path"
}

# measure <name> <ticks> -> MEAS_ARENA_BYTES MEAS_RSS_MB MEAS_RC
measure() {
    local name="$1" ticks="$2" src bin tlog out
    src="$(emit_fixture "$name" "$ticks")"
    bin="$WORK/${name}_${ticks}.bin"
    out="$WORK/${name}_${ticks}.out"
    tlog="$WORK/${name}_${ticks}.time"
    if ! ( cd "$WORK" && "$ESHKOL_RUN" "$src" -o "$bin" ) > "$WORK/${name}_${ticks}.compile.log" 2>&1; then
        MEAS_ARENA_BYTES=""; MEAS_RSS_MB=0; MEAS_RC=127
        return
    fi
    chmod +x "$bin"
    local timer=(env ESHKOL_ARENA_REPORT=1)
    if [ "$TIME_MODE" = "bsd" ]; then
        timer=(env ESHKOL_ARENA_REPORT=1 /usr/bin/time -l)
    elif [ "$TIME_MODE" = "gnu" ]; then
        timer=(env ESHKOL_ARENA_REPORT=1 /usr/bin/time -v)
    fi
    ( cd "$WORK" && "${timer[@]}" perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec: $!\n"' \
        "$TIMEOUT_S" "$bin" ) > "$out" 2> "$tlog"
    MEAS_RC=$?
    grep -q "^PASS$" "$out" 2>/dev/null || MEAS_RC=$(( MEAS_RC == 0 ? 126 : MEAS_RC ))
    MEAS_ARENA_BYTES=$(awk -F= '/global_total_allocated_bytes/{print $2}' "$tlog" | tail -1)
    if [ "$TIME_MODE" = "bsd" ]; then
        MEAS_RSS_MB=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$tlog")
    elif [ "$TIME_MODE" = "gnu" ]; then
        MEAS_RSS_MB=$(awk -F: '/Maximum resident set size/{printf "%d", $2/1024}' "$tlog")
    else
        MEAS_RSS_MB=0
    fi
    [ -n "$MEAS_RSS_MB" ] || MEAS_RSS_MB=0
    rm -f "$bin" "$src"
    disk_cap_check
}

echo "=========================================================="
echo "  SW-53 long-horizon residency gate"
echo "  short=${SHORT_TICKS} ticks   long=${LONG_TICKS} ticks"
echo "  signal=global arena bytes (exact)   rss=${TIME_MODE:-unavailable} (advisory)"
echo "=========================================================="
echo

fail=0
DELTA_TICKS=$(( LONG_TICKS - SHORT_TICKS ))

# ── A. zero-growth channels: byte-identical retention across an 8x horizon ──
echo "--- [A] zero-growth channels (must retain EXACTLY the same arena bytes) ---"
printf '  %-20s %14s %14s %12s %10s\n' fixture "arena@short" "arena@long" "bytes/tick" "rss@long"
for name in transient vector_set_imm hash_set_imm set_cdr_imm set_bang_imm; do
    measure "$name" "$SHORT_TICKS"; a_short="$MEAS_ARENA_BYTES"; rc_short="$MEAS_RC"
    measure "$name" "$LONG_TICKS";  a_long="$MEAS_ARENA_BYTES";  rc_long="$MEAS_RC"; rss_long="$MEAS_RSS_MB"
    if [ "$rc_short" -ne 0 ] || [ "$rc_long" -ne 0 ]; then
        printf '  %-20s %s\n' "$name" "FAIL: run failed (rc short=$rc_short long=$rc_long)"
        fail=1; continue
    fi
    if [ -z "$a_short" ] || [ -z "$a_long" ]; then
        printf '  %-20s %s\n' "$name" "FAIL: no ESHKOL_ARENA_REPORT line — arena probe missing from this build"
        fail=1; continue
    fi
    bpt=$(awk -v d="$(( a_long - a_short ))" -v t="$DELTA_TICKS" 'BEGIN{printf "%.3f", d/t}')
    printf '  %-20s %14s %14s %12s %9sMB' "$name" "$a_short" "$a_long" "$bpt" "$rss_long"
    if [ "$a_long" -ne "$a_short" ]; then
        echo "   FAIL"
        echo "      $name retains $(( a_long - a_short )) more arena bytes at ${LONG_TICKS} ticks than at"
        echo "      ${SHORT_TICKS} (${bpt} bytes/tick). This loop publishes nothing freshly"
        echo "      allocated, so its steady-state retention must be exactly zero."
        fail=1
    else
        echo "   ok"
    fi
done
echo

# ── B. published-bytes rate stays pinned to what the program publishes ──────
echo "--- [B] published-bytes rate (ESH-0214e fixture shape, 5 fresh cons/tick) ---"
measure publish_five_cons "$SHORT_TICKS"; p_short="$MEAS_ARENA_BYTES"; prc_short="$MEAS_RC"
measure publish_five_cons "$LONG_TICKS";  p_long="$MEAS_ARENA_BYTES";  prc_long="$MEAS_RC"; p_rss="$MEAS_RSS_MB"
if [ "$prc_short" -ne 0 ] || [ "$prc_long" -ne 0 ] || [ -z "$p_short" ] || [ -z "$p_long" ]; then
    echo "  FAIL: publish_five_cons did not run cleanly (rc short=$prc_short long=$prc_long)."
    fail=1
else
    pbt=$(awk -v d="$(( p_long - p_short ))" -v t="$DELTA_TICKS" 'BEGIN{printf "%.1f", d/t}')
    echo "  arena@short=${p_short}  arena@long=${p_long}  rate=${pbt} bytes/tick  rss@long=${p_rss}MB"
    over=$(awk -v r="$pbt" -v c="$PUBLISH_CEILING_BPT" 'BEGIN{print (r>c)?1:0}')
    if [ "$over" -eq 1 ]; then
        echo "  FAIL: ${pbt} bytes/tick exceeds the ${PUBLISH_CEILING_BPT} bytes/tick ceiling."
        echo "        The fixture publishes 5 cons cells per tick (240.0 B/tick at 48 B/cons);"
        echo "        anything above the ceiling is retention this program did not ask for."
        fail=1
    else
        echo "  PASS: within the ${PUBLISH_CEILING_BPT} bytes/tick ceiling (published bytes only)."
    fi
fi
echo

if [ "$fail" -eq 0 ]; then
    echo "resident_longrun_flat_gate.sh: PASS"
else
    echo "resident_longrun_flat_gate.sh: FAIL"
fi
exit "$fail"
