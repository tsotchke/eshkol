#!/usr/bin/env bash
# tests/memory/region_callcc_flat_rss_test.sh — SW-74 region-pin lifecycle gate.
#
# WHAT WENT WRONG
#
# `codegenCallCC` classifies every capture as escape-only or possibly-escaping
# (callCCContinuationStaysLocal, lib/backend/llvm_codegen.cpp) and used that
# classification for ONE decision: whether to snapshot the C stack. The region
# pin was taken unconditionally, from eshkol_make_continuation_state(), on
# nothing but "is a region open". A pinned region's arena was then not freed at
# all — region_destroy skipped arena_destroy and dropped the block chain.
#
# So `(with-region (call/cc (lambda (k) ... (k v))))` in a loop leaked one whole
# region arena per iteration, forever, for a continuation the compiler had
# already PROVEN could not outlive its frame. SW-59's own measurements scoped
# with-region out ("no `with-region` involved"), so the combination was ungated.
#
# WHAT THIS GATE PINS
#
#   A. escape_only_region — an escape-only `call/cc` inside `with-region`, in a
#      resident loop, retains EXACTLY the same number of arena bytes at an 8x
#      horizon as at the short one. Not "within a factor": byte-identical,
#      0.000 bytes/tick. This is the SW-74 repro; before the fix it grew without
#      bound. The signal is the arena's own counter (ESHKOL_ARENA_REPORT=1),
#      not peak RSS — see the block comment in resident_longrun_flat_gate.sh for
#      why peak RSS is the wrong instrument.
#
#   B. escape_only_no_region — the same loop with NO `with-region`, as the
#      instrument check. Every native `call/cc` allocates its continuation state
#      and closure from the current arena; with no region open that arena is the
#      process arena and the bytes stay for the run. So B must show STRICTLY
#      POSITIVE growth. That is what makes A's exact zero mean something: the
#      counter can see call/cc allocations, and in A the region reclaimed them.
#
#   C. escaping_region_bounded — a capture that DOES escape (it is stored in a
#      global) still pins, by design, and its region is promoted into the
#      enclosing arena rather than reclaimed. That is a documented cost, not a
#      leak: the ceiling here says the promoted bytes stay proportional to what
#      the region actually allocated, so a future regression that promotes more
#      than the region held is caught. It also fails if the promotion ever costs
#      NOTHING, which would mean the pin stopped working.
#
#   D. answers — every fixture prints the answer it is supposed to, at both
#      horizons. A flat memory curve bought with a wrong result is not a pass.
#
#   E. no_pin_note — the escape-only fixture prints no region-pin note on
#      stderr (eshkol_region_pin_notice()), and the escaping one does. The note
#      is the user-visible statement that a region was retained, so it must
#      track the pin exactly.
#
#   F. handle_close_inside_callcc — the carve-out, under ESHKOL_ARENA_POISON=1.
#      An escape-only capture skips the pin because no LEXICAL `with-region` can
#      be torn down inside the capture's extent. `(region-close h)` can: it is
#      an ordinary call and runs anywhere. So a handle-owned open region makes
#      an escape-only capture pin after all, and
#      tests/continuations/region_handle_close_inside_callcc.esk closes a handle
#      from inside the `call/cc` procedure and then invokes the continuation.
#      With the arena poisoned, losing that carve-out reads 0xCB and crashes
#      instead of being accidentally right.
#
# Usage: tests/memory/region_callcc_flat_rss_test.sh [--short N] [--long N]
#                                                    [--timeout S]
#   BUILD_DIR selects the build directory (default: build).
#   ESHKOL_RUN overrides the eshkol-run binary path.
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
. "$REPO_ROOT/scripts/lib/durable_work_root.sh"

BUILD_DIR="${BUILD_DIR:-build}"
if [ -z "${ESHKOL_RUN:-}" ]; then
    case "$BUILD_DIR" in
        /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
        *)  ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
    esac
fi
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "region_callcc_flat_rss_test.sh: $ESHKOL_RUN not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

SHORT_TICKS=100000
LONG_TICKS=800000
ESCAPING_TICKS=500         # C promotes a region per tick: keep it small
TIMEOUT_S=300
while [ $# -gt 0 ]; do
    case "$1" in
        --short) shift; SHORT_TICKS="${1:?}" ;;
        --long) shift; LONG_TICKS="${1:?}" ;;
        --timeout) shift; TIMEOUT_S="${1:?}" ;;
        *) echo "region_callcc_flat_rss_test.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
if [ "$LONG_TICKS" -le "$SHORT_TICKS" ]; then
    echo "region_callcc_flat_rss_test.sh: --long ($LONG_TICKS) must exceed --short ($SHORT_TICKS)." >&2
    exit 2
fi

if eshkol_durable_enabled; then
    WORK="$(eshkol_durable_prepare_dir region-callcc-flat)" || exit $?
else
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-rcf.XXXXXX")"
    trap 'rm -rf "$WORK"' EXIT INT TERM
fi

# Disk cap: fixtures and one binary at a time, but a horizon knob invites
# accidents (see feedback on the P7b harness that filled a 58 GB volume).
DISK_CAP_MB="${ESHKOL_GATE_DISK_CAP_MB:-512}"
disk_cap_check() {
    local used
    used=$(du -sm "$WORK" 2>/dev/null | awk '{print $1}')
    if [ "${used:-0}" -gt "$DISK_CAP_MB" ]; then
        echo "region_callcc_flat_rss_test.sh: work dir exceeded ${DISK_CAP_MB}MB (${used}MB) — aborting." >&2
        exit 3
    fi
}

# ── fixtures ────────────────────────────────────────────────────────────────
#
# The per-tick body is the only thing that differs between A and B, and the only
# thing that differs between A and C is whether the captured continuation is
# stored. `probe` keeps `k` in operator position so the compiler classifies it
# escape-only; `probe` in fixture C stores it, which is a bare reference and is
# classified as escaping.
emit_fixture() { # <name> <ticks> -> path
    local name="$1" ticks="$2" path="$WORK/${1}_${2}.esk" prelude probe
    prelude=""
    case "$name" in
      escape_only_region)
        probe='(define (probe i)
  (with-region
    (call/cc (lambda (k) (if (> i 0) (k i) 0)))))' ;;
      escape_only_no_region)
        probe='(define (probe i)
  (call/cc (lambda (k) (if (> i 0) (k i) 0))))' ;;
      escaping_region)
        prelude='(define held #f)'
        probe='(define (probe i)
  (with-region
    (call/cc (lambda (k) (set! held k) i))))' ;;
      *) echo "region_callcc_flat_rss_test.sh: unknown fixture $name" >&2; exit 2 ;;
    esac
    cat > "$path" <<EOF
;; generated by tests/memory/region_callcc_flat_rss_test.sh — $name @ $ticks ticks
(define ticks $ticks)
$prelude
$probe
(define (tick i)
  (if (>= i ticks)
      i
      (if (= (probe i) i) (tick (+ i 1)) (- 0 i))))
(define result (tick 0))
(if (= result ticks)
    (begin (display "PASS") (newline))
    (begin (display "FAIL result=") (display result) (newline) (exit 1)))
EOF
    printf '%s\n' "$path"
}

# measure <name> <ticks> -> MEAS_ARENA_BYTES MEAS_RC MEAS_ERR
measure() {
    local name="$1" ticks="$2" src bin
    src="$(emit_fixture "$name" "$ticks")"
    bin="$WORK/${name}_${ticks}.bin"
    MEAS_ERR="$WORK/${name}_${ticks}.err"
    local out="$WORK/${name}_${ticks}.out"
    if ! ( cd "$WORK" && "$ESHKOL_RUN" "$src" -o "$bin" ) > "$WORK/${name}_${ticks}.compile.log" 2>&1; then
        MEAS_ARENA_BYTES=""; MEAS_RC=127
        return
    fi
    chmod +x "$bin"
    ( cd "$WORK" && env ESHKOL_ARENA_REPORT=1 \
        perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec: $!\n"' \
        "$TIMEOUT_S" "$bin" ) > "$out" 2> "$MEAS_ERR"
    MEAS_RC=$?
    grep -q "^PASS$" "$out" 2>/dev/null || MEAS_RC=$(( MEAS_RC == 0 ? 126 : MEAS_RC ))
    MEAS_ARENA_BYTES=$(awk -F= '/global_total_allocated_bytes/{print $2}' "$MEAS_ERR" | tail -1)
    rm -f "$bin" "$src"
    disk_cap_check
}

PASS=0
FAIL=0
check() { # name failure-description result(0=ok)
    if [ "$3" -eq 0 ]; then
        echo "PASSED tests/memory/region_callcc_flat_rss_test.sh::$1"
        PASS=$((PASS + 1))
    else
        echo "FAILED tests/memory/region_callcc_flat_rss_test.sh::$1 — $2"
        FAIL=$((FAIL + 1))
    fi
}

echo "=========================================================="
echo "  SW-74 continuation region-pin lifecycle gate"
echo "  short=${SHORT_TICKS} ticks   long=${LONG_TICKS} ticks"
echo "  signal=global arena bytes (exact, ESHKOL_ARENA_REPORT=1)"
echo "=========================================================="
echo

DELTA_TICKS=$(( LONG_TICKS - SHORT_TICKS ))
NOTE_SUBSTRING="could not be reclaimed because a continuation was captured inside it"

# ── A. escape-only capture inside a region: byte-identical across 8x ────────
printf '  %-24s %14s %14s %12s\n' fixture "arena@short" "arena@long" "bytes/tick"
measure escape_only_region "$SHORT_TICKS"; a_short="$MEAS_ARENA_BYTES"; rc_short="$MEAS_RC"
a_short_err="$MEAS_ERR"
measure escape_only_region "$LONG_TICKS";  a_long="$MEAS_ARENA_BYTES";  rc_long="$MEAS_RC"
if [ "$rc_short" -ne 0 ] || [ "$rc_long" -ne 0 ] || [ -z "$a_short" ] || [ -z "$a_long" ]; then
    check "escape_only_region_answer" "run failed (rc short=$rc_short long=$rc_long)" 1
    check "escape_only_region_flat" "not measured: the run failed or printed no ESHKOL_ARENA_REPORT line" 1
    check "escape_only_region_no_pin_note" "not measured: the run failed" 1
    a_bpt="n/a"
else
    check "escape_only_region_answer" "" 0
    a_bpt=$(awk -v d="$(( a_long - a_short ))" -v t="$DELTA_TICKS" 'BEGIN{printf "%.3f", d/t}')
    printf '  %-24s %14s %14s %12s\n' escape_only_region "$a_short" "$a_long" "$a_bpt"
    if [ "$a_long" -ne "$a_short" ]; then
        echo "      escape_only_region retains $(( a_long - a_short )) more arena bytes at"
        echo "      ${LONG_TICKS} ticks than at ${SHORT_TICKS} (${a_bpt} bytes/tick). An escape-only"
        echo "      continuation cannot outlive the frame that captured it, so nothing it"
        echo "      touches may survive the iteration that created it."
        check "escape_only_region_flat" "retention grows with the horizon (${a_bpt} bytes/tick)" 1
    else
        check "escape_only_region_flat" "" 0
    fi
    if grep -qF "$NOTE_SUBSTRING" "$a_short_err"; then rc=1; else rc=0; fi
    check "escape_only_region_no_pin_note" "printed the region-pin note for a capture that must not pin" $rc
fi
echo

# ── B. the same loop with no region: the instrument check ───────────────────
measure escape_only_no_region "$SHORT_TICKS"; b_short="$MEAS_ARENA_BYTES"; brc_short="$MEAS_RC"
measure escape_only_no_region "$LONG_TICKS";  b_long="$MEAS_ARENA_BYTES";  brc_long="$MEAS_RC"
if [ "$brc_short" -ne 0 ] || [ "$brc_long" -ne 0 ] || [ -z "$b_short" ] || [ -z "$b_long" ]; then
    check "escape_only_no_region_answer" "run failed (rc short=$brc_short long=$brc_long)" 1
    check "escape_only_no_region_visible" "not measured: the run failed" 1
else
    check "escape_only_no_region_answer" "" 0
    b_bpt=$(awk -v d="$(( b_long - b_short ))" -v t="$DELTA_TICKS" 'BEGIN{printf "%.3f", d/t}')
    printf '  %-24s %14s %14s %12s\n' escape_only_no_region "$b_short" "$b_long" "$b_bpt"
    if [ "$b_long" -gt "$b_short" ]; then rc=0; else rc=1; fi
    check "escape_only_no_region_visible" \
        "with no region open, a per-capture continuation state must accumulate in the process arena; measuring zero means the arena counter cannot see call/cc allocations at all, which would make the ${a_bpt} bytes/tick above meaningless" $rc
fi
echo

# ── C. an escaping capture still pins, and the pin still costs something ────
measure escaping_region "$ESCAPING_TICKS"; e_bytes="$MEAS_ARENA_BYTES"; e_rc="$MEAS_RC"
e_err="$MEAS_ERR"
measure escaping_region "$(( ESCAPING_TICKS / 2 ))"; e_half="$MEAS_ARENA_BYTES"; e_half_rc="$MEAS_RC"
if [ "$e_rc" -ne 0 ] || [ "$e_half_rc" -ne 0 ] || [ -z "$e_bytes" ] || [ -z "$e_half" ]; then
    check "escaping_region_answer" "run failed (rc=$e_rc half=$e_half_rc)" 1
    check "escaping_region_pins" "not measured: the run failed" 1
else
    check "escaping_region_answer" "" 0
    echo "  escaping_region: arena@$(( ESCAPING_TICKS / 2 ))=${e_half}  arena@${ESCAPING_TICKS}=${e_bytes}"
    # A capture that escapes pins every open region, and a pinned region is
    # promoted into the enclosing arena. That MUST show up as retention: if it
    # does not, the pin has stopped being taken and the next escaping
    # continuation will read a reclaimed arena.
    if [ "$e_bytes" -gt "$e_half" ]; then rc=0; else rc=1; fi
    check "escaping_region_pins" "an escaping capture inside with-region retained nothing — the pin is not being taken" $rc
    if grep -qF "$NOTE_SUBSTRING" "$e_err"; then rc=0; else rc=1; fi
    check "escaping_region_pin_note" "an escaping capture inside with-region printed no region-pin note" $rc
fi
echo

# ── F. the handle carve-out, with the arena poisoned ────────────────────────
HANDLE_SRC="$REPO_ROOT/tests/continuations/region_handle_close_inside_callcc.esk"
HANDLE_EXPECTED="$REPO_ROOT/tests/continuations/expected/region_handle_close_inside_callcc.txt"
if [ ! -f "$HANDLE_SRC" ] || [ ! -f "$HANDLE_EXPECTED" ]; then
    check "handle_close_inside_callcc_jit" "fixture or expected transcript missing" 1
    check "handle_close_inside_callcc_aot" "fixture or expected transcript missing" 1
else
    want=$(tr -d '\n' < "$HANDLE_EXPECTED")

    ( cd "$WORK" && env ESHKOL_ARENA_POISON=1 "$ESHKOL_RUN" -r "$HANDLE_SRC" ) \
        > "$WORK/handle_jit.out" 2> "$WORK/handle_jit.err"
    jit_rc=$?
    got=$(tr -d '\n' < "$WORK/handle_jit.out")
    if [ "$jit_rc" -eq 0 ] && [ "$got" = "$want" ]; then rc=0; else rc=1; fi
    check "handle_close_inside_callcc_jit" \
        "native JIT under ESHKOL_ARENA_POISON=1 gave rc=$jit_rc and \"$got\" (want \"$want\") — a region handle closed inside a call/cc extent must still pin" $rc

    HANDLE_BIN="$WORK/handle_close_inside_callcc"
    if ( cd "$WORK" && "$ESHKOL_RUN" "$HANDLE_SRC" -o "$HANDLE_BIN" ) \
            > "$WORK/handle_compile.log" 2>&1 && [ -x "$HANDLE_BIN" ]; then
        ( cd "$WORK" && env ESHKOL_ARENA_POISON=1 "$HANDLE_BIN" ) \
            > "$WORK/handle_aot.out" 2> "$WORK/handle_aot.err"
        aot_rc=$?
        got=$(tr -d '\n' < "$WORK/handle_aot.out")
        if [ "$aot_rc" -eq 0 ] && [ "$got" = "$want" ]; then rc=0; else rc=1; fi
        check "handle_close_inside_callcc_aot" \
            "native AOT under ESHKOL_ARENA_POISON=1 gave rc=$aot_rc and \"$got\" (want \"$want\")" $rc
        rm -f "$HANDLE_BIN"
    else
        check "handle_close_inside_callcc_aot" "AOT compile failed" 1
    fi
    disk_cap_check
fi
echo

echo "  region-callcc-flat-rss: $PASS passed, $FAIL failed"
if [ "$FAIL" -eq 0 ]; then
    echo "region_callcc_flat_rss_test.sh: PASS"
    exit 0
fi
echo "region_callcc_flat_rss_test.sh: FAIL"
exit 1
