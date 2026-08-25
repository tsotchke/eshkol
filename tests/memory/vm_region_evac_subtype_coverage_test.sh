#!/usr/bin/env bash
# tests/memory/vm_region_evac_subtype_coverage_test.sh
#
# VM counterpart of region_evac_subtype_coverage_test.sh, and the coverage half
# of the SW-14 close.
#
# WHAT IT PINS. The Stage-1 VM region evacuator (lib/backend/vm_region_evac.c)
# reclaims a `with-region` body by MARKING from the VM root set and releasing
# the arena blocks nothing needs. Payload buffers are covered conservatively by
# a pointer scan, so exactly one class of mistake is dangerous: a heap-object
# INDEX reachable only through a field the per-subtype walk forgot. An index is
# a small integer, invisible to any pointer scan, so a missed field frees a LIVE
# object. This gate builds one instance of every subtype a Scheme program can
# construct inside a region, escapes it, and reads its CONTENTS back afterwards.
#
# It runs the fixture in four configurations, because each can fail on its own:
#
#   1. poison    ESHKOL_ARENA_POISON=1 — dead blocks are stamped 0xCB and kept
#                mapped, and retired object indices are NOT recycled, so a
#                missed reference reads a cleared slot instead of aliasing a
#                freshly allocated object. This is the configuration in which a
#                coverage hole is loud rather than lucky.
#   2. verify    ESHKOL_VM_REGION_VERIFY_FATAL=1 — after each pop, an audit
#                independent of the mark scans the object table for a surviving
#                reference to an index being retired, and exits nonzero if it
#                finds one.
#   3. default   reclamation on, compaction on: the configuration users get.
#   4. off       ESHKOL_VM_REGION_EVAC=0: the same answers with reclamation
#                disabled, which is what makes "the evacuator changed no answer"
#                a measured claim rather than an assertion.
#
# Exit 0 = PASS, 1 = FAIL, 2 = cannot gate (missing binary, missing fixture).
#
# Usage: tests/memory/vm_region_evac_subtype_coverage_test.sh [--ceiling-mb N] [--timeout S]
#   BUILD_DIR selects the build directory (default: build).
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
. "$REPO_ROOT/scripts/lib/durable_work_root.sh"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) VM="$BUILD_DIR/eshkol-vm-standalone-test" ;;
    *)  VM="$REPO_ROOT/$BUILD_DIR/eshkol-vm-standalone-test" ;;
esac
if [ ! -x "$VM" ]; then
    echo "vm_region_evac_subtype_coverage_test.sh: $VM not found — run \`cmake --build $BUILD_DIR\` first." >&2
    exit 2
fi

SRC="$REPO_ROOT/tests/memory/vm_region_evac_subtype_coverage_test.esk"
if [ ! -f "$SRC" ]; then
    echo "vm_region_evac_subtype_coverage_test.sh: $SRC not found." >&2
    exit 2
fi

# Poison mode neither frees nor compacts, so it holds the whole run's arena;
# 260 MB clears its measured ~130 MB with wide margin while still catching a
# regression to the pre-evacuator behaviour, which on this fixture is unbounded.
CEILING_MB=260
TIMEOUT_S=300
while [ $# -gt 0 ]; do
    case "$1" in
        --ceiling-mb) CEILING_MB="${2:-}"; shift 2 ;;
        --timeout)    TIMEOUT_S="${2:-}"; shift 2 ;;
        *) echo "vm_region_evac_subtype_coverage_test.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done

# Peak-RSS reporting differs by platform; detect once, refuse to gate if neither.
#   macOS (BSD time): `/usr/bin/time -l` prints "N  maximum resident set size" in BYTES.
#   Linux (GNU time): `/usr/bin/time -v` prints "Maximum resident set size (kbytes): N".
TIME_MODE=""
if /usr/bin/time -l true >/dev/null 2>/tmp/.eshkol_vm_evac_probe.$$ &&
   grep -q "maximum resident set size" /tmp/.eshkol_vm_evac_probe.$$; then
    TIME_MODE=bsd
elif /usr/bin/time -v true >/dev/null 2>/tmp/.eshkol_vm_evac_probe.$$ &&
     grep -q "Maximum resident set size" /tmp/.eshkol_vm_evac_probe.$$; then
    TIME_MODE=gnu
fi
rm -f /tmp/.eshkol_vm_evac_probe.$$
if [ -z "$TIME_MODE" ]; then
    echo "vm_region_evac_subtype_coverage_test.sh: no usable /usr/bin/time; cannot gate." >&2
    exit 2
fi

if eshkol_durable_enabled; then
    WORK="$(eshkol_durable_prepare_dir vm-region-evac-subtype-coverage)" || exit $?
else
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-vm-region-evac.XXXXXX")"
    trap 'rm -rf "$WORK"' EXIT INT TERM
fi

PASS=0
FAIL=0
check() { # name description result(0=ok)
    if [ "$3" -eq 0 ]; then
        echo "PASSED tests/memory/vm_region_evac_subtype_coverage_test.sh::$1"
        PASS=$((PASS + 1))
    else
        echo "FAILED tests/memory/vm_region_evac_subtype_coverage_test.sh::$1 — $2"
        FAIL=$((FAIL + 1))
    fi
}

# `perl -e alarm` is the portable timeout wrapper the other memory gates use —
# GNU `timeout` is not present on every supported host.
RSS_MB=0
RUN_RC=0
run_case() { # tag env...
    tag="$1"; shift
    if [ "$TIME_MODE" = bsd ]; then
        env "$@" ESHKOL_VM_NO_DISASM=1 ESHKOL_VM_HEAP_BUDGET_MB=0 ESHKOL_VM_REGION_QUIET=1 \
            /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
            "$TIMEOUT_S" "$VM" "$SRC" >"$WORK/$tag.out" 2>"$WORK/$tag.time"
        RUN_RC=$?
        RSS_MB=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$WORK/$tag.time")
    else
        env "$@" ESHKOL_VM_NO_DISASM=1 ESHKOL_VM_HEAP_BUDGET_MB=0 ESHKOL_VM_REGION_QUIET=1 \
            /usr/bin/time -v perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
            "$TIMEOUT_S" "$VM" "$SRC" >"$WORK/$tag.out" 2>"$WORK/$tag.time"
        RUN_RC=$?
        RSS_MB=$(awk -F: '/Maximum resident set size/{printf "%d", $2/1024}' "$WORK/$tag.time")
    fi
    [ -n "$RSS_MB" ] || RSS_MB=0
}

echo "  SW-14 close: VM region evacuator subtype coverage (ceiling ${CEILING_MB} MB, time=$TIME_MODE)"

# ── 1. poison: a coverage hole must be loud, not lucky ──────────────────────
run_case poison ESHKOL_ARENA_POISON=1
POISON_RSS=$RSS_MB
if [ "$RUN_RC" -eq 0 ] && grep -q "^PASS$" "$WORK/poison.out"; then rc=0; else rc=1; fi
check poison_pass \
  "under ESHKOL_ARENA_POISON=1 a promoted value did not read back intact — a 0xcbcb.. crash or a #f flag means a subtype's interior references are not being walked" $rc
[ "$rc" -eq 0 ] || sed -n '1,40p' "$WORK/poison.out" "$WORK/poison.time"

# ── 2. the independent post-sweep audit finds no surviving reference ────────
run_case verify ESHKOL_VM_REGION_VERIFY=1 ESHKOL_VM_REGION_VERIFY_FATAL=1
if [ "$RUN_RC" -eq 0 ] && grep -q "^PASS$" "$WORK/verify.out"; then rc=0; else rc=1; fi
check audit_clean \
  "the post-sweep audit found a live reference to a retired heap index (see REGION EVACUATOR AUDIT on stderr)" $rc
if grep -q "REGION EVACUATOR AUDIT" "$WORK/verify.time"; then
    check audit_silent "the audit reported a dangling reference" 1
else
    check audit_silent "" 0
fi

# ── 3. the shipping configuration ───────────────────────────────────────────
run_case default
DEFAULT_RSS=$RSS_MB
if [ "$RUN_RC" -eq 0 ] && grep -q "^PASS$" "$WORK/default.out"; then rc=0; else rc=1; fi
check default_pass "the default configuration did not read every promoted value back intact" $rc

# ── 4. reclamation changes no answer ────────────────────────────────────────
run_case off ESHKOL_VM_REGION_EVAC=0
OFF_RSS=$RSS_MB
if [ "$RUN_RC" -eq 0 ] && grep -q "^PASS$" "$WORK/off.out"; then rc=0; else rc=1; fi
check reclaim_off_pass "the fixture does not pass with reclamation disabled" $rc

flags_of() { grep '^\[vm_region_evac_subtype_coverage\]' "$1" | tr -d ' \n'; }
A="$(flags_of "$WORK/default.out")"
B="$(flags_of "$WORK/off.out")"
C="$(flags_of "$WORK/poison.out")"
if [ -n "$A" ] && [ "$A" = "$B" ] && [ "$A" = "$C" ]; then rc=0; else rc=1; fi
check answers_unchanged \
  "reclamation changed the fixture's per-subtype results (default='$A' off='$B' poison='$C')" $rc

# ── 5. memory ceiling ───────────────────────────────────────────────────────
if [ "${POISON_RSS:-0}" -gt "$CEILING_MB" ]; then rc=1; else rc=0; fi
check poison_rss_ceiling \
  "poison-mode peak RSS ${POISON_RSS} MB exceeds ${CEILING_MB} MB" $rc
if [ "${DEFAULT_RSS:-0}" -gt "$CEILING_MB" ]; then rc=1; else rc=0; fi
check default_rss_ceiling \
  "default peak RSS ${DEFAULT_RSS} MB exceeds ${CEILING_MB} MB — per-region reclamation regressed" $rc

echo "  vm-region-evac-subtype-coverage: $PASS passed, $FAIL failed"
echo "  peak RSS: default=${DEFAULT_RSS} MB, poison=${POISON_RSS} MB, reclaim-off=${OFF_RSS} MB"
if [ "$FAIL" -eq 0 ]; then
    echo "vm_region_evac_subtype_coverage_test.sh: PASS"
    exit 0
fi
echo "vm_region_evac_subtype_coverage_test.sh: FAIL"
exit 1
