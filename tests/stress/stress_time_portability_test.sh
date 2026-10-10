#!/usr/bin/env bash
set -u
export LC_ALL=C LC_CTYPE=C LANG=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GUARD="${1:-$ROOT/scripts/lib/guarded_exec.pl}"
SCRATCH="${2:-$ROOT/.scratch/stress-time-test}"
mkdir -p "$SCRATCH"
WORK="$(mktemp -d "$SCRATCH/test.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
. "$ROOT/scripts/lib/stress_time.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }
check() { [ "$1" -eq 0 ] || fail "$2"; }

MODE=$(eshkol_stress_time_detect /usr/bin/time "$WORK/probe") || fail 'real /usr/bin/time has no supported peak-RSS mode'
case "$MODE" in bsd|gnu) ;; *) fail "unexpected detected mode: $MODE" ;; esac

eshkol_stress_time_run /usr/bin/time "$MODE" "$GUARD" 5 "$WORK/out" "$WORK/time" \
    /bin/sh -c 'printf "child stdout\n"; printf "child stderr\n" >&2'
RC=$?
check "$RC" 'successful child exit status changed'
RSS=$(eshkol_stress_time_rss_mb "$MODE" "$WORK/time") || fail 'real time output did not yield RSS'
case "$RSS" in ''|*[!0-9]*) fail "RSS was not an integer: $RSS" ;; esac
grep -q '^child stdout$' "$WORK/out" || fail 'child stdout was lost'
eshkol_stress_time_append_program_stderr "$MODE" "$WORK/time" "$WORK/out"
grep -q '^child stderr$' "$WORK/out" || fail 'child stderr was not preserved'

eshkol_stress_time_run /usr/bin/time "$MODE" "$GUARD" 5 "$WORK/fail.out" "$WORK/fail.time" \
    /bin/sh -c 'printf "expected failure diagnostic\n" >&2; exit 7'
RC=$?
[ "$RC" -eq 7 ] || fail "child exit 7 changed to $RC"
RSS=$(eshkol_stress_time_rss_mb "$MODE" "$WORK/fail.time") || fail 'failed child lost its resource measurement'
eshkol_stress_time_append_program_stderr "$MODE" "$WORK/fail.time" "$WORK/fail.out"
grep -q '^expected failure diagnostic$' "$WORK/fail.out" || fail 'failure stderr was lost'

eshkol_stress_time_run /usr/bin/time "$MODE" "$GUARD" 5 "$WORK/signal.out" "$WORK/signal.time" \
    /bin/sh -c 'kill -SEGV $$'
RC=$?
[ "$RC" -eq 139 ] || fail "genuine SIGSEGV classification changed to $RC"

printf 'Maximum resident set size (kbytes): 2048\n' > "$WORK/gnu.time"
RSS=$(eshkol_stress_time_rss_mb gnu "$WORK/gnu.time") || fail 'GNU fixture was rejected'
[ "$RSS" = 2 ] || fail "GNU RSS conversion was $RSS, expected 2"
printf '2048  maximum resident set size\n' > "$WORK/bsd.time"
RSS=$(eshkol_stress_time_rss_mb bsd "$WORK/bsd.time") || fail 'BSD fixture was rejected'
[ "$RSS" = 0 ] || fail "BSD RSS conversion was $RSS, expected 0"
for fixture in missing invalid; do
    case "$fixture" in
        missing) : > "$WORK/$fixture.time" ;;
        invalid) printf 'Maximum resident set size (kbytes): unknown\n' > "$WORK/$fixture.time" ;;
    esac
    if eshkol_stress_time_rss_mb gnu "$WORK/$fixture.time" >/dev/null 2>&1; then
        fail "$fixture RSS report was accepted"
    fi
done

# A missing time report is harness infrastructure failure even on a row whose
# program failure is normally accepted as XKNOWN. Use a fake runner for the
# documented-open row and a fake time executable that reports RSS during its
# feature probe, then deliberately omits it for measured commands.
mkdir -p "$WORK/fake-build"
cat > "$WORK/fake-build/eshkol-run" <<'MOCK_RUNNER'
#!/bin/sh
printf 'OK 3\n'
printf 'mock child stderr\n' >&2
MOCK_RUNNER
chmod +x "$WORK/fake-build/eshkol-run"
cat > "$WORK/fake-time" <<'MOCK_TIME'
#!/bin/sh
if [ "$1" = "-l" ] && [ "${2:-}" = "true" ]; then
    printf '1 maximum resident set size\n' >&2
    exit 0
fi
if [ "$1" = "-v" ] && [ "${2:-}" = "true" ]; then
    printf 'Maximum resident set size (kbytes): 1\n' >&2
    exit 0
fi
[ "$1" = "-l" ] || exit 2
shift
"$@"
MOCK_TIME
chmod +x "$WORK/fake-time"
TRACE_FILE="$WORK/xknown-infra.jsonl" \
STRESS_SCRATCH_DIR="$WORK/harness-scratch" \
STRESS_TIME_BIN="$WORK/fake-time" \
BUILD_DIR="$WORK/fake-build" \
    bash "$ROOT/scripts/run_stress.sh" --no-aot --only closure_loop_global_set \
        > "$WORK/xknown-infra.log" 2>&1
HARNESS_RC=$?
[ "$HARNESS_RC" -ne 0 ] || fail 'missing RSS passed the XKNOWN harness control'
grep -q 'INFRA .*closure_loop_global_set' "$WORK/xknown-infra.log" ||
    fail 'missing RSS was not classified as infrastructure failure'
grep -q 'execution classification=FAIL' "$WORK/xknown-infra.log" ||
    fail 'underlying row execution classification was not preserved'
grep -q 'mock child stderr' "$WORK/xknown-infra.log" || fail 'XKNOWN child stderr was lost'
grep -q '"value":"INFRA"' "$WORK/xknown-infra.jsonl" ||
    fail 'infrastructure failure was not written to the trace'
if grep -q 'XKNOWN[[:space:]]\+tests/stress/found/closure_loop_global_set' "$WORK/xknown-infra.log"; then
    fail 'missing RSS was swallowed by XKNOWN'
fi

eshkol_stress_time_run /usr/bin/time "$MODE" "$GUARD" 1 "$WORK/timeout.out" "$WORK/timeout.time" \
    /bin/sh -c 'sleep 3'
RC=$?
[ "$RC" -eq 124 ] || fail "guard timeout changed to $RC"

echo 'PASS: stress time portability and failure preservation'
