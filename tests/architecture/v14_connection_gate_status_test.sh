#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
gate="$root/scripts/run_v14_connection_gate.sh"
helper="$(sed -n '/^probe_unverified() {/,/^}/p' "$gate")"
[ -n "$helper" ]
ci_helper="$(sed -n '/^probe_external_ci_evidence() {/,/^}/p' "$gate")"
[ -n "$ci_helper" ]
run_helper="$(sed -n '/^probe_run() {/,/^}/p' "$gate")"
[ -n "$run_helper" ]

out="$(bash -c '
    PROBE_TOTAL=0
    PROBE_FAILURES=0
    emit_event() { printf "EVENT %s %s %s\\n" "$1" "$2" "$3"; }
    eval "$1"
    probe_unverified v14_example "requires source-bound implementation evidence"
    printf "COUNTS %s %s\\n" "$PROBE_TOTAL" "$PROBE_FAILURES"
' _ "$helper")"
case "$out" in
    *"EVENT v14_example FAIL UNVERIFIED: requires source-bound implementation evidence"*) ;;
    *) printf 'FAIL: unverified event missing or not failing\n%s\n' "$out" >&2; exit 1 ;;
esac

# A zero exit code cannot turn a test's SKIP/UNVERIFIED report into PASS.
skip_out="$(bash -c '
    PROBE_TOTAL=0
    PROBE_FAILURES=0
    emit_event() { printf "EVENT %s %s %s\\n" "$1" "$2" "$3"; }
    eval "$1"
    probe_run v14_skipped "fixture" '\''printf "SKIP: unavailable\\n"; exit 0'\''
    printf "COUNTS %s %s\\n" "$PROBE_TOTAL" "$PROBE_FAILURES"
' _ "$run_helper")"
case "$skip_out" in
    *"EVENT v14_skipped FAIL SKIP: unavailable"*"COUNTS 1 1"*) ;;
    *) printf 'FAIL: zero-exit SKIP output certified or miscounted\n%s\n' "$skip_out" >&2; exit 1 ;;
esac

# BUILD_DIR accepts a checkout-relative or absolute directory and reports a
# missing directory as a setup error (2), without attempting a build.
missing_build="$root/.scratch/v14-gate-missing-build-$$"
set +e
BUILD_DIR="$missing_build" bash "$gate" >/dev/null 2>&1
missing_status=$?
set -e
[ "$missing_status" -eq 2 ] || { echo "FAIL: missing BUILD_DIR status was $missing_status, expected 2" >&2; exit 1; }
for missing in "$missing_build" "${missing_build#$root/}"; do
    set +e
    BUILD_DIR="$missing" bash "$root/tests/v1_2_edge_cases/metrics_http_roundtrip_test.sh" >/dev/null 2>&1
    missing_status=$?
    set -e
    [ "$missing_status" -eq 2 ] || { echo "FAIL: metrics missing BUILD_DIR status was $missing_status, expected 2" >&2; exit 1; }
done
case "$out" in
    *"COUNTS 1 1"*) ;;
    *) printf 'FAIL: unverified probe accounting changed\n%s\n' "$out" >&2; exit 1 ;;
esac

ci_out="$(bash -c '
    PROBE_TOTAL=0
    PROBE_FAILURES=0
    emit_event() { printf "EVENT %s %s %s\\n" "$1" "$2" "$3"; }
    eval "$1"
    eval "$2"
    probe_external_ci_evidence v14_ci_example workflow.yml "receipt required"
    printf "COUNTS %s %s\\n" "$PROBE_TOTAL" "$PROBE_FAILURES"
' _ "$helper" "$ci_helper")"
case "$ci_out" in
    *"EVENT v14_ci_example FAIL UNVERIFIED: no source-bound CI receipt supplied for workflow.yml — receipt required"*) ;;
    *) printf 'FAIL: CI receipt status missing or not failing\n%s\n' "$ci_out" >&2; exit 1 ;;
esac
case "$ci_out" in
    *"COUNTS 1 1"*) ;;
    *) printf 'FAIL: CI receipt probe accounting changed\n%s\n' "$ci_out" >&2; exit 1 ;;
esac

if rg -n -i 'NOT YET IMPLEMENTED|confirmed absent|zero hits|no [^[:cntrl:]]+ exists anywhere|this box has no self-hosted|no PJRT.*reference exists|detect_leaks=0' "$gate"; then
    echo 'FAIL: stale source-absence or runner-availability claim remains' >&2
    exit 1
fi

echo 'PASS: v1.4 connection gate unverified status'
