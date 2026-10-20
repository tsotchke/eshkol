#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
gate="$root/scripts/run_v14_connection_gate.sh"
helper="$(sed -n '/^probe_unverified() {/,/^}/p' "$gate")"
[ -n "$helper" ]
ci_helper="$(sed -n '/^probe_external_ci_evidence() {/,/^}/p' "$gate")"
[ -n "$ci_helper" ]

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
