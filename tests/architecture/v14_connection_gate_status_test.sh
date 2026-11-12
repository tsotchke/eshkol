#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
gate="$root/scripts/run_v14_connection_gate.sh"
source "$root/scripts/lib/connection_gate_probe.sh"

work="$(mktemp -d "${TMPDIR:-/tmp}/v14-connection-status-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT
events="$work/events"
PROBE_TOTAL=0
PROBE_FAILURES=0
emit_event() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$events"; }

assert_event() {
    local name="$1" value="$2" expected="$3"
    if ! grep -Fq "$name" "$events" || ! grep -Fq "$value" "$events" || { [ -n "$expected" ] && ! grep -Fq "$expected" "$events"; }; then
        printf 'FAIL: expected %s %s event containing %s\n' "$name" "$value" "$expected" >&2
        cat "$events" >&2
        exit 1
    fi
}

# Exercise the exact helper that the production gate sources. Marker checks
# happen only after the producer finishes writing the captured output file.
probe_run producer_pass 'successful producer' 'printf "PASS: fixture\\n"' '^PASS: fixture$'
assert_event producer_pass PASS 'successful producer: OK'
probe_run producer_exit7 'nonzero producer' "bash -c 'printf \"PASS: fixture\\\\n\"; exit 7'" '^PASS: fixture$'
assert_event producer_exit7 FAIL 'PASS: fixture'
probe_run missing_marker 'missing marker' 'printf "different output\\n"' '^PASS: fixture$'
assert_event missing_marker FAIL 'different output'
completion_marker="$work/producer-completed"
probe_run producer_complete 'large successful producer' \
    "COMPLETION_MARKER='$completion_marker' python3 -c 'import os,sys; print(\"PASS: fixture\"); sys.stdout.write(\"x\" * 2000000); open(os.environ[\"COMPLETION_MARKER\"], \"w\").write(\"done\")'" \
    '^PASS: fixture$'
assert_event producer_complete PASS 'large successful producer: OK'
[ -f "$completion_marker" ] || { echo 'FAIL: producer did not finish after emitting its marker' >&2; exit 1; }
probe_run skip_large 'large SKIP output' \
    "python3 -c 'import sys; print(\"SKIP: fixture\"); sys.stdout.write(\"x\" * 2000000); print()'" \
    'SKIP: fixture'
assert_event skip_large FAIL ''
probe_run unverified_large 'large UNVERIFIED output' \
    "python3 -c 'import sys; print(\"UNVERIFIED: fixture\"); sys.stdout.write(\"x\" * 2000000); print()'" \
    'UNVERIFIED: fixture'
assert_event unverified_large FAIL ''

[ "$PROBE_TOTAL" -eq 6 ] && [ "$PROBE_FAILURES" -eq 4 ] || {
    printf 'FAIL: probe totals were %s/%s, expected 6/4\n' "$PROBE_TOTAL" "$PROBE_FAILURES" >&2
    exit 1
}

# Preserve UNVERIFIED and external CI accounting from the production gate.
helper="$(sed -n '/^probe_unverified() {/,/^}/p' "$gate")"
ci_helper="$(sed -n '/^probe_external_ci_evidence() {/,/^}/p' "$gate")"
[ -n "$helper" ] && [ -n "$ci_helper" ]
out="$(bash -c '
    PROBE_TOTAL=0
    PROBE_FAILURES=0
    emit_event() { printf "EVENT %s %s %s\\n" "$1" "$2" "$3"; }
    eval "$1"
    probe_unverified v14_example "requires source-bound implementation evidence"
    printf "COUNTS %s %s\\n" "$PROBE_TOTAL" "$PROBE_FAILURES"
' _ "$helper")"
case "$out" in
    *"EVENT v14_example FAIL UNVERIFIED: requires source-bound implementation evidence"*"COUNTS 1 1"*) ;;
    *) printf 'FAIL: unverified probe status changed\n%s\n' "$out" >&2; exit 1 ;;
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
    *"EVENT v14_ci_example FAIL UNVERIFIED: no source-bound CI receipt supplied for workflow.yml — receipt required"*"COUNTS 1 1"*) ;;
    *) printf 'FAIL: CI receipt status changed\n%s\n' "$ci_out" >&2; exit 1 ;;
esac

# Coverage: probe_portable_tail_transfer must not grade an incomplete read
# of scripts/run_recursion_depth.sh's trace file as "no mutual_tail cells".
# guarded_exec.pl SIGKILLs any process a cell left behind in its group on a
# normal exit too, without waiting for that kill to land, so a descendant can
# still be alive (and, hypothetically, still writing) for a moment after the
# producer's own process has already exited. That alone would only delay
# $(...) (a descendant holding the captured pipe open blocks its EOF), not
# corrupt the read -- the actually-reproduced shape needs the descendant's
# own stdio detached from that pipe (exactly what redirecting a background
# writer away from it, as below, or any out-of-band writer, looks like): then
# the producer's exit lets $(...) return immediately while the real content
# lands moments later. Confirmed against the pre-fix function: this exact
# fixture reproduced the real "no mutual_tail cells" false verdict before the
# completion-record wait was added, and is silent with it.
tt_fn="$(sed -n '/^probe_portable_tail_transfer() {/,/^}/p' "$gate")"
[ -n "$tt_fn" ]
fake_repo="$work/tail-transfer-race-repo"
mkdir -p "$fake_repo/scripts/icc_traces"
cat > "$fake_repo/scripts/run_recursion_depth.sh" <<'FAKE'
#!/usr/bin/env bash
trace="$(cd "$(dirname "$0")/.." && pwd)/scripts/icc_traces/recursion_depth.jsonl"
: > "$trace"
(
    sleep 0.5
    printf '{"kind":"recursion_depth","name":"rec_mutual_tail_cond_d100000000_r","value":"PASS","snippet":"x","confidence":0.95}\n' >> "$trace"
    printf '{"kind":"recursion_depth","name":"rec_mutual_tail_cond_d100000000_aot","value":"PASS","snippet":"x","confidence":0.95}\n' >> "$trace"
    printf '{"kind":"recursion_depth","name":"recursion_depth_gate","value":"PASS","snippet":"total=2 pass=2 fail=0","confidence":0.95}\n' >> "$trace"
) </dev/null >/dev/null 2>&1 &
disown
exit 0
FAKE
chmod +x "$fake_repo/scripts/run_recursion_depth.sh"
tt_out="$(bash -c '
    PROBE_TOTAL=0
    PROBE_FAILURES=0
    REPO_ROOT="$2"
    BUILD_DIR_PATH="$2/build"
    emit_event() { printf "EVENT %s %s %s\n" "$1" "$2" "$3"; }
    eval "$1"
    probe_portable_tail_transfer
    printf "COUNTS %s %s\n" "$PROBE_TOTAL" "$PROBE_FAILURES"
' _ "$tt_fn" "$fake_repo")"
case "$tt_out" in
    *"EVENT v14_1_portable_tail_transfer PASS"*"OK:2 cells"*"COUNTS 1 0"*) ;;
    *) printf 'FAIL: the race fixture was not read correctly after waiting for the completion record\n%s\n' "$tt_out" >&2; exit 1 ;;
esac
echo 'PASS: portable-tail-transfer survives an orphaned-writer race'

# BUILD_DIR failures remain setup errors and do not attempt a compiler build.
missing_build="$work/missing"
empty_build="$work/empty"
mkdir -p "$empty_build"
set +e
BUILD_DIR="$missing_build" bash "$gate" >/dev/null 2>&1
status=$?
[ "$status" -eq 2 ] || { echo "FAIL: missing BUILD_DIR status was $status, expected 2" >&2; exit 1; }
BUILD_DIR="$empty_build" bash "$root/tests/v1_2_edge_cases/metrics_http_roundtrip_test.sh" >/dev/null 2>&1
status=$?
set -e
[ "$status" -eq 2 ] || { echo "FAIL: empty BUILD_DIR status was $status, expected 2" >&2; exit 1; }

# Shared isolation setup must retain each smoke test's own WORK cleanup.
for smoke in "$root/tests/v1_2_edge_cases/http_server_smoke_test.sh" \
             "$root/tests/v1_2_edge_cases/metrics_http_roundtrip_test.sh"; do
    grep -Fq 'ESHKOL_TEST_ISOLATION_NO_TRAP=1 eshkol_test_isolation_init' "$smoke"
    grep -Fq "trap 'eshkol_test_isolation_cleanup; rm -rf \"\$WORK\"' EXIT" "$smoke"
done

if grep -Ein 'NOT YET IMPLEMENTED|confirmed absent|zero hits|no [^[:cntrl:]]+ exists anywhere|this box has no self-hosted|no PJRT.*reference exists|detect_leaks=0' "$gate"; then
    echo 'FAIL: stale source-absence or runner-availability claim remains' >&2
    exit 1
fi

echo 'PASS: v1.4 connection gate status semantics'
