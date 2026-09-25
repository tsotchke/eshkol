#!/usr/bin/env bash
# Shared probe execution and measured evidence for smoke and release gates.
# Requires harness_outcome.sh and TRACE_FILE; INFRA never becomes a test_result.
# Emit one trace line as a JSON-L event with explicit `kind`. ICC's
# runtime_evidence parser was extended (2026-05-07) to recognize records
# carrying an explicit `kind` field as pre-shaped events, instead of
# walking their keys with the ML-training-log heuristic.
#
# The oracle criterion matches:
#     event_kinds: [eshkol_smoke]
#     event_names: ["<probe_id>"]
#     event_values: ["PASS"]
emit_event() {
    local probe_id="$1" status="$2" snippet="$3"
    eshkol_outcome_emit_event "$TRACE_FILE" eshkol_smoke "$probe_id" "$status" "$snippet" 0.95
    eshkol_outcome_emit_test_result "$TRACE_FILE" "$probe_id" "$status" "$snippet"
}

PROBE_TOTAL=0
PROBE_FAILURES=0
PROBE_INFRA=0

probe() {
    local probe_id="$1" label="$2" cmd="$3"
    local out status snippet class
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    # Capture combined stdout+stderr so the snippet is informative when
    # something fails. Bound the snippet so a multi-MB log doesn't blow
    # up the trace file.
    out=$(eval "$cmd" 2>&1)
    status=$?
    if [ "$status" -eq 0 ]; then
        snippet="${label}: OK"
        emit_event "$probe_id" PASS "$snippet"
        printf '  ✓ %-40s %s\n' "$probe_id" "$label"
        return
    fi
    # A probe body is an ad hoc `eval`'d shell snippet, most of which call
    # eshkol-run/a compiled binary directly with no timeout wrapper at all —
    # so today the ONLY exit codes this classifier can recognize as
    # "harness could not run" rather than "the code is wrong" are the
    # small, well-known set scripts/lib/harness_outcome.sh defines: a
    # SIGKILL/SIGTERM/SIGINT the environment sent (137/143/130), or the 124/
    # 125/142 shapes any probe that DOES wrap itself in
    # eshkol_outcome_guarded (directly, or transitively through a script
    # that sources this file) can now produce. Everything else stays FAIL,
    # per eshkol_outcome_classify_exit's own principle: an unrecognized
    # nonzero exit is a claim about the CODE until a harness explicitly
    # says otherwise.
    class=$(eshkol_outcome_classify_exit "$status")
    if [ "$class" = INFRA ]; then
        PROBE_INFRA=$((PROBE_INFRA + 1))
        snippet=$(printf '%s' "$out" | tail -c 200)
        emit_event "$probe_id" INFRA "$snippet"
        printf '  ⚠ %-40s %s (infra, exit %d — no verdict obtained)\n' "$probe_id" "$label" "$status"
    else
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        snippet=$(printf '%s' "$out" | tail -c 200)
        emit_event "$probe_id" FAIL "$snippet"
        printf '  ✗ %-40s %s (exit %d)\n' "$probe_id" "$label" "$status"
    fi
}
