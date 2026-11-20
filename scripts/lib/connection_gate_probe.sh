# Shared probe runner used by the v1.4 connection gate and its shell regression.
# The command completes into a file before its output is inspected. This keeps
# marker checks from closing a producer pipe early or hiding its exit status.
probe_run() {
    local probe_id="$1" label="$2" cmd="$3" marker="${4:-}"
    local output_file status skip_status snippet
    PROBE_TOTAL=$((PROBE_TOTAL + 1))
    output_file="$(mktemp "${TMPDIR:-/tmp}/eshkol-v14-probe.XXXXXX")" || {
        status=$?
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        emit_event "$probe_id" FAIL "could not create probe output file (exit $status)"
        printf '  \xE2\x9C\x97 %-42s could not capture probe output\n' "$probe_id"
        return 0
    }

    if eval "$cmd" >"$output_file" 2>&1; then
        status=0
    else
        status=$?
    fi

    skip_status=0
    if grep -Eqi '(^|[^[:alpha:]])(SKIP|UNVERIFIED)(:|[[:space:]]|$)' "$output_file"; then
        skip_status=0
    else
        skip_status=$?
    fi
    if [ "$skip_status" -gt 1 ]; then
        status="$skip_status"
    elif [ -n "$marker" ] && ! grep -Eq "$marker" "$output_file"; then
        status=1
    fi

    if [ "$status" -eq 0 ] && [ "$skip_status" -ne 0 ]; then
        emit_event "$probe_id" PASS "${label}: OK"
        printf '  \xE2\x9C\x93 %-42s %s\n' "$probe_id" "$label"
    else
        PROBE_FAILURES=$((PROBE_FAILURES + 1))
        snippet="$(tail -c 220 "$output_file")"
        emit_event "$probe_id" FAIL "$snippet"
        printf '  \xE2\x9C\x97 %-42s %s (exit %d)\n' "$probe_id" "$label" "$status"
    fi
    rm -f "$output_file"
}
