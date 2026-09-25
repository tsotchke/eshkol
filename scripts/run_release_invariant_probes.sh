#!/usr/bin/env bash
# Produce real test_result receipts before the first ICC architecture grade.
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in /*) ;; *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;; esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
# Evidence paths are absolute before first use (scripts/lib/evidence_paths.sh).
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/release_invariant_probes.jsonl"
: > "$TRACE_FILE"
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "Release invariant probes require a built compiler: $ESHKOL_RUN" >&2
    exit 2
fi
. "$REPO_ROOT/scripts/lib/harness_outcome.sh"
. "$REPO_ROOT/scripts/lib/icc_probe.sh"
. "$REPO_ROOT/scripts/lib/release_invariant_probes.sh"
. "$REPO_ROOT/scripts/lib/build_fingerprint.sh"
eshkol_emit_build_fingerprint_event "$TRACE_DIR" release_invariant_probes "$BUILD_DIR" eshkol-run || exit $?
eshkol_release_invariant_probes
echo "Release invariant probes: $PROBE_TOTAL total, $PROBE_FAILURES failures, $PROBE_INFRA infrastructure failures"
[ "$PROBE_FAILURES" -eq 0 ] && [ "$PROBE_INFRA" -eq 0 ] && [ "$PROBE_TOTAL" -eq 4 ]
