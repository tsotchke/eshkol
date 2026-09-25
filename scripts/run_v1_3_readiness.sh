#!/usr/bin/env bash
# Run the phased v1.3.5-evolve evidence recipe and ask ICC for readiness.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
. "$REPO_ROOT/scripts/lib/durable_work_root.sh"

PHASE=all
if [ "$#" -eq 2 ] && [ "$1" = "--phase" ]; then
    PHASE="$2"
elif [ "$#" -ne 0 ]; then
    echo "usage: $0 [--phase baseline|smoke|final-evidence|readiness]" >&2
    exit 2
fi
case "$PHASE" in
    all|baseline|smoke|final-evidence|readiness) ;;
    *) echo "unknown release readiness phase: $PHASE" >&2; exit 2 ;;
esac

ICC_BIN="${ICC_BIN:-icc}"
ICC_REPO="${ICC_REPO:-eshkol_lang}"
if eshkol_durable_enabled; then
    READINESS_WORK="$(eshkol_durable_prepare_dir v1-3-readiness)" || exit $?
    TRACE_DIR="${TRACE_DIR:-$READINESS_WORK/traces}"
    mkdir -p "$TRACE_DIR"
else
    TRACE_DIR="${TRACE_DIR:-scripts/icc_traces}"
fi
# Evidence paths are absolute before first use (scripts/lib/evidence_paths.sh).
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
ARCH_MODEL="${ARCH_MODEL:-.icc/architecture-model.yaml}"
ARCH_TRACE_GLOB="${ARCH_TRACE_GLOB:-.icc/runtime-traces-oracle-view/*architecture-model-verify-*.jsonl}"
if eshkol_durable_enabled; then
    READINESS_JSON="$(eshkol_durable_file "$READINESS_WORK" readiness.json)" || exit $?
else
    mkdir -p "$REPO_ROOT/.scratch"
    READINESS_JSON="$(mktemp "$REPO_ROOT/.scratch/eshkol-v135-readiness.XXXXXX.json")"
fi
: "${READINESS_JSON:?READINESS_JSON must be set}"
if ! eshkol_durable_enabled; then trap 'rm -f -- "${READINESS_JSON:?}"' EXIT; fi

BUILD_DIR="${BUILD_DIR:-build}"
QUANTUM_BUILD_DIR="${QUANTUM_BUILD_DIR:-build-quantum}"
export BUILD_DIR QUANTUM_BUILD_DIR TRACE_DIR ICC_BIN ICC_REPO
export ICC_TRACE_DIR="$TRACE_DIR" ESHKOL_TRACE_DIR="$TRACE_DIR"
export ESH0103_TRACE_DIR="$TRACE_DIR"

PHASE_ID="${ESHKOL_RELEASE_PHASE_ID:-}"
if [ -z "$PHASE_ID" ] && [ -n "${GITHUB_RUN_ID:-}" ]; then
    PHASE_ID="${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT:-1}"
fi
if [ -z "$PHASE_ID" ] && [ "$PHASE" = "all" ]; then
    PHASE_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
fi
if [ -z "$PHASE_ID" ]; then
    echo "split readiness phases require ESHKOL_RELEASE_PHASE_ID" >&2
    exit 2
fi
PHASE_STATE="$TRACE_DIR/release-phase-state.json"
COHORT_MANIFEST="$TRACE_DIR/release-build-cohort.json"

start_cohort() {
    mkdir -p "$TRACE_DIR"
    # Preserve a previous run outside ICC's active trace root. It cannot
    # provide PASS evidence for this release attempt.
    TRACE_ARCHIVE_ROOT="${ESHKOL_RELEASE_TRACE_ARCHIVE_ROOT:-$REPO_ROOT/.scratch/release-readiness-history}"
    python3 scripts/archive_release_trace_cohort.py --trace-dir "$TRACE_DIR" --archive-root "$TRACE_ARCHIVE_ROOT"
    python3 scripts/check_release_build_cohort.py capture --build-dir "$BUILD_DIR" --manifest "$COHORT_MANIFEST" --trace "$TRACE_DIR/release-build-cohort.jsonl"
    python3 scripts/release_phase_state.py begin --repo-root "$REPO_ROOT" --state "$PHASE_STATE" --phase-id "$PHASE_ID"
}

check_cohort() {
    python3 scripts/check_release_build_cohort.py check --build-dir "$BUILD_DIR" --manifest "$COHORT_MANIFEST" --trace "$TRACE_DIR/release-build-cohort.jsonl"
}

require_phase() {
    python3 scripts/release_phase_state.py require --repo-root "$REPO_ROOT" --state "$PHASE_STATE" --phase-id "$PHASE_ID" --phase "$1"
    check_cohort
}

mark_phase() {
    python3 scripts/release_phase_state.py mark --repo-root "$REPO_ROOT" --state "$PHASE_STATE" --phase-id "$PHASE_ID" --phase "$1"
}

run_baseline_phase() {
    # Coverage runs the complete suite once and records its prerequisite result.
    scripts/run_language_coverage.sh
    python3 - "$TRACE_DIR/language_surface_coverage_prereq.jsonl" <<'PY'
import json, sys
found = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("kind") == "language_coverage_prereq" and event.get("name") == "core_suite":
            found.append(event)
if len(found) != 1 or found[0].get("value") != "PASS":
    raise SystemExit("run_all_tests.sh prerequisite did not produce exactly one PASS")
PY
    . scripts/lib/harness_outcome.sh
    eshkol_outcome_emit_test_result "$TRACE_DIR/release_test_actions.jsonl" release_action::run_tco_tests PASS "run_all_tests.sh completed all suites including run_tco_tests.sh"
    eshkol_outcome_emit_test_result "$TRACE_DIR/release_test_actions.jsonl" release_action::run_control_flow_tests PASS "run_all_tests.sh completed all suites including run_control_flow_tests.sh"
    scripts/run_vm_parity.sh
    python3 scripts/check_release_phase_receipts.py baseline --trace-dir "$TRACE_DIR"
    mark_phase baseline
}

run_smoke_phase() {
    export ESHKOL_LANGUAGE_COVERAGE_ALREADY_RUN=1
    BUILD_DIR="$BUILD_DIR" TRACE_DIR="$TRACE_DIR" scripts/run_mono_equiv_ad_taylor_gate.sh
    python3 scripts/run_eskm_model_fuzz.py --smoke --self-test --probe "${ESKM_FUZZ_BUILD_DIR:-build-fuzz}/tests/fuzz/eskm_model_fuzz_probe" --trace-file "$TRACE_DIR/eskm_model_fuzz.jsonl"
    scripts/run_icc_smoke.sh
    python3 scripts/check_release_phase_receipts.py smoke --trace-dir "$TRACE_DIR"
    mark_phase smoke
}

run_final_evidence_phase() {
    # Doc-claims grading queries the ICC index; refresh it before that producer.
    "$ICC_BIN" reindex --repo "$ICC_REPO" --full
    scripts/run_v1_3_release_producers.sh
    python3 scripts/check_release_build_cohort.py verify --build-dir "$BUILD_DIR" --manifest "$COHORT_MANIFEST" --trace "$TRACE_DIR/release-build-cohort.jsonl"
    python3 scripts/verify_v1_3_release_evidence.py --trace-dir "$TRACE_DIR"
    "$ICC_BIN" architecture-verify --repo "$ICC_REPO" --model "$ARCH_MODEL" --trace-dir "$TRACE_DIR" --emit-trace --format markdown
    mark_phase final-evidence
}

run_readiness_phase() {
    python3 scripts/verify_v1_3_release_evidence.py --trace-dir "$TRACE_DIR"
    "$ICC_BIN" readiness --repo "$ICC_REPO" --target v1.3.5-evolve --trace-dir "$TRACE_DIR" --trace-latest "$ARCH_TRACE_GLOB" --format json > "${READINESS_JSON:?}"
    "$ICC_BIN" readiness --repo "$ICC_REPO" --target v1.3.5-evolve --trace-dir "$TRACE_DIR" --trace-latest "$ARCH_TRACE_GLOB" --format markdown

    status="$(jq -r '.status // ""' "$READINESS_JSON")"
    if [ "$status" != "ready" ]; then
        echo "v1.3.5-evolve readiness is $status, expected ready" >&2
        exit 1
    fi
}

case "$PHASE" in
    all)
        start_cohort
        run_baseline_phase
        run_smoke_phase
        run_final_evidence_phase
        run_readiness_phase
        ;;
    baseline)
        start_cohort
        run_baseline_phase
        ;;
    smoke)
        require_phase baseline
        python3 scripts/check_release_phase_receipts.py baseline --trace-dir "$TRACE_DIR"
        run_smoke_phase
        ;;
    final-evidence)
        require_phase smoke
        python3 scripts/check_release_phase_receipts.py baseline --trace-dir "$TRACE_DIR"
        python3 scripts/check_release_phase_receipts.py smoke --trace-dir "$TRACE_DIR"
        run_final_evidence_phase
        ;;
    readiness)
        require_phase final-evidence
        python3 scripts/verify_v1_3_release_evidence.py --trace-dir "$TRACE_DIR"
        run_readiness_phase
        ;;
esac
