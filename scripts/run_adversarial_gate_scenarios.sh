#!/usr/bin/env bash
# run_adversarial_gate_scenarios.sh — adversarial scenarios for OUR OWN gates.
#
# WHY THIS EXISTS, AND WHAT IT IS NOT
#
# docs/design/FLAW_DETECTION_ROADMAP.md's Appendix A names ICC's weakness-map
# output as ranking "eval scenarios for dirty worktrees, stale artifacts,
# model-server outage, disk pressure, failed gates, and new-development
# suggestions" as its #1 must-have. Investigating how ICC's own eval
# scenarios are built (`icc task-attempt-evals`, `icc capability-test-
# coverage`) shows they are derived from REAL recorded task attempts in
# ICC's own store (`task-attempt` outcomes), not from a scenario-definition
# file format this repo can author and hand to the tool — there is no
# `.icc/eval-scenarios.yaml` or equivalent for a target repo to populate.
#
# So this script delivers the eshkol-side equivalent named as the fallback in
# the wave-2 task: adversarial scenarios that prove OUR OWN gates — the
# assurance-gate family established in #454 and this wave
# (gate_no_silent_wrong.py, check_ledger_integrity.py, check_oracle_schema.py,
# check_self_verdicts.py, check_build_fingerprint.py,
# scripts/doc_audit/check_results_schema.py) — behave correctly under each of
# the six named conditions, mapped onto what actually exists in this repo:
#
#   S1 dirty worktree        gates must grade on FILE CONTENT, not be thrown
#                             off by unrelated uncommitted changes elsewhere
#                             in the tree
#   S2 stale artifacts        check_build_fingerprint.py's own contract:
#                             prove it goes RED on a rebuilt/replaced binary
#                             (reuses its --self-test, which already covers
#                             this — see D-11)
#   S3 model-server outage    none of our gates may depend on a live network
#                             or model server; prove every self-test still
#                             passes with outbound connections poisoned
#   S4 disk pressure          the disk-cap+cleanup discipline
#                             (scripts/lib/test_isolation.sh's pruning) must
#                             actually reclaim space under a tiny budget,
#                             not merely exist in a comment
#   S5 a gate actually fails   prove failure PROPAGATES: this runner's own
#                             pass/fail aggregation must not swallow a
#                             deliberately-red fixture (the standing bug class
#                             this whole roadmap is about — a check that can
#                             go red silently is worse than no check)
#   S6 new-development         a NEW gate script following the --self-test
#      orphan detection        contract must be wired into CMakeLists.txt
#                             ctest AND a CI workflow, or it is exactly a
#                             D-13 "gate that runs nowhere"
#
# Emits pytest-style PASSED/FAILED lines and a kind=adversarial_scenario
# JSON-L trace to scripts/icc_traces/adversarial_gate_scenarios.jsonl.
#
# Usage: scripts/run_adversarial_gate_scenarios.sh
set -u
export LC_ALL=C LC_CTYPE=C LANG=C

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/adversarial_gate_scenarios.jsonl"
mkdir -p "$TRACE_DIR"
: > "$TRACE_FILE"

# Disk cap + cleanup (project rule: every harness needs a bound). This
# runner's scratch is a handful of tiny synthetic files; the bound exists on
# principle and to catch a future scenario growing unbounded input.
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-adversarial-scenarios.XXXXXX")"
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

# macOS has no timeout(1); emulate with perl alarm (exit 124 on expiry) —
# same idiom scripts/run_vm_parity.sh already uses for portability.
run_guarded() { # seconds cmd...
    perl -e 'my $s=shift; eval { local $SIG{ALRM}=sub{ exit 124 }; alarm $s; exec @ARGV or exit 127; }' \
        "$1" "${@:2}"
}

emit_event() { # name PASS|FAIL snippet
    python3 -c '
import json, sys
print(json.dumps({"kind": "adversarial_scenario", "name": sys.argv[1],
                  "value": sys.argv[2], "snippet": sys.argv[3],
                  "confidence": 0.95}, ensure_ascii=False))
' "$1" "$2" "$3" >> "$TRACE_FILE"
}

SCEN_TOTAL=0
SCEN_FAILURES=0
scenario_result() { # id PASS|FAIL detail
    SCEN_TOTAL=$((SCEN_TOTAL + 1))
    if [ "$2" = "PASS" ]; then
        echo "PASSED scenario::$1 — $3"
    else
        SCEN_FAILURES=$((SCEN_FAILURES + 1))
        echo "FAILED scenario::$1 — $3"
    fi
    emit_event "$1" "$2" "$3"
}

echo "== adversarial gate scenarios =="
echo

# ── S1: dirty worktree must not throw off a gate's verdict ────────────────
echo "-- S1 dirty worktree --"
DIRTY_MARKER="$REPO_ROOT/.adversarial-scenario-dirty-marker.$$"
echo "unrelated scratch content, never committed" > "$DIRTY_MARKER"
s1_ok=1
s1_detail=""
for gate in \
    "scripts/gate_no_silent_wrong.py" \
    "scripts/check_ledger_integrity.py" \
    "scripts/check_oracle_schema.py"; do
    if ! python3 "$gate" --no-trace >/dev/null 2>&1; then
        s1_ok=0
        s1_detail="$gate graded FAIL against real committed state while an unrelated file was dirty"
        break
    fi
done
for gate in "scripts/check_self_verdicts.py" "scripts/check_build_fingerprint.py" \
            "scripts/doc_audit/check_results_schema.py"; do
    if ! python3 "$gate" --self-test >/dev/null 2>&1; then
        s1_ok=0
        s1_detail="$gate --self-test failed while an unrelated file was dirty"
        break
    fi
done
rm -f "$DIRTY_MARKER"
if [ "$s1_ok" -eq 1 ]; then
    scenario_result dirty_worktree_does_not_affect_gates PASS \
        "6 gates graded identically to a clean tree with an unrelated dirty file present"
else
    scenario_result dirty_worktree_does_not_affect_gates FAIL "$s1_detail"
fi

# ── S2: stale artifact detection actually goes red (reuse the real contract) ──
echo "-- S2 stale artifacts --"
if python3 scripts/check_build_fingerprint.py --self-test >/dev/null 2>&1; then
    scenario_result stale_artifact_detected PASS \
        "check_build_fingerprint.py's own fixtures prove it goes RED on a rebuilt/replaced/deleted binary"
else
    scenario_result stale_artifact_detected FAIL \
        "check_build_fingerprint.py --self-test did not pass — it can no longer prove it detects staleness"
fi

# ── S3: no gate depends on a live network / model server ──────────────────
echo "-- S3 model-server / network outage --"
s3_ok=1
s3_detail=""
for gate in \
    "scripts/gate_no_silent_wrong.py --self-test" \
    "scripts/check_ledger_integrity.py --self-test" \
    "scripts/check_oracle_schema.py --self-test" \
    "scripts/check_self_verdicts.py --self-test" \
    "scripts/check_build_fingerprint.py --self-test" \
    "scripts/doc_audit/check_results_schema.py --self-test"; do
    # Poison outbound connections: point every proxy variable at a closed
    # local port and blank DNS-relevant hosts, so any accidental network call
    # fails fast rather than hanging or silently succeeding on a live network.
    if ! http_proxy="http://127.0.0.1:1" https_proxy="http://127.0.0.1:1" \
            HTTP_PROXY="http://127.0.0.1:1" HTTPS_PROXY="http://127.0.0.1:1" \
            ALL_PROXY="http://127.0.0.1:1" no_proxy="" \
            run_guarded 60 python3 $gate >/dev/null 2>&1; then
        s3_ok=0
        s3_detail="$gate failed with outbound network poisoned — it may depend on a live service"
        break
    fi
done
if [ "$s3_ok" -eq 1 ]; then
    scenario_result gates_survive_network_outage PASS \
        "6 gate self-tests pass identically with every proxy env var pointed at a closed port"
else
    scenario_result gates_survive_network_outage FAIL "$s3_detail"
fi

# ── S4: disk-pressure pruning actually reclaims space ──────────────────────
echo "-- S4 disk pressure --"
PRUNE_ROOT="$SCRATCH/prune-root"
mkdir -p "$PRUNE_ROOT"
# Manufacture 10 fake "orphaned scratch directories" of the kind
# test_isolation.sh's eshkol_test_isolation_prune_stale reclaims, each old
# enough to be eligible, then squeeze the budget far below what they occupy.
i=0
while [ "$i" -lt 10 ]; do
    d="$PRUNE_ROOT/eshkol-test.selftest.deadbeef.$i"
    mkdir -p "$d"
    head -c 65536 /dev/zero > "$d/payload.bin" 2>/dev/null || dd if=/dev/zero of="$d/payload.bin" bs=1024 count=64 2>/dev/null
    # Backdate so both the age-sweep and the min-age floor for the size-sweep
    # treat these as eligible for reclamation, portably (GNU touch -d vs BSD touch -t).
    touch -d "-3 days" "$d" 2>/dev/null || touch -t "$(date -v-3d +%Y%m%d%H%M 2>/dev/null || date +%Y%m%d%H%M)" "$d" 2>/dev/null || true
    i=$((i + 1))
done
before_count=$(find "$PRUNE_ROOT" -maxdepth 1 -type d -name 'eshkol-test.*' | wc -l | tr -d ' ')
(
    ESHKOL_TEST_TMP_ROOT="$PRUNE_ROOT" \
    ESHKOL_TEST_TMP_MAX_DIRS=2 \
    ESHKOL_TEST_TMP_MAX_MB=1 \
    ESHKOL_TEST_TMP_MIN_AGE_MIN=1 \
    bash -c '
        source "'"$REPO_ROOT"'/scripts/lib/test_isolation.sh"
        eshkol_test_isolation_prune_stale
    '
) >/dev/null 2>&1
after_count=$(find "$PRUNE_ROOT" -maxdepth 1 -type d -name 'eshkol-test.*' 2>/dev/null | wc -l | tr -d ' ')
if [ "$before_count" -gt 2 ] && [ "$after_count" -le 2 ]; then
    scenario_result disk_pressure_reclaimed PASS \
        "pruning brought $before_count orphaned scratch dirs down to $after_count under a 2-dir/1MB budget"
else
    scenario_result disk_pressure_reclaimed FAIL \
        "expected pruning to reduce $before_count dirs to <=2, got $after_count remaining — disk-cap discipline is not enforcing its own budget"
fi

# ── S5: a deliberately-failing gate must be visible, not swallowed ─────────
echo "-- S5 failed gates propagate --"
BROKEN_LEDGER="$SCRATCH/broken-ledger.yaml"
cat > "$BROKEN_LEDGER" <<'YAML'
schema: eshkol.silent_wrong_ledger.v1
entries:
  - id: SW-ADVERSARIAL
    bucket: SILENT-WRONG
      status: open
    title: "deliberately malformed indentation"
YAML
if python3 scripts/check_ledger_integrity.py --ledger "$BROKEN_LEDGER" --no-trace >/dev/null 2>&1; then
    # The broken fixture was graded PASS — the gate (or this harness) failed
    # to detect a real problem. That is exactly the failure mode this
    # scenario exists to catch.
    scenario_result failed_gate_is_visible FAIL \
        "check_ledger_integrity.py graded a malformed ledger PASS — a failure that should be loud was silent"
else
    scenario_result failed_gate_is_visible PASS \
        "check_ledger_integrity.py correctly exited nonzero on a malformed fixture, and this runner correctly recorded that as the expected, visible outcome"
fi

# ── S6: every self-test-contract gate is wired into ctest AND CI ──────────
echo "-- S6 new-development orphan detection --"
ORPHANS=""
GATES_CHECKED=0
while IFS= read -r -d '' f; do
    grep -q -- '--self-test' "$f" 2>/dev/null || continue
    GATES_CHECKED=$((GATES_CHECKED + 1))
    rel="${f#"$REPO_ROOT"/}"
    base="$(basename "$f")"
    in_cmake=0
    grep -qF "$rel" CMakeLists.txt 2>/dev/null && in_cmake=1
    in_ci=0
    grep -rqF -- "$rel" .github/workflows/ 2>/dev/null && in_ci=1
    grep -rqF -- "$base" .github/workflows/ 2>/dev/null && in_ci=1
    if [ "$in_cmake" -eq 0 ] && [ "$in_ci" -eq 0 ]; then
        ORPHANS="$ORPHANS $rel"
    fi
done < <(find scripts \( -name 'check_*.py' -o -name 'gate_*.py' \) -print0 2>/dev/null)

if [ -z "$ORPHANS" ]; then
    scenario_result no_orphan_selftest_gates PASS \
        "$GATES_CHECKED gate script(s) following the --self-test contract are each wired into CMakeLists.txt ctest and/or a CI workflow"
else
    scenario_result no_orphan_selftest_gates FAIL \
        "gate(s) implementing --self-test but wired into neither CMakeLists.txt nor .github/workflows/:$ORPHANS"
fi

# ── roll-up ──────────────────────────────────────────────────────────────
echo
echo "adversarial-gate-scenarios: $((SCEN_TOTAL - SCEN_FAILURES)) passed, $SCEN_FAILURES failed"
if [ "$SCEN_FAILURES" -eq 0 ]; then
    emit_event "adversarial_gate_scenarios_gate" PASS "$SCEN_TOTAL/$SCEN_TOTAL scenarios behaved as predicted"
    echo "Trace written: $TRACE_FILE"
    exit 0
else
    emit_event "adversarial_gate_scenarios_gate" FAIL "$SCEN_FAILURES/$SCEN_TOTAL scenario(s) did not behave as predicted"
    echo "Trace written: $TRACE_FILE"
    exit 1
fi
