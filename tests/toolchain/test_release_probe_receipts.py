#!/usr/bin/env python3
"""Exercise the real shared shell probe and its two evidence consumers."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ProbeReceipts(unittest.TestCase):
    def test_smoke_transport_persists_full_nested_failure_and_exact_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            path = Path(directory)
            trace = path / "trace.jsonl"
            trace.touch()
            passing = path / "passing-producer.sh"
            passing.write_text("#!/usr/bin/env bash\nprintf success-marker\n", encoding="utf-8")
            failing = path / "nested-producer.sh"
            failing.write_text("#!/usr/bin/env bash\nprintf nested-marker >&2\nexit 7\n", encoding="utf-8")
            script = '''set -u
. "$REPO_ROOT/scripts/lib/harness_outcome.sh"
. "$REPO_ROOT/scripts/lib/icc_probe.sh"
PROBE_TOTAL=0; PROBE_FAILURES=0; PROBE_INFRA=0
mkdir -p "$ICC_PROBE_LOG_DIR"
export ICC_PROBE_LOG_DIR
probe passing 'passing control' 'out=$(bash "$PASS_SCRIPT" 2>&1); printf "%s\\n" "$out"'
probe nested_failure 'nested failure control' 'out=$(bash "$FAIL_SCRIPT" 2>&1); rc=$?; if [ "$rc" -ne 0 ]; then printf "%s\\n" "$out" >&2; exit "$rc"; fi'
test "$PROBE_TOTAL" -eq 2
test "$PROBE_FAILURES" -eq 1
test "$(cat "$ICC_PROBE_LOG_DIR/passing.exit-status")" = 0
test "$(cat "$ICC_PROBE_LOG_DIR/nested_failure.exit-status")" = 7
grep -q success-marker "$ICC_PROBE_LOG_DIR/passing.log"
grep -q nested-marker "$ICC_PROBE_LOG_DIR/nested_failure.log"
python3 - "$TRACE_FILE" <<'PYCODE'
import json, sys
events = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
assert any(e.get("kind") == "eshkol_smoke" and e.get("name") == "nested_failure" and e.get("value") == "FAIL" for e in events)
assert any(e.get("kind") == "eshkol_smoke" and e.get("name") == "passing" and e.get("snippet") == "passing control: OK" for e in events), events
PYCODE
'''
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                env={**os.environ, "REPO_ROOT": str(ROOT), "TRACE_FILE": str(trace), "ICC_PROBE_LOG_DIR": str(path / "logs"), "PASS_SCRIPT": str(passing), "FAIL_SCRIPT": str(failing)})
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_receipt_write_failure_preserves_observed_verdict_and_blocks_durability(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            path = Path(directory)
            trace = path / "trace.jsonl"
            trace.touch()
            unusable = path / "not-a-directory"
            unusable.write_text("block receipt writes\n", encoding="utf-8")
            script = '''set -u
. "$REPO_ROOT/scripts/lib/harness_outcome.sh"
. "$REPO_ROOT/scripts/lib/icc_probe.sh"
probe receipt_pass 'positive receipt failure' 'printf positive-marker'
probe receipt_fail 'negative receipt failure' 'printf wrong-answer-marker; exit 7'
test "$PROBE_TOTAL" -eq 2
test "$PROBE_FAILURES" -eq 1
test "$PROBE_INFRA" -eq 2
python3 - "$TRACE_FILE" <<'PYCODE'
import json, sys
events = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
smoke = {(e.get("name"), e.get("value")) for e in events if e.get("kind") == "eshkol_smoke"}
assert ("receipt_pass", "INFRA") in smoke
assert ("receipt_pass", "PASS") not in smoke
assert ("receipt_fail", "FAIL") in smoke
assert ("receipt_fail_evidence_persistence", "INFRA") in smoke
results = {(e.get("name"), e.get("value", {}).get("passed")) for e in events if e.get("kind") == "test_result"}
assert ("receipt_fail", False) in results
assert not any(name == "receipt_pass" for name, _ in results)
PYCODE
'''
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                env={**os.environ, "REPO_ROOT": str(ROOT), "TRACE_FILE": str(trace), "ICC_PROBE_LOG_DIR": str(unusable)})
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_completed_outcomes_have_typed_receipts_but_infra_does_not(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.touch()
            script = '''set -u
. "$REPO_ROOT/scripts/lib/harness_outcome.sh"
. "$REPO_ROOT/scripts/lib/icc_probe.sh"
probe measured_pass 'pass with "quoted" label' ':'
probe measured_failure 'wrong answer' '(exit 7)'
probe missing_verdict 'no result obtained' '(exit 124)'
test "$PROBE_TOTAL" -eq 3
test "$PROBE_FAILURES" -eq 1
test "$PROBE_INFRA" -eq 1
'''
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                env={**os.environ, "REPO_ROOT": str(ROOT), "TRACE_FILE": str(trace)})
            self.assertEqual(result.returncode, 0, result.stderr)
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            smoke = {e["name"]: e["value"] for e in events if e["kind"] == "eshkol_smoke"}
            self.assertEqual(smoke, {"measured_pass": "PASS", "measured_failure": "FAIL", "missing_verdict": "INFRA"})
            pass_receipt = next(e for e in events if e.get("kind") == "eshkol_smoke" and e.get("name") == "measured_pass")
            self.assertEqual(pass_receipt["snippet"], 'pass with "quoted" label: OK')
            results = {e["name"]: e["value"]["passed"] for e in events if e["kind"] == "test_result"}
            self.assertEqual(results, {"measured_pass": True, "measured_failure": False})

    def test_missing_build_cannot_leave_a_stale_success_receipt(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            path = Path(directory)
            trace = path / "release_invariant_probes.jsonl"
            trace.write_text('{"kind":"test_result","name":"abi_layout_pin","value":{"passed":true}}\n')
            result = subprocess.run(["bash", str(ROOT / "scripts/run_release_invariant_probes.sh")],
                capture_output=True, text=True, env={**os.environ, "TRACE_DIR": str(path), "BUILD_DIR": str(path / "absent")})
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(trace.read_text(), "")

    def test_release_recipe_prepares_all_receipts_before_architecture_verification(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        readiness_gate = workflow.index("      - name: ICC readiness gate")
        readiness_call = workflow.index("scripts/run_v1_3_readiness.sh", readiness_gate)
        wrapper = (ROOT / "scripts/run_v1_3_readiness.sh").read_text()
        baseline = wrapper.split("run_baseline_phase() {", 1)[1].split("\n}\n", 1)[0]
        measurements = wrapper.split("run_full_measurements() {", 1)[1].split("\n}\n", 1)[0]
        coverage = baseline.index("run_baseline_coverage_phase")
        vm = baseline.index("run_baseline_measurements_phase")
        self.assertLess(measurements.index("scripts/run_ctest_gate.sh"),
                        measurements.index("scripts/run_vm_parity.sh"))
        smoke_step = wrapper.index("scripts/run_icc_smoke.sh")
        producers = wrapper.index("scripts/run_v1_3_release_producers.sh")
        grade = wrapper.index('"$ICC_BIN" architecture-verify')
        verify = wrapper.index("scripts/verify_v1_3_release_evidence.py")
        self.assertLess(readiness_call, len(workflow))
        self.assertLess(coverage, vm)
        self.assertLess(wrapper.index("run_baseline_phase() {"), smoke_step)
        self.assertLess(smoke_step, producers)
        self.assertLess(producers, verify)
        self.assertLess(verify, grade)
        smoke = (ROOT / "scripts/run_icc_smoke.sh").read_text()
        self.assertIn('scripts/lib/icc_probe.sh', smoke)
        self.assertIn('eshkol_release_invariant_probes', smoke)
        self.assertNotIn('probe abi_layout_pin ', smoke)
        recipe = (ROOT / "scripts/lib/release_invariant_probes.sh").read_text()
        for name in ("abi_layout_pin", "abi_object_header_ratchet", "closed_enum_dispatch_exhaustive", "ad_exactness_gate"):
            self.assertIn("probe " + name + " ", recipe)
        vm_script = (ROOT / "scripts/run_vm_parity.sh").read_text()
        self.assertIn('emit_test_result "vm_parity_gate" "PASS"', vm_script)
        self.assertIn('emit_test_result "vm_parity_gate" "FAIL"', vm_script)

    def test_vm_infrastructure_outcome_has_no_typed_verdict(self):
        vm_script = (ROOT / "scripts/run_vm_parity.sh").read_text()
        gate = vm_script[vm_script.index("# ── gate ──"):]
        infra_start = gate.index('elif [ $infra -gt 0 ]; then')
        infra_end = gate.index('\nelse\n', infra_start)
        infra_branch = gate[infra_start:infra_end]
        self.assertIn('emit_event "vm_parity_gate" "INFRA"', infra_branch)
        self.assertIn('rc=2', infra_branch)
        self.assertNotIn('emit_test_result', infra_branch)
        self.assertIn('elif [ $infra -gt 0 ]; then', gate)


if __name__ == "__main__":
    (ROOT / ".scratch").mkdir(exist_ok=True)
    unittest.main()
