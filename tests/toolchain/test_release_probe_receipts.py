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
        coverage = wrapper.index("scripts/run_language_coverage.sh")
        vm = wrapper.index("scripts/run_vm_parity.sh")
        smoke_step = wrapper.index("scripts/run_icc_smoke.sh")
        producers = wrapper.index("scripts/run_v1_3_release_producers.sh")
        grade = wrapper.index('"$ICC_BIN" architecture-verify')
        verify = wrapper.index("scripts/verify_v1_3_release_evidence.py")
        self.assertLess(readiness_call, len(workflow))
        self.assertLess(coverage, vm)
        self.assertLess(vm, smoke_step)
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
