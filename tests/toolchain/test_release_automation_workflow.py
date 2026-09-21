#!/usr/bin/env python3
"""Static negative controls for strict release-readiness workflow wiring."""

from pathlib import Path
import re
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/release.yml"


class ReleaseAutomationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        cls.workflow = yaml.load(cls.text, Loader=yaml.BaseLoader)
        cls.job = cls.workflow["jobs"]["release-readiness-gate"]
        cls.steps = {step.get("name"): step for step in cls.job["steps"]}

    def test_dispatch_strict_input_is_opt_in_boolean(self):
        strict = self.workflow["on"]["workflow_dispatch"]["inputs"]["strict_readiness"]
        self.assertEqual(strict["type"], "boolean")
        self.assertEqual(strict["default"], "false")
        self.assertEqual(strict["required"], "false")
        self.assertIn("inputs.strict_readiness", self.text)

    def test_missing_icc_fails_on_tag_or_strict_dispatch(self):
        resolve = self.steps["Resolve ICC binary (block on push, never fail-open)"]["run"]
        self.assertIn('rm -f "$RUNNER_TEMP/release-readiness-receipt.json"', resolve)
        self.assertIn('"${GITHUB_EVENT_NAME}" == "push" || "${STRICT_READINESS}" == "true"', resolve)
        self.assertIn("exit 1", resolve)
        self.assertIn("STRICT_READINESS: ${{ github.event_name == 'workflow_dispatch' && inputs.strict_readiness || false }}", self.text)
        # Ordinary dry-runs retain their warning-and-continue behavior.
        self.assertIn("This non-publishing dry-run continues", resolve)

    def test_unready_strict_dispatch_is_blocking_and_advisory_stays_advisory(self):
        gate = self.steps["ICC readiness gate (tag push or strict dry run requires ready/100)"]
        run = gate["run"]
        self.assertIn('"${GITHUB_EVENT_NAME}" == "push" || "${STRICT_READINESS}" == "true"', run)
        self.assertIn('if [[ "$block" == 1 ]]; then', run)
        self.assertIn('if [[ "$status" != "ready" ]]; then', run)
        self.assertIn("score:100", run)
        self.assertIn("--verdict \"$readiness_json\"", run)
        self.assertIn("readiness_command_failed=1", run)
        self.assertIn('if [[ "$readiness_command_failed" == 1 ]]; then', run)

    def test_receipt_is_bound_to_exact_commit_and_run_attempt(self):
        gate = self.steps["ICC readiness gate (tag push or strict dry run requires ready/100)"]
        run = gate["run"]
        self.assertIn('echo "receipt_created=false" >> "$GITHUB_OUTPUT"', run)
        self.assertIn('--arg sha "$GITHUB_SHA"', run)
        self.assertIn('--argjson run_id "$GITHUB_RUN_ID"', run)
        self.assertIn('--argjson run_attempt "$GITHUB_RUN_ATTEMPT"', run)
        self.assertIn('schema:"eshkol.release-readiness.v1"', run)
        self.assertIn('status:"ready",score:100,target:"v1.3.5-evolve"', run)
        self.assertIn('if ! jq -n', run)
        self.assertIn('rm -f "$RUNNER_TEMP/release-readiness-receipt.json"', run)
        self.assertNotIn("$RELEASE_TAG", run[run.find("jq -n "):])
        self.assertLess(run.find('--verdict "$readiness_json"'), run.find("jq -n "))
        self.assertLess(run.find('(.status == "ready")'), run.find("jq -n "))

    def test_receipt_upload_requires_created_proof_and_uses_sha_name(self):
        upload = self.steps["Upload bound release-readiness receipt"]
        self.assertEqual(upload["if"], "steps.readiness.outputs.receipt_created == 'true'")
        self.assertEqual(upload["with"]["name"], "release-readiness-receipt-${{ github.sha }}")
        self.assertEqual(upload["with"]["path"], "${{ runner.temp }}/release-readiness-receipt.json")
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertNotIn("always()", upload["if"])

    def test_readiness_budget_and_build_parallelism_cover_the_expanded_recipe(self):
        self.assertEqual(self.job["timeout-minutes"], "720")
        commands = "\n".join(step.get("run", "") for step in self.job["steps"])
        builds = [line.strip() for line in commands.splitlines() if "cmake --build" in line]
        self.assertEqual(len(builds), 3)
        self.assertTrue(all("--parallel 4" in line for line in builds), builds)

    def test_explicit_readiness_reread_uses_canonical_architecture_trace(self):
        wrapper = (ROOT / "scripts/run_v1_3_readiness.sh").read_text(encoding="utf-8")
        match = re.search(r'ARCH_TRACE_GLOB="\$\{ARCH_TRACE_GLOB:-([^}]+)\}"', wrapper)
        self.assertIsNotNone(match)
        canonical_trace = match.group(1)
        gate = self.steps["ICC readiness gate (tag push or strict dry run requires ready/100)"]["run"]
        self.assertIn('--trace-latest "$ARCH_TRACE_GLOB"', wrapper)
        self.assertIn(f"--trace-latest '{canonical_trace}'", gate)


if __name__ == "__main__":
    unittest.main()
