#!/usr/bin/env python3
"""Static negative controls for strict release-readiness workflow wiring."""

from pathlib import Path
import os
import re
import subprocess
import tempfile
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
        self.assertEqual(self.workflow["on"]["workflow_dispatch"]["inputs"]["candidate_tag"]["default"],
                         "v1.3.6-evolve")

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
        self.assertIn('status:"ready",score:100,target:$target', run)
        self.assertIn('--target "$RELEASE_TARGET"', run)
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

    def test_configured_readiness_toolchain_rejects_missing_or_relative_files(self):
        self.assertEqual(self.job["env"]["CMAKE_TOOLCHAIN_FILE"],
                         "${{ vars.RELEASE_CMAKE_TOOLCHAIN_FILE }}")
        script = self.steps["Validate configured readiness toolchain"]["run"]
        scratch = ROOT / ".scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="readiness-toolchain-", dir=scratch) as directory:
            valid = Path(directory) / "toolchain.cmake"
            valid.write_text("# Test toolchain fixture\n", encoding="utf-8")
            for path, expected in [("", 0), (str(valid), 0),
                                   (str(valid) + ".missing", 1), (valid.name, 1),
                                   (directory, 1)]:
                with self.subTest(path=path):
                    result = subprocess.run(["bash", "-c", script],
                                            env=dict(os.environ, CMAKE_TOOLCHAIN_FILE=path),
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_readiness_library_paths_are_validated_and_preserve_existing_search_path(self):
        self.assertEqual(self.job["env"]["READINESS_LIBRARY_PATH"],
                         "${{ vars.RELEASE_RUNTIME_LIBRARY_PATH }}")
        script = self.steps["Validate configured readiness toolchain"]["run"]
        scratch = ROOT / ".scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="readiness-libraries-", dir=scratch) as directory:
            output = Path(directory) / "github-env"
            for path, expected in [(directory, 0), (directory + ".missing", 1),
                                   ("relative", 1), (directory + "\nINJECTED=1", 1),
                                   (":" + directory, 1), (directory + ":", 1),
                                   (directory + "::" + directory, 1),
                                   (directory + "\rINJECTED=1", 1)]:
                with self.subTest(path=path):
                    output.write_text("", encoding="utf-8")
                    result = subprocess.run(["bash", "-c", script],
                                            env=dict(os.environ, CMAKE_TOOLCHAIN_FILE="",
                                                     READINESS_LIBRARY_PATH=path,
                                                     LD_LIBRARY_PATH="/existing/path",
                                                     GITHUB_ENV=str(output)),
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    if expected == 0:
                        self.assertEqual(output.read_text(),
                                         f"LD_LIBRARY_PATH={directory}:/existing/path\n")
                    else:
                        self.assertEqual(output.read_text(), "")

    def test_release_packages_and_evidence_builds_require_image_io(self):
        for job_name, expected in [("unix-release-matrix", 1),
                                   ("windows-release-matrix", 1),
                                   ("release-readiness-gate", 3)]:
            commands = "\n".join(step.get("run", "")
                                 for step in self.workflow["jobs"][job_name]["steps"])
            with self.subTest(job=job_name):
                self.assertEqual(commands.count("-DESHKOL_REQUIRE_IMAGE_IO=ON"), expected)

    def test_readiness_python_dependencies_match_the_qualified_profile(self):
        script = self.steps["Prepare isolated Python binding test environment"]["run"]
        self.assertIn("pybind11==3.1.0 numpy==2.5.3 pyyaml==6.0.3", script)
        self.assertIn('python3 -m venv "$python_env"', script)


if __name__ == "__main__":
    unittest.main()
