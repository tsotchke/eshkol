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

    def test_memory_diagnostic_mode_is_opt_in_and_nonpublishing(self):
        diagnostic = self.workflow["on"]["workflow_dispatch"]["inputs"]["memory_diagnostics"]
        self.assertEqual(diagnostic["type"], "boolean")
        self.assertEqual(diagnostic["default"], "false")
        self.assertEqual(diagnostic["required"], "false")
        self.assertEqual(self.workflow["jobs"]["unix-release-matrix"]["if"],
                         "inputs.memory_diagnostics != true")
        self.assertEqual(self.workflow["jobs"]["prefetch-windows-llvm-archives"]["if"],
                         "inputs.memory_diagnostics != true")
        self.assertEqual(self.workflow["jobs"]["windows-release-matrix"]["if"],
                         "inputs.memory_diagnostics != true")
        self.assertEqual(self.workflow["jobs"]["publish-release"]["if"],
                         "inputs.memory_diagnostics != true")
        reject = self.steps["Reject incompatible memory diagnostic options"]
        self.assertIn('[[ "$STRICT_READINESS" == "true" ]]', reject["run"])
        resolve = self.steps["Resolve ICC binary (block on push, never fail-open)"]
        self.assertNotIn("if", resolve)
        self.assertIn('if [[ "${MEMORY_DIAGNOSTICS}" == "true" ]]; then', resolve["run"])
        self.assertIn("exit 1", resolve["run"])
        self.assertEqual(self.steps["Bind ICC to this release checkout and commit"]["if"],
                         "env.ICC_AVAILABLE == 'true'")
        run = self.steps["Run nonpublishing memory diagnostics"]
        self.assertEqual(run["if"], "env.MEMORY_DIAGNOSTICS == 'true'")
        self.assertEqual(run["env"]["BUILD_DIR"], "build")
        for test_script in (
            "tests/memory/region_evac_subtype_coverage_test.sh",
            "tests/memory/vm_region_flat_rss_test.sh",
            "tests/memory/iter_scope_partial_reclaim_test.sh",
        ):
            self.assertIn(test_script, run["run"])
        self.assertIn("status.tsv", run["run"])
        self.assertIn('if (( failures != 0 )); then', run["run"])
        self.assertIn("${{ github.run_id }}", run["env"]["ESHKOL_DURABLE_WORK_ROOT"])
        self.assertIn("${{ github.run_attempt }}", run["env"]["ESHKOL_DURABLE_WORK_ROOT"])
        self.assertNotIn("scripts/run_v1_3_readiness.sh", run["run"])
        self.assertNotIn("receipt_created", run["run"])
        self.assertEqual(self.steps["Configure and build (readiness evidence)"]["if"],
                         "env.ICC_AVAILABLE == 'true'")
        self.assertEqual(self.steps["Configure and build fuzz tree (readiness evidence)"]["if"],
                         "env.ICC_AVAILABLE == 'true' && env.MEMORY_DIAGNOSTICS != 'true'")
        self.assertEqual(self.steps["Configure and build quantum tree (readiness evidence)"]["if"],
                         "env.ICC_AVAILABLE == 'true' && env.MEMORY_DIAGNOSTICS != 'true'")
        self.assertIn("env.MEMORY_DIAGNOSTICS != 'true'", self.steps[
            "ICC readiness gate (tag push or strict dry run requires ready/100)"]["if"])
        self.assertEqual(self.steps["Upload bound release-readiness receipt"]["if"],
                         "steps.readiness.outputs.receipt_created == 'true'")

    def test_memory_diagnostics_run_all_gates_and_retain_each_failure_log_and_status(self):
        run = self.steps["Run nonpublishing memory diagnostics"]["run"]
        tests = (
            "tests/memory/region_evac_subtype_coverage_test.sh",
            "tests/memory/vm_region_flat_rss_test.sh",
            "tests/memory/iter_scope_partial_reclaim_test.sh",
        )
        with tempfile.TemporaryDirectory(prefix="memory-diagnostic-") as directory:
            root = Path(directory)
            fake_tests = root / "fake-tests"
            fake_tests.mkdir()
            for index, source in enumerate(tests):
                test = fake_tests / Path(source).name
                test.write_text(f'echo "captured-{index}"\nexit {index}\n', encoding="utf-8")
                run = run.replace(source, str(test))
            evidence = root / "evidence"
            result = subprocess.run(
                ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", run],
                cwd=ROOT,
                env=dict(os.environ, ESHKOL_DURABLE_WORK_ROOT=str(evidence), BUILD_DIR="build"),
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            rows = (evidence / "status.tsv").read_text(encoding="utf-8").splitlines()
            self.assertEqual([row.rsplit("\t", 1)[1] for row in rows], ["0", "1", "2"])
            for index, source in enumerate(tests):
                log = evidence / (Path(source).name + ".log")
                self.assertIn(f"captured-{index}", log.read_text(encoding="utf-8"))
                self.assertIn(str(log), result.stdout)

    def test_missing_icc_fails_on_tag_or_strict_dispatch(self):
        resolve = self.steps["Resolve ICC binary (block on push, never fail-open)"]["run"]
        self.assertIn('rm -f "$RUNNER_TEMP/release-readiness-receipt.json"', resolve)
        self.assertIn('"${GITHUB_EVENT_NAME}" == "push" || "${STRICT_READINESS}" == "true"', resolve)
        self.assertIn("exit 1", resolve)
        self.assertIn("STRICT_READINESS: ${{ github.event_name == 'workflow_dispatch' && inputs.strict_readiness || false }}", self.text)
        # Ordinary dry-runs retain their warning-and-continue behavior.
        self.assertIn("This non-publishing dry-run continues", resolve)

    def test_release_icc_selector_prefers_release_override_and_keeps_ci_default(self):
        self.assertEqual(self.job["env"]["ICC_BIN_OVERRIDE"],
                         "${{ vars.RELEASE_ICC_BIN || vars.ICC_BIN }}")
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("ICC_BIN_OVERRIDE: ${{ vars.ICC_BIN }}", ci)
        self.assertNotIn("RELEASE_ICC_BIN", ci)
        pillars = (ROOT / ".github/workflows/pillars-nightly.yml").read_text(encoding="utf-8")
        self.assertIn("ICC_BIN_OVERRIDE: ${{ vars.ICC_BIN }}", pillars)
        self.assertNotIn("RELEASE_ICC_BIN", pillars)

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

    def test_each_release_phase_owns_fresh_durable_root_uploaded_before_cleanup(self):
        phases = ("baseline", "smoke", "final-evidence", "readiness")
        roots = []
        for phase in phases:
            step_name = {
                "baseline": "Release evidence baseline (coverage and VM parity)",
                "smoke": "Release smoke evidence",
                "final-evidence": "Remaining release producers and architecture verification",
                "readiness": "ICC readiness gate (tag push or strict dry run requires ready/100)",
            }[phase]
            step = self.steps[step_name]
            root = step["env"]["ESHKOL_DURABLE_WORK_ROOT"]
            self.assertIn("${{ github.run_id }}", root)
            self.assertIn("${{ github.run_attempt }}", root)
            self.assertTrue(root.endswith("/" + phase), root)
            roots.append(root)
            self.assertIn('mkdir -p "$ESHKOL_DURABLE_WORK_ROOT"', step["run"])
        self.assertEqual(len(set(roots)), len(phases))
        upload = self.steps["Upload readiness evidence"]
        self.assertEqual(upload["if"], "always()")
        self.assertIn("release-evidence-${{ github.run_id }}-${{ github.run_attempt }}/", upload["with"]["path"])
        diagnostic_root = self.steps["Run nonpublishing memory diagnostics"]["env"]["ESHKOL_DURABLE_WORK_ROOT"]
        self.assertIn("${{ github.run_id }}", diagnostic_root)
        self.assertIn("${{ github.run_attempt }}", diagnostic_root)
        self.assertIn("release-evidence-${{ github.run_id }}-${{ github.run_attempt }}/", upload["with"]["path"])
        cleanup_index = next(i for i, step in enumerate(self.job["steps"]) if step.get("name") == "Reclaim build trees")
        upload_index = next(i for i, step in enumerate(self.job["steps"]) if step.get("name") == "Upload readiness evidence")
        self.assertLess(upload_index, cleanup_index)
        self.assertNotIn(".scratch/v1-3-readiness", upload["with"]["path"])
        self.assertNotIn(".scratch/v1-3-readiness", self.steps["Reclaim build trees"]["run"])

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
                                            env=dict(os.environ, CMAKE_TOOLCHAIN_FILE=path,
                                                     READINESS_TOOL_BIN_DIR=""),
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_readiness_tool_selector_validates_and_appends_only_after_all_paths_pass(self):
        self.assertEqual(self.job["env"]["READINESS_TOOL_BIN_DIR"],
                         "${{ vars.RELEASE_TOOLCHAIN_BIN_DIR }}")
        names = [step.get("name") for step in self.job["steps"]]
        self.assertLess(names.index("Validate configured readiness toolchain"),
                        names.index("Toolchain preflight (self-hosted; provisioned out of band)"))
        script = self.steps["Validate configured readiness toolchain"]["run"]
        scratch = ROOT / ".scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="readiness-tool-bin-", dir=scratch) as directory:
            valid = Path(directory) / "bin"
            valid.mkdir()
            regular_file = Path(directory) / "file"
            regular_file.write_text("tool", encoding="utf-8")
            path_output = Path(directory) / "github-path"
            env_output = Path(directory) / "github-env"
            cases = [("", 0), (str(valid), 0), (str(valid) + ".missing", 1),
                     (valid.name, 1), (str(regular_file), 1),
                     (str(valid) + ":" + str(valid), 1),
                     (str(valid) + "\nINJECTED=1", 1),
                     (str(valid) + "\rINJECTED=1", 1)]
            for tool_dir, expected in cases:
                with self.subTest(tool_dir=tool_dir):
                    path_output.write_text("", encoding="utf-8")
                    env_output.write_text("", encoding="utf-8")
                    result = subprocess.run(
                        ["bash", "-c", script],
                        env=dict(os.environ,
                                 CMAKE_TOOLCHAIN_FILE="",
                                 READINESS_LIBRARY_PATH="",
                                 READINESS_TOOL_BIN_DIR=tool_dir,
                                 GITHUB_PATH=str(path_output),
                                 GITHUB_ENV=str(env_output)),
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, expected,
                                     result.stdout + result.stderr)
                    if expected == 0 and tool_dir:
                        self.assertEqual(path_output.read_text(), f"{valid}\n")
                    else:
                        self.assertEqual(path_output.read_text(), "")
                    self.assertEqual(env_output.read_text(), "")

            combined_cases = [
                (str(valid), "relative-library", 1),
                (str(valid) + ".missing", directory, 1),
                (str(valid), directory, 0),
            ]
            for tool_dir, library_path, expected in combined_cases:
                with self.subTest(tool_dir=tool_dir, library_path=library_path):
                    path_output.write_text("", encoding="utf-8")
                    env_output.write_text("", encoding="utf-8")
                    result = subprocess.run(
                        ["bash", "-c", script],
                        env=dict(os.environ,
                                 CMAKE_TOOLCHAIN_FILE="",
                                 READINESS_LIBRARY_PATH=library_path,
                                 READINESS_TOOL_BIN_DIR=tool_dir,
                                 LD_LIBRARY_PATH="/existing/path",
                                 GITHUB_PATH=str(path_output),
                                 GITHUB_ENV=str(env_output)),
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, expected,
                                     result.stdout + result.stderr)
                    if expected:
                        self.assertEqual(path_output.read_text(), "")
                        self.assertEqual(env_output.read_text(), "")
                    else:
                        self.assertEqual(path_output.read_text(), f"{valid}\n")
                        self.assertEqual(env_output.read_text(),
                                         f"LD_LIBRARY_PATH={directory}:/existing/path\n")

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
                                                     READINESS_TOOL_BIN_DIR="",
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
