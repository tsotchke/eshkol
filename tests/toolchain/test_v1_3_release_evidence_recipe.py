#!/usr/bin/env python3
"""Failure-injection checks for the release evidence manifest and CTest receipts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_v1_3_release_evidence as verifier  # noqa: E402
import check_release_build_cohort as cohort  # noqa: E402


class ReleaseEvidenceRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        cls.readiness = (ROOT / "scripts/run_v1_3_readiness.sh").read_text(encoding="utf-8")
        cls.smoke = (ROOT / "scripts/run_icc_smoke.sh").read_text(encoding="utf-8")

    def test_readiness_owns_one_trace_cohort_and_runs_producers_before_grading(self):
        capture = self.readiness.index("check_release_build_cohort.py capture")
        archive = self.readiness.index("archive_release_trace_cohort.py")
        coverage = self.readiness.index("scripts/run_language_coverage.sh")
        parity = self.readiness.index("scripts/run_vm_parity.sh")
        smoke = self.readiness.index("scripts/run_icc_smoke.sh")
        reindex = self.readiness.index('"$ICC_BIN" reindex')
        producers = self.readiness.index("scripts/run_v1_3_release_producers.sh")
        cohort_verify = self.readiness.index("check_release_build_cohort.py verify")
        verify = self.readiness.index("scripts/verify_v1_3_release_evidence.py")
        architecture = self.readiness.index('"$ICC_BIN" architecture-verify')
        self.assertLess(archive, capture)
        self.assertLess(capture, coverage)
        self.assertLess(coverage, parity)
        self.assertLess(parity, smoke)
        self.assertLess(smoke, reindex)
        self.assertLess(reindex, producers)
        self.assertLess(producers, cohort_verify)
        self.assertLess(cohort_verify, verify)
        self.assertLess(verify, architecture)
        self.assertEqual(self.readiness.count("scripts/run_language_coverage.sh"), 1)
        self.assertNotIn("scripts/run_all_tests.sh", self.smoke)
        self.assertIn('export ICC_TRACE_DIR="$TRACE_DIR" ESHKOL_TRACE_DIR="$TRACE_DIR"', self.readiness)

    def test_workflow_provisions_python_bindings_and_delegates_one_readiness_recipe(self):
        job = self.workflow[self.workflow.index("  release-readiness-gate:"):]
        self.assertIn("fetch-depth: 0", job)
        self.assertIn("Prepare isolated Python binding test environment", job)
        self.assertIn("pip install --disable-pip-version-check pybind11 numpy pyyaml", job)
        self.assertIn("command -v sbcl", job)
        self.assertIn("command -v prlimit", job)
        self.assertIn("-DESHKOL_PYTHON_BINDINGS=ON", job)
        self.assertIn("-Dpybind11_DIR=", job)
        parsed = yaml.load(self.workflow, Loader=yaml.BaseLoader)
        commands = "\n".join(step.get("run", "") for step in parsed["jobs"]["release-readiness-gate"]["steps"])
        self.assertNotIn("scripts/run_language_coverage.sh", commands)
        self.assertNotIn('architecture-verify \\\n', commands)
        self.assertIn("scripts/run_v1_3_readiness.sh --phase readiness", job)
        positions = [commands.index(f"scripts/run_v1_3_readiness.sh --phase {phase}")
                     for phase in ("baseline", "smoke", "final-evidence")]
        self.assertEqual(positions, sorted(positions))
        phase_steps = {step["name"]: step for step in parsed["jobs"]["release-readiness-gate"]["steps"] if "name" in step}
        for name in ("Release evidence baseline (coverage and VM parity)", "Release smoke evidence",
                     "Remaining release producers and architecture verification"):
            self.assertEqual(phase_steps[name]["timeout-minutes"], "360")
        for phase in ("baseline", "smoke", "final-evidence"):
            self.assertIn(f"scripts/run_v1_3_readiness.sh --phase {phase}", commands)
        producer = (ROOT / "scripts/run_v1_3_release_producers.sh").read_text(encoding="utf-8")
        self.assertIn("tests/toolchain/test_v1_3_release_evidence_recipe.py", producer)

    def test_smoke_has_no_in_place_stdlib_mutation_or_duplicate_coverage(self):
        self.assertNotIn("touch lib/stdlib.esk", self.smoke)
        self.assertIn("compile_stdlib_isolated.py", self.smoke)
        isolated = (ROOT / "scripts/compile_stdlib_isolated.py").read_text()
        self.assertIn('compiler = build_dir / "eshkol-run"', isolated)
        self.assertIn('"--shared-lib",', isolated)
        self.assertNotIn("FETCHCONTENT_BASE_DIR", self.smoke + isolated)
        self.assertIn("language_surface_coverage.jsonl", self.smoke)
        self.assertIn('ESHKOL_LANGUAGE_COVERAGE_ALREADY_RUN:-0', self.smoke)
        self.assertIn('ICC_TRACE_DIR="$TRACE_DIR" ./scripts/run_language_coverage.sh', self.smoke)
        self.assertIn('--trace-file "$TRACE_DIR/engine_parity_coverage.jsonl"', self.smoke)
        parity = (ROOT / "scripts/run_engine_parity_coverage.py").read_text()
        self.assertIn('os.environ.get("TRACE_DIR",', parity)

    def test_release_producers_use_approved_lsan_suppressions_and_offline_context_mode(self):
        producers = (ROOT / "scripts/run_v1_3_release_producers.sh").read_text()
        self.assertIn("ASAN_OPTIONS=detect_leaks=1", producers)
        self.assertIn("LSAN_OPTIONS=\"suppressions=$REPO_ROOT/.icc/lsan-suppressions.txt:print_suppressions=0\"", producers)
        self.assertNotIn("detect_leaks=0", producers)
        self.assertIn("check_required_context_consistency.py --offline", producers)
        self.assertIn("ESHKOL_BUILD_JOBS=4 BUILD_DIR=build-asan-ubsan", producers)
        self.assertIn("run_sanitizer_fuzz.sh --skip-build", producers)
        sanitizer = (ROOT / "scripts/run_sanitizer_fuzz.sh").read_text()
        self.assertIn("sanitizer build failed; refusing to run a stale binary", sanitizer)
        build_script = (ROOT / "scripts/build-sanitizer.sh").read_text()
        self.assertIn('"-DLLVM_CONFIG_EXECUTABLE=${LLVM_CONFIG}"', build_script)

    def test_prior_trace_cohort_is_archived_outside_active_root(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as temp:
            root = Path(temp)
            trace, archive = root / "icc_traces", root / "history"
            trace.mkdir()
            old_trace = trace / "old-pass.jsonl"
            old_trace.write_text('{"kind":"eshkol_smoke","name":"example","value":"PASS"}\n')
            old_trace_contents = old_trace.read_text()
            (trace / "release-build-cohort.json").write_text('{"sha256":"old"}\n')
            (trace / "release-phase-state.json").write_text('{"phase_id":"old"}\n')
            nested = trace / "nested" / "older-pass.jsonl"
            nested.parent.mkdir()
            nested.write_text('{"kind":"runtime_event","name":"example","value":"PASS"}\n')
            nested_contents = nested.read_text()
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/archive_release_trace_cohort.py"),
                 "--trace-dir", str(trace), "--archive-root", str(archive)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(list(trace.rglob("*.jsonl")), [])
            self.assertFalse((trace / "release-build-cohort.json").exists())
            self.assertFalse((trace / "release-phase-state.json").exists())
            archived = list(archive.glob("*"))
            self.assertEqual(len(archived), 1)
            self.assertEqual((archived[0] / old_trace.name).read_text(), old_trace_contents)
            self.assertEqual((archived[0] / "nested" / nested.name).read_text(), nested_contents)
            self.assertTrue((archived[0] / "release-build-cohort.json").is_file())
            self.assertTrue((archived[0] / "release-phase-state.json").is_file())

    def test_split_phase_state_is_bound_to_head_run_and_order(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as temp:
            state = Path(temp) / "phase-state.json"

            def invoke(action: str, phase: str | None = None, phase_id: str = "run-7-1"):
                command = [sys.executable, str(ROOT / "scripts/release_phase_state.py"), action,
                           "--repo-root", str(ROOT), "--state", str(state), "--phase-id", phase_id]
                if phase is not None:
                    command.extend(("--phase", phase))
                return subprocess.run(command, capture_output=True, text=True)

            self.assertEqual(invoke("begin").returncode, 0)
            self.assertNotEqual(invoke("require", "smoke").returncode, 0)
            self.assertNotEqual(invoke("mark", "smoke").returncode, 0)
            self.assertNotEqual(invoke("mark", "baseline", phase_id="other-run").returncode, 0)
            self.assertEqual(invoke("mark", "baseline").returncode, 0)
            self.assertEqual(invoke("require", "baseline").returncode, 0)
            self.assertNotEqual(invoke("require", "smoke").returncode, 0)
            self.assertEqual(invoke("mark", "smoke").returncode, 0)
            self.assertNotEqual(invoke("require", "final-evidence").returncode, 0)
            self.assertEqual(invoke("mark", "final-evidence").returncode, 0)
            self.assertEqual(invoke("require", "final-evidence").returncode, 0)

    def test_readiness_wrapper_rejects_bad_phase_and_missing_main_build_before_work(self):
        bad_phase = subprocess.run(
            ["bash", str(ROOT / "scripts/run_v1_3_readiness.sh"), "--phase", "unsupported"],
            capture_output=True, text=True, env={**os.environ, "GITHUB_RUN_ID": "", "ESHKOL_RELEASE_PHASE_ID": ""})
        self.assertEqual(bad_phase.returncode, 2)
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as temp:
            trace_dir = Path(temp) / "traces"
            missing_build = subprocess.run(
                ["bash", str(ROOT / "scripts/run_v1_3_readiness.sh"), "--phase", "baseline"],
                capture_output=True, text=True,
                env={**os.environ, "TRACE_DIR": str(trace_dir), "BUILD_DIR": str(Path(temp) / "absent"),
                     "ESHKOL_RELEASE_PHASE_ID": "negative-plumbing-test", "GITHUB_RUN_ID": "", "GITHUB_RUN_ATTEMPT": ""})
            self.assertNotEqual(missing_build.returncode, 0)
            self.assertIn("required main-build artifact missing", missing_build.stdout + missing_build.stderr)

    def test_verifier_rejects_missing_duplicate_failed_and_unmapped_evidence(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as temp:
            temp_root = Path(temp)
            repo = temp_root / "repo"
            trace = repo / "trace"
            (repo / ".icc").mkdir(parents=True)
            trace.mkdir()
            (repo / ".icc/completion-oracles.yaml").write_bytes((ROOT / ".icc/completion-oracles.yaml").read_bytes())
            records = []
            for criterion in __import__("yaml").safe_load((repo / ".icc/completion-oracles.yaml").read_text())["oracles"]:
                if criterion["name"] == "v1.3.5-evolve":
                    target = criterion
                    break
            for criterion in target["requires"]:
                if "runtime_event" in criterion:
                    event = criterion["runtime_event"]
                    records.extend({"kind": kind, "name": name, "value": "PASS"} for kind in event["event_kinds"] for name in event["event_names"])
                elif "test_evidence" in criterion:
                    records.append({
                        "kind": "test_result",
                        "name": verifier.TEST_ACTIONS[criterion["action"]],
                        "value": {"passed": True, "summary": "fixture"},
                    })
            (trace / "evidence.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            with (trace / "evidence.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": "release_build_cohort", "name": "release_build_cohort_clean", "value": "PASS"}) + "\n")

            def check() -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [sys.executable, str(ROOT / "scripts/verify_v1_3_release_evidence.py"),
                     "--repo-root", str(repo), "--trace-dir", str(trace)],
                    capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")})

            result = check()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            original = (trace / "evidence.jsonl").read_text(encoding="utf-8")
            lines = original.splitlines()
            (trace / "evidence.jsonl").write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
            self.assertNotEqual(check().returncode, 0)
            (trace / "evidence.jsonl").write_text(original + lines[0] + "\n", encoding="utf-8")
            self.assertNotEqual(check().returncode, 0)
            (trace / "evidence.jsonl").write_text(original.replace('"passed": true', '"passed": false', 1), encoding="utf-8")
            self.assertNotEqual(check().returncode, 0)

    def test_main_build_fingerprint_covers_runtime_artifacts_and_detects_mutation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as temp:
            root = Path(temp)
            build, manifest, trace = root / "build", root / "cohort.json", root / "cohort.jsonl"
            build.mkdir()
            for name in cohort.ARTIFACTS:
                (build / name).write_bytes(name.encode("utf-8"))
            capture = subprocess.run(
                [sys.executable, str(ROOT / "scripts/check_release_build_cohort.py"), "capture",
                 "--build-dir", str(build), "--manifest", str(manifest), "--trace", str(trace)],
                capture_output=True, text=True)
            self.assertEqual(capture.returncode, 0, capture.stdout + capture.stderr)
            (build / "stdlib.o").write_bytes(b"changed")
            verify = subprocess.run(
                [sys.executable, str(ROOT / "scripts/check_release_build_cohort.py"), "verify",
                 "--build-dir", str(build), "--manifest", str(manifest), "--trace", str(trace)],
                capture_output=True, text=True)
            self.assertNotEqual(verify.returncode, 0)
            self.assertIn("stdlib.o", verify.stdout)
            self.assertEqual(set(cohort.ARTIFACTS), {
                "eshkol-run", "eshkol-vm-standalone-test", "stdlib.o", "stdlib.bc",
                "libeshkol-runtime.a", "libeshkol-static.a", "libeshkol-agent-ffi.a"})

    def test_ctest_junit_recorder_requires_every_named_action_once_and_green(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as temp:
            root = Path(temp)
            junit, trace = root / "results.xml", root / "results.jsonl"
            suite = ET.Element("testsuite", tests="5", failures="0", errors="0")
            for name in verifier.TEST_ACTIONS.values():
                if name.startswith("ctest::"):
                    ET.SubElement(suite, "testcase", name=name.removeprefix("ctest::"))
            ET.ElementTree(suite).write(junit, encoding="utf-8", xml_declaration=True)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/record_release_ctest_evidence.py"),
                 "--junit", str(junit), "--trace", str(trace), "--ctest-exit-code", "0"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            receipts = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(sum(record["value"]["passed"] for record in receipts), 5)

            suite.find("testcase[@name='python_bindings_capsule_lifetime']").append(ET.Element("failure"))
            ET.ElementTree(suite).write(junit, encoding="utf-8", xml_declaration=True)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/record_release_ctest_evidence.py"),
                 "--junit", str(junit), "--trace", str(trace), "--ctest-exit-code", "0"],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Traceback", result.stderr)

            suite.remove(suite.find("testcase[@name='python_bindings_capsule_lifetime']"))
            ET.ElementTree(suite).write(junit, encoding="utf-8", xml_declaration=True)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/record_release_ctest_evidence.py"),
                 "--junit", str(junit), "--trace", str(trace), "--ctest-exit-code", "0"],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing from JUnit report", result.stdout)

            ET.SubElement(suite, "testcase", name="python_bindings_capsule_lifetime")
            ET.SubElement(suite, "testcase", name="python_bindings_capsule_lifetime")
            ET.ElementTree(suite).write(junit, encoding="utf-8", xml_declaration=True)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/record_release_ctest_evidence.py"),
                 "--junit", str(junit), "--trace", str(trace), "--ctest-exit-code", "8"],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("appeared 2 times", result.stdout)
            self.assertIn("ctest exited 8", result.stdout)


    def test_every_environment_supplied_evidence_path_is_made_absolute(self):
        """A gate that reads TRACE_DIR from its environment normalises it first.

        `ctest --test-dir build --output-junit P` resolves a relative P inside
        build/, so a relative TRACE_DIR made the release recorder look for the
        JUnit report in one place while ctest wrote it in another, and five
        passing required tests were graded as not executed.
        """
        import re
        reads_env = re.compile(r"TRACE_DIR=.*\$\{(TRACE_DIR|ICC_TRACE_DIR):-")
        tracked = subprocess.run(
            ["git", "ls-files", "scripts/*.sh", "scripts/**/*.sh", "tests/*.sh", "tests/**/*.sh"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        consumers = []
        for rel in sorted(set(tracked)):
            lines = (ROOT / rel).read_text(encoding="utf-8", errors="replace").splitlines()
            reads = [i for i, line in enumerate(lines) if reads_env.search(line)]
            if not reads:
                continue
            consumers.append(rel)
            normalised = [i for i, line in enumerate(lines)
                          if line.strip().startswith("eshkol_evidence_abs_var TRACE_DIR ")]
            self.assertTrue(normalised, f"{rel}: reads TRACE_DIR from the environment without normalising it")
            self.assertGreater(normalised[0], reads[-1], f"{rel}: normalises TRACE_DIR before its last assignment")
            between = lines[reads[-1] + 1:normalised[0]]
            uses = [line for line in between
                    if "TRACE_DIR" in line and not line.strip().startswith(("#", "mkdir -p", "."))]
            self.assertEqual(uses, [], f"{rel}: uses TRACE_DIR before it is absolute: {uses}")
            self.assertTrue(any("scripts/lib/evidence_paths.sh" in line for line in lines[:normalised[0]]),
                            f"{rel}: does not source scripts/lib/evidence_paths.sh")
        self.assertGreaterEqual(len(consumers), 19, consumers)
        self.assertIn("scripts/run_v1_3_release_producers.sh", consumers)
        for line in self.workflow.splitlines():
            if line.strip().startswith("TRACE_DIR:"):
                self.assertIn("${{ github.workspace }}/", line, "release.yml passes a relative TRACE_DIR")

    def test_evidence_path_helper_contract(self):
        helper = ROOT / "scripts/lib/evidence_paths.sh"

        def run(script):
            return subprocess.run(["bash", "-c", f'. "{helper}"; {script}'],
                                  capture_output=True, text=True)

        cases = {
            'eshkol_abs_path scripts/icc_traces /repo': "/repo/scripts/icc_traces",
            'eshkol_abs_path ./scripts/icc_traces /repo/': "/repo/scripts/icc_traces",
            'eshkol_abs_path /already/abs /repo': "/already/abs",
            'T=rel/dir; eshkol_evidence_abs_var T /base; printf "%s\\n" "$T"': "/base/rel/dir",
            'T=/abs/dir; eshkol_evidence_abs_var T /base; printf "%s\\n" "$T"': "/abs/dir",
        }
        for script, expected in cases.items():
            result = run(script)
            self.assertEqual(result.returncode, 0, script + result.stderr)
            self.assertEqual(result.stdout.strip(), expected, script)
        for script in ('eshkol_abs_path "" /repo', 'eshkol_abs_path rel relative-base',
                       'T=; eshkol_evidence_abs_var T /base', 'eshkol_evidence_abs_var "bad name" /base'):
            result = run(script)
            self.assertNotEqual(result.returncode, 0, script)
            self.assertEqual(result.stdout, "", script)

    def test_relative_trace_dir_reaches_ctest_as_an_absolute_junit_path(self):
        """Run the producers' own prologue with a relative TRACE_DIR and a stub ctest."""
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as temp:
            stub_dir = Path(temp) / "bin"
            stub_dir.mkdir()
            seen = Path(temp) / "junit-argument"
            stub = stub_dir / "ctest"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                "while [ $# -gt 0 ]; do\n"
                "  if [ \"$1\" = --output-junit ]; then printf '%s' \"$2\" > \"$SEEN\"; fi\n"
                "  shift\n"
                "done\n", encoding="utf-8")
            stub.chmod(0o755)
            producers = (ROOT / "scripts/run_v1_3_release_producers.sh").read_text(encoding="utf-8")
            start = producers.index('ctest_junit="$TRACE_DIR/v1_3_required_ctest.junit.xml"')
            end = producers.index("python3 scripts/record_release_ctest_evidence.py")
            prologue = producers[:producers.index(". scripts/lib/harness_outcome.sh")]
            script = prologue + producers[start:end]
            relative = Path(temp).relative_to(ROOT) / "traces"
            result = subprocess.run(
                ["bash", "-c", script, str(ROOT / "scripts/run_v1_3_release_producers.sh")],
                cwd=ROOT, capture_output=True, text=True,
                env={**os.environ, "PATH": f"{stub_dir}:{os.environ['PATH']}", "SEEN": str(seen),
                     "TRACE_DIR": str(relative), "BUILD_DIR": "build"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(seen.read_text(), str(ROOT / relative / "v1_3_required_ctest.junit.xml"))


if __name__ == "__main__":
    (ROOT / ".scratch").mkdir(exist_ok=True)
    unittest.main()
