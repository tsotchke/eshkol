#!/usr/bin/env python3
"""Offline contract checks for reusing a fully successful CI tree."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ci_reference_run", ROOT / "scripts/ci_reference_run.py")
ci_reference_run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ci_reference_run)
FIXTURE = json.loads((ROOT / "tests/toolchain/fixtures/ci_reference_run_success.json")
                     .read_text(encoding="utf-8"))
SHA = FIXTURE["run"]["head_sha"]
WORKFLOW = "CI"


class CiReferenceRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run = copy.deepcopy(FIXTURE["run"])
        self.jobs = copy.deepcopy(FIXTURE["jobs"])

    def assert_rejected(self) -> None:
        accepted, _ = ci_reference_run.verify(self.run, self.jobs, SHA, WORKFLOW)
        self.assertFalse(accepted)

    def test_genuine_full_reference_passes(self) -> None:
        accepted, reason = ci_reference_run.verify(self.run, self.jobs, SHA, WORKFLOW)
        self.assertTrue(accepted, reason)
        self.assertEqual(self.jobs["total_count"], 30)
        self.assertEqual(sum(j["conclusion"] == "success" for j in self.jobs["jobs"]), 28)

    def test_label_contract_is_under_github_limit_and_tree_only(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("ci-tree:\\([0-9a-fA-F]\\{40\\}\\)$", workflow)
        self.assertNotIn("ci-equivalent:", workflow)
        self.assertIn('"$equivalence_kind" == "tree"', workflow)
        self.assertEqual(len("ci-tree:" + "a" * 40), 48)

    def test_failed_label_lookup_falls_back_under_runner_bash_errexit(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        step = workflow.split("        run: |\n", 1)[1]
        shell_options = step.split('          changed_files=""', 1)[0]
        reference = '          equivalent_to=""' + step.split(
            '          equivalent_to=""', 1)[1].split(
                "          # docs_only stays", 1)[0]
        script = textwrap.dedent(shell_options) + '\nimpact="full"\n' + textwrap.dedent(reference)
        script += '\nprintf "fallback=%s\\n" "$impact"\n'
        with tempfile.TemporaryDirectory() as temp:
            gh = Path(temp) / "gh"
            gh.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            gh.chmod(0o755)
            env = dict(os.environ, PATH=temp + os.pathsep + os.environ["PATH"],
                       EVENT_NAME="pull_request", PR_NUMBER="763", REPO="fixture/repo")
            result = subprocess.run(["bash", "-e", "-c", script], env=env,
                                    text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fallback=full", result.stdout)

    def test_newer_failed_run_cannot_reuse_older_success(self) -> None:
        failed = dict(self.run, id=self.run["id"] + 1,
                      run_number=self.run["run_number"] + 1,
                      conclusion="failure")
        payload = [{"total_count": 2, "workflow_runs": [self.run, failed]}]
        selected = ci_reference_run.latest_run(payload, SHA, WORKFLOW)
        self.assertEqual(selected["id"], failed["id"])
        self.run = selected
        self.assert_rejected()

    def test_rejects_skipped_matrix_lane_and_equivalence_only_job_set(self) -> None:
        lane = next(j for j in self.jobs["jobs"] if j["name"] == "linux-x64-lite")
        lane["conclusion"] = "skipped"
        self.assert_rejected()
        self.setUp()
        self.jobs["jobs"] = [j for j in self.jobs["jobs"] if j["name"] == "changes"]
        self.jobs["total_count"] = 1
        self.assert_rejected()

    def test_rejects_failed_advisory_cancelled_and_incomplete_jobs(self) -> None:
        advisory = next(j for j in self.jobs["jobs"]
                        if j["name"] == "mesh-gate-advisory (advisory, no-op)")
        advisory["conclusion"] = "failure"
        self.assert_rejected()
        self.setUp()
        advisory = next(j for j in self.jobs["jobs"]
                        if j["name"] == "mesh-gate-advisory (advisory, no-op)")
        advisory["conclusion"] = "cancelled"
        self.assert_rejected()
        self.setUp()
        next(j for j in self.jobs["jobs"] if j["name"] == "surface-manifest")["status"] = "in_progress"
        self.assert_rejected()
        self.run["status"] = "in_progress"
        self.assert_rejected()

    def test_rejects_truncated_or_wrong_attempt_job_payloads(self) -> None:
        self.jobs["total_count"] += 1
        self.assert_rejected()
        self.setUp()
        self.jobs["jobs"][0]["run_attempt"] += 1
        self.assert_rejected()

    def test_partial_rerun_retains_same_commit_successful_jobs(self) -> None:
        # Model the API's filter=latest response after a successful retry:
        # one newly executed job and the same run's retained passing jobs.
        self.run["run_attempt"] = 2
        self.jobs["jobs"][0]["run_attempt"] = 2
        accepted, reason = ci_reference_run.verify(self.run, self.jobs, SHA, WORKFLOW)
        self.assertTrue(accepted, reason)
        self.jobs["jobs"][0]["run_attempt"] = 1
        self.assert_rejected()

    def test_rejects_cross_run_or_cross_commit_job_records(self) -> None:
        self.jobs["jobs"][0]["run_id"] += 1
        self.assert_rejected()
        self.setUp()
        self.jobs["jobs"][0]["head_sha"] = "f" * 40
        self.assert_rejected()

    def test_rejects_malformed_duplicate_and_missing_job_records(self) -> None:
        for mutation in ("malformed", "duplicate_id", "duplicate_name", "missing"):
            with self.subTest(mutation=mutation):
                self.setUp()
                rows = self.jobs["jobs"]
                if mutation == "malformed":
                    rows[0] = None
                elif mutation == "duplicate_id":
                    rows[0]["id"] = rows[1]["id"]
                elif mutation == "duplicate_name":
                    rows[0]["name"] = rows[1]["name"]
                else:
                    rows.pop()
                    self.jobs["total_count"] -= 1
                self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
