#!/usr/bin/env python3
"""Release decisions must reject stale evidence and incomplete prerequisites."""
from datetime import datetime, timezone
import base64
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("release_autopilot", ROOT / "scripts/release_autopilot.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SHA = "a" * 40


class Decisions(unittest.TestCase):
    def test_required_check_cannot_be_absent_skipped_or_running(self):
        for conclusion in (None, "", "SKIPPED", "FAILURE", "CANCELLED"):
            with self.subTest(conclusion=conclusion), self.assertRaises(module.Wait):
                module.require_checks([{"name": "sanitizer", "conclusion": conclusion}], {"sanitizer"})
        with self.assertRaises(module.Wait):
            module.require_checks([], {"sanitizer"})
        module.require_checks([{"name": "sanitizer", "conclusion": "SUCCESS"}], {"sanitizer"})

    def test_latest_rerun_wins_and_nonrequired_failures_still_block(self):
        rows = [{"name": "sanitizer", "conclusion": "SUCCESS", "startedAt": "2026-09-13"},
                {"name": "sanitizer", "conclusion": "FAILURE", "startedAt": "2026-09-14"}]
        for ordering in (rows, rows[::-1]):
            with self.assertRaises(module.Wait):
                module.require_checks(ordering, {"sanitizer"})
        with self.assertRaises(module.Wait):
            module.require_checks([rows[0], {"name": "optional", "conclusion": "FAILURE"}], {"sanitizer"})

    def test_receipt_requires_exact_identity_and_numeric_ready_100(self):
        good = {"schema": "eshkol.release-readiness.v1", "sha": SHA, "run_id": 10,
                "run_attempt": 2, "target": "v1.3.5-evolve", "status": "ready", "score": 100}
        run = {"id": 10, "run_attempt": 2}
        module.require_receipt(good, SHA, run)
        for key, value in (("sha", "b" * 40), ("run_id", 9), ("run_attempt", 1),
                           ("target", "v1.3.4-evolve"), ("status", "blocked"),
                           ("score", "100"), ("score", True), ("score", 99), ("schema", "other")):
            with self.subTest(key=key, value=value), self.assertRaises(module.Wait):
                module.require_receipt({**good, key: value}, SHA, run)
        with self.assertRaises(module.Wait):
            module.require_receipt({}, SHA, run)

    def test_montreal_publication_window_is_enforced_in_utc(self):
        config = {"not_before": "2026-09-14T09:00:00-04:00", "expires_at": "2026-09-15T00:00:00-04:00"}
        for time in ("2026-09-14T12:59:59Z", "2026-09-15T04:00:00Z"):
            with self.assertRaises(module.Wait):
                module.require_window(config, module.timestamp(time))
        module.require_window(config, module.timestamp("2026-09-14T13:00:00Z"))
        module.require_window(config, module.timestamp("2026-09-14T16:00:00Z"))
        with self.assertRaises(ValueError):
            module.timestamp("2026-09-14T09:00:00")

    def test_hold_is_binding_until_resolved_or_expired(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as folder:
            path = Path(folder)
            (path / "thoughts.jsonl").write_text("")
            hold = {"stream": "dispatch", "kind": "hold", "task_id": "eshkol-release",
                    "ts": "2026-09-14T12:00:00Z", "why": "eshkol release is on hold"}
            (path / "dispatch.jsonl").write_text(json.dumps(hold) + "\n")
            now = module.timestamp("2026-09-14T13:00:00Z")
            self.assertTrue(module.pending_hold(path, now))
            release = {**hold, "kind": "complete", "ts": "2026-09-14T12:30:00Z"}
            with (path / "dispatch.jsonl").open("a") as out:
                out.write(json.dumps(release) + "\n")
            self.assertIsNone(module.pending_hold(path, now))


class Controller(unittest.TestCase):
    def setUp(self):
        (ROOT / ".scratch").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".scratch")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.config = {"repo": "tsotchke/eshkol", "tag": "v1.3.5-evolve", "gh": "gh",
                       "checkout": str(self.path / "checkout"), "required_contexts": ["sanitizer"],
                       "lanes": [{"branch": "fix/pending", "worktree": str(self.path / "absent")}],
                       "presence_root": str(self.path), "not_before": "2026-09-14T09:00:00-04:00",
                       "expires_at": "2026-09-15T00:00:00-04:00"}
        self.release = module.Release(self.config, self.path / "state")

    def test_preview_never_executes_mutation(self):
        with patch.object(module.subprocess, "run") as run, self.assertRaises(module.Wait):
            self.release.gh("pr", "merge", "628", mutate=True)
        run.assert_not_called()

    def test_protection_reduction_is_not_accepted(self):
        with patch.object(self.release, "api", return_value={"contexts": []}), self.assertRaises(module.Wait):
            self.release.required()

    def test_missing_open_or_unpublished_hardening_blocks(self):
        for prs in ([], [{"state": "OPEN"}], [{"state": "CLOSED"}]):
            with patch.object(self.release, "gh", return_value=json.dumps(prs)), self.assertRaises(module.Wait):
                self.release.lanes_ready()
        prs = [{"number": 700, "state": "MERGED", "headRefOid": SHA, "baseRefName": "master"}]
        with patch.object(self.release, "gh", return_value=json.dumps(prs)), patch.object(self.release, "api", return_value=[]):
            self.release.lanes_ready()
        self.assertEqual(self.release.state["lane_prs"]["fix/pending"], 700)

    def test_merged_hardening_on_unrelated_base_does_not_satisfy_lane(self):
        prs = [{"number": 700, "state": "MERGED", "headRefOid": SHA,
                "baseRefName": "feature/unrelated"}]
        with patch.object(self.release, "gh", return_value=json.dumps(prs)), \
                self.assertRaisesRegex(module.Wait, "outside the release branches"):
            self.release.lanes_ready()

    def test_merged_lane_commit_must_be_in_release_ancestry(self):
        self.release.state["lane_prs"] = {"fix/pending": 700}
        pr = {"mergeCommit": {"oid": SHA}}
        results = [module.subprocess.CompletedProcess([], 1),
                   module.subprocess.CompletedProcess([], 1)]
        with patch.object(self.release, "pr", return_value=pr), \
                patch.object(module.subprocess, "run", side_effect=results):
            with self.assertRaisesRegex(module.Wait, "absent from the release source"):
                self.release.require_lane_inclusion("origin/master", "origin/integration/astra-v135")

    def test_merged_lane_commit_in_any_release_ref_is_accepted(self):
        self.release.state["lane_prs"] = {"fix/pending": 700}
        pr = {"mergeCommit": {"oid": SHA}}
        results = [module.subprocess.CompletedProcess([], 1),
                   module.subprocess.CompletedProcess([], 0)]
        with patch.object(self.release, "pr", return_value=pr), \
                patch.object(module.subprocess, "run", side_effect=results) as run:
            self.release.require_lane_inclusion("origin/master", "origin/integration/astra-v135")
        self.assertEqual(run.call_count, 2)

    def test_no_duplicate_workflow_dispatch_while_another_run_is_active(self):
        data = {"workflow_runs": [{"id": 10, "head_sha": "b" * 40,
                                  "event": "workflow_dispatch", "status": "in_progress"}]}
        with patch.object(self.release, "api", return_value=data), patch.object(self.release, "gh") as gh:
            with self.assertRaisesRegex(module.Wait, "no duplicate"):
                self.release.proof("master", SHA)
        gh.assert_not_called()

    def test_failed_run_is_not_blindly_retried(self):
        data = {"workflow_runs": [{"id": 10, "head_sha": SHA, "event": "workflow_dispatch",
                                  "status": "completed", "conclusion": "cancelled", "html_url": "run"}]}
        with patch.object(self.release, "api", return_value=data), patch.object(self.release, "gh") as gh:
            with self.assertRaisesRegex(module.Wait, "no blind retry"):
                self.release.proof("master", SHA)
        gh.assert_not_called()

    def test_green_advisory_run_without_receipt_cannot_authorize_tag(self):
        data = {"workflow_runs": [{"id": 10, "head_sha": SHA, "event": "workflow_dispatch",
                                  "status": "completed", "conclusion": "success", "html_url": "run"}]}
        with patch.object(self.release, "api", side_effect=[data, {"artifacts": []}]):
            with self.assertRaisesRegex(module.Wait, "lacks strict readiness"):
                self.release.proof("master", SHA)

    def test_pause_prevents_even_remote_queries(self):
        (self.release.directory / "PAUSE").touch()
        with patch.object(self.release, "required") as required:
            with self.assertRaisesRegex(module.Wait, "Paused"):
                self.release.step()
        required.assert_not_called()

    def test_homebrew_fallback_preserves_resource_pins_and_uses_blob_precondition(self):
        formula = ('class Eshkol < Formula\n'
            '  url "https://github.com/tsotchke/eshkol/archive/refs/tags/v1.3.4-evolve.tar.gz"\n'
            '  sha256 "' + '1' * 64 + '"\n'
            '  resource "dependency" do\n    url "https://example.test/dependency.tar.gz"\n'
            '    sha256 "' + '2' * 64 + '"\n  end\nend\n')
        data = {"sha": "old-blob", "content": base64.b64encode(formula.encode()).decode()}
        self.release.execute = True
        with patch.object(self.release, "gh", return_value=json.dumps(data)) as gh:
            with patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(b"source archive")):
                with self.assertRaisesRegex(module.Wait, "formula updated"):
                    self.release.ensure_homebrew()
        args = gh.call_args.args
        self.assertIn("sha=old-blob", args)
        content = next(a.removeprefix("content=") for a in args if a.startswith("content="))
        rewritten = base64.b64decode(content).decode()
        self.assertIn('    sha256 "' + '2' * 64 + '"', rewritten)
        self.assertIn(hashlib.sha256(b"source archive").hexdigest(), rewritten)
        self.assertIn("v1.3.5-evolve.tar.gz", rewritten)
        self.assertTrue(gh.call_args.kwargs["mutate"])

    def test_publication_does_not_finish_without_homebrew_verification(self):
        self.config["assets"] = [["linux-x64-lite", "tar.gz"]]
        release = {"draft": False, "prerelease": False, "html_url": "release", "assets": [
            {"name": "eshkol-v1.3.5-evolve-linux-x64-lite.tar.gz", "size": 100},
            {"name": "SHA256SUMS.txt", "size": 100}]}
        runs = {"workflow_runs": [{"head_sha": SHA, "event": "push", "head_branch": "v1.3.5-evolve",
                                   "conclusion": "success"}]}
        with patch.object(self.release, "api", side_effect=[release, runs]):
            with patch.object(self.release, "ensure_homebrew", side_effect=module.Wait("tap pending")):
                with self.assertRaisesRegex(module.Wait, "tap pending"):
                    self.release.verify_published(SHA)
        self.assertFalse(self.release.state.get("completed"))

    def test_refreshed_candidate_notes_are_finalized_only_with_recorded_proof(self):
        self.release.checkout.mkdir()
        path = self.release.checkout / "RELEASE_NOTES.md"
        path.write_text('# Eshkol v1.3.5-evolve — Release Notes\n\n'
            '**Status:** refreshed release candidate; final verification and publication are pending. '
            'Hardening remains open. The September 11 measurements below are historical.\n\n'
            '<!-- RELEASE_EVIDENCE_PENDING -->\n')
        self.release.execute = True
        with patch.object(self.release, "git"):
            with self.assertRaisesRegex(module.Wait, "notes finalized"):
                self.release.finalize_notes({"headRefName": "release/v135-cut-final"}, {"html_url": "https://github.com/run/10"})
        content = path.read_text()
        self.assertNotIn("Hardening remains open", content)
        self.assertNotIn("RELEASE_EVIDENCE_PENDING", content)
        self.assertIn("September 11 measurements below are historical", content)
        self.assertIn("https://github.com/run/10", content)

    def test_restart_preserves_completed_release_and_does_not_publish_twice(self):
        self.release.state.update(completed=True, release_url="https://github.com/tsotchke/eshkol/releases/tag/v1.3.5-evolve")
        module.write_json(self.release.state_path, self.release.state)
        restarted = module.Release(self.config, self.release.directory, execute=True)
        with patch.object(restarted, "api") as api:
            self.assertIn("Already complete", restarted.step())
        api.assert_not_called()


if __name__ == "__main__":
    (ROOT / ".scratch").mkdir(exist_ok=True)
    unittest.main()
