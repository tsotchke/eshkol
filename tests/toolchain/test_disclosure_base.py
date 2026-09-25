#!/usr/bin/env python3
"""Focused integration checks for CI's incoming disclosure scan base."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.select_disclosure_base import choose_base


class DisclosureBaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="disclosure-base-")
        self.root = pathlib.Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-b", "master")
        self.git("config", "user.email", "ci@example.invalid")
        self.git("config", "user.name", "CI test")
        self.write_commit("common", "common")
        self.common = self.git("rev-parse", "HEAD")
        self.git("checkout", "-b", "release")
        self.write_commit("release-one", "release-one")
        self.write_commit("release-two", "release-two")
        self.release_head = self.git("rev-parse", "HEAD")
        self.git("checkout", "master")
        self.write_commit("default-only", "default-only")
        self.default_head = self.git("rev-parse", "HEAD")
        self.git("init", "--bare", str(self.root / "origin.git"), cwd=self.root)
        self.git("remote", "add", "origin", str(self.root / "origin.git"))
        self.git("push", "origin", "master")
        self.original_cwd = pathlib.Path.cwd()
        import os
        os.chdir(self.repo)

    def tearDown(self) -> None:
        import os
        os.chdir(self.original_cwd)
        self.temp.cleanup()

    def git(self, *args: str, cwd: pathlib.Path | None = None) -> str:
        result = subprocess.run(
            ["git", *args], cwd=cwd or self.repo, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout.strip()

    def write_commit(self, filename: str, content: str) -> None:
        (self.repo / filename).write_text(content + "\n", encoding="utf-8")
        self.git("add", filename)
        self.git("commit", "-m", filename)

    def test_manual_release_branch_scans_from_merge_base_with_fetched_default(self) -> None:
        base = choose_base(
            event="workflow_dispatch", head=self.release_head,
            ref="refs/heads/release", default_branch="master",
        )
        self.assertEqual(base, self.common)

    def test_manual_default_branch_scans_its_actual_parent_diff(self) -> None:
        base = choose_base(
            event="workflow_dispatch", head=self.default_head,
            ref="refs/heads/master", default_branch="master",
        )
        self.assertEqual(base, self.common)

    def test_push_before_and_merge_group_base_are_preserved(self) -> None:
        pushed = choose_base(
            event="push", head=self.default_head, ref="refs/heads/master",
            default_branch="master", before=self.common,
        )
        grouped = choose_base(
            event="merge_group", head=self.release_head,
            ref="refs/heads/gh-readonly-queue/master/pr-1", default_branch="master",
            merge_group_base=self.common,
        )
        self.assertEqual(pushed, self.common)
        self.assertEqual(grouped, self.common)

    def test_unresolvable_manual_history_fails_closed(self) -> None:
        other = self.root / "unrelated"
        other.mkdir()
        subprocess.run(["git", "init", "-b", "master"], cwd=other, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "config", "user.email", "ci@example.invalid"], cwd=other, check=True)
        subprocess.run(["git", "config", "user.name", "CI test"], cwd=other, check=True)
        (other / "only").write_text("unrelated\n", encoding="utf-8")
        subprocess.run(["git", "add", "only"], cwd=other, check=True)
        subprocess.run(["git", "commit", "-m", "unrelated"], cwd=other, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.git("remote", "set-url", "origin", str(other))
        with self.assertRaises(subprocess.CalledProcessError):
            choose_base(
                event="workflow_dispatch", head=self.release_head,
                ref="refs/heads/release", default_branch="master",
            )

    def test_workflow_uses_selector_for_non_pr_scan(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("python3 scripts/select_disclosure_base.py", workflow)
        self.assertIn("DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}", workflow)
        self.assertIn("MERGE_GROUP_BASE_SHA: ${{ github.event.merge_group.base_sha }}", workflow)
        self.assertIn("PUSH_BEFORE: ${{ github.event.before }}", workflow)
        self.assertIn("python3 scripts/check_disclosure.py --base \"$BASE_SHA\" --head \"$HEAD_SHA\"", workflow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
