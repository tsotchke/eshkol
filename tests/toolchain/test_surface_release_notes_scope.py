#!/usr/bin/env python3
"""Focused coverage for release-note identity scoping in the surface gate."""
import importlib.util
import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "surface_counts", ROOT / "scripts/check_surface_counts.py")
surface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surface)

TAG = "v1.3.5-evolve"
TARGET = f"# Eshkol {TAG} — Release Notes"
RECORD = {"tag": TAG, "version": TAG[1:], "date": datetime.date(2026, 9, 22),
          "status": "SHIPPED", "ctest_total": None, "vm_parity_total": 388}


class ReleaseNotesScopeTests(unittest.TestCase):
    def grade(self, text):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "RELEASE_NOTES.md").write_text(text, encoding="utf-8")
            with patch.object(surface, "REPO_ROOT", temp), \
                    patch.object(surface, "RELEASE_DATE_ANCHORS", ["RELEASE_NOTES.md"]):
                return surface.check_release_doc("RELEASE_NOTES.md", RECORD)

    def test_shipped_notes_alone_pass(self):
        text = TARGET + "\n\n**Release date:** Tuesday, September 22, 2026.\n**Status:** released.\n"
        findings, edits = self.grade(text)
        self.assertEqual(findings, [])
        self.assertEqual(edits, [])

    def test_candidate_before_shipped_archive_does_not_contaminate_scope(self):
        text = ("# Eshkol v1.3.6-evolve — Release Notes\n\n"
                "**Status:** candidate; release evidence is pending.\n\n---\n" +
                TARGET + "\n\n**Release date:** Tuesday, September 22, 2026.\n"
                "**Status:** released.\n")
        findings, edits = self.grade(text)
        self.assertEqual(findings, [])
        self.assertEqual(edits, [])

    def test_archived_wrong_date_and_status_fail_and_edit_is_scoped(self):
        prefix = "# Eshkol v1.3.6-evolve — Release Notes\nPending candidate.\n\n---\n"
        text = (prefix + TARGET + "\n\n**Release date:** Monday, September 21, 2026.\n"
                "**Status:** verification pending.\n")
        findings, edits = self.grade(text)
        self.assertIn("release_date", {f["quantity"] for f in findings})
        self.assertIn("release_status", {f["quantity"] for f in findings})
        self.assertEqual(len(edits), 1)
        edit = edits[0]
        self.assertGreaterEqual(edit["start"], len(prefix))
        self.assertEqual(text[edit["start"]:edit["end"]], "Monday, September 21, 2026")

    def test_candidate_wording_outside_scoped_section_is_excluded(self):
        text = ("# Eshkol v1.3.6-evolve — Release Notes\n"
                "verification is pending; release candidate.\n\n---\n" +
                TARGET + "\n\n**Release date:** Tuesday, September 22, 2026.\n")
        findings, _ = self.grade(text)
        self.assertEqual(findings, [])

    def test_missing_duplicate_and_wrong_tag_identity_fail(self):
        cases = ["# Eshkol v1.3.6-evolve — Release Notes\n",
                 TARGET + "\n---\n" + TARGET + "\n",
                 "# Eshkol v1.3.4-evolve — Release Notes\n"]
        for text in cases:
            with self.subTest(text=text):
                findings, _ = self.grade(text)
                identity = [f for f in findings if f["quantity"] == "release_notes_identity"]
                self.assertEqual(len(identity), 1)


if __name__ == "__main__":
    unittest.main()
