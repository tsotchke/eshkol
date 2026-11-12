#!/usr/bin/env python3
"""Unit checks for scripts/doc_truth_ratchet.py on synthetic ICC claims.

The phase-1 CTest (doc_truth_ratchet_phase1) runs the real ranking; this
test pins the ranking order, the quota comparison, the phase-2 --enforce
exit status and the no-raise rule of update-quota. Outputs go to a
directory under the current working directory (the build tree in CTest).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "doc_truth_ratchet.py"
sys.path.insert(0, str(SCRIPT.parent))
import doc_truth_ratchet as dtr  # noqa: E402


CONTRADICTED = dtr.ICC_CONTRADICTED_STATUS  # ICC's status for a contradicted claim


def claim(path: str, status: str, ctype: str = "file_path") -> dict:
    return {"claim_type": ctype, "status": status, "location": {"path": path, "line": 1}}


PAYLOAD = {
    "schema_version": "1.0",
    "docs": [{"path": p} for p in ("docs/a.md", "docs/b.md", "docs/c.md", "docs/reports/DOC_TRUTH_RANKING.md")],
    "claims": [
        claim("docs/a.md", "grounded"), claim("docs/a.md", "unknown", "arity"),
        claim("docs/b.md", CONTRADICTED), claim("docs/b.md", "grounded"),
        claim("docs/c.md", "table_row_mismatch", "table_row"), claim("docs/c.md", "unknown"),
        claim("docs/c.md", "unknown"),
        claim("docs/reports/DOC_TRUTH_RANKING.md", CONTRADICTED),
        claim("README.md", CONTRADICTED),
    ],
    "findings": [],
}


class DocTruthRatchetTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="doc-truth-test-", dir=os.getcwd()))

    def test_ranking_order_and_scope(self):
        r = dtr.rank(PAYLOAD, "docs/", "test")
        order = [d["path"] for d in r["documents"]]
        # contradicted first (b: 1 contradicted; c: 1 table-contradicted, more unsupported)
        self.assertEqual(order, ["docs/c.md", "docs/b.md", "docs/a.md"])
        self.assertNotIn("docs/reports/DOC_TRUTH_RANKING.md", order)  # generated output excluded
        self.assertEqual(r["totals"]["claims"], 7)  # README.md is outside docs/
        self.assertEqual(r["totals"]["contradicted_total"], 2)
        self.assertEqual(r["totals"]["unsupported"], 5)

    def test_compare_and_enforce(self):
        ranking = dtr.rank(PAYLOAD, "docs/", "test")
        quota_path = self.work / "quota.json"
        summary = self.work / "summary.md"
        dtr.write_quota(ranking, quota_path, summary, None)
        quota = json.loads(quota_path.read_text())
        self.assertEqual(dtr.compare(ranking, quota)["over_quota"], [])
        worse = json.loads(json.dumps(PAYLOAD))
        worse["claims"].append(claim("docs/a.md", CONTRADICTED))
        claims = self.work / "worse.json"
        claims.write_text(json.dumps(worse))
        base = [sys.executable, str(SCRIPT), "check", "--source", str(ROOT), "--build-dir", str(self.work),
                "--quota", str(quota_path), "--claims", str(claims)]
        report = subprocess.run(base, text=True, capture_output=True)
        self.assertEqual(report.returncode, 0, report.stdout + report.stderr)
        self.assertIn("DOC-TRUTH REPORT (phase 1, not release-blocking)", report.stdout)
        self.assertIn("over quota: 1", report.stdout)
        enforced = subprocess.run(base + ["--enforce"], text=True, capture_output=True)
        self.assertEqual(enforced.returncode, 1)
        self.assertIn("FAIL: documents above their doc-truth quota", enforced.stdout)
        cmp = json.loads((self.work / "doc_truth" / "comparison.json").read_text())
        self.assertTrue(cmp["recomputed"])

    def test_update_quota_never_raises_without_consent(self):
        ranking = dtr.rank(PAYLOAD, "docs/", "test")
        quota_path = self.work / "quota.json"
        dtr.write_quota(ranking, quota_path, self.work / "summary.md", None)
        worse = json.loads(json.dumps(PAYLOAD))
        worse["claims"].append(claim("docs/b.md", CONTRADICTED))
        rpath = self.work / "ranking.json"
        rpath.write_text(json.dumps(dtr.rank(worse, "docs/", "test")))
        cmd = [sys.executable, str(SCRIPT), "update-quota", "--ranking", str(rpath),
               "--quota", str(quota_path), "--summary", str(self.work / "summary.md")]
        refused = subprocess.run(cmd, text=True, capture_output=True)
        self.assertEqual(refused.returncode, 1)
        accepted = subprocess.run(cmd + ["--allow-increase"], text=True, capture_output=True)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_unavailable_icc_is_reported_and_fails_only_when_enforced(self):
        ranking = dtr.rank(PAYLOAD, "docs/", "test")
        quota_path = self.work / "quota.json"
        dtr.write_quota(ranking, quota_path, self.work / "summary.md", None)
        env = dict(os.environ, ESHKOL_ICC_BIN=str(self.work / "missing"), PATH="/usr/bin:/bin",
                   HOME=str(self.work))
        base = [sys.executable, str(SCRIPT), "check", "--source", str(self.work), "--build-dir",
                str(self.work), "--quota", str(quota_path)]
        out = subprocess.run(base, text=True, capture_output=True, env=env)
        self.assertEqual(out.returncode, 0)
        self.assertIn("DOC-TRUTH NOT RECOMPUTED", out.stdout)
        out = subprocess.run(base + ["--enforce"], text=True, capture_output=True, env=env)
        self.assertEqual(out.returncode, 1)


if __name__ == "__main__":
    unittest.main()
