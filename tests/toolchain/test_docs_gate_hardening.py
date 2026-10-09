#!/usr/bin/env python3
"""Negative controls for release documentation evidence gates."""
import importlib.util
from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


catalogue = load("build_example_catalogue")
api = load("check_public_api_docs")
generator = load("gen_api_docs")


class ReceiptControls(unittest.TestCase):
    def _ingest_fixture(self, jit_log=None, aot_log=None, omit=None):
        entry = {"path": "examples/fixture.esk", "source_sha256": "a" * 64, "registrations": [], "key_output": []}
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            base = Path(directory)
            (base / "docs/examples").mkdir(parents=True)
            (base / "examples").mkdir()
            (base / "examples/fixture.esk").write_text("fixture\n", encoding="utf-8")
            records = {
                "path": entry["path"], "source_sha256": entry["source_sha256"],
                "jit": {"status": "PASS", "exit": 0, "result_line": "RESULT: ALL PASS", "log": "jit.log"},
                "aot": {"status": "PASS", "exit": 0, "compile_exit": 0, "result_line": "RESULT: ALL PASS", "log": "aot.log"},
            }
            if omit:
                records[omit].pop("log")
            receipt = base / "receipts.json"
            receipt.write_text(json.dumps({"schema": "eshkol.example-receipts.v1", "release_sha": "b" * 40, "records": [records]}), encoding="utf-8")
            if jit_log is not None:
                (base / "jit.log").write_text(jit_log, encoding="utf-8")
            if aot_log is not None:
                (base / "aot.log").write_text(aot_log, encoding="utf-8")
            with patch.object(catalogue, "inventory", return_value=[entry["path"]]), \
                 patch.object(catalogue, "registration_matrix", return_value={}), \
                 patch.object(catalogue, "load_catalogue", return_value={"entries": [entry]}), \
                 patch.object(catalogue, "load_measurements", return_value={}):
                return catalogue.ingest(base, receipt)

    def test_ingest_binds_pass_to_each_observed_log(self):
        with self.assertRaises(catalogue.CatalogueError):
            self._ingest_fixture("RESULT: ALL PASS\n", "wrong verdict\n")

    def test_ingest_rejects_missing_nonexistent_and_empty_logs(self):
        cases = (
            (None, "RESULT: ALL PASS\n", None),
            ("RESULT: ALL PASS\n", None, None),
            ("", "RESULT: ALL PASS\n", None),
            ("RESULT: ALL PASS\n", "", None),
            ("RESULT: ALL PASS\n", "RESULT: ALL PASS\n", "jit"),
        )
        for jit, aot, omit in cases:
            with self.subTest(jit=jit, aot=aot, omit=omit), self.assertRaises(catalogue.CatalogueError):
                self._ingest_fixture(jit, aot, omit)

    def test_ingest_preserves_both_not_run_records(self):
        entry = {"path": "examples/fixture.esk", "source_sha256": "a" * 64, "registrations": [], "key_output": []}
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as directory:
            base = Path(directory)
            (base / "docs/examples").mkdir(parents=True)
            (base / "examples").mkdir()
            receipt = base / "receipts.json"
            receipt.write_text(json.dumps({"schema": "eshkol.example-receipts.v1", "release_sha": "b" * 40, "records": []}), encoding="utf-8")
            with patch.object(catalogue, "inventory", return_value=[entry["path"]]), \
                 patch.object(catalogue, "registration_matrix", return_value={}), \
                 patch.object(catalogue, "load_catalogue", return_value={"entries": [entry]}), \
                 patch.object(catalogue, "load_measurements", return_value={}):
                result = catalogue.ingest(base, receipt, not_run={entry["path"]: "lane not scheduled"})
            self.assertEqual(result["records"], 1)

    def test_pass_requires_verdict_and_ran_ok_rejects_one(self):
        with self.assertRaises(catalogue.CatalogueError):
            catalogue._validate_run({"status": "PASS", "exit": 0, "result_line": None}, False)
        with self.assertRaises(catalogue.CatalogueError):
            catalogue._validate_run({"status": "RAN-OK", "exit": 0, "result_line": "RESULT: ALL PASS"}, False)

    def test_status_and_compile_exit_are_coherent(self):
        with self.assertRaises(catalogue.CatalogueError):
            catalogue._validate_run({"status": "PASS", "exit": 3, "result_line": "RESULT: ALL PASS"}, False)
        with self.assertRaises(catalogue.CatalogueError):
            catalogue._validate_run({"status": "RAN-OK", "exit": 0, "compile_exit": 2, "result_line": None}, True)
        with self.assertRaises(catalogue.CatalogueError):
            catalogue._validate_run({"status": "COMPILE-FAIL", "compile_exit": 0}, True)

    def test_required_logs_are_fail_closed(self):
        run = {"status": "RAN-OK", "exit": 0, "result_line": None}
        with self.assertRaises(catalogue.CatalogueError):
            catalogue._validate_run(run, False, "fixture", require_log=True)


class ManifestControls(unittest.TestCase):
    def test_reviewed_manifest_counts_are_pinned(self):
        headers = api.read_manifest(ROOT / "docs/api/public_surface.tsv", 5)
        exports = api.read_manifest(ROOT / "docs/reference/stdlib/public_exports.tsv", 4)
        self.assertEqual(len(headers), api.EXPECTED_HEADER_COUNT)
        self.assertEqual(len(exports), api.EXPECTED_EXPORT_COUNT)
        self.assertEqual(len(generator.load_public_surface_manifest()), generator.EXPECTED_PUBLIC_SURFACE_COUNT)

    def test_truncated_manifest_is_rejected(self):
        rows = api.read_manifest(ROOT / "docs/api/public_surface.tsv", 5)[:-1]
        errors = api.check_manifest(rows, api.EXPECTED_HEADER_COUNT, ROOT, ROOT / "docs/api/INDEX.md", "docs/api/public_surface.md")
        self.assertTrue(any("expected 132" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
