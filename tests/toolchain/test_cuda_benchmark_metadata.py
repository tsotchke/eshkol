#!/usr/bin/env python3
"""CPU-only reducer fixtures for Ozaki benchmark backend metadata."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
REDUCER_PATH = ROOT / "bench/axes/02_ozaki_gemm_reduce.py"
SPEC = importlib.util.spec_from_file_location("ozaki_reducer", REDUCER_PATH)
REDUCER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REDUCER)


class OzakiReducerMetadataTests(unittest.TestCase):
    def fixture(self, directory, backend, dispatched=True, available=True):
        work = Path(directory) / "work"
        work.mkdir()
        for tag in ("amx", "ozaki", "ozaki-fast"):
            (work / f"throughput.{tag}.out").write_text(
                "BENCH n=1024 ns_samples=[1000000,1100000]\n")
            (work / f"accuracy.{tag}.out").write_text(
                "SAMPLE r=0 c=0 approx=1.0000001 exact=1.0 relerr=1e-7\n")
        if backend == "cuda":
            kernel = "INT8-Ozaki T=6" if dispatched else "cublasDgemm"
            for tag in ("ozaki", "ozaki-fast"):
                (work / f"throughput.{tag}.stderr").write_text(
                    f"[GPU] matmul 1024x1024 @ 1024x1024 -> {kernel} (fixture)\n")
                (work / f"accuracy.{tag}.stderr").write_text(
                    f"[GPU] matmul 1024x1024 @ 1024x1024 -> {kernel} (fixture)\n")
        out_json = Path(directory) / "result.json"
        out_md = Path(directory) / "result.md"
        argv = [
            "reducer", "--workdir", str(work), "--amx-ok", "1",
            "--ozaki-ok", "1" if available else "0",
            "--ozaki-fast-ok", "1" if available else "0",
            "--gpu-backend", backend, "--acc-n", "1024" if backend == "cuda" else "64",
            "--json-out", str(out_json), "--md-out", str(out_md),
        ]
        with patch("sys.argv", argv):
            REDUCER.main()
        return json.loads(out_json.read_text()), out_md.read_text()

    def test_cuda_reports_measured_error_separately_from_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            result, markdown = self.fixture(directory, "cuda")
        self.assertEqual(result["cuda_t6_relative_error_bound"], 1e-13)
        self.assertEqual(result["accuracy_vs_exact_rational_reference"]["ozaki"]["max_relerr"], 1e-7)
        self.assertIn("CUDA INT8-Ozaki T=6", markdown)
        self.assertIn("max measured relerr", markdown)
        self.assertIn("not a measured value or fixture certification", markdown)
        self.assertNotIn("CRT exact f64 GEMM", markdown)
        self.assertNotIn("bit-exact", " ".join(result["claims_tested"]))

    def test_cuda_fallback_is_reported_as_not_dispatched(self):
        with tempfile.TemporaryDirectory() as directory:
            result, markdown = self.fixture(directory, "cuda", dispatched=False)
        self.assertTrue(result["throughput_gflops_by_n"]["ozaki"][0]["not_dispatched"])
        self.assertIsNone(result["throughput_gflops_by_n"]["ozaki"][0]["gflops_median"])
        self.assertIn("ozaki", result["accuracy_not_dispatched"])
        self.assertIsNone(result["accuracy_vs_exact_rational_reference"]["ozaki"])
        self.assertIn("not dispatched (cublasDgemm)", markdown)

    def test_metal_labels_remain_and_unavailable_rows_are_not_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            result, markdown = self.fixture(directory, "metal", available=False)
        self.assertEqual(result["gpu_backend"], "metal")
        self.assertIsNone(result["throughput_gflops_by_n"]["ozaki"])
        self.assertIsNone(result["accuracy_vs_exact_rational_reference"]["ozaki"])
        self.assertIn("Ozaki-II exact", markdown)
        self.assertIn("Ozaki-II fast", markdown)
        self.assertIn("unavailable", markdown)


if __name__ == "__main__":
    unittest.main()
