"""CPU-only validators for generated Eshkol manifests and trace artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import check_depth_coverage
from scripts import gen_ad_depth
from scripts import gen_nesting_depth
from scripts import gen_numeric_depth
from scripts import gen_recursion_depth
from scripts import run_generative_differential
from scripts import stage_third_party_licenses


class GeneratedArtifactValidatorTest(unittest.TestCase):
    def test_third_party_license_staging_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "build"
            package_dir = root / "package"
            notice = root / "THIRD_PARTY_NOTICES.md"
            notice.write_text("notices\n")
            for spec in stage_third_party_licenses.REQUIRED_LICENSES:
                source = build_dir / "_deps" / spec.source_relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(f"license: {spec.output_name}\n")
            curl_source = (
                build_dir / "_deps" /
                stage_third_party_licenses.CURL_LICENSE.source_relative
            )
            curl_source.parent.mkdir(parents=True, exist_ok=True)
            curl_source.write_text("curl license\n")
            eigen_source = (
                build_dir / "_deps" /
                stage_third_party_licenses.WINDOWS_EIGEN_LICENSE.source_relative
            )
            eigen_source.parent.mkdir(parents=True, exist_ok=True)
            eigen_source.write_text("Eigen MPL-2.0 license\n")
            package_lib = package_dir / "lib"
            package_lib.mkdir(parents=True)
            (package_lib / "eshkol-agent-curl.a").write_bytes(b"archive")
            staged = stage_third_party_licenses.stage_licenses(
                build_dir, package_dir, notice_file=notice,
            )
            self.assertEqual(
                len(staged),
                len(stage_third_party_licenses.REQUIRED_LICENSES) + 4,
            )
            self.assertTrue((package_dir / "THIRD_PARTY_NOTICES.md").is_file())
            self.assertTrue((package_dir / "licenses" / "sqlite-PUBLIC-DOMAIN.txt").is_file())
            self.assertTrue((package_dir / "licenses" / "curl-COPYING.txt").is_file())
            self.assertTrue(
                (package_dir / "licenses" / "eigen-COPYING.MPL2.txt").is_file()
            )

    def test_third_party_license_staging_rejects_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "build" / "_deps").mkdir(parents=True)
            notice = root / "THIRD_PARTY_NOTICES.md"
            notice.write_text("notices\n")
            with self.assertRaisesRegex(ValueError, "license for"):
                stage_third_party_licenses.stage_licenses(
                    root / "build", root / "package", notice_file=notice,
                )

    def test_depth_coverage_trace_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(check_depth_coverage.write_depth_coverage_trace(
                tmp,
                passed=True,
                swept=7,
                composables=8,
                pct=87.5,
                gap_count=1,
                error_count=0,
            ))
            self.assertEqual(path.name, "depth_coverage.jsonl")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                [row["name"] for row in rows],
                ["depth_coverage_gate", "depth_coverage_pct"],
            )
            self.assertEqual(rows[0]["value"], "PASS")

    def test_ad_depth_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generator = gen_ad_depth.Gen(tmp, max_depth=1)
            generator.generate()
            manifest = Path(tmp) / "MANIFEST.txt"
            self.assertTrue(manifest.is_file())
            self.assertTrue(
                manifest.read_text().startswith("# depth-parametric AD oracle")
            )
            self.assertTrue(generator.files)

    def test_nesting_depth_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["gen_nesting_depth.py", "--outdir", tmp, "--depths", "1"]
            with mock.patch.object(sys, "argv", argv):
                gen_nesting_depth.main()
            manifest = Path(tmp) / "MANIFEST.txt"
            rows = [
                line for line in manifest.read_text().splitlines()
                if not line.startswith("#")
            ]
            self.assertEqual(len(rows), len(gen_nesting_depth.CONSTRUCT_ORDER))

    def test_numeric_depth_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["gen_numeric_depth.py", "--out-dir", tmp, "--scale", "0.1"]
            with mock.patch.object(sys, "argv", argv):
                gen_numeric_depth.main()
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
            self.assertEqual(manifest["scale"], 0.1)
            self.assertEqual(len(manifest["families"]), len(gen_numeric_depth.GENERATORS))

    def test_recursion_depth_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["gen_recursion_depth.py", "--outdir", tmp]
            with mock.patch.object(sys, "argv", argv):
                gen_recursion_depth.main()
            manifest = Path(tmp) / "MANIFEST.txt"
            self.assertTrue(manifest.is_file())
            text = manifest.read_text()
            self.assertTrue(all(kind in text for kind in gen_recursion_depth.KIND_ORDER))

    def test_divergence_artifact_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_generative_differential.OracleResult("jit")
            result.rc = 1
            result.out = "stdout\n"
            result.err = "stderr\n"
            harness = SimpleNamespace(diverg_dir=tmp)
            divergences = [{
                "kind": "VALUE_MISMATCH",
                "program": "case-1",
                "detail": "expected fixture mismatch",
            }]
            run_generative_differential.write_divergence_artifacts(
                harness, "numeric", "case-1", {"jit": result}, divergences,
            )
            record = Path(tmp) / "case-1" / "DIVERGENCES.txt"
            self.assertEqual(
                record.read_text(),
                "VALUE_MISMATCH :: case-1 :: expected fixture mismatch\n",
            )


if __name__ == "__main__":
    unittest.main()
