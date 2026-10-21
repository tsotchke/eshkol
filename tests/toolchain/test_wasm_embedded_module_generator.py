#!/usr/bin/env python3
"""Guard deterministic embedded-module generation and fail-closed WASM builds."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate_wasm_embedded_modules.py"
BUILD_SCRIPT = ROOT / "scripts/build-wasm-repl.sh"


class EmbeddedModuleGeneratorTest(unittest.TestCase):
    def test_check_detects_generated_header_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eshkol-embedded-modules-") as tmp:
            output = Path(tmp) / "vm_embedded_modules.h"
            generated = subprocess.run(
                [sys.executable, str(GENERATOR), "--root", str(ROOT), "--output", str(output)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)

            matches = subprocess.run(
                [sys.executable, str(GENERATOR), "--root", str(ROOT), "--output", str(output), "--check"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(matches.returncode, 0, matches.stderr)
            self.assertIn("matches 4 sources", matches.stdout)

            output.write_bytes(output.read_bytes() + b"/* intentional test drift */\n")
            drifted = subprocess.run(
                [sys.executable, str(GENERATOR), "--root", str(ROOT), "--output", str(output), "--check"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(drifted.returncode, 0)
            self.assertIn("generated header drift", drifted.stderr)

    def test_product_build_stops_when_generator_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eshkol-wasm-generator-failure-") as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "bin"
            build_dir = tmp_path / "build"
            fake_bin.mkdir()
            (build_dir / "generated/eshkol").mkdir(parents=True)
            (build_dir / "generated/eshkol/build_config.h").write_text("/* fixture */\n")

            python_stub = fake_bin / "python3"
            python_stub.write_text("#!/bin/sh\nexit 47\n")
            python_stub.chmod(0o755)
            emcc_marker = tmp_path / "emcc-was-called"
            emcc_stub = fake_bin / "emcc"
            emcc_stub.write_text("#!/bin/sh\nprintf called > \"$EMCC_MARKER\"\nexit 0\n")
            emcc_stub.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
                    "BUILD_DIR": str(build_dir),
                    "OUT": str(tmp_path / "product.js"),
                    "EMCC_MARKER": str(emcc_marker),
                }
            )
            result = subprocess.run(
                ["/bin/bash", str(BUILD_SCRIPT)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("failed to generate embedded module sources", result.stderr)
            self.assertFalse(emcc_marker.exists(), "emcc ran after embedded-source generation failed")


if __name__ == "__main__":
    unittest.main()
