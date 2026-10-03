#!/usr/bin/env python3
"""Exercise the production native image-I/O CMake module in small projects."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "cmake" / "EshkolImageIO.cmake"
SCRATCH = ROOT / ".scratch" / "v136-release-finalization-20261002" / "image-capability-contract"


class NativeImageIOContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        cls.task_dir = Path(tempfile.mkdtemp(prefix="cmake-tests-", dir=SCRATCH))
        cls.source = cls.task_dir / "fixture"
        cls.source.mkdir(parents=True)
        (cls.source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.14)\n"
            "project(ImageIOContractFixture LANGUAGES C)\n"
            "set(ESHKOL_EXTRA_LINK_LIBS \"\")\n"
            "if(FORCE_CODEC_BRANCH)\n"
            "  set(APPLE FALSE)\n"
            "  set(WIN32 FALSE)\n"
            "endif()\n"
            f'include("{MODULE.as_posix()}")\n',
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.task_dir, ignore_errors=True)

    def configure(self, name: str, *, required: bool, force_codec_branch: bool = False,
                  disable_png: bool = False) -> subprocess.CompletedProcess[str]:
        build = self.task_dir / name
        command = [
            "cmake", "-S", str(self.source), "-B", str(build),
            f"-DESHKOL_REQUIRE_IMAGE_IO={'ON' if required else 'OFF'}",
            f"-DFORCE_CODEC_BRANCH={'ON' if force_codec_branch else 'OFF'}",
        ]
        if disable_png:
            command.append("-DCMAKE_DISABLE_FIND_PACKAGE_PNG=ON")
        return subprocess.run(command, text=True, capture_output=True, timeout=30)

    def test_missing_png_is_optional_by_default_and_rejected_when_required(self):
        optional = self.configure("missing-png-optional", required=False,
                                  force_codec_branch=True, disable_png=True)
        optional_output = optional.stdout + optional.stderr
        self.assertEqual(optional.returncode, 0, optional_output)
        self.assertIn("Image I/O capability: NONE", optional_output)
        self.assertIn("ESHKOL_IMAGE_IO_BACKEND:INTERNAL=NONE",
                      (self.task_dir / "missing-png-optional" / "CMakeCache.txt").read_text())

        required = self.configure("missing-png-required", required=True,
                                  force_codec_branch=True, disable_png=True)
        required_output = required.stdout + required.stderr
        self.assertNotEqual(required.returncode, 0)
        self.assertIn(
            "ESHKOL_REQUIRE_IMAGE_IO=ON requires a native image I/O backend",
            required_output,
        )

    def test_available_host_backend_is_accepted_when_required(self):
        result = self.configure("available-host-backend", required=True)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertRegex(output, re.compile(r"Image I/O capability: (APPLE|GDIPLUS|LIBPNG)"))
        self.assertRegex((self.task_dir / "available-host-backend" / "CMakeCache.txt").read_text(),
                         r"ESHKOL_IMAGE_IO_BACKEND:INTERNAL=(APPLE|GDIPLUS|LIBPNG)")


if __name__ == "__main__":
    unittest.main()
