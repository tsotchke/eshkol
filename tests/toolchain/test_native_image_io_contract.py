#!/usr/bin/env python3
"""Exercise the production image-I/O CMake module in small projects.

Image I/O uses codecs built into the runtime, so the module must resolve the
BUILTIN capability on every platform without searching for any codec
library, even when CMake is told the usual codec packages do not exist.
"""

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
SCRATCH = ROOT / ".scratch" / "image-capability-contract"


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
            f'include("{MODULE.as_posix()}")\n',
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.task_dir, ignore_errors=True)

    def configure(self, name: str, *, required: bool,
                  disable_codec_packages: bool = False) -> subprocess.CompletedProcess[str]:
        build = self.task_dir / name
        command = [
            "cmake", "-S", str(self.source), "-B", str(build),
            f"-DESHKOL_REQUIRE_IMAGE_IO={'ON' if required else 'OFF'}",
        ]
        if disable_codec_packages:
            command += [
                "-DCMAKE_DISABLE_FIND_PACKAGE_PNG=ON",
                "-DCMAKE_DISABLE_FIND_PACKAGE_JPEG=ON",
                "-DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=ON",
            ]
        return subprocess.run(command, text=True, capture_output=True, timeout=30)

    def test_module_searches_for_no_codec_library(self):
        text = MODULE.read_text(encoding="utf-8")
        code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        for probe in ("find_package", "find_library", "pkg_check_modules", "FindPkgConfig"):
            self.assertNotIn(probe, code, f"{MODULE.name} must not call {probe}")
        self.assertNotIn("ESHKOL_EXTRA_LINK_LIBS", code)

    def test_builtin_codecs_without_codec_packages(self):
        for required in (False, True):
            name = f"builtin-{'required' if required else 'optional'}"
            result = self.configure(name, required=required, disable_codec_packages=True)
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("Image I/O capability: BUILTIN", output)
            self.assertIn("no external codec library is used or linked", output)
            self.assertIn("ESHKOL_IMAGE_IO_BACKEND:INTERNAL=BUILTIN",
                          (self.task_dir / name / "CMakeCache.txt").read_text())


if __name__ == "__main__":
    unittest.main()
