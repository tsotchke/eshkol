#!/usr/bin/env python3
"""Path-safety controls for the macOS runtime staging helper."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import stage_macos_openssl_runtime as stage  # noqa: E402


class StagePathSafetyTest(unittest.TestCase):
    def test_os_temp_alias_is_canonicalized(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eshkol-stage-alias-") as temp:
            requested = Path(temp) / "package"
            requested.mkdir()
            canonical = stage.canonical_package_root(requested)
            self.assertTrue(canonical.is_dir())
            self.assertFalse(canonical.is_symlink())

    def test_package_root_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eshkol-stage-root-link-") as temp:
            root = Path(temp)
            target = root / "target"
            target.mkdir()
            link = root / "package"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked package staging root"):
                stage.canonical_package_root(link)

    def test_descendant_symlinks_and_escapes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eshkol-stage-child-link-") as temp:
            root = Path(temp) / "package"
            root.mkdir()
            canonical = stage.canonical_package_root(root)
            target = Path(temp) / "outside"
            target.mkdir()
            link = canonical / "lib"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked staging path"):
                stage.require_no_symlink_descendants(canonical, link)
            with self.assertRaisesRegex(ValueError, "escapes package root"):
                stage.require_no_symlink_descendants(canonical, target)


if __name__ == "__main__":
    unittest.main(verbosity=2)
