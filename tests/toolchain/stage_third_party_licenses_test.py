#!/usr/bin/env python3
"""FetchContent source resolution controls for release license staging."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import stage_third_party_licenses as licenses  # noqa: E402


class FetchContentSourceResolutionTest(unittest.TestCase):
    def test_configured_override_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eshkol-license-override-") as temp:
            root = Path(temp)
            build = root / "build"
            build.mkdir()
            override = root / "configured-source"
            override.mkdir()
            fallback = build / "_deps/eshkol_pcre2-src"
            fallback.mkdir(parents=True)
            (build / "CMakeCache.txt").write_text(
                f"FETCHCONTENT_SOURCE_DIR_ESHKOL_PCRE2:PATH={override}\n"
                f"FETCHCONTENT_BASE_DIR:PATH={root / 'other-content'}\n",
                encoding="utf-8",
            )
            self.assertEqual(
                licenses.fetchcontent_source_dir(build, "ESHKOL_PCRE2"), override
            )

    def test_default_source_is_used_without_override(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eshkol-license-default-") as temp:
            build = Path(temp) / "build"
            source = build / "_deps/eshkol_pcre2-src"
            source.mkdir(parents=True)
            (build / "CMakeCache.txt").write_text(
                "FETCHCONTENT_FULLY_DISCONNECTED:BOOL=ON\n", encoding="utf-8"
            )
            self.assertEqual(
                licenses.fetchcontent_source_dir(build, "ESHKOL_PCRE2"), source
            )

    def test_configured_fetchcontent_base_dir_is_used(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eshkol-license-base-") as temp:
            root = Path(temp)
            build = root / "build"
            build.mkdir()
            custom_base = root / "configured-content"
            source = custom_base / "eshkol_pcre2-src"
            source.mkdir(parents=True)
            (build / "CMakeCache.txt").write_text(
                f"FETCHCONTENT_BASE_DIR:PATH={custom_base}\n", encoding="utf-8"
            )
            self.assertEqual(
                licenses.fetchcontent_source_dir(build, "ESHKOL_PCRE2"), source
            )

    def test_missing_override_does_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eshkol-license-missing-") as temp:
            root = Path(temp)
            build = root / "build"
            build.mkdir()
            fallback = build / "_deps/eshkol_pcre2-src"
            fallback.mkdir(parents=True)
            missing_override = root / "missing-configured-source"
            (build / "CMakeCache.txt").write_text(
                f"FETCHCONTENT_SOURCE_DIR_ESHKOL_PCRE2:PATH={missing_override}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "configured FETCHCONTENT_SOURCE_DIR"):
                licenses.fetchcontent_source_dir(build, "ESHKOL_PCRE2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
