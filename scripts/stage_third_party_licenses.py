#!/usr/bin/env python3
"""Stage exact pinned dependency licenses into an Eshkol binary package."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTICE_FILE = ROOT / "THIRD_PARTY_NOTICES.md"


@dataclass(frozen=True)
class LicenseSpec:
    output_name: str
    source_relative: str


REQUIRED_LICENSES = (
    LicenseSpec("pcre2-LICENCE.md", "eshkol_pcre2-src/LICENCE.md"),
    LicenseSpec("pcre2-sljit-LICENSE.txt", "eshkol_pcre2-src/deps/sljit/LICENSE"),
    LicenseSpec("zlib-LICENSE.txt", "eshkol_zlib-src/LICENSE"),
    LicenseSpec("tree-sitter-LICENSE.txt", "eshkol_tree_sitter-src/LICENSE"),
    LicenseSpec(
        "tree-sitter-unicode-LICENSE.txt",
        "eshkol_tree_sitter-src/lib/src/unicode/LICENSE",
    ),
    LicenseSpec("tree-sitter-javascript-LICENSE.txt", "eshkol_ts_javascript-src/LICENSE"),
    LicenseSpec("tree-sitter-typescript-LICENSE.txt", "eshkol_ts_typescript-src/LICENSE"),
    LicenseSpec("tree-sitter-python-LICENSE.txt", "eshkol_ts_python-src/LICENSE"),
    LicenseSpec("tree-sitter-rust-LICENSE.txt", "eshkol_ts_rust-src/LICENSE"),
    LicenseSpec("tree-sitter-go-LICENSE.txt", "eshkol_ts_go-src/LICENSE"),
    LicenseSpec("tree-sitter-c-LICENSE.txt", "eshkol_ts_c-src/LICENSE"),
    LicenseSpec("tree-sitter-cpp-LICENSE.txt", "eshkol_ts_cpp-src/LICENSE"),
    LicenseSpec("tree-sitter-java-LICENSE.txt", "eshkol_ts_java-src/LICENSE"),
    LicenseSpec("tree-sitter-ruby-LICENSE.txt", "eshkol_ts_ruby-src/LICENSE"),
    LicenseSpec("tree-sitter-bash-LICENSE.txt", "eshkol_ts_bash-src/LICENSE"),
    LicenseSpec("yoga-LICENSE.txt", "eshkol_yoga-src/LICENSE"),
)

CURL_LICENSE = LicenseSpec("curl-COPYING.txt", "eshkol_curl-src/COPYING")

SQLITE_PUBLIC_DOMAIN_NOTICE = """SQLite public-domain notice

The author disclaims copyright to the SQLite source code. In place of a legal
notice, the amalgamation carries this blessing:

    May you do good and not evil.
    May you find forgiveness for yourself and forgive others.
    May you share freely, never taking more than you give.

SQLite's authors have dedicated the deliverable code and documentation to the
public domain. Canonical terms: https://www.sqlite.org/copyright.html
"""


def _regular_nonempty(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{description} is missing, empty, or symlinked: {path}")


def stage_licenses(
    build_dir: Path,
    package_dir: Path,
    *,
    notice_file: Path = NOTICE_FILE,
) -> list[Path]:
    build_dir = build_dir.resolve()
    package_dir = package_dir.resolve()
    deps_dir = build_dir / "_deps"
    _regular_nonempty(notice_file, "third-party notice index")
    if not deps_dir.is_dir() or deps_dir.is_symlink():
        raise ValueError(f"FetchContent dependency directory is unavailable: {deps_dir}")

    package_dir.mkdir(parents=True, exist_ok=True)
    licenses_dir = package_dir / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []

    notice_target = package_dir / "THIRD_PARTY_NOTICES.md"
    shutil.copyfile(notice_file, notice_target)
    staged.append(notice_target)

    for spec in REQUIRED_LICENSES:
        source = deps_dir / spec.source_relative
        _regular_nonempty(source, f"license for {spec.output_name}")
        target = licenses_dir / spec.output_name
        shutil.copyfile(source, target)
        staged.append(target)

    sqlite_target = licenses_dir / "sqlite-PUBLIC-DOMAIN.txt"
    sqlite_target.write_text(SQLITE_PUBLIC_DOMAIN_NOTICE, encoding="utf-8")
    staged.append(sqlite_target)

    curl_archive_names = ("eshkol-agent-curl.a", "eshkol-agent-curl.lib")
    package_has_curl = any((package_dir / "lib" / name).is_file() for name in curl_archive_names)
    if package_has_curl:
        source = deps_dir / CURL_LICENSE.source_relative
        _regular_nonempty(source, "curl license")
        target = licenses_dir / CURL_LICENSE.output_name
        shutil.copyfile(source, target)
        staged.append(target)

    for path in staged:
        _regular_nonempty(path, "staged license artifact")
    return staged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--notice-file", type=Path, default=NOTICE_FILE)
    args = parser.parse_args()
    try:
        staged = stage_licenses(
            args.build_dir,
            args.package_dir,
            notice_file=args.notice_file,
        )
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: staged {len(staged)} third-party notice/license files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
