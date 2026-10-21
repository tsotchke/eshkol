#!/usr/bin/env python3
"""Stage exact pinned dependency licenses into an Eshkol binary package."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTICE_FILE = ROOT / "THIRD_PARTY_NOTICES.md"

FETCHCONTENT_SOURCE_NAMES = {
    "ESHKOL_PCRE2": "eshkol_pcre2-src",
    "ESHKOL_ZLIB": "eshkol_zlib-src",
    "ESHKOL_TREE_SITTER": "eshkol_tree_sitter-src",
    "ESHKOL_TS_JAVASCRIPT": "eshkol_ts_javascript-src",
    "ESHKOL_TS_TYPESCRIPT": "eshkol_ts_typescript-src",
    "ESHKOL_TS_PYTHON": "eshkol_ts_python-src",
    "ESHKOL_TS_RUST": "eshkol_ts_rust-src",
    "ESHKOL_TS_GO": "eshkol_ts_go-src",
    "ESHKOL_TS_C": "eshkol_ts_c-src",
    "ESHKOL_TS_CPP": "eshkol_ts_cpp-src",
    "ESHKOL_TS_JAVA": "eshkol_ts_java-src",
    "ESHKOL_TS_RUBY": "eshkol_ts_ruby-src",
    "ESHKOL_TS_BASH": "eshkol_ts_bash-src",
    "ESHKOL_YOGA": "eshkol_yoga-src",
    "ESHKOL_EIGEN": "eshkol_eigen-src",
    "ESHKOL_CURL": "eshkol_curl-src",
}


@dataclass(frozen=True)
class LicenseSpec:
    output_name: str
    dependency: str
    license_relative: str

    @property
    def source_relative(self) -> str:
        """Legacy `_deps`-relative location retained for existing callers/tests."""
        if not self.dependency:
            return self.license_relative
        return str(Path(FETCHCONTENT_SOURCE_NAMES[self.dependency]) /
                   self.license_relative)


REQUIRED_LICENSES = (
    LicenseSpec("pcre2-LICENCE.md", "ESHKOL_PCRE2", "LICENCE.md"),
    LicenseSpec("pcre2-sljit-LICENSE.txt", "ESHKOL_PCRE2", "deps/sljit/LICENSE"),
    LicenseSpec("zlib-LICENSE.txt", "ESHKOL_ZLIB", "LICENSE"),
    LicenseSpec("tree-sitter-LICENSE.txt", "ESHKOL_TREE_SITTER", "LICENSE"),
    LicenseSpec(
        "tree-sitter-unicode-LICENSE.txt",
        "ESHKOL_TREE_SITTER", "lib/src/unicode/LICENSE",
    ),
    LicenseSpec("tree-sitter-javascript-LICENSE.txt", "ESHKOL_TS_JAVASCRIPT", "LICENSE"),
    LicenseSpec("tree-sitter-typescript-LICENSE.txt", "ESHKOL_TS_TYPESCRIPT", "LICENSE"),
    LicenseSpec("tree-sitter-python-LICENSE.txt", "ESHKOL_TS_PYTHON", "LICENSE"),
    LicenseSpec("tree-sitter-rust-LICENSE.txt", "ESHKOL_TS_RUST", "LICENSE"),
    LicenseSpec("tree-sitter-go-LICENSE.txt", "ESHKOL_TS_GO", "LICENSE"),
    LicenseSpec("tree-sitter-c-LICENSE.txt", "ESHKOL_TS_C", "LICENSE"),
    LicenseSpec("tree-sitter-cpp-LICENSE.txt", "ESHKOL_TS_CPP", "LICENSE"),
    LicenseSpec("tree-sitter-java-LICENSE.txt", "ESHKOL_TS_JAVA", "LICENSE"),
    LicenseSpec("tree-sitter-ruby-LICENSE.txt", "ESHKOL_TS_RUBY", "LICENSE"),
    LicenseSpec("tree-sitter-bash-LICENSE.txt", "ESHKOL_TS_BASH", "LICENSE"),
    LicenseSpec("yoga-LICENSE.txt", "ESHKOL_YOGA", "LICENSE"),
)

WINDOWS_EIGEN_LICENSE = LicenseSpec(
    "eigen-COPYING.MPL2.txt", "ESHKOL_EIGEN", "COPYING.MPL2"
)

CURL_LICENSE = LicenseSpec("curl-COPYING.txt", "ESHKOL_CURL", "COPYING")

# The PJRT C API header is vendored directly in the source tree (deps/pjrt/),
# not fetched via CMake FetchContent, so its license lives next to it rather
# than under the build directory's _deps/ cache.
PJRT_LICENSE = LicenseSpec("pjrt-LICENSE.txt", "", "pjrt/LICENSE")

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


def _xla_enabled(build_dir: Path) -> bool:
    """Whether this build configured ESHKOL_XLA_ENABLED=ON (the *-xla release
    packages), read from the build directory's own CMakeCache.txt."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return False
    text = cache.read_text(encoding="utf-8", errors="replace")
    return re.search(r"^ESHKOL_XLA_ENABLED:BOOL=ON$", text, re.M) is not None


def _cmake_cache(build_dir: Path) -> dict[str, str]:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return {}
    values: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#") or ":" not in line or "=" not in line:
            continue
        key, typed_value = line.split(":", 1)
        _, value = typed_value.split("=", 1)
        values[key] = value
    return values


def fetchcontent_source_dir(build_dir: Path, dependency: str) -> Path:
    """Resolve one FetchContent source, honoring a configured override exactly."""
    cache_values = _cmake_cache(build_dir)
    cache_key = f"FETCHCONTENT_SOURCE_DIR_{dependency}"
    override = cache_values.get(cache_key, "")
    if override:
        source = Path(override)
        origin = f"configured {cache_key} override"
    else:
        source_name = FETCHCONTENT_SOURCE_NAMES.get(dependency)
        if source_name is None:
            raise ValueError(f"unsupported FetchContent dependency: {dependency}")
        base_dir = cache_values.get("FETCHCONTENT_BASE_DIR", "")
        source = (Path(base_dir) if base_dir else build_dir / "_deps") / source_name
        origin = "FetchContent default source directory"
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"{origin} is missing or invalid for {dependency}: {source}")
    return source


def stage_licenses(
    build_dir: Path,
    package_dir: Path,
    *,
    notice_file: Path = NOTICE_FILE,
) -> list[Path]:
    build_dir = build_dir.resolve()
    package_dir = package_dir.resolve()
    _regular_nonempty(notice_file, "third-party notice index")

    package_dir.mkdir(parents=True, exist_ok=True)
    licenses_dir = package_dir / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []

    notice_target = package_dir / "THIRD_PARTY_NOTICES.md"
    shutil.copyfile(notice_file, notice_target)
    staged.append(notice_target)

    for spec in REQUIRED_LICENSES:
        try:
            source_root = fetchcontent_source_dir(build_dir, spec.dependency)
        except ValueError as exc:
            raise ValueError(f"license for {spec.output_name}: {exc}") from exc
        source = source_root / spec.license_relative
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
        try:
            source_root = fetchcontent_source_dir(build_dir, CURL_LICENSE.dependency)
        except ValueError as exc:
            raise ValueError(f"license for {CURL_LICENSE.output_name}: {exc}") from exc
        source = source_root / CURL_LICENSE.license_relative
        _regular_nonempty(source, "curl license")
        target = licenses_dir / CURL_LICENSE.output_name
        shutil.copyfile(source, target)
        staged.append(target)

    # Native ClangCL/MSVC packages compile Eigen's vectorized kernels into the
    # runtime. Other platforms use their system BLAS and do not fetch Eigen.
    cache_values = _cmake_cache(build_dir)
    eigen_key = f"FETCHCONTENT_SOURCE_DIR_{WINDOWS_EIGEN_LICENSE.dependency}"
    base_dir = cache_values.get("FETCHCONTENT_BASE_DIR", "")
    eigen_default = (Path(base_dir) if base_dir else build_dir / "_deps") / "eshkol_eigen-src"
    eigen_configured = bool(cache_values.get(eigen_key, ""))
    if eigen_configured or eigen_default.is_dir():
        try:
            eigen_source = (fetchcontent_source_dir(build_dir, WINDOWS_EIGEN_LICENSE.dependency) /
                            WINDOWS_EIGEN_LICENSE.license_relative)
        except ValueError as exc:
            raise ValueError(f"license for {WINDOWS_EIGEN_LICENSE.output_name}: {exc}") from exc
    else:
        eigen_source = None
    if eigen_source is not None and eigen_source.is_file():
        _regular_nonempty(eigen_source, "Eigen MPL-2.0 license")
        target = licenses_dir / WINDOWS_EIGEN_LICENSE.output_name
        shutil.copyfile(eigen_source, target)
        staged.append(target)
    elif eigen_configured:
        _regular_nonempty(eigen_source, "Eigen MPL-2.0 license")

    # *-xla packages compile the vendored Apache-2.0 PJRT C API header
    # (deps/pjrt/pjrt_c_api.h) into the runtime; other packages never include
    # an XLA backend and do not need this notice.
    if _xla_enabled(build_dir):
        pjrt_source = ROOT / "deps" / PJRT_LICENSE.source_relative
        _regular_nonempty(pjrt_source, "PJRT Apache-2.0 license")
        target = licenses_dir / PJRT_LICENSE.output_name
        shutil.copyfile(pjrt_source, target)
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
