#!/usr/bin/env python3
"""Verify that a staged Eshkol release package is self-contained and runnable.

The release archives must support the default ``eshkol-run -r`` persistent
run-cache path without consulting an older system-wide Eshkol installation.
This verifier deliberately performs both a cold cache build and a warm cache
execution, and rejects the otherwise-successful in-process JIT fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import NoReturn


DEFAULT_TIMEOUT_SECONDS = 900
EXPECTED_SMOKE_OUTPUT = "hello from windows smoke"
EXPECTED_AGENT_SMOKE_OUTPUT = "release agent smoke"
COMMON_LICENSE_FILES = (
    "pcre2-LICENCE.md",
    "pcre2-sljit-LICENSE.txt",
    "sqlite-PUBLIC-DOMAIN.txt",
    "zlib-LICENSE.txt",
    "tree-sitter-LICENSE.txt",
    "tree-sitter-unicode-LICENSE.txt",
    "tree-sitter-javascript-LICENSE.txt",
    "tree-sitter-typescript-LICENSE.txt",
    "tree-sitter-python-LICENSE.txt",
    "tree-sitter-rust-LICENSE.txt",
    "tree-sitter-go-LICENSE.txt",
    "tree-sitter-c-LICENSE.txt",
    "tree-sitter-cpp-LICENSE.txt",
    "tree-sitter-java-LICENSE.txt",
    "tree-sitter-ruby-LICENSE.txt",
    "tree-sitter-bash-LICENSE.txt",
    "yoga-LICENSE.txt",
)
LINUX_CODEC_LICENSE_FILES = (
    "linux-libpng-copyright.txt",
    "linux-libjpeg-copyright.txt",
    "linux-libwebp-copyright.txt",
    "linux-zlib-copyright.txt",
)
LINUX_RUNTIME_SUBDIR = Path("lib/eshkol/runtime-deps")
LINUX_REQUIRED_CODEC_ALIASES = (
    "libpng.so",
    "libpng16.so",
    "libjpeg.so",
    "libwebp.so",
    "libz.so",
)


def fail(message: str, *, stdout: str = "", stderr: str = "") -> NoReturn:
    print(f"FAIL: {message}", file=sys.stderr)
    if stdout:
        print("--- stdout ---", file=sys.stderr)
        print(stdout, file=sys.stderr, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print("--- stderr ---", file=sys.stderr)
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
    raise SystemExit(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    description: str,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        fail(
            f"{description} timed out after {timeout}s",
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )
    if result.returncode != 0:
        fail(
            f"{description} exited with status {result.returncode}",
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return result


def verify_linux_runtime_dependencies(
    package_dir: Path,
    binaries: tuple[Path, ...],
    env: dict[str, str],
) -> None:
    dependency_dir = package_dir / LINUX_RUNTIME_SUBDIR
    if dependency_dir.is_symlink() or not dependency_dir.is_dir():
        fail(f"packaged Linux runtime dependency directory is missing or symlinked: {dependency_dir}")
    manifest_path = dependency_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        fail(f"packaged Linux runtime dependency manifest is missing or symlinked: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"packaged Linux runtime dependency manifest is invalid: {exc}")
    if manifest.get("schema_version") != 1:
        fail("packaged Linux runtime dependency manifest has unsupported schema")
    if manifest.get("runtime_subdir") != LINUX_RUNTIME_SUBDIR.as_posix():
        fail("packaged Linux runtime dependency manifest records the wrong directory")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        fail("packaged Linux runtime dependency manifest has no files")
    manifest_licenses = manifest.get("licenses")
    if not isinstance(manifest_licenses, dict) or not manifest_licenses:
        fail("packaged Linux runtime dependency manifest has no licenses")

    for name, metadata in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            fail(f"unsafe runtime dependency manifest filename: {name!r}")
        path = dependency_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            fail(f"runtime dependency is missing, empty, or symlinked: {path}")
        if not isinstance(metadata, dict) or metadata.get("sha256") != sha256(path):
            fail(f"runtime dependency checksum mismatch: {path}")
        if not metadata.get("source_package") or not metadata.get("source_version"):
            fail(f"runtime dependency lacks package provenance: {path}")
        dynamic = run_checked(
            ["readelf", "-d", str(path)],
            cwd=package_dir,
            env=env,
            timeout=60,
            description=f"RUNPATH inspection for {path.name}",
        )
        if "$ORIGIN" not in dynamic.stdout:
            fail(f"runtime dependency lacks a relative $ORIGIN RUNPATH: {path}")

    licenses_dir = package_dir / "licenses"
    for name, metadata in manifest_licenses.items():
        if not isinstance(name, str) or Path(name).name != name:
            fail(f"unsafe runtime dependency license filename: {name!r}")
        path = licenses_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            fail(f"runtime dependency license is missing, empty, or symlinked: {path}")
        if not isinstance(metadata, dict) or metadata.get("sha256") != sha256(path):
            fail(f"runtime dependency license checksum mismatch: {path}")
        if not metadata.get("source_package") or not metadata.get("source_version"):
            fail(f"runtime dependency license lacks package provenance: {path}")

    for alias in LINUX_REQUIRED_CODEC_ALIASES:
        if alias not in files:
            fail(f"required Linux image-codec link alias is absent: {alias}")

    codec_pattern = re.compile(r"^(libpng(?:16)?|libjpeg|libwebp|libsharpyuv|libz)\.so")
    required_families = {"libpng", "libjpeg", "libwebp", "libz"}
    dependency_root = dependency_dir.resolve()
    for binary in binaries:
        result = run_checked(
            ["ldd", str(binary)],
            cwd=package_dir,
            env=env,
            timeout=60,
            description=f"shared-library resolution for {binary.name}",
        )
        resolved_families: set[str] = set()
        for raw_line in result.stdout.splitlines():
            match = re.match(
                r"^\s*(\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-fA-F]+\)\s*$",
                raw_line,
            )
            if not match:
                continue
            soname, resolved_text = match.groups()
            family_match = codec_pattern.match(soname)
            if not family_match:
                continue
            family = family_match.group(1)
            if family.startswith("libpng"):
                family = "libpng"
            resolved = Path(resolved_text).resolve()
            if not resolved.is_relative_to(dependency_root):
                fail(
                    f"{binary.name} resolves {soname} outside its package: {resolved}",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            resolved_families.add(family)
        missing = sorted(required_families - resolved_families)
        if missing:
            fail(
                f"{binary.name} does not resolve the complete packaged codec closure: "
                f"{', '.join(missing)}",
                stdout=result.stdout,
                stderr=result.stderr,
            )


def verify_cache_run(
    runner: Path,
    smoke_source: Path,
    package_dir: Path,
    env: dict[str, str],
    timeout: int,
    expected_trace: str,
    expected_output: str,
) -> None:
    result = run_checked(
        [str(runner), "-r", str(smoke_source)],
        cwd=package_dir,
        env=env,
        timeout=timeout,
        description=f"package smoke ({expected_trace.strip()})",
    )
    if expected_output not in result.stdout:
        fail(
            f"package smoke did not print {expected_output!r}",
            stdout=result.stdout,
            stderr=result.stderr,
        )
    forbidden = (
        "[jit-cache] compile-failed",
        "[jit-cache] compile-timeout",
        "[jit-cache] store-failed",
    )
    if any(marker in result.stderr for marker in forbidden):
        fail(
            "persistent run-cache failed and silently fell back to in-process JIT",
            stdout=result.stdout,
            stderr=result.stderr,
        )
    if expected_trace not in result.stderr:
        fail(
            f"package smoke did not report required trace {expected_trace!r}",
            stdout=result.stdout,
            stderr=result.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--smoke-source", required=True, type=Path)
    parser.add_argument("--agent-smoke-source", required=True, type=Path)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    smoke_source = args.smoke_source.resolve()
    agent_smoke_source = args.agent_smoke_source.resolve()
    if not package_dir.is_dir():
        fail(f"package directory does not exist: {package_dir}")
    if not smoke_source.is_file():
        fail(f"smoke source does not exist: {smoke_source}")
    if not agent_smoke_source.is_file():
        fail(f"agent smoke source does not exist: {agent_smoke_source}")
    if args.timeout_seconds <= 0:
        fail("--timeout-seconds must be positive")

    windows = os.name == "nt"
    runner = package_dir / "bin" / ("eshkol-run.exe" if windows else "eshkol-run")
    repl = package_dir / "bin" / ("eshkol-repl.exe" if windows else "eshkol-repl")
    if windows:
        archive_names = (
            "eshkol-runtime.lib",
            "eshkol-agent-ffi.lib",
            "eshkol-agent-pcre2.lib",
            "eshkol-agent-sqlite3.lib",
            "eshkol-agent-zlib.lib",
            "eshkol-agent-tree-sitter-grammars.lib",
            "eshkol-agent-tree-sitter.lib",
            "eshkol-agent-yoga.lib",
        )
    else:
        archive_names = (
            "libeshkol-runtime.a",
            "libeshkol-agent-ffi.a",
            "eshkol-agent-pcre2.a",
            "eshkol-agent-sqlite3.a",
            "eshkol-agent-zlib.a",
            "eshkol-agent-tree-sitter-grammars.a",
            "eshkol-agent-tree-sitter.a",
            "eshkol-agent-yoga.a",
        )
        if platform.system() == "Linux":
            archive_names += ("eshkol-agent-curl.a",)

    if not runner.is_file():
        fail(f"packaged runner is missing: {runner}")
    if not repl.is_file():
        fail(f"packaged REPL is missing: {repl}")

    notice_file = package_dir / "THIRD_PARTY_NOTICES.md"
    if notice_file.is_symlink() or not notice_file.is_file() or notice_file.stat().st_size == 0:
        fail(f"third-party notice index is missing, empty, or symlinked: {notice_file}")
    license_names = list(COMMON_LICENSE_FILES)
    if platform.system() == "Linux":
        license_names.append("curl-COPYING.txt")
        license_names.extend(LINUX_CODEC_LICENSE_FILES)
    if windows:
        license_names.append("eigen-COPYING.MPL2.txt")
    licenses_dir = package_dir / "licenses"
    if licenses_dir.is_symlink() or not licenses_dir.is_dir():
        fail(f"third-party license directory is missing or symlinked: {licenses_dir}")
    for license_name in license_names:
        license_path = licenses_dir / license_name
        if license_path.is_symlink() or not license_path.is_file() or license_path.stat().st_size == 0:
            fail(f"third-party license is missing, empty, or symlinked: {license_path}")

    for archive_name in archive_names:
        archive_paths = (
            package_dir / "lib" / archive_name,
            package_dir / "lib" / "eshkol" / archive_name,
        )
        for archive_path in archive_paths:
            if not archive_path.is_file() or archive_path.stat().st_size == 0:
                fail(f"packaged archive is missing or empty: {archive_path}")
        if sha256(archive_paths[0]) != sha256(archive_paths[1]):
            fail(f"packaged archive copies differ: {archive_name}")

    base_env = os.environ.copy()
    if platform.system() == "Linux":
        verify_linux_runtime_dependencies(package_dir, (runner, repl), base_env)
    version = run_checked(
        [str(runner), "--version"],
        cwd=package_dir,
        env=base_env,
        timeout=60,
        description="packaged eshkol-run --version",
    )
    version_text = version.stdout + version.stderr
    if args.expected_version not in version_text:
        fail(
            f"packaged version does not contain {args.expected_version!r}",
            stdout=version.stdout,
            stderr=version.stderr,
        )

    with tempfile.TemporaryDirectory(prefix="eshkol-release-package-smoke-") as tmp:
        state_dir = Path(tmp)
        home_dir = state_dir / "home"
        cache_dir = state_dir / "jit-cache"
        home_dir.mkdir()

        env = base_env.copy()
        env["HOME"] = str(home_dir)
        env["XDG_CACHE_HOME"] = str(state_dir / "xdg-cache")
        env["LOCALAPPDATA"] = str(state_dir / "local-app-data")
        env["ESHKOL_JIT_CACHE_DIR"] = str(cache_dir)
        env["ESHKOL_JIT_CACHE_TRACE"] = "1"

        verify_cache_run(
            runner,
            smoke_source,
            package_dir,
            env,
            args.timeout_seconds,
            "[jit-cache] store ",
            EXPECTED_SMOKE_OUTPUT,
        )
        verify_cache_run(
            runner,
            smoke_source,
            package_dir,
            env,
            args.timeout_seconds,
            "[jit-cache] hit ",
            EXPECTED_SMOKE_OUTPUT,
        )
        verify_cache_run(
            runner,
            agent_smoke_source,
            package_dir,
            env,
            args.timeout_seconds,
            "[jit-cache] store ",
            EXPECTED_AGENT_SMOKE_OUTPUT,
        )
        verify_cache_run(
            runner,
            agent_smoke_source,
            package_dir,
            env,
            args.timeout_seconds,
            "[jit-cache] hit ",
            EXPECTED_AGENT_SMOKE_OUTPUT,
        )

    print(
        "PASS: release package is self-contained; core and agent cold stores and warm hits succeeded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
