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
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import NoReturn


DEFAULT_TIMEOUT_SECONDS = 900
EXPECTED_SMOKE_OUTPUT = "hello from windows smoke"


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


def verify_cache_run(
    runner: Path,
    smoke_source: Path,
    package_dir: Path,
    env: dict[str, str],
    timeout: int,
    expected_trace: str,
) -> None:
    result = run_checked(
        [str(runner), "-r", str(smoke_source)],
        cwd=package_dir,
        env=env,
        timeout=timeout,
        description=f"package smoke ({expected_trace.strip()})",
    )
    if EXPECTED_SMOKE_OUTPUT not in result.stdout:
        fail(
            f"package smoke did not print {EXPECTED_SMOKE_OUTPUT!r}",
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
    parser.add_argument("--expected-version", required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    smoke_source = args.smoke_source.resolve()
    if not package_dir.is_dir():
        fail(f"package directory does not exist: {package_dir}")
    if not smoke_source.is_file():
        fail(f"smoke source does not exist: {smoke_source}")
    if args.timeout_seconds <= 0:
        fail("--timeout-seconds must be positive")

    windows = os.name == "nt"
    runner = package_dir / "bin" / ("eshkol-run.exe" if windows else "eshkol-run")
    runtime_name = "eshkol-runtime.lib" if windows else "libeshkol-runtime.a"
    runtime_paths = (
        package_dir / "lib" / runtime_name,
        package_dir / "lib" / "eshkol" / runtime_name,
    )

    if not runner.is_file():
        fail(f"packaged runner is missing: {runner}")
    for runtime_path in runtime_paths:
        if not runtime_path.is_file() or runtime_path.stat().st_size == 0:
            fail(f"packaged runtime archive is missing or empty: {runtime_path}")
    if sha256(runtime_paths[0]) != sha256(runtime_paths[1]):
        fail("packaged runtime archive copies differ")

    base_env = os.environ.copy()
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
        )
        verify_cache_run(
            runner,
            smoke_source,
            package_dir,
            env,
            args.timeout_seconds,
            "[jit-cache] hit ",
        )

    print(
        "PASS: release package is self-contained; cold run-cache store and warm hit succeeded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
