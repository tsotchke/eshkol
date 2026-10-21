#!/usr/bin/env python3
"""Run public encrypted-PEM TLS rejection through JIT and AOT without a TTY."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile


PROMPT = re.compile(
    r"(?i)(enter.{0,40}(?:pass ?phrase|password)|"
    r"(?:pass ?phrase|password).{0,40}(?:enter|prompt))"
)
EXPECTED = "PASS: public TLS rejects encrypted keys without a prompt"


def run_open_stdin(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int) -> None:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        # Keep the pipe open through process exit. This reproduces the case
        # where OpenSSL would block waiting for an interactive PEM password.
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        stdout, stderr = process.communicate()
        raise RuntimeError(
            f"public TLS context constructor hung with stdin open: {' '.join(command)}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )

    stdout, stderr = process.communicate()
    combined = stdout + "\n" + stderr
    if process.returncode != 0:
        raise RuntimeError(
            f"public TLS test exited {process.returncode}: {' '.join(command)}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    if EXPECTED not in stdout:
        raise RuntimeError(f"public TLS test did not report both encrypted-key rejections:\n{combined}")
    if PROMPT.search(combined):
        raise RuntimeError(f"public TLS test emitted a password prompt:\n{combined}")


def run_checked(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                            text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(
            f"command exited {result.returncode}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--lib-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    args = parser.parse_args()
    runner = args.runner.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    library_dir = args.lib_dir.resolve(strict=True)
    source = source_root / "tests/network/tls_encrypted_key_public.esk"
    if not source.is_file():
        parser.error(f"test source missing: {source}")

    env = os.environ.copy()
    env["ESHKOL_PATH"] = str(source_root / "lib")
    env["ESHKOL_LIB_DIR"] = str(library_dir)
    env["ESHKOL_JIT_CACHE"] = "0"
    run_open_stdin([str(runner), "-r", str(source)], cwd=source_root,
                   env=env, timeout=args.timeout_seconds)
    print("PASS: public TLS encrypted-key JIT rejects nonempty and empty passwords without prompting")

    with tempfile.TemporaryDirectory(prefix="eshkol-tls-encrypted-aot-") as tmp:
        output = Path(tmp) / "tls_encrypted_key_public"
        run_checked([str(runner), "-O", "1", "-o", str(output), str(source),
                     "-L", str(library_dir)], cwd=source_root, env=env, timeout=90)
        run_open_stdin([str(output)], cwd=source_root, env=env,
                       timeout=args.timeout_seconds)
    print("PASS: public TLS encrypted-key AOT rejects nonempty and empty passwords without prompting")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
