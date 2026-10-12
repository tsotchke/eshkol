#!/usr/bin/env python3
"""Run the bounded stdout/stderr contract smoke for an Eshkol REPL."""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys
from typing import Sequence


SMOKE_FORM = "(+ 1 2)\n"
EXPECTED_STDOUT = "3"
ACCEPTED_STDOUT = (EXPECTED_STDOUT, EXPECTED_STDOUT + "\n")


class ReplSmokeError(RuntimeError):
    def __init__(self, message: str, *, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


def verify_repl_smoke(
    command: Sequence[str],
    *,
    timeout_seconds: float = 120.0,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Require one successful machine-mode evaluation, retaining failure output."""
    try:
        result = subprocess.run(
            [*command, "--machine"],
            input=SMOKE_FORM,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_output(exc.stdout)
        stderr = _decode_output(exc.stderr)
        raise ReplSmokeError(
            f"REPL machine smoke timed out after {timeout_seconds:g}s",
            stdout=stdout,
            stderr=stderr,
        ) from exc
    except OSError as exc:
        raise ReplSmokeError(f"could not launch REPL machine smoke: {exc}") from exc

    if result.returncode != 0:
        raise ReplSmokeError(
            f"REPL machine smoke exited with status {result.returncode}",
            stdout=result.stdout,
            stderr=result.stderr,
        )
    if result.stdout not in ACCEPTED_STDOUT:
        raise ReplSmokeError(
            f"REPL stdout was {result.stdout!r}, expected exactly {EXPECTED_STDOUT!r} "
            "with at most one final newline",
            stdout=result.stdout,
            stderr=result.stderr,
        )

    frames = [line.strip() for line in result.stderr.splitlines()]
    try:
        ready_index = frames.index("EREPL READY")
    except ValueError:
        raise ReplSmokeError(
            "REPL machine smoke did not emit EREPL READY",
            stdout=result.stdout,
            stderr=result.stderr,
        ) from None
    try:
        done_index = frames.index("EREPL DONE", ready_index + 1)
    except ValueError:
        raise ReplSmokeError(
            "REPL machine smoke did not emit EREPL DONE after READY",
            stdout=result.stdout,
            stderr=result.stderr,
        ) from None
    if any(line == "EREPL FAIL" for line in frames):
        raise ReplSmokeError(
            "REPL machine smoke emitted EREPL FAIL",
            stdout=result.stdout,
            stderr=result.stderr,
        )
    if done_index <= ready_index:
        raise ReplSmokeError(
            "REPL machine smoke emitted DONE before READY",
            stdout=result.stdout,
            stderr=result.stderr,
        )


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repl", required=True, help="path to eshkol-repl executable")
    parser.add_argument("--cwd", type=Path, help="working directory for the REPL process")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    try:
        verify_repl_smoke(
            [args.repl], cwd=args.cwd, timeout_seconds=args.timeout_seconds
        )
    except ReplSmokeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        if exc.stdout:
            print("--- stdout ---", file=sys.stderr)
            print(exc.stdout, file=sys.stderr, end="" if exc.stdout.endswith("\n") else "\n")
        if exc.stderr:
            print("--- stderr ---", file=sys.stderr)
            print(exc.stderr, file=sys.stderr, end="" if exc.stderr.endswith("\n") else "\n")
        return 1
    print("PASS: eshkol-repl --machine evaluates (+ 1 2) with clean stdout and READY/DONE framing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
