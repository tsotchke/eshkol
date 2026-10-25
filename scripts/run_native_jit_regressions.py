#!/usr/bin/env python3
"""Compare forced in-process JIT with AOT controls for small native programs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import time


CASES = {
    "forward_ad": "12\n",
    "callcc_escape": "42\n",
    "guard_raise": "42\n",
    "parallel_map": "(1 4 9 16 25)\n",
    "parallel_continuation_escape": (
        "escaped\n(short 2)\nthunk\nfiltered\n(-1 2 -3 4 -5 6 -7 8)\n"
        "wound\n(after)\nparallel continuation escape PASS\n"
    ),
    "callcc_multishot_returned_frame": "1\n37\n73\ndone\n",
    "guard_handler_snapshot": "(caught 3 boom)\n" * 3,
}
CASE_SOURCE_PATHS = {
    "parallel_continuation_escape": "tests/vm_parity/corpus/parallel_continuation_escape.esk",
    "guard_handler_snapshot": "tests/continuations/guard_handler_snapshot.esk",
}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ERROR = re.compile(
    r"(?:^|\n)\s*(?:ERROR:|\[ERROR\]|Unhandled exception:|"
    r"\[Eshkol\] Unhandled Windows exception|LLVM module verification failed)",
    re.IGNORECASE,
)


def run_command(command, *, cwd, env, timeout):
    """Preserve failure evidence; never turn a crash or timeout into empty success."""
    started = time.monotonic()
    try:
        result = subprocess.run(
            [str(arg) for arg in command], cwd=cwd, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
        record = {
            "status": "completed", "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        def decode(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
        record = {
            "status": "timeout", "exit_code": None,
            "stdout": decode(exc.stdout), "stderr": decode(exc.stderr),
        }
    except OSError as exc:
        record = {
            "status": "launch_error", "exit_code": None,
            "stdout": "", "stderr": str(exc),
        }
    record["seconds"] = round(time.monotonic() - started, 3)
    record["command"] = [str(arg) for arg in command]
    return record


def transcript_passes(record, expected):
    # Text-mode subprocess reads normalize CRLF. Do not strip spaces, discard
    # banners, round numbers, or accept a PASS substring in unrelated output.
    return (
        record["status"] == "completed"
        and record["exit_code"] == 0
        and record["stdout"] == expected
        and ERROR.search(ANSI.sub("", record["stderr"])) is None
    )


def compilation_succeeded(record):
    return (record["status"] == "completed" and record["exit_code"] == 0
            and ERROR.search(ANSI.sub("", record["stderr"])) is None)


def classify_case(compile_result, aot_result, jit_result, expected):
    if not compilation_succeeded(compile_result) or aot_result is None:
        return "inconclusive_compile_failure"
    if not transcript_passes(aot_result, expected):
        return "inconclusive_aot_failure"
    if jit_result["status"] == "launch_error":
        return "inconclusive_jit_launch_failure"
    return "pass" if transcript_passes(jit_result, expected) else "jit_regression"


def sanitize(value, replacements):
    """Make the published JSON usable without embedding reviewer home paths."""
    if isinstance(value, str):
        for original, replacement in replacements:
            value = value.replace(original, replacement)
            value = value.replace(original.replace("\\", "/"), replacement)
        return value
    if isinstance(value, list):
        return [sanitize(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: sanitize(item, replacements) for key, item in value.items()}
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", default=os.environ.get("ESHKOL_RUN_OVERRIDE", "eshkol-run"),
                        help="Compiler executable (default: ESHKOL_RUN_OVERRIDE or eshkol-run on PATH)")
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="Parent for a new, retained evidence directory")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds per subprocess (default: 30)")
    parser.add_argument("--case", action="append", choices=CASES,
                        help="Run selected cases; default is all cases")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    resolved = shutil.which(args.runner)
    if resolved is None and Path(args.runner).is_file():
        resolved = args.runner
    if resolved is None:
        parser.error("runner not found; pass --runner or set ESHKOL_RUN_OVERRIDE")
    runner = Path(resolved).resolve()
    repository = Path(__file__).resolve().parents[1]
    fixture_dir = repository / "tests" / "toolchain" / "native_jit_regressions"
    work = args.work_dir.resolve() / f"run-{time.time_ns()}-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=False)
    for name in ("tmp", "jit-cache", "module-cache"):
        (work / name).mkdir()
    env = dict(os.environ, ESHKOL_JIT_CACHE="0",
               ESHKOL_JIT_CACHE_DIR=str(work / "jit-cache"),
               ESHKOL_AOT_MODULE_CACHE_DIR=str(work / "module-cache"),
               TMPDIR=str(work / "tmp"), TMP=str(work / "tmp"), TEMP=str(work / "tmp"))
    # All fixtures use compiler primitives and run --no-stdlib on both routes.
    # No installed stdlib object cache needs to be touched or trusted.
    results = []
    replacements = [(str(work), "<work>"), (str(repository), "<source>"),
                    (str(runner), "<eshkol-run>")]
    digest = hashlib.sha256()
    with runner.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "sha256": digest.hexdigest(),
        "version": run_command([runner, "--version"], cwd=work, env=env, timeout=args.timeout),
        "features": run_command([runner, "--features"], cwd=work, env=env, timeout=args.timeout),
    }
    for name in args.case or CASES:
        fixture = (repository / CASE_SOURCE_PATHS[name] if name in CASE_SOURCE_PATHS
                   else fixture_dir / f"{name}.esk")
        expected = CASES[name]
        binary = work / (name + (".exe" if os.name == "nt" else ".out"))
        compile_result = run_command(
            [runner, "--no-stdlib", fixture, "-o", binary], cwd=work, env=env, timeout=args.timeout)
        aot_result = None
        if compilation_succeeded(compile_result) and binary.is_file():
            aot_result = run_command([binary], cwd=work, env=env, timeout=args.timeout)
        jit_result = run_command(
            [runner, "--no-stdlib", "-r", fixture], cwd=work, env=env, timeout=args.timeout)
        result = {
            "case": name, "expected_stdout": expected,
            "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "status": classify_case(compile_result, aot_result, jit_result, expected),
            "aot_compile": compile_result, "aot_run": aot_result, "jit_run": jit_result,
        }
        results.append(result)
        report = {"schema": "eshkol.native-jit-regressions.v1",
                  "platform": sys.platform,
                  "machine": platform.machine() or sysconfig.get_platform(), "runner": metadata,
                  "jit_cache": "disabled", "stdlib": "disabled",
                  "timeout_seconds": args.timeout, "cases": results}
        (work / "results.json").write_text(
            json.dumps(sanitize(report, replacements), indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {result['status']}", flush=True)
    counts = {status: sum(case["status"] == status for case in results)
              for status in sorted({case["status"] for case in results})}
    print("Results: " + json.dumps(counts, sort_keys=True))
    print(f"Evidence: {work / 'results.json'}")
    if any(case["status"].startswith("inconclusive_") for case in results):
        return 2
    return 1 if any(case["status"] == "jit_regression" for case in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
