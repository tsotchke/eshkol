#!/usr/bin/env python3
"""Compile the current stdlib with the release compiler into private outputs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def cache_value(cache: Path, name: str) -> str:
    prefix = name + ":"
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix) and "=" in line:
            return line.split("=", 1)[1]
    raise RuntimeError(f"{name} is absent from {cache}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    build_dir = args.build_dir if args.build_dir.is_absolute() else repo_root / args.build_dir
    build_dir = build_dir.resolve()
    compiler = build_dir / "eshkol-run"
    cache = build_dir / "CMakeCache.txt"
    source = repo_root / "lib" / "stdlib.esk"
    launcher = repo_root / "scripts" / "lib" / "stack_limit_exec.sh"
    for required in (compiler, cache, source, launcher):
        if not required.exists():
            raise SystemExit(f"stdlib isolated compile: required path missing: {required}")
    if not os.access(compiler, os.X_OK):
        raise SystemExit(f"stdlib isolated compile: compiler is not executable: {compiler}")

    try:
        target_cpu = cache_value(cache, "ESHKOL_STDLIB_TARGET_CPU")
        target_features = cache_value(cache, "ESHKOL_STDLIB_TARGET_FEATURES")
    except (OSError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    env = os.environ.copy()
    env["ESHKOL_PATH"] = str(repo_root / "lib")
    if target_cpu:
        env["ESHKOL_TARGET_CPU"] = target_cpu
        env["ESHKOL_TARGET_FEATURES"] = target_features
    elif target_features:
        env["ESHKOL_TARGET_FEATURES"] = target_features

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="icc-stdlib-rebuild-", dir=output_root) as temp:
        output_prefix = Path(temp) / "stdlib"
        command = [
            shutil.which("sh") or "/bin/sh",
            str(launcher),
            str(compiler),
            "--shared-lib",
            "-c",
            "-o",
            str(output_prefix),
            str(source),
        ]
        result = subprocess.run(command, cwd=build_dir, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            raise SystemExit(f"stdlib isolated compile failed with exit {result.returncode}")
        outputs = [output_prefix.with_suffix(".o"), output_prefix.with_suffix(".bc")]
        missing = [str(path) for path in outputs if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise SystemExit("stdlib isolated compile did not produce non-empty outputs: " + ", ".join(missing))
        print("stdlib isolated compile: PASS (fresh private .o/.bc outputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
