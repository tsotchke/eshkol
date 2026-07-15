#!/usr/bin/env python3
"""Reject release stdlib bitcode that inherits builder-only CPU features."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Iterable


CPU_CACHE_KEY = "ESHKOL_STDLIB_TARGET_CPU"
FEATURES_CACHE_KEY = "ESHKOL_STDLIB_TARGET_FEATURES"

# Eshkol exposes no scalable-vector source primitive.  These constructs can
# therefore only come from LLVM optimizing the stdlib for the build runner's
# SVE/SVE2 CPU.  Likewise, target attributes for optional wide-vector ISA
# families would make an otherwise portable bitcode/object pair host-specific.
FORBIDDEN_IR_PATTERNS = (
    (re.compile(r"@llvm\.vscale(?:\.|\()"), "LLVM scalable-vector intrinsic"),
    (re.compile(r"<vscale\s+x\s+"), "LLVM scalable-vector type"),
    (
        re.compile(
            r'"target-features"="[^"]*\+(?:sve|sme|avx|amx)[^"]*"',
            re.IGNORECASE,
        ),
        "optional wide-vector target feature",
    ),
)


def parse_cmake_cache(cache_path: Path) -> dict[str, str]:
    """Return non-comment CMake cache entries without interpreting escapes."""
    entries: dict[str, str] = {}
    for raw_line in cache_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        entries[key] = value
    return entries


def validate_cmake_cache(cache_path: Path, expected_cpu: str) -> None:
    entries = parse_cmake_cache(cache_path)
    actual_cpu = entries.get(CPU_CACHE_KEY)
    if actual_cpu != expected_cpu:
        raise ValueError(
            f"{CPU_CACHE_KEY} must be {expected_cpu!r}, found {actual_cpu!r}"
        )
    actual_features = entries.get(FEATURES_CACHE_KEY)
    if actual_features != "":
        raise ValueError(
            f"{FEATURES_CACHE_KEY} must be explicitly empty for the release "
            f"baseline, found {actual_features!r}"
        )


def scan_ir(lines: Iterable[str]) -> list[str]:
    failures: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        for pattern, label in FORBIDDEN_IR_PATTERNS:
            if pattern.search(line):
                failures.append(f"line {line_number}: {label}: {line.strip()}")
                break
    return failures


def disassemble_and_validate(bitcode: Path, llvm_dis: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="eshkol-stdlib-portability-") as tmp:
        ir_path = Path(tmp) / "stdlib.ll"
        result = subprocess.run(
            [str(llvm_dis), str(bitcode), "-o", str(ir_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"llvm-dis failed for {bitcode} (exit {result.returncode}): {detail}"
            )

        with ir_path.open("r", encoding="utf-8") as handle:
            failures = scan_ir(handle)
        if failures:
            preview = "\n".join(failures[:20])
            remainder = len(failures) - min(len(failures), 20)
            suffix = f"\n... and {remainder} more" if remainder else ""
            raise ValueError(
                "precompiled stdlib is not baseline-portable:\n"
                f"{preview}{suffix}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitcode", required=True, type=Path)
    parser.add_argument("--llvm-dis", required=True, type=Path)
    parser.add_argument("--cmake-cache", required=True, type=Path)
    parser.add_argument("--expected-cpu", default="generic")
    args = parser.parse_args()

    for label, path in (
        ("stdlib bitcode", args.bitcode),
        ("llvm-dis", args.llvm_dis),
        ("CMake cache", args.cmake_cache),
    ):
        if not path.is_file():
            parser.error(f"{label} not found: {path}")

    try:
        validate_cmake_cache(args.cmake_cache, args.expected_cpu)
        disassemble_and_validate(args.bitcode, args.llvm_dis)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        "PASS: precompiled stdlib uses the configured generic CPU and contains "
        "no builder-only scalable/wide-vector IR"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
