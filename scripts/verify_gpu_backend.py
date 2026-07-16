#!/usr/bin/env python3
"""Verify that a GPU-labeled build graph contains the real requested backend."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def read_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_with_type, value = line.split("=", 1)
        key = key_with_type.split(":", 1)[0]
        values[key] = value
    return values


def build_graph_text(build_dir: Path) -> str:
    candidates = [build_dir / "build.ninja"]
    candidates.extend(build_dir.rglob("*.vcxproj"))
    texts: list[str] = []
    for candidate in candidates:
        if candidate.is_file():
            texts.append(candidate.read_text(encoding="utf-8", errors="replace"))
    if not texts:
        raise ValueError("no Ninja or Visual Studio build graph found")
    return "\n".join(texts).replace("\\", "/")


def verify(build_dir: Path, expected: str) -> list[str]:
    failures: list[str] = []
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        return [f"CMake cache not found: {cache_path}"]

    cache = read_cache(cache_path)
    expected = expected.upper()
    checks = {
        "ESHKOL_GPU_ENABLED": "ON",
        "ESHKOL_REQUIRE_GPU_BACKEND": "ON",
        "ESHKOL_GPU_BACKEND": expected,
    }
    for key, wanted in checks.items():
        actual = cache.get(key)
        if actual != wanted:
            failures.append(f"{key} is {actual!r}, expected {wanted!r}")

    if expected == "CUDA":
        compiler = cache.get("CMAKE_CUDA_COMPILER", "")
        if not compiler or compiler.endswith("-NOTFOUND"):
            failures.append("CMAKE_CUDA_COMPILER is absent or unresolved")
        architectures = {
            item.removesuffix("-real").removesuffix("-virtual")
            for item in cache.get("CMAKE_CUDA_ARCHITECTURES", "").split(";")
            if item
        }
        for required_arch in ("72", "86"):
            if required_arch not in architectures:
                failures.append(
                    f"portable CUDA architecture {required_arch} is missing from "
                    "CMAKE_CUDA_ARCHITECTURES"
                )
        try:
            graph = build_graph_text(build_dir)
        except ValueError as error:
            failures.append(str(error))
        else:
            for required in ("gpu_memory_cuda.cpp", "gpu_cuda_kernels.cu"):
                if required not in graph:
                    failures.append(f"real CUDA source missing from build graph: {required}")
            if "gpu_memory_stub.cpp" in graph:
                failures.append("CPU GPU stub is present in a CUDA-labeled build graph")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--expected", choices=("CUDA", "METAL"), required=True)
    args = parser.parse_args()

    failures = verify(args.build_dir.resolve(), args.expected)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"PASS: build graph contains the real {args.expected} GPU backend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
