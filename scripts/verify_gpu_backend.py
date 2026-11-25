#!/usr/bin/env python3
"""Verify that a GPU-labeled build graph contains the real requested backend."""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
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


_GPU_RUNTIME_TARGET = "eshkol-gpu-runtime-obj"
_CUDA_SOURCES = ("gpu_memory_cuda.cpp", "gpu_cuda_kernels.cu")


def ninja_gpu_runtime_sources(build_dir: Path) -> set[str] | None:
    """Return target source basenames, or None when its compile edge is absent."""
    sources: set[str] = set()
    found_target_edge = False
    target_output = re.compile(
        rf"(?:^|/)CMakeFiles/{re.escape(_GPU_RUNTIME_TARGET)}\.dir/"
    )
    # Single-config Ninja stores edges in build.ninja. Ninja Multi-Config
    # keeps them in CMakeFiles/impl-<Config>.ninja. In both cases scope source
    # checks to outputs belonging to eshkol-gpu-runtime-obj; recursively
    # scanning all graph text also sees unrelated generated targets/deps.
    for path in build_dir.rglob("*.ninja"):
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.replace("\\", "/")
            if not line.startswith("build "):
                continue
            outputs, separator, inputs = line[6:].partition(":")
            if not separator or not target_output.search(outputs):
                continue
            found_target_edge = True
            # Ninja's implicit/order-only prerequisites follow ` |` or ` ||`;
            # only explicit inputs are compiled sources for this edge.
            explicit_inputs = re.split(r"\s\|", inputs, maxsplit=1)[0]
            for source in (*_CUDA_SOURCES, "gpu_memory_stub.cpp"):
                if re.search(
                    rf"(?<![A-Za-z0-9_.-]){re.escape(source)}(?![A-Za-z0-9_.-])",
                    explicit_inputs,
                ):
                    sources.add(source)
    return sources if found_target_edge else None


def visual_studio_gpu_runtime_sources(build_dir: Path) -> set[str] | None:
    """Return target source basenames, or None when its project is absent."""
    project_name = f"{_GPU_RUNTIME_TARGET}.vcxproj".casefold()
    projects = [
        path for path in build_dir.rglob("*.vcxproj")
        if path.name.casefold() == project_name
    ]
    sources: set[str] = set()
    for path in projects:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as error:
            raise ValueError(f"cannot parse GPU runtime project {path}: {error}") from error
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            if tag not in {"ClCompile", "CudaCompile", "CustomBuild"}:
                continue
            include = element.attrib.get("Include", "").replace("\\", "/")
            for source in (*_CUDA_SOURCES, "gpu_memory_stub.cpp"):
                if include.endswith(source):
                    sources.add(source)
    return sources


def gpu_runtime_sources(build_dir: Path) -> set[str]:
    ninja_files = list(build_dir.rglob("*.ninja"))
    visual_studio_projects = list(build_dir.rglob("*.vcxproj"))
    if ninja_files:
        sources = ninja_gpu_runtime_sources(build_dir)
        if sources is not None:
            return sources
    if visual_studio_projects:
        sources = visual_studio_gpu_runtime_sources(build_dir)
        if sources is not None:
            return sources
    if not ninja_files and not visual_studio_projects:
        raise ValueError("no Ninja or Visual Studio build graph found")
    raise ValueError(
        f"{_GPU_RUNTIME_TARGET} source edges not found in the generated build graph"
    )


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
        try:
            cuda_major = int(cache.get("ESHKOL_HOST_CUDA_MAJOR", "0"))
        except ValueError:
            cuda_major = 0
            failures.append("ESHKOL_HOST_CUDA_MAJOR is not an integer")
        # The toolkit's own supported set, recorded by the configure
        # (cmake/EshkolCudaArchitectures.cmake): nvcc's architecture list,
        # or "RANGE;low;high" from the documented range of its version.
        # Without it (an older cache), the CUDA major decides.
        supported = [
            item for item in cache.get("ESHKOL_CUDA_SUPPORTED_ARCHITECTURES", "").split(";")
            if item
        ]

        def arch_supported(arch: str) -> bool:
            digits = "".join(ch for ch in arch if ch.isdigit()) or "0"
            number = int(digits)
            if supported and supported[0] == "RANGE" and len(supported) == 3:
                return int(supported[1]) <= number <= int(supported[2])
            if supported:
                return str(number) in supported
            return not (cuda_major >= 13 and number < 75)

        for arch in sorted(architectures):
            if arch[:1].isdigit() and not arch_supported(arch):
                failures.append(
                    f"CUDA {cuda_major} does not support portable architecture {arch}"
                )
        # The portable floor is the oldest default generation the toolkit
        # still compiles for: SM72 where supported, SM75 otherwise.
        floor = "72" if arch_supported("72") else "75"
        required_arches = (floor, "86")
        for required_arch in required_arches:
            if required_arch not in architectures:
                failures.append(
                    f"portable CUDA architecture {required_arch} is missing from "
                    "CMAKE_CUDA_ARCHITECTURES"
                )
        try:
            sources = gpu_runtime_sources(build_dir)
        except ValueError as error:
            failures.append(str(error))
        else:
            for required in _CUDA_SOURCES:
                if required not in sources:
                    failures.append(f"real CUDA source missing from build graph: {required}")
            if "gpu_memory_stub.cpp" in sources:
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
