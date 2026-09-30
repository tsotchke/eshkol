#!/usr/bin/env python3
"""Build and test the active feature integration on a provisioned Linux node."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FLAGS = {
    "ESHKOL_BUILD_TESTS": "ON",
    "ESHKOL_ALLOCATION_TESTING": "ON",
    "ESHKOL_ENABLE_EXPERIMENTAL_ESKM_V2": "ON",
    "ESHKOL_GPU_ENABLED": "OFF",
    "ESHKOL_XLA_ENABLED": "OFF",
    "ESHKOL_QUANTUM_ENABLED": "OFF",
}
TARGETS = [
    "eshkol-run", "stdlib", "eshkol-vm-standalone-test",
    "allocation-hardening-tests", "eskm_v2_preflight_test",
    "eskm_v2_preflight_cpp_test", "eskm_v2_backend_test",
    "eskm_v2_runtime_test", "eskm_v2_vm_allocation_test", "eskm_model_fuzz_probe",
]
TEST_PATTERN = (
    "^(runtime_allocation_hardening_test|constructor_allocation_aot|"
    "checked_constructor_ir_dominance|eskm_v2_.*)$"
)


def owned_path(raw):
    path = (ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if path == ROOT or ROOT not in path.parents:
        raise ValueError("qualification paths must be strictly inside this checkout")
    return path


def verify_cache(build):
    values = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.split(":", 1)[0]] = value
    if Path(values.get("CMAKE_HOME_DIRECTORY", "")).resolve() != ROOT:
        raise ValueError("CMake cache belongs to a different source checkout")
    for name, expected in FLAGS.items():
        if values.get(name) != expected:
            raise ValueError(f"{name} must be {expected}, got {values.get(name)!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("build", "test"), required=True)
    parser.add_argument("--build-dir", default="build-active-qualification")
    parser.add_argument("--work-dir", default=".scratch/active-qualification/results")
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("allocator fault qualification requires Linux GNU linker wrapping")
    build = owned_path(args.build_dir)
    work = owned_path(args.work_dir)
    if args.phase == "build":
        if shutil.disk_usage(ROOT).free < 35 * 2**30:
            raise RuntimeError("refusing a new build below the 35 GiB disk floor")
        llvm = os.environ.get("LLVM_CONFIG") or shutil.which("llvm-config-21")
        if not llvm:
            raise RuntimeError("provisioned LLVM21 llvm-config is required")
        version = subprocess.check_output([llvm, "--version"], text=True).strip()
        if version.split(".", 1)[0] != "21":
            raise RuntimeError(f"LLVM21 required, found {version}")
        command = ["cmake", "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
                   "-DCMAKE_BUILD_TYPE=Release", "-DESHKOL_REQUIRED_LLVM_MAJOR=21",
                   f"-DLLVM_CONFIG_EXECUTABLE={llvm}", "-DESHKOL_BUILD_AGENT_FFI=ON",
                   "-DESHKOL_PYTHON_BINDINGS=OFF", "-DESHKOL_ENABLE_FUZZ=ON"]
        command.extend(f"-D{name}={value}" for name, value in FLAGS.items())
        subprocess.run(command, check=True)
        verify_cache(build)
        subprocess.run(["cmake", "--build", str(build), "--parallel", str(args.jobs),
                        "--target", *TARGETS], check=True)
        return 0
    verify_cache(build)
    if work.exists() and any(work.iterdir()):
        raise RuntimeError("refusing to replace existing qualification results")
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ctest", "--test-dir", str(build), "--output-on-failure",
                    "--no-tests=error", "--output-junit", str(work / "focused-tests.xml"), "-R", TEST_PATTERN], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_eskm_model_fuzz.py"),
                    "--smoke", "--self-test", "--probe", str(build / "tests/fuzz/eskm_model_fuzz_probe"),
                    "--trace-file", str(work / "model-fuzz.jsonl")], check=True)
    subprocess.run(["bash", str(ROOT / "tests/limits/resource_limits_enforcement_gate.sh"),
                    str(build / "eshkol-run"), str(build / "eshkol-vm-standalone-test"),
                    str(work / "resource-limits")], check=True)
    print("PASS: active integration Linux allocation, ESKM and resource qualification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
