#!/usr/bin/env python3
"""Check the real CMake identity-to-CTest boundary without a product build."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile


def need(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--ctest", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    cmake_source = source / "CMakeLists.txt"
    template = source / "cmake/stdlib_identity.h.in"
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [cmake_source, template]}
    text = cmake_source.read_text()
    start = text.index("set(ESHKOL_STDLIB_RELEASE_STRING")
    end = text.index("@ONLY)", start) + len("@ONLY)")
    identity_block = text[start:end]
    start = text.index("if(ESHKOL_BUILD_TESTS AND TARGET eshkol-run AND NOT WIN32)")
    lines, depth = [], 0
    for line in text[start:].splitlines():
        lines.append(line)
        if line.strip().startswith("if("):
            depth += 1
        elif line.strip().startswith("endif("):
            depth -= 1
            if not depth:
                break
    need(depth == 0, "unbalanced identity registration block")
    registration = "\n".join(lines).replace("${CMAKE_CURRENT_SOURCE_DIR}", str(source))
    good = '--expected-release "${ESHKOL_STDLIB_RELEASE_STRING}"'
    bad = '--expected-release "${ESHKOL_STDLIB_IDENTITY_RELEASE}"'
    need(registration.count(good) == 1, "raw release argument is not uniquely registered")
    args.work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="identity-registration-", dir=args.work_root)).resolve()
    rows = []
    for name, block, expected in [
        ("correct", registration, True),
        ("normalized-control", registration.replace(good, bad), False),
    ]:
        project = work / name
        project.mkdir()
        (project / "cmake").mkdir()
        shutil.copy2(template, project / "cmake/stdlib_identity.h.in")
        prefix = """cmake_minimum_required(VERSION 3.20)
project(IdentityRegistrationContract NONE)
enable_testing()
set(ESHKOL_BUILD_TESTS ON)
set(ESHKOL_VERSION 1.4.0)
set(ESHKOL_STDLIB_IDENTITY_ABI object-abi-v2)
set(ESHKOL_STDLIB_SOURCE_FINGERPRINT aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)
set(ESHKOL_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
add_executable(eshkol-run IMPORTED)
set_target_properties(eshkol-run PROPERTIES IMPORTED_LOCATION "${CMAKE_COMMAND}")
set(ESHKOL_STDLIB_IDENTITY_LLVM_AS "${CMAKE_COMMAND}")
set(ESHKOL_STDLIB_IDENTITY_LLC "${CMAKE_COMMAND}")
"""
        prefix += 'set(Python3_EXECUTABLE "' + sys.executable + '")\n'
        (project / "CMakeLists.txt").write_text(prefix + identity_block + "\n" + block + "\n")
        build = project / "build"
        command = [args.cmake, "-S", str(project), "-B", str(build),
                   "-DCMAKE_MAKE_PROGRAM=" + (shutil.which("make") or "/usr/bin/make")]
        run = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (project / "configure.stdout").write_text(run.stdout)
        (project / "configure.stderr").write_text(run.stderr)
        need(run.returncode == 0, "metadata-only CMake configure failed: " + run.stderr[-1000:])
        listing = subprocess.run([args.ctest, "--test-dir", str(build), "--show-only=json-v1"],
                                 capture_output=True, text=True, timeout=30)
        (project / "CTEST-DECLARATIONS.json").write_text(listing.stdout)
        need(listing.returncode == 0, "CTest listing failed")
        tests = [t for t in json.loads(listing.stdout)["tests"]
                 if t["name"] == "stdlib_identity_cli_metadata"]
        need(len(tests) == 1, "metadata test is missing or duplicated")
        header = (build / "generated/eshkol/stdlib_identity_config.h").read_text()
        golden = {k.lower(): v for k, v in re.findall(
            r'^#define ESHKOL_STDLIB_IDENTITY_(RELEASE|ABI|SOURCE|SYMBOL) "([^"]*)"$',
            header, re.M)}
        need(len(golden) == 4 and golden["release"] == "1.4.0-evolve",
             "generated raw release/config does not exercise normalization")
        argv = tests[0]["command"]
        actual = {k: argv[argv.index("--expected-" + k) + 1] for k in golden}
        matches = actual == golden
        need(matches == expected, name + ": identity/CTest contract has the wrong verdict")
        rows.append({"case": name, "expected_match": expected, "actual_match": matches,
                     "generated_identity": golden, "CTest_expected_identity": actual})
    need(before == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before},
         "source changed during registration test")
    result = {"status": "PASS_GENERATED_IDENTITY_CTEST_ARGUMENT_CONTRACT",
              "cases": rows, "source_sha256": before, "product_builds": 0,
              "product_executions": 0, "control_reproduces_normalized_release_failure": True}
    (work / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(rows),
                      "output": str(work), "product_executions": 0}))


if __name__ == "__main__":
    main()
