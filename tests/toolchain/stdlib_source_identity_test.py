#!/usr/bin/env python3
"""Exercise the actual CMake source-material producer and independent checker."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def need(value, message):
    if not value:
        raise RuntimeError(message)


def run(argv, directory, name):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    (directory / (name + ".stdout")).write_text(result.stdout)
    (directory / (name + ".stderr")).write_text(result.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    start_time = time.monotonic()
    source = args.source_dir.resolve()
    checker = source / "scripts/check_stdlib_source_identity.py"
    collector = source / "cmake/StdlibIdentityInputs.cmake"
    cmake = (source / "CMakeLists.txt").read_text()
    begin = cmake.index("include(cmake/StdlibIdentityInputs.cmake)")
    end_marker = '     "${_stdlib_identity_source_material}")'
    end = cmake.index(end_marker, cmake.index("file(WRITE", begin)) + len(end_marker)
    producer = cmake[begin:end]
    args.work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="source-identity-", dir=args.work_root)).resolve()
    project = work / "baseline/source"
    project.mkdir(parents=True)
    for directory in ("inc/eshkol", "lib/backend", "lib/core", "lib/repl", "cmake",
                      "lib/math", "lib/signal", "lib/random", "lib/web", "lib/tensor",
                      "lib/quantum", "lib/ml", "exe"):
        d = project / directory
        d.mkdir(parents=True, exist_ok=True)
        (d / "nested").mkdir()
        (d / "nested/input.esk").write_text("; " + directory + "\n")
        (d / ".hidden.esk").write_text("; hidden\n")
        (d / "uppercase.ESK").write_text("; platform glob behavior\n")
        (d / "not-a-module.txt").write_text("scope-control\n")
    shutil.copy2(collector, project / "cmake/StdlibIdentityInputs.cmake")
    for name in ("exe/eshkol-run.cpp", "lib/stdlib.esk", "lib/math.esk"):
        (project / name).write_text("; literal " + name + "\n")
    (project / ".gitignore").write_text("/inc/eshkol/.ignored-source\n")
    prefix = '''cmake_minimum_required(VERSION 3.20)
project(SourceIdentityContract NONE)
set(CMAKE_C_COMPILER_ID FixtureC)
set(CMAKE_C_COMPILER_VERSION 1.2.3)
set(CMAKE_CXX_COMPILER_ID FixtureCXX)
set(CMAKE_CXX_COMPILER_VERSION 4.5.6)
set(ESHKOL_LLVM_VERSION 21.1.8)
unset(LLVM_PACKAGE_VERSION)
set(ESHKOL_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
'''
    (project / "CMakeLists.txt").write_text(prefix + producer + '''
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fingerprint.txt" "${ESHKOL_STDLIB_SOURCE_FINGERPRINT}")
''')
    baseline = work / "baseline"
    build = baseline / "build"
    configured = run([args.cmake, "-S", str(project), "-B", str(build)], baseline, "configure")
    need(configured.returncode == 0, "actual producer configure failed: " + configured.stderr[-1200:])
    golden = (build / "generated/eshkol/stdlib_identity_source.txt").read_bytes()
    fingerprint = (build / "fingerprint.txt").read_text()
    need(hashlib.sha256(golden).hexdigest() == fingerprint, "CMake export/hash parity failed")
    expected_toolchain = b"toolchain:c=FixtureC-1.2.3;cxx=FixtureCXX-4.5.6;llvm=21.1.8\n"
    need(expected_toolchain and golden.endswith(expected_toolchain),
         "canonical LLVM identity suffix is missing or incorrect")

    # A package-advertised version can be stale after the selected LLVM has
    # already been validated.  Configure a separate fixture to ensure it
    # cannot replace the canonical ESHKOL_LLVM_VERSION identity input.
    stale = work / "stale-package-variable"
    shutil.copytree(project, stale / "source")
    stale_cmake_path = stale / "source/CMakeLists.txt"
    stale_cmake = stale_cmake_path.read_text().replace(
        "unset(LLVM_PACKAGE_VERSION)", "set(LLVM_PACKAGE_VERSION 99.9.9)")
    need(stale_cmake != stale_cmake_path.read_text(),
         "stale package fixture was not injected")
    stale_cmake_path.write_text(stale_cmake)
    stale_configured = run([args.cmake, "-S", str(stale / "source"),
                            "-B", str(stale / "build")], stale, "configure")
    need(stale_configured.returncode == 0,
         "stale package-variable configure failed: " + stale_configured.stderr[-1200:])
    stale_golden = (stale / "build/generated/eshkol/stdlib_identity_source.txt").read_bytes()
    need(stale_golden.endswith(expected_toolchain),
         "stale LLVM package variable overrode canonical identity suffix")

    # Verify the refactor preserves the old collector's exact namespace, with
    # its native '*' and Scheme-only '*.esk' scopes and platform glob rules.
    old_collection = '''
file(GLOB_RECURSE old_native CONFIGURE_DEPENDS
 "${CMAKE_CURRENT_SOURCE_DIR}/inc/eshkol/*" "${CMAKE_CURRENT_SOURCE_DIR}/lib/backend/*"
 "${CMAKE_CURRENT_SOURCE_DIR}/lib/core/*" "${CMAKE_CURRENT_SOURCE_DIR}/lib/repl/*"
 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*")
file(GLOB_RECURSE old_scheme CONFIGURE_DEPENDS
 "${CMAKE_CURRENT_SOURCE_DIR}/lib/core/*.esk" "${CMAKE_CURRENT_SOURCE_DIR}/lib/math/*.esk"
 "${CMAKE_CURRENT_SOURCE_DIR}/lib/signal/*.esk" "${CMAKE_CURRENT_SOURCE_DIR}/lib/random/*.esk"
 "${CMAKE_CURRENT_SOURCE_DIR}/lib/web/*.esk" "${CMAKE_CURRENT_SOURCE_DIR}/lib/tensor/*.esk"
 "${CMAKE_CURRENT_SOURCE_DIR}/lib/quantum/*.esk" "${CMAKE_CURRENT_SOURCE_DIR}/lib/ml/*.esk"
 "${CMAKE_CURRENT_SOURCE_DIR}/lib/math.esk")
set(old_inputs ${old_native} ${old_scheme} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
 "${CMAKE_CURRENT_SOURCE_DIR}/exe/eshkol-run.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/lib/stdlib.esk")
list(REMOVE_DUPLICATES old_inputs)
list(SORT old_inputs)
if(NOT "${old_inputs}" STREQUAL "${_stdlib_identity_inputs}")
 message(FATAL_ERROR "collector namespace changed")
endif()
'''
    parity = work / "collector-parity"
    shutil.copytree(project, parity / "source")
    with (parity / "source/CMakeLists.txt").open("a") as stream:
        stream.write(old_collection)
    result = run([args.cmake, "-S", str(parity / "source"), "-B", str(parity / "build")], parity, "configure")
    need(result.returncode == 0, "legacy collector namespace parity failed")

    rows = []
    def check(name, expected, mutation=None, mutate_material=None, expected_fp=None,
              source_template=project, material_template=golden):
        case = work / name
        shutil.copytree(source_template, case / "source")
        material = case / "material.txt"
        material.write_bytes(material_template if mutate_material is None else mutate_material(material_template))
        if mutation:
            mutation(case / "source", case)
        pinned = hashlib.sha256(material.read_bytes()).hexdigest() if expected_fp == "rehash-control" else expected_fp or fingerprint
        argv = [sys.executable, str(checker), "--source-root", str(case / "source"),
                "--material", str(material), "--expected-source-fingerprint", pinned]
        result = run(argv, case, "check")
        parsed = json.loads(result.stdout)
        need(result.returncode == (0 if expected else 1) and parsed["status"] == ("PASS" if expected else "FAIL"),
             name + ": unexpected checker verdict: " + result.stdout + result.stderr)
        rows.append({"case": name, "status": parsed["status"], "expected_pass": expected,
                     "reason": parsed.get("reason")})
        return case

    check("unchanged", True)
    check("stale-package-admission", True, expected_fp="rehash-control",
          source_template=stale / "source", material_template=stale_golden)
    check("edited-native", False, lambda p, _: (p / "exe/eshkol-run.cpp").write_text("changed\n"))
    check("new-hidden", False, lambda p, _: (p / "inc/eshkol/.new-hidden").write_text("new\n"))
    check("new-nested-scheme", False, lambda p, _: (p / "lib/math/nested/new.esk").write_text("new\n"))
    check("missing-literal", False, lambda p, c: (p / "lib/stdlib.esk").rename(c / "retained-stdlib.esk"))
    check("renamed-native", False, lambda p, _: (p / "inc/eshkol/.hidden.esk").rename(p / "inc/eshkol/renamed.esk"))
    check("tampered-material", False, mutate_material=lambda b: b + b"extra\n")
    check("wrong-expected", False, expected_fp="0" * 64)
    check("invalid-expected", False, expected_fp="not-a-digest")
    check("wrong-toolchain", False, mutate_material=lambda b: b.replace(b"FixtureC-", b"WrongC-"))
    check("self-consistent-omission", False, mutate_material=lambda b: b.split(b"\n", 1)[1],
          expected_fp="rehash-control")
    check("malformed-toolchain", False, mutate_material=lambda b: b.replace(b"toolchain:c=", b"toolchain:invalid="),
          expected_fp="rehash-control")
    check("unsupported-collector", False,
          lambda p, _: (p / "cmake/StdlibIdentityInputs.cmake").write_bytes(collector.read_bytes() + b"# changed\n"))
    check("outside-configured-scopes", True, lambda p, _: (p / "README.md").write_text("not fingerprinted\n"))
    check("scheme-nonmodule", True, lambda p, _: (p / "lib/math/new.txt").write_text("not fingerprinted\n"))
    if os.name != "nt":
        check("symlink-file", False, lambda p, _: (p / "inc/eshkol/link").symlink_to(p / "lib/stdlib.esk"))
        check("symlink-directory", False, lambda p, _: (p / "inc/eshkol/linked-dir").symlink_to(p / "lib", target_is_directory=True))
        check("fifo", False, lambda p, _: os.mkfifo(p / "inc/eshkol/pipe"))
        def symlink_material(p, case):
            (case / "material.txt").rename(case / "retained-material.txt")
            (case / "material.txt").symlink_to(case / "retained-material.txt")
        check("symlink-material", False, symlink_material)
        def symlink_scope(p, case):
            (p / "inc/eshkol").rename(case / "retained-headers")
            (p / "inc/eshkol").symlink_to(case / "retained-headers", target_is_directory=True)
        check("symlink-scope", False, symlink_scope)
    git = shutil.which("git")
    need(git, "git required for ignored-source control")
    def ignored(p, case):
        for suffix, argv in [("init", ["init", "-q", str(p)]),
                             ("add", ["-C", str(p), "add", "."]),
                             ("commit", ["-C", str(p), "-c", "user.name=Identity Fixture", "-c",
                                         "user.email=fixture@invalid", "-c", "commit.gpgsign=false",
                                         "commit", "-qm", "retained fixture"])]:
            need(run([git] + argv, case, "git-" + suffix).returncode == 0, "fixture git " + suffix + " failed")
        (p / "inc/eshkol/.ignored-source").write_text("ignored but identity-relevant\n")
        status = run([git, "-C", str(p), "status", "--porcelain", "--untracked-files=all"], case, "git-status")
        need(status.returncode == 0 and not status.stdout.strip(), "ignored-source control is not git-clean")
    check("git-clean-ignored-input", False, ignored)

    # Deterministic mutation injection between hashing inputs and relisting.
    spec = importlib.util.spec_from_file_location("source_checker", checker)
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    race = work / "namespace-mutation"
    shutil.copytree(project, race / "source")
    (race / "material.txt").write_bytes(golden)
    original_namespace = module.namespace
    calls = [0]
    def mutating_namespace(*argv):
        calls[0] += 1
        if calls[0] == 2:
            (race / "source/inc/eshkol/late-input").write_text("late\n")
        return original_namespace(*argv)
    module.namespace = mutating_namespace
    try:
        module.verify(race / "source", race / "material.txt", fingerprint)
        raise RuntimeError("namespace mutation was accepted")
    except ValueError as error:
        need("namespace changed" in str(error), "mutation refused for wrong reason")
        rows.append({"case": "namespace-mutation", "status": "FAIL", "expected_pass": False, "reason": str(error)})
    module.namespace = original_namespace
    original_read = module.stable_bytes
    target = race / "source/exe/eshkol-run.cpp"
    fired = [False]
    def mutating_read(path):
        value = original_read(path)
        if path == target and not fired[0]:
            fired[0] = True
            target.write_text("changed after reading\n")
        return value
    # Use a fresh retained namespace: the prior late-input remains preserved.
    changed = work / "bytes-mutation"
    shutil.copytree(project, changed / "source")
    (changed / "material.txt").write_bytes(golden)
    target = changed / "source/exe/eshkol-run.cpp"
    module.stable_bytes = mutating_read
    try:
        module.verify(changed / "source", changed / "material.txt", fingerprint)
        raise RuntimeError("byte mutation was accepted")
    except ValueError as error:
        need("source changed during verification" in str(error), "byte mutation refused for wrong reason")
        rows.append({"case": "bytes-mutation", "status": "FAIL", "expected_pass": False, "reason": str(error)})
    summary = {"status": "PASS", "cases": rows, "case_count": len(rows),
               "producer_digest": fingerprint, "legacy_collector_parity": "PASS",
               "elapsed_seconds": round(time.monotonic() - start_time, 3), "retained_work": str(work)}
    (work / "RESULT.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
