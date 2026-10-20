#!/usr/bin/env python3
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
"""Non-vacuous checks for the scoped P11 and P6 v1.4 AD evidence gates.

This module validates CTest configuration/report evidence. It never emits an
ICC PASS event; only the production runner writes an event after source/build
identity and all required runtime cases have been checked.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

PROFILES = {
    # Keep P11 as the default for callers and scripts that predate profiles.
    "p11": {
        "tests": {"taylor_numerics_runtime_smoke", "taylor_numerics_aot_smoke"},
        "fixture": "tests/ad/taylor_numerics_test.esk",
        "verdict": "PASS: tower user-numerics 41/41 checks",
        "checks": 41,
        "scope": "v1.4 ODE/root/inversion examples",
        "schema": "eshkol.ad-numerics-gate.v1",
        "binary": "eshkol-run",
        "kind": "ad_numerics",
        "event": "ad_taylor_p11_v14_user_numerics",
        "harness": "run_ad_numerics_gate",
    },
    "p6": {
        "tests": {"exact_taylor_runtime_smoke", "exact_taylor_aot_smoke",
                  "exact_taylor_vm_runtime_smoke"},
        "fixture": "tests/ad/exact_taylor_test.esk",
        "verdict": "PASS: exact-coefficient Taylor tier 192/192 checks",
        "checks": 192,
        "scope": "v1.4 exact rational/bignum Taylor coefficients",
        "schema": "eshkol.ad-numerics-gate.v1",
        "binary": "eshkol-run",
        "kind": "ad_exact",
        "event": "ad_taylor_p6_exact_coefficients",
        "harness": "run_ad_p6_gate",
    },
}
PROFILES = MappingProxyType({
    name: MappingProxyType({**profile, "tests": frozenset(profile["tests"])})
    for name, profile in PROFILES.items()
})
DEFAULT_PROFILE = "p11"


def get_profile(name: str = DEFAULT_PROFILE) -> Mapping:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise GateError(f"unknown AD evidence profile: {name}") from exc


class GateError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered_test_names(payload: str, profile_name: str = DEFAULT_PROFILE) -> set[str]:
    profile = get_profile(profile_name)
    expected = profile["tests"]
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise GateError(f"CTest configuration listing is invalid JSON: {exc}") from exc
    tests = decoded.get("tests") if isinstance(decoded, dict) else None
    if not isinstance(tests, list):
        raise GateError("CTest configuration listing has no tests array")
    names = [item.get("name") for item in tests if isinstance(item, dict)]
    selected = [name for name in names if name in expected]
    if set(selected) != expected or len(selected) != len(expected):
        raise GateError(
            f"configured {profile_name.upper()} CTests do not match the required exact set: "
            f"expected={sorted(expected)} observed={selected}"
        )
    return set(selected)


def validate_test_commands(payload: str, repo_root: Path, build_dir: Path,
                           profile_name: str = DEFAULT_PROFILE) -> set[str]:
    profile = get_profile(profile_name)
    names = registered_test_names(payload, profile_name)
    decoded = json.loads(payload)
    by_name = {item.get("name"): item for item in decoded["tests"] if isinstance(item, dict)}
    binary = str((build_dir / "eshkol-run").resolve())
    fixture = str((repo_root / profile["fixture"]).resolve())
    if profile_name == "p11":
        runtime_name, aot_name = "taylor_numerics_runtime_smoke", "taylor_numerics_aot_smoke"
        runtime = by_name[runtime_name].get("command")
    else:
        runtime_name, aot_name = "exact_taylor_runtime_smoke", "exact_taylor_aot_smoke"
        runtime = by_name[runtime_name].get("command")
    if (not isinstance(runtime, list) or len(runtime) < 3 or runtime[0] != binary
            or runtime[1:3] != ["-r", fixture]):
        raise GateError("runtime CTest command does not run the source-bound fixture with this build binary")
    if profile_name == "p6" and (len(runtime) != 4 or runtime[3] != "-L" + str(build_dir.resolve())):
        raise GateError("P6 JIT command does not use the owned build's standard library path")
    aot = by_name[aot_name].get("command")
    if (not isinstance(aot, list) or len(aot) != 3
            or not isinstance(aot[0], str) or Path(aot[0]).name != "sh" or aot[1] != "-c"):
        raise GateError("AOT CTest command is not the expected shell compile-and-run command")
    stem = "taylor_numerics_aot" if profile_name == "p11" else "exact_taylor_aot"
    aot_output = str((build_dir / stem).resolve())
    try:
        lexer = shlex.shlex(str(aot[2]), posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError as exc:
        raise GateError(f"AOT CTest command cannot be parsed safely: {exc}") from exc
    expected_compile = [binary, "-o", aot_output, fixture, "-L" + str(build_dir.resolve())]
    if tokens != expected_compile + ["&&", aot_output]:
        raise GateError("AOT CTest command does not compile and run the source-bound fixture")
    if profile_name == "p6":
        vm_name = "exact_taylor_vm_runtime_smoke"
        vm = by_name[vm_name].get("command")
        vm_binary = str((build_dir / "eshkol-vm-standalone-test").resolve())
        if not isinstance(vm, list) or vm != [vm_binary, fixture]:
            raise GateError("P6 VM CTest command does not directly execute the source-bound fixture")
    return names


def validate_report(junit_path: Path, verbose_log_path: Path,
                    profile_name: str = DEFAULT_PROFILE) -> dict:
    profile = get_profile(profile_name)
    expected, verdict = profile["tests"], profile["verdict"]
    try:
        root = ET.parse(junit_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise GateError(f"missing or invalid CTest JUnit report: {exc}") from exc
    cases = root.findall(".//testcase")
    names = [case.attrib.get("name", "") for case in cases]
    try:
        total_count = int(root.attrib.get("tests", "-1"))
        failures = int(root.attrib.get("failures", "0"))
        errors = int(root.attrib.get("errors", "0"))
        skipped = int(root.attrib.get("skipped", "0"))
        disabled = int(root.attrib.get("disabled", "0"))
    except ValueError as exc:
        raise GateError(f"CTest JUnit report has invalid suite counts: {exc}") from exc
    if total_count != len(expected) or any((failures, errors, skipped, disabled)):
        raise GateError(
            "CTest suite counts do not prove the exact passing set: "
            f"tests={total_count} expected={len(expected)} failures={failures} "
            f"errors={errors} skipped={skipped} disabled={disabled}"
        )
    if len(cases) != len(expected) or set(names) != expected or len(set(names)) != len(expected):
        raise GateError(
            "CTest JUnit report did not contain exactly the required set: "
            f"expected={sorted(expected)} observed={names}"
        )
    bad = [case.attrib.get("name", "?") for case in cases
           if case.find("failure") is not None or case.find("error") is not None
           or case.find("skipped") is not None]
    if bad:
        raise GateError(f"CTest cases failed, errored, or skipped: {bad}")
    try:
        log = verbose_log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise GateError(f"missing CTest verbose log: {exc}") from exc
    starts = list(re.finditer(r"(?m)^[ \t]*Start\s+\d+:\s*(\S+)\s*$", log))
    blocks: dict[str, str] = {}
    for index, match in enumerate(starts):
        name = match.group(1)
        end = starts[index + 1].start() if index + 1 < len(starts) else len(log)
        blocks.setdefault(name, log[match.start():end])
    if set(blocks) != expected:
        raise GateError(
            "verbose CTest log did not show exactly the required cases: "
            f"expected={sorted(expected)} observed={sorted(blocks)}"
        )
    for name in sorted(expected):
        lines = blocks[name].splitlines()
        verdict_index = next((i for i, line in enumerate(lines) if verdict in line), None)
        if verdict_index is None:
            raise GateError(f"{name} did not print the explicit {verdict!r} verdict")
        if not re.search(rf"{profile['checks']}/{profile['checks']} checks", lines[verdict_index]):
            raise GateError(f"{name} printed a zero or incorrect check count")
        runtime_errors = [line for line in lines
                           if re.search(r"\bERROR:|\bJIT session error:|\bFAIL\b", line)]
        if runtime_errors:
            raise GateError(f"{name} printed runtime error diagnostics: {runtime_errors}")
    return {"cases": sorted(expected), "verdict": verdict,
            "checks_per_case": profile["checks"], "profile": profile_name}


def validate_build_source(repo_root: Path, build_dir: Path,
                          profile_name: str = DEFAULT_PROFILE) -> dict:
    cache_path = build_dir / "CMakeCache.txt"
    ctest_file = build_dir / "CTestTestfile.cmake"
    binary = build_dir / "eshkol-run"
    if not cache_path.is_file() or not ctest_file.is_file():
        raise GateError("build is missing CMakeCache.txt or CTestTestfile.cmake")
    if not binary.is_file() or not binary.stat().st_mode & 0o111:
        raise GateError("build is missing executable build/eshkol-run")
    configured = None
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
            configured = line.partition("=")[2]
    if not configured:
        raise GateError("CMakeCache.txt has no CMAKE_HOME_DIRECTORY")
    configured_root = Path(configured).resolve()
    if configured_root != repo_root.resolve():
        raise GateError(f"build source mismatch: configured={configured_root} expected={repo_root.resolve()}")
    result = {
        "configured_source_root": str(configured_root),
        "cmake_cache_sha256": sha256_file(cache_path),
        "ctest_file_sha256": sha256_file(ctest_file),
        "binary_path": str(binary.resolve()),
        "binary_sha256": sha256_file(binary),
        "binary_size": binary.stat().st_size,
    }
    if profile_name == "p6":
        vm_binary = build_dir / "eshkol-vm-standalone-test"
        if not vm_binary.is_file() or not vm_binary.stat().st_mode & 0o111:
            raise GateError("P6 requires executable build/eshkol-vm-standalone-test; refusing without the VM lane")
        result["vm_binary"] = {
            "path": str(vm_binary.resolve()), "sha256": sha256_file(vm_binary),
            "size": vm_binary.stat().st_size,
        }
    return result


def source_identity(repo_root: Path) -> dict:
    def git(*args: str) -> str:
        proc = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)
        if proc.returncode:
            raise GateError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout.strip()

    top = Path(git("rev-parse", "--show-toplevel")).resolve()
    if top != repo_root.resolve():
        raise GateError(f"source root is not its own Git checkout: top={top} expected={repo_root.resolve()}")
    dirty = git("status", "--porcelain")
    if dirty:
        raise GateError("source checkout is dirty; commit or remove changes before recording evidence")
    return {
        "root": str(top),
        "git_head": git("rev-parse", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def fingerprint_events(run_dir: Path, build: dict, source: dict,
                       profile_name: str = DEFAULT_PROFILE) -> list[dict]:
    path = run_dir / "build_fingerprint.jsonl"
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"missing or invalid shared build fingerprint record: {exc}") from exc
    harness = get_profile(profile_name)["harness"]
    records = [record for record in records if record.get("harness") == harness]
    expected_binaries = {"eshkol-run": build["binary_path"]}
    if profile_name == "p6":
        expected_binaries["eshkol-vm-standalone-test"] = build["vm_binary"]["path"]
    groups = {name: [record for record in records if record.get("binary") == name]
              for name in expected_binaries}
    if len(records) != 2 * len(expected_binaries) or any(len(group) != 2 for group in groups.values()):
        raise GateError(f"expected pre/post fingerprints for {sorted(expected_binaries)}, found {len(records)}")
    for name, group in groups.items():
        for record in group:
            if record.get("path") != expected_binaries[name]:
                raise GateError(f"build fingerprint describes a different {name} path")
            if record.get("git_sha") != source["git_head"]:
                raise GateError("build fingerprint source revision does not match current source")
            if record.get("sha256") in (None, "unavailable"):
                raise GateError("build fingerprint has no binary content digest")
    stable_fields = ("path", "size", "mtime", "sha256", "git_sha")
    for name, group in groups.items():
        if any(group[0].get(key) != group[1].get(key) for key in stable_fields):
            raise GateError(f"build binary or source identity changed during the gate: {name}")
        expected_digest = (build["binary_sha256"] if name == "eshkol-run"
                           else build["vm_binary"]["sha256"])
        if group[1].get("sha256") != expected_digest:
            raise GateError(f"shared fingerprint digest does not match current binary: {name}")
    return records


def verify_build_freshness(repo_root: Path, build_dir: Path, run_dir: Path,
                           profile_name: str = DEFAULT_PROFILE) -> dict:
    command = [sys.executable, str(repo_root / "scripts/check_build_fingerprint.py"),
               "--repo-root", str(repo_root), "--build-dir", str(build_dir),
               "--binary", "eshkol-run", "--trace-dir", str(run_dir), "--no-trace", "--format", "json"]
    proc = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise GateError(f"build fingerprint freshness check returned invalid JSON: {exc}; {proc.stderr.strip()}") from exc
    if proc.returncode or report.get("status") != "PASS":
        raise GateError(f"build fingerprint freshness check failed: {report}")
    if profile_name == "p6":
        vm_command = command.copy()
        vm_command[vm_command.index("eshkol-run")] = "eshkol-vm-standalone-test"
        vm_proc = subprocess.run(vm_command, cwd=repo_root, capture_output=True, text=True, check=False)
        try:
            vm_report = json.loads(vm_proc.stdout)
        except json.JSONDecodeError as exc:
            raise GateError(f"VM build fingerprint freshness check returned invalid JSON: {exc}; {vm_proc.stderr.strip()}") from exc
        if vm_proc.returncode or vm_report.get("status") != "PASS":
            raise GateError(f"VM build fingerprint freshness check failed: {vm_report}")
        return {"jit": report, "vm": vm_report}
    return report


def write_json_exclusive(path: Path, value: dict) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_and_write(repo_root: Path, build_dir: Path, run_dir: Path,
                     profile_name: str = DEFAULT_PROFILE) -> dict:
    profile = get_profile(profile_name)
    source = source_identity(repo_root)
    build = validate_build_source(repo_root, build_dir, profile_name)
    test_list = (run_dir / "ctest-list.json").read_text(encoding="utf-8")
    validate_test_commands(test_list, repo_root, build_dir, profile_name)
    result = validate_report(run_dir / "ctest.xml", run_dir / "ctest.verbose.log", profile_name)
    fingerprints = fingerprint_events(run_dir, build, source, profile_name)
    freshness = verify_build_freshness(repo_root, build_dir, run_dir, profile_name)
    fixture_path = repo_root / profile["fixture"]
    if not fixture_path.is_file():
        raise GateError(f"{profile_name.upper()} fixture is missing")
    if source_identity(repo_root) != source:
        raise GateError("source identity changed while the gate was running")
    build_after = validate_build_source(repo_root, build_dir, profile_name)
    if build_after != build:
        raise GateError("build identity changed while the gate was running")
    receipt = {
        "schema": profile["schema"],
        "status": "PASS",
        "profile": profile_name,
        "scope": profile["scope"],
        "source": source,
        "build": build,
        "ctest_listing_sha256": sha256_file(run_dir / "ctest-list.json"),
        "build_fingerprint_events": fingerprints,
        "build_freshness": freshness,
        "fixture": {"path": profile["fixture"], "sha256": sha256_file(fixture_path),
                    "size": fixture_path.stat().st_size},
        "tests": result,
        "recorded_epoch": int(time.time()),
    }
    if profile_name == "p6":
        aot_path = build_dir / "exact_taylor_aot"
        if not aot_path.is_file() or not aot_path.stat().st_mode & 0o111:
            raise GateError("P6 AOT CTest did not leave its compiled executable")
        receipt["aot_binary"] = {"path": str(aot_path.resolve()), "sha256": sha256_file(aot_path),
                                 "size": aot_path.stat().st_size}
    write_json_exclusive(run_dir / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return receipt


def write_failure_receipt(repo_root: Path, build_dir: Path, run_dir: Path, reason: str,
                          profile_name: str = DEFAULT_PROFILE) -> None:
    profile = get_profile(profile_name)
    receipt = {"schema": profile["schema"], "status": "FAIL", "profile": profile_name,
               "scope": profile["scope"], "reason": reason,
               "recorded_epoch": int(time.time())}
    try:
        receipt["source"] = source_identity(repo_root)
    except GateError:
        receipt["source"] = {"root": str(repo_root.resolve()), "identity": "unavailable"}
    try:
        receipt["build"] = validate_build_source(repo_root, build_dir, profile_name)
    except (OSError, GateError):
        receipt["build"] = {"identity": "unavailable"}
    path = run_dir / "receipt.json"
    if not path.exists():
        write_json_exclusive(path, receipt)


def cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", nargs=2, metavar=("REPO_ROOT", "BUILD_DIR"))
    parser.add_argument("--check-tests", nargs=3, metavar=("JSON", "REPO_ROOT", "BUILD_DIR"))
    parser.add_argument("--verify-and-write", nargs=3, metavar=("REPO_ROOT", "BUILD_DIR", "RUN_DIR"))
    parser.add_argument("--write-failure-receipt", nargs=4, metavar=("REPO_ROOT", "BUILD_DIR", "RUN_DIR", "REASON"))
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test(args.profile)
        elif args.preflight:
            repo, build = map(Path, args.preflight)
            source_identity(repo)
            validate_build_source(repo, build, args.profile)
            print(f"PASS: {args.profile.upper()} build configured from clean source checkout")
        elif args.check_tests:
            path, repo, build = args.check_tests
            payload = Path(path).read_text(encoding="utf-8")
            validate_test_commands(payload, Path(repo), Path(build), args.profile)
            print(f"PASS: exact {args.profile.upper()} CTest set registered")
        elif args.verify_and_write:
            verify_and_write(*map(Path, args.verify_and_write), args.profile)
        elif args.write_failure_receipt:
            repo, build, run_dir, reason = args.write_failure_receipt
            write_failure_receipt(Path(repo), Path(build), Path(run_dir), reason, args.profile)
        else:
            parser.error("choose --self-test, --preflight, --check-tests, --verify-and-write, or --write-failure-receipt")
    except (GateError, OSError) as exc:
        print(f"{args.profile.upper()} gate validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


def self_test(profile_name: str = DEFAULT_PROFILE) -> None:
    profile = get_profile(profile_name)
    expected_tests, verdict = profile["tests"], profile["verdict"]
    import tempfile

    def fixtures(case: str, root: Path) -> tuple[Path, Path]:
        names = sorted(expected_tests)
        if case in {"missing", "renamed"}:
            names = names[:1] if case == "missing" else ["renamed_test", names[1]]
        xml = [f'<testsuite tests="{len(names)}" failures="{int(case == "failed")}" '
               f'errors="{int(case == "error-count")}" skipped="{int(case == "skipped")}" disabled="0">']
        log = []
        for idx, name in enumerate(names, 1):
            skipped = '<skipped/>' if case == "skipped" and idx == 1 else ""
            failure = '<failure>injected</failure>' if case == "failed" and idx == 1 else ""
            printed = (profile["verdict"].replace(f"{profile['checks']}/{profile['checks']}", "0/0")
                       if case == "zero-count" else verdict)
            xml.append(f'<testcase name="{name}">{skipped}{failure}</testcase>')
            log.extend((f"  Start {idx}: {name}", f"{idx}: {printed}"))
            if case == "error-after-verdict" and idx == 1:
                log.append("1: ERROR: injected post-verdict diagnostic")
            elif case == "jit-session-error" and idx == 1:
                log.append("1: JIT session error: injected post-verdict diagnostic")
            elif case == "fail-after-verdict" and idx == 1:
                log.append("1: FAIL injected post-verdict diagnostic")
            elif case == "linker-warning" and idx == 1:
                log.append("1: ld: warning: injected benign linker warning")
        xml.append("</testsuite>")
        junit, verbose = root / f"{case}.xml", root / f"{case}.log"
        junit.write_text("".join(xml), encoding="utf-8")
        verbose.write_text("\n".join(log) + "\n", encoding="utf-8")
        return junit, verbose

    with tempfile.TemporaryDirectory(prefix="ad-numerics-gate-") as temp:
        root = Path(temp)
        junit, verbose = fixtures("linker-warning", root)
        validate_report(junit, verbose, profile_name)
        for case in ("missing", "renamed", "skipped", "zero-count", "failed", "error-count",
                     "error-after-verdict", "jit-session-error", "fail-after-verdict"):
            junit, verbose = fixtures(case, root)
            try:
                validate_report(junit, verbose, profile_name)
            except GateError:
                pass
            else:
                raise AssertionError(f"failure injection {case!r} was accepted")
        expected = sorted(expected_tests)
        for case, names in (("missing", expected[:1]), ("renamed", ["renamed_test", expected[1]])):
            try:
                registered_test_names(json.dumps({"tests": [{"name": name} for name in names]}), profile_name)
            except GateError:
                pass
            else:
                raise AssertionError(f"CTest configuration failure injection {case!r} was accepted")
        source, wrong, build = root / "source", root / "wrong-source", root / "build"
        for path in (source, wrong, build):
            path.mkdir()
        (build / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={wrong}\n", encoding="utf-8")
        (build / "CTestTestfile.cmake").write_text("", encoding="utf-8")
        binary = build / "eshkol-run"
        binary.write_text("fixture", encoding="utf-8")
        binary.chmod(0o755)
        try:
            validate_build_source(source, build)
        except GateError:
            pass
        else:
            raise AssertionError("wrong-source build was accepted")
        (build / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n", encoding="utf-8")
        binary.unlink()
        try:
            validate_build_source(source, build)
        except GateError:
            pass
        else:
            raise AssertionError("build without eshkol-run was accepted")
        (build / "CTestTestfile.cmake").unlink()
        try:
            validate_build_source(source, build)
        except GateError:
            pass
        else:
            raise AssertionError("build without CTest configuration was accepted")
        (build / "CTestTestfile.cmake").write_text("", encoding="utf-8")
        binary.write_text("fixture", encoding="utf-8")
        binary.chmod(0o755)
        if profile_name == "p6":
            try:
                validate_build_source(source, build, profile_name)
            except GateError:
                pass
            else:
                raise AssertionError("P6 build without standalone VM was accepted")
            vm_binary = build / "eshkol-vm-standalone-test"
            vm_binary.write_text("fixture", encoding="utf-8")
            vm_binary.chmod(0o755)
            validate_build_source(source, build, profile_name)
        fixture = str((source / profile["fixture"]).resolve())
        binary_path = str((build / "eshkol-run").resolve())
        if profile_name == "p11":
            aot_path = str((build / "taylor_numerics_aot").resolve())
            good_commands = {"tests": [
                {"name": "taylor_numerics_runtime_smoke", "command": [binary_path, "-r", fixture]},
                {"name": "taylor_numerics_aot_smoke", "command": ["sh", "-c", f"{binary_path} -o {aot_path} {fixture} -L{build.resolve()} && {aot_path}"]},
            ]}
        else:
            aot_path = str((build / "exact_taylor_aot").resolve())
            vm_path = str((build / "eshkol-vm-standalone-test").resolve())
            good_commands = {"tests": [
                {"name": "exact_taylor_runtime_smoke", "command": [binary_path, "-r", fixture, "-L" + str(build.resolve())]},
                {"name": "exact_taylor_aot_smoke", "command": ["sh", "-c", f"{binary_path} -o {aot_path} {fixture} -L{build.resolve()} && {aot_path}"]},
                {"name": "exact_taylor_vm_runtime_smoke", "command": [vm_path, fixture]},
            ]}
        validate_test_commands(json.dumps(good_commands), source, build, profile_name)
        absolute_shell = json.loads(json.dumps(good_commands))
        aot_case = "taylor_numerics_aot_smoke" if profile_name == "p11" else "exact_taylor_aot_smoke"
        next(test for test in absolute_shell["tests"] if test["name"] == aot_case)["command"][0] = "/bin/sh"
        validate_test_commands(json.dumps(absolute_shell), source, build, profile_name)
        junit, verbose = fixtures("good", root)
        verbose.write_text(verbose.read_text().replace(verdict, "1: ERROR: injected before verdict\n" + verdict, 1))
        try:
            validate_report(junit, verbose, profile_name)
        except GateError:
            pass
        else:
            raise AssertionError("runtime ERROR before a valid verdict was accepted")
        wrong_commands = json.loads(json.dumps(good_commands))
        wrong_commands["tests"][0]["command"][2] = str((wrong / profile["fixture"]).resolve())
        try:
            validate_test_commands(json.dumps(wrong_commands), source, build, profile_name)
        except GateError:
            pass
        else:
            raise AssertionError("wrong-source CTest fixture command was accepted")
        compile_only = json.loads(json.dumps(good_commands))
        aot_case_obj = next(test for test in compile_only["tests"] if test["name"] == aot_case)
        aot_case_obj["command"][2] = f"{binary_path} -o {aot_path} {fixture} -L{build.resolve()}"
        echo_command = json.loads(json.dumps(good_commands))
        echo_command["tests"][0]["command"] = ["echo", binary_path, "-r", fixture]
        for label, payload in (("compile-only AOT", compile_only), ("echo runtime", echo_command)):
            try:
                validate_test_commands(json.dumps(payload), source, build, profile_name)
            except GateError:
                pass
            else:
                raise AssertionError(f"{label} CTest command was accepted")
    print(f"PASS: {profile_name.upper()} gate controls rejected missing/renamed/skipped/0-count/failed/post-verdict ERROR/JIT-session-error/FAIL/wrong-source/compile-only/echo commands; benign linker warning accepted")


if __name__ == "__main__":
    raise SystemExit(cli())
