#!/usr/bin/env python3
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
"""Non-vacuous checks for the scoped P11 v1.4 numerical evidence gate.

This module validates CTest configuration/report evidence. It never emits an
ICC PASS event; only the production runner writes an event after source/build
identity and both runtime cases have been checked.
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

EXPECTED_TESTS = {"taylor_numerics_runtime_smoke", "taylor_numerics_aot_smoke"}
VERDICT = "PASS: tower user-numerics 41/41 checks"


class GateError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered_test_names(payload: str) -> set[str]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise GateError(f"CTest configuration listing is invalid JSON: {exc}") from exc
    tests = decoded.get("tests") if isinstance(decoded, dict) else None
    if not isinstance(tests, list):
        raise GateError("CTest configuration listing has no tests array")
    names = [item.get("name") for item in tests if isinstance(item, dict)]
    selected = [name for name in names if name in EXPECTED_TESTS]
    if set(selected) != EXPECTED_TESTS or len(selected) != len(EXPECTED_TESTS):
        raise GateError(
            "configured P11 CTests do not match the required exact pair: "
            f"expected={sorted(EXPECTED_TESTS)} observed={selected}"
        )
    return set(selected)


def validate_test_commands(payload: str, repo_root: Path, build_dir: Path) -> set[str]:
    names = registered_test_names(payload)
    decoded = json.loads(payload)
    by_name = {item.get("name"): item for item in decoded["tests"] if isinstance(item, dict)}
    binary = str((build_dir / "eshkol-run").resolve())
    fixture = str((repo_root / "tests/ad/taylor_numerics_test.esk").resolve())
    runtime = by_name["taylor_numerics_runtime_smoke"].get("command")
    aot = by_name["taylor_numerics_aot_smoke"].get("command")
    if (not isinstance(runtime, list) or len(runtime) < 3 or runtime[0] != binary
            or runtime[1:3] != ["-r", fixture]):
        raise GateError("runtime CTest command does not run the source-bound fixture with this build binary")
    if not isinstance(aot, list) or len(aot) != 3 or aot[:2] != ["sh", "-c"]:
        raise GateError("AOT CTest command is not the expected shell compile-and-run command")
    aot_output = str((build_dir / "taylor_numerics_aot").resolve())
    try:
        lexer = shlex.shlex(str(aot[2]), posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError as exc:
        raise GateError(f"AOT CTest command cannot be parsed safely: {exc}") from exc
    expected_compile = [binary, "-o", aot_output, fixture, "-L" + str(build_dir.resolve())]
    if tokens != expected_compile + ["&&", aot_output]:
        raise GateError("AOT CTest command does not compile and run the source-bound fixture")
    return names


def validate_report(junit_path: Path, verbose_log_path: Path) -> dict:
    try:
        root = ET.parse(junit_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise GateError(f"missing or invalid CTest JUnit report: {exc}") from exc
    cases = root.findall(".//testcase")
    names = [case.attrib.get("name", "") for case in cases]
    if len(cases) != 2 or set(names) != EXPECTED_TESTS or len(set(names)) != 2:
        raise GateError(
            "CTest JUnit report did not contain exactly the required pair: "
            f"expected={sorted(EXPECTED_TESTS)} observed={names}"
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
    if set(blocks) != EXPECTED_TESTS:
        raise GateError(
            "verbose CTest log did not show exactly the required cases: "
            f"expected={sorted(EXPECTED_TESTS)} observed={sorted(blocks)}"
        )
    for name in sorted(EXPECTED_TESTS):
        lines = blocks[name].splitlines()
        verdict_index = next((i for i, line in enumerate(lines) if VERDICT in line), None)
        if verdict_index is None:
            raise GateError(f"{name} did not print the explicit {VERDICT!r} verdict")
        trailing_errors = [line for line in lines[verdict_index + 1:]
                           if re.search(r"\bERROR:|\bJIT session error:|\bFAIL\b", line)]
        if trailing_errors:
            raise GateError(f"{name} printed an error after its 41/41 verdict: {trailing_errors}")
    return {"cases": sorted(EXPECTED_TESTS), "verdict": VERDICT, "checks_per_case": 41}


def validate_build_source(repo_root: Path, build_dir: Path) -> dict:
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
    return {
        "configured_source_root": str(configured_root),
        "cmake_cache_sha256": sha256_file(cache_path),
        "ctest_file_sha256": sha256_file(ctest_file),
        "binary_path": str(binary.resolve()),
        "binary_sha256": sha256_file(binary),
        "binary_size": binary.stat().st_size,
    }


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


def fingerprint_events(run_dir: Path, build: dict, source: dict) -> list[dict]:
    path = run_dir / "build_fingerprint.jsonl"
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"missing or invalid shared build fingerprint record: {exc}") from exc
    records = [record for record in records if record.get("harness") == "run_ad_numerics_gate"]
    if len(records) != 2:
        raise GateError(f"expected pre/post build fingerprints, found {len(records)}")
    expected_path = build["binary_path"]
    for record in records:
        if record.get("path") != expected_path or record.get("binary") != "eshkol-run":
            raise GateError("build fingerprint describes a different binary path")
        if record.get("git_sha") != source["git_head"]:
            raise GateError("build fingerprint source revision does not match current source")
        if record.get("sha256") in (None, "unavailable"):
            raise GateError("build fingerprint has no binary content digest")
    stable_fields = ("path", "size", "mtime", "sha256", "git_sha")
    if any(records[0].get(key) != records[1].get(key) for key in stable_fields):
        raise GateError("build binary or source identity changed during the gate")
    if records[1].get("sha256") != build["binary_sha256"]:
        raise GateError("shared fingerprint digest does not match current binary")
    return records


def verify_build_freshness(repo_root: Path, build_dir: Path, run_dir: Path) -> dict:
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
    return report


def write_json_exclusive(path: Path, value: dict) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_and_write(repo_root: Path, build_dir: Path, run_dir: Path) -> dict:
    source = source_identity(repo_root)
    build = validate_build_source(repo_root, build_dir)
    test_list = (run_dir / "ctest-list.json").read_text(encoding="utf-8")
    validate_test_commands(test_list, repo_root, build_dir)
    result = validate_report(run_dir / "ctest.xml", run_dir / "ctest.verbose.log")
    fingerprints = fingerprint_events(run_dir, build, source)
    freshness = verify_build_freshness(repo_root, build_dir, run_dir)
    fixture_path = repo_root / "tests/ad/taylor_numerics_test.esk"
    if not fixture_path.is_file():
        raise GateError("P11 numerical fixture is missing")
    if source_identity(repo_root) != source:
        raise GateError("source identity changed while the gate was running")
    build_after = validate_build_source(repo_root, build_dir)
    if build_after != build:
        raise GateError("build identity changed while the gate was running")
    receipt = {
        "schema": "eshkol.ad-numerics-gate.v1",
        "status": "PASS",
        "scope": "v1.4 ODE/root/inversion examples",
        "source": source,
        "build": build,
        "ctest_listing_sha256": sha256_file(run_dir / "ctest-list.json"),
        "build_fingerprint_events": fingerprints,
        "build_freshness": freshness,
        "fixture": {"path": "tests/ad/taylor_numerics_test.esk", "sha256": sha256_file(fixture_path),
                    "size": fixture_path.stat().st_size},
        "tests": result,
        "recorded_epoch": int(time.time()),
    }
    write_json_exclusive(run_dir / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return receipt


def write_failure_receipt(repo_root: Path, build_dir: Path, run_dir: Path, reason: str) -> None:
    receipt = {"schema": "eshkol.ad-numerics-gate.v1", "status": "FAIL",
               "scope": "v1.4 ODE/root/inversion examples", "reason": reason,
               "recorded_epoch": int(time.time())}
    try:
        receipt["source"] = source_identity(repo_root)
    except GateError:
        receipt["source"] = {"root": str(repo_root.resolve()), "identity": "unavailable"}
    try:
        receipt["build"] = validate_build_source(repo_root, build_dir)
    except (OSError, GateError):
        receipt["build"] = {"identity": "unavailable"}
    path = run_dir / "receipt.json"
    if not path.exists():
        write_json_exclusive(path, receipt)


def cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", nargs=2, metavar=("REPO_ROOT", "BUILD_DIR"))
    parser.add_argument("--check-tests", nargs=3, metavar=("JSON", "REPO_ROOT", "BUILD_DIR"))
    parser.add_argument("--verify-and-write", nargs=3, metavar=("REPO_ROOT", "BUILD_DIR", "RUN_DIR"))
    parser.add_argument("--write-failure-receipt", nargs=4, metavar=("REPO_ROOT", "BUILD_DIR", "RUN_DIR", "REASON"))
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        elif args.preflight:
            repo, build = map(Path, args.preflight)
            source_identity(repo)
            validate_build_source(repo, build)
            print("PASS: P11 build configured from clean source checkout")
        elif args.check_tests:
            path, repo, build = args.check_tests
            payload = Path(path).read_text(encoding="utf-8")
            validate_test_commands(payload, Path(repo), Path(build))
            print("PASS: exact P11 CTest pair registered")
        elif args.verify_and_write:
            verify_and_write(*map(Path, args.verify_and_write))
        elif args.write_failure_receipt:
            repo, build, run_dir, reason = args.write_failure_receipt
            write_failure_receipt(Path(repo), Path(build), Path(run_dir), reason)
        else:
            parser.error("choose --self-test, --preflight, --check-tests, --verify-and-write, or --write-failure-receipt")
    except (GateError, OSError) as exc:
        print(f"P11 gate validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


def self_test() -> None:
    import tempfile

    def fixtures(case: str, root: Path) -> tuple[Path, Path]:
        names = sorted(EXPECTED_TESTS)
        if case in {"missing", "renamed"}:
            names = names[:1] if case == "missing" else ["renamed_test", names[1]]
        xml = ["<testsuites><testsuite>"]
        log = []
        for idx, name in enumerate(names, 1):
            skipped = '<skipped/>' if case == "skipped" and idx == 1 else ""
            failure = '<failure>injected</failure>' if case == "failed" and idx == 1 else ""
            verdict = "PASS: tower user-numerics 0/0 checks" if case == "zero-count" else VERDICT
            xml.append(f'<testcase name="{name}">{skipped}{failure}</testcase>')
            log.extend((f"  Start {idx}: {name}", f"{idx}: {verdict}"))
            if case == "error-after-verdict" and idx == 1:
                log.append("1: ERROR: injected post-verdict diagnostic")
            elif case == "jit-session-error" and idx == 1:
                log.append("1: JIT session error: injected post-verdict diagnostic")
            elif case == "fail-after-verdict" and idx == 1:
                log.append("1: FAIL injected post-verdict diagnostic")
            elif case == "linker-warning" and idx == 1:
                log.append("1: ld: warning: injected benign linker warning")
        xml.append("</testsuite></testsuites>")
        junit, verbose = root / f"{case}.xml", root / f"{case}.log"
        junit.write_text("".join(xml), encoding="utf-8")
        verbose.write_text("\n".join(log) + "\n", encoding="utf-8")
        return junit, verbose

    with tempfile.TemporaryDirectory(prefix="ad-numerics-gate-") as temp:
        root = Path(temp)
        junit, verbose = fixtures("linker-warning", root)
        validate_report(junit, verbose)
        for case in ("missing", "renamed", "skipped", "zero-count", "failed",
                     "error-after-verdict", "jit-session-error", "fail-after-verdict"):
            junit, verbose = fixtures(case, root)
            try:
                validate_report(junit, verbose)
            except GateError:
                pass
            else:
                raise AssertionError(f"failure injection {case!r} was accepted")
        expected = sorted(EXPECTED_TESTS)
        for case, names in (("missing", expected[:1]), ("renamed", ["renamed_test", expected[1]])):
            try:
                registered_test_names(json.dumps({"tests": [{"name": name} for name in names]}))
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
        fixture = str((source / "tests/ad/taylor_numerics_test.esk").resolve())
        binary_path = str((build / "eshkol-run").resolve())
        aot_path = str((build / "taylor_numerics_aot").resolve())
        good_commands = {"tests": [
            {"name": "taylor_numerics_runtime_smoke", "command": [binary_path, "-r", fixture]},
            {"name": "taylor_numerics_aot_smoke", "command": ["sh", "-c", f"{binary_path} -o {aot_path} {fixture} -L{build.resolve()} && {aot_path}"]},
        ]}
        validate_test_commands(json.dumps(good_commands), source, build)
        wrong_commands = json.loads(json.dumps(good_commands))
        wrong_commands["tests"][0]["command"][2] = str((wrong / "tests/ad/taylor_numerics_test.esk").resolve())
        try:
            validate_test_commands(json.dumps(wrong_commands), source, build)
        except GateError:
            pass
        else:
            raise AssertionError("wrong-source CTest fixture command was accepted")
        compile_only = json.loads(json.dumps(good_commands))
        compile_only["tests"][1]["command"][2] = f"{binary_path} -o {aot_path} {fixture} -L{build.resolve()}"
        echo_command = json.loads(json.dumps(good_commands))
        echo_command["tests"][0]["command"] = ["echo", binary_path, "-r", fixture]
        for label, payload in (("compile-only AOT", compile_only), ("echo runtime", echo_command)):
            try:
                validate_test_commands(json.dumps(payload), source, build)
            except GateError:
                pass
            else:
                raise AssertionError(f"{label} CTest command was accepted")
    print("PASS: P11 gate controls rejected missing/renamed/skipped/0-count/failed/post-verdict ERROR/JIT-session-error/FAIL/wrong-source/compile-only/echo commands; benign linker warning accepted")


if __name__ == "__main__":
    raise SystemExit(cli())
