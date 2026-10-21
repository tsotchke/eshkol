#!/usr/bin/env python3
"""Require MessagePack wire golden tests and strict UTF-8 refusal tests."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path,
                        default=root / os.environ.get("BUILD_DIR", "build") / "eshkol-run")
    parser.add_argument("--all-routes", action="store_true",
                        help="also require fresh O1 AOT, VM source and emitted ESKB")
    parser.add_argument("--vm-runner", type=Path,
                        help="standalone VM (default: beside --runner)")
    args = parser.parse_args()
    runner = args.runner.resolve()
    vm_name = "eshkol-vm-standalone-test" + (".exe" if os.name == "nt" else "")
    vm = (args.vm_runner or runner.parent / vm_name).resolve()
    env = dict(os.environ, ESHKOL_PATH=str(root / "lib"),
               ESHKOL_LIB_DIR=str(runner.parent), ESHKOL_JIT_CACHE="0",
               ESHKOL_VM_NO_DISASM="1")

    def run(label, command, markers=()):
        try:
            result = subprocess.run(
                command, cwd=root, env=env,
                capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"FAIL: {label}: {error}", file=sys.stderr)
            return False
        print(result.stdout, end="")
        lines = {line.strip() for line in result.stdout.splitlines()}
        if (result.returncode or not set(markers).issubset(lines)
                or "FAIL:" in result.stdout + result.stderr):
            print(result.stderr, end="", file=sys.stderr)
            print(f"FAIL: {label} (runner exit {result.returncode})", file=sys.stderr)
            return False
        print(f"PASS: {label}")
        return True

    with tempfile.TemporaryDirectory(prefix="eshkol-msgpack-") as work:
        for test, markers in [
                ("msgpack_test.esk", ("RESULT: OK", "Passed: 42", "Failed: 0")),
                ("msgpack_utf8_test.esk", ("PASS: MessagePack UTF-8 80 checks",))]:
            source = root / "tests/v1_2_edge_cases" / test
            if not run(f"{test} JIT", [str(runner), "-r", str(source),
                                      "-L" + str(runner.parent)], markers):
                return 1
            if not args.all_routes:
                continue
            binary = Path(work) / (source.stem + (".exe" if os.name == "nt" else ".bin"))
            bytecode = Path(work) / (source.stem + ".eskb")
            stages = [
                ("O1 AOT compile", [str(runner), "-O", "1", "-o", str(binary),
                                    str(source), "-L" + str(runner.parent)], ()),
                ("O1 AOT execute", [str(binary)], markers),
                ("VM source", [str(vm), str(source)], markers),
                ("ESKB emit", [str(runner), "--profile", "hosted-vm", "--emit-eskb",
                               str(bytecode), str(source), "-L" + str(runner.parent)], ()),
                ("ESKB execute", [str(vm), str(bytecode)], markers),
            ]
            for label, command, expected in stages:
                if not run(f"{test} {label}", command, expected):
                    return 1
    if args.all_routes:
        print("PASS: MessagePack four execution routes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
