#!/usr/bin/env python3
"""Run public bytevector constructor conformance on all four execution routes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--vm-runner", type=Path, required=True)
    args = parser.parse_args()
    runner, vm = args.runner.resolve(), args.vm_runner.resolve()
    source = root / "tests/features/bytevector_constructor_conformance.esk"
    env = dict(os.environ, ESHKOL_PATH=str(root / "lib"),
               ESHKOL_LIB_DIR=str(runner.parent), ESHKOL_JIT_CACHE="0",
               ESHKOL_VM_NO_DISASM="1")
    markers = {"Passed: 16", "Failed: 0", "RESULT: PASS"}

    def run(label, command, expected=markers):
        try:
            result = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                    text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"FAIL: {label}: {error}", file=sys.stderr)
            return False
        print(result.stdout, end="")
        if (result.returncode or not set(expected).issubset(
                {line.strip() for line in result.stdout.splitlines()})
                or any(line.strip().startswith("FAIL:")
                       for line in (result.stdout + result.stderr).splitlines())):
            print(result.stderr, end="", file=sys.stderr)
            print(f"FAIL: {label} (exit {result.returncode})", file=sys.stderr)
            return False
        print(f"PASS: {label}")
        return True

    with tempfile.TemporaryDirectory(prefix="eshkol-bytevector-") as temp:
        temp = Path(temp)
        env["ESHKOL_JIT_CACHE_DIR"] = str(temp / "jit-cache")
        (temp / "jit-cache").mkdir()
        binary, bytecode = temp / "bytevector-aot", temp / "bytevector.eskb"
        debug_binary = temp / "bytevector-debug-aot"
        routes = [
            ("nativeJIT", [str(runner), "-r", str(source), "-L" + str(runner.parent)]),
            ("freshO1AOT compile", [str(runner), "-O", "1", "-o", str(binary),
                                    str(source), "-L" + str(runner.parent)]),
            ("VMsource", [str(vm), str(source)]),
        ]
        if not run(*routes[0]):
            return 1
        if not run(routes[1][0], routes[1][1], expected=()):
            return 1
        if not run("freshO1AOT", [str(binary)]):
            return 1
        if not run("debug AOT compile", [str(runner), "-g", "-O", "1", "-o",
                                           str(debug_binary), str(source),
                                           "-L" + str(runner.parent)], expected=()):
            return 1
        if not run("debug AOT execute", [str(debug_binary)]):
            return 1
        if not run(*routes[2]):
            return 1
        if not run("ESKB emit", [str(runner), "--profile", "hosted-vm",
                                 "--emit-eskb", str(bytecode), str(source),
                                 "-L" + str(runner.parent)], expected=()):
            return 1
        if not run("actualESKB", [str(vm), str(bytecode)]):
            return 1
    fatal_dir = root / "tests/vm_parity/fatal"
    for case in ("bytevector_constructor_type.esk",
                 "bytevector_constructor_negative.esk",
                 "bytevector_constructor_high.esk"):
        for route, command in (
                ("native", [str(runner), "-n", "-r", str(fatal_dir / case),
                             "-L" + str(runner.parent)]),
                ("VM", [str(vm), str(fatal_dir / case)])):
            try:
                result = subprocess.run(command, cwd=root, env=env,
                                        capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.TimeoutExpired) as error:
                print(f"FAIL: fatal {case} {route}: {error}", file=sys.stderr)
                return 1
            if result.returncode == 0:
                print(result.stdout, end="")
                print(f"FAIL: {fatal_dir / case} {route} accepted invalid byte", file=sys.stderr)
                return 1
    print("PASS: bytevector constructor four execution routes (16 checks each), six fatal refusals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
