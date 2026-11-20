#!/usr/bin/env python3
"""Run a wire-format suite (MessagePack or Protocol Buffers) on every route.

Each suite is a list of Eshkol test programs with the output markers they
must print. With --all-routes every program also runs as a fresh O1 AOT
binary, on the VM from source, and from emitted ESKB bytecode. A suite may
also name native golden-vector test binaries (--native-test) that must exit
0 and print their PASS marker.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


SUITES = {
    "msgpack": {
        "title": "MessagePack",
        "tests": [
            ("msgpack_test.esk", ("RESULT: OK", "Passed: 42", "Failed: 0")),
            ("msgpack_utf8_test.esk", ("PASS: MessagePack UTF-8 80 checks",)),
        ],
        "native_marker": None,
    },
    "protobuf": {
        "title": "Protocol Buffers",
        "tests": [
            ("protobuf_test.esk", ("RESULT: OK", "Passed: 190", "Failed: 0")),
            ("protobuf_proto_test.esk", ("RESULT: OK", "Passed: 31", "Failed: 0")),
        ],
        "native_marker": "PASS: protobuf wire golden vectors",
    },
}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--runner", type=Path,
                        default=root / os.environ.get("BUILD_DIR", "build") / "eshkol-run")
    parser.add_argument("--all-routes", action="store_true",
                        help="also require fresh O1 AOT, VM source and emitted ESKB")
    parser.add_argument("--vm-runner", type=Path,
                        help="standalone VM (default: beside --runner)")
    parser.add_argument("--native-test", type=Path, action="append", default=[],
                        help="native golden-vector test binary to run first")
    args = parser.parse_args(argv)
    suite = SUITES[args.suite]
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

    for native in args.native_test:
        if suite["native_marker"] is None:
            print(f"FAIL: suite {args.suite} has no native golden-vector test", file=sys.stderr)
            return 1
        if not run(f"{native.name} native", [str(native.resolve())],
                   (suite["native_marker"],)):
            return 1

    with tempfile.TemporaryDirectory(prefix=f"eshkol-{args.suite}-") as work:
        for test, markers in suite["tests"]:
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
        print(f"PASS: {suite['title']} four execution routes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
