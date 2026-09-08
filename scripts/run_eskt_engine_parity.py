#!/usr/bin/env python3
"""Valid ESKT 4x4 producer/consumer matrix; no malformed inputs are executed."""

import argparse
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/core/eskt_engine_parity.esk"
ENGINES = ("jit", "aot", "vm-source", "vm-bytecode")
CASES = {
    "vector": ((6,), (1.5, -2.0, 0.0, -0.0, 0.125, 1024.5)),
    "matrix": ((2, 3), (3.0, -4.5, 8.0, 0.25, -16.0, 2.0)),
    "rank8": ((1, 1, 1, 1, 1, 1, 1, 2), (-0.5, 7.25)),
}


class Failure(Exception):
    pass


class Infrastructure(Exception):
    pass


def expected_bytes(shape, values):
    # '=': host byte order, fixed widths, no alignment. ESKT v1 has no dtype
    # or count field; payload is binary64 and count is the shape product.
    assert math.prod(shape) == len(values)
    return (struct.pack("=III", 0x45534B54, 1, len(shape))
            + struct.pack(f"={len(shape)}q", *shape)
            + struct.pack(f"={len(values)}d", *values))


def verify_bytes(path, expected):
    if not path.is_file():
        raise Failure(f"missing output: {path}")
    actual = path.read_bytes()
    if actual != expected:
        raise Failure(f"exact file bytes differ: {path} "
                      f"(actual {len(actual)}, expected {len(expected)} bytes)")


def run(command, directory, label, env, timeout, marker=False):
    with (directory / f"{label}.stdout").open("w") as out, \
         (directory / f"{label}.stderr").open("w") as err:
        try:
            result = subprocess.run(command, cwd=directory, env=env,
                                    stdout=out, stderr=err, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise Infrastructure(f"{label} timed out after {timeout}s") from exc
    if result.returncode:
        if result.returncode in (-9, -2, 125, 127):
            raise Infrastructure(f"{label} exited {result.returncode}")
        raise Failure(f"{label} exited {result.returncode}")
    if marker:
        lines = (directory / f"{label}.stdout").read_text().splitlines()
        if lines.count("ESKT-PARITY:PASS") != 1 or any("FAIL" in line for line in lines):
            raise Failure(f"{label} semantic checks did not pass")


def matrix(args, work):
    compiler, vm = args.compiler.resolve(), args.vm.resolve()
    for binary in (compiler, vm):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise Infrastructure(f"missing executable: {binary}")
    env = dict(os.environ, ESHKOL_VM_NO_DISASM="1")
    env["ESHKOL_PATH"] = str(ROOT / "lib") + os.pathsep + env.get("ESHKOL_PATH", "")
    aot, bytecode = work / "fixture-aot", work / "fixture.eskb"
    run([str(compiler), "--no-stdlib", str(SOURCE), "-o", str(aot),
         f"-L{compiler.parent}"], work, "compile-aot", env, args.timeout)
    run([str(compiler), "--profile", "hosted-vm", "--emit-eskb", str(bytecode),
         str(SOURCE)], work, "compile-bytecode", env, args.timeout)
    if not aot.is_file() or not bytecode.is_file() or not bytecode.stat().st_size:
        raise Failure("compiler did not create executable/bytecode")
    commands = {
        "jit": [str(compiler), "--no-stdlib", "-r", str(SOURCE), f"-L{compiler.parent}"],
        "aot": [str(aot)],
        "vm-source": [str(vm), str(SOURCE)],
        "vm-bytecode": [str(vm), str(bytecode)],
    }
    expected = {name: expected_bytes(*case) for name, case in CASES.items()}
    for producer in ENGINES:
        directory = work / f"produce-{producer}"
        directory.mkdir()
        run(commands[producer], directory, "produce", dict(env, ESKT_PARITY_MODE="produce"),
            args.timeout, marker=True)
        for name, data in expected.items():
            verify_bytes(directory / f"{name}.eskt", data)
        print(f"PASS: {producer} producer: 3 exact ESKT files", flush=True)

    negative_count = 0
    for consumer in ENGINES:
        directory = work / f"consume-{consumer}"
        directory.mkdir()
        for producer in ENGINES:
            for name in CASES:
                shutil.copyfile(work / f"produce-{producer}" / f"{name}.eskt",
                                directory / f"{producer}-{name}.eskt")
        run(commands[consumer], directory, "consume", dict(env, ESKT_PARITY_MODE="consume"),
            args.timeout, marker=True)
        for producer in ENGINES:
            for name, data in expected.items():
                # Check the input copy is unchanged, as well as the independently
                # reserialized loaded tensor. Neither comparison is text based.
                verify_bytes(directory / f"{producer}-{name}.eskt", data)
                output = directory / f"rewrite-{producer}-{name}.eskt"
                verify_bytes(output, data)
                if args.self_test:
                    # Change ONLY Python's expectation, never a file given to a reader.
                    wrong = data[:-1] + bytes([data[-1] ^ 1])
                    try:
                        verify_bytes(output, wrong)
                    except Failure:
                        negative_count += 1
                    else:
                        raise Failure(f"incorrect expected bytes accepted: {output}")
            print(f"PASS: {producer} -> {consumer}: metadata and 3 exact rewrites", flush=True)
    if args.self_test:
        if negative_count != 48:
            raise Failure(f"incomplete negative controls: {negative_count}/48")
        print("PASS: 48/48 incorrect expected-byte oracles refused (valid files unchanged)")
    print("PASS: ESKT producer/consumer matrix 16/16; 12 producer files, 48 rewrites")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path, nargs="?", default=ROOT / "build/eshkol-run")
    parser.add_argument("vm", type=Path, nargs="?", default=ROOT / "build/eshkol-vm-standalone-test")
    parser.add_argument("--self-test", action="store_true",
                        help="also require refusal of wrong Python byte expectations")
    parser.add_argument("--timeout", type=int, default=120, help="seconds per engine invocation")
    parser.add_argument("--keep", action="store_true", help="retain isolated outputs and logs")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    work = None
    keep = args.keep or bool(os.environ.get("ESHKOL_TEST_KEEP_TMPDIR"))
    try:
        base = os.environ.get("ESHKOL_TEST_TMPDIR") or os.environ.get("ESHKOL_TEST_TMP_ROOT")
        work = Path(tempfile.mkdtemp(prefix="eshkol-eskt-parity-", dir=base)).resolve()
        matrix(args, work)
        return 0
    except Failure as exc:
        keep = True
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except (Infrastructure, OSError) as exc:
        keep = True
        print(f"INFRA: {exc}", file=sys.stderr)
        return 125
    finally:
        if work:
            if keep:
                print(f"ESKT artifacts: {work}", file=sys.stderr)
            else:
                shutil.rmtree(work)


if __name__ == "__main__":
    sys.exit(main())
