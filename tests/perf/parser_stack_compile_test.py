#!/usr/bin/env python3
"""Compile the shipped stdlib and execute 16k nesting with an 8 MiB stack.

This gate is independent of ESH-0103's unchanged time/RSS scaling checks.
The hard limit prevents the compiler or a subprocess from enlarging its stack.
"""
import os
from pathlib import Path
import resource
import shutil
import struct
import subprocess
import sys
import tempfile

STACK_BYTES = 8 * 1024 * 1024
DEPTH = 16000


def limit_stack():
    resource.setrlimit(resource.RLIMIT_STACK, (STACK_BYTES, STACK_BYTES))


def bounded_stack_executable(source, destination):
    """Enforce the same executable stack mapping as Linux on Darwin.

    RLIMIT_STACK alone cannot shrink Darwin's LC_MAIN reservation. Change only
    that load-command field in a test copy, then renew its ad-hoc signature;
    the compiler's instructions, linked libraries and production binary stay
    identical. Generated AOT executables need the same treatment.
    """
    if sys.platform != "darwin":
        return source
    shutil.copy2(source, destination)
    data = bytearray(destination.read_bytes())
    if struct.unpack_from("<I", data)[0] != 0xFEEDFACF:
        raise RuntimeError("stack gate requires a native 64-bit Mach-O executable")
    commands = struct.unpack_from("<I", data, 16)[0]
    offset = 32
    for _ in range(commands):
        command, size = struct.unpack_from("<II", data, offset)
        if command == 0x80000028:  # LC_MAIN: cmd, cmdsize, entryoff, stacksize
            struct.pack_into("<Q", data, offset + 16, STACK_BYTES)
            destination.write_bytes(data)
            subprocess.run(["codesign", "--force", "--sign", "-", str(destination)],
                           check=True, capture_output=True)
            return destination
        offset += size
    raise RuntimeError("Mach-O executable has no LC_MAIN stack reservation")


def main():
    compiler = Path(sys.argv[1]).resolve()
    root = Path(__file__).resolve().parents[2]
    scratch = root / ".scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="parser-stack-", dir=scratch) as work:
        work = Path(work)
        compiler = bounded_stack_executable(compiler, work / "compiler-8m")
        # A private directory alone still permits -r to build an AOT cache
        # entry. Disable that shortcut so this case exercises ORC in process.
        env = dict(os.environ, ESHKOL_JIT_CACHE="0",
                   ESHKOL_JIT_CACHE_DIR=str(work / "jit-cache"))
        commands = [
            ("stdlib", [str(compiler), "--shared-lib", "-c", "-o",
                        str(work / "stdlib"), "stdlib.esk"], root / "lib"),
        ]
        source = work / "nested.esk"
        source.write_text("(display " + "(+ 1 " * DEPTH + "0" + ")" * DEPTH + ")(newline)\n")
        # Alternate left/right nested noncommutative calls and keep observable
        # calls on both sides of the continuation chain. This catches a driver
        # that reverses child evaluation or loses a suspended operand.
        ordered = work / "ordered.esk"
        prefixes, suffixes = [], []
        for i in range(DEPTH):
            prefixes.append("(+ 1 " if i % 2 == 0 else "(- ")
            suffixes.append(")" if i % 2 == 0 else " 1)")
        ordered.write_text(
            "(define (mark x) (display x) (newline) x)\n"
            "(display (- (mark 9) " + "".join(prefixes) + "(mark 4)" +
            "".join(reversed(suffixes)) + "))(newline)\n")
        commands += [
            ("jit", [str(compiler), "-n", "-r", str(source)], root),
            ("aot", [str(compiler), "-n", "-O0", str(source), "-o", str(work / "nested")], root),
            ("aot-run", [str(work / "nested")], root),
            ("jit-order", [str(compiler), "-n", "-r", str(ordered)], root),
            ("aot-order", [str(compiler), "-n", "-O0", str(ordered), "-o",
                           str(work / "ordered")], root),
            ("aot-order-run", [str(work / "ordered")], root),
        ]
        for name, argv, cwd in commands:
            if name in ("aot-run", "aot-order-run"):
                executable = Path(argv[0])
                argv[0] = str(bounded_stack_executable(
                    executable, work / (executable.name + "-8m")))
            result = subprocess.run(argv, cwd=cwd, env=env, preexec_fn=limit_stack,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=180)
            if result.returncode:
                print(result.stdout[-16000:], file=sys.stderr)
                raise RuntimeError(f"{name} failed on 8 MiB stack: {result.returncode}")
            if name in ("jit", "aot-run") and str(DEPTH) not in result.stdout.splitlines():
                raise RuntimeError(f"{name} did not produce {DEPTH}: {result.stdout[-2000:]}")
            if name in ("jit-order", "aot-order-run"):
                numeric_lines = [line for line in result.stdout.splitlines()
                                 if line.strip().lstrip("-").isdigit()]
                if numeric_lines != ["9", "4", "5"]:
                    raise RuntimeError(f"{name} lost operand order: {result.stdout[-2000:]}")
            if name == "stdlib":
                for suffix in (".o", ".bc"):
                    if not (work / ("stdlib" + suffix)).stat().st_size:
                        raise RuntimeError(f"empty stdlib{suffix}")
            print(f"PASS: {name} on 8 MiB process stack", flush=True)


if __name__ == "__main__":
    main()
