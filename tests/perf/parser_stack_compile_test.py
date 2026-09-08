#!/usr/bin/env python3
"""Compile the shipped stdlib and execute 16k nesting with an 8 MiB stack.

This gate is independent of ESH-0103's unchanged time/RSS scaling checks.
The hard limit prevents the compiler or a subprocess from enlarging its stack.
"""
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile

STACK_BYTES = 8 * 1024 * 1024
DEPTH = 16000


def limit_stack():
    resource.setrlimit(resource.RLIMIT_STACK, (STACK_BYTES, STACK_BYTES))


def main():
    compiler = Path(sys.argv[1]).resolve()
    root = Path(__file__).resolve().parents[2]
    scratch = root / ".scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="parser-stack-", dir=scratch) as work:
        work = Path(work)
        env = dict(os.environ, ESHKOL_JIT_CACHE_DIR=str(work / "jit-cache"))
        commands = [
            ("stdlib", [str(compiler), "--shared-lib", "-c", "-o",
                        str(work / "stdlib"), "stdlib.esk"], root / "lib"),
        ]
        source = work / "nested.esk"
        source.write_text("(display " + "(+ 1 " * DEPTH + "0" + ")" * DEPTH + ")(newline)\n")
        commands += [
            ("jit", [str(compiler), "-n", "-r", str(source)], root),
            ("aot", [str(compiler), "-n", "-O0", str(source), "-o", str(work / "nested")], root),
            ("aot-run", [str(work / "nested")], root),
        ]
        for name, argv, cwd in commands:
            result = subprocess.run(argv, cwd=cwd, env=env, preexec_fn=limit_stack,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=180)
            if result.returncode:
                print(result.stdout[-16000:], file=sys.stderr)
                raise RuntimeError(f"{name} failed on 8 MiB stack: {result.returncode}")
            if name in ("jit", "aot-run") and str(DEPTH) not in result.stdout.splitlines():
                raise RuntimeError(f"{name} did not produce {DEPTH}: {result.stdout[-2000:]}")
            if name == "stdlib":
                for suffix in (".o", ".bc"):
                    if not (work / ("stdlib" + suffix)).stat().st_size:
                        raise RuntimeError(f"empty stdlib{suffix}")
            print(f"PASS: {name} on 8 MiB process stack", flush=True)


if __name__ == "__main__":
    main()
