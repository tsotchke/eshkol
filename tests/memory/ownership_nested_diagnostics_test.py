#!/usr/bin/env python3
"""Keep nested ownership diagnostics active through iterative AST traversal."""

from pathlib import Path
import subprocess
import sys
import tempfile


DEPTH = 256


def nested(expression):
    return "(begin 0 " * DEPTH + expression + ")" * DEPTH


def expect_diagnostic(compiler, work, name, source_text, expected):
    source = work / f"{name}.esk"
    source.write_text(source_text)
    result = subprocess.run(
        [str(compiler), "-n", "-O0", str(source), "-o", str(work / name)],
        cwd=work,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    if result.returncode == 0 or expected not in result.stdout:
        raise RuntimeError(
            f"{name}: expected {expected!r} with a failing compile; "
            f"exit={result.returncode}\n{result.stdout[-4000:]}"
        )


def main():
    compiler = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="ownership-nested-") as directory:
        work = Path(directory)
        expect_diagnostic(
            compiler,
            work,
            "use_after_move",
            "(define x (owned (cons 1 2)))\n"
            "(define y (move x))\n"
            f"{nested('x')}\n",
            "Use of moved value 'x'",
        )
        expect_diagnostic(
            compiler,
            work,
            "move_while_borrowed",
            "(define x (owned (cons 1 2)))\n"
            "(borrow x\n"
            f"  {nested('(move x)')}\n"
            ")\n",
            "Cannot move 'x' while it is borrowed",
        )
    print("PASS: nested moved-value and borrowed-move diagnostics")


if __name__ == "__main__":
    main()
