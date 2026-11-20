"""Check the source ownership verdict before execution on each engine."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run(command, work, env, source=None):
    return subprocess.run(command, input=source, cwd=work, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)


def main():
    compiler, vm, repl = (Path(arg).resolve() for arg in sys.argv[1:4])
    case = sys.argv[4]
    source, diagnostic = {
        "move": ("(define x (owned 17))\n(move x)\n(display x)\n",
                 "Use of moved value 'x'"),
        "borrow": ("(define x (owned 17))\n(borrow x (borrow x 0) (move x))\n",
                   "Cannot move 'x' while it is borrowed"),
    }[case]
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        env = dict(os.environ, ESHKOL_JIT_CACHE="0", ESHKOL_VM_NO_DISASM="1")
        fixture = work / "ownership.esk"
        fixture.write_text(source)
        out = work / "compiled"
        for command in ([compiler, "-n", "-r", fixture], [compiler, "-n", fixture, "-o", out],
                        [vm, fixture]):
            checked = run(command, work, env)
            if checked.returncode == 0 or diagnostic not in checked.stdout or out.exists():
                raise AssertionError(f"{command}: {checked.returncode}\n{checked.stdout}")
        imported = work / "imported.esk"
        imported.write_text('(load "ownership.esk")\n')
        for command in ([compiler, "-n", "-r", imported], [compiler, "-n", imported, "-o", out],
                        [vm, imported]):
            result = run(command, work, env)
            if result.returncode == 0 or diagnostic not in result.stdout or out.exists():
                raise AssertionError(f"{command}: {result.returncode}\n{result.stdout}")
        checked = run([repl], work, env, source)
        if diagnostic not in checked.stdout:
            raise AssertionError(f"interactive {case}: {checked.stdout}")
        # Both arms transfer the same owner exactly once. Reading a fresh binding
        # that shadows the transferred name is valid afterwards.
        fixture.write_text("(define x (owned 17))\n(if (> x 0) (move x) (move x))\n"
                           '(let ((x 8)) (display x)) (newline)\n')
        for command in ([compiler, "-n", "-r", fixture], [vm, fixture],
                        [compiler, "-n", fixture, "-o", out], [out]):
            result = run(command, work, env)
            executing = "-o" not in command
            if result.returncode or (executing and "8" not in result.stdout.splitlines()):
                raise AssertionError(f"{command}: {result.returncode}\n{result.stdout}")

        fixture.write_text('(define x (owned 17))\n(borrow x (move x))\n'
                           '(display "legacy-ok") (newline)\n')
        legacy = work / "legacy"
        for command in ([compiler, "--unsafe", "-n", "-r", fixture],
                        [vm, "--unsafe", fixture],
                        [compiler, "--unsafe", "-n", fixture, "-o", legacy], [legacy]):
            result = run(command, work, env)
            if result.returncode or ("-o" not in command and
                                     "legacy-ok" not in result.stdout.splitlines()):
                raise AssertionError(f"{command}: {result.returncode}\n{result.stdout}")


if __name__ == "__main__":
    main()
