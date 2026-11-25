"""Check the source ownership verdict before execution on each engine."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run(command, work, env, source=None):
    return subprocess.run(command, input=source, cwd=work, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)


ACCEPTED = (
    # An alias read before the move, a procedure that reads the owner before
    # the move, a borrowed owner passed to a procedure that does not keep it,
    # and a loop whose variables carry the alias.
    "(define x (owned (list 1 2)))\n"
    "(define y (cons x '()))\n"
    "(define (peek) (car x))\n"
    "(define (show v) (display v) (newline))\n"
    "(show (car (car y)))\n"
    "(show (peek))\n"
    "(borrow x (show (cadr x)))\n"
    "(show (let loop ((acc '()) (n 2)) (if (= n 0) (car (car (car acc))) (loop (cons y acc) (- n 1)))))\n"
    "(define z (move x))\n"
    "(show (car z))\n"
)


def accepted():
    compiler, vm, repl = (Path(arg).resolve() for arg in sys.argv[1:4])
    expected = ["1", "1", "2", "1", "1"]
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        env = dict(os.environ, ESHKOL_JIT_CACHE="0", ESHKOL_VM_NO_DISASM="1")
        fixture = work / "accepted.esk"
        fixture.write_text(ACCEPTED)
        out = work / "accepted"
        for command in ([compiler, "-n", "-r", fixture], [vm, fixture],
                        [compiler, "-n", fixture, "-o", out], [out]):
            result = run(command, work, env)
            lines = [line.strip() for line in result.stdout.splitlines()
                     if line.strip() and not line.startswith(("===", "[WARN]"))]
            if result.returncode or ("-o" not in command and lines[-5:] != expected):
                raise AssertionError(f"{command}: {result.returncode}\n{result.stdout}")
        checked = run([repl], work, env, ACCEPTED)
        if "ownership" in checked.stdout.lower() or "moved value" in checked.stdout:
            raise AssertionError(f"interactive accepted: {checked.stdout}")


def loaded_alias():
    # The owner and its alias live in a loaded file; the loading program moves
    # the owner and reads the alias. Every engine checks the loaded file as
    # part of the program that loads it.
    compiler, vm, repl = (Path(arg).resolve() for arg in sys.argv[1:4])
    diagnostic = "Use of 'y', which refers to moved value 'x'"
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        env = dict(os.environ, ESHKOL_JIT_CACHE="0", ESHKOL_VM_NO_DISASM="1")
        (work / "owner.esk").write_text("(define x (owned (list 1 2)))\n(define y (cons x '()))\n")
        program = work / "program.esk"
        program.write_text('(load "owner.esk")\n(define z (move x))\n(display (car y))\n')
        out = work / "compiled"
        for command in ([compiler, "-n", "-r", program], [compiler, "-n", program, "-o", out],
                        [vm, program]):
            result = run(command, work, env)
            if result.returncode == 0 or diagnostic not in result.stdout or out.exists():
                raise AssertionError(f"{command}: {result.returncode}\n{result.stdout}")
        checked = run([repl], work, env, program.read_text())
        if diagnostic not in checked.stdout:
            raise AssertionError(f"interactive loaded_alias: {checked.stdout}")


def main():
    compiler, vm, repl = (Path(arg).resolve() for arg in sys.argv[1:4])
    case = sys.argv[4]
    source, diagnostic = {
        "move": ("(define x (owned 17))\n(move x)\n(display x)\n",
                 "Use of moved value 'x'"),
        "borrow": ("(define x (owned 17))\n(borrow x (borrow x 0) (move x))\n",
                   "Cannot move 'x' while it is borrowed"),
        "joined_borrow": ("(define x (owned 17))\n(if (> x 1) (move x) 0)\n(borrow x 0)\n",
                          "Cannot borrow moved value 'x'"),
        "borrow_result": ("(define x (owned 17))\n(define y (borrow x (begin 0 (the integer x))))\n",
                          "Borrowed value 'x' escapes its borrow scope"),
        # A structure that holds the owned value outlives its move.
        "alias_data": ("(define x (owned (list 1 2)))\n(define y (cons x '()))\n"
                       "(define z (move x))\n(display (car y))\n",
                       "Use of 'y', which refers to moved value 'x'"),
        # A procedure that reads the owner is called after the move.
        "alias_closure": ("(define x (owned (list 1 2)))\n(define (peek) (car x))\n"
                          "(define z (move x))\n(display (peek))\n",
                          "Use of 'peek', which refers to moved value 'x'"),
        # A closure keeps a local alias after its scope ends.
        "closure_let_alias": ("(define x (owned (list 1 2)))\n"
                              "(define get (let ((alias (list x))) (lambda () (car alias))))\n"
                              "(define z (move x))\n(display (get))\n",
                              "Use of 'get', which refers to moved value 'x'"),
        "borrow_closure": ("(define x (owned (list 1 2)))\n(define f (borrow x (lambda () (car x))))\n",
                           "Borrowed value 'x' escapes its borrow scope"),
        "escape_set": ("(define keep '())\n(define x (owned (list 1 2)))\n(borrow x (set! keep (list x)))\n",
                       "Assignment to 'keep' would keep a reference to owned value 'x'"),
        "escape_call": ("(define cache '())\n(define (remember! v) (set! cache v))\n"
                        "(define x (owned (list 1 2)))\n(borrow x (remember! x))\n",
                        "'remember!' may keep a reference to owned value 'x'"),
        "escape_hof": ("(define cache '())\n(define x (owned (list 1 2)))\n"
                       "(for-each (lambda (e) (set! cache e)) (list x))\n",
                       "Assignment to 'cache' would keep a reference to owned value 'x'"),
        "escape_unknown": ("(define x (owned (list 1 2)))\n(define procs (list car))\n"
                           "(borrow x ((car procs) x))\n",
                           "Owned value 'x' is passed to a procedure the ownership checker cannot resolve"),
        "captured_move": ("(define x (owned (list 1 2)))\n(define (take) (move x))\n(take)\n(take)\n",
                          "Cannot move 'x' inside a procedure or loop body"),
        "loop_move": ("(define x (owned (list 1 2)))\n"
                      "(let loop ((i 0)) (if (< i 2) (begin (move x) (loop (+ i 1))) i))\n",
                      "Cannot move 'x' inside a procedure or loop body"),
        "unfollowed": ("(define x (owned (list 1 2)))\n(define y (case 1 ((1) x) (else 0)))\n",
                       "The ownership checker does not follow case"),
    }.get(case, (None, None))
    if case == "accepted":
        accepted()
        return
    if case == "loaded_alias":
        loaded_alias()
        return
    if source is None:
        raise SystemExit(f"unknown case {case}")
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
