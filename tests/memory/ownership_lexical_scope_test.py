#!/usr/bin/env python3
"""Exercise lexical ownership identities in native JIT and AOT paths."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run(command, work, env):
    return subprocess.run(command, cwd=work, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=90)


def fails(compiler, source, work, env, text):
    output = work / (source.stem + "-bad")
    result = run([str(compiler), "-n", "-O0", str(source), "-o", str(output)], work, env)
    if result.returncode != 1 or text not in result.stdout or output.exists():
        raise RuntimeError(f"{source.name}: expected failing compile with {text!r}; "
                           f"exit={result.returncode}\n{result.stdout[-4000:]}")


def main():
    compiler = Path(sys.argv[1]).resolve()
    fixtures = Path(sys.argv[2]).resolve()
    with tempfile.TemporaryDirectory(prefix="ownership-lexical-") as directory:
        work = Path(directory)
        env = os.environ.copy()
        env["ESHKOL_JIT_CACHE_DIR"] = str(work / "jit-cache")
        shadow = fixtures / "ownership_lexical_shadow_move.esk"
        jit = run([str(compiler), "-n", "-r", "-O0", str(shadow)], work, env)
        if jit.returncode or jit.stdout.strip() != "1":
            raise RuntimeError(f"shadow JIT failed: {jit.returncode}\n{jit.stdout[-4000:]}")
        executable = work / "shadow-aot"
        compiled = run([str(compiler), "-n", "-O0", str(shadow), "-o", str(executable)], work, env)
        if compiled.returncode:
            raise RuntimeError(f"shadow AOT compile failed:\n{compiled.stdout[-4000:]}")
        aot = run([str(executable)], work, env)
        if aot.returncode or aot.stdout.strip() != "1":
            raise RuntimeError(f"shadow AOT run failed: {aot.returncode}\n{aot.stdout[-4000:]}")

        fails(compiler, fixtures / "use_after_move.esk", work, env, "Use of moved value 'x'")
        fails(compiler, fixtures / "ownership_parallel_initializer_move_fail.esk", work, env,
              "Double move of 'x'")
        fails(compiler, fixtures / "ownership_initializer_outer_use_fail.esk", work, env,
              "Use of moved value 'x'")
        fails(compiler, fixtures / "ownership_nested_borrow_move_fail.esk", work, env,
              "Cannot move 'x' while it is borrowed")

        star_source = work / "let-star.esk"
        star_source.write_text("(display (let* ((x 1) (y x)) y)) (newline)\n")
        star = run([str(compiler), "-n", "-r", "-O0", str(star_source)], work, env)
        if star.returncode or star.stdout.strip() != "1":
            raise RuntimeError(f"let* visibility changed: {star.returncode}\n{star.stdout[-4000:]}")
        shadow_star = fixtures / "ownership_let_star_shadow_move.esk"
        sequential_move = run([str(compiler), "-n", "-r", "-O0", str(shadow_star)], work, env)
        if sequential_move.returncode or sequential_move.stdout.strip() != "1":
            raise RuntimeError(f"let* shadow move failed: {sequential_move.returncode}\n"
                               f"{sequential_move.stdout[-4000:]}")
        borrow_shadow = fixtures / "ownership_borrow_shadow_move.esk"
        for mode in ("jit", "aot"):
            if mode == "jit":
                command = [str(compiler), "-n", "-r", "-O0", str(borrow_shadow)]
            else:
                binary = work / "borrow-shadow-aot"
                built = run([str(compiler), "-n", "-O0", str(borrow_shadow),
                             "-o", str(binary)], work, env)
                if built.returncode:
                    raise RuntimeError(f"borrow shadow AOT compile failed:\n{built.stdout[-4000:]}")
                command = [str(binary)]
            result = run(command, work, env)
            if result.returncode or result.stdout.strip() != "3\n1\n1":
                raise RuntimeError(f"borrow shadow {mode} failed: {result.returncode}\n"
                                   f"{result.stdout[-4000:]}")
    print("PASS: lexical shadow move JIT/AOT, outer and initializer moves, nested borrow, let* visibility")


if __name__ == "__main__":
    main()
