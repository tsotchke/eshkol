"""Check the source-level arities and reducer-name requirement on each engine."""
import subprocess
import sys
from pathlib import Path


def main(engine, runner, build_dir):
    build = Path(build_dir).resolve()
    work = build / "seeded_reduce_syntax" / engine
    work.mkdir(parents=True, exist_ok=True)
    cases = [
        ("short", "(tensor-reduce #(1) +)", "Arity mismatch:"),
        ("extra-axis", "(tensor-reduce #(1) + 0 0 0)", "Arity mismatch:"),
        ("full-short", "(tensor-reduce-all #(1) +)", "Arity mismatch:"),
        ("full-axis", "(tensor-reduce-all #(1) + 0 0)", "Arity mismatch:"),
        ("procedure-expression", "(tensor-reduce #(1) (lambda (a b) (+ a b)) 0)",
         "simple function names"),
    ]
    for name, form, diagnostic in cases:
        source = work / (name + ".esk")
        source.write_text(form + "\n", encoding="utf-8")
        command = [runner]
        if engine == "jit":
            command.append("-r")
        elif engine == "aot":
            command.extend(["-o", str(work / name)])
        command.append(str(source))
        if engine != "vm":
            command.append("-L" + str(build))
        result = subprocess.run(command, cwd=work, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=90)
        if result.returncode <= 0 or diagnostic not in result.stdout:
            raise SystemExit(f"{engine} {name} returned {result.returncode}\n{result.stdout}")
    print(f"PASS {engine} reduction syntax ({len(cases)} cases)")


if __name__ == "__main__":
    main(*sys.argv[1:])
