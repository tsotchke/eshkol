"""Check project module lookup and catchable module conditions through each engine."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def namespace(root, env):
    cwd = root / "project"
    configured = root / "search"
    library = root / "libraries"
    env.update(ESHKOL_PATH=str(configured), ESHKOL_LIB_DIR=str(library))
    # Each project tier carries an empty source with the installed spelling.
    for parent in (cwd, configured, library):
        for module in ("test/modules/dd10_loaded_private.esk", "core/test-module.esk",
                       "test/modules/internal_forward_helper.esk", "core/list/query.esk"):
            write(parent / module, "; project fixture\n")
    write(library / "stdlib.esk", "; project library root\n")
    write(library / "extra/offset.esk",
          "(require sibling)\n(provide offset)\n(define (offset) (unit-value))\n")
    write(library / "extra/sibling.esk",
          "(provide unit-value)\n(define (unit-value) 1)\n")
    write(configured / "local/provider.esk",
          "(require sibling)\n(require extra.offset)\n(provide project-value)\n"
          "(define (project-value) (+ (sibling-value) (offset)))\n")
    write(configured / "local/sibling.esk",
          "(provide sibling-value)\n(define (sibling-value) 30)\n")
    write(cwd / "working.esk",
          "(require local.provider)\n(provide result)\n"
          "(define (result) (+ (project-value) 11))\n")
    return ("(require test.modules.dd10_loaded_private)\n(require working)\n"
            '(load "core/test-module.esk")\n(import "./test/modules/internal_forward_helper.esk")\n'
            '(require "core/list/query.esk")\n'
            "(display (+ (loaded-private) (result)))\n(newline)\n"
            "(display (= (test-add 1 2) 3))\n(newline)\n"
            "(display (= (public-value 1) 42))\n(newline)\n", ["53", "#t", "#t"])



def conditions(root):
    lines = []
    cwd = root / "project"
    (cwd / "local").mkdir()
    write(root / "present.esk", "; test fixture\n")
    write(cwd / "present.esk", "; test fixture\n")
    names = [str(root / "absent.esk"), "local/../absent.esk", "../absent.esk",
             str(root / "present.esk"), "local/../present.esk", "../present.esk"]
    for form, name in zip(("require", "import", "load") * 2, names):
        literal = json.dumps(name)
        lines.append(f'(display (string=? (guard (e (else (error-object-message e))) '
                     f'({form} {literal}) "accepted") {json.dumps("Refused module reference " + name)}))')
        lines.append("(newline)")
    return "\n".join(lines) + "\n", ["#t"] * len(names)


def main():
    case, engine, executable = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="module-resolution-") as directory:
        root = Path(directory)
        cwd = root / "project"
        cwd.mkdir()
        env = dict(os.environ, ESHKOL_JIT_CACHE="0", ESHKOL_VM_NO_DISASM="1")
        if case == "roots":
            write(root / "lib/outside-only.esk", "; unconfigured directory\n")
            source, expected = "(require outside-only)\n", []
        else:
            source, expected = namespace(root, env) if case == "namespace" else conditions(root)
        entry = (cwd if case == "conditions" else root) / "entry.esk"
        write(entry, source)

        def run(args, stdin=None):
            proc = subprocess.run(args, input=stdin, cwd=cwd, env=env, capture_output=True,
                                  text=True, timeout=150)
            if case == "roots":
                diagnostic = (proc.stdout + proc.stderr).lower()
                missing = re.search(r"module(?: source)? not found: outside-only|"
                                    r"module 'outside-only' not found", diagnostic)
                if (engine != "repl" and proc.returncode == 0) or not missing:
                    raise AssertionError(f"{engine} selected an unconfigured root: {diagnostic}")
                return ""
            if proc.returncode:
                raise AssertionError((args, proc.returncode, proc.stdout, proc.stderr))
            return proc.stdout

        if engine == "aot":
            program = root / "program"
            run([executable, str(entry), "-o", str(program)])
            output = "" if case == "roots" else run([str(program)])
        elif engine == "repl":
            if case == "namespace":
                source = source.replace("(require test.modules.dd10_loaded_private)",
                                        ":load test/modules/dd10_loaded_private.esk")
            output = run([executable], source + ":quit\n")
        else:
            args = [executable, "-r"] if engine == "jit" else [executable]
            output = run([*args, str(entry)])
        observed = re.findall(r"(?<![\w.])(?:53|#t|#f)(?![\w.])", output)
        if observed != expected:
            raise AssertionError(f"{engine} expected {expected!r} in {output!r}")


if __name__ == "__main__":
    main()
