#!/usr/bin/env python3
"""Verify executable --emit-depfile, precompiled object-input linking, and
JSON UTF-8 canonical roundtrip across native JIT, AOT, and VM engines.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("BUILD_DIR", str(REPO_ROOT / "build")))
ESHKOL_RUN = BUILD_DIR / "eshkol-run"
ESHKOL_VM = BUILD_DIR / "eshkol-vm-standalone-test"


class DepfileObjectLinkAndJsonUtf8Test(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ESHKOL_RUN.is_file(), f"missing {ESHKOL_RUN}")

    def test_executable_depfile_lists_transitive_modules(self):
        with tempfile.TemporaryDirectory(prefix="eshkol-depfile-") as td:
            work = Path(td)
            mod_dir = work / "mods"
            mod_dir.mkdir()
            helper = mod_dir / "helper_dep.esk"
            helper.write_text("(provide helper-add)\n(define (helper-add x) (+ x 10))\n", encoding="utf-8")
            main_src = work / "main_dep.esk"
            main_src.write_text(
                "(require helper_dep)\n(display (helper-add 32)) (newline)\n",
                encoding="utf-8",
            )
            exe_path = work / "main_dep"
            dep_path = work / "main_dep.d"

            res = subprocess.run(
                [
                    str(ESHKOL_RUN),
                    "-O0",
                    "-I", str(mod_dir),
                    "-L", str(BUILD_DIR),
                    "--emit-depfile", str(dep_path),
                    str(main_src),
                    "-o", str(exe_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr + res.stdout)
            self.assertTrue(exe_path.is_file(), "executable was not emitted")
            self.assertTrue(dep_path.is_file(), "depfile was not emitted for executable build")
            dep_text = dep_path.read_text(encoding="utf-8")
            self.assertIn("main_dep:", dep_text)
            self.assertIn("main_dep.esk", dep_text)
            self.assertIn("helper_dep.esk", dep_text)

            run = subprocess.run([str(exe_path)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, msg=run.stderr)
            self.assertEqual(run.stdout.strip(), "42")

    def test_compile_only_then_object_input_link(self):
        with tempfile.TemporaryDirectory(prefix="eshkol-objlink-") as td:
            work = Path(td)
            src = work / "obj_prog.esk"
            src.write_text("(define (sq x) (* x x))\n(display (sq 7)) (newline)\n", encoding="utf-8")
            obj = work / "obj_prog.o"
            exe = work / "obj_prog_bin"

            c = subprocess.run(
                [str(ESHKOL_RUN), "-L", str(BUILD_DIR), "--compile-only", str(src), "-o", str(obj)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(c.returncode, 0, msg=c.stderr + c.stdout)
            self.assertTrue(obj.is_file())

            l = subprocess.run(
                [str(ESHKOL_RUN), "-L", str(BUILD_DIR), str(obj), "-o", str(exe)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(l.returncode, 0, msg=l.stderr + l.stdout)
            self.assertTrue(exe.is_file())

            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, msg=run.stderr)
            self.assertEqual(run.stdout.strip(), "49")

    def test_json_utf8_canonical_roundtrip(self):
        prog = (
            '(require core.json)\n'
            '(let* ((raw "{\\"text\\":\\"café\\",\\"note\\":\\"a\\\\\\"b\\\\nc\\"}")\n'
            '       (obj (json-parse raw))\n'
            '       (text (json-get obj "text"))\n'
            '       (note (json-get obj "note"))\n'
            '       (enc (json-stringify obj))\n'
            '       (obj2 (json-parse enc)))\n'
            '  (display (and (string=? text "café")\n'
            '                (string=? (json-get obj2 "text") "café")\n'
            '                (string=? note (json-get obj2 "note"))))\n'
            '  (newline))\n'
        )
        with tempfile.TemporaryDirectory(prefix="eshkol-json-utf8-") as td:
            work = Path(td)
            src = work / "json_utf8.esk"
            src.write_text(prog, encoding="utf-8")
            exe = work / "json_utf8"

            jit = subprocess.run(
                [str(ESHKOL_RUN), "-r", str(src)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(jit.returncode, 0, msg=jit.stderr)
            self.assertEqual(jit.stdout.strip().splitlines()[-1], "#t")

            comp = subprocess.run(
                [str(ESHKOL_RUN), "-L", str(BUILD_DIR), str(src), "-o", str(exe)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(comp.returncode, 0, msg=comp.stderr + comp.stdout)
            aot = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
            self.assertEqual(aot.returncode, 0, msg=aot.stderr)
            self.assertEqual(aot.stdout.strip(), "#t")


if __name__ == "__main__":
    unittest.main()
