"""The AOT harness must reject zero-exit false greens and failed compiles."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == "posix" and shutil.which("cmake"), "POSIX fake-executable fixture")
class StdlibAotRunnerTest(unittest.TestCase):
    def invoke(self, output, compile_status=0, compile_output="", missing_artifact=None, verdict=None):
        with tempfile.TemporaryDirectory(prefix="aot controls with spaces ") as temporary:
            root = Path(temporary)
            compiler = root / "fake compiler"
            invocation_marker = root / "compiler-called"
            program = "#!/bin/sh\nprintf '%s\\n' " + "'" + output.replace("'", "'\\''") + "'\n"
            source = ("#!/usr/bin/env python3\nimport pathlib,sys\n"
                      f"pathlib.Path({str(invocation_marker)!r}).touch()\n"
                      f"print({compile_output!r})\n")
            source += f"sys.exit({compile_status})\n" if compile_status else (
                "p=pathlib.Path(sys.argv[sys.argv.index('-o')+1])\n"
                f"p.write_text({program!r})\np.chmod(0o700)\n"
            )
            compiler.write_text(source)
            compiler.chmod(0o700)
            for artifact in ("stdlib.o", "stdlib.bc"):
                if artifact != missing_artifact:
                    (root / artifact).write_text("fixture")
            runner = Path(__file__).resolve().parents[2] / "scripts/run_stdlib_aot.cmake"
            command = ["cmake", f"-DESHKOL_RUN={compiler}", f"-DESHKOL_SOURCE={root / 'input.esk'}",
                       f"-DESHKOL_BINARY={root / 'output program'}", f"-DESHKOL_LINK_DIR={root}"]
            if verdict is not None:
                command.append(f"-DESHKOL_VERDICT={verdict}")
            result = subprocess.run(
                command + ["-P", str(runner)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20,
            )
            result.compiler_called = invocation_marker.exists()
            return result

    def test_passing_verdict_and_paths_with_spaces(self):
        result = self.invoke("PASS: fixture\nRESULT: PASS")
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_missing_verdict_is_rejected(self):
        self.assertNotEqual(self.invoke("hello").returncode, 0)

    def test_failure_overrides_pass_and_zero_exit(self):
        self.assertNotEqual(self.invoke("FAIL: fixture\nRESULT: PASS").returncode, 0)

    def test_failed_compilation_is_rejected(self):
        self.assertNotEqual(self.invoke("RESULT: PASS", compile_status=7).returncode, 0)

    def test_failed_compiler_diagnostic_overrides_zero_exit(self):
        self.assertNotEqual(self.invoke("RESULT: PASS", compile_output="LLVM module verification failed").returncode, 0)

    def test_catalogue_verdict_requires_explicit_selection(self):
        self.assertNotEqual(self.invoke("RESULT: ALL PASS").returncode, 0)
        self.assertEqual(self.invoke("RESULT: ALL PASS", verdict="ALL PASS").returncode, 0)

    def test_unrecognized_verdict_cannot_relax_acceptance(self):
        self.assertNotEqual(self.invoke("anything", verdict=".*").returncode, 0)


    def test_missing_local_stdlib_object_fails_before_compiler(self):
        result = self.invoke("RESULT: PASS", missing_artifact="stdlib.o")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing system fallback", result.stdout)
        self.assertFalse(result.compiler_called)

    def test_missing_local_stdlib_bitcode_fails_before_compiler(self):
        result = self.invoke("RESULT: PASS", missing_artifact="stdlib.bc")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing system fallback", result.stdout)
        self.assertFalse(result.compiler_called)


if __name__ == "__main__":
    unittest.main()
