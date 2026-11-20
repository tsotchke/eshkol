#!/usr/bin/env python3
"""Negative controls for the native JIT probe's process and transcript checks."""

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_native_jit_regressions import (  # noqa: E402
    classify_case, compilation_succeeded, run_command, sanitize, transcript_passes,
)


class NativeJitHarnessTests(unittest.TestCase):
    def execute(self, source, timeout=5):
        return run_command([sys.executable, "-c", source], cwd=ROOT,
                           env=os.environ.copy(), timeout=timeout)

    def test_exact_transcript_required(self):
        self.assertTrue(transcript_passes(self.execute("print(42)"), "42\n"))
        for source in ("print('()')", "print('PASS: 42')", "print('42 ')",
                       "print(42); print('extra')"):
            with self.subTest(source=source):
                self.assertFalse(transcript_passes(self.execute(source), "42\n"))

    def test_error_on_stderr_is_not_success(self):
        result = self.execute("import sys; print(42); print('\\x1b[31m ERROR: workers not registered\\x1b[0m', file=sys.stderr)")
        self.assertFalse(transcript_passes(result, "42\n"))

    def test_nonzero_exit_with_correct_output_fails(self):
        result = self.execute("print(42); raise SystemExit(7)")
        self.assertEqual(result["exit_code"], 7)
        self.assertFalse(transcript_passes(result, "42\n"))

    def test_timeout_is_retained(self):
        result = self.execute("import time; print('partial', flush=True); time.sleep(5)", timeout=0.5)
        self.assertEqual(result["status"], "timeout")
        self.assertFalse(transcript_passes(result, ""))

    def test_launch_failure_is_inconclusive(self):
        result = run_command([str(ROOT / "no-such-jit-probe-runner")], cwd=ROOT,
                             env=os.environ.copy(), timeout=1)
        self.assertEqual(result["status"], "launch_error")
        compile_result = self.execute("print('compiler progress')")
        correct = self.execute("print(42)")
        self.assertTrue(compilation_succeeded(compile_result))
        self.assertEqual(classify_case(compile_result, correct, result, "42\n"),
                         "inconclusive_jit_launch_failure")

    def test_failed_aot_control_cannot_confirm_jit_regression(self):
        compile_result = self.execute("raise SystemExit(1)")
        wrong = self.execute("print('()')")
        self.assertEqual(classify_case(compile_result, None, wrong, "42\n"),
                         "inconclusive_compile_failure")
        good_compile = self.execute("pass")
        self.assertEqual(classify_case(good_compile, wrong, wrong, "42\n"),
                         "inconclusive_aot_failure")
        correct = self.execute("print(42)")
        self.assertEqual(classify_case(good_compile, correct, wrong, "42\n"),
                         "jit_regression")

    def test_report_paths_are_sanitized_recursively(self):
        result = sanitize({"command": ["C:\\private\\run.exe"],
                           "stderr": "at C:/private/run.exe"},
                          [("C:\\private\\run.exe", "<runner>")])
        self.assertEqual(result, {"command": ["<runner>"], "stderr": "at <runner>"})


if __name__ == "__main__":
    unittest.main()
