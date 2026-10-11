#!/usr/bin/env python3
"""Portable negative controls for the Windows release REPL smoke helper."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from repl_smoke import ReplSmokeError, verify_repl_smoke


class ReplSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="repl-smoke-test-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_fixture(self, body: str, *, timeout: float = 2.0) -> None:
        fixture = self.root / "fake_repl.py"
        fixture.write_text(
            "import sys\n"
            "input_text = sys.stdin.read()\n"
            "if input_text != '(+ 1 2)\\n':\n"
            "    print('unexpected input: ' + repr(input_text), file=sys.stderr)\n"
            "    raise SystemExit(99)\n"
            + body,
            encoding="utf-8",
        )
        verify_repl_smoke([sys.executable, str(fixture)], timeout_seconds=timeout)

    def test_success_requires_input_and_machine_framing(self) -> None:
        self.run_fixture(
            "print('3', end='')\n"
            "print('EREPL READY', file=sys.stderr)\n"
            "print('EREPL DONE', file=sys.stderr)\n"
        )

    def test_one_final_newline_is_allowed(self) -> None:
        self.run_fixture(
            "print('3')\n"
            "print('EREPL READY', file=sys.stderr)\n"
            "print('EREPL DONE', file=sys.stderr)\n"
        )

    def test_registry_failure_keeps_jit_diagnostics(self) -> None:
        with self.assertRaises(ReplSmokeError) as raised:
            self.run_fixture(
                "print('Failed to create LLJIT: no targets registered', file=sys.stderr)\n"
                "print('EREPL FAIL', file=sys.stderr)\n"
                "raise SystemExit(1)\n"
            )
        self.assertIn("no targets registered", raised.exception.stderr)
        self.assertIn("EREPL FAIL", raised.exception.stderr)

    def test_wrong_result_fails_even_with_valid_frames(self) -> None:
        with self.assertRaisesRegex(ReplSmokeError, "expected exactly '3'"):
            self.run_fixture(
                "print('4', end='')\n"
                "print('EREPL READY', file=sys.stderr)\n"
                "print('EREPL DONE', file=sys.stderr)\n"
            )

    def test_extra_stdout_is_rejected(self) -> None:
        with self.assertRaises(ReplSmokeError):
            self.run_fixture(
                "print('3\\n')\n"
                "print('EREPL READY', file=sys.stderr)\n"
                "print('EREPL DONE', file=sys.stderr)\n"
            )

    def test_missing_done_frame_fails_with_streams(self) -> None:
        with self.assertRaises(ReplSmokeError) as raised:
            self.run_fixture(
                "print('3', end='')\n"
                "print('EREPL READY', file=sys.stderr)\n"
                "print('startup detail', file=sys.stderr)\n"
            )
        self.assertIn("EREPL READY", raised.exception.stderr)
        self.assertIn("startup detail", raised.exception.stderr)
        self.assertEqual(raised.exception.stdout, "3")

    def test_timeout_is_bounded_and_reports_partial_output(self) -> None:
        with self.assertRaises(ReplSmokeError) as raised:
            self.run_fixture(
                "print('partial stdout', flush=True)\n"
                "print('partial stderr', file=sys.stderr, flush=True)\n"
                "import time; time.sleep(10)\n",
                # Allow Python startup on a busy Windows runner before the
                # intentional ten-second stall exercises timeout capture.
                timeout=2.0,
            )
        self.assertIn("timed out", str(raised.exception))
        self.assertIn("partial stdout", raised.exception.stdout)
        self.assertIn("partial stderr", raised.exception.stderr)

    def test_fail_frame_rejects_an_otherwise_correct_result(self) -> None:
        with self.assertRaisesRegex(ReplSmokeError, "emitted EREPL FAIL"):
            self.run_fixture(
                "print('3', end='')\n"
                "print('EREPL READY', file=sys.stderr)\n"
                "print('EREPL FAIL', file=sys.stderr)\n"
                "print('EREPL DONE', file=sys.stderr)\n"
            )

    def test_done_before_ready_is_not_a_success(self) -> None:
        with self.assertRaisesRegex(ReplSmokeError, "DONE after READY"):
            self.run_fixture(
                "print('3', end='')\n"
                "print('EREPL DONE', file=sys.stderr)\n"
                "print('EREPL READY', file=sys.stderr)\n"
            )


if __name__ == "__main__":
    unittest.main()
