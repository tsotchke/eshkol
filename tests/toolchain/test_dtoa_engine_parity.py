#!/usr/bin/env python3
"""End-to-end JIT, AOT, and VM parity verification for binary64 shortest
round-trip rendering (dtoa_shortest.h), including asymmetric normal powers of
two (such as +/-2^-44), subnormals, extremes, signed zero, and non-finite forms.
"""
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(os.environ.get("BUILD_DIR", str(REPO_ROOT / "build")))
ESHKOL_RUN = BUILD_DIR / "eshkol-run"
ESHKOL_VM = BUILD_DIR / "eshkol-vm-standalone-test"


def _build_corpus():
    vals = [
        0.0, -0.0, 1.0, -1.0, 0.1, 0.2, 0.3, 1.0 / 3.0, math.pi, math.e,
        math.ldexp(1.0, -44), -math.ldexp(1.0, -44),
        math.ldexp(1.0, -53), -math.ldexp(1.0, -53),
        math.ldexp(1.0, -1074), -math.ldexp(1.0, -1074),  # DBL_TRUE_MIN
        math.ldexp(1.0, -1022), -math.ldexp(1.0, -1022),  # DBL_MIN
        1.7976931348623157e+308, -1.7976931348623157e+308,  # DBL_MAX
    ]
    # All powers of two across representative exponents including the 92 asymmetric
    # cases near normal powers of two.
    for exp in range(-1021, 1024, 13):
        v = math.ldexp(1.0, exp)
        vals.append(v)
        vals.append(-v)
    return vals


def _extract_lines(stdout: str):
    lines = []
    for line in stdout.splitlines():
        if line.startswith("VAL:"):
            lines.append(line[4:])
    return lines


class DtoaEngineParityTest(unittest.TestCase):
    def test_jit_aot_vm_shortest_roundtrip_parity(self):
        self.assertTrue(ESHKOL_RUN.is_file(), f"missing {ESHKOL_RUN}")
        self.assertTrue(ESHKOL_VM.is_file(), f"missing {ESHKOL_VM}")
        vals = _build_corpus()

        forms = []
        for v in vals:
            if v == 0.0 and math.copysign(1.0, v) < 0:
                expr = "(- 0.0)"
            else:
                expr = repr(v)
            forms.append(
                f'(let* ((x {expr}) (s (number->string x))) '
                f'(display "VAL:") (display s) (newline))'
            )
        forms.append('(display "VAL:") (display (number->string (/ 1.0 0.0))) (newline)')
        forms.append('(display "VAL:") (display (number->string (/ -1.0 0.0))) (newline)')
        forms.append('(display "VAL:") (display (number->string (/ 0.0 0.0))) (newline)')

        with tempfile.TemporaryDirectory(prefix="eshkol-dtoa-parity-") as td:
            work = Path(td)
            src = work / "dtoa_parity.esk"
            exe = work / "dtoa_parity"
            src.write_text("\n".join(forms) + "\n", encoding="utf-8")

            jit = subprocess.run(
                [str(ESHKOL_RUN), "-r", str(src)],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(jit.returncode, 0, msg=jit.stderr)
            jit_lines = _extract_lines(jit.stdout)

            comp = subprocess.run(
                [str(ESHKOL_RUN), "-L", str(BUILD_DIR), str(src), "-o", str(exe)],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(comp.returncode, 0, msg=comp.stderr + comp.stdout)
            aot = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(aot.returncode, 0, msg=aot.stderr)
            aot_lines = _extract_lines(aot.stdout)

            vm = subprocess.run(
                [str(ESHKOL_VM), str(src)],
                cwd=REPO_ROOT,
                env=dict(os.environ, ESHKOL_VM_NO_DISASM="1"),
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(vm.returncode, 0, msg=vm.stderr + vm.stdout)
            vm_lines = _extract_lines(vm.stdout)

            expected_count = len(vals) + 3
            self.assertEqual(len(jit_lines), expected_count)
            self.assertEqual(jit_lines, aot_lines, "JIT vs AOT dtoa mismatch")
            self.assertEqual(jit_lines, vm_lines, "JIT vs VM dtoa mismatch")

            # Verify exact IEEE-754 bit round-trip and shortest significant digits
            # against Python's Ryu/David-Gay shortest repr for every finite value.
            for idx, v in enumerate(vals):
                s = jit_lines[idx]
                parsed = float(s)
                self.assertEqual(
                    struct.pack("!d", parsed),
                    struct.pack("!d", v),
                    f"bit mismatch at idx={idx} v={v!r} rendered={s!r}",
                )
                # Ensure inexact spelling ('.' or 'e'/'E') is preserved.
                self.assertTrue("." in s or "e" in s or "E" in s, f"lost inexactness: {s!r}")
                # Check significant digits <= Python's shortest repr significant digits.
                def sig_digits(tok: str) -> int:
                    mant = tok.lower().split("e")[0].lstrip("+-").replace(".", "").strip("0")
                    return max(1, len(mant))
                self.assertLessEqual(
                    sig_digits(s),
                    sig_digits(repr(v)),
                    f"excess significant digits for {v!r}: eshkol={s!r} python={repr(v)!r}",
                )

            self.assertEqual(jit_lines[-3:], ["+inf.0", "-inf.0", "+nan.0"])


if __name__ == "__main__":
    unittest.main()
