"""The required VM output gate must reject false greens and incomplete runs."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_curvature_vm_output.py"
SPEC = importlib.util.spec_from_file_location("curvature_verdict", SCRIPT)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class CurvatureVmOutputTest(unittest.TestCase):
    def test_all_fixture_contracts_and_corruptions(self):
        for fixture, (count, style) in GATE.CONTRACTS.items():
            lines = [f"PASS: check {i}" if style == "prefix" else f"check {i}: PASS"
                     for i in range(count)]
            with self.subTest(fixture=fixture, case="complete"):
                GATE.validate(fixture, "\n".join(lines) + "\n=== Execution complete ===\n")
            corruptions = {
                "empty": "",
                "missing": "\n".join(lines[:-1]),
                "extra": "\n".join(lines + ["PASS: extra" if style == "prefix" else "extra: PASS"]),
                "duplicate": "\n".join(lines + [lines[0]]),
                "prefix_failure": "\n".join(lines + ["FAIL: assertion"]),
                "suffix_failure": "\n".join(lines + ["assertion: FAIL"]),
                "error": "\n".join(lines + ["ERROR: runtime"]),
                "native_call": "\n".join(lines + ["unhandled native call"]),
                "signal": "\n".join(lines + ["fatal signal"]),
                "mixed_style": "\n".join(lines + ["extra: PASS" if style == "prefix" else "PASS: extra"]),
            }
            for case, text in corruptions.items():
                with self.subTest(fixture=fixture, case=case):
                    with self.assertRaises(ValueError):
                        GATE.validate(fixture, text)


if __name__ == "__main__":
    unittest.main()
