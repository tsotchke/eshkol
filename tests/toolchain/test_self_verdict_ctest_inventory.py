"""Exercise expectation-aware auditing against real CTest verdicts."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_self_verdicts as audit


class CTestExpectations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmake = shutil.which("cmake")
        cls.ctest = shutil.which("ctest")
        if not cls.cmake or not cls.ctest:
            raise RuntimeError("CTest inventory controls require cmake and ctest")
        scratch = ROOT / ".scratch"
        scratch.mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="self-verdict-ctest-", dir=scratch)
        cls.work = Path(cls.temporary.name)
        cls.build = cls.work / "build"
        cls.serial = 0
        (cls.work / "probe.py").write_text(
            "import sys\n"
            "mode = sys.argv[1]\n"
            "if mode in ('expected_failure', 'false_green'):\n"
            "    print('FAIL: deliberate fixture rejection')\n"
            "else:\n"
            "    print('RESULT: ALL PASS')\n"
            "sys.exit(1 if mode == 'expected_failure' else 0)\n"
        )
        (cls.work / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.21)
project(SelfVerdictControls NONE)
enable_testing()
foreach(mode expected_failure plain_pass false_green unexpected_success)
  add_test(NAME ${mode} COMMAND "${TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/probe.py" ${mode})
endforeach()
set_tests_properties(expected_failure unexpected_success PROPERTIES
  WILL_FAIL TRUE
  PASS_REGULAR_EXPRESSION "RESULT: ALL PASS"
  FAIL_REGULAR_EXPRESSION "FAIL:|RESULT: FAILURES DETECTED|Heap limit exceeded|fatal signal")
''')
        subprocess.run([cls.cmake, "-S", str(cls.work), "-B", str(cls.build),
                        "-DTEST_PYTHON:FILEPATH=" + sys.executable],
                       check=True, capture_output=True, text=True, timeout=30)
        cls.inventory = cls.work / "inventory.json"
        cls.inventory.write_text(subprocess.check_output(
            [cls.ctest, "--test-dir", str(cls.build), "--show-only=json-v1"],
            text=True, timeout=30))
        cls.expectations = audit.load_ctest_expectations(str(cls.inventory))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_cases(self, *names):
        type(self).serial += 1
        junit = self.work / f"junit-{self.serial}.xml"
        result = subprocess.run(
            [self.ctest, "--test-dir", str(self.build), "--output-junit", str(junit),
             "-R", "^(" + "|".join(names) + ")$"],
            capture_output=True, text=True, timeout=30)
        self.assertTrue(junit.is_file(), result.stdout + result.stderr)
        return result, junit

    def test_expected_rejection_is_explicit_and_raw_output_is_unchanged(self):
        result, junit = self.run_cases("expected_failure", "plain_pass")
        self.assertEqual(result.returncode, 0, result.stdout)
        original = junit.read_bytes()
        legacy, _ = audit.scan_junit(str(junit), None)
        self.assertEqual([x.name for x in legacy], ["expected_failure"])
        expected = []
        contradictions, examined = audit.scan_junit(
            str(junit), None, ctest_expectations=self.expectations,
            expected_failures=expected)
        self.assertEqual((contradictions, examined), ([], 2))
        self.assertEqual([x["name"] for x in expected], ["expected_failure"])
        self.assertEqual(expected[0]["failure_markers"], ["FAIL: deliberate fixture rejection"])
        self.assertEqual(junit.read_bytes(), original)

    def test_ordinary_false_green_still_fails_audit(self):
        result, junit = self.run_cases("false_green")
        self.assertEqual(result.returncode, 0, result.stdout)
        expected = []
        contradictions, _ = audit.scan_junit(
            str(junit), None, ctest_expectations=self.expectations,
            expected_failures=expected)
        self.assertEqual([x.name for x in contradictions], ["false_green"])
        self.assertEqual(expected, [])

    def test_unexpected_success_still_fails_ctest(self):
        result, junit = self.run_cases("unexpected_success")
        self.assertNotEqual(result.returncode, 0)
        case = next(ET.parse(junit).getroot().iter("testcase"))
        self.assertIsNotNone(case.find("failure"))
        expected = []
        audit.scan_junit(str(junit), None, ctest_expectations=self.expectations,
                         expected_failures=expected)
        self.assertEqual(expected, [])

    def test_false_expectation_does_not_exempt_failure_output(self):
        _, junit = self.run_cases("expected_failure")
        expectations = dict(self.expectations, expected_failure=False)
        contradictions, _ = audit.scan_junit(str(junit), None, ctest_expectations=expectations)
        self.assertEqual([x.name for x in contradictions], ["expected_failure"])

    def test_unknown_or_duplicate_junit_names_are_rejected(self):
        _, junit = self.run_cases("expected_failure")
        with self.assertRaisesRegex(ValueError, "unknown or duplicate"):
            audit.scan_junit(str(junit), None, ctest_expectations={})
        tree = ET.parse(junit)
        suite = tree.getroot()
        if suite.tag != "testsuite":
            suite = suite.find("testsuite")
        suite.append(copy.deepcopy(next(suite.iter("testcase"))))
        duplicate = self.work / "duplicate.xml"
        tree.write(duplicate)
        with self.assertRaisesRegex(ValueError, "unknown or duplicate"):
            audit.scan_junit(str(duplicate), None, ctest_expectations=self.expectations)

    def test_malformed_inventory_is_rejected(self):
        original = json.loads(self.inventory.read_text())
        variants = []
        duplicate = copy.deepcopy(original)
        duplicate["tests"].append(copy.deepcopy(duplicate["tests"][0]))
        variants.append(duplicate)
        for value in ["TRUE", 1, None, []]:
            bad = copy.deepcopy(original)
            bad["tests"][0]["properties"] = [{"name": "WILL_FAIL", "value": value}]
            variants.append(bad)
        duplicate_prop = copy.deepcopy(original)
        duplicate_prop["tests"][0]["properties"] = [
            {"name": "WILL_FAIL", "value": True}, {"name": "WILL_FAIL", "value": False}]
        variants.append(duplicate_prop)
        wrong_kind = copy.deepcopy(original)
        wrong_kind["kind"] = "untrusted-list"
        variants.append(wrong_kind)
        for version in [{"major": True, "minor": 0}, {"major": 1, "minor": "0"}, {"major": 2, "minor": 0}]:
            wrong_version = copy.deepcopy(original)
            wrong_version["version"] = version
            variants.append(wrong_version)
        for number, value in enumerate(variants):
            with self.subTest(number=number):
                path = self.work / f"bad-{number}.json"
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    audit.load_ctest_expectations(str(path))
        duplicate_key = self.work / "duplicate-key.json"
        duplicate_key.write_text('{"kind":"ctestInfo","kind":"ctestInfo"}')
        with self.assertRaisesRegex(ValueError, "duplicate inventory key"):
            audit.load_ctest_expectations(str(duplicate_key))

    def test_cli_reports_expected_cases_and_rejects_missing_inventory(self):
        _, junit = self.run_cases("expected_failure", "plain_pass")
        command = [sys.executable, str(ROOT / "scripts/check_self_verdicts.py"),
                   "--junit", str(junit), "--no-trace", "--format", "json",
                   "--ctest-inventory"]
        result = subprocess.run(command + [str(self.inventory)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["passed"])
        self.assertEqual([x["name"] for x in report["expected_failure_cases"]], ["expected_failure"])
        missing = subprocess.run(command + [str(self.work / "missing.json")],
                                 capture_output=True, text=True)
        self.assertNotEqual(missing.returncode, 0)
        self.assertTrue(json.loads(missing.stdout)["read_errors"])

    @unittest.skipIf(os.name == "nt", "the shell producer runs on Unix release runners")
    def test_real_producer_uses_its_captured_inventory(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("RELEASE_", "ESHKOL_DURABLE_"))}
        trace = self.work / "producer-trace"
        env.update(BUILD_DIR=str(self.build), TRACE_DIR=str(trace),
                   ESHKOL_DURABLE_WORK_ROOT=str(self.work / "producer-work"))
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/run_ctest_gate.sh"), "--", "-R", "^expected_failure$"],
            env=env, capture_output=True, text=True, timeout=30)
        # This miniature fixture intentionally lacks the real release pillar
        # groups, so the aggregate must fail even though its self-scan is clean.
        self.assertNotEqual(result.returncode, 0)
        events = [json.loads(line) for line in (trace / "ctest_gate.jsonl").read_text().splitlines()]
        scan = [e for e in events if e.get("kind") == "ctest" and e.get("name") == "ctest_self_verdict_scan"]
        self.assertEqual([e["value"] for e in scan], ["PASS"], result.stdout + result.stderr)
        self.assertIn("configured expected-failure cases : 1", result.stdout)
        self.assertTrue(any(e.get("name") == "ctest_suite_green" and e.get("value") == "FAIL" for e in events))


if __name__ == "__main__":
    unittest.main()
