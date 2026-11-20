"""Exercise the input boundaries of release evidence producers."""
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


class ReleaseGateInputs(unittest.TestCase):
    def test_candidate_cannot_select_readiness_target(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
        target = workflow["env"]["RELEASE_TARGET"]
        self.assertEqual(target, "v1.4-connection")
        for tag, oracle, expected in (
            ("v1.4.0-connection", target, 0),
            ("v1.3.6-evolve", target, 1),
            ("v1.3.6-evolve", "v1.3.6-evolve", 1),
            ("v1.4.0-connection", "v1.4.0-connection", 1),
        ):
            with self.subTest(tag=tag, oracle=oracle):
                result = subprocess.run([sys.executable, SCRIPTS / "release_target.py",
                                         "--tag", tag, "--target", oracle], capture_output=True)
                self.assertEqual(result.returncode, expected, result.stderr)

    def test_criterion_count_matches_oracle_requires(self):
        from verify_v1_3_release_evidence import EXPECTED_CRITERION_COUNTS
        data = yaml.safe_load((ROOT / ".icc/completion-oracles.yaml").read_text(encoding="utf-8"))
        for target, expected_count in EXPECTED_CRITERION_COUNTS.items():
            with self.subTest(target=target):
                oracle = next((item for item in data["oracles"] if item.get("name") == target), None)
                self.assertIsNotNone(oracle, f"{target} oracle is absent")
                self.assertEqual(len(oracle.get("requires", [])), expected_count,
                                  f"{target} pinned criterion count has drifted from the oracle file")

    @unittest.skipIf(os.name == "nt", "connection probe uses bash")
    def test_qubit_probe_requires_control_and_specific_refusal(self):
        gate = (SCRIPTS / "run_v14_connection_gate.sh").read_text()
        start = gate.index("probe_run v14_linear_consume_exactly_once_static ")
        invocation = gate[start:gate.index("\n\n", start)]
        harness = """
. scripts/lib/connection_gate_probe.sh
PROBE_TOTAL=0
PROBE_FAILURES=0
emit_event() { printf '%s %s\\n' "$1" "$2"; }
""" + invocation + '\ntest "$PROBE_FAILURES" -eq 0'

        scratch = ROOT / ".scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            runner = Path(directory) / "eshkol-run"
            runner.write_text("""#!/usr/bin/env python3
import os, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index('-o') + 1])
mode = os.environ['PROBE_MODE']
if '--strict-types' not in sys.argv:
    sys.exit(2)
if any(arg.endswith('qubit_linear_valid_test.esk') for arg in sys.argv):
    if mode == 'refuse-control':
        sys.exit(1)
    out.write_text('#!/bin/sh\\nexit 0\\n')
    out.chmod(0o755)
    sys.exit(0)
if mode == 'artifact':
    out.write_text('clone')
print("parser error" if mode == 'parse' else "linear variable 'q' was consumed more than once",
      file=sys.stderr)
sys.exit(0 if mode == 'accept' else 1)
""")
            runner.chmod(0o755)
            for mode in ("valid", "parse", "refuse-control", "artifact", "accept"):
                with self.subTest(mode=mode):
                    result = subprocess.run(["bash", "-c", harness], cwd=ROOT, capture_output=True,
                                            env={**os.environ, "PROBE_MODE": mode, "ESHKOL_RUN": str(runner),
                                                 "WORK": directory, "TMPDIR": directory})
                    self.assertEqual(result.returncode == 0, mode == "valid", result.stderr)

    def test_wasm_checker_accepts_build_dir_and_rejects_unknown_override(self):
        command = [sys.executable, SCRIPTS / "check_wasm_imports.py", "--selftest", "--build-dir", "build"]
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        result = subprocess.run(command + ["--server", "absent"], capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"unrecognized arguments: --server", result.stderr)

    def test_package_version_requires_exact_compiler_and_staged_tag(self):
        from verify_release_package import verify_version
        scratch = ROOT / ".scratch"
        scratch.mkdir(exist_ok=True)
        tag = "v1.4.0-connection"
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            package = Path(directory)
            stamp = package / "share/eshkol/release-tag"
            stamp.parent.mkdir(parents=True)
            stamp.write_text(tag + "\n")
            verify_version(package, tag, f"Eshkol Compiler {tag}\n", "")
            old_tag = "v1.3.6-evolve"
            for staged, expected, reported in (
                (tag, tag, tag + "-other"),
                (tag, tag, "v1.4.0-evolve"),
                (old_tag, old_tag, old_tag),
                (old_tag, tag, tag),
            ):
                with self.subTest(staged=staged, expected=expected, reported=reported):
                    stamp.write_text(staged + "\n")
                    diagnostic = io.StringIO()
                    with redirect_stderr(diagnostic), self.assertRaises(SystemExit) as refusal:
                        verify_version(package, expected, f"Eshkol Compiler {reported}\n", tag)
                    self.assertEqual(refusal.exception.code, 1)
                    self.assertIn("expected package and compiler version", diagnostic.getvalue())


if __name__ == "__main__":
    unittest.main()
