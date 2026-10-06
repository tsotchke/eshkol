#!/usr/bin/env python3
"""Exercise release harness path resolution and durable run isolation."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ReleaseHarnessPaths(unittest.TestCase):
    def setUp(self):
        (ROOT / ".scratch").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="harness paths ", dir=ROOT / ".scratch")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.build = self.root / "build with spaces"
        self.build.mkdir()
        # These doubles prove executable selection, including the AOT output
        # selected by the real pin-budget harness. They make no runtime claim.
        native = self.build / "eshkol-run"
        native.write_text('''#!/bin/bash
if [ "${1:-}" = -o ]; then
  printf '#!/bin/bash\necho "continuation region-pin budget exceeded" >&2\nexit 1\n' > "$2"
  chmod +x "$2"
  exit 0
fi
echo 'continuation region-pin budget exceeded' >&2
exit 1
''')
        native.chmod(0o755)
        vm = self.build / "eshkol-vm-standalone-test"
        vm.write_text('''#!/bin/bash
case "$1" in
  *region_pin_budget*) echo 'continuation region-pin budget exceeded' >&2 ;;
  *) echo 'Arity mismatch: controlled harness double' >&2 ;;
esac
exit 1
''')
        vm.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("ESHKOL_")}

    def run_harness(self, script, **env):
        return subprocess.run(["bash", str(ROOT / script)], cwd=self.root,
                              env={**self.env, **env}, capture_output=True, text=True,
                              timeout=20)

    def test_absolute_and_relative_builds_with_spaces(self):
        for build in (str(self.build), str(self.build.relative_to(ROOT))):
            for script in ("tests/memory/continuation_pin_budget_boundary_test.sh",
                           "tests/integration/vm_first_class_builtin_arity_test.sh"):
                with self.subTest(build=build, script=script):
                    result = self.run_harness(script, BUILD_DIR=build)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_explicit_runners_override_build_directory(self):
        result = self.run_harness("tests/memory/continuation_pin_budget_boundary_test.sh",
                                  BUILD_DIR="absent", ESHKOL_RUN=str(self.build / "eshkol-run"),
                                  ESHKOL_VM=str(self.build / "eshkol-vm-standalone-test"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_eskm_modes_have_distinct_preserved_evidence_and_refuse_reuse(self):
        # Deliberately fail compilation after the real fixture and directory
        # checks. Both roles must retain their own failure output unchanged.
        compiler = self.build / "reject-compile"
        compiler.write_text("#!/bin/bash\necho controlled-compile-failure >&2\nexit 42\n")
        compiler.chmod(0o755)
        evidence = self.root / "evidence"
        evidence.mkdir()
        for mode, name in (([], "eskm-v1-model-io-parity"),
                           (["--self-test"], "eskm-v1-model-io-parity-selftest")):
            command = ["bash", str(ROOT / "scripts/run_eskm_v1_model_load_parity.sh"),
                       *mode, str(compiler), str(self.build / "eshkol-vm-standalone-test")]
            env = {**self.env, "ESHKOL_DURABLE_WORK_ROOT": str(evidence)}
            result = subprocess.run(command, cwd=self.root, env=env,
                                    capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            output = evidence / name / "aot-compile.err"
            self.assertIn("controlled-compile-failure", output.read_text())
            before = output.read_bytes()
            retry = subprocess.run(command, cwd=self.root, env=env,
                                   capture_output=True, text=True, timeout=20)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn("durable evidence target already exists", retry.stderr)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
