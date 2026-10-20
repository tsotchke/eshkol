#!/usr/bin/env python3
"""Regression tests for the FFI shell harness using a fake eshkol-run."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "run_ffi_tests.sh"

FAKE_RUNNER = r'''#!/usr/bin/env bash
set -eu
phase=compile
output=
source_file=
while [ "$#" -gt 0 ]; do
    case "$1" in
        -r) phase=jit ;;
        -o) shift; output="${1:-}" ;;
        *.esk) source_file="$1" ;;
    esac
    shift
done
test_name="$(basename "${source_file:-unknown.esk}" .esk)"
mode="${FAKE_FFI_MODE:-pass}"

if [ "$phase" = jit ]; then
    if [ "${ESHKOL_JIT_CACHE:-}" != 0 ]; then
        echo 'fake runner requires ESHKOL_JIT_CACHE=0' >&2
        exit 97
    fi
    case "$mode" in
        failures)
            if [ "$test_name" = emergency_exit_surface_test ]; then
                echo 'fake JIT runtime diagnostic' >&2
                exit 37
            fi
            ;;
        timeout)
            if [ "$test_name" = emergency_exit_surface_test ]; then
                echo 'fake JIT timeout diagnostic'
                exec sleep 3
            fi
            ;;
        exit-137) [ "$test_name" != emergency_exit_surface_test ] || exit 137 ;;
        exit-143) [ "$test_name" != emergency_exit_surface_test ] || exit 143 ;;
        assertion)
            if [ "$test_name" = emergency_exit_surface_test ]; then
                echo 'FAIL: fake JIT assertion diagnostic'
                exit 0
            fi
            ;;
    esac
    echo 'PASS: fake JIT program'
    exit 0
fi

if [ "$mode" = failures ] && [ "$test_name" = emergency_exit_surface_test ]; then
    echo 'fake AOT compile diagnostic' >&2
    exit 23
fi
cat > "$output" <<'PROGRAM'
#!/usr/bin/env bash
mode="${FAKE_FFI_MODE:-pass}"
case "$mode" in
    failures|aot-run-fail) echo 'fake AOT runtime diagnostic' >&2; exit 41 ;;
    assertion) echo 'FAIL: fake AOT assertion diagnostic'; exit 0 ;;
    *) echo 'PASS: fake AOT program'; exit 0 ;;
esac
PROGRAM
chmod +x "$output"
'''


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "POSIX Bash harness")
class FfiHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix=".ffi-harness-", dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.build = self.root / "fake-build"
        self.build.mkdir()
        runner = self.build / "eshkol-run"
        runner.write_text(FAKE_RUNNER)
        runner.chmod(0o755)

    def invoke(self, mode="pass", diagnostic_dir=None, build_dir=None):
        env = os.environ.copy()
        env.pop("RUNNER_TEMP", None)
        env.pop("ESHKOL_DURABLE_WORK_ROOT", None)
        env["BUILD_DIR"] = build_dir or self.build.relative_to(ROOT).as_posix()
        env["ESHKOL_FFI_TEST_TIMEOUT_SECONDS"] = "1"
        env["FAKE_FFI_MODE"] = mode
        if diagnostic_dir is None:
            env.pop("ESHKOL_FFI_DIAGNOSTIC_DIR", None)
        else:
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            env["ESHKOL_FFI_DIAGNOSTIC_DIR"] = str(diagnostic_dir)
        return subprocess.run(
            ["bash", str(HARNESS)], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=20, check=False,
        )

    def test_success_uses_cache_disabled_orc(self):
        diagnostics = self.root / "success-diagnostics"
        result = self.invoke(diagnostic_dir=diagnostics)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Passed:         8", result.stdout)
        self.assertIn("Failed:         0", result.stdout)
        status = (diagnostics / "emergency_exit_surface_test.jit-orc-cache0.status").read_text()
        self.assertIn("ESHKOL_JIT_CACHE=0", status)

    def test_compile_runtime_and_jit_failures_keep_diagnostics(self):
        diagnostics = self.root / "failure-diagnostics"
        result = self.invoke("failures", diagnostics)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        for witness in (
            "fake AOT compile diagnostic",
            "fake AOT runtime diagnostic",
            "fake JIT runtime diagnostic",
        ):
            self.assertIn(witness, result.stdout)
        self.assertIn("exit_status=23", (diagnostics / "emergency_exit_surface_test.aot-compile.status").read_text())
        self.assertIn("exit_status=37", (diagnostics / "emergency_exit_surface_test.jit-orc-cache0.status").read_text())

    def test_exit_zero_failure_markers_print_aot_and_jit_output(self):
        result = self.invoke("assertion", self.root / "assertion-diagnostics")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("fake AOT assertion diagnostic", result.stdout)
        self.assertIn("fake JIT assertion diagnostic", result.stdout)

    def test_timeout_is_distinct_from_child_exit_137_and_143(self):
        for mode, expected_status, timed_out in (
            ("timeout", 143, "yes"),
            ("exit-137", 137, "no"),
            ("exit-143", 143, "no"),
        ):
            with self.subTest(mode=mode):
                diagnostics = self.root / (mode + "-diagnostics")
                result = self.invoke(mode, diagnostics)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                status = (diagnostics / "emergency_exit_surface_test.jit-orc-cache0.status").read_text()
                self.assertIn("exit_status=%d" % expected_status, status)
                self.assertIn("timed_out=%s" % timed_out, status)
                if mode == "timeout":
                    self.assertIn("JIT ORC TIMEOUT (1s)", result.stdout)
                else:
                    self.assertIn("JIT ORC FAIL (exit %d)" % expected_status, result.stdout)

    def test_local_diagnostics_persist_and_missing_build_refuses(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout)
        line = next(line for line in result.stdout.splitlines() if line.startswith("FFI diagnostics: "))
        diagnostics = Path(line.split(": ", 1)[1])
        try:
            self.assertIn(Path(tempfile.gettempdir()).resolve(), diagnostics.resolve().parents)
            self.assertTrue((diagnostics / "emergency_exit_surface_test.jit-orc-cache0.status").is_file())
        finally:
            shutil.rmtree(diagnostics, ignore_errors=True)

        missing = self.invoke(
            diagnostic_dir=self.root / "missing-build-diagnostics",
            build_dir=self.root.joinpath("missing-build").relative_to(ROOT).as_posix(),
        )
        self.assertEqual(missing.returncode, 1, missing.stdout)
        self.assertIn("build directory not found", missing.stdout)


if __name__ == "__main__":
    unittest.main()
