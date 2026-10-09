#!/usr/bin/env python3
"""Fast external-program controls for the PR751 shell regression harnesses."""

from pathlib import Path
import os
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / ".scratch"


FAKE_RUN = r'''#!/bin/sh
set -eu
if [ "${FAKE_KIND:-}" = cache ]; then
    n=0
    [ -f "$FAKE_STATE" ] && n=$(cat "$FAKE_STATE")
    n=$((n + 1)); printf '%s\n' "$n" > "$FAKE_STATE"
    case "${FAKE_MODE:-positive}:$n" in
        failed-second:2) exit 9 ;;
        unknown-second:2) echo '[jit-cache] strange' ;;
        empty-second:2|store-only:1) echo '[jit-cache] store KEY' ;;
        duplicate-lookup:1) echo '[jit-cache] miss KEY'; echo '[jit-cache] hit KEY' ;;
        positive:1|wrong-result:1|mutate:1|failed-second:1|unknown-second:1|empty-second:1) echo '[jit-cache] miss KEY'; echo '[jit-cache] store KEY' ;;
        positive:2|wrong-result:2|mutate:2|failed-second:3|unknown-second:3|empty-second:3) echo '[jit-cache] miss KEY'; echo '[jit-cache] store KEY' ;;
        positive:3|wrong-result:3|mutate:3|failed-second:4|unknown-second:4|empty-second:4) echo '[jit-cache] hit KEY' ;;
    esac
    [ "${FAKE_MODE:-}" = wrong-result ] && echo 3 || echo 24
    if [ "${FAKE_MODE:-}" = mutate ] && [ "$n" -eq 1 ]; then
        printf '# changed during run\n' >> "$0"
    fi
    exit 0
fi
out=''
prev=''
is_xla=0
for arg do
    [ "$prev" = -o ] && out=$arg
    case "$arg" in *xla_region_reclaim_test.esk) is_xla=1 ;; esac
    prev=$arg
done
[ -n "$out" ]
if [ "$is_xla" -eq 1 ]; then
    printf '%s\n' '#!/bin/sh' 'echo PASS' 'echo global_total_allocated_bytes=4096' > "$out"
else
    printf '%s\n' '#!/bin/sh' 'echo PASS' > "$out"
fi
chmod +x "$out"
'''


class HarnessControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=SCRATCH, prefix="pr751-harness-")
        cls.work = Path(cls.temp.name)
        cls.build = cls.work / "build"
        cls.build.mkdir()
        cls.bin = cls.work / "bin"
        cls.bin.mkdir()
        timeout = cls.bin / "timeout"
        timeout.write_text("#!/bin/sh\nshift\nexec \"$@\"\n")
        timeout.chmod(0o755)
        cls.fake = cls.build / "eshkol-run"
        cls.fake.write_text(FAKE_RUN)
        cls.fake.chmod(0o755)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, script, mode=None):
        env = os.environ.copy()
        env.update(BUILD_DIR=str(self.build), ESHKOL_RUN=str(self.fake),
                   ESHKOL_TEST_TMP_ROOT=str(SCRATCH),
                   PATH=f"{self.bin}:{env['PATH']}")
        if mode:
            state = self.work / f"{mode}.state"
            state.unlink(missing_ok=True)
            env.update(FAKE_KIND="cache", FAKE_MODE=mode, FAKE_STATE=str(state))
        return subprocess.run(["bash", str(ROOT / script)], cwd=ROOT, env=env,
                              text=True, capture_output=True, timeout=30)

    def test_timeout_helper_propagates_success_timeout_and_failure(self):
        helper = ROOT / "scripts/lib/test_isolation.sh"
        def run(command):
            env = os.environ.copy()
            env.update(ESHKOL_TEST_TIMEOUT_FORCE_PYTHON="1",
                       ESHKOL_TEST_TMP_ROOT=str(SCRATCH))
            return subprocess.run(["bash", "-c", f'. "{helper}"; {command}'],
                                  cwd=ROOT, env=env, timeout=10)

        self.assertEqual(run("eshkol_test_timeout 2 sh -c 'exit 0'").returncode, 0)
        self.assertEqual(run("eshkol_test_timeout 0.1 sh -c 'sleep 2'").returncode, 124)
        self.assertEqual(run("eshkol_test_timeout 2 sh -c 'exit 7'").returncode, 7)

    @unittest.skipUnless(os.name == "posix", "process-group assertion is POSIX-specific")
    def test_timeout_process_tree_cleanup_and_signal_status(self):
        helper = ROOT / "scripts/lib/test_isolation.sh"
        pid_file = self.work / "timeout-child.pid"
        env = os.environ.copy()
        env.update(ESHKOL_TEST_TIMEOUT_FORCE_PYTHON="1",
                   ESHKOL_TEST_TMP_ROOT=str(SCRATCH))
        command = (f". \"{helper}\"; "
                   f"eshkol_test_timeout 0.1 sh -c 'sleep 10 & echo $! > \"{pid_file}\"; wait'")
        timed = subprocess.run(["bash", "-c", command], cwd=ROOT, env=env, timeout=10)
        self.assertEqual(timed.returncode, 124)
        child_pid = int(pid_file.read_text().strip())
        for _ in range(20):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            self.fail("timed-out process-tree grandchild survived")

        signaled = subprocess.run(
            ["bash", "-c", f". \"{helper}\"; eshkol_test_timeout 2 sh -c 'kill -TERM $$'"],
            cwd=ROOT, env=env, timeout=10)
        self.assertEqual(signaled.returncode, 143)

    def test_xla_and_pool_harnesses_use_external_program_successfully(self):
        for script in ("tests/xla/xla_region_reclaim_test.sh",
                       "tests/memory/arena_block_pool_test.sh"):
            with self.subTest(script=script):
                result = self.invoke(script)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cache_positive_requires_miss_miss_hit(self):
        result = self.invoke("tests/codegen/run_cache_xla_threshold_key_test.sh", "positive")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("miss / miss / hit", result.stdout)

    def test_cache_failed_second_run_is_rejected(self):
        result = self.invoke("tests/codegen/run_cache_xla_threshold_key_test.sh", "failed-second")
        self.assertNotEqual(result.returncode, 0)

    def test_cache_wrong_result_is_rejected(self):
        result = self.invoke("tests/codegen/run_cache_xla_threshold_key_test.sh", "wrong-result")
        self.assertNotEqual(result.returncode, 0)

    def test_toolchain_verification_failure_is_rejected(self):
        result = self.invoke("tests/codegen/run_cache_xla_threshold_key_test.sh", "mutate")
        self.assertNotEqual(result.returncode, 0)

    def test_cache_unknown_or_empty_second_status_is_rejected(self):
        for mode in ("unknown-second", "empty-second"):
            with self.subTest(mode=mode):
                result = self.invoke("tests/codegen/run_cache_xla_threshold_key_test.sh", mode)
                self.assertNotEqual(result.returncode, 0)

    def test_cache_store_only_and_duplicate_lookup_are_rejected(self):
        for mode in ("store-only", "duplicate-lookup"):
            with self.subTest(mode=mode):
                result = self.invoke("tests/codegen/run_cache_xla_threshold_key_test.sh", mode)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
