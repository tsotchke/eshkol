#!/usr/bin/env python3
"""Regression gate: an instrumented site reaches the coverage runtime once.

Language-coverage instrumentation used to call into the runtime every time
an instrumented site executed. The runtime keeps only the first execution,
so a site inside a loop of N iterations paid N runtime calls for one record,
and a loop-heavy example ran more than an order of magnitude slower under the
release readiness gate than without it.

Generated code now guards each site with a private first-execution byte and
the VM guards each coverage marker the same way. This test is deterministic:
it never compares wall-clock times. The runtime counts every entry into an
execution hook and reports the total at exit when
ESHKOL_LANGUAGE_COVERAGE_HOOK_STATS is set. The same program run with a small
and a large iteration count must enter the hooks the same number of times,
and must write the same set of coverage records.
"""

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
STATS_RE = re.compile(r"eshkol-language-coverage: exec-hook-entries=(\d+)")

SMALL = 7
LARGE = 20000

PROGRAM = """(define (step x) (abs x))
(define (spin n acc)
  (if (= n 0)
      acc
      (spin (- n 1) (+ acc (step n) (expt 2 1)))))
(define total
  (let loop ((i 0) (acc 0))
    (if (< i ITERATIONS)
        (loop (+ i 1) (+ acc (string-length "ab")))
        acc)))
(display (+ total (spin ITERATIONS 0)))
(newline)
"""


def expected_output(iterations):
    return str(2 * iterations + iterations * (iterations + 1) // 2 +
               2 * iterations)


class LanguageCoverageHookGuardTest(unittest.TestCase):
    eshkol_run = None
    eshkol_vm = None
    lib_dir = None
    work_root = None

    def environment(self, trace_dir):
        env = os.environ.copy()
        env["ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR"] = str(trace_dir)
        env["ESHKOL_LANGUAGE_COVERAGE_HOOK_STATS"] = "1"
        env["ESHKOL_VM_NO_DISASM"] = "1"
        env.setdefault("ESHKOL_PATH", str(REPO / "lib"))
        return env

    def run_checked(self, argv, env, cwd):
        result = subprocess.run(
            argv, cwd=cwd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=600, check=False)
        self.assertEqual(result.returncode, 0,
                         "%s\n%s\n%s" % (argv, result.stdout, result.stderr))
        return result

    @staticmethod
    def hook_entries(result):
        matches = STATS_RE.findall(result.stderr)
        return int(matches[-1]) if matches else None

    @staticmethod
    def execution_records(trace_dir, source):
        """O/C records for the test source, plus every V record."""
        records = set()
        for path in pathlib.Path(trace_dir).glob("*.tsv"):
            for raw in path.read_text(encoding="utf-8").splitlines():
                fields = raw.split("\t")
                if fields[0] == "V":
                    records.add(raw)
                elif fields[0] in ("O", "C") and len(fields) > 1 and \
                        pathlib.Path(fields[1]).name == source.name:
                    records.add("\t".join([fields[0], source.name] + fields[2:]))
        return records

    def measure(self, mode, iterations):
        root = pathlib.Path(tempfile.mkdtemp(prefix="coverage-guard-",
                                             dir=self.work_root))
        self.addCleanup(shutil.rmtree, root, True)
        trace_dir = root / "trace"
        trace_dir.mkdir()
        source = root / "coverage_guard_loop.esk"
        source.write_text(PROGRAM.replace("ITERATIONS", str(iterations)),
                          encoding="utf-8")
        env = self.environment(trace_dir)
        if mode == "jit":
            result = self.run_checked(
                [self.eshkol_run, "-r", str(source), "-L" + self.lib_dir],
                env, root)
        elif mode == "aot":
            binary = root / "coverage_guard_loop"
            self.run_checked(
                [self.eshkol_run, str(source), "-o", str(binary),
                 "-L" + self.lib_dir], env, root)
            result = self.run_checked([str(binary)], env, root)
        elif mode == "vm":
            result = self.run_checked([self.eshkol_vm, str(source)], env, root)
        else:
            raise AssertionError(mode)
        self.assertIn(expected_output(iterations), result.stdout)
        entries = self.hook_entries(result)
        self.assertIsNotNone(
            entries, "runtime printed no hook statistics:\n" + result.stderr)
        return entries, self.execution_records(trace_dir, source)

    def assert_hooks_independent_of_iterations(self, mode):
        small_entries, small_records = self.measure(mode, SMALL)
        large_entries, large_records = self.measure(mode, LARGE)
        self.assertGreater(small_entries, 0,
                           "%s run entered no execution hook" % mode)
        # Before the guard, LARGE iterations entered the hooks thousands of
        # times more often than SMALL ones.
        self.assertEqual(
            small_entries, large_entries,
            "%s: hook entries grew with the iteration count (%d at %d "
            "iterations, %d at %d)" % (mode, small_entries, SMALL,
                                       large_entries, LARGE))
        self.assertTrue(small_records, "%s wrote no execution records" % mode)
        self.assertEqual(small_records, large_records)

    def test_jit_site_in_loop_reaches_runtime_once(self):
        self.assert_hooks_independent_of_iterations("jit")

    def test_aot_site_in_loop_reaches_runtime_once(self):
        self.assert_hooks_independent_of_iterations("aot")

    def test_vm_marker_in_loop_reaches_runtime_once(self):
        self.assert_hooks_independent_of_iterations("vm")

    def test_generated_hook_calls_sit_behind_their_guard(self):
        """Structural proof: each hook call is in a block entered only when
        that site's private guard byte was zero, and that block sets it."""
        root = pathlib.Path(tempfile.mkdtemp(prefix="coverage-guard-ir-",
                                             dir=self.work_root))
        self.addCleanup(shutil.rmtree, root, True)
        trace_dir = root / "trace"
        trace_dir.mkdir()
        source = root / "coverage_guard_ir.esk"
        source.write_text(PROGRAM.replace("ITERATIONS", str(SMALL)),
                          encoding="utf-8")
        env = self.environment(trace_dir)
        self.run_checked(
            [self.eshkol_run, "--dump-ir", str(source), "-o",
             str(root / "coverage_guard_ir"), "-L" + self.lib_dir],
            env, root)
        dumps = sorted(root.glob("*.ll"))
        self.assertTrue(dumps, "no IR dump in %s" % root)
        ir = "\n".join(path.read_text(encoding="utf-8") for path in dumps)

        guards = set(re.findall(
            r"^(@eshkol_language_coverage_guard[\w.]*) = private global i8 0",
            ir, re.M))
        self.assertTrue(guards, "no private per-site guard globals")

        hook_calls = 0
        for function in re.findall(r"^define [^\n]*\{\n(.*?)^\}", ir,
                                   re.M | re.S):
            blocks = re.split(r"^(?=[\w.$-]+:)", function, flags=re.M)
            by_label = {}
            for block in blocks:
                label = re.match(r"([\w.$-]+):", block)
                by_label[label.group(1) if label else None] = block
            for label, block in by_label.items():
                calls = re.findall(
                    r"call void @eshkol_language_coverage_exec_(?:op|call)\(",
                    block)
                if not calls:
                    continue
                hook_calls += len(calls)
                store = re.search(
                    r"store i8 1, ptr (@eshkol_language_coverage_guard[\w.]*)",
                    block)
                self.assertIsNotNone(
                    store, "hook call in block %r does not set a guard" % label)
                guard = store.group(1)
                self.assertIn(guard, guards)
                branch = re.compile(
                    r"(%%[\w.]+) = load i8, ptr %s\b.*?"
                    r"(%%[\w.]+) = icmp eq i8 \1, 0.*?"
                    r"br i1 \2, label %%%s, label %%[\w.$-]+"
                    % (re.escape(guard), re.escape(label)), re.S)
                self.assertTrue(
                    branch.search(function),
                    "block %r is not entered through guard %s" % (label, guard))
        self.assertGreater(hook_calls, 0, "no coverage hook calls in the IR")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eshkol-run", required=True)
    parser.add_argument("--eshkol-vm", required=True)
    parser.add_argument("--lib-dir", required=True)
    parser.add_argument("--work-dir", default=None,
                        help="parent for the fresh run directories")
    args, unittest_args = parser.parse_known_args()
    if args.work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
        LanguageCoverageHookGuardTest.work_root = os.path.abspath(args.work_dir)
    LanguageCoverageHookGuardTest.eshkol_run = os.path.abspath(args.eshkol_run)
    LanguageCoverageHookGuardTest.eshkol_vm = os.path.abspath(args.eshkol_vm)
    LanguageCoverageHookGuardTest.lib_dir = os.path.abspath(args.lib_dir)
    unittest.main(argv=[__file__] + unittest_args)


if __name__ == "__main__":
    main()
