#!/usr/bin/env python3
"""Regression tests for execution-backed language-surface evidence."""

import argparse
import importlib.util
import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock


REPO = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "language_coverage", REPO / "scripts" / "language_coverage.py")
LANGUAGE_COVERAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LANGUAGE_COVERAGE)


class RuntimeEvidencePolicyTest(unittest.TestCase):
    def synthetic_evidence(self, records):
        with tempfile.TemporaryDirectory() as trace_dir:
            pathlib.Path(trace_dir, "language-coverage-1.tsv").write_text(
                "\n".join(records) + "\n", encoding="utf-8")
            return LANGUAGE_COVERAGE.load_runtime_evidence([trace_dir])

    def test_dead_generated_call_gets_no_credit(self):
        evidence = self.synthetic_evidence([
            "P\ttests/dead.esk\t1\t2\t7\tabs",
            "G\ttests/dead.esk\t1\t2\t7",
        ])
        self.assertNotIn("abs", evidence["covered_names"])

    def test_executed_call_gets_credit(self):
        evidence = self.synthetic_evidence([
            "P\ttests/live.esk\t1\t2\t7\tabs",
            "G\ttests/live.esk\t1\t2\t7",
            "O\ttests/live.esk\t1\t2\t7",
            "C\ttests/live.esk\t1\t2\tabs",
        ])
        self.assertIn("abs", evidence["covered_names"])

    def test_lowered_operation_matches_unambiguous_source_position(self):
        evidence = self.synthetic_evidence([
            "P\ttests/lowered.esk\t1\t2\t89\tcase-lambda",
            "G\ttests/lowered.esk\t1\t2\t12",
            "O\ttests/lowered.esk\t1\t2\t12",
        ])
        self.assertIn("case-lambda", evidence["covered_names"])

    def test_lowered_position_does_not_credit_ambiguous_spellings(self):
        evidence = self.synthetic_evidence([
            "P\ttests/ambiguous.esk\t1\t2\t89\tcase-lambda",
            "P\ttests/ambiguous.esk\t1\t2\t12\tlambda",
            "O\ttests/ambiguous.esk\t1\t2\t12",
        ])
        self.assertNotIn("case-lambda", evidence["covered_names"])
        self.assertIn("lambda", evidence["covered_names"])

    def test_compile_time_form_requires_accept_or_codegen(self):
        rejected = self.synthetic_evidence([
            "P\ttests/form.esk\t1\t2\t8\tdefine",
        ])
        accepted = self.synthetic_evidence([
            "P\ttests/form.esk\t1\t2\t8\tdefine",
            "A\ttests/form.esk\t1\t2\t8",
        ])
        self.assertNotIn("define", rejected["covered_names"])
        self.assertIn("define", accepted["covered_names"])

    def test_exact_vm_dispatch_gets_alias_credit(self):
        evidence = self.synthetic_evidence([
            "V\t<vm>\t0\t0\t80\twrite-string",
        ])
        self.assertIn("write-string", evidence["covered_names"])
        self.assertEqual(evidence["event_counts"]["V"], 1)
        self.assertEqual(evidence["vm_dispatch_names"], {"write-string"})

    def test_exact_serialized_vm_scheme_call_hash_gets_credit(self):
        marker = LANGUAGE_COVERAGE.vm_call_name_hash("emit!")
        evidence = self.synthetic_evidence([
            "V\t<vm>\t0\t0\t%d\t@call" % marker,
        ])
        self.assertIn("emit!", evidence["covered_names"])
        self.assertIn("emit!", evidence["vm_dispatch_names"])

    def test_ambiguous_serialized_vm_scheme_call_hash_is_rejected(self):
        fake_manifest = {
            "builtins": [{"name": "surface-a"}, {"name": "surface-b"}],
            "special_forms": [],
            "prelude": [],
        }
        with mock.patch.object(
                LANGUAGE_COVERAGE, "load_manifest",
                return_value=fake_manifest), mock.patch.object(
                    LANGUAGE_COVERAGE, "vm_call_name_hash", return_value=77):
            with self.assertRaisesRegex(
                    RuntimeError, "ambiguous serialized-VM call hash"):
                self.synthetic_evidence(["V\t<vm>\t0\t0\t77\t@call"])

    def test_expected_negative_form_requires_rejection_event(self):
        parsed_only = self.synthetic_evidence([
            "P\ttests/negative.esk\t1\t2\t95\tsyntax-error",
        ])
        rejected = self.synthetic_evidence([
            "P\ttests/negative.esk\t1\t2\t95\tsyntax-error",
            "R\ttests/negative.esk\t1\t2\t95\tsyntax-error",
        ])
        self.assertNotIn("syntax-error", parsed_only["covered_names"])
        self.assertIn("syntax-error", rejected["covered_names"])


class RuntimeInstrumentationTest(unittest.TestCase):
    eshkol_run = None
    eshkol_vm = None

    def run_program(self, trace_dir, instrumented):
        program = """(define live-result (abs -3))
(if #f (acos 0.5) (display \"TRACE-PASS\"))
(newline)
"""
        source = pathlib.Path(trace_dir).parent / "runtime-coverage-branch.esk"
        source.write_text(program, encoding="utf-8")
        env = os.environ.copy()
        env.pop("ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR", None)
        if instrumented:
            env["ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR"] = str(trace_dir)
        result = subprocess.run(
            [self.eshkol_run, "-r", str(source)],
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TRACE-PASS", result.stdout)

    def test_real_untaken_branch_is_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            trace_dir = pathlib.Path(root) / "trace"
            trace_dir.mkdir()
            self.run_program(trace_dir, instrumented=True)
            evidence = LANGUAGE_COVERAGE.load_runtime_evidence([str(trace_dir)])
            self.assertIn("abs", evidence["covered_names"])
            self.assertIn("display", evidence["covered_names"])
            self.assertNotIn("acos", evidence["covered_names"])

    def test_unset_environment_emits_no_trace(self):
        with tempfile.TemporaryDirectory() as root:
            trace_dir = pathlib.Path(root) / "trace"
            trace_dir.mkdir()
            self.run_program(trace_dir, instrumented=False)
            self.assertEqual(list(trace_dir.iterdir()), [])

    def test_imported_module_keeps_its_own_source_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = pathlib.Path(root)
            trace_dir = root_path / "trace"
            trace_dir.mkdir()
            imported = root_path / "runtime-coverage-imported.esk"
            imported.write_text(
                "(define (imported-covered-value) (abs -9))\n",
                encoding="utf-8",
            )
            caller = root_path / "runtime-coverage-caller.esk"
            caller.write_text(
                '(import "%s")\n(display (imported-covered-value))\n(newline)\n'
                % str(imported).replace("\\", "\\\\"),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR"] = str(trace_dir)
            result = subprocess.run(
                [self.eshkol_run, "-r", str(caller)],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("9", result.stdout)

            abs_sources = set()
            for trace_path in trace_dir.glob("*.tsv"):
                for raw in trace_path.read_text(encoding="utf-8").splitlines():
                    fields = raw.split("\t")
                    if fields[0] == "C" and fields[4] == "abs":
                        abs_sources.add(pathlib.Path(fields[1]).resolve())
            self.assertEqual(abs_sources, {imported.resolve()})

    def test_serialized_vm_dispatch_preserves_exact_aliases(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = pathlib.Path(root)
            trace_dir = root_path / "trace"
            trace_dir.mkdir()
            source = root_path / "vm-alias-evidence.esk"
            module = root_path / "vm-alias-evidence.eskb"
            source.write_text(
                '(display (expt 2 3))\n(display (pow 2 3))\n(newline)\n',
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR"] = str(trace_dir)
            compiled = subprocess.run(
                [self.eshkol_run, "--profile", "hosted-vm", "--emit-eskb",
                 str(module), str(source)],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0,
                             compiled.stdout + compiled.stderr)
            executed = subprocess.run(
                [self.eshkol_vm, str(module)],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
            self.assertEqual(executed.returncode, 0,
                             executed.stdout + executed.stderr)
            evidence = LANGUAGE_COVERAGE.load_runtime_evidence([str(trace_dir)])
            self.assertIn("expt", evidence["vm_dispatch_names"])
            self.assertIn("pow", evidence["vm_dispatch_names"])

    def test_serialized_vm_executes_and_credits_direct_scheme_calls(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = pathlib.Path(root)
            trace_dir = root_path / "trace"
            trace_dir.mkdir()
            module = root_path / "vm-direct-call-evidence.eskb"
            source = REPO / "tests" / "vm" / "event_emitter_surface_regression.esk"
            env = os.environ.copy()
            env["ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR"] = str(trace_dir)
            compiled = subprocess.run(
                [self.eshkol_run, "--profile", "hosted-vm", "--emit-eskb",
                 str(module), str(source)],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0,
                             compiled.stdout + compiled.stderr)
            executed = subprocess.run(
                [self.eshkol_vm, str(module)],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
            self.assertEqual(executed.returncode, 0,
                             executed.stdout + executed.stderr)
            evidence = LANGUAGE_COVERAGE.load_runtime_evidence([str(trace_dir)])
            self.assertIn("emit!", evidence["vm_dispatch_names"])
            marker = str(LANGUAGE_COVERAGE.vm_call_name_hash("emit!"))
            raw_trace = "\n".join(
                path.read_text(encoding="utf-8")
                for path in trace_dir.glob("*.tsv"))
            self.assertIn("\t%s\t@call" % marker, raw_trace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eshkol-run", required=True)
    parser.add_argument("--eshkol-vm", required=True)
    args, unittest_args = parser.parse_known_args()
    RuntimeInstrumentationTest.eshkol_run = os.path.abspath(args.eshkol_run)
    RuntimeInstrumentationTest.eshkol_vm = os.path.abspath(args.eshkol_vm)
    unittest.main(argv=[__file__] + unittest_args)


if __name__ == "__main__":
    main()
