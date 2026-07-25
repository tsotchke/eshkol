#!/usr/bin/env python3
"""Build-free regression guard for the execution-backed coverage gate (A1).

These tests need no compiler: they drive the pure gating logic of
``scripts/language_coverage.py`` with synthetic TSV evidence and synthetic
deficit ledgers. They lock in the A1 assurance property — a construct is
"covered" only with runtime execution evidence (or a bounded compile-time-form
witness); lexical name-presence earns zero credit — and they pin the monotonic
deficit ratchet so a future refactor cannot silently walk the claim down.

Run standalone (no ``--eshkol-run`` needed):

    python3 scripts/test_language_coverage_gate.py
"""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "language_coverage", REPO / "scripts" / "language_coverage.py")
LC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LC)


def evidence_from(records):
    with tempfile.TemporaryDirectory() as trace_dir:
        pathlib.Path(trace_dir, "language-coverage-1.tsv").write_text(
            "\n".join(records) + "\n", encoding="utf-8")
        return LC.load_runtime_evidence([trace_dir])


class NamePresenceEarnsNoCredit(unittest.TestCase):
    """The A1 property: spelling a construct is never enough to certify it."""

    def test_parsed_only_spelling_is_uncovered(self):
        # A construct whose name merely APPEARS (parsed, and even code was
        # generated for it) but never executed earns zero credit.
        ev = evidence_from([
            "P\ttests/x.esk\t1\t2\t7\tvector-map",
            "G\ttests/x.esk\t1\t2\t7",
        ])
        self.assertNotIn("vector-map", ev["covered_names"])

    def test_execution_earns_credit(self):
        ev = evidence_from([
            "P\ttests/x.esk\t1\t2\t7\tvector-map",
            "O\ttests/x.esk\t1\t2\t7",
        ])
        self.assertIn("vector-map", ev["covered_names"])

    def test_direct_call_and_vm_dispatch_earn_credit(self):
        ev = evidence_from([
            "C\ttests/x.esk\t3\t4\tstring-upcase",
            "V\t<vm>\t0\t0\t80\twrite-string",
        ])
        self.assertIn("string-upcase", ev["covered_names"])
        self.assertIn("write-string", ev["covered_names"])

    def test_codegen_credit_is_bounded_to_compile_time_allowlist(self):
        # A non-allowlist runtime construct with only parse+codegen (no O/C/V)
        # stays uncovered; an allowlisted compile-time form is credited.
        non_allowlist = evidence_from([
            "P\ttests/x.esk\t5\t1\t9\tvector-ref",
            "A\ttests/x.esk\t5\t1\t9",
        ])
        allowlisted = evidence_from([
            "P\ttests/x.esk\t6\t1\t9\tdefine-syntax",
            "A\ttests/x.esk\t6\t1\t9",
        ])
        self.assertNotIn("vector-ref", non_allowlist["covered_names"])
        self.assertIn("define-syntax", allowlisted["covered_names"])


class ExecutionBackedInvariant(unittest.TestCase):
    """verify_execution_backed_invariant is the permanent demotion tripwire."""

    def test_passes_when_covered_subset_of_runtime_evidence(self):
        LC.verify_execution_backed_invariant(
            covered={"map": "higher_order"},
            runtime_names={"map", "for-each"},
            surface={"map": "higher_order", "for-each": "higher_order"},
            source_heads={"map", "for-each"})

    def test_raises_when_a_construct_lacks_runtime_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "name-presence leak"):
            LC.verify_execution_backed_invariant(
                covered={"map": "higher_order", "ghost": "misc"},
                runtime_names={"map"},
                surface={"map": "higher_order", "ghost": "misc"},
                source_heads={"map", "ghost"})


class DeficitRatchet(unittest.TestCase):
    """The deficit list is monotonic: it may shrink, never grow."""

    def make(self, covered, fraction, deficit_names):
        surface = {name: "misc" for name in
                   list(covered) + list(deficit_names)}
        by_cat = {"misc": list(deficit_names)} if deficit_names else {}
        return LC.build_execution_deficit(
            surface=surface,
            covered={name: "misc" for name in covered},
            uncovered_by_category=by_cat,
            execution_fraction=fraction,
            lexical_covered=len(covered),
            spelled_but_unproven=[],
            execution_only_names=[])

    def test_no_baseline_passes(self):
        current = self.make({"a", "b"}, 1.0, set())
        self.assertTrue(LC.evaluate_deficit_ratchet(current, None)["pass"])

    def test_holding_the_line_passes(self):
        baseline = self.make({"a", "b"}, 1.0, set())
        current = self.make({"a", "b"}, 1.0, set())
        self.assertTrue(LC.evaluate_deficit_ratchet(current, baseline)["pass"])

    def test_new_deficit_name_fails(self):
        baseline = self.make({"a", "b"}, 1.0, set())
        current = self.make({"a"}, 0.5, {"b"})
        verdict = LC.evaluate_deficit_ratchet(current, baseline)
        self.assertFalse(verdict["pass"])
        self.assertTrue(verdict["deficit_grew"])
        self.assertIn("b", verdict["new_deficit_names"])

    def test_shrinking_deficit_passes(self):
        baseline = self.make({"a"}, 0.5, {"b", "c"})
        current = self.make({"a", "b", "c"}, 1.0, set())
        self.assertTrue(LC.evaluate_deficit_ratchet(current, baseline)["pass"])

    def test_coverage_drop_without_new_name_still_fails(self):
        # Same deficit set membership, but fewer constructs covered overall.
        baseline = self.make({"a", "b", "c"}, 1.0, set())
        current = self.make({"a", "b"}, 0.9, set())
        verdict = LC.evaluate_deficit_ratchet(current, baseline)
        self.assertFalse(verdict["pass"])
        self.assertTrue(verdict["coverage_dropped"])


class LedgerRoundTrip(unittest.TestCase):
    def test_build_and_load_are_consistent(self):
        surface = {"a": "numeric", "b": "numeric", "c": "misc"}
        deficit = LC.build_execution_deficit(
            surface=surface,
            covered={"a": "numeric"},
            uncovered_by_category={"numeric": ["b"], "misc": ["c"]},
            execution_fraction=1 / 3,
            lexical_covered=2,
            spelled_but_unproven=["b"],
            execution_only_names=[])
        self.assertEqual(deficit["evidence_mode"], "runtime-execution")
        self.assertEqual(deficit["deficit_total"], 2)
        self.assertEqual(sorted(deficit["deficit_names"]), ["b", "c"])
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root, "execution_deficit.json")
            import json
            path.write_text(json.dumps(deficit), encoding="utf-8")
            loaded = LC.load_execution_deficit(str(path))
        self.assertEqual(loaded["deficit_names"], deficit["deficit_names"])
        self.assertIsNone(LC.load_execution_deficit(
            str(pathlib.Path(root, "absent.json"))))


POLICY = REPO / "tests" / "coverage" / "coverage_policy.json"
LEDGER = REPO / "tests" / "coverage" / "execution_deficit.json"


def live_surface_names():
    """The surface set exactly as scripts/language_coverage.py computes it."""
    manifest = json.loads(
        (REPO / "tests" / "coverage" / "language_surface.json").read_text(
            encoding="utf-8"))
    surface = {}
    for builtin in manifest["builtins"]:
        surface[builtin["name"]] = builtin["category"]
    for form in manifest["special_forms"]:
        surface.setdefault(form["name"], form["category"])
    for entry in manifest["prelude"]:
        surface.setdefault(entry["name"], entry["category"])
    return {name for name in surface if not name.startswith("_")}


class CommittedFloorTracksTheLiveSurface(unittest.TestCase):
    """The floor must equal the surface, not trail it.

    `minimum_covered_fraction` is 1.0, so the fraction condition already forces
    100% of whatever the manifest holds. The *count* floors are the second lock,
    and a count floor that trails the manifest silently donates slack: with the
    floor at N and the surface at N+k, k constructs could stop executing and the
    count condition would still report PASS. That is exactly the drift this test
    exists to catch — the surface grew to 1,082 when the `i128`/bitwise builtins
    landed while the floor and the ledger stayed at 1,078, so four `numeric`
    (high-risk) constructs sat outside the count ratchet. Every future builtin
    must ratchet the floor in the same commit that declares it.
    """

    def test_policy_count_floors_equal_the_live_surface_total(self):
        surface_total = len(live_surface_names())
        policy = json.loads(POLICY.read_text(encoding="utf-8"))
        self.assertEqual(
            policy["baseline_surface_total"], surface_total,
            "coverage_policy.json baseline_surface_total trails the manifest; "
            "ratchet it to %d" % surface_total)
        self.assertEqual(
            policy["minimum_covered"], surface_total,
            "coverage_policy.json minimum_covered trails the manifest; "
            "ratchet it to %d" % surface_total)

    def test_policy_fraction_floor_is_never_walked_down(self):
        policy = json.loads(POLICY.read_text(encoding="utf-8"))
        self.assertEqual(policy["minimum_covered_fraction"], 1.0)
        self.assertEqual(policy["completion"]["minimum_covered_fraction"], 1.0)
        self.assertEqual(policy["completion"]["maximum_uncovered_high_risk"], 0)

    def test_ledger_baseline_equals_the_live_surface_total(self):
        surface_total = len(live_surface_names())
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
        self.assertEqual(ledger["evidence_mode"], "runtime-execution")
        self.assertEqual(
            ledger["surface_total"], surface_total,
            "execution_deficit.json surface_total trails the manifest; "
            "regenerate the ledger (--write-execution-deficit)")
        self.assertEqual(
            ledger["baseline_execution_backed_covered"], surface_total,
            "execution_deficit.json records fewer proven constructs than the "
            "manifest declares; the claim is only substantiated at %d"
            % surface_total)
        self.assertEqual(ledger["baseline_execution_backed_fraction"], 1.0)
        self.assertEqual(ledger["deficit_total"], 0)
        self.assertEqual(ledger["deficit_names"], [])


class TheGateCanActuallyFail(unittest.TestCase):
    """Drive the real gate against the real committed policy AND ledger.

    A gate nobody has ever seen go red is worth nothing. These two runs are a
    matched pair: full surface must exit 0, and the same surface minus exactly
    one construct must exit nonzero and name the construct it lost.
    """

    @staticmethod
    def write_full_surface_trace(directory, drop=None):
        # C records are the executed-direct-call channel, so this synthesizes a
        # run in which every surface construct dispatched at runtime.
        records = [
            "C\ttests/synthetic_gate_probe.esk\t%d\t1\t%s" % (index + 1, name)
            for index, name in enumerate(sorted(live_surface_names()))
            if name != drop
        ]
        pathlib.Path(directory, "language-coverage-synthetic.tsv").write_text(
            "\n".join(records) + "\n", encoding="utf-8")

    def run_gate(self, drop=None):
        with tempfile.TemporaryDirectory() as root:
            trace_dir = pathlib.Path(root, "trace")
            trace_dir.mkdir()
            self.write_full_surface_trace(trace_dir, drop=drop)
            completed = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "language_coverage.py"),
                 "--runtime-trace-dir", str(trace_dir),
                 "--policy", str(POLICY),
                 "--execution-deficit", str(LEDGER),
                 "--require-zero-high-risk",
                 # Never write the tracked sidecar from a test.
                 "--json", str(pathlib.Path(root, "coverage_run.json"))],
                cwd=str(REPO), text=True, timeout=300, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            return completed.returncode, completed.stdout

    def test_full_surface_passes(self):
        code, output = self.run_gate()
        self.assertEqual(code, 0, output)
        self.assertIn("100.0%", output)

    def test_losing_one_construct_turns_the_gate_red(self):
        victim = "tensor-matmul"
        self.assertIn(victim, live_surface_names())
        code, output = self.run_gate(drop=victim)
        self.assertNotEqual(
            code, 0,
            "the coverage gate stayed green after a surface construct lost all "
            "execution evidence:\n%s" % output)
        self.assertIn(victim, output)


if __name__ == "__main__":
    unittest.main()
