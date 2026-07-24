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
import pathlib
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


if __name__ == "__main__":
    unittest.main()
