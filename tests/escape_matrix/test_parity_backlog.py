#!/usr/bin/env python3
"""Build-free regression tests for the parity backlog gates."""

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_script(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GAPS = load_script("scripts/canonicalize_vm_gaps.py", "canonicalize_vm_gaps")
ARITY = load_script("scripts/p8/p8_arity_sweep.py", "p8_arity_sweep")


class GapCanonicalizationTests(unittest.TestCase):
    def test_every_current_gap_has_a_disposition_and_live_route(self):
        passed, errors = GAPS.check()
        self.assertTrue(passed, "\n".join(errors))
        rows = GAPS.canonical_rows(GAPS.read_parity())
        self.assertEqual(len(rows), 328)
        self.assertTrue(all(row[1] in GAPS.VALID for row in rows))
        self.assertTrue(all(row[2] for row in rows))

    def test_generated_probe_is_name_specific_and_deterministic(self):
        first = GAPS.generated_probe("missing-builtin")
        self.assertEqual(first, GAPS.generated_probe("missing-builtin"))
        self.assertIn("GAP-PROBE: missing-builtin", first)
        self.assertIn("(missing-builtin)", first)


class ArityOutcomeTests(unittest.TestCase):
    def test_matching_fatal_diagnostic_is_not_a_divergence(self):
        self.assertEqual(ARITY.diagnostic_class("ERROR: Arity mismatch", rc=1), "arity")
        self.assertEqual(
            ARITY.canonical_outcome(1, {}, "arity"),
            "FATAL:arity",
        )

    def test_values_are_compared_even_when_process_is_successful(self):
        self.assertEqual(
            ARITY.canonical_outcome(0, {0: "42"}, ""),
            "VALUE:42",
        )

    def test_timeout_can_never_become_a_fatal_or_value(self):
        self.assertEqual(
            ARITY.canonical_outcome(None, {}, "", timed_out=True),
            "TIMEOUT",
        )


if __name__ == "__main__":
    unittest.main()
