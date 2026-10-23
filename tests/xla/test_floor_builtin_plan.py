"""Host checks of the shared signed floor-division lowering data (no device claim)."""
import builtins
import importlib.util
import operator
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("plan", ROOT / "scripts/gen_builtin_parity_plan.py")
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)
NAMES = ("modulo", "mod", "floor-remainder", "floor-quotient")
TRUNC_NAMES = ("quotient", "remainder")


def trunc_div(a, b):
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def evaluate(row, a, b, omit_correction=False):
    values = {"a0": a, "a1": b}
    binary = {"Divide": trunc_div, "Remainder": lambda a, b: a-trunc_div(a, b)*b,
              "Add": operator.add, "Subtract": operator.sub, "Multiply": operator.mul,
              "And": operator.and_, "Xor": operator.xor}
    for step in row["lowering"]:
        if step["op"] == "const": result = int(step["value"])
        elif step["op"] == "binary": result = binary[step["kind"]](values[step["lhs"]], values[step["rhs"]])
        elif step["op"] == "compare":
            result = {"NE": operator.ne, "LT": operator.lt}[step["dir"]](values[step["lhs"]], values[step["rhs"]])
        elif step["op"] == "convert": result = int(values[step["in"]])
        else: raise AssertionError("unsupported lowering op " + step["op"])
        values[step["out"]] = 0 if omit_correction and step["out"] == "correction" else result
    return values[row["result"]]


class FloorPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = plan.load_table(plan.TABLE)
        plan.validate_floor_contract(cls.rows)

    def test_small_signed_domain(self):
        for name in NAMES:
            row = self.rows[name]
            self.assertEqual(row["etype"], "s64")
            for a in range(-37, 38):
                for b in range(-9, 10):
                    if b:
                        expected = a // b if name == "floor-quotient" else a % b
                        self.assertEqual(evaluate(row, a, b), expected, (name, a, b))

    def test_truncation_and_floor_are_distinct_contracts(self):
        for a, b in ((-13, 5), (13, -5), (-13, -5), (13, 5)):
            self.assertEqual(evaluate(self.rows["quotient"], a, b), trunc_div(a, b))
            self.assertEqual(evaluate(self.rows["remainder"], a, b), a-trunc_div(a, b)*b)
            self.assertEqual(evaluate(self.rows["floor-quotient"], a, b), a // b)
            for name in ("modulo", "mod", "floor-remainder"):
                self.assertEqual(evaluate(self.rows[name], a, b), a % b)

    def test_fail_closed_on_missing_name_wrong_dtype_or_bad_table_entry(self):
        missing = dict(self.rows)
        del missing["floor-remainder"]
        with self.assertRaisesRegex(ValueError, "missing signed-division"):
            plan.validate_floor_contract(missing)

        for rejected_type in ("f64", "i128"):
            wrong_dtype = dict(self.rows)
            wrong_dtype["floor-quotient"] = dict(
                self.rows["floor-quotient"], etype=rejected_type)
            with self.assertRaisesRegex(ValueError, "etype s64"):
                plan.validate_floor_contract(wrong_dtype)

        wrong_lowering = dict(self.rows)
        wrong_lowering["floor-quotient"] = dict(
            self.rows["floor-quotient"], lowering=self.rows["quotient"]["lowering"])
        with self.assertRaisesRegex(ValueError, "truncating-remainder basis"):
            plan.validate_floor_contract(wrong_lowering)

    def test_unsupported_s64_overflow_and_exact_numeric_domains(self):
        overflow = dict(self.rows)
        row = dict(self.rows["floor-quotient"])
        row["inputs"] = [{"values": [-(1 << 63)]}, {"values": [-1]}]
        overflow["floor-quotient"] = row
        with self.assertRaisesRegex(ValueError, "overflowing s64 MIN / -1"):
            plan.validate_floor_contract(overflow)

        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML unavailable for classification contract check")
        with open(plan.os.path.join(plan.REPO, "lib/backend/xla/builtin_classification.yaml"),
                  encoding="utf-8") as fh:
            classification = yaml.safe_load(fh)["builtins"]
        for name in ("exact", "exact->inexact", "inexact->exact", "/rational",
                     "make-rational", "i128-floor-remainder"):
            self.assertEqual(classification[name]["label"], "host", name)
            self.assertNotIn(name, self.rows, name)

    def test_cases_kill_missing_correction_and_cover_signs(self):
        for name in NAMES:
            row = self.rows[name]
            a, b = [list(map(int, plan.values(x))) for x in row["inputs"]]
            self.assertTrue({(-1,-1),(-1,1),(1,-1),(1,1)} <= {(1 if x>0 else -1,1 if y>0 else -1) for x,y in zip(a,b) if x})
            self.assertTrue(any(x == 0 for x in a))
            self.assertTrue(any(x % y == 0 and x*y < 0 for x,y in zip(a,b)))
            self.assertTrue(any(evaluate(row,x,y,True) != evaluate(row,x,y) for x,y in zip(a,b)))

    def test_dependency_free_reader_preserves_explicit_cases(self):
        original_import = builtins.__import__
        def without_yaml(name, *args, **kwargs):
            if name == "yaml": raise ImportError("exercise minimal reader")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=without_yaml):
            fallback = plan.load_table(plan.TABLE)
        self.assertEqual(set(fallback), set(self.rows))
        for name in self.rows:
            self.assertEqual([plan.values(x) for x in fallback[name]["inputs"]],
                             [plan.values(x) for x in self.rows[name].get("inputs", [])])

    def test_explicit_inputs_and_integer_program(self):
        self.assertEqual(plan.values({"values": "13 -13 0"}), [13,-13,0])
        self.assertEqual(plan.values({"values": [13,-13,0]}), [13,-13,0])
        with self.assertRaises(ValueError): plan.values({"values": "", "count": 0})
        with self.assertRaises(ValueError): plan.values({"values": ""})
        program = plan.build_program({name:self.rows[name] for name in NAMES})
        self.assertIn("(modulo -13 5)", program)
        self.assertNotIn("(modulo -13.0 5.0)", program)
        row = dict(self.rows["modulo"], inputs=[{"values":"1 2"},{"values":"3"}])
        with self.assertRaises(ValueError): plan.build_program({"modulo":row})

    def test_s64_plan_text_never_round_trips_through_float(self):
        values = plan.parse_refs("@q -9223372036854775808 9223372036854775807\n", {"q"})["q"]
        self.assertEqual(values, [-(1 << 63), (1 << 63)-1])
        self.assertEqual([plan.fmt_typed(v, "s64") for v in values],
                         ["-9223372036854775808", "9223372036854775807"])


if __name__ == "__main__": unittest.main()
