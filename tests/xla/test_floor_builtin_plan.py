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

    def test_small_signed_domain(self):
        for name in NAMES:
            row = self.rows[name]
            self.assertEqual(row["etype"], "s64")
            for a in range(-37, 38):
                for b in range(-9, 10):
                    if b:
                        expected = a // b if name == "floor-quotient" else a % b
                        self.assertEqual(evaluate(row, a, b), expected, (name, a, b))

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


if __name__ == "__main__": unittest.main()
