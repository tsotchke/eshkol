#!/usr/bin/env python3
"""Unit tests for the Windows bounded-runtime export validator."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_windows_runtime_exports",
    ROOT / "scripts" / "verify_windows_runtime_exports.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WindowsRuntimeExportsValidatorTest(unittest.TestCase):
    def test_parses_llvm_readobj_exports(self) -> None:
        output = """
Export {
  Ordinal: 1
  Name: eshkol_raise
  RVA: 0x1000
}
Export {
  Ordinal: 2
  Name: arena_allocate
  RVA: 0x2000
}
"""
        self.assertEqual(
            MODULE.parse_exports(output), {"eshkol_raise", "arena_allocate"}
        )

    def test_accepts_complete_bounded_namespace(self) -> None:
        exports = {f"eshkol_generated_{index}" for index in range(800)}
        exports.update(MODULE.REQUIRED_EXPORTS)
        self.assertEqual(MODULE.verify_exports(exports, 800), [])

    def test_rejects_tree_sitter_only_export_table(self) -> None:
        exports = {"tree_sitter_c", "tree_sitter_cpp"}
        errors = MODULE.verify_exports(exports, 800)
        self.assertEqual(len(errors), 2)
        self.assertIn("missing required runtime exports", errors[0])
        self.assertIn("found 0", errors[1])

    def test_requires_taylor_tower_data_exports(self) -> None:
        exports = {f"eshkol_generated_{index}" for index in range(800)}
        exports.update(MODULE.REQUIRED_EXPORTS)
        exports.remove("__ad_tower_active")
        exports.remove("__ad_tower_order")
        errors = MODULE.verify_exports(exports, 800)
        self.assertEqual(len(errors), 1)
        self.assertIn("__ad_tower_active", errors[0])
        self.assertIn("__ad_tower_order", errors[0])


if __name__ == "__main__":
    unittest.main()
