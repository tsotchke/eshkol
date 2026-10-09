#!/usr/bin/env python3
"""Focused language-selection tests for the ABI semantic scanner."""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import abi_header_inventory as inventory


class ToolchainFlagSelectionTests(unittest.TestCase):
    def test_suffix_classification(self):
        self.assertFalse(inventory._is_cxx_translation_unit("probe.c", ["cc", "probe.c"]))
        self.assertTrue(inventory._is_cxx_translation_unit("probe.cpp", ["c++", "probe.cpp"]))

    def test_explicit_x_overrides_suffix(self):
        self.assertTrue(inventory._is_cxx_translation_unit("probe.c", ["cc", "-x", "c++", "probe.c"]))
        self.assertTrue(inventory._is_cxx_translation_unit("probe.c", ["cc", "-xc++", "probe.c"]))
        self.assertFalse(inventory._is_cxx_translation_unit("probe.cpp", ["c++", "-x", "c", "probe.cpp"]))

    def test_c_tu_does_not_receive_cxx_system_paths(self):
        self.assertNotIn("-isystem", inventory._toolchain_flags(False))

    def test_real_vm_model_c_entry_is_classified_as_c(self):
        compdb = ROOT / ".scratch" / "native-xla-review" / "compile_commands.json"
        entries = json.loads(compdb.read_text())
        entry = next(e for e in entries if e["file"].endswith("vm_model_io_fail_closed_test.c"))
        self.assertFalse(inventory._is_cxx_translation_unit(entry["file"], entry["command"].split()))


if __name__ == "__main__":
    unittest.main()
