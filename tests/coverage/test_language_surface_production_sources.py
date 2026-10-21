#!/usr/bin/env python3
"""Guard production surface generation against retired compiler ghosts."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]


class ProductionSurfaceSources(unittest.TestCase):
    def test_generation_is_independent_of_legacy_compiler(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            shutil.copy(REPO / "scripts/gen_language_surface.py", root / "scripts")
            for relative in (
                "lib/backend/eshkol_vm.c", "lib/backend/llvm_codegen.cpp",
                "lib/backend/vm_prelude_source.h", "lib/frontend/parser.cpp",
                "inc/eshkol/eshkol.h", "lib/stdlib.esk",
                "lib/agent/quantum.esk", "lib/agent/pqc.esk",
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(REPO / relative, target)
            legacy = root / "lib/backend/eshkol_compiler.c"
            legacy.write_text("GHOST_ONLY_BUILTIN_33\n")
            out = root / "surface.json"
            command = ["python3", str(root / "scripts/gen_language_surface.py"),
                       "--out", str(out)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            first = out.read_bytes()
            legacy.write_text('"SECOND_GHOST_ONLY_BUILTIN"\n')
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(first, out.read_bytes())
            manifest = json.loads(first)
            names = {row["name"] for row in manifest["builtins"]}
            self.assertNotIn("GHOST_ONLY_BUILTIN_33", names)
            self.assertNotIn("SECOND_GHOST_ONLY_BUILTIN", names)

    def test_live_builtin_provenance_and_ids_are_vm_grounded(self):
        spec = importlib.util.spec_from_file_location(
            "gen_language_surface", REPO / "scripts/gen_language_surface.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = module.build_manifest()
        vm = module.extract_builtin_table(module.VM_C)
        self.assertNotIn("native_builtins", manifest["_meta"]["sources"])
        for row in manifest["builtins"]:
            self.assertNotIn("native", row["backends"])
            if "vm" in row["backends"]:
                members = [row["name"]] + row.get("aliases", [])
                source_ids = sorted({nid for name in members for nid in vm.get(name, {}).get("ids", [])})
                self.assertEqual(source_ids, row["ids"])
                if any(name in vm for name in members):
                    self.assertTrue(source_ids)
                    self.assertIsNotNone(row["arity"])
                else:
                    vm_prelude = module.extract_vm_prelude_defs()
                    self.assertTrue(any(name in vm_prelude for name in members))
        self.assertIn("vm_bootstrap", manifest["_meta"]["sources"])
        self.assertIn("stdlib", manifest["_meta"]["sources"])


if __name__ == "__main__":
    unittest.main()
