#!/usr/bin/env python3
"""Guard production surface generation against retired compiler ghosts."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gen_language_surface", REPO / "scripts/gen_language_surface.py")
SURFACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SURFACE)


class ProductionSurfaceSources(unittest.TestCase):
    def test_scheme_reader_ignores_nested_comment_string_and_quoted_ghosts(self):
        source = r''' ; (define line-comment-ghost 0)
#| (define block-comment-ghost 0)
   (require core.block-comment-ghost) |#
#; (define datum-comment-ghost 0)
(define (public-name x)
  (let () (define local-ghost 0) x))
(define-values (first-value second-value) (values 1 2))
(define text "(define string-ghost 0) (require core.string-ghost)")
'(define quoted-ghost 0)
(define nested-require
  (lambda () #; (require core.datum-local-only)
    (require core.local-only)))
(provide public-name first-value
  ;; local-looking entries never occur at top level
  (rename second-value renamed-second))
'''
        forms = SURFACE.parse_scheme_forms(source)
        definitions = {name for form in forms
                       for name, _ in SURFACE._top_level_definition(form)}
        exports = {name for form in forms
                   for name, _ in SURFACE._top_level_exports(form)}
        requires = {name for form in forms
                    for name, _ in SURFACE._require_module_names(form)}
        self.assertEqual(
            {"public-name", "first-value", "second-value", "text", "nested-require"},
            definitions)
        self.assertEqual({"public-name", "first-value", "renamed-second"}, exports)
        self.assertEqual(set(), requires)
        for ghost in ("line-comment-ghost", "block-comment-ghost",
                      "datum-comment-ghost", "local-ghost", "string-ghost",
                      "quoted-ghost", "local-only", "datum-local-only"):
            self.assertNotIn(ghost, definitions | exports | requires)

    def test_stdlib_closure_uses_exports_and_root_definitions_with_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "lib/core").mkdir(parents=True)
            (root / "lib/stdlib.esk").write_text('''
; (require core.line-ghost)
#| (require core.block-ghost) |#
#; (require core.datum-ghost)
(define (stdlib-root-api x)
  (let () (define local-ghost 0) x))
(define source-text "(define string-ghost 1) (require core.string-ghost)")
'(require core.quoted-ghost)
(require core.alpha)
''')
            (root / "lib/core/alpha.esk").write_text('''
(provide alpha-public)
(require (only core.beta beta-public))
(define (alpha-public x)
  (let () (define alpha-local-ghost 0) x))
(define alpha-private-helper 1)
'(provide quoted-export-ghost)
''')
            (root / "lib/core/beta.esk").write_text('''
#| (provide block-export-ghost)
   (define (block-export-ghost) 0) |#
(provide beta-public)
(define (beta-public) 42)
''')

            result = SURFACE.extract_stdlib_surface(str(root))
            names = set(result["names"])
            self.assertTrue({"stdlib-root-api", "source-text", "alpha-public",
                             "beta-public"}.issubset(names))
            self.assertFalse({"alpha-private-helper", "alpha-local-ghost",
                              "line-ghost", "block-ghost", "datum-ghost",
                              "string-ghost", "quoted-ghost",
                              "quoted-export-ghost", "block-export-ghost"} & names)
            modules = {item["module"]: item for item in result["modules"]}
            self.assertEqual({"stdlib", "core.alpha", "core.beta"}, set(modules))
            self.assertEqual(["alpha-public"],
                             [e["name"] for e in modules["core.alpha"]["exports"]])
            self.assertEqual(["alpha-private-helper", "alpha-public"],
                             [e["name"] for e in modules["core.alpha"]["top_level_definitions"]])
            self.assertEqual("module_export", result["names"]["alpha-public"][0]["kind"])
            self.assertEqual("lib/core/alpha.esk", result["names"]["alpha-public"][0]["path"])
        self.assertTrue(all(export["definition_origin"]
                            for module in result["modules"]
                            for export in module["exports"]))

    def test_generation_is_independent_of_legacy_compiler(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "scripts").mkdir()
            shutil.copy(REPO / "scripts/gen_language_surface.py", root / "scripts")
            module_surface = SURFACE.extract_stdlib_surface(str(REPO))
            inputs = {
                "lib/backend/eshkol_vm.c", "lib/backend/llvm_codegen.cpp",
                "lib/backend/vm_prelude_source.h", "lib/frontend/parser.cpp",
                "inc/eshkol/eshkol.h", "lib/agent/quantum.esk",
                "lib/agent/pqc.esk",
            }
            inputs.update(entry["path"] for entry in module_surface["modules"])
            for relative in inputs:
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

    def test_live_builtin_ids_and_stdlib_provenance_are_source_grounded(self):
        manifest = SURFACE.build_manifest()
        vm = SURFACE.extract_builtin_table(SURFACE.VM_C)
        self.assertNotIn("native_builtins", manifest["_meta"]["sources"])
        for row in manifest["builtins"]:
            self.assertNotIn("native", row["backends"])
            if "vm" in row["backends"]:
                members = [row["name"]] + row.get("aliases", [])
                source_ids = sorted({nid for name in members
                                     for nid in vm.get(name, {}).get("ids", [])})
                self.assertEqual(source_ids, row["ids"])
                if any(name in vm for name in members):
                    self.assertIsNotNone(row["arity"])
                else:
                    vm_bootstrap = SURFACE.extract_vm_prelude_defs()
                    self.assertTrue(any(name in vm_bootstrap for name in members))

        closure = {item["module"]: item
                   for item in manifest["_meta"]["sources"]["stdlib_module_closure"]}
        self.assertIn("stdlib", closure)
        self.assertIn("core.io", closure)
        for entry in manifest["prelude"]:
            if "stdlib" in entry["sources"]:
                self.assertTrue(entry["provenance"])
                for source in entry["provenance"]:
                    self.assertIn(source["module"], closure)
                    self.assertTrue(source["path"].startswith("lib/"))
                    self.assertGreater(source["line"], 0)
        for item in closure.values():
            self.assertTrue(item["path"].startswith("lib/"))
            for exported in item["exports"]:
                self.assertTrue(exported["name"])
                self.assertGreater(exported["line"], 0)
                self.assertTrue(exported["definition_origin"])

        sys.path.insert(0, str(REPO / "scripts/p8"))
        import five_way_surface
        present, _, _, _, native_prelude = five_way_surface.collect_manifest(str(REPO))
        vm_only = next(entry["name"] for entry in manifest["prelude"]
                       if entry["sources"] == ["vm_bootstrap"])
        stdlib_only = next(entry["name"] for entry in manifest["prelude"]
                           if entry["sources"] == ["stdlib"])
        self.assertNotIn(vm_only, native_prelude)
        self.assertIn(vm_only, present)
        self.assertIn(stdlib_only, native_prelude)


if __name__ == "__main__":
    unittest.main()
