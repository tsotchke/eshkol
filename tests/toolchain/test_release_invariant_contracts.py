#!/usr/bin/env python3
"""Negative controls for the source contracts graded by ICC release invariants."""
from pathlib import Path
import re
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL = yaml.safe_load((ROOT / ".icc/architecture-model.yaml").read_text())


def grade(name, replacements=None):
    invariant = next(i for i in MODEL["invariants"] if i["id"] == name)
    spaces = []
    for site in invariant["sites"]:
        text = (replacements or {}).get(site["path"], (ROOT / site["path"]).read_text())
        spaces.append(set(re.findall(site["key_pattern"], text)))
    return bool(spaces) and all(spaces) and all(s == spaces[0] for s in spaces[1:])


def capability(name):
    return next(c for c in MODEL["capabilities"] if c["id"] == name)


def text(path, replacements=None):
    return (replacements or {}).get(path, (ROOT / path).read_text())


class IdentityTests(unittest.TestCase):
    name = "INV-node-identity-single-substrate"

    def test_actual_field_allocator_and_query_keys_agree(self):
        self.assertTrue(grade(self.name))

    def test_ast_field_type_fork_is_rejected(self):
        path = "inc/eshkol/eshkol.h"
        source = (ROOT / path).read_text().replace("eshkol_node_id_t node_id", "uint64_t node_id")
        self.assertFalse(grade(self.name, {path: source}))

    def test_each_semantic_query_must_use_the_parser_key(self):
        path = "inc/eshkol/frontend/semantic_identity.h"
        original = (ROOT / path).read_text()
        for query in ("eshkol_binding_id_for_node", "eshkol_typed_expr_info"):
            with self.subTest(query=query):
                source = original.replace(query + "(eshkol_node_id_t node_id", query + "(eshkol_binding_id_t node_id")
                self.assertNotEqual(source, original)
                self.assertFalse(grade(self.name, {path: source}))

    def test_allocator_type_fork_or_missing_declaration_is_rejected(self):
        path = "inc/eshkol/frontend/node_identity.h"
        source = (ROOT / path).read_text()
        changed = source.replace("eshkol_node_id_t eshkol_node_id_new", "uint64_t eshkol_node_id_new")
        self.assertNotEqual(changed, source)
        self.assertFalse(grade(self.name, {path: changed}))
        self.assertFalse(grade(self.name, {path: ""}))


class DeepWalkTests(unittest.TestCase):
    name = "INV-oalr-interior-pointer-deepwalk"
    header = "inc/eshkol/eshkol.h"
    runtime = "lib/core/runtime_regions.cpp"

    def test_actual_canonical_enum_matches_actual_deep_walk_dispatch(self):
        self.assertTrue(grade(self.name))

    def test_last_grouped_leaf_is_not_incorrectly_reported_as_deep_walk(self):
        invariant = next(i for i in MODEL["invariants"] if i["id"] == self.name)
        site = invariant["sites"][1]
        source = (ROOT / site["path"]).read_text()
        keys = set(re.findall(site["key_pattern"], source))
        self.assertNotIn("SDNC", keys)
        self.assertIn("CONS", keys)

    def test_changing_a_deep_walk_to_a_leaf_is_rejected(self):
        original = (ROOT / self.runtime).read_text()
        source = original.replace("case HEAP_SUBTYPE_CONS:        return EVAC_CONS;",
                                  "case HEAP_SUBTYPE_CONS:        return EVAC_LEAF;")
        self.assertNotEqual(original, source)
        self.assertFalse(grade(self.name, {self.runtime: source}))

    def test_new_interior_pointer_subtype_requires_real_dispatch(self):
        source = (ROOT / self.header).read_text() + "\nHEAP_SUBTYPE_NEW_INTERIOR = 200, // [DEEPWALK]\n"
        self.assertFalse(grade(self.name, {self.header: source}))


class ReleaseInvariantContractTests(unittest.TestCase):
    def test_ad_counter_spec_is_armed_and_its_live_negative_control_is_pinned(self):
        spec = capability("ad_counter_measurement")
        self.assertEqual(spec["arming"]["kind"], "runtime_event")
        self.assertEqual(spec["arming"]["path"], "scripts/run_icc_smoke.sh")
        self.assertEqual(spec["pattern"], r"(?m)^probe ad_exactness_gate")
        self.assertEqual(
            spec["dependency_constructor"],
            r'run_case "fd-counter" tests/ad/fd_counter_negative_test\.esk'
        )

        smoke_path = "scripts/run_icc_smoke.sh"
        gate_path = "scripts/run_ad_exactness_gate.sh"
        test_path = "tests/ad/fd_counter_negative_test.esk"
        smoke = text(smoke_path)
        gate = text(gate_path)
        negative = text(test_path)
        self.assertRegex(smoke, spec["pattern"])
        self.assertRegex(gate, spec["dependency_constructor"])
        self.assertIn("run_one_pass_gradient_gate.sh", gate)
        self.assertIn("matmul_tape_node_count_test.esk", gate)
        self.assertIn("(= fd-count-after 4)", negative)
        self.assertIn("(not (= fd-count-after 0))", negative)

        invariant = next(
            i for i in MODEL["invariants"]
            if i["id"] == "INV-ad-counters-measure-real-events"
        )
        self.assertEqual(invariant["kind"], "intended-invariant")
        self.assertEqual(invariant["fidelity"], "runtime")
        self.assertEqual(invariant["evidence"]["trace_name_pattern"], "^ad_exactness_gate$")
        oracle = (ROOT / ".icc/completion-oracles.yaml").read_text()
        self.assertIn('event_names: ["ad_exactness_gate"]', oracle)
        self.assertIn("both engines", oracle)

        no_verification = negative.replace("(not (= fd-count-after 0))", "#f")
        self.assertNotEqual(no_verification, negative)
        self.assertNotIn("(not (= fd-count-after 0))", no_verification)
        no_negative_gate = gate.replace("tests/ad/fd_counter_negative_test.esk", "")
        self.assertNotEqual(no_negative_gate, gate)
        self.assertNotRegex(no_negative_gate, spec["dependency_constructor"])
        no_smoke_probe = smoke.replace("probe ad_exactness_gate", "")
        self.assertNotEqual(no_smoke_probe, smoke)
        self.assertNotRegex(no_smoke_probe, spec["pattern"])

    def test_package_manifest_spec_requires_receipt_and_release_verification_call(self):
        spec = capability("package_surface_manifest")
        self.assertEqual(spec["arming"]["path"], "scripts/check_package_manifest.py")
        self.assertEqual(spec["pattern"], "package_manifest_complete")
        workflow_path = ".github/workflows/release.yml"
        manifest_path = ".icc/package-manifest.yaml"
        workflow = text(workflow_path)
        manifest = text(manifest_path)
        checker = text("scripts/check_package_manifest.py")
        self.assertRegex(workflow, spec["dependency_constructor"])
        self.assertIn("package_surface:", manifest)
        self.assertIn('PROBE_ID = "package_manifest_complete"', checker)
        self.assertIn("tar -czf", workflow)
        self.assertIn("Compress-Archive", workflow)

        omitted_linux = workflow.replace("python3 scripts/check_package_manifest.py", "python3 check_package_manifest.py")
        self.assertNotEqual(omitted_linux, workflow)
        self.assertNotRegex(omitted_linux, spec["dependency_constructor"])
        omitted_windows = workflow.replace("python scripts/check_package_manifest.py", "python check_package_manifest.py")
        self.assertNotEqual(omitted_windows, workflow)
        self.assertNotRegex(omitted_windows, spec["dependency_constructor"])

    def test_ad_bridge_registry_matches_actual_definitions(self):
        name = "INV-ad-node-declared-in-registry"
        self.assertTrue(grade(name))
        registry_path = "inc/eshkol/ad_node_registry.def"
        registry = text(registry_path)
        removed_row, count = re.subn(
            r"(?m)^ESHKOL_AD_NODE\(TENSOR_MATMUL,[^\n]*\n", "", registry, count=1
        )
        self.assertEqual(count, 1)
        self.assertNotEqual(removed_row, registry)
        self.assertFalse(grade(name, {registry_path: removed_row}))

        definition_path = "lib/bridge/tensor_backward.cpp"
        definitions = text(definition_path)
        removed_definition = definitions.replace(
            'extern "C" void tensor_matmul_backward(ad_node_t* node) {',
            'extern "C" void tensor_matmul_backward(ad_node_t* node);', 1
        )
        self.assertNotEqual(removed_definition, definitions)
        self.assertFalse(grade(name, {definition_path: removed_definition}))

    def test_squared_distance_registration_uses_its_definition_not_dispatcher_declaration(self):
        name = "INV-ad-squared-distance-backward-defined-and-registered"
        self.assertTrue(grade(name))
        registry_path = "inc/eshkol/ad_node_registry.def"
        registry = text(registry_path)
        omitted = registry.replace(
            "ESHKOL_AD_NODE(SQUARED_DISTANCE,", "ESHKOL_AD_NODE(UNREGISTERED_SQUARED_DISTANCE,", 1
        )
        self.assertNotEqual(omitted, registry)
        self.assertFalse(grade(name, {registry_path: omitted}))


if __name__ == "__main__":
    unittest.main()
