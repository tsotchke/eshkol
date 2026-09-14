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


if __name__ == "__main__":
    unittest.main()
