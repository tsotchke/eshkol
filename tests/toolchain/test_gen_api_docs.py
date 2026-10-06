#!/usr/bin/env python3
"""Focused controls for C/C++ declaration classification in API docs."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gen_api_docs", ROOT / "scripts" / "gen_api_docs.py"
)
gen_api_docs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gen_api_docs
SPEC.loader.exec_module(gen_api_docs)


class DeclarationClassificationTest(unittest.TestCase):
    def test_statement_macros_are_not_functions_but_macro_definitions_remain(self):
        for statement in (
            'ESHKOL_ABI_ASSERT(sizeof(T) == 8, "layout changed");',
            'static_assert(sizeof(T) == 8, "layout changed");',
            'CHECK_LAYOUT(sizeof(T));',
        ):
            with self.subTest(statement=statement):
                self.assertIsNone(gen_api_docs.classify_chunk(statement))

        symbol, _ = gen_api_docs.classify_chunk(
            "#define ESHKOL_ABI_ASSERT(cond, msg) static_assert(cond, msg)"
        )
        self.assertEqual((symbol.kind, symbol.name), ("macro", "ESHKOL_ABI_ASSERT"))

    def test_prototypes_return_pointer_and_annotated_declarator_are_functions(self):
        for declaration, expected in (
            ("int parse_value(const char *text);", "parse_value"),
            ("void *allocate_value(size_t bytes);", "allocate_value"),
            ("EXPORT_TYPE(int) exported_value(double input);", "exported_value"),
            ("__declspec(dllexport) int windows_value(void);", "windows_value"),
        ):
            with self.subTest(declaration=declaration):
                result = gen_api_docs.classify_chunk(declaration)
                self.assertIsNotNone(result)
                self.assertEqual((result[0].kind, result[0].name), ("function", expected))

    def test_balanced_declaration_heads_reject_calls_and_resolve_grouped_names(self):
        controls = (
            ("CHECK_LAYOUT (sizeof(T));", None, None),
            ("Other ();", "Widget", None),
            ('[[deprecated("path/why?!")]] int real_api(void);', None,
             ("function", "real_api")),
            ("Array<(1+2)> real_api(void);", None, ("function", "real_api")),
            ("Array<(sizeof(T) < 8)> sized(void);", None,
             ("function", "sized")),
            ("Array<(sizeof(T) > 8)> sized(void);", None,
             ("function", "sized")),
            ("Array<(sizeof(T) >= 8)> sized(void);", None,
             ("function", "sized")),
            ("int (real_api)(double);", None, ("function", "real_api")),
            ("int stored = real_api(1.0);", None, ("variable", "stored")),
            ('[[deprecated("path=why")]] int stored = real_api(1.0);', None,
             ("variable", "stored")),
            ("Array<(1==2)> stored = factory();", None,
             ("variable", "stored")),
            ("int (*factory(void))(double);", None, ("function", "factory")),
            ("int (*factory(void (*callback)(int)))(double);", None,
             ("function", "factory")),
            ("int (* const callback)(double);", None,
             ("variable", "callback")),
            ("Widget& operator=(const Widget&) = delete;", "Widget",
             ("function", "operator=")),
            ("ns::factory();", None, None),
            ("int ns::factory();", None, ("function", "ns::factory")),
            ("Array<(1+2)>\n real_api (void);", None,
             ("function", "real_api")),
            ("Map<int, Array<(3+4)>> convert(\n"
             "    Map<int, Array<(5+6)>> value = fallback(choose(1, 2)));",
             None, ("function", "convert")),
        )
        for declaration, scope, expected in controls:
            with self.subTest(declaration=declaration):
                result = gen_api_docs.classify_chunk(
                    declaration, class_scope=scope
                )
                observed = (result[0].kind, result[0].name) if result else None
                self.assertEqual(observed, expected)

    def test_function_pointer_variable_stays_a_variable(self):
        result = gen_api_docs.classify_chunk("extern int (*callback)(double);")
        self.assertIsNotNone(result)
        self.assertEqual((result[0].kind, result[0].name), ("variable", "callback"))

        result = gen_api_docs.classify_chunk(
            "Task* submit(void* (*callback)(void*), void* arg);"
        )
        self.assertIsNotNone(result)
        self.assertEqual((result[0].kind, result[0].name), ("function", "submit"))

    def test_matching_and_qualified_constructors_and_destructor_are_functions(self):
        for declaration, scope in (
            ("Widget();", "Widget"),
            ("~Widget();", "Widget"),
            ("Widget::Widget();", None),
        ):
            with self.subTest(declaration=declaration):
                result = gen_api_docs.classify_chunk(declaration, class_scope=scope)
                self.assertIsNotNone(result)
                self.assertEqual(result[0].kind, "function")

        self.assertIsNone(gen_api_docs.classify_chunk("Other();", class_scope="Widget"))

    def test_header_parser_preserves_constructor_docs_and_omits_assertions(self):
        source = '''\
/** @brief A widget. */
class Widget {
/** @brief Make a widget. */
Widget();
/** @brief Destroy a widget. */
~Widget();
};
/** @brief Out-of-class constructor. */
Widget::Widget();
ESHKOL_ABI_ASSERT(sizeof(Widget) > 0, "layout changed");
'''
        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / "widget.h"
            header.write_text(source, encoding="utf-8")
            symbols, _ = gen_api_docs.parse_header(header)

        functions = [symbol for symbol in symbols if symbol.kind == "function"]
        self.assertEqual([symbol.name for symbol in functions], [
            "Widget::Widget", "Widget::~Widget", "Widget::Widget",
        ])
        self.assertTrue(all(symbol.documented for symbol in functions))
        self.assertFalse(any(symbol.name == "ESHKOL_ABI_ASSERT" for symbol in symbols))

    def test_real_abi_header_keeps_assert_macro_definitions_without_counting_calls(self):
        symbols, _ = gen_api_docs.parse_header(ROOT / "inc/eshkol/abi_fingerprint.h")
        assertions = [symbol for symbol in symbols if symbol.name == "ESHKOL_ABI_ASSERT"]
        self.assertEqual(len(assertions), 3)
        self.assertTrue(all(symbol.kind == "macro" for symbol in assertions))


if __name__ == "__main__":
    unittest.main()
