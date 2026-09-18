#!/usr/bin/env python3
"""Source gate for ADR-0020: AST string payloads have exactly one owner.

Every producer of AST string payloads allocates through
inc/eshkol/frontend/ast_strings.h, and no consumer frees one. This gate reads
the translation units that build or rewrite AST nodes and fails on:

  * a raw ``new char[...]``, ``strdup(...)`` or ``strndup(...)``: an
    allocation that bypasses the owner;
  * a ``delete[]``/``free()`` of an AST string field: a consumer releasing
    storage it does not own (and, before the owner existed, the source of
    alloc-dealloc mismatches).

tests/frontend/ast_strings_test.cpp checks the same rule dynamically on real
parser, expander, rename and copy output; this gate catches a regression in a
path that test does not reach, before it can surface as a leak report.

Exit 0 when clean, 1 with one line per violation otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Every translation unit that produces or rewrites AST string payloads.
PRODUCERS = (
    "lib/frontend/parser.cpp",
    "lib/frontend/macro_expander.cpp",
    "lib/core/module_visibility.cpp",
    "lib/core/ast.cpp",
    "lib/core/sexp_to_ast.cpp",
    "lib/core/introspection.cpp",
    "lib/types/type_checker.cpp",
    "lib/repl/repl_jit.cpp",
    "exe/eshkol-run.cpp",
)

RAW_ALLOC = re.compile(r"new\s+char\s*\[|(?<![\w])str(?:n)?dup\s*\(")
AST_FIELD = (
    r"(?:variable\.id|str_val\.ptr|rest_param|eshkol_func\.id|var_name|type_vars"
    r"|\b(?:define_op|set_op|let_op|lambda_op|extern_op|extern_var_op"
    r"|type_annotation_op|define_type_op|guard_op|with_region_op|diff_op"
    r"|import_op|require_op|provide_op|let_values_op|logic_var_op)\.\w+"
    r"|->name|->identifier|->literals)"
)
RAW_RELEASE = re.compile(
    r"(?:delete\s*\[\s*\]\s*|(?<![\w])free\s*\(\s*(?:\(\s*void\s*\*\s*\)\s*)?)[^;]*" + AST_FIELD
)


def code_part(line: str) -> str:
    """Drop // comments and whole-line block-comment text."""
    stripped = line.lstrip()
    if stripped.startswith(("*", "/*", "//")):
        return ""
    cut = line.find("//")
    return line if cut < 0 else line[:cut]


def scan_text(rel: str, text: str) -> list[str]:
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        code = code_part(line)
        if not code:
            continue
        if RAW_ALLOC.search(code) and "eshkol_ast_str" not in code:
            problems.append(
                f"{rel}:{number}: raw string allocation bypasses the AST string owner: {line.strip()}"
            )
        if RAW_RELEASE.search(code):
            problems.append(
                f"{rel}:{number}: AST string payload released by a consumer: {line.strip()}"
            )
    return problems


def scan(root: Path) -> list[str]:
    problems: list[str] = []
    for rel in PRODUCERS:
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: producer file missing (update PRODUCERS)")
            continue
        problems.extend(scan_text(rel, path.read_text(encoding="utf-8")))
    return problems


SELF_TEST_BAD = {
    "new char": "    ast.variable.id = new char[len + 1];",
    "strdup": "    op->set_op.name = strdup(name);",
    "strndup": "    p = strndup(s, n);",
    "delete[] field": "    delete[] ast->variable.id;",
    "free field": "    free(result.variable.id);",
    "free cast field": "    free((void*)var_ast.variable.id);",
    "free macro name": "    free(macro->name);",
}
SELF_TEST_GOOD = (
    "    ast.variable.id = eshkol_ast_strdup(name);",
    "    slot = eshkol_ast_string_copy(name);",
    "    // a comment that says strdup() and new char[] is fine",
    " * `strdup()`ed here -- block comment text is not code",
    "    delete[] ast.operation.call_op.variables;  // node storage, not a string",
    "    free(bindings);",
)


def self_test() -> int:
    failures = 0
    for label, line in SELF_TEST_BAD.items():
        if not scan_text("self-test", line):
            print(f"self-test: planted violation not caught: {label}")
            failures += 1
    for line in SELF_TEST_GOOD:
        found = scan_text("self-test", line)
        if found:
            print(f"self-test: false positive: {found[0]}")
            failures += 1
    if failures:
        print(f"FAIL: AST string owner gate self-test ({failures})")
        return 1
    print(f"PASS: AST string owner gate self-test ({len(SELF_TEST_BAD)} planted, "
          f"{len(SELF_TEST_GOOD)} clean)")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=Path(__file__).resolve().parent.parent, type=Path)
    parser.add_argument("--self-test", action="store_true",
                        help="prove the patterns catch planted violations and pass clean code")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    problems = scan(args.root)
    if problems:
        for p in problems:
            print(p)
        print(f"FAIL: AST string owner gate ({len(problems)} violation(s)); see ADR-0020")
        return 1
    print(f"PASS: AST string owner gate ({len(PRODUCERS)} producer files clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
