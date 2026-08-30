#!/usr/bin/env python3
"""Check that the parser binding registry is complete and backend-shared."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "inc/eshkol/frontend/binding_forms.h"
OBSERVATION = ROOT / "inc/eshkol/backend/mutation_observation.h"


def load_rows() -> list[tuple[str, str, str]]:
    text = REGISTRY.read_text(encoding="utf-8")
    match = re.search(
        r"#define ESHKOL_PARSER_BINDING_FORM_TABLE\(X\) \\\n(?P<body>.*?)(?=\n\n|\n\nenum)",
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError("parser binding-form table is missing")
    rows = re.findall(r"(?:^|\n)\s*X\(([^,]+),\s*\"([^\"]+)\",\s*([^\)]+)\) ?\\?", match.group("body"))
    if not rows:
        raise ValueError("parser binding-form table has no rows")
    return [(ident.strip(), spelling, flags.strip()) for ident, spelling, flags in rows]


def check() -> tuple[bool, str]:
    rows = load_rows()
    ids = [row[0] for row in rows]
    spellings = [row[1] for row in rows]
    if len(ids) != len(set(ids)):
        return False, "parser binding-form table has duplicate ids"
    if len(spellings) != len(set(spellings)):
        return False, "parser binding-form table has duplicate spellings"

    registry = REGISTRY.read_text(encoding="utf-8")
    observation = OBSERVATION.read_text(encoding="utf-8")
    if "ESHKOL_PARSER_BINDING_FORM_TABLE(X)" not in observation:
        return False, "mutation observation table is not generated from parser registry"
    if "ESHKOL_PARSER_BINDING_FORM_COUNT" not in observation:
        return False, "mutation observation count is not tied to parser registry count"
    if "ESHKOL_PARSER_BINDING_FORM_TABLE(ESHKOL_PARSER_BINDING_FORM_ENUM)" not in registry:
        return False, "parser binding-form enum is not generated from the registry"
    return True, f"parser binding-form table complete: {len(rows)} entries; observation table shares registry"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="run the table completeness check")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("--self-test is required")
    try:
        ok, detail = check()
    except (OSError, ValueError) as exc:
        ok, detail = False, str(exc)
    print(f"check_binding_form_table.py self-test: {'PASS' if ok else 'FAIL'}")
    print(detail)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
