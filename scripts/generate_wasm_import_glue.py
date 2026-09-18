#!/usr/bin/env python3
"""Generate and verify shared flat-AD imports in both browser WASM bundles.

The fragment is embedded into the two self-contained classic JS bundles. The
JSON contract records every core `eshkol_*` and `region_*` import; adapter-only
DOM keys are intentionally excluded. `--check` compares generated blocks and
the actual keys found by check_wasm_imports.extract_env_keys.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parent.parent
FRAGMENT = ROOT / "scripts" / "wasm_flat_ad_imports.fragment.js"
CONTRACT = ROOT / "scripts" / "wasm_core_import_keys.json"
JS_FILES = (
    ROOT / "web" / "eshkol-repl.js",
    ROOT / "site" / "static" / "eshkol-runtime.js",
)
BEGIN = "// BEGIN GENERATED FLAT-AD IMPORTS"
END = "// END GENERATED FLAT-AD IMPORTS"
BLOCK_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)" + re.escape(BEGIN)
    + r"\n(?P<body>.*?)\n(?P=indent)" + re.escape(END), re.DOTALL
)


def _extract_env_keys(js_text: str) -> set[str]:
    """Use the import scanner's real tokenizer, including its lexical checks."""
    spec = importlib.util.spec_from_file_location(
        "check_wasm_imports_for_glue", ROOT / "scripts" / "check_wasm_imports.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/check_wasm_imports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_env_keys(js_text)


def _core_keys(keys: set[str]) -> set[str]:
    return {key for key in keys if key.startswith(("eshkol_", "region_"))}


def expected_block() -> str:
    indent = "                "
    fragment = FRAGMENT.read_text().rstrip().splitlines()
    return "\n".join(
        [indent + BEGIN]
        + [indent + line for line in fragment]
        + [indent + END]
    )


def check(
    js_text_by_path: Mapping[Path, str] | None = None,
) -> list[str]:
    """Return generated-block and core-contract failures without modifying files."""
    failures: list[str] = []
    try:
        contract_data = json.loads(CONTRACT.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read core-key contract {CONTRACT}: {exc}"]
    if not isinstance(contract_data, list) or any(
        not isinstance(key, str) for key in contract_data
    ):
        return [f"invalid core-key contract {CONTRACT}: expected a JSON string array"]
    contract = set(contract_data)
    if len(contract) != len(contract_data):
        failures.append("core-key contract contains duplicate keys")
    if contract != _core_keys(contract):
        failures.append("core-key contract has a non-core key (expected eshkol_/region_ prefix)")

    block = expected_block()
    by_path = js_text_by_path or {path: path.read_text() for path in JS_FILES if path.exists()}
    extracted_core: dict[Path, set[str]] = {}
    for path in JS_FILES:
        label = path.relative_to(ROOT).as_posix()
        if path not in by_path:
            failures.append(f"required glue file missing: {label}")
            continue
        js_text = by_path[path]
        blocks = list(BLOCK_RE.finditer(js_text))
        if len(blocks) != 1:
            failures.append(f"{label}: expected one generated flat-AD block, found {len(blocks)}")
        elif blocks[0].group(0) != block:
            failures.append(f"{label}: generated flat-AD block is stale; run generate_wasm_import_glue.py --write")
        keys = _core_keys(_extract_env_keys(js_text))
        extracted_core[path] = keys
        missing = sorted(contract - keys)
        extra = sorted(keys - contract)
        if missing:
            failures.append(f"{label}: missing core contract keys: {', '.join(missing)}")
        if extra:
            failures.append(f"{label}: core keys absent from contract: {', '.join(extra)}")
    if len(extracted_core) == len(JS_FILES):
        first, second = (extracted_core[path] for path in JS_FILES)
        if first != second:
            failures.append("the two glue files provide different eshkol_/region_ import sets")
    return failures


def write() -> list[str]:
    """Replace each marked region with the canonical fragment."""
    block = expected_block()
    failures: list[str] = []
    for path in JS_FILES:
        if not path.exists():
            failures.append(f"required glue file missing: {path.relative_to(ROOT)}")
            continue
        text = path.read_text()
        matches = list(BLOCK_RE.finditer(text))
        if len(matches) != 1:
            failures.append(
                f"{path.relative_to(ROOT)}: expected one generated flat-AD block, found {len(matches)}"
            )
            continue
        path.write_text(BLOCK_RE.sub(lambda _match: block, text, count=1))
    return failures


def selftest() -> list[str]:
    """Pin stale-generated-block and missing-contract-key rejection."""
    failures: list[str] = []
    baseline = {path: path.read_text() for path in JS_FILES if path.exists()}
    if len(baseline) != len(JS_FILES):
        return ["required glue file missing during generator self-test"]
    for failure in check(baseline):
        failures.append(f"checked-in generated glue rejected: {failure}")
    first = JS_FILES[0]
    stale = dict(baseline)
    stale[first] = baseline[first].replace("eshkol_ad_tower_carry_result: () => 0,", "eshkol_ad_tower_carry_result: () => 1,", 1)
    if not any("generated flat-AD block is stale" in f for f in check(stale)):
        failures.append("deliberately stale generated flat-AD block was accepted")
    missing = dict(baseline)
    missing[first] = baseline[first].replace("eshkol_ad_nested_capture_unsupported: () => {", "eshkol_ad_nested_capture_unsupported_REMOVED: () => {", 1)
    if not any("missing core contract keys" in f for f in check(missing)):
        failures.append("deliberately missing core import key was accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="verify blocks and actual core keys")
    group.add_argument("--write", action="store_true", help="regenerate the marked blocks")
    group.add_argument("--selftest", action="store_true", help="run positive and negative checks")
    args = parser.parse_args()
    if args.write:
        failures = write()
        if failures:
            for failure in failures:
                print(f"error: {failure}", file=sys.stderr)
            return 1
    failures = selftest() if args.selftest else check()
    if failures:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        return 1
    print("OK — shared flat-AD block and core-key contract are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
