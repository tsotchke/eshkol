#!/usr/bin/env python3
"""Check (or refresh) the reviewed-source pins of the strict resource contracts.

inc/eshkol/strict_resource_contracts.h trusts a module's handle contracts only
for the exact source they were reviewed against. A module edit keeps compiling
in the default profile, but `--strict-resources` refuses it until the contract
table is reviewed and the pin is refreshed:

  python3 scripts/strict_resource_pins.py --check    # CTest strict_resource_contract_pins
  python3 scripts/strict_resource_pins.py --update   # after reviewing the contract rows

The check also verifies that each module's provide list names exactly the
operations its contract table describes.
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "inc" / "eshkol" / "strict_resource_contracts.h"
ENTRY = re.compile(r'\{"(?P<module>[a-z._]+)", "(?P<source>[a-z_/]+\.esk)",\s*"(?P<sha>[0-9a-f]{64})",\s*'
                   r'(?P<table>k[A-Za-z]+Contracts)', re.S)


def table_names(text: str, table: str) -> set:
    start = text.index(f"inline constexpr Contract {table}[] = {{")
    end = text.index("};", start)
    return set(re.findall(r'^\s*\{"([^"]+)", Role::', text[start:end], re.M))


def provided_names(source: str) -> set:
    match = re.search(r"\(provide\s(.*?)\)", source, re.S)
    if not match:
        return set()
    body = re.sub(r";[^\n]*", "", match.group(1))
    return set(body.split())


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update", action="store_true")
    args = parser.parse_args()
    text = HEADER.read_text()
    entries = list(ENTRY.finditer(text))
    if not entries:
        print(f"FAIL no module contracts found in {HEADER.relative_to(ROOT)}")
        return 1
    failures = 0
    updated = text
    for entry in entries:
        source_path = ROOT / "lib" / entry["source"]
        source = source_path.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        names = table_names(text, entry["table"])
        provided = provided_names(source.decode("utf-8"))
        if names != provided:
            failures += 1
            print(f"FAIL {entry['module']}: provide list and contract table differ "
                  f"(only provided: {sorted(provided - names)}, only in table: {sorted(names - provided)})")
        if digest != entry["sha"]:
            if args.update:
                updated = updated.replace(entry["sha"], digest)
                print(f"UPDATED {entry['module']} {digest}")
            else:
                failures += 1
                print(f"FAIL {entry['module']}: lib/{entry['source']} is {digest}, the contract is pinned "
                      f"to {entry['sha']}; review the contract rows, then run "
                      f"scripts/strict_resource_pins.py --update")
        else:
            print(f"PASS {entry['module']} pinned {digest[:12]} ({len(names)} operations)")
    if args.update and updated != text:
        HEADER.write_text(updated)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
