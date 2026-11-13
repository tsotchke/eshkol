#!/usr/bin/env python3
"""An iterator is compared only against the end() of the container it came from.

`it != a.end() || it != b.end()` (or the `==`/`&&` forms) tests one iterator
against two containers. Whatever it was meant to say, it cannot say it: at most
one of the two comparisons is defined, and the lookup result of the other
container is not consulted at all. Each container is searched with its own
iterator and that iterator is compared with that container's end().

Usage: iterator_end_comparison_test.py <repo-root>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERN = re.compile(
    r"\b(\w+)\s*(?:!=|==)\s*([\w.>\-]+?)(?:\.|->)end\(\)\s*(?:\|\||&&)\s*"
    r"\1\s*(?:!=|==)\s*([\w.>\-]+?)(?:\.|->)end\(\)"
)
SUFFIXES = {".cpp", ".h", ".hpp", ".c", ".cc", ".mm"}


def findings(root: Path) -> list[str]:
    out = []
    for top in ("lib", "inc", "exe"):
        for path in sorted((root / top).rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in PATTERN.finditer(text):
                if match.group(2) == match.group(3):
                    continue
                line = text.count("\n", 0, match.start()) + 1
                out.append(f"{path.relative_to(root)}:{line}: {match.group(0)}")
    return out


def self_test() -> None:
    assert PATTERN.search("if (it != global_symbol_table.end() || it != symbol_table.end()) {")
    assert PATTERN.search("if (it == a->end() && it == b.end())")
    assert not PATTERN.search("if (a_it != a.end() || b_it != b.end())")


def main() -> int:
    self_test()
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    found = findings(root)
    for item in found:
        print(f"FAIL: one iterator compared against two containers: {item}")
    if found:
        return 1
    print("PASS: every iterator is compared against its own container's end()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
