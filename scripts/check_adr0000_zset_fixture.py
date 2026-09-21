#!/usr/bin/env python3
"""Deterministic reference serialization for the ADR-0000 Stage 1 gate."""

from pathlib import Path
import hashlib
import sys


def canonical_bytes(path: Path) -> bytes:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        fields = raw.split("|")
        if len(fields) != 3 or not fields[0] or not fields[1]:
            raise ValueError(f"malformed reference row: {raw}")
        rows.append((fields[1], int(fields[2])))
    consolidated = {}
    for row, weight in rows:
        consolidated[row] = consolidated.get(row, 0) + weight
    return "".join(f"{row}|{weight}\n" for row, weight in sorted(consolidated.items()) if weight).encode()


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("tests/adr0000/zset_reference.tsv")
    first = canonical_bytes(path)
    second = canonical_bytes(path)
    if first != second:
        print("FAIL: Z-set canonical serialization is not byte-stable")
        return 1
    print("PASS: Z-set canonical serialization")
    print("  sha256: " + hashlib.sha256(first).hexdigest())
    print(first.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
