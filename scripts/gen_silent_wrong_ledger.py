#!/usr/bin/env python3
"""Generate the compatibility silent-wrong ledger from per-entry files.

The checked-in aggregate remains the input consumed by existing gates.  The
split files are the editable source: one YAML mapping per entry in
``.icc/ledger/entries`` and the top-level preamble in ``.icc/ledger/meta.yaml``.
The entry text is rendered without YAML re-serialization so the migration can
prove a byte-for-byte match with the historical aggregate.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


ENTRY_RE = re.compile(r"^id:\s*(\S+)\s*$", re.MULTILINE)


def paths(repo: Path) -> tuple[Path, Path, Path, Path]:
    root = repo / ".icc" / "ledger"
    return root / "meta.yaml", root / "entries", root / "order.txt", repo / ".icc" / "silent-wrong-ledger.yaml"


def read_entries(directory: Path) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: entry is not a YAML mapping")
        entry_id = data.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError(f"{path}: entry has no non-empty string id")
        if entry_id in seen:
            raise ValueError(f"duplicate ledger id: {entry_id}")
        seen.add(entry_id)
        if path.stem != entry_id:
            raise ValueError(f"{path}: filename must be {entry_id}.yaml")
        if not text.endswith("\n"):
            text += "\n"
        result.append((entry_id, text))
    if not result:
        raise ValueError(f"no ledger entries found under {directory}")
    return result


def render(meta: Path, entry_dir: Path, order_path: Path) -> bytes:
    preamble = meta.read_text(encoding="utf-8")
    if not preamble.endswith("\n"):
        preamble += "\n"
    if "\nentries:\n" in preamble:
        raise ValueError(f"{meta}: meta.yaml must not contain an entries key")
    entries = read_entries(entry_dir)
    by_id = dict(entries)
    order = order_path.read_text(encoding="utf-8").splitlines()
    if set(order) != set(by_id) or len(order) != len(by_id):
        raise ValueError(f"{order_path}: must list each entry id exactly once")
    # The aggregate's historical order is intentional evidence order.  Keep it
    # in a separate stable manifest so adding a file never rewrites old bytes.
    body = "entries:\n"
    for entry_id in order:
        text = by_id[entry_id]
        lines = text.splitlines()
        id_line = next(i for i, line in enumerate(lines) if line.startswith("id:"))
        body += "".join(("\n" if not line else "  " + line + "\n") for line in lines[:id_line])
        body += "  - " + lines[id_line] + "\n"
        body += "".join(("\n" if not line else "    " + line + "\n") for line in lines[id_line + 1:])
    return (preamble + body).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="fail if the aggregate is stale")
    args = parser.parse_args()
    meta, entry_dir, order_path, aggregate = paths(args.repo_root)
    try:
        generated = render(meta, entry_dir, order_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.check:
        actual = aggregate.read_bytes() if aggregate.exists() else b""
        if actual != generated:
            print(f"{aggregate.relative_to(args.repo_root)} is stale", file=sys.stderr)
            return 1
        print("silent-wrong ledger aggregate is up to date.")
        return 0
    aggregate.write_bytes(generated)
    print(f"Wrote {aggregate.relative_to(args.repo_root)} from {len(read_entries(entry_dir))} entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
