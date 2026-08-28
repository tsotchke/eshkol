#!/usr/bin/env python3
"""Fail-closed DD-11 coverage gate for reviewed public API documentation."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADER_MANIFEST = REPO_ROOT / "docs" / "api" / "public_surface.tsv"
EXPORT_MANIFEST = REPO_ROOT / "docs" / "reference" / "stdlib" / "public_exports.tsv"
TRACE_NAME = "public_api_docs_clean"


def emit_trace(path: Path, status: str, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "kind": "eshkol_smoke",
        "name": TRACE_NAME,
        "value": status,
        "snippet": detail[:2000],
        "confidence": 1.0,
    }) + "\n", encoding="utf-8")


def read_manifest(path: Path, expected_fields: int) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = tuple(raw.split("\t"))
        if len(fields) != expected_fields:
            raise ValueError(f"{path}: expected {expected_fields} tab-separated fields: {raw}")
        rows.append(fields)
    return rows


def slug(symbol: str) -> str:
    """Match the stable GitHub-style slug used by the Markdown headings."""
    value = symbol.lower()
    value = re.sub(r"[^a-z0-9._-]+", "", value)
    return value


def has_anchor(doc: str, symbol: str) -> bool:
    anchor = slug(symbol)
    explicit = re.search(rf'<a\s+id=["\']{re.escape(anchor)}["\']\s*/?>', doc, re.I)
    heading = re.search(rf"^#{{1,6}}\s+`{re.escape(symbol)}`\s*$", doc, re.M)
    return bool(explicit or heading)


def has_index_link(index: str, symbol: str, target: str) -> bool:
    label = re.escape(symbol)
    link = re.escape(target)
    return bool(re.search(rf"(?:\[`{label}`\]|\[{label}\])\({link}\)", index))


def check_manifest(
    rows: list[tuple[str, ...]],
    expected_count: int,
    doc_root: Path,
    index_path: Path,
    doc_path: str,
) -> list[str]:
    errors: list[str] = []
    if len(rows) != expected_count:
        errors.append(f"{doc_path}: manifest has {len(rows)} rows, expected {expected_count}")
    symbols = [row[3] for row in rows]
    for symbol in sorted({name for name in symbols if symbols.count(name) > 1}):
        errors.append(f"duplicate manifest symbol: {symbol}")

    index = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    doc = doc_root / doc_path
    doc_text = doc.read_text(encoding="utf-8") if doc.exists() else ""
    for row in rows:
        path, line, kind, symbol = row[:4]
        source = doc_root / path
        if not source.exists():
            errors.append(f"missing source: {path}")
            continue
        source_text = source.read_text(encoding="utf-8", errors="replace")
        source_lines = source_text.splitlines()
        line_number = int(line)
        if kind != "module" and symbol not in source_text:
            errors.append(f"source does not contain {symbol}: {path}:{line}")
        elif not (1 <= line_number <= len(source_lines)):
            errors.append(f"source location is out of range for {symbol}: {path}:{line}")
        if not doc.exists():
            errors.append(f"missing documentation page: {doc_path}")
        elif not has_anchor(doc_text, symbol):
            errors.append(f"missing documentation anchor: {doc_path}#{slug(symbol)}")
        target = f"{Path(doc_path).name}#{slug(symbol)}"
        if not has_index_link(index, symbol, target):
            errors.append(f"missing index link: {index_path.relative_to(REPO_ROOT)} -> {target}")
    return errors


def self_test() -> int:
    sample = '<a id="foo"></a>\n### `bar?`\n'
    assert has_anchor(sample, "foo")
    assert has_anchor(sample, "bar?")
    assert not has_anchor(sample, "missing")
    assert has_index_link("[`bar?`](x#bar)", "bar?", "x#bar")
    print("check_public_api_docs self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--trace-dir", type=Path, default=REPO_ROOT / "scripts" / "icc_traces")
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    root = args.repo_root
    header_manifest = root / HEADER_MANIFEST.relative_to(REPO_ROOT)
    export_manifest = root / EXPORT_MANIFEST.relative_to(REPO_ROOT)
    try:
        headers = read_manifest(header_manifest, 5)
        exports = read_manifest(export_manifest, 4)
    except (OSError, ValueError) as exc:
        print(f"public_api_docs: FAIL: {exc}", file=sys.stderr)
        return 1

    errors = check_manifest(
        headers, 68, root, root / "docs/api/INDEX.md", "docs/api/public_surface.md"
    )
    errors.extend(check_manifest(
        exports, 59, root, root / "docs/reference/stdlib/INDEX.md",
        "docs/reference/stdlib/shipped_exports.md"
    ))
    if errors:
        print("public_api_docs: FAIL")
        for error in errors:
            print(f"- {error}")
        if not args.no_trace:
            emit_trace(args.trace_dir / "public_api_docs_gate.jsonl", "FAIL", "; ".join(errors))
        return 1
    print("public_api_docs: PASS")
    print("Header symbols: 68/68 documented and indexed")
    print("Eshkol exports: 59/59 documented and indexed")
    if not args.no_trace:
        emit_trace(args.trace_dir / "public_api_docs_gate.jsonl", "PASS",
                   "68 header symbols and 59 Eshkol exports are documented and indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
