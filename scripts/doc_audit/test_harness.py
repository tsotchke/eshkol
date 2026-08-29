#!/usr/bin/env python3
"""Regression tests for the documentation example parser and runners."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SCRATCH = REPO_ROOT / ".scratch"
sys.path.insert(0, str(HERE))

import check_output_blocks  # noqa: E402
import extract_examples  # noqa: E402


def run() -> bool:
    SCRATCH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="doc-audit-test-", dir=SCRATCH) as temp:
        root = Path(temp)
        (root / "README.md").write_text(
            "~~~scheme\n"
            "(display 1) ; => one\n"
            "(display 2) ; returns: two\n"
            "(display 3) ; prints three\n"
            "~~~\n"
            "\n"
            "~~~text\n"
            "1\n"
            "2\n"
            "3\n"
            "~~~\n",
            encoding="utf-8",
        )
        original_scope = extract_examples.SCOPE
        extract_examples.SCOPE = ["README.md"]
        try:
            lines = (root / "README.md").read_text(encoding="utf-8").splitlines()
            opening = extract_examples.FENCE_RE.match(lines[0])
            if opening is None:
                return False
            close = extract_examples.closing_fence(lines, 0, opening)
            expectation_lines = lines[1:close]
            expectations = [
                extract_examples.EXPECT_RE.search(line).group(1).strip()
                for line in expectation_lines
                if extract_examples.EXPECT_RE.search(line)
            ]
            pairs = check_output_blocks.collect(str(root))
        finally:
            extract_examples.SCOPE = original_scope

    return (
        close == 4
        and expectations == ["one", "two", "three"]
        and len(pairs) == 1
        and pairs[0]["expected"] == "1\n2\n3"
    )


def main() -> int:
    ok = run()
    print("doc_audit_harness: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
