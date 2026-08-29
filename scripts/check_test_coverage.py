#!/usr/bin/env python3
"""Check that TEST_COVERAGE.md describes the current suite inventory."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from generate_test_coverage import ROOT, inventory

DOC = ROOT / "docs" / "TEST_COVERAGE.md"
ROW_RE = re.compile(
    r'^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|'
)


def check(doc: Path = DOC, runner: Path | None = None) -> list[str]:
    expected = inventory(runner or ROOT / "scripts" / "run_all_tests.sh")
    text = doc.read_text(encoding="utf-8")
    errors: list[str] = []
    title_count = re.search(r'All (\d+) suites are orchestrated', text)
    if not title_count:
        errors.append("documentation has no generated suite-count statement")
    elif int(title_count.group(1)) != expected["suite_count"]:
        errors.append("documentation suite count does not match TEST_SCRIPTS")

    rows = []
    for line in text.splitlines():
        match = ROW_RE.match(line)
        if match:
            rows.append((int(match.group(1)), match.group(3), match.group(4).strip()))
    actual = [script for _, script, _ in rows]
    expected_scripts = [row["script"] for row in expected["suites"]]
    if set(actual) != set(expected_scripts) or len(actual) != len(expected_scripts):
        errors.append("documentation suite rows do not match TEST_SCRIPTS")
    if [number for number, _, _ in sorted(rows)] != list(range(1, len(rows) + 1)):
        errors.append("documentation suite row numbers are not consecutive")
    for number, script, count in rows:
        if count != "--":
            errors.append("row %d (%s) carries an ungenerated per-suite count %r"
                          % (number, script, count))
    if "generate_test_coverage.py" not in text:
        errors.append("documentation does not identify generate_test_coverage.py")
    return errors


def self_test() -> bool:
    (ROOT / ".scratch").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test-coverage-check-", dir=ROOT / ".scratch") as temp:
        root = Path(temp)
        runner = root / "run_all_tests.sh"
        runner.write_text('TEST_SCRIPTS=(\n  "run_a.sh"\n  "run_b.sh"\n)\n', encoding="utf-8")
        good = root / "TEST_COVERAGE.md"
        good.write_text(
            "All 2 suites are orchestrated by run_all_tests.sh.\n"
            "Use generate_test_coverage.py.\n"
            "| 1 | A | `run_a.sh` | -- | x |\n"
            "| 2 | B | `run_b.sh` | -- | x |\n",
            encoding="utf-8",
        )
        bad = root / "bad.md"
        bad.write_text(good.read_text(encoding="utf-8").replace("| -- |", "| 7 |", 1), encoding="utf-8")
        return not check(good, runner) and bool(check(bad, runner))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        ok = self_test()
        print("check_test_coverage.py self-test: %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    errors = check(args.doc, args.runner)
    if errors:
        print("test_coverage_inventory: FAIL")
        for error in errors:
            print("  - %s" % error)
        return 1
    data = inventory(args.runner or ROOT / "scripts" / "run_all_tests.sh")
    print("test_coverage_inventory: PASS (%d suites; per-suite counts are generated-only)"
          % data["suite_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
