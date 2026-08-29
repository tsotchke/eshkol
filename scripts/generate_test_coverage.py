#!/usr/bin/env python3
"""Generate the source-of-truth test-suite inventory from run_all_tests.sh."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_all_tests.sh"
SCRIPT_RE = re.compile(r'^\s+"([^"]+\.sh)"\s*$')


def suite_scripts(runner: Path = RUNNER) -> list[str]:
    text = runner.read_text(encoding="utf-8")
    match = re.search(r"TEST_SCRIPTS=\(\n(.*?)\n\)", text, re.DOTALL)
    if not match:
        raise ValueError("TEST_SCRIPTS array not found in %s" % runner)
    scripts = [m.group(1) for line in match.group(1).splitlines()
               if (m := SCRIPT_RE.match(line))]
    if not scripts:
        raise ValueError("TEST_SCRIPTS array is empty in %s" % runner)
    if len(set(scripts)) != len(scripts):
        raise ValueError("TEST_SCRIPTS contains duplicate entries")
    return scripts


def inventory(runner: Path = RUNNER) -> dict:
    scripts = suite_scripts(runner)
    return {
        "schema": "eshkol.test_suite_inventory.v1",
        "source": str(runner.relative_to(ROOT)),
        "suite_count": len(scripts),
        "suites": [{"number": i, "script": script, "test_count": None}
                   for i, script in enumerate(scripts, 1)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=RUNNER)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = inventory(args.runner)
    rendered = json.dumps(data, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
