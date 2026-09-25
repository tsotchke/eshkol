#!/usr/bin/env python3
"""Focused regression test for generated flat-AD import glue."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    generator = ROOT / "scripts" / "generate_wasm_import_glue.py"
    node_test = Path(__file__).with_suffix(".js")
    for command in (
        [sys.executable, str(generator), "--selftest"],
        ["node", str(node_test)],
    ):
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
