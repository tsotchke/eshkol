#!/usr/bin/env python3
"""Require MessagePack wire golden tests and strict UTF-8 refusal tests.

The route runner is shared with the Protocol Buffers suite and lives in
scripts/run_wire_format_gate.py; this entry point keeps the MessagePack
command line used by CTest and scripts/run_v14_connection_gate.sh.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_wire_format_gate import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main(["--suite", "msgpack"] + sys.argv[1:]))
