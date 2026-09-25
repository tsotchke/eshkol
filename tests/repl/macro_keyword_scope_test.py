#!/usr/bin/env python3
"""A macro keyword persists between inputs but never leaks between sessions."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "tools"))
from erepl_client import EReplClient

binary = sys.argv[1]
def execute(client, code, expected):
    result = client.execute(code, timeout=60)
    if result["error"] is not None or result["stdout"] != expected:
        raise SystemExit(f"FAIL: {code}: {result}")

with EReplClient(binary) as client:
    for value in (777, 888):
        execute(client, f"(define-syntax when (syntax-rules () ((_ test body) {value})))", "")
        execute(client, "(display (when #f 42))", str(value))
with EReplClient(binary) as client:
    execute(client, "(display (when #f 42))", "")
print("PASS: REPL keyword macro persistence, replacement, and isolation")
