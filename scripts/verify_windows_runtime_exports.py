#!/usr/bin/env python3
"""Verify that a Windows Eshkol host publishes its bounded JIT runtime ABI."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


RUNTIME_PREFIXES = (
    "eshkol_",
    "arena_",
    "__eshkol",
    "__repl_shared_arena",
    "g_eshkol",
    "g_current_exception",
    "g_exception_handler_stack",
    "g_lambda_registry",
)

REQUIRED_EXPORTS = {
    "arena_allocate",
    "eshkol_ad_node_custom_backward",
    "eshkol_closure_get_arity",
    "eshkol_make_parameter_ptr",
    "eshkol_raise",
    "eshkol_repl_capture_last_value",
}


def parse_exports(output: str) -> set[str]:
    return set(re.findall(r"^\s*Name:\s+(\S+)\s*$", output, re.MULTILINE))


def verify_exports(exports: set[str], minimum_runtime_exports: int) -> list[str]:
    runtime_exports = {
        name for name in exports if name.startswith(RUNTIME_PREFIXES)
    }
    errors: list[str] = []
    missing = sorted(REQUIRED_EXPORTS - exports)
    if missing:
        errors.append("missing required runtime exports: " + ", ".join(missing))
    if len(runtime_exports) < minimum_runtime_exports:
        errors.append(
            "bounded runtime export table is incomplete: "
            f"found {len(runtime_exports)}, expected at least {minimum_runtime_exports}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--llvm-readobj", required=True, type=pathlib.Path)
    parser.add_argument("--minimum-runtime-exports", type=int, default=800)
    args = parser.parse_args()

    for label, path in (("binary", args.binary), ("llvm-readobj", args.llvm_readobj)):
        if not path.is_file():
            parser.error(f"{label} is not a regular file: {path}")

    completed = subprocess.run(
        [str(args.llvm_readobj), "--coff-exports", str(args.binary)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        print(completed.stdout, end="", file=sys.stderr)
        print(completed.stderr, end="", file=sys.stderr)
        print(
            f"FAIL: llvm-readobj exited with {completed.returncode}",
            file=sys.stderr,
        )
        return 1

    exports = parse_exports(completed.stdout)
    errors = verify_exports(exports, args.minimum_runtime_exports)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1

    runtime_count = sum(name.startswith(RUNTIME_PREFIXES) for name in exports)
    print(
        "PASS: Windows host publishes "
        f"{runtime_count} bounded Eshkol runtime exports for cache-disabled JIT"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
