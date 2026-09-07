#!/usr/bin/env python3
"""Build-free gate: the committed VM prelude cache names every VM builtin.

`lib/backend/vm_prelude_cache.h` is a checked-in, pre-compiled bytecode image
of the bytecode VM's Scheme prelude. `emit_builtin_preamble()` in
lib/backend/eshkol_vm.c turns every entry of the `BUILTINS[]` dispatch table
into one prelude local (a first-class closure wrapping that native), so any
addition, removal or rename in `BUILTINS[]` invalidates the cache — and
nothing before this gate ever re-checked that invalidation happened. SW-49:
the committed cache drifted 28 names behind `BUILTINS[]` (missing
`string-length`, `string-ref`, `integer?` and the whole c[ad]{3,4}r family)
with every native lane green, because the cache's only consumer is the
Emscripten-built WASM REPL (vm_wasm_repl.c, the one site that defines
ESHKOL_VM_NO_DISASM as a macro) — no ctest target and no native binary loads
it.

This gate closes that blind spot WITHOUT compiling anything: it extracts the
`{"name", id, arity}` entries from the `BUILTINS[]` initializer in
eshkol_vm.c and the `"name"` entries from the `prelude_local_names[]` array
in vm_prelude_cache.h by text, and asserts the first set is a subset of the
second. It cannot detect an ordering or bytecode-content bug (that needs the
real generator — see scripts/regenerate_vm_prelude_cache.sh and the
`vm_prelude_cache_is_current` ctest it wires up when the VM unity build is
available) but it CAN detect exactly the SW-49 shape — a builtin the cache
has simply never heard of — build-free, on every PR, including docs-only
lanes that never configure CMake.

Usage
    scripts/check_vm_prelude_cache_builtins.py
    scripts/check_vm_prelude_cache_builtins.py --format json
    scripts/check_vm_prelude_cache_builtins.py --selftest

Exit:
    0  every BUILTINS[] name appears in the committed prelude_local_names[]
    1  one or more BUILTINS[] names are missing from the committed cache
    2  a source file is missing or neither array could be parsed — neutral
       fail (this gate refuses to pass on "I could not check")

"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VM_SOURCE = REPO_ROOT / "lib" / "backend" / "eshkol_vm.c"
CACHE_HEADER = REPO_ROOT / "lib" / "backend" / "vm_prelude_cache.h"

_STRING_RE = r'"(?:[^"\\]|\\.)*"'


def _strip_c_comments(text: str) -> str:
    """Remove /* */ and // comments without disturbing string literals.

    A hand-rolled scanner rather than a single regex: BUILTINS[] carries
    long block comments containing quote characters and the word `//` in
    prose (e.g. arity notes), so a comment-vs-string regex race would be
    fragile. This walks the text once, character by character, tracking
    whether we are inside a string, a block comment or a line comment.
    """
    out = []
    i = 0
    n = len(text)
    in_string = False
    in_block_comment = False
    in_line_comment = False
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
                out.append(c)
            i += 1
            continue
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(nxt)
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        # Not in string/comment
        if c == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if c == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _extract_braced_block(text: str, anchor: str) -> str:
    """Return the `{ ... }` body that follows `anchor` in `text`.

    Brace-balances from the first `{` after the anchor to its match, so
    nested struct-literal braces (`{"sin", 20, 1}` inside the array) do not
    end the scan early.
    """
    idx = text.find(anchor)
    if idx < 0:
        raise ValueError(f"anchor not found: {anchor!r}")
    start = text.find("{", idx)
    if start < 0:
        raise ValueError(f"no '{{' after anchor: {anchor!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    raise ValueError(f"unbalanced braces after anchor: {anchor!r}")


def extract_builtin_names(source_text: str) -> list[str]:
    """Every `name` in eshkol_vm.c's `static const BuiltinDef BUILTINS[] = {...}`.

    Includes internal (`_`-prefixed) entries: emit_builtin_preamble() gives
    ALL of them a prelude local regardless of whether the prelude's own
    Scheme-level bindings re-expose them under a public name, so all of them
    must appear in prelude_local_names[] too.
    """
    clean = _strip_c_comments(source_text)
    body = _extract_braced_block(clean, "static const BuiltinDef BUILTINS[] = ")
    # Each live entry looks like {"name", <int>, <int>}; the sentinel is
    # {NULL, 0, 0} and carries no string to match.
    names = re.findall(r'\{\s*(' + _STRING_RE + r')\s*,', body)
    return [json.loads(n) for n in names]


def extract_cache_local_names(header_text: str) -> list[str]:
    """Every entry in vm_prelude_cache.h's `prelude_local_names[]` array."""
    clean = _strip_c_comments(header_text)
    body = _extract_braced_block(clean, "static const char* prelude_local_names[] = ")
    names = re.findall(_STRING_RE, body)
    return [json.loads(n) for n in names]


def check(vm_source_path: Path, cache_header_path: Path) -> tuple[bool, dict]:
    if not vm_source_path.is_file():
        return False, {"error": f"missing VM source: {vm_source_path}"}
    if not cache_header_path.is_file():
        return False, {"error": f"missing prelude cache: {cache_header_path}"}

    source_text = vm_source_path.read_text(encoding="utf-8", errors="replace")
    header_text = cache_header_path.read_text(encoding="utf-8", errors="replace")

    try:
        builtin_names = extract_builtin_names(source_text)
    except ValueError as exc:
        return False, {"error": f"could not parse BUILTINS[] in {vm_source_path}: {exc}"}
    try:
        cache_names = extract_cache_local_names(header_text)
    except ValueError as exc:
        return False, {"error": f"could not parse prelude_local_names[] in {cache_header_path}: {exc}"}

    if not builtin_names:
        return False, {"error": f"BUILTINS[] parsed with zero entries in {vm_source_path}"}
    if not cache_names:
        return False, {"error": f"prelude_local_names[] parsed with zero entries in {cache_header_path}"}

    cache_set = set(cache_names)
    missing = sorted(name for name in builtin_names if name not in cache_set)

    result = {
        "builtins_total": len(builtin_names),
        "cache_locals_total": len(cache_names),
        "missing_from_cache": missing,
    }
    return (len(missing) == 0), result


def selftest() -> bool:
    """Prove the two extractors are correct and prove the gate can go red."""
    ok = True

    src = (
        'typedef struct { const char* name; int native_id; int arity; } BuiltinDef;\n'
        'static const BuiltinDef BUILTINS[] = {\n'
        '    /* a comment with a "quoted string" and // a slash-slash */\n'
        '    {"sin", 20, 1}, {"cos", 21, 1},\n'
        '    // {"commented-out", 99, 1},\n'
        '    {"_internal-helper", 5, 2},\n'
        '    {"weird\\"name", 6, 1},\n'
        '    {NULL, 0, 0}\n'
        '};\n'
    )
    names = extract_builtin_names(src)
    expected = ["sin", "cos", "_internal-helper", 'weird"name']
    if names != expected:
        print(f"SELFTEST FAIL: extract_builtin_names got {names!r}, want {expected!r}")
        ok = False

    hdr = (
        'static const char* prelude_local_names[] = {\n'
        '    "sin",\n'
        '    "cos",\n'
        '    "_internal-helper",\n'
        '    "weird\\"name"\n'
        '};\n'
    )
    cache_names = extract_cache_local_names(hdr)
    if cache_names != expected:
        print(f"SELFTEST FAIL: extract_cache_local_names got {cache_names!r}, want {expected!r}")
        ok = False

    # The gate must actually go red on a real drift, not just report a
    # vacuous "0 missing" no matter what it is handed (VACUOUS-ASSURANCE
    # shape from the ledger's own bucket vocabulary).
    stale_hdr = (
        'static const char* prelude_local_names[] = {\n'
        '    "sin"\n'
        '};\n'
    )
    with_stale = extract_cache_local_names(stale_hdr)
    missing = sorted(n for n in names if n not in set(with_stale))
    if missing != sorted(["cos", "_internal-helper", 'weird"name']):
        print(f"SELFTEST FAIL: drift detection got {missing!r}")
        ok = False

    if ok:
        print("SELFTEST PASS: check_vm_prelude_cache_builtins extractors are correct "
              "and the gate can fail")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vm-source", type=Path, default=VM_SOURCE)
    parser.add_argument("--cache-header", type=Path, default=CACHE_HEADER)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return 0 if selftest() else 1

    passed, result = check(args.vm_source, args.cache_header)

    if args.format == "json":
        print(json.dumps({"pass": passed, **result}, indent=2, sort_keys=True))
    else:
        if "error" in result:
            print(f"FAIL: {result['error']}")
        elif passed:
            print(
                f"PASS vm_prelude_cache_builtins: all {result['builtins_total']} "
                f"BUILTINS[] entries are present in the committed "
                f"prelude_local_names[] ({result['cache_locals_total']} entries)"
            )
        else:
            print(
                f"FAIL vm_prelude_cache_builtins: {len(result['missing_from_cache'])} "
                f"of {result['builtins_total']} BUILTINS[] entries are missing from "
                "the committed lib/backend/vm_prelude_cache.h — regenerate it with "
                "scripts/regenerate_vm_prelude_cache.sh and commit the result."
            )
            for name in result["missing_from_cache"]:
                print(f"  missing: {name}")

    if "error" in result:
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
