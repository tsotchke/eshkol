#!/usr/bin/env python3
"""Verify that the JS WASM glue (web/eshkol-repl.js + site/static/eshkol-runtime.js)
provides every `env` import that the eshkol-compiled WASM modules ask for.

Compiles a representative set of Eshkol programs to WASM, parses each
`.wasm` import section, collects all `env.*` field names, then compares
against the keys defined in the `env: { … }` object literal of each JS
glue file. Fails if any WASM import is missing from the JS side.

This catches the failure mode where a new C runtime helper
(eshkol_intern_symbol_lookup, region_create, eshkol_make_parameter_ptr, …)
gets introduced on the host side and the WASM build silently picks it up
as an `env` import, but the JS glue is never updated — the website then
fails with `function import requires a callable` at instantiation.

Both glue files are mandatory and each is diffed against the FULL import set:
a stub added to only one of them still breaks the other, so a missing file is a
hard failure, never a skip.

Usage:
    scripts/check_wasm_imports.py [--build-dir build] [--server PATH]
    scripts/check_wasm_imports.py --selftest     # scanner regression suite only

Exit:
    0  every WASM env import has a matching JS env stub
    1  one or more imports missing
    2  no .wasm produced (toolchain / build issue), a required JS glue file
       missing, or the scanner's own self-test failed — neutral fail

Notes:
    - The WASM binary parser is intentionally minimal: it walks only the
      header + import section, so it works on any well-formed wasm32 file
      regardless of whether the rest of the module is reachable.
    - JS env extraction TOKENIZES the glue (comments, the three string
      flavours, `${…}` substitutions and regex literals) and then brace-matches
      over the token stream, so nested object literals contribute no keys and
      no comment or literal can desynchronise the scan.  selftest_extractor()
      pins that behaviour and runs before every verdict.
    - Smoke programs deliberately exercise different feature surfaces
      (symbols, arena, AD, tensors, parameters) so each unique import
      gets at least one chance to appear in the WASM output.
"""
from __future__ import annotations

import argparse
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_ROOT = REPO_ROOT / ".scratch"
JS_FILES = [
    REPO_ROOT / "web" / "eshkol-repl.js",
    REPO_ROOT / "site" / "static" / "eshkol-runtime.js",
]

# The C++ WASM entry check reports this ordered geometry tuple to both glue
# implementations. Values absent from an ABI use the unsigned 32-bit sentinel.
# Keep this independent of the generated WASM so the negative test can prove
# that a changed JS offset is rejected even when emscripten is unavailable.
WASM_ABI_GEOMETRY = {
    "abiVersion": 1,
    "pointerWidth": 4,
    "objectHeaderSize": 8,
    "objectHeaderAlign": 4,
    "objectPayloadAlign": 8,
    "objectSubtypeOffset": 0,
    "objectFlagsOffset": 1,
    "objectRefCountOffset": 2,
    "objectSizeOffset": 4,
    "objectLayoutIdOffset": 0xFFFFFFFF,
    "objectIdOffset": 0xFFFFFFFF,
    "objectHomeOffset": 0xFFFFFFFF,
    "objectAuxOffset": 0xFFFFFFFF,
    "taggedValueSize": 16,
    "taggedValueAlign": 8,
    "taggedValueTypeOffset": 0,
    "taggedValueFlagsOffset": 1,
    "taggedValueReservedOffset": 2,
    "taggedValueDataOffset": 8,
    "taggedValuePaddingOffset": 4,
}
_WASM_ABI_GEOMETRY_RE = re.compile(
    r"const\s+wasmAbiGeometry\s*=\s*Object\.freeze\(\s*\{(?P<body>.*?)\}\s*\)",
    re.DOTALL,
)
_WASM_ABI_GEOMETRY_FIELD_RE = re.compile(
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*:\s*(?P<value>0[xX][0-9A-Fa-f]+|[0-9]+)"
)


def extract_wasm_abi_geometry(js_text: str) -> dict[str, int] | None:
    """Extract the literal geometry object used by a JS glue file."""
    match = _WASM_ABI_GEOMETRY_RE.search(js_text)
    if not match:
        return None
    return {
        field.group("name"): int(field.group("value"), 0)
        for field in _WASM_ABI_GEOMETRY_FIELD_RE.finditer(match.group("body"))
    }


def validate_wasm_abi_geometry(js_text: str) -> list[str]:
    """Return contract violations for one JS glue file's ABI geometry."""
    actual = extract_wasm_abi_geometry(js_text)
    if actual is None:
        return ["missing const wasmAbiGeometry = Object.freeze({ ... })"]
    failures: list[str] = []
    for name, expected in WASM_ABI_GEOMETRY.items():
        if name not in actual:
            failures.append(f"{name}: missing (expected {expected})")
        elif actual[name] != expected:
            failures.append(f"{name}: expected {expected}, got {actual[name]}")
    for name in sorted(set(actual) - set(WASM_ABI_GEOMETRY)):
        failures.append(f"{name}: unexpected geometry field")
    return failures

# Smoke programs — each picks a different runtime surface. Add new programs
# here when a new runtime helper is introduced and you want CI to exercise it.
SMOKE_PROGRAMS = {
    "symbols": "(define x 'foo) (display (eq? x 'foo)) (newline)",
    "arena_lists": "(define xs (list 1 2 3)) (display (length xs)) (newline)",
    "math_tensor": "(define v (vector 1.0 2.0 3.0)) (display (vector-length v)) (newline)",
    "string_io": "(define p (open-output-string)) (display \"hi\" p) (display (get-output-string p)) (newline)",
    "parameters": "(define p (make-parameter 1)) (display (p)) (newline)",
    "regions": "(with-region (lambda () (define x (list 1 2 3)) (display (car x)) (newline)))",
    "bignum_rational": "(display (* 100000000000 100000000000)) (newline) (display (/ 1 3)) (newline)",
    "ad": "(display (derivative (lambda (x) (* x x)) 3.0)) (newline)",
    "control": "(call/cc (lambda (k) (display \"hello\") (newline) (k 0)))",
    "match_guard":
        "(guard (cond ((string? cond) (display cond) (newline)))"
        "  (raise \"caught\"))",
    # ESH-0011: the event-loop surface emits env imports of its own, and the
    # browser has no descriptors for it to watch. Compiling it here is what
    # proves the fail-closed JS stubs in both glue files are actually declared,
    # rather than discovering it when the lite lane breaks.
    "event_loop":
        "(define l (make-event-loop 4))"
        " (display (if l (event-loop-backend) \"no-event-loop\")) (newline)"
        " (if l (event-loop-close l) #f)",
    # The closure environment must carry every free variable through map;
    # this is the regression shape for the former 32-entry VM capture table.
    "large_closure_map":
        "(let ((c01 1) (c02 2) (c03 3) (c04 4) (c05 5) (c06 6)"
        " (c07 7) (c08 8) (c09 9) (c10 10) (c11 11) (c12 12)"
        " (c13 13) (c14 14) (c15 15) (c16 16) (c17 17) (c18 18)"
        " (c19 19) (c20 20) (c21 21) (c22 22) (c23 23) (c24 24)"
        " (c25 25) (c26 26) (c27 27) (c28 28) (c29 29) (c30 30)"
        " (c31 31) (c32 32) (c33 33))"
        " (display (map (lambda (x) (+ x c01 c02 c03 c04 c05 c06 c07"
        " c08 c09 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20 c21"
        " c22 c23 c24 c25 c26 c27 c28 c29 c30 c31 c32 c33)) '(1)))"
        " (newline))",
}


# --------------------------------------------------------------------------- #
#  WASM parser — header + import section only
# --------------------------------------------------------------------------- #


def _read_uleb128(data: bytes, off: int) -> tuple[int, int]:
    """Decode an unsigned LEB128 integer. Returns (value, new_off)."""
    result = 0
    shift = 0
    while True:
        byte = data[off]
        off += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, off
        shift += 7
        if shift > 63:
            raise ValueError("LEB128 too long")


def _read_string(data: bytes, off: int) -> tuple[str, int]:
    length, off = _read_uleb128(data, off)
    s = data[off : off + length].decode("utf-8")
    return s, off + length


def parse_env_imports(wasm_path: Path) -> set[str]:
    """Return the set of `env.<name>` import field names in this WASM module."""
    data = wasm_path.read_bytes()
    if data[:4] != b"\x00asm":
        raise ValueError(f"{wasm_path} does not have WASM magic")
    if struct.unpack_from("<I", data, 4)[0] != 1:
        raise ValueError(f"{wasm_path} unsupported WASM version")

    off = 8
    imports: set[str] = set()
    while off < len(data):
        section_id = data[off]
        off += 1
        section_size, off = _read_uleb128(data, off)
        section_end = off + section_size
        if section_id == 2:  # Import section
            count, off = _read_uleb128(data, off)
            for _ in range(count):
                module_name, off = _read_string(data, off)
                field_name, off = _read_string(data, off)
                kind = data[off]
                off += 1
                # Skip the type signature — we only care about names.
                if kind == 0:  # function: type index (uleb128)
                    _, off = _read_uleb128(data, off)
                elif kind == 1:  # table
                    off += 1  # elem type
                    flags, off = _read_uleb128(data, off)
                    _, off = _read_uleb128(data, off)  # min
                    if flags & 1:
                        _, off = _read_uleb128(data, off)  # max
                elif kind == 2:  # memory
                    flags, off = _read_uleb128(data, off)
                    _, off = _read_uleb128(data, off)  # min
                    if flags & 1:
                        _, off = _read_uleb128(data, off)  # max
                elif kind == 3:  # global
                    off += 1  # value type
                    off += 1  # mutable flag
                else:
                    raise ValueError(f"unknown import kind {kind} at offset {off-1}")
                if module_name == "env":
                    imports.add(field_name)
        off = section_end
    return imports


# --------------------------------------------------------------------------- #
#  JS env-key extractor — token-based scan around `env: {`
# --------------------------------------------------------------------------- #
#
# WHY A TOKENIZER AND NOT REGEXES.  The previous extractor matched braces and
# quote characters with a character scan that did not know where comments were.
# Two things in the real glue files broke it:
#
#   * an apostrophe inside a `//` comment ("WASM can't longjmp out of host
#     frames") opened a phantom string literal, after which the scan skipped
#     every `{`/`}` until the next apostrophe.  Skipping an unmatched brace
#     desynchronises the depth counter, so the `env: { … }` block ended in the
#     wrong place — the extractor then reported keys it had simply stopped
#     reading as MISSING.  A gate that names a symbol that is right there in
#     the file trains people to ignore it.
#   * a regex literal containing backticks or braces
#     (`html.replace(/`([^`]+)`/g, …)`) desynchronises it the same way.
#
# So the scan must know JS lexical structure: line comments, block comments,
# the three string flavours (including `${…}` substitutions, which nest), and
# regex literals.  That is what _tokenize does; extract_env_keys then walks a
# token stream where a brace is always a real brace.  selftest_extractor()
# below pins both failure directions with fixtures, and main() runs it before
# trusting any result, so this scanner can never again go quietly wrong.


_IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_NUM_RE = re.compile(r"[0-9][0-9A-Za-z_.]*")

# After these tokens a `/` starts a REGEX literal, not a division: a regex can
# only appear where an expression may start.  (`}` is treated as
# regex-permitting — a block close — because `}` / division is vanishingly rare
# and misreading a regex as division is the damaging direction.)
_REGEX_OK_KEYWORDS = frozenset({
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void",
    "do", "else", "case", "throw", "yield", "await",
})


class _Tok:
    """One JS token: kind in {ident, num, string, regex, punct}."""

    __slots__ = ("kind", "text", "pos")

    def __init__(self, kind: str, text: str, pos: int):
        self.kind = kind
        self.text = text
        self.pos = pos

    def is_punct(self, ch: str) -> bool:
        return self.kind == "punct" and self.text == ch

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.kind} {self.text!r} @{self.pos}>"


def _scan_string(js: str, i: int) -> int:
    """Index just past the string/template literal starting at `js[i]`.

    Handles backslash escapes and, for templates, `${…}` substitutions — which
    may themselves contain strings and further nested templates.
    """
    quote = js[i]
    n = len(js)
    i += 1
    while i < n:
        ch = js[i]
        if ch == "\\":
            i += 2
            continue
        if ch == quote:
            return i + 1
        if quote == "`" and ch == "$" and i + 1 < n and js[i + 1] == "{":
            # Substitution: balance braces, recursing through nested literals.
            i += 2
            depth = 1
            while i < n and depth:
                c = js[i]
                if c in "\"'`":
                    i = _scan_string(js, i)
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                i += 1
            continue
        i += 1
    return n  # unterminated — treat the remainder as literal


def _scan_regex(js: str, i: int) -> int:
    """Index just past the regex literal starting at `js[i]` (a `/`)."""
    n = len(js)
    i += 1
    in_class = False
    while i < n:
        ch = js[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "\n":
            return i  # unterminated regex; do not run past the line
        if in_class:
            if ch == "]":
                in_class = False
        elif ch == "[":
            in_class = True
        elif ch == "/":
            i += 1
            while i < n and js[i].isalpha():  # flags
                i += 1
            return i
        i += 1
    return n


def _tokenize(js: str) -> list[_Tok]:
    """Tokenize JS well enough to find object-literal keys.

    Comments are dropped.  Strings, templates and regex literals become single
    opaque tokens, so no character inside one can be mistaken for structure.
    """
    toks: list[_Tok] = []
    i = 0
    n = len(js)
    while i < n:
        ch = js[i]
        if ch in " \t\r\n\f\v":
            i += 1
            continue
        if ch == "/" and i + 1 < n and js[i + 1] == "/":
            while i < n and js[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and js[i + 1] == "*":
            close = js.find("*/", i + 2)
            i = n if close < 0 else close + 2
            continue
        if ch in "\"'`":
            end = _scan_string(js, i)
            toks.append(_Tok("string", js[i:end], i))
            i = end
            continue
        if ch == "/":
            prev = toks[-1] if toks else None
            regex_ok = (
                prev is None
                or (prev.kind == "ident" and prev.text in _REGEX_OK_KEYWORDS)
                or (prev.kind == "punct" and prev.text not in (")", "]"))
            )
            if regex_ok:
                end = _scan_regex(js, i)
                toks.append(_Tok("regex", js[i:end], i))
                i = end
                continue
            toks.append(_Tok("punct", ch, i))
            i += 1
            continue
        m = _IDENT_RE.match(js, i)
        if m:
            toks.append(_Tok("ident", m.group(0), i))
            i = m.end()
            continue
        m = _NUM_RE.match(js, i)
        if m:
            toks.append(_Tok("num", m.group(0), i))
            i = m.end()
            continue
        toks.append(_Tok("punct", ch, i))
        i += 1
    return toks


def _string_token_value(text: str) -> str | None:
    """The contents of a simple quoted key token, or None if it is not one."""
    if len(text) < 2 or text[0] not in "\"'" or text[-1] != text[0]:
        return None
    inner = text[1:-1]
    return inner if _IDENT_RE.fullmatch(inner) else None


def extract_env_keys(js_text: str) -> set[str]:
    """Find every `env: { … }` block in the JS text and return the set of keys
    defined directly inside it (depth-1 only, so nested object literals don't
    contribute).

    A key is an identifier or quoted name that (a) sits at brace depth 1 of the
    block, (b) is preceded by the block's `{` or by a `,`, and (c) is followed
    by `:`.  Requiring the `{`/`,` anchor is what keeps a ternary's `… ? a : b`
    inside a property VALUE from being harvested as a property NAME.
    """
    toks = _tokenize(js_text)
    keys: set[str] = set()

    for idx in range(len(toks) - 2):
        if not (toks[idx].kind == "ident" and toks[idx].text == "env"):
            continue
        if not (toks[idx + 1].is_punct(":") and toks[idx + 2].is_punct("{")):
            continue
        depth = 1
        j = idx + 3
        anchored = True  # right after `{`, a key may start here
        while j < len(toks) and depth > 0:
            tok = toks[j]
            if tok.is_punct("{"):
                depth += 1
            elif tok.is_punct("}"):
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1:
                if tok.is_punct(","):
                    anchored = True
                    j += 1
                    continue
                if anchored and j + 1 < len(toks) and toks[j + 1].is_punct(":"):
                    name = (
                        tok.text
                        if tok.kind == "ident"
                        else _string_token_value(tok.text)
                        if tok.kind == "string"
                        else None
                    )
                    if name:
                        keys.add(name)
                anchored = False
            j += 1
    return keys


# --------------------------------------------------------------------------- #
#  Self-test — the gate on the gate
# --------------------------------------------------------------------------- #


# Every fixture below is a (description, js, expected-keys) triple.  They pin
# the constructs that have desynchronised character-scanning extractors in this
# file's history, plus the plain cases, so a future "simplification" that
# reintroduces the bug fails here instead of in a CI run six weeks later.
_SELFTEST_FIXTURES: list[tuple[str, str, set[str]]] = [
    (
        "plain keys, one per line",
        "const x = { env: { alpha: () => 0,\n  beta: () => 1,\n } };",
        {"alpha", "beta"},
    ),
    (
        "several keys on one line",
        "const x = { env: { sin: Math.sin, cos: Math.cos } };",
        {"sin", "cos"},
    ),
    (
        "apostrophe inside a // comment must not open a string",
        "const x = { env: {\n"
        "  before: () => 0,\n"
        "  // Continuations — WASM can't longjmp out of host frames\n"
        "  after: () => 0,\n"
        "} };",
        {"before", "after"},
    ),
    (
        # THE REPORTED DEFECT, minimised. The apostrophe in "can't" opened a
        # phantom string; the next real quote closed it, so the `}` inside
        # `'}'` was read as code, closed the env block early, and every key
        # after this line was reported MISSING though present.
        "apostrophe in a // comment followed by a brace-bearing string",
        "const x = { env: {\n"
        "  before: () => 0,\n"
        "  // Continuations — WASM can't longjmp out of host frames\n"
        "  fmt: () => '}',\n"
        "  after: () => 0,\n"
        "} };",
        {"before", "fmt", "after"},
    ),
    (
        "unbalanced brace inside a // comment must not shift depth",
        "const x = { env: {\n"
        "  before: () => 0,\n"
        "  // the closing } of this comment is not code\n"
        "  after: () => 0,\n"
        "} };",
        {"before", "after"},
    ),
    (
        "regex literal containing backticks and braces",
        "const x = { env: {\n"
        "  before: (s) => s.replace(/`([^`]+)`/g, '<code>'),\n"
        "  braces: (s) => s.replace(/[{}]/g, ''),\n"
        "  after: () => 0,\n"
        "} };",
        {"before", "braces", "after"},
    ),
    (
        "template literal with a ${…} substitution",
        "const x = { env: {\n"
        "  before: (n) => { throw new Error(`bad ${n} value}`); },\n"
        "  after: () => 0,\n"
        "} };",
        {"before", "after"},
    ),
    (
        "nested object literal contributes no keys",
        "const x = { env: {\n"
        "  outer: new WebAssembly.Global({ value: 'i32', mutable: false }, 0),\n"
        "  after: () => 0,\n"
        "} };",
        {"outer", "after"},
    ),
    (
        "ternary in a value is not a key",
        "const x = { env: { pick: (a) => a ? left : right, after: () => 0 } };",
        {"pick", "after"},
    ),
    (
        "block comment mentioning env: { is not a second block",
        "/* env: { ghost: 0 } */\nconst x = { env: { real: () => 0 } };",
        {"real"},
    ),
    (
        "string containing a // sequence is not a comment",
        "const x = { env: {\n"
        "  url: () => 'http://example.invalid/x',\n"
        "  after: () => 0,\n"
        "} };",
        {"url", "after"},
    ),
    (
        "quoted key",
        "const x = { env: { 'quoted': () => 0, plain: () => 1 } };",
        {"quoted", "plain"},
    ),
    (
        "division is not a regex",
        "const x = { env: { half: (n) => n / 2, after: () => 0 } };",
        {"half", "after"},
    ),
]


def selftest_extractor() -> list[str]:
    """Run the extractor fixtures.  Returns a list of failure descriptions."""
    failures: list[str] = []
    for name, js, expected in _SELFTEST_FIXTURES:
        got = extract_env_keys(js)
        if got != expected:
            failures.append(
                f"{name}: expected {sorted(expected)}, got {sorted(got)}"
            )

    # Direction 2 of the gate proof, at unit scale: a key that is absent must
    # be reported absent.  (The whole-file version is in the CI step; this is
    # the assertion that the comparison itself can still fail.)
    present = extract_env_keys("const x = { env: { kept: () => 0 } };")
    if "removed" in present:
        failures.append("absent key reported present")
    if "kept" not in present:
        failures.append("present key reported absent")
    return failures


def selftest_geometry() -> list[str]:
    """Prove both the source contract and its deliberate mismatch direction."""
    fields = "\n".join(f"            {name}: {value},"
                         for name, value in WASM_ABI_GEOMETRY.items())
    valid = f"const wasmAbiGeometry = Object.freeze({{\n{fields}\n        }});"
    failures = validate_wasm_abi_geometry(valid)
    if failures:
        return [f"valid geometry rejected: {failure}" for failure in failures]

    wrong = valid.replace("objectHeaderSize: 8", "objectHeaderSize: 9", 1)
    if not validate_wasm_abi_geometry(wrong):
        failures.append("deliberately wrong objectHeaderSize was accepted")

    for js in JS_FILES:
        if not js.exists():
            failures.append(f"required JS glue file not found: {js}")
            continue
        for failure in validate_wasm_abi_geometry(js.read_text()):
            failures.append(f"{js.relative_to(REPO_ROOT)}: {failure}")
    return failures


# --------------------------------------------------------------------------- #
#  Build / compile helpers
# --------------------------------------------------------------------------- #


def find_server_binary(build_dir: Path, override: Path | None) -> Path | None:
    if override:
        return override if override.exists() else None
    candidate = build_dir / "eshkol-server"
    if candidate.exists():
        return candidate
    return None


def compile_smoke_wasm(server_bin: Path, source: str) -> bytes | None:
    """Drive eshkol-server's compile-to-WASM pipeline directly via a tiny
    HTTP POST.  Falls back to None if the server isn't usable."""
    # We don't want to spin up the actual HTTP listener; instead we use the
    # compile-to-wasm CLI shim if present.  As a fallback, write the source
    # to a tempfile and use eshkol-run --emit-wasm if that flag exists.
    # For now, just return None and fall through to the CLI path.
    return None


def compile_smoke_wasm_via_cli(
    eshkol_run: Path, source: str, timeout: int
) -> bytes | None:
    """Use `eshkol-run --wasm <src> -o <out.wasm>` to produce the .wasm.
    Returns the WASM bytes, or None if the toolchain refuses."""
    with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as tmp:
        src_path = Path(tmp) / "smoke.esk"
        out_path = Path(tmp) / "smoke.wasm"
        src_path.write_text(source)
        try:
            res = subprocess.run(
                [str(eshkol_run), "--wasm", str(src_path), "-o", str(out_path)],
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"warning: wasm smoke compile timed out after {timeout}s", file=sys.stderr)
            return None
        except FileNotFoundError:
            print(f"warning: {eshkol_run} not found", file=sys.stderr)
            return None
        if res.returncode != 0 or not out_path.exists():
            stderr = res.stderr.decode("utf-8", errors="replace").strip()
            if stderr:
                print(stderr[-4000:], file=sys.stderr)
            print(
                f"warning: wasm smoke compile failed with exit code {res.returncode}",
                file=sys.stderr,
            )
            return None
        return out_path.read_bytes()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    ap.add_argument("--server", type=Path, default=None,
                    help="Path to a pre-built eshkol-server (overrides --build-dir).")
    ap.add_argument("--prebuilt-wasm-dir", type=Path, default=None,
                    help="Directory of pre-compiled .wasm files (one per smoke surface). "
                         "If passed, skip compilation and just check imports against JS.")
    ap.add_argument("--compile-timeout", type=int, default=180,
                    help="Per-surface timeout, in seconds, for eshkol-run --wasm "
                         "smoke compiles (default: 180).")
    ap.add_argument("--strict", action="store_true",
                    help="Also fail if the JS provides keys that no WASM import requested "
                         "(catches stale stubs).")
    ap.add_argument("--selftest", action="store_true",
                    help="Only run the env-key extractor self-test and exit "
                         "(no build, no WASM compile).")
    args = ap.parse_args()

    # The extractor's own regression suite runs FIRST, unconditionally. If the
    # scanner is broken, every verdict it produces afterwards is worthless —
    # either a phantom red naming a symbol that is present, or a silent green.
    # Failing here is a hard tooling failure (exit 2), not a content failure.
    selftest_failures = selftest_extractor() + selftest_geometry()
    if selftest_failures:
        print("error: WASM import/geometry self-test FAILED — refusing to report "
              "a verdict from a broken scanner:", file=sys.stderr)
        for f in selftest_failures:
            print(f"  {f}", file=sys.stderr)
        return 2
    if args.selftest:
        print(f"OK — WASM import/geometry self-test passed "
              f"({len(_SELFTEST_FIXTURES)} extractor fixtures; deliberate "
              "geometry mismatch rejected).")
        return 0

    eshkol_run = args.build_dir / "eshkol-run"

    # Collect imports from every smoke WASM.
    all_imports: set[str] = set()
    surfaces_built: list[str] = []
    surfaces_failed: list[str] = []

    if args.prebuilt_wasm_dir:
        for wasm_file in sorted(args.prebuilt_wasm_dir.glob("*.wasm")):
            try:
                imps = parse_env_imports(wasm_file)
                all_imports |= imps
                surfaces_built.append(wasm_file.stem)
            except Exception as e:
                print(f"warning: {wasm_file}: {e}", file=sys.stderr)
                surfaces_failed.append(wasm_file.stem)
    else:
        if not eshkol_run.exists():
            print(
                f"error: {eshkol_run} not found.  Build first or pass "
                "--prebuilt-wasm-dir.",
                file=sys.stderr,
            )
            return 2

        SCRATCH_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH_ROOT) as tmp:
            tmpdir = Path(tmp)
            for surface, src in SMOKE_PROGRAMS.items():
                print(f"compiling WASM smoke surface: {surface}", file=sys.stderr)
                wasm = compile_smoke_wasm_via_cli(
                    eshkol_run, src, args.compile_timeout
                )
                if wasm is None:
                    surfaces_failed.append(surface)
                    continue
                wasm_path = tmpdir / f"{surface}.wasm"
                wasm_path.write_bytes(wasm)
                try:
                    imps = parse_env_imports(wasm_path)
                    all_imports |= imps
                    surfaces_built.append(surface)
                except Exception as e:
                    print(f"warning: {surface}: parse failed: {e}", file=sys.stderr)
                    surfaces_failed.append(surface)

    if not surfaces_built:
        print(
            "error: no smoke surfaces compiled to WASM.  Either eshkol-run "
            "doesn't support --emit-wasm, or every program in SMOKE_PROGRAMS "
            "failed.  Pass --prebuilt-wasm-dir to bypass compilation.",
            file=sys.stderr,
        )
        return 2

    # Collect JS env keys from every glue file.
    #
    # BOTH files are mandatory, and each is checked against the FULL import set
    # independently: the lite lane instantiates the site runtime and the REPL
    # from separate glue, so a stub added to only one of them still breaks the
    # other with `function import requires a callable`. A silently-skipped file
    # would turn that into a green run, so a missing file is a hard failure
    # rather than a warning.
    js_provided: dict[Path, set[str]] = {}
    missing_files = [js for js in JS_FILES if not js.exists()]
    if missing_files:
        for js in missing_files:
            print(f"error: required JS glue file not found: {js}", file=sys.stderr)
        print("error: every file in JS_FILES must be present — both the site "
              "runtime and the REPL glue provide the `env` imports and both are "
              "instantiated by the lite lane.", file=sys.stderr)
        return 2
    for js in JS_FILES:
        js_provided[js] = extract_env_keys(js.read_text())

    if not js_provided:
        print("error: no JS glue files configured", file=sys.stderr)
        return 2

    # Cross-check.
    failed = False
    for js in JS_FILES:
        geometry_failures = validate_wasm_abi_geometry(js.read_text())
        if geometry_failures:
            failed = True
            rel = js.relative_to(REPO_ROOT)
            print(f"\n=== ABI GEOMETRY MISMATCH in {rel} ===")
            for failure in geometry_failures:
                print(f"  {failure}")
    for js, keys in js_provided.items():
        missing = sorted(all_imports - keys)
        extra = sorted(keys - all_imports) if args.strict else []
        rel = js.relative_to(REPO_ROOT)
        if missing:
            failed = True
            print(f"\n=== MISSING in {rel} ({len(missing)}) ===")
            for m in missing:
                print(f"  {m}")
        if extra:
            print(f"\n=== UNUSED in {rel} ({len(extra)}, --strict mode) ===")
            for e in extra:
                print(f"  {e}")

    print(
        f"\nsurfaces compiled: {len(surfaces_built)} "
        f"({', '.join(surfaces_built)})"
    )
    if surfaces_failed:
        print(
            f"surfaces FAILED:    {len(surfaces_failed)} "
            f"({', '.join(surfaces_failed)})"
        )
    print(f"unique env imports: {len(all_imports)}")
    print(f"JS keys per file:   {{{', '.join(f'{p.name}={len(k)}' for p,k in js_provided.items())}}}")

    if failed:
        print("\nFIX: add stubs for the MISSING imports above to the JS env "
              "object literal(s).  Pattern: `<name>: () => 0` for opaque "
              "returns, or implement properly when the WASM-side semantics "
              "depend on the helper.")
        return 1

    if surfaces_failed:
        print(
            "\nerror: one or more WASM smoke surfaces failed to compile; "
            "the import coverage would be incomplete.",
            file=sys.stderr,
        )
        return 2

    print("OK — all WASM env imports are provided by the JS glue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
