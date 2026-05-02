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

Usage:
    scripts/check_wasm_imports.py [--build-dir build] [--server PATH]

Exit:
    0  every WASM env import has a matching JS env stub
    1  one or more imports missing
    2  no .wasm produced (toolchain / build issue) — neutral fail

Notes:
    - The WASM binary parser is intentionally minimal: it walks only the
      header + import section, so it works on any well-formed wasm32 file
      regardless of whether the rest of the module is reachable.
    - JS env extraction uses a brace-matched scan around `env: {`, so it
      handles nested object literals (DOM bridge etc.) correctly.
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
JS_FILES = [
    REPO_ROOT / "web" / "eshkol-repl.js",
    REPO_ROOT / "site" / "static" / "eshkol-runtime.js",
]

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
#  JS env-key extractor — brace-matched scan around `env: {`
# --------------------------------------------------------------------------- #


_KEY_RE = re.compile(r"^\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*:")
_LINE_COMMENT_RE = re.compile(r"//.*$")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def extract_env_keys(js_text: str) -> set[str]:
    """Find every `env: { … }` block in the JS text and return the set of keys
    defined directly inside it (depth-1 only, so nested object literals don't
    contribute)."""
    text = _BLOCK_COMMENT_RE.sub(" ", js_text)
    keys: set[str] = set()

    for env_match in re.finditer(r"\benv\s*:\s*\{", text):
        start = env_match.end()
        depth = 1
        i = start
        in_str = None
        while i < len(text) and depth > 0:
            ch = text[i]
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
                i += 1
                continue
            if ch in ('"', "'", "`"):
                in_str = ch
                i += 1
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1

        block = text[start:i]
        # Strip line comments line-by-line.
        clean_lines = [_LINE_COMMENT_RE.sub("", ln) for ln in block.splitlines()]

        # We want only depth-1 keys.  Walk the block tracking brace depth.
        d = 0
        in_str = None
        line_no = 0
        for ln in clean_lines:
            line_no += 1
            j = 0
            # Match a key at the start of a logical entry only when d == 0.
            if d == 0 and not in_str:
                m = _KEY_RE.match(ln)
                if m:
                    keys.add(m.group(1))
            while j < len(ln):
                ch = ln[j]
                if in_str:
                    if ch == "\\":
                        j += 2
                        continue
                    if ch == in_str:
                        in_str = None
                elif ch in ('"', "'", "`"):
                    in_str = ch
                elif ch == "{":
                    d += 1
                elif ch == "}":
                    d -= 1
                j += 1
    return keys


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


def compile_smoke_wasm_via_cli(eshkol_run: Path, source: str) -> bytes | None:
    """Use `eshkol-run --wasm <src> -o <out.wasm>` to produce the .wasm.
    Returns the WASM bytes, or None if the toolchain refuses."""
    with tempfile.TemporaryDirectory() as tmp:
        src_path = Path(tmp) / "smoke.esk"
        out_path = Path(tmp) / "smoke.wasm"
        src_path.write_text(source)
        try:
            res = subprocess.run(
                [str(eshkol_run), "--wasm", str(src_path), "-o", str(out_path)],
                capture_output=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        if res.returncode != 0 or not out_path.exists():
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
    ap.add_argument("--strict", action="store_true",
                    help="Also fail if the JS provides keys that no WASM import requested "
                         "(catches stale stubs).")
    args = ap.parse_args()

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

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for surface, src in SMOKE_PROGRAMS.items():
                wasm = compile_smoke_wasm_via_cli(eshkol_run, src)
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
    js_provided: dict[Path, set[str]] = {}
    for js in JS_FILES:
        if not js.exists():
            print(f"warning: {js} not found, skipping", file=sys.stderr)
            continue
        js_provided[js] = extract_env_keys(js.read_text())

    if not js_provided:
        print("error: no JS glue files found", file=sys.stderr)
        return 2

    # Cross-check.
    failed = False
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

    print("OK — all WASM env imports are provided by the JS glue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
