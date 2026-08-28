#!/usr/bin/env python3
"""Release gate: every tracked PowerShell script is ASCII, or explicitly
marks itself as UTF-8 with a byte-order mark.

Motivating incident (verified on a real Windows PowerShell 5.1 host,
2026-08-27): a BOM-less `.ps1` file containing two U+2014 em dashes parsed
and ran without error under CI, because CI runs PowerShell scripts under
`pwsh` 7, which defaults to UTF-8. Windows PowerShell 5.1 -- the version
that actually ships on a stock Windows machine, and the only one many users
will ever have -- has no such default: with no BOM to tell it otherwise, it
decodes a `.ps1` file in the process's system ANSI code page. Under that
codepage the two non-ASCII bytes of each 3-byte UTF-8 em dash sequence
(`E2 80 94`) decode to two unrelated single-byte characters apiece, planted
in the middle of string literals. The result was 18 cascading parse errors
on a script that CI had just marked green.

CI cannot catch this class of defect by running the script, because CI's
`pwsh` 7 and a user's Windows PowerShell 5.1 do not even agree on what bytes
the file contains once decoded -- the only thing that can be checked is the
file's own bytes, independent of which interpreter later reads them. That is
what this gate does.

Rule enforced, per tracked `*.ps1` / `*.psm1` file:
    - If the file starts with the UTF-8 byte-order mark (`EF BB BF`), any
      byte value is allowed after it: the BOM removes the ambiguity Windows
      PowerShell 5.1 would otherwise resolve via the system code page, so a
      BOM'd file is decoded the same way -- as UTF-8 -- by 5.1 and by pwsh 7
      alike.
    - Otherwise, every byte in the file must be pure ASCII (< 0x80). A
      non-ASCII byte with no BOM is exactly the condition that decodes
      differently between the two interpreters.

The file set is read via `git ls-files -- '*.ps1' '*.psm1'`, so it walks only
tracked files and automatically respects `.gitignore` -- an untracked or
ignored script was never going to run in anyone's CI or ship in a release
either.

Grading
    PASS  every tracked `.ps1`/`.psm1` file is ASCII-only, or opens with the
          UTF-8 BOM.
    FAIL  at least one BOM-less file contains a byte >= 0x80. Every offending
          location is reported as `file:line:col: <codepoint>`.

Usage
    python3 scripts/check_ps1_encoding.py
    python3 scripts/check_ps1_encoding.py --format json
    python3 scripts/check_ps1_encoding.py --self-test

Exit status is 0 on PASS and 1 on FAIL (including under --self-test, where it
reports whether the gate itself is capable of failing).

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "ps1_encoding_gate.jsonl"
PROBE_ID = "ps1_encoding_clean"

UTF8_BOM = b"\xef\xbb\xbf"
PS1_PATTERNS = ("*.ps1", "*.psm1")


class Ps1EncodingError(Exception):
    """Raised when the tracked-file listing itself cannot be produced."""


def list_ps1_files(repo_root: str = REPO_ROOT) -> list[str]:
    """Tracked *.ps1 / *.psm1 paths, relative to repo_root, via git ls-files
    (so untracked/ignored files -- which cannot run in anyone's CI or ship in
    a release -- are never scanned)."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--"] + list(PS1_PATTERNS),
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Ps1EncodingError(f"git ls-files failed: {exc}") from exc
    return [line for line in proc.stdout.splitlines() if line.strip()]


def scan_bytes(data: bytes) -> list[tuple[int, int, str]]:
    """Return a list of (line, col, description) violations for one file's
    raw bytes. Empty means clean. `line`/`col` are 1-based; `col` counts
    decoded characters on lines that are valid UTF-8, or raw bytes on lines
    that are not even that."""
    if data.startswith(UTF8_BOM):
        return []  # BOM present: unambiguous UTF-8 for 5.1 and pwsh 7 alike.

    violations: list[tuple[int, int, str]] = []
    for line_no, line_bytes in enumerate(data.split(b"\n"), start=1):
        if all(b < 0x80 for b in line_bytes):
            continue
        try:
            text = line_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Not even valid UTF-8: report the raw offending bytes directly,
            # there is no codepoint to decode them into.
            for col, b in enumerate(line_bytes, start=1):
                if b >= 0x80:
                    violations.append((line_no, col, f"raw byte 0x{b:02X} (not valid UTF-8 either)"))
            continue
        for col, ch in enumerate(text, start=1):
            if ord(ch) >= 0x80:
                violations.append((line_no, col, f"U+{ord(ch):04X} ({ch!r})"))
    return violations


def check(paths: list[str] | None = None, repo_root: str = REPO_ROOT) -> dict:
    """Grade the given repo-relative paths (default: every tracked
    *.ps1/*.psm1 file). Returns a result dict with 'passed' and per-file
    'findings' (repo-relative path -> list of (line, col, description))."""
    files = list_ps1_files(repo_root) if paths is None else paths
    findings: dict[str, list[tuple[int, int, str]]] = {}
    missing: list[str] = []
    for rel in files:
        full = os.path.join(repo_root, rel)
        if not os.path.isfile(full):
            missing.append(rel)
            continue
        with open(full, "rb") as handle:
            data = handle.read()
        violations = scan_bytes(data)
        if violations:
            findings[rel] = violations

    passed = not findings and not missing
    return {
        "passed": passed,
        "files_scanned": len(files),
        "findings": findings,
        "missing": missing,
    }


def emit_trace(trace_dir: str, status: str, snippet: str) -> str:
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, TRACE_BASENAME)
    event = {
        "kind": "eshkol_smoke",
        "name": PROBE_ID,
        "value": status,
        "snippet": snippet[:2000],
        "confidence": 1.0,
    }
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


def _format_findings(findings: dict[str, list[tuple[int, int, str]]]) -> list[str]:
    lines = []
    for rel in sorted(findings):
        for line_no, col, desc in findings[rel]:
            lines.append(f"{rel}:{line_no}:{col}: {desc}")
    return lines


# ───────────────────────────── self-test ─────────────────────────────
#
# "A gate that cannot fail is not a gate." Each fixture below feeds the gate
# deliberately-broken bytes and asserts it grades FAIL; well-formed fixtures
# assert the gate does NOT fail everything indiscriminately.

_ASCII_ONLY = b'Write-Host "hello, world -- no surprises here"\r\n$x = 1 -- ASCII hyphens only\r\n'

_BOM_UTF8_EM_DASH = UTF8_BOM + "Write-Host \"quantum computing — state of the art\"\r\n".encode("utf-8")

# The exact motivating fixture: no BOM, two U+2014 EM DASH characters landing
# inside string literals on different lines -- Windows PowerShell 5.1 decodes
# each dash's 3 UTF-8 bytes as three separate ANSI-codepage characters,
# corrupting the quoted-string boundary and cascading into parse errors well
# past this line.
_EM_DASH_LINE_1 = "Write-Host \"quantum computing — state of the art\"\r\n"
_EM_DASH_LINE_2 = "Write-Host \"second line — also non-ASCII\"\r\n"
_BOM_LESS_EM_DASH = _EM_DASH_LINE_1.encode("utf-8") + _EM_DASH_LINE_2.encode("utf-8")


def _run_fixture(name: str, data: bytes, tmp_dir: str) -> tuple[bool, list[str]]:
    ext = ".psm1" if name.endswith("_psm1") else ".ps1"
    path = os.path.join(tmp_dir, name + ext)
    with open(path, "wb") as handle:
        handle.write(data)
    violations = scan_bytes(data)
    return not violations, _format_findings({name + ext: violations} if violations else {})


def self_test() -> bool:
    """Run the gate against fixtures with known-bad and known-good bytes.

    Fixtures are written to a temp directory INSIDE the repo (never /tmp),
    exercised as real files on disk exactly as the gate reads them in CI,
    and removed before this function returns.
    """
    cases = [
        ("ascii_only", _ASCII_ONLY, True),
        ("bom_utf8_em_dash", _BOM_UTF8_EM_DASH, True),
        ("bom_less_em_dash", _BOM_LESS_EM_DASH, False),
    ]

    all_ok = True
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-ps1-encoding-") as tmp_dir:
        print("check_ps1_encoding.py self-test:")
        for name, data, expect_pass in cases:
            passed, detail = _run_fixture(name, data, tmp_dir)
            ok = passed == expect_pass
            all_ok = all_ok and ok
            verdict = "OK" if ok else "GATE IS BROKEN"
            print(f"  [{verdict}] {name}: expected passed={expect_pass}, got passed={passed}")
            if detail:
                for line in detail:
                    print(f"           {line}")

        # The bom_less_em_dash fixture must additionally point at the RIGHT
        # location, not merely fail somewhere -- a gate that flags the wrong
        # line is nearly as unhelpful as one that does not flag at all.
        # Expected positions are derived from the fixture text itself (its
        # own decoded character index), never hand-counted, so a rewrite of
        # the fixture strings above can never silently desync from a stale
        # hardcoded column number.
        violations = scan_bytes(_BOM_LESS_EM_DASH)
        expected = [
            (1, _EM_DASH_LINE_1.rstrip("\r\n").index("—") + 1, f"U+2014 ({chr(0x2014)!r})"),
            (2, _EM_DASH_LINE_2.rstrip("\r\n").index("—") + 1, f"U+2014 ({chr(0x2014)!r})"),
        ]
        location_ok = violations == expected
        all_ok = all_ok and location_ok
        verdict = "OK" if location_ok else "GATE IS BROKEN"
        print(f"  [{verdict}] bom_less_em_dash: reports exact file:line:col")
        if not location_ok:
            print(f"           expected {expected}, got {violations}")

    if all_ok:
        print("self-test: PASS -- the gate fails on BOM-less non-ASCII bytes, passes ASCII and BOM'd UTF-8, "
              "and reports the exact offending location")
    else:
        print("self-test: FAIL -- the gate did not discriminate broken input from good input", file=sys.stderr)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    try:
        result = check()
    except Ps1EncodingError as exc:
        snippet = f"could not enumerate tracked PowerShell scripts: {exc}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "FAIL", snippet)
        if args.format == "json":
            print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: FAIL -- {exc}", file=sys.stderr)
        return 1

    status = "PASS" if result["passed"] else "FAIL"
    finding_lines = _format_findings(result["findings"])
    if result["passed"]:
        snippet = f"{result['files_scanned']} tracked .ps1/.psm1 file(s), all ASCII or BOM'd UTF-8"
    else:
        parts = finding_lines[:5]
        if result["missing"]:
            parts.append(f"missing: {', '.join(result['missing'][:5])}")
        snippet = f"{len(finding_lines)} non-ASCII byte(s) with no BOM: " + "; ".join(parts)

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({
            "status": status,
            "files_scanned": result["files_scanned"],
            "findings": {rel: [[l, c, d] for l, c, d in v] for rel, v in result["findings"].items()},
            "missing": result["missing"],
        }, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  files scanned: {result['files_scanned']}")
        if finding_lines:
            print("  VIOLATIONS (BOM-less file, non-ASCII byte -- decodes differently under "
                  "Windows PowerShell 5.1's system code page than under pwsh 7's UTF-8 default):")
            for line in finding_lines:
                print(f"    {line}")
        if result["missing"]:
            print("  MISSING (tracked by git but not present on disk):")
            for rel in result["missing"]:
                print(f"    {rel}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
