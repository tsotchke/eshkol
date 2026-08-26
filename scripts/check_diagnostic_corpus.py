#!/usr/bin/env python3
"""Release gate: the diagnostic golden corpus has not drifted (ADR-0010 gap A11).

Motivating gap (v1.4-connection oracle, 2026-08-25): "no diagnostic
golden-corpus (input -> expected diagnostic code + span) directory exists
anywhere under tests/." Diagnostic quality was a single boolean per fixture
(does the compiler still reject this program) with no assertion on WHAT it
says or WHERE it points. `tests/typesystem/*.esk` already asserts message
substrings (`;; EXPECT-STDERR:`) and the compile/no-binary contract
(`;; EXPECT-COMPILE:`), but nothing anywhere pins the reported
file:line:col span — so ESH-0365 (an import diagnostic that reported the
CLOSING paren's position instead of the `(import` keyword's) or ESH-0364 (a
required module's diagnostic naming the REQUIRING file instead of the
module that actually contains the error) could silently regress and every
existing suite would still say PASS, because both bugs printed a
plausible-looking diagnostic — just at the wrong place, or about the wrong
file.

Corpus layout
    tests/diagnostics/<case-name>/
        input.esk            (or another entry file named by `entry`)
        [other .esk files the entry requires/imports]
        expected.json        (see the module docstring in `evaluate_case`)

Usage
    python3 scripts/check_diagnostic_corpus.py --build-dir build
    python3 scripts/check_diagnostic_corpus.py --self-test

Exit status is 0 on PASS and 1 on FAIL (including under --self-test, where it
reports whether the gate itself is capable of failing).

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import platform as _platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CORPUS_DIR = REPO_ROOT / "tests" / "diagnostics"
DEFAULT_TRACE_DIR = REPO_ROOT / "scripts" / "icc_traces"
TRACE_BASENAME = "diagnostic_corpus_gate.jsonl"
PROBE_ID = "diagnostic_corpus_clean"

# Matches Eshkol's diagnostic line format exactly: `eshkol_error_at` /
# `eshkol_warn_at` (lib/core/logger.cpp) print `file:line:col: error: msg`
# (or `warning:`), one primary line per diagnostic, optionally followed by a
# source snippet and a caret line this regex does not need to parse.
DIAGNOSTIC_LINE_RE = re.compile(
    r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):(?P<col>\d+):\s*(?P<severity>error|warning):\s*(?P<message>.*)$"
)

# The HoTT type checker (gradual/strict-types diagnostics) uses a SECOND,
# file-less format: `[ERROR] Type error: <msg> (line L:C)` /
# `[WARN] Type warning: <msg> (line L:C)` — no `file:` prefix at all, so a
# span assertion against this format can only pin line/col, never file (an
# expected.json entry for one of these simply omits `span.file`).
HOTT_DIAGNOSTIC_LINE_RE = re.compile(
    r"^\[(?P<severity_tag>ERROR|WARN)\]\s*Type (?:error|warning):\s*(?P<message>.*?)\s*\(line (?P<line>\d+):(?P<col>\d+)\)\s*$"
)

# The compiler's diagnostic printer (lib/core/logger.cpp) emits ANSI SGR
# color codes unconditionally, even when stderr is not a TTY (confirmed by
# running it under a captured subprocess). Strip them before matching, or
# every diagnostic line fails the anchored `file:line:col:` match because it
# actually starts with an escape sequence.
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def parse_diagnostic_lines(stderr: str) -> list[dict]:
    """Extract every `file:line:col: error|warning: msg` line from stderr."""
    parsed = []
    for raw_line in strip_ansi(stderr).splitlines():
        stripped_line = raw_line.strip()
        m = DIAGNOSTIC_LINE_RE.match(stripped_line)
        if m:
            parsed.append(
                {
                    "file": m.group("file"),
                    "line": int(m.group("line")),
                    "col": int(m.group("col")),
                    "severity": m.group("severity"),
                    "message": m.group("message"),
                    "raw": raw_line,
                }
            )
            continue
        m = HOTT_DIAGNOSTIC_LINE_RE.match(stripped_line)
        if m:
            parsed.append(
                {
                    "file": None,
                    "line": int(m.group("line")),
                    "col": int(m.group("col")),
                    "severity": "error" if m.group("severity_tag") == "ERROR" else "warning",
                    "message": m.group("message"),
                    "raw": raw_line,
                }
            )
    return parsed


def _diagnostic_matches_span(diag: dict, span: dict) -> bool:
    expected_file = span.get("file")
    if expected_file is not None:
        # Accept either an exact match or a basename match, since the
        # compiler may report a path exactly as given on the command line
        # (relative, absolute, or "./"-prefixed) depending on how the
        # source was reached (direct compile vs require/import).
        actual = diag["file"]
        if actual is None:
            return False
        if actual != expected_file and os.path.basename(actual) != os.path.basename(expected_file):
            return False
    if "line" in span and diag["line"] != span["line"]:
        return False
    if "col" in span and diag["col"] != span["col"]:
        return False
    return True


def evaluate_case(
    expected: dict,
    *,
    compile_exit: int,
    binary_exists: bool,
    stderr: str,
    ran: bool,
    run_exit: int | None,
    stdout: str,
) -> tuple[bool, list[str]]:
    """Grade one corpus case. Never raises; returns (passed, reasons).

    `expected` schema (tests/diagnostics/<case>/expected.json):
      description   : str, human-readable (required)
      esh_ref        : str, optional traceability tag (e.g. "ESH-0364")
      compile        : "fail" | "ok"                        (required)
      diagnostics    : list of {contains: str, span?: {file?, line?, col?}}
                       every entry's `contains` must appear as a substring
                       of some parsed diagnostic's message; if `span` is
                       given, that SAME diagnostic line's file/line/col must
                       match it exactly (this is the span-drift assertion
                       ESH-0364/ESH-0365 exist to pin)
      forbidden      : list of str, must NOT appear anywhere in stderr
      run            : bool, only meaningful when compile == "ok" — execute
                       the compiled binary after a successful compile
      stdout_contains: list of str, checked against captured stdout when run
      exit_code      : int, expected process exit code when run (default 0)
    """

    reasons: list[str] = []
    clean_stderr = strip_ansi(stderr)
    parsed = parse_diagnostic_lines(clean_stderr)

    compile_mode = expected.get("compile")
    if compile_mode == "fail":
        if compile_exit == 0:
            reasons.append("expected a NONZERO compile exit, got 0")
        if binary_exists:
            reasons.append(
                "expected compile to FAIL but a binary was written — the diagnosed "
                "program must not build and run"
            )
    elif compile_mode == "ok":
        if compile_exit != 0:
            reasons.append(f"expected a ZERO compile exit, got {compile_exit}")
        if not binary_exists:
            reasons.append("expected compile to succeed but no binary was produced")
    else:
        reasons.append(f"expected.json has no valid `compile` (fail|ok), got {compile_mode!r}")

    for entry in expected.get("diagnostics", []):
        contains = entry.get("contains", "")
        span = entry.get("span")
        candidates = [d for d in parsed if contains in d["message"] or contains in d["raw"]]
        if not candidates:
            reasons.append(f"no diagnostic line contains {contains!r} (stderr had {len(parsed)} parsed diagnostic line(s))")
            continue
        if span is not None:
            span_ok = any(_diagnostic_matches_span(d, span) for d in candidates)
            if not span_ok:
                got = [(d["file"], d["line"], d["col"]) for d in candidates]
                reasons.append(
                    f"diagnostic containing {contains!r} did not report the expected span "
                    f"{span} — actual span(s) on matching lines: {got}"
                )

    for forbidden in expected.get("forbidden", []):
        if forbidden in clean_stderr:
            reasons.append(f"forbidden stderr text present: {forbidden!r}")

    if compile_mode == "ok" and expected.get("run"):
        if not ran:
            reasons.append("expected.json declares run:true but the case runner did not execute the binary")
        else:
            expected_exit = expected.get("exit_code", 0)
            if run_exit != expected_exit:
                reasons.append(f"expected run exit code {expected_exit}, got {run_exit}")
            for text in expected.get("stdout_contains", []):
                if text not in stdout:
                    reasons.append(f"stdout missing expected text: {text!r}")

    return (not reasons, reasons)


def _check_coverage_rows(rows: list[str], entry: str, construct: str) -> tuple[bool, list[str]]:
    """Pure logic behind check_coverage_position_pin — see that function's
    docstring for the mechanism. Split out so --self-test can exercise it
    against synthetic TSV rows with no compiler invocation."""
    reasons: list[str] = []
    dispatch_pos = None
    accept_pos = None
    for i, row in enumerate(rows):
        fields = row.split("\t")
        if len(fields) < 5:
            continue
        kind, file_field = fields[0], fields[1]
        if kind == "P" and file_field == entry and len(fields) >= 6 and fields[5] == construct:
            dispatch_pos = (int(fields[2]), int(fields[3]))
            # The accept row for a desugared construct is emitted
            # immediately after its dispatch row, same file, kind "A".
            for follow in rows[i + 1 :]:
                ffields = follow.split("\t")
                if len(ffields) < 4:
                    continue
                if ffields[0] == "A" and ffields[1] == file_field:
                    accept_pos = (int(ffields[2]), int(ffields[3]))
                break
            break

    if dispatch_pos is None:
        reasons.append(f"no parser-dispatch ('P') row for construct {construct!r} in file {entry!r} was found in the coverage trace")
    elif accept_pos is None:
        reasons.append(f"found the dispatch row for {construct!r} at {dispatch_pos} but no following accept ('A') row for {entry!r}")
    elif dispatch_pos != accept_pos:
        reasons.append(
            f"dispatch/accept position mismatch for {construct!r}: dispatch at {dispatch_pos}, "
            f"accept at {accept_pos} — this is exactly the ESH-0365 shape (the desugared node took "
            f"its position from the wrong token)"
        )

    return (not reasons, reasons)


def check_coverage_position_pin(
    case_dir: Path, entry: str, construct: str, eshkol_run: Path, build_dir: Path, flags: list[str]
) -> tuple[bool, list[str]]:
    """ESH-0365 regression pin: the parser-dispatch event for `construct`
    (recorded at the operator token) and the codegen-accept event for the
    same construct (recorded when the desugared form's node is emitted)
    must land at the EXACT SAME source position.

    This is the literal mechanism ESH-0365 broke: `(import ...)` desugars to
    a `require` node, and the accept event took its position from the import
    form's spec loop, which exits on the CLOSING PAREN — not from the
    `(import` operator token the dispatch event used. The coverage tracker's
    "same construct is covered" rule is position-only (it deliberately
    ignores operation-kind), so a dispatch/accept position mismatch is
    exactly what a regression here would reintroduce, and it is directly
    observable in `scripts/language_coverage.py`'s own TSV trace format
    (`P <file> <line> <col> <category> <name>` / `A <file> <line> <col>
    <category>`), which this function reads verbatim rather than
    reimplementing the tracker's parsing.
    """
    with tempfile.TemporaryDirectory(dir=case_dir, prefix=".diagnostic-corpus-cov-") as tmp:
        trace_dir = Path(tmp) / "trace"
        trace_dir.mkdir()
        out_bin = Path(tmp) / "cov_out_bin"
        env = dict(os.environ)
        env["ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR"] = str(trace_dir)
        subprocess.run(
            [str(eshkol_run.resolve()), entry, f"-L{build_dir.resolve()}", *flags, "-o", str(out_bin)],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        tsv_files = list(trace_dir.glob("language-coverage-*.tsv"))
        if not tsv_files:
            return False, [f"no language-coverage trace was written to {trace_dir} (ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR not honored?)"]

        rows = tsv_files[0].read_text(encoding="utf-8").splitlines()
        return _check_coverage_rows(rows, entry, construct)


def _eshkol_run_path(build_dir: Path) -> Path:
    suffix = ".exe" if _platform.system().lower() == "windows" else ""
    return build_dir / f"eshkol-run{suffix}"


def _mode_flags(mode: str | None) -> list[str]:
    if mode == "strict-types":
        return ["--strict-types"]
    if mode == "unsafe":
        return ["--unsafe"]
    return []


def run_case(case_dir: Path, expected: dict, eshkol_run: Path, build_dir: Path) -> tuple[bool, list[str], dict]:
    entry = expected.get("entry", "input.esk")
    flags = _mode_flags(expected.get("mode"))

    with tempfile.TemporaryDirectory(dir=case_dir, prefix=".diagnostic-corpus-out-") as tmp:
        out_bin = Path(tmp) / "out_bin"
        proc = subprocess.run(
            [str(eshkol_run.resolve()), entry, f"-L{build_dir.resolve()}", *flags, "-o", str(out_bin)],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        compile_exit = proc.returncode
        stderr = proc.stderr
        binary_exists = out_bin.is_file()

        ran = False
        run_exit: int | None = None
        stdout = ""
        if expected.get("compile") == "ok" and expected.get("run") and binary_exists:
            run_proc = subprocess.run([str(out_bin)], cwd=case_dir, capture_output=True, text=True, timeout=60)
            ran = True
            run_exit = run_proc.returncode
            stdout = run_proc.stdout

    passed, reasons = evaluate_case(
        expected,
        compile_exit=compile_exit,
        binary_exists=binary_exists,
        stderr=stderr,
        ran=ran,
        run_exit=run_exit,
        stdout=stdout,
    )

    coverage_pin = expected.get("coverage_position_pin")
    if coverage_pin:
        pin_passed, pin_reasons = check_coverage_position_pin(
            case_dir, entry, coverage_pin["construct"], eshkol_run, build_dir, flags
        )
        passed = passed and pin_passed
        reasons = reasons + pin_reasons

    detail = {"compile_exit": compile_exit, "stderr": stderr, "stdout": stdout, "run_exit": run_exit}
    return passed, reasons, detail


def discover_cases(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.is_dir():
        return []
    return sorted(p.parent for p in corpus_dir.glob("*/expected.json"))


def emit_trace(trace_dir: Path, status: str, snippet: str) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / TRACE_BASENAME
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


# ───────────────────────────── self-test ─────────────────────────────
#
# Pure-function fixtures against evaluate_case() — no compiler invocation
# needed, matching check_ledger_integrity.py's style, so this self-test runs
# anywhere Python + this file are available (the lightweight assurance-gates
# CI job does not build the compiler).


def self_test() -> bool:
    all_ok = True
    print("check_diagnostic_corpus.py self-test:")

    cases = [
        (
            "well_formed_span_match",
            {"compile": "fail", "diagnostics": [{"contains": "expects 2 arguments, got 4", "span": {"file": "input.esk", "line": 9, "col": 10}}]},
            dict(compile_exit=1, binary_exists=False, stderr="input.esk:9:10: error: expects 2 arguments, got 4\n", ran=False, run_exit=None, stdout=""),
            True,
        ),
        (
            "wrong_span_is_drift",
            {"compile": "fail", "diagnostics": [{"contains": "expects 2 arguments, got 4", "span": {"file": "input.esk", "line": 9, "col": 10}}]},
            # Same message, but the compiler now reports a DIFFERENT
            # location — this is exactly the ESH-0365 shape of regression.
            dict(compile_exit=1, binary_exists=False, stderr="input.esk:1:1: error: expects 2 arguments, got 4\n", ran=False, run_exit=None, stdout=""),
            False,
        ),
        (
            "wrong_file_is_drift",
            {"compile": "fail", "diagnostics": [{"contains": "boom", "span": {"file": "module.esk", "line": 3, "col": 5}}]},
            # ESH-0364 shape: the diagnostic fires but blames the wrong file.
            dict(compile_exit=1, binary_exists=False, stderr="main.esk:3:5: error: boom\n", ran=False, run_exit=None, stdout=""),
            False,
        ),
        (
            "compile_fail_but_binary_written",
            {"compile": "fail", "diagnostics": [{"contains": "boom"}]},
            # The "diagnosed program must not build and run" contract: even
            # with the right message, a binary written on a fail case is a
            # violation on its own.
            dict(compile_exit=1, binary_exists=True, stderr="input.esk:1:1: error: boom\n", ran=False, run_exit=None, stdout=""),
            False,
        ),
        (
            "compile_ok_run_contract",
            {"compile": "ok", "run": True, "exit_code": 0, "stdout_contains": ["caught it"]},
            dict(compile_exit=0, binary_exists=True, stderr="", ran=True, run_exit=0, stdout="caught it\n"),
            True,
        ),
        (
            "compile_ok_run_wrong_stdout",
            {"compile": "ok", "run": True, "exit_code": 0, "stdout_contains": ["caught it"]},
            dict(compile_exit=0, binary_exists=True, stderr="", ran=True, run_exit=0, stdout="something else\n"),
            False,
        ),
        (
            "forbidden_text_present",
            {"compile": "ok", "forbidden": ["Unknown function"]},
            dict(compile_exit=0, binary_exists=True, stderr="Unknown function: the\n", ran=False, run_exit=None, stdout=""),
            False,
        ),
    ]

    for name, expected, actual, expect_pass in cases:
        passed, reasons = evaluate_case(expected, **actual)
        ok = passed == expect_pass
        all_ok = all_ok and ok
        verdict = "OK" if ok else "GATE IS BROKEN"
        print(f"  [{verdict}] {name}: expected passed={expect_pass}, got passed={passed}")
        if reasons:
            print(f"           {'; '.join(reasons)}")

    coverage_cases = [
        (
            "coverage_pin_matching_positions",
            ["P\tinput.esk\t1\t2\t31\timport", "A\tinput.esk\t1\t2\t32"],
            True,
        ),
        (
            "coverage_pin_esh_0365_shape_mismatch",
            # Dispatch at the `(import` keyword (1,2); accept at the
            # closing paren (2,15) instead — the exact pre-fix defect shape.
            ["P\tinput.esk\t1\t2\t31\timport", "A\tinput.esk\t2\t15\t32"],
            False,
        ),
        (
            "coverage_pin_missing_dispatch_row",
            ["P\tinput.esk\t1\t2\t31\tdisplay", "A\tinput.esk\t1\t2\t7"],
            False,
        ),
        (
            "coverage_pin_missing_accept_row",
            ["P\tinput.esk\t1\t2\t31\timport"],
            False,
        ),
    ]
    for name, rows, expect_pass in coverage_cases:
        passed, reasons = _check_coverage_rows(rows, "input.esk", "import")
        ok = passed == expect_pass
        all_ok = all_ok and ok
        verdict = "OK" if ok else "GATE IS BROKEN"
        print(f"  [{verdict}] {name}: expected passed={expect_pass}, got passed={passed}")
        if reasons:
            print(f"           {'; '.join(reasons)}")

    if all_ok:
        print("self-test: PASS — the gate fails on every broken fixture and passes the well-formed one")
    else:
        print("self-test: FAIL — the gate did not discriminate broken input from good input", file=sys.stderr)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--build-dir", default="build", help="directory containing a built eshkol-run")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    corpus_dir = Path(args.corpus_dir)
    build_dir = Path(args.build_dir)
    eshkol_run = _eshkol_run_path(build_dir)

    if not eshkol_run.is_file():
        snippet = f"eshkol-run not found at {eshkol_run} (build first, or pass --build-dir)"
        if not args.no_trace:
            emit_trace(Path(args.trace_dir), "FAIL", snippet)
        print(f"{PROBE_ID}: FAIL — {snippet}", file=sys.stderr)
        return 1

    case_dirs = discover_cases(corpus_dir)
    if not case_dirs:
        snippet = f"no cases found under {corpus_dir} (expected.json missing everywhere) — the gate fails closed"
        if not args.no_trace:
            emit_trace(Path(args.trace_dir), "FAIL", snippet)
        print(f"{PROBE_ID}: FAIL — {snippet}", file=sys.stderr)
        return 1

    all_passed = True
    results = []
    for case_dir in case_dirs:
        expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
        passed, reasons, detail = run_case(case_dir, expected, eshkol_run, build_dir)
        all_passed = all_passed and passed
        results.append({"case": case_dir.name, "passed": passed, "reasons": reasons})
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {case_dir.name}")
        if not passed:
            for reason in reasons:
                print(f"      reason: {reason}")
            if detail["stderr"]:
                print(f"      stderr: {detail['stderr'][:400]}")

    status = "PASS" if all_passed else "FAIL"
    n_pass = sum(1 for r in results if r["passed"])
    snippet = f"{n_pass}/{len(results)} diagnostic corpus cases passed"

    if not args.no_trace:
        emit_trace(Path(args.trace_dir), status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, "results": results}, indent=2))
    else:
        print(f"{PROBE_ID}: {status} ({snippet})")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
