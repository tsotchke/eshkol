#!/usr/bin/env python3
"""Validate `results-<mode>.json`, the artifact `scripts/doc_audit/run_examples.py`
writes for every documentation-example execution run.

WHY THIS EXISTS

The v1.3.4 ICC readiness run scored ready/97 with exactly one low-severity
gap: `artifact_without_test_or_trace` on `Artifact:results-%s.json` —
produced by `run_examples.py` (~line 107) with nothing validating it. ICC's
suggested action was `add_artifact_validator_or_runtime_probe`. This is the
validator half of that fix; `run_examples.py` also now emits a runtime_event
trace (the "do both, cheap" half — see its own header comment) recording
whether this validator passed against the results it just wrote.

WHAT IT CHECKS

  1. The file parses as a JSON list.
  2. Every record has the required fields `run_examples.py` promises in its
     own docstring: file, start_line, end_line, klass, mode, exit, stdout,
     stderr, seconds, expects — with the right shapes (start_line/end_line/
     exit/seconds numeric, file/klass/mode strings, expects a list).
  3. No duplicate (file, start_line, mode) triple — the same example
     recorded twice hides whichever result comes second.
  4. Records are sorted by (file, start_line), matching exactly what
     `run_examples.py` itself does before writing
     (`results.sort(key=lambda r: (r["file"], r["start_line"]))`) — an
     out-of-order file was hand-edited or produced by a different tool
     entirely and should not be trusted as this harness's own output.
  5. Optional: when `--examples FILE` (the `examples.json` the run was
     driven from) is given, the record count must equal the number of
     examples discovered there (after applying the same `--only` filter, if
     the caller used one) — proving the results file is neither truncated
     nor padded relative to what was supposed to run.

Grading
    PASS  the file parses, every record is well-formed, no duplicates, the
          file is sorted, and (if --examples given) the count matches.
    FAIL  a parse error, a malformed record, a duplicate, an out-of-order
          file, or a count mismatch.

The gate FAILS CLOSED: a missing or unparseable results file is FAIL, not a
silent pass — the same discipline scripts/check_ledger_integrity.py and
scripts/check_oracle_schema.py already use.

Usage
    python3 scripts/doc_audit/check_results_schema.py path/to/results-jit.json
    python3 scripts/doc_audit/check_results_schema.py results-jit.json --examples examples.json
    python3 scripts/doc_audit/check_results_schema.py results-jit.json --format json
    python3 scripts/doc_audit/check_results_schema.py --self-test

Exit status is 0 on PASS and 1 on FAIL (including under --self-test).

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "doc_examples_results_gate.jsonl"
PROBE_ID = "doc_examples_results_valid"

REQUIRED_FIELDS = ("file", "start_line", "end_line", "klass", "mode", "exit", "stdout", "stderr", "seconds", "expects")
STRING_FIELDS = ("file", "klass", "mode")
INT_FIELDS = ("start_line", "end_line", "exit")
NUMERIC_FIELDS = ("seconds",)


class ResultsSchemaError(Exception):
    """The results file could not be read or parsed at all."""


def _load_json(path: str) -> object:
    if not os.path.isfile(path):
        raise ResultsSchemaError(f"results file not found at {path} (the gate fails closed)")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001 - report any parse failure uniformly
        raise ResultsSchemaError(f"results file at {path} is not valid JSON: {exc}") from exc


def check(data: object, expected_count: int | None = None) -> dict:
    """Validate a parsed results document. Never raises; returns a report."""
    errors: list[str] = []

    if not isinstance(data, list):
        return {"passed": False, "errors": ["results document is not a JSON list"],
                "record_count": 0, "duplicates": [], "sorted": False}

    seen: dict[tuple, int] = {}
    duplicates: list[str] = []
    keys: list[tuple] = []

    for position, rec in enumerate(data, start=1):
        label = f"record #{position}"
        if not isinstance(rec, dict):
            errors.append(f"{label} is not an object")
            continue

        if "file" in rec and "start_line" in rec:
            label = f"record #{position} ({rec.get('file')}:{rec.get('start_line')})"

        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            errors.append(f"{label} is missing required field(s): {', '.join(missing)}")

        for f in STRING_FIELDS:
            if f in rec and not isinstance(rec[f], str):
                errors.append(f"{label}: field {f!r} must be a string, got {type(rec[f]).__name__}")
        for f in INT_FIELDS:
            if f in rec and not isinstance(rec[f], int):
                errors.append(f"{label}: field {f!r} must be an integer, got {type(rec[f]).__name__}")
        for f in NUMERIC_FIELDS:
            if f in rec and not isinstance(rec[f], (int, float)):
                errors.append(f"{label}: field {f!r} must be numeric, got {type(rec[f]).__name__}")
        if "expects" in rec and not isinstance(rec["expects"], list):
            errors.append(f"{label}: field 'expects' must be a list, got {type(rec['expects']).__name__}")

        if "file" in rec and "start_line" in rec and "mode" in rec:
            key = (rec["file"], rec["start_line"], rec["mode"])
            if key in seen:
                duplicates.append(f"{key[0]}:{key[1]} (mode={key[2]})")
            else:
                seen[key] = position
            keys.append((rec.get("file"), rec.get("start_line")))

    is_sorted = keys == sorted(keys, key=lambda k: (k[0] is None, k[0], k[1] is None, k[1]))
    if keys and not is_sorted:
        errors.append(
            "records are not sorted by (file, start_line) — run_examples.py always sorts "
            "before writing; an unsorted file was not produced by this harness as-is"
        )

    if duplicates:
        errors.append(f"{len(duplicates)} duplicate (file, start_line, mode) record(s): " + "; ".join(duplicates[:5]))

    if expected_count is not None and len(data) != expected_count:
        errors.append(
            f"record count {len(data)} does not match {expected_count} example(s) discovered "
            f"in the --examples manifest — the results file is truncated or padded"
        )

    passed = not errors
    return {
        "passed": passed,
        "errors": errors,
        "record_count": len(data),
        "duplicates": duplicates,
        "sorted": is_sorted,
    }


def count_examples(examples_path: str, only: str | None = None) -> int:
    with open(examples_path, "r", encoding="utf-8") as fh:
        recs = json.load(fh)
    if only:
        sel = set(only.split(","))
        recs = [r for r in recs if r["file"] in sel or ("%s:%d" % (r["file"], r["start_line"])) in sel]
    return len(recs)


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


# ───────────────────────────────── self-test ────────────────────────────────

_GOOD = [
    {"file": "docs/A.md", "start_line": 10, "end_line": 12, "klass": "esk", "mode": "jit",
     "exit": 0, "stdout": "ok\n", "stderr": "", "seconds": 0.1, "expects": ["ok"]},
    {"file": "docs/A.md", "start_line": 20, "end_line": 22, "klass": "esk", "mode": "jit",
     "exit": 0, "stdout": "42\n", "stderr": "", "seconds": 0.2, "expects": []},
    {"file": "docs/B.md", "start_line": 5, "end_line": 6, "klass": "esk", "mode": "jit",
     "exit": 1, "stdout": "", "stderr": "error: boom\n", "seconds": 0.05, "expects": []},
]

_MISSING_FIELD = [
    {"file": "docs/A.md", "start_line": 10, "klass": "esk", "mode": "jit",
     "exit": 0, "stdout": "", "stderr": "", "seconds": 0.1, "expects": []},
]

_DUPLICATE = _GOOD + [dict(_GOOD[0])]

_UNSORTED = [_GOOD[1], _GOOD[0], _GOOD[2]]

_WRONG_TYPE = [
    {"file": "docs/A.md", "start_line": "10", "end_line": 12, "klass": "esk", "mode": "jit",
     "exit": 0, "stdout": "", "stderr": "", "seconds": 0.1, "expects": []},
]


def self_test() -> bool:
    all_ok = True
    print("check_results_schema.py self-test:")

    cases = [
        ("well_formed", _GOOD, None, True),
        ("missing_field", _MISSING_FIELD, None, False),
        ("duplicate_record", _DUPLICATE, None, False),
        ("unsorted", _UNSORTED, None, False),
        ("wrong_type", _WRONG_TYPE, None, False),
        ("count_matches", _GOOD, 3, True),
        ("count_mismatch", _GOOD, 5, False),
    ]

    for name, data, expected_count, expect_pass in cases:
        result = check(data, expected_count)
        ok = result["passed"] == expect_pass
        all_ok = all_ok and ok
        verdict = "OK" if ok else "GATE IS BROKEN"
        print(f"  [{verdict}] {name}: expected passed={expect_pass}, got passed={result['passed']}")
        if result["errors"]:
            print(f"           {'; '.join(result['errors'][:2])}")

    # End-to-end: real files on disk, including a real examples.json for the
    # count cross-check, inside a repo-local temp directory.
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-doc-results-schema-") as tmp_dir:
        results_path = os.path.join(tmp_dir, "results-jit.json")
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(_GOOD, fh)
        examples_path = os.path.join(tmp_dir, "examples.json")
        with open(examples_path, "w", encoding="utf-8") as fh:
            json.dump([{"file": r["file"], "start_line": r["start_line"]} for r in _GOOD], fh)

        data = _load_json(results_path)
        n = count_examples(examples_path)
        result = check(data, n)
        ok = result["passed"] and n == 3
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] end_to_end_file_and_examples_count: "
              f"passed={result['passed']}, examples_count={n}")

        missing_path = os.path.join(tmp_dir, "does_not_exist.json")
        try:
            _load_json(missing_path)
            ok2 = False
        except ResultsSchemaError:
            ok2 = True
        all_ok = all_ok and ok2
        print(f"  [{'OK' if ok2 else 'GATE IS BROKEN'}] missing_file_fails_closed: raised={ok2}")

    if all_ok:
        print("self-test: PASS — the gate fails on every broken fixture and passes the well-formed one")
    else:
        print("self-test: FAIL — the gate did not discriminate broken input from good input", file=sys.stderr)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="?", help="path to results-<mode>.json")
    parser.add_argument("--examples", default=None, help="examples.json to cross-check the record count against")
    parser.add_argument("--only", default=None, help="same --only filter used for the run_examples.py invocation being checked")
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    if not args.results:
        parser.error("the results positional argument is required unless --self-test is given")

    try:
        data = _load_json(args.results)
        expected_count = count_examples(args.examples, args.only) if args.examples else None
        result = check(data, expected_count)
    except ResultsSchemaError as exc:
        snippet = f"results file unusable: {exc}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "FAIL", snippet)
        if args.format == "json":
            print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: FAIL — {exc}", file=sys.stderr)
        return 1

    status = "PASS" if result["passed"] else "FAIL"
    if result["passed"]:
        snippet = f"{result['record_count']} records, sorted, no duplicates"
    else:
        snippet = f"{len(result['errors'])} schema error(s): " + "; ".join(result["errors"][:5])

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  results file : {args.results}")
        print(f"  records      : {result['record_count']}")
        print(f"  sorted       : {result['sorted']}")
        if result["errors"]:
            print("  ERRORS:")
            for error in result["errors"]:
                print(f"    - {error}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
