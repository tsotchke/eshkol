#!/usr/bin/env python3
"""Check the Navier-Stokes proof ledger for internal and cross-file consistency.

`.icc/navier-stokes-proof-ledger.yaml` is the machine-readable twin of
`docs/design/NAVIER_STOKES_PROOF_LEDGER.md`, which in turn gives a status to
every row of the pipeline table in
`docs/design/NAVIER_STOKES_BLOWUP_MECHANIZATION.md` Section 2. None of these
three files enforces the others by construction — a hand edit to any one of
them (adding a row to the mechanization doc, renaming a CTest, relaxing a
status from ANALYTIC-ONLY to EXACT without wiring a program) can drift
silently. This gate is the check that catches that drift.

It answers five questions a YAML parser alone does not:

  1. Does the ledger parse, and does every row carry the minimum shape (an
     `id`, a `row` number, a `status` from the fixed three-word vocabulary,
     and — depending on status — either a nonempty `programs` list or a
     nonempty `missing_capability` string)?

  2. Does the ledger cover EXACTLY the row numbers the mechanization doc's
     own Section 2 pipeline table declares — no row invented, none dropped?
     (`--doc` / DEFAULT_DOC)

  3. For every row whose status is EXACT or VALIDATED, does each program it
     names resolve to a REAL, currently-wired `ns_*` CTest entry in
     CMakeLists.txt (`ESHKOL_NS_EXAMPLES`)? A row cannot be "checked by a
     passing program" if the program is not actually wired into CTest.

  4. For every row whose status is EXACT specifically, does the referenced
     example program actually LOOK like exact-rational Eshkol code (it uses
     `exact?` or carries rational literals), rather than being float-only?
     This is a syntactic heuristic, not a proof — the real evidence is the
     `ctest -R 'ns_'` run in the release gate — but it is enough to catch the
     concrete failure mode this gate exists to prevent: a status hand-edited
     to EXACT for a program that never touches the exact tower at all.

  5. Does `.icc/completion-oracles.yaml` carry the `navier-stokes-mechanization`
     oracle with the `navier_stokes_proof_ledger_consistent` criterion this
     gate itself is bound to?

Grading
    PASS  the ledger parses, covers exactly the doc's row set, every
          EXACT/VALIDATED row's programs are real CTest entries that look
          exact-rational where claimed EXACT, every ANALYTIC-ONLY row names a
          missing capability, and the oracle criterion exists.
    FAIL  any of the above is violated, including a missing or unparseable
          ledger (the gate fails closed).

Usage
    python3 scripts/check_ns_proof_ledger.py
    python3 scripts/check_ns_proof_ledger.py --ledger path/to/ledger.yaml
    python3 scripts/check_ns_proof_ledger.py --format json
    python3 scripts/check_ns_proof_ledger.py --no-trace
    python3 scripts/check_ns_proof_ledger.py --self-test

Exit status is 0 on PASS and 1 on FAIL (including under --self-test, where it
reports whether the gate itself is capable of failing).

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LEDGER = os.path.join(REPO_ROOT, ".icc", "navier-stokes-proof-ledger.yaml")
DEFAULT_DOC = os.path.join(REPO_ROOT, "docs", "design", "NAVIER_STOKES_BLOWUP_MECHANIZATION.md")
DEFAULT_CMAKE = os.path.join(REPO_ROOT, "CMakeLists.txt")
DEFAULT_ORACLES = os.path.join(REPO_ROOT, ".icc", "completion-oracles.yaml")
DEFAULT_EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "ns_proof_ledger_gate.jsonl"
PROBE_ID = "navier_stokes_proof_ledger_consistent"

STATUS_VOCAB = ("EXACT", "VALIDATED", "ANALYTIC-ONLY")
ORACLE_NAME = "navier-stokes-mechanization"
ORACLE_CRITERION = "navier_stokes_proof_ledger_consistent"

DOC_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|", re.M)
CMAKE_LIST_RE = re.compile(r"set\(ESHKOL_NS_EXAMPLES(.*?)\)", re.S)
CMAKE_PAIR_RE = re.compile(r"(ns_[A-Za-z0-9_]+)\s+(mathematics_navier_stokes_[A-Za-z0-9_]+)")

EXACT_MARKER_RE = re.compile(r"\(exact\?\s|\b\d+/\d+\b")


class LedgerError(Exception):
    """The ledger, doc, CMake file or oracle file could not be read/parsed."""


def _load_yaml_text(text: str, what: str):
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LedgerError("PyYAML is required (pip install pyyaml)") from exc
    try:
        return yaml.safe_load(text)
    except Exception as exc:
        raise LedgerError(f"{what} is not parseable YAML: {exc}") from exc


def _read(path: str, what: str) -> str:
    if not os.path.isfile(path):
        raise LedgerError(f"{what} not found at {path} (the gate fails closed)")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def parse_doc_row_ids(doc_text: str) -> set[int]:
    """Row numbers the mechanization doc's Section 2 pipeline table declares.

    Restricted to the '## 2. The proof as a pipeline' .. '## 3.' span so a
    bare leading integer in an unrelated table can never be mistaken for a
    pipeline row.
    """
    start = doc_text.find("## 2. The proof as a pipeline")
    end = doc_text.find("## 3. Capability ledger")
    if start == -1 or end == -1 or end <= start:
        raise LedgerError("could not locate the Section 2 pipeline table in the mechanization doc")
    span = doc_text[start:end]
    return {int(m.group(1)) for m in DOC_ROW_RE.finditer(span)}


def parse_cmake_ns_examples(cmake_text: str) -> dict[str, str]:
    """{ctest base name: example file basename (no .esk)} from ESHKOL_NS_EXAMPLES."""
    m = CMAKE_LIST_RE.search(cmake_text)
    if not m:
        raise LedgerError("ESHKOL_NS_EXAMPLES list not found in CMakeLists.txt")
    body = m.group(1)
    pairs = CMAKE_PAIR_RE.findall(body)
    if not pairs:
        raise LedgerError("ESHKOL_NS_EXAMPLES list parsed but yielded no name/file pairs")
    return {name: fname for name, fname in pairs}


def program_looks_exact(examples_dir: str, file_basename: str) -> tuple[bool, str]:
    path = os.path.join(examples_dir, file_basename + ".esk")
    if not os.path.isfile(path):
        return False, f"example file not found: {path}"
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if EXACT_MARKER_RE.search(text):
        return True, "contains an (exact? ...) check or a rational literal"
    return False, "no (exact? ...) check and no rational literal found — looks float-only"


def check_oracle_has_criterion(oracles_data) -> tuple[bool, str]:
    if not isinstance(oracles_data, dict) or not isinstance(oracles_data.get("oracles"), list):
        return False, "completion-oracles.yaml has no top-level 'oracles' list"
    for oracle in oracles_data["oracles"]:
        if not isinstance(oracle, dict):
            continue
        if oracle.get("name") != ORACLE_NAME:
            continue
        requires = oracle.get("requires") or []
        for criterion in requires:
            if not isinstance(criterion, dict):
                continue
            label = criterion.get("label", "")
            action = criterion.get("action", "")
            re_block = criterion.get("runtime_event") or {}
            names = re_block.get("event_names") or []
            if ORACLE_CRITERION in names or ORACLE_CRITERION in str(label) or ORACLE_CRITERION in str(action):
                return True, f"found under oracle '{ORACLE_NAME}'"
        return False, f"oracle '{ORACLE_NAME}' exists but has no '{ORACLE_CRITERION}' criterion"
    return False, f"oracle '{ORACLE_NAME}' not found in completion-oracles.yaml"


def check(ledger_data, doc_text: str, cmake_text: str, oracles_data,
          examples_dir: str) -> dict:
    errors: list[str] = []

    if not isinstance(ledger_data, dict) or not isinstance(ledger_data.get("rows"), list):
        raise LedgerError("ledger has no top-level 'rows' list")

    rows = ledger_data["rows"]
    cmake_names = parse_cmake_ns_examples(cmake_text)
    doc_row_ids = parse_doc_row_ids(doc_text)

    seen_rows: dict[int, int] = {}
    status_counts = {s: 0 for s in STATUS_VOCAB}
    exact_ok = 0

    for entry in rows:
        if not isinstance(entry, dict):
            errors.append(f"non-mapping row entry: {entry!r}")
            continue
        rid = entry.get("id", "<missing id>")
        row_no = entry.get("row")
        status = entry.get("status")
        programs = entry.get("programs") or []
        missing_cap = entry.get("missing_capability")

        if row_no is None or not isinstance(row_no, int):
            errors.append(f"{rid}: missing or non-integer 'row'")
        else:
            seen_rows[row_no] = seen_rows.get(row_no, 0) + 1

        if status not in STATUS_VOCAB:
            errors.append(f"{rid}: status {status!r} is not one of {STATUS_VOCAB}")
            continue
        status_counts[status] += 1

        if status in ("EXACT", "VALIDATED"):
            if not programs:
                errors.append(f"{rid}: status {status} but 'programs' is empty (no passing program named)")
            for prog in programs:
                if prog not in cmake_names:
                    errors.append(
                        f"{rid}: program '{prog}' is not a currently-wired ns_* CTest name "
                        f"in CMakeLists.txt ESHKOL_NS_EXAMPLES"
                    )
                    continue
                if status == "EXACT":
                    ok, why = program_looks_exact(examples_dir, cmake_names[prog])
                    if not ok:
                        errors.append(f"{rid}: status EXACT but program '{prog}' {why}")
                    else:
                        exact_ok += 1
            if missing_cap:
                errors.append(f"{rid}: status {status} but 'missing_capability' is set (should be null)")
        elif status == "ANALYTIC-ONLY":
            if not missing_cap or not str(missing_cap).strip():
                errors.append(f"{rid}: status ANALYTIC-ONLY but 'missing_capability' is empty")

    dup_rows = {k: v for k, v in seen_rows.items() if v > 1}
    for row_no, count in sorted(dup_rows.items()):
        errors.append(f"row {row_no} appears {count} times in the ledger (must be unique)")

    ledger_row_ids = set(seen_rows.keys())
    missing_from_ledger = doc_row_ids - ledger_row_ids
    extra_in_ledger = ledger_row_ids - doc_row_ids
    if missing_from_ledger:
        errors.append(
            f"{len(missing_from_ledger)} row(s) in the mechanization doc have no ledger entry: "
            + ", ".join(str(r) for r in sorted(missing_from_ledger)[:10])
            + (" ..." if len(missing_from_ledger) > 10 else "")
        )
    if extra_in_ledger:
        errors.append(
            f"{len(extra_in_ledger)} ledger row(s) do not correspond to any doc row: "
            + ", ".join(str(r) for r in sorted(extra_in_ledger)[:10])
            + (" ..." if len(extra_in_ledger) > 10 else "")
        )

    oracle_ok, oracle_detail = check_oracle_has_criterion(oracles_data)
    if not oracle_ok:
        errors.append(f"oracle wiring: {oracle_detail}")

    passed = not errors
    return {
        "passed": passed,
        "errors": errors,
        "row_count": len(rows),
        "doc_row_count": len(doc_row_ids),
        "status_counts": status_counts,
        "exact_programs_verified": exact_ok,
        "oracle_detail": oracle_detail,
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


# ───────────────────────────── self-test ─────────────────────────────
#
# "A gate that cannot fail is not a gate." Each fixture below feeds the gate
# deliberately-broken input and asserts it grades FAIL; one well-formed
# fixture asserts the gate does NOT grade every input FAIL regardless of
# content.

_GOOD_DOC = """
## 2. The proof as a pipeline

| # | Paper reference | Operation | Eshkol primitive or BUILD ITEM | Status | Version | Gate |
|---|---|---|---|---|---|---|
| 1 | ref | op one | prim | SHIPPED | v1.3.0 | `ns_gate_one` |
| 2 | ref | op two | prim | PLANNED | v1.4.0 | `ns_gate_two` |

## 3. Capability ledger
"""

_GOOD_CMAKE = """
    set(ESHKOL_NS_EXAMPLES
        ns_gate_one   mathematics_navier_stokes_selftest_good)
"""

_GOOD_ORACLES = {
    "oracles": [
        {
            "name": ORACLE_NAME,
            "requires": [
                {
                    "runtime_event": {"event_names": [ORACLE_CRITERION]},
                    "severity": "high",
                    "label": "self-test criterion",
                    "action": "./scripts/check_ns_proof_ledger.py",
                }
            ],
        }
    ]
}

_GOOD_LEDGER_YAML = """
rows:
  - id: ns-proof-001
    row: 1
    status: EXACT
    programs: [ns_gate_one]
    missing_capability: null
  - id: ns-proof-002
    row: 2
    status: ANALYTIC-ONLY
    programs: []
    missing_capability: "some named missing capability"
"""

_MALFORMED_YAML = """
rows:
  - id: ns-proof-001
    row: 1
      status: EXACT
    programs: [ns_gate_one]
"""

_DUPLICATE_ROW_YAML = """
rows:
  - id: ns-proof-001a
    row: 1
    status: EXACT
    programs: [ns_gate_one]
    missing_capability: null
  - id: ns-proof-001b
    row: 1
    status: ANALYTIC-ONLY
    programs: []
    missing_capability: "duplicate row number"
  - id: ns-proof-002
    row: 2
    status: ANALYTIC-ONLY
    programs: []
    missing_capability: "some named missing capability"
"""

_EMPTY_MISSING_CAP_YAML = """
rows:
  - id: ns-proof-001
    row: 1
    status: EXACT
    programs: [ns_gate_one]
    missing_capability: null
  - id: ns-proof-002
    row: 2
    status: ANALYTIC-ONLY
    programs: []
    missing_capability: ""
"""

_UNWIRED_PROGRAM_YAML = """
rows:
  - id: ns-proof-001
    row: 1
    status: EXACT
    programs: [gate_that_does_not_exist_in_cmake]
    missing_capability: null
  - id: ns-proof-002
    row: 2
    status: ANALYTIC-ONLY
    programs: []
    missing_capability: "some named missing capability"
"""

_INEXACT_PROGRAM_LEDGER_YAML = """
rows:
  - id: ns-proof-001
    row: 1
    status: EXACT
    programs: [ns_gate_float_only]
    missing_capability: null
  - id: ns-proof-002
    row: 2
    status: ANALYTIC-ONLY
    programs: []
    missing_capability: "some named missing capability"
"""

_INEXACT_PROGRAM_CMAKE = """
    set(ESHKOL_NS_EXAMPLES
        ns_gate_float_only   mathematics_navier_stokes_selftest_float)
"""

_FLOAT_ONLY_PROGRAM_SOURCE = """
;; self-test fixture: no exact? call, no rational literal, float-only.
(define x 3.14)
(display (* x 2.0)) (newline)
"""

_MISSING_ROW_LEDGER_YAML = """
rows:
  - id: ns-proof-001
    row: 1
    status: EXACT
    programs: [ns_gate_one]
    missing_capability: null
"""


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _run_fixture(tmp_dir: str, ledger_yaml: str, doc_text: str, cmake_text: str,
                  oracles_data, examples_files: dict[str, str]) -> tuple[bool, str]:
    examples_dir = os.path.join(tmp_dir, "examples")
    for fname, content in examples_files.items():
        _write(os.path.join(examples_dir, fname + ".esk"), content)
    try:
        ledger_data = _load_yaml_text(ledger_yaml, "ledger")
        result = check(ledger_data, doc_text, cmake_text, oracles_data, examples_dir)
    except LedgerError as exc:
        return False, f"error (may be expected): {exc}"
    return result["passed"], "; ".join(result["errors"][:3]) or "no errors"


def self_test() -> bool:
    """Run the gate against fixtures with known-bad and known-good shape.

    Fixtures live in a temp directory INSIDE the repo (never /tmp) so the
    gate is exercised against real files on disk exactly as it runs in CI.
    """
    good_examples = {"mathematics_navier_stokes_selftest_good": "(exact? (/ 1 3))\n"}
    float_examples = {"mathematics_navier_stokes_selftest_float": _FLOAT_ONLY_PROGRAM_SOURCE}

    cases = [
        ("malformed_yaml", _MALFORMED_YAML, _GOOD_DOC, _GOOD_CMAKE, _GOOD_ORACLES, good_examples, False),
        ("duplicate_row", _DUPLICATE_ROW_YAML, _GOOD_DOC, _GOOD_CMAKE, _GOOD_ORACLES, good_examples, False),
        ("empty_missing_capability", _EMPTY_MISSING_CAP_YAML, _GOOD_DOC, _GOOD_CMAKE, _GOOD_ORACLES, good_examples, False),
        ("unwired_program", _UNWIRED_PROGRAM_YAML, _GOOD_DOC, _GOOD_CMAKE, _GOOD_ORACLES, good_examples, False),
        ("inexact_program_claims_exact", _INEXACT_PROGRAM_LEDGER_YAML, _GOOD_DOC, _INEXACT_PROGRAM_CMAKE,
         _GOOD_ORACLES, float_examples, False),
        ("missing_doc_row", _MISSING_ROW_LEDGER_YAML, _GOOD_DOC, _GOOD_CMAKE, _GOOD_ORACLES, good_examples, False),
        ("well_formed", _GOOD_LEDGER_YAML, _GOOD_DOC, _GOOD_CMAKE, _GOOD_ORACLES, good_examples, True),
    ]

    all_ok = True
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-ns-proof-ledger-") as tmp_dir:
        print("check_ns_proof_ledger.py self-test:")
        for name, ledger_yaml, doc_text, cmake_text, oracles_data, examples_files, expect_pass in cases:
            passed, detail = _run_fixture(tmp_dir, ledger_yaml, doc_text, cmake_text, oracles_data, examples_files)
            ok = passed == expect_pass
            all_ok = all_ok and ok
            verdict = "OK" if ok else "GATE IS BROKEN"
            print(f"  [{verdict}] {name}: expected passed={expect_pass}, got passed={passed}")
            print(f"           {detail}")

    if all_ok:
        print("self-test: PASS — the gate fails on every broken fixture and passes the well-formed one")
    else:
        print("self-test: FAIL — the gate did not discriminate broken input from good input", file=sys.stderr)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", default=os.environ.get("ESHKOL_NS_PROOF_LEDGER", DEFAULT_LEDGER))
    parser.add_argument("--doc", default=DEFAULT_DOC)
    parser.add_argument("--cmake", default=DEFAULT_CMAKE)
    parser.add_argument("--oracles", default=DEFAULT_ORACLES)
    parser.add_argument("--examples-dir", default=DEFAULT_EXAMPLES_DIR)
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    try:
        ledger_data = _load_yaml_text(_read(args.ledger, "ledger"), "ledger")
        doc_text = _read(args.doc, "mechanization doc")
        cmake_text = _read(args.cmake, "CMakeLists.txt")
        oracles_data = _load_yaml_text(_read(args.oracles, "completion-oracles.yaml"), "completion-oracles.yaml")
        result = check(ledger_data, doc_text, cmake_text, oracles_data, args.examples_dir)
    except LedgerError as exc:
        snippet = f"ledger unusable: {exc}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "FAIL", snippet)
        if args.format == "json":
            print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: FAIL — {exc}", file=sys.stderr)
        return 1

    status = "PASS" if result["passed"] else "FAIL"
    if result["passed"]:
        snippet = (
            f"{result['row_count']} rows ({result['status_counts']}), "
            f"{result['exact_programs_verified']} EXACT program references verified exact-rational"
        )
    else:
        snippet = f"{len(result['errors'])} consistency error(s): " + "; ".join(result["errors"][:5])

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  ledger        : {args.ledger}")
        print(f"  rows          : {result['row_count']} (doc declares {result['doc_row_count']})")
        for s, count in result["status_counts"].items():
            print(f"  {s:<14}: {count}")
        print(f"  oracle wiring : {result['oracle_detail']}")
        if result["errors"]:
            print("  ERRORS:")
            for error in result["errors"]:
                print(f"    - {error}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
