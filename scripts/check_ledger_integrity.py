#!/usr/bin/env python3
"""Release gate: the silent-wrong flaw ledger is a well-formed ledger.

`.icc/silent-wrong-ledger.yaml` is hand-edited by every branch that opens or
closes a flaw, and it is merged like any other text file: two branches that
each add an entry near the same place produce a textual merge, not a content
merge, and git happily accepts it even when the result is unusable as a
ledger. This gate is the check that a bare `git merge` cannot perform.

It answers three questions a YAML parser alone does not:

  1. Does the file parse at all? A stray `- id:` at the wrong indent, an
     unclosed quote, or a duplicated block key breaks every consumer
     downstream (gate_no_silent_wrong.py, the completion-oracle criteria that
     read its trace) with no signal beyond a stack trace nobody was watching
     for.

  2. Does every id appear exactly once ACROSS THE WHOLE LEDGER? A textual
     merge does not conflict when two branches each independently pick the
     same next-free id — both edits "apply" cleanly and the file still
     parses. The result is two entries answering to one name: a lookup finds
     whichever one comes first, edits meant for one land on the other, and a
     human auditing "is SW-42 closed" gets an arbitrary answer. This is not
     hypothetical: SW-33 was independently allocated three times, SW-35 twice
     and SW-42 twice, on stacked branches that never conflicted with each
     other. The ledger's own `renumbered_from` fields record how each
     collision was eventually resolved by hand; this gate exists so the next
     one is caught before merge instead of after.

  3. Does every entry carry the minimum shape a reader (or gate_no_silent_
     wrong.py) needs to make sense of it: an `id`, a `bucket` that is one of
     the buckets this file itself declares, a `status` from the vocabulary
     the gate understands, a `title`, and — for any entry whose status is not
     `open` — at least one closure-evidence field naming what closed it
     (`closed_at`, `evidence`, `fixed_by`, `resolution` or `guarded_by`).
     "Believed fixed" with nothing to point at is not evidence; the ledger's
     own header states this as INVARIANT 1 and 3.

Grading
    PASS  the file parses, has no duplicate id anywhere in `entries`, and
          every entry satisfies the minimum shape above.
    FAIL  a parse error, any duplicate id, or any entry missing a required
          field / carrying a bucket or status this gate does not recognise /
          closed-like with no closure evidence.

The gate FAILS CLOSED: a missing or unparseable ledger is FAIL, never a silent
pass, for the same reason gate_no_silent_wrong.py fails closed — absence of
the ledger is not evidence of an absence of defects in it.

Usage
    python3 scripts/check_ledger_integrity.py
    python3 scripts/check_ledger_integrity.py --ledger path/to/ledger.yaml
    python3 scripts/check_ledger_integrity.py --format json
    python3 scripts/check_ledger_integrity.py --self-test

Exit status is 0 on PASS and 1 on FAIL (including under --self-test, where it
reports whether the gate itself is capable of failing).

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LEDGER = os.path.join(REPO_ROOT, ".icc", "silent-wrong-ledger.yaml")
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "ledger_integrity_gate.jsonl"
PROBE_ID = "ledger_integrity_clean"

KNOWN_STATUSES = ("open", "closed", "fixed", "guarded")
TERMINAL_STATUSES = ("closed", "fixed", "guarded")
CLOSURE_EVIDENCE_FIELDS = ("closed_at", "evidence", "fixed_by", "resolution", "guarded_by")
REQUIRED_ENTRY_FIELDS = ("id", "bucket", "status", "title")


class LedgerIntegrityError(Exception):
    """The ledger could not be read or parsed at all."""


def _load_yaml(path: str) -> object:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LedgerIntegrityError(
            "PyYAML is required to check the ledger (pip install pyyaml)"
        ) from exc

    if not os.path.isfile(path):
        raise LedgerIntegrityError(f"ledger not found at {path} (the gate fails closed)")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        raise LedgerIntegrityError(f"ledger at {path} is not parseable: {exc}") from exc


def check(data: object) -> dict:
    """Validate a parsed ledger document. Never raises; returns a report."""

    errors: list[str] = []

    if not isinstance(data, dict):
        return {"passed": False, "errors": ["ledger document is not a mapping"],
                "entry_count": 0, "duplicate_ids": {}, "bucket_counts": {}}

    buckets = data.get("buckets")
    declared_buckets = set(buckets.keys()) if isinstance(buckets, dict) else set()
    if not declared_buckets:
        errors.append("ledger has no `buckets` mapping (or it is empty) — nothing to validate entry buckets against")

    entries = data.get("entries")
    if not isinstance(entries, list):
        errors.append("ledger has no `entries` list")
        entries = []

    seen_ids: dict[str, int] = {}
    duplicate_ids: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}

    for position, raw in enumerate(entries, start=1):
        if not isinstance(raw, dict):
            errors.append(f"entry #{position} is not a mapping")
            continue

        label = f"entry #{position}"
        entry_id = raw.get("id")
        if entry_id:
            label = f"entry #{position} ({entry_id!r})"
            key = str(entry_id)
            if key in seen_ids:
                duplicate_ids[key] = duplicate_ids.get(key, seen_ids[key]) + 1
            else:
                seen_ids[key] = 1

        missing = [f for f in REQUIRED_ENTRY_FIELDS if not raw.get(f)]
        if missing:
            errors.append(f"{label} is missing required field(s): {', '.join(missing)}")
            # Still keep validating what IS present below — a missing id
            # should not hide an unrelated missing-evidence problem.

        bucket = raw.get("bucket")
        if bucket:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            if declared_buckets and bucket not in declared_buckets:
                errors.append(f"{label} has bucket {bucket!r}, which is not declared in `buckets`")

        status = raw.get("status")
        if status is not None and status not in KNOWN_STATUSES:
            errors.append(
                f"{label} has status {status!r}, expected one of {'/'.join(KNOWN_STATUSES)}"
            )
        elif status in TERMINAL_STATUSES:
            if not any(raw.get(field) for field in CLOSURE_EVIDENCE_FIELDS):
                errors.append(
                    f"{label} has status {status!r} but no closure evidence "
                    f"({'/'.join(CLOSURE_EVIDENCE_FIELDS)})"
                )

    # Duplicate ids that only ever collide with themselves are recorded once
    # above (seen_ids counts the FIRST occurrence, duplicate_ids the rest);
    # normalise to "how many times each id appears" for reporting.
    occurrence_counts: dict[str, int] = {}
    for raw in entries:
        if isinstance(raw, dict) and raw.get("id"):
            key = str(raw["id"])
            occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
    true_duplicates = {k: v for k, v in occurrence_counts.items() if v > 1}
    for entry_id, count in sorted(true_duplicates.items()):
        errors.append(f"id {entry_id!r} is used by {count} entries (must be unique across the whole ledger)")

    passed = not errors
    return {
        "passed": passed,
        "errors": errors,
        "entry_count": len(entries),
        "duplicate_ids": true_duplicates,
        "bucket_counts": bucket_counts,
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
# content (a gate that always fails is exactly as useless as one that never
# does).

_GOOD_LEDGER = """
schema: eshkol.silent_wrong_ledger.v1
buckets:
  SILENT-WRONG: "wrong value, no diagnostic, exit 0"
entries:
  - id: SW-SELFTEST-01
    bucket: SILENT-WRONG
    status: closed
    title: "self-test entry, closed with evidence"
    closed_at: "deadbeef"
  - id: SW-SELFTEST-02
    bucket: SILENT-WRONG
    status: open
    title: "self-test entry, open"
"""

_MALFORMED_YAML = """
schema: eshkol.silent_wrong_ledger.v1
entries:
  - id: SW-X
    bucket: SILENT-WRONG
      status: open
    title: "broken indentation opens a block-mapping the parser cannot close"
"""

_DUPLICATE_ID_LEDGER = """
schema: eshkol.silent_wrong_ledger.v1
buckets:
  SILENT-WRONG: "wrong value, no diagnostic, exit 0"
entries:
  - id: SW-42
    bucket: SILENT-WRONG
    status: closed
    title: "first SW-42, allocated on one branch"
    closed_at: "aaaaaaaa"
  - id: SW-42
    bucket: SILENT-WRONG
    status: open
    title: "second SW-42, allocated independently on another branch"
"""

_MISSING_FIELD_LEDGER = """
schema: eshkol.silent_wrong_ledger.v1
buckets:
  SILENT-WRONG: "wrong value, no diagnostic, exit 0"
entries:
  - id: SW-INCOMPLETE
    status: open
    title: "entry with no bucket at all"
  - id: SW-UNEVIDENCED
    bucket: SILENT-WRONG
    status: closed
    title: "closed with nothing pointing at what closed it"
"""


def _run_fixture(name: str, text: str, tmp_dir: str) -> tuple[bool, str]:
    path = os.path.join(tmp_dir, name + ".yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    try:
        data = _load_yaml(path)
    except LedgerIntegrityError as exc:
        return False, f"parse error (expected for a malformed-YAML fixture): {exc}"
    result = check(data)
    return result["passed"], "; ".join(result["errors"]) or "no errors"


def self_test() -> bool:
    """Run the gate against fixtures with known-bad and known-good shape.

    Fixtures live in a temp directory INSIDE the repo (never /tmp) so the
    gate is exercised against real files on disk exactly as it runs in CI,
    and the directory is removed before this function returns.
    """

    cases = [
        ("malformed_yaml", _MALFORMED_YAML, False),
        ("duplicate_id", _DUPLICATE_ID_LEDGER, False),
        ("missing_field", _MISSING_FIELD_LEDGER, False),
        ("well_formed", _GOOD_LEDGER, True),
    ]

    all_ok = True
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-ledger-integrity-") as tmp_dir:
        print("check_ledger_integrity.py self-test:")
        for name, text, expect_pass in cases:
            passed, detail = _run_fixture(name, text, tmp_dir)
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
    parser.add_argument("--ledger", default=os.environ.get("ESHKOL_FLAW_LEDGER", DEFAULT_LEDGER))
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    try:
        data = _load_yaml(args.ledger)
        result = check(data)
    except LedgerIntegrityError as exc:
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
            f"{result['entry_count']} entries, no duplicate ids, "
            f"all required fields present across {len(result['bucket_counts'])} buckets"
        )
    else:
        snippet = f"{len(result['errors'])} integrity error(s): " + "; ".join(result["errors"][:5])

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  ledger      : {args.ledger}")
        print(f"  entries     : {result['entry_count']}")
        for bucket, count in sorted(result.get("bucket_counts", {}).items()):
            print(f"  {bucket:<24}: {count}")
        if result["duplicate_ids"]:
            print("  DUPLICATE IDS:")
            for entry_id, count in sorted(result["duplicate_ids"].items()):
                print(f"    - {entry_id}  used by {count} entries")
        if result["errors"]:
            print("  ERRORS:")
            for error in result["errors"]:
                print(f"    - {error}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
