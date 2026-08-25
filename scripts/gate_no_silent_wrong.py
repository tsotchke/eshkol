#!/usr/bin/env python3
"""Release gate: no open, unwaived SILENT-WRONG flaw may ship.

Reads the silent-wrong flaw ledger (.icc/silent-wrong-ledger.yaml by default)
and emits one ICC runtime_event into scripts/icc_traces/ so the
`no_open_silent_wrong` criterion in .icc/completion-oracles.yaml can grade it.

A SILENT-WRONG flaw is one where the compiler, the VM or the runtime produces
a wrong value, a wrong derivative or a wrong memory outcome with no diagnostic
and a zero exit status.  The maintainer's release rule is that these block the
tag: a loud failure is honest, a silent one is a lie told to a consumer.

Grading
    PASS  every SILENT-WRONG entry is `status: closed`, or carries a `waiver`
          with an owner, a reason and an `expires` date still in the future.
          Entries in other buckets are reported but never gate.
    FAIL  any SILENT-WRONG entry is open and unwaived, any waiver has expired
          or is missing a required field, or the ledger is absent, unreadable,
          unparseable or schema-invalid.

The gate FAILS CLOSED on purpose.  A missing ledger is not evidence that no
silent-wrong defects exist, and a readiness score computed without the ledger
would be the exact class of false green this gate was added to prevent.

Entries whose bucket is IN-FLIGHT but which carry `silent_wrong_class: true`
are counted as SILENT-WRONG for gating: an open PR is not a fix, and the tag
decision is about what the cut contains, not about what is being written.

Usage
    python3 scripts/gate_no_silent_wrong.py
    python3 scripts/gate_no_silent_wrong.py --ledger path/to/ledger.yaml
    python3 scripts/gate_no_silent_wrong.py --format json
    python3 scripts/gate_no_silent_wrong.py --self-test

Exit status is 0 on PASS and 1 on FAIL, so the script also works as a plain
CI step without ICC.

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LEDGER = os.path.join(REPO_ROOT, ".icc", "silent-wrong-ledger.yaml")
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "silent_wrong_gate.jsonl"

EXPECTED_SCHEMA = "eshkol.silent_wrong_ledger.v1"
GATING_BUCKET = "SILENT-WRONG"
PROBE_ID = "no_open_silent_wrong"

# Terminal statuses — the silent-wrong property is gone.
#   closed   the defect was removed and re-measured at a named SHA
#   fixed    the defect was removed; evidence is a committed test rather than a SHA
#   guarded  the silent path was converted into a loud, catchable one
# All three answer the gate's only question ("can this still return a wrong
# answer with no diagnostic?") with no. They are NOT interchangeable for a
# reader: `guarded` in particular means the wrong VALUE is gone but the
# operation still cannot compute the right one, so the entry stays on the
# ledger with a different shape. `fixed` and `guarded` were introduced by the
# closure PRs merged onto master (#423, #427); this gate learns them rather
# than forcing those entries back into a vocabulary that does not fit.
TERMINAL_STATUSES = ("closed", "fixed", "guarded")

WAIVER_REQUIRED_FIELDS = ("owner", "reason", "expires")


class LedgerError(Exception):
    """The ledger could not be read, parsed or validated."""


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LedgerError(
            "PyYAML is required to grade the silent-wrong ledger "
            "(pip install pyyaml)"
        ) from exc

    if not os.path.isfile(path):
        raise LedgerError(f"ledger not found at {path} (the gate fails closed)")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:
        raise LedgerError(f"ledger at {path} is not parseable: {exc}") from exc

    if not isinstance(data, dict):
        raise LedgerError(f"ledger at {path} is not a mapping")
    if data.get("schema") != EXPECTED_SCHEMA:
        raise LedgerError(
            f"ledger schema is {data.get('schema')!r}, expected {EXPECTED_SCHEMA!r}"
        )
    if not isinstance(data.get("entries"), list):
        raise LedgerError("ledger has no `entries` list")
    return data


def _as_date(value) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value).strip())


def _gates(entry: dict) -> bool:
    """Does this entry participate in the silent-wrong gate?"""
    if entry.get("bucket") == GATING_BUCKET:
        return True
    return bool(entry.get("silent_wrong_class"))


def grade(data: dict, today: _dt.date) -> dict:
    blocking: list[dict] = []
    waived: list[dict] = []
    closed: list[dict] = []
    invalid: list[dict] = []
    by_bucket: dict[str, int] = {}

    for raw in data["entries"]:
        if not isinstance(raw, dict):
            invalid.append({"id": "<non-mapping entry>", "why": "entry is not a mapping"})
            continue

        entry_id = raw.get("id", "<missing id>")
        bucket = raw.get("bucket", "<missing bucket>")
        by_bucket[bucket] = by_bucket.get(bucket, 0) + 1

        if not _gates(raw):
            continue

        record = {"id": entry_id, "title": raw.get("title", ""), "bucket": bucket}

        status = raw.get("status")
        if status in TERMINAL_STATUSES:
            # Nothing closes by bare assertion. A terminal entry must carry
            # either a re-measurement SHA or concrete evidence (a committed
            # test that would fail if the defect came back). Entries closed
            # without a SHA are accepted but reported, so the weaker form
            # stays visible instead of quietly becoming the norm.
            if not raw.get("closed_at") and not raw.get("evidence"):
                invalid.append({
                    **record,
                    "why": f"status {status} with neither closed_at nor evidence",
                })
            else:
                entry = {**record, "status": status}
                if not raw.get("closed_at"):
                    entry["no_sha"] = True
                closed.append(entry)
            continue
        if status != "open":
            invalid.append({
                **record,
                "why": f"status must be open or one of {'/'.join(TERMINAL_STATUSES)}, got {status!r}",
            })
            continue

        waiver = raw.get("waiver")
        if not waiver:
            blocking.append(record)
            continue
        if not isinstance(waiver, dict):
            invalid.append({**record, "why": "waiver is not a mapping"})
            continue

        missing = [f for f in WAIVER_REQUIRED_FIELDS if not waiver.get(f)]
        if missing:
            invalid.append({**record, "why": f"waiver missing {', '.join(missing)}"})
            continue
        try:
            expires = _as_date(waiver["expires"])
        except Exception:
            invalid.append({**record, "why": f"waiver expires is not an ISO date: {waiver['expires']!r}"})
            continue
        if expires < today:
            invalid.append({**record, "why": f"waiver expired on {expires.isoformat()}"})
            continue

        waived.append({**record, "owner": waiver["owner"], "expires": expires.isoformat()})

    passed = not blocking and not invalid
    return {
        "passed": passed,
        "blocking": blocking,
        "waived": waived,
        "closed": closed,
        "invalid": invalid,
        "by_bucket": by_bucket,
        "measured_at_sha": data.get("measured_at_sha"),
        "generated_at": str(data.get("generated_at", "")),
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
    # Rewrite rather than append: a stale FAIL from an earlier run must not
    # keep the failure_free invariant red after the ledger is cleaned up, and
    # a stale PASS must never survive a ledger that has regressed.
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


# ───────────────────────────── self-test ─────────────────────────────
#
# "A gate that cannot fail is not a gate." Each fixture feeds grade() (or the
# YAML loader) deliberately-broken input and asserts a FAIL; one well-formed
# fixture asserts the gate does not fail on everything indiscriminately.

_MALFORMED_YAML = """
schema: eshkol.silent_wrong_ledger.v1
entries:
  - id: SW-X
    bucket: SILENT-WRONG
      status: open
"""

_WRONG_SCHEMA_LEDGER = """
schema: eshkol.some_other_ledger.v1
entries: []
"""

_OPEN_UNWAIVED_LEDGER = """
schema: eshkol.silent_wrong_ledger.v1
entries:
  - id: SW-SELFTEST-OPEN
    bucket: SILENT-WRONG
    status: open
    title: "open with no waiver at all"
"""

_EXPIRED_WAIVER_LEDGER = """
schema: eshkol.silent_wrong_ledger.v1
entries:
  - id: SW-SELFTEST-EXPIRED
    bucket: SILENT-WRONG
    status: open
    title: "waiver expired last year"
    waiver:
      owner: nobody
      reason: "self-test fixture"
      expires: "2020-01-01"
"""

_GOOD_LEDGER = """
schema: eshkol.silent_wrong_ledger.v1
entries:
  - id: SW-SELFTEST-CLOSED
    bucket: SILENT-WRONG
    status: closed
    title: "closed with a re-measurement SHA"
    closed_at: "deadbeef"
  - id: SW-SELFTEST-WAIVED
    bucket: SILENT-WRONG
    status: open
    title: "open with an unexpired waiver"
    waiver:
      owner: someone
      reason: "self-test fixture"
      expires: "2999-01-01"
"""


def _run_fixture(name: str, text: str, tmp_dir: str) -> tuple[bool, str]:
    path = os.path.join(tmp_dir, name + ".yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    try:
        data = _load_yaml(path)
    except LedgerError as exc:
        return False, f"ledger error (expected for a malformed/wrong-schema fixture): {exc}"
    result = grade(data, _dt.date.today())
    ids = ", ".join(e["id"] for e in result["blocking"]) or "none"
    bad = ", ".join(f"{e['id']}:{e['why']}" for e in result["invalid"]) or "none"
    return result["passed"], f"blocking=[{ids}] invalid=[{bad}]"


def self_test() -> bool:
    cases = [
        ("malformed_yaml", _MALFORMED_YAML, False),
        ("wrong_schema", _WRONG_SCHEMA_LEDGER, False),
        ("open_unwaived", _OPEN_UNWAIVED_LEDGER, False),
        ("expired_waiver", _EXPIRED_WAIVER_LEDGER, False),
        ("well_formed", _GOOD_LEDGER, True),
    ]

    all_ok = True
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-no-silent-wrong-") as tmp_dir:
        print("gate_no_silent_wrong.py self-test:")
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
        result = grade(data, _dt.date.today())
    except LedgerError as exc:
        snippet = f"silent-wrong ledger unusable: {exc}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "FAIL", snippet)
        if args.format == "json":
            print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        else:
            print(f"no_open_silent_wrong: FAIL — {exc}", file=sys.stderr)
        return 1

    status = "PASS" if result["passed"] else "FAIL"
    no_sha = [e for e in result["closed"] if e.get("no_sha")]
    if result["passed"]:
        snippet = (
            f"no open unwaived SILENT-WRONG entries "
            f"({len(result['closed'])} closed, {len(result['waived'])} waived, "
            f"{len(no_sha)} closed without a re-measurement SHA) "
            f"at {result['measured_at_sha']}"
        )
    else:
        ids = ", ".join(e["id"] for e in result["blocking"]) or "none"
        bad = ", ".join(f"{e['id']}:{e['why']}" for e in result["invalid"]) or "none"
        snippet = (
            f"{len(result['blocking'])} open unwaived SILENT-WRONG entries [{ids}]; "
            f"{len(result['invalid'])} invalid [{bad}]"
        )

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
    else:
        print(f"no_open_silent_wrong: {status}")
        print(f"  ledger      : {args.ledger}")
        print(f"  measured at : {result['measured_at_sha']}")
        for bucket, count in sorted(result["by_bucket"].items()):
            print(f"  {bucket:<16}: {count}")
        if result["blocking"]:
            print("  BLOCKING (open, unwaived SILENT-WRONG):")
            for entry in result["blocking"]:
                print(f"    - {entry['id']}  {entry['title']}")
        if result["invalid"]:
            print("  INVALID (schema or waiver problem — treated as blocking):")
            for entry in result["invalid"]:
                print(f"    - {entry['id']}  {entry['why']}")
        if result["waived"]:
            print("  WAIVED (expiring):")
            for entry in result["waived"]:
                print(f"    - {entry['id']}  owner={entry['owner']} expires={entry['expires']}")
        if no_sha:
            print("  CLOSED WITHOUT A RE-MEASUREMENT SHA (accepted on evidence; not blocking):")
            for entry in no_sha:
                print(f"    - {entry['id']}  status={entry['status']}  {entry['title'][:70]}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
