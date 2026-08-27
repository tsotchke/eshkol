#!/usr/bin/env python3
"""Release gate: runtime-event evidence backing a `severity: high`
`.icc/completion-oracles.yaml` criterion must not be older than N days.

Motivating gap: ADR-0010 gap A13 ("GPU/quantum correctness produce no
hosted evidence; fail *open* to SKIP") names its own proposed fix --
"no evidence in N days = FAIL" -- and, as of the 2026-08-25 architectural
audit (finding F19), `.icc/completion-oracles.yaml` implemented it for
ZERO targets. The severity fix in this same changeset (F19) closes the
"zero evidence at all" half of A13: every criterion is now `severity:high`
(or an explicit ADVISORY), so a target with NO trace record for a
criterion's `event_kinds` correctly reads `blocked`. This script closes
the other half: a trace record that DOES exist but is old enough that it
no longer says anything true about the current tree.

Why this is a repo-side script and not a `max_age_days:` key in the oracle
YAML itself: ICC's own grader
(`completion_oracle_criteria._evaluate_criterion` /
`_status_for_evidence`, in the private ICC tool this repo does not vendor)
only consults a criterion's `max_age_days` for the `performance_budget`
kind -- `_status_for_evidence`, which grades `runtime_event` (the kind
every criterion in this file uses), never reads staleness at all, and
`runtime_event`'s own trace records do not carry a `timestamp` field
(see e.g. `check_oracle_schema.py`'s `emit_trace`, which never wrote
one). Declaring `max_age_days` on a `runtime_event` criterion in the YAML
would be silently ignored, which is a worse failure mode than not
declaring it -- it would read as a staleness rule while enforcing nothing.
So until ICC's `runtime_event` grading path gains the same `max_age_days`
support `performance_budget` already has, staleness is enforced HERE, as
its own repo-side gate over the trace corpus, with its own PASS/FAIL
verdict fed back into the oracle file as an ordinary `eshkol_smoke`
runtime_event (the same pattern `check_oracle_schema.py`,
`check_ledger_integrity.py` and `gate_no_silent_wrong.py` already use) so
`icc readiness` still sees the result even though it cannot compute it
itself. See the note filed for the ICC feedback channel at
~/.tsotchke/state/feedback/ for the upstream ask.

What "freshness" means for a trace record here, in order of preference:
    1. an explicit numeric `timestamp` field on the JSON record itself
       (epoch seconds) -- if a probe ever starts emitting one, this script
       already prefers it over (2) without any further change.
    2. the mtime of the `.jsonl` file the record was read from. Every
       probe in this repo (`run_icc_smoke.sh`, `run_stress.sh`, the gate
       scripts themselves) OVERWRITES or appends to its trace file each
       run, so file mtime is a faithful proxy for "when was this evidence
       last produced" even though the JSON payload itself is undated.

Default staleness window: 14 days (`--max-age-days`, env
`ESHKOL_EVIDENCE_MAX_AGE_DAYS`). Rationale: this repo cuts CI-visible
commits and merges daily-to-several-times-daily (see CHANGELOG.md), so a
14-day-old GPU/stress/AD trace was produced against a tree that is, on
this project's own cadence, many merges stale -- long enough to survive a
missed weekend or a single delayed nightly run, short enough that a trace
cannot silently outlive the feature it was measuring by a release cycle.
Override per-CI-lane or per-workstation with `--max-age-days` if a
slower-moving evidence class (e.g. a quarterly hosted-GPU run) needs a
wider window; that is a deliberate, visible choice, not a silent default.

Checks
    For every `severity: high` `runtime_event` criterion in every oracle
    target: find the newest matching trace record across `--trace-dir`
    (matched by `event_kinds`, and by `event_names`/`event_values` when the
    criterion declares them). If none exists, this gate has nothing to say
    -- that is the absence case the severity fix already handles. If one
    exists, compute its age; a criterion whose newest matching evidence is
    older than `--max-age-days` is STALE.

Grading (PASS / FAIL / NO_DATA, the same three-way vocabulary
`check_required_context_consistency.py` already uses)
    PASS      at least one `severity: high` criterion has matching evidence,
              and every criterion that has evidence is newer than the window.
    FAIL      at least one criterion's newest matching evidence has aged
              past the window.
    NO_DATA   zero `severity: high` criteria have ANY matching evidence at
              all -- nothing was actually graded. Exit 2. By default this is
              not a failure (a bare local invocation with no trace corpus
              yet has nothing to say -- the absence case the severity gate
              already handles), but it is also no longer indistinguishable
              from a real PASS: `--require-trace-dir` callers (a CI step
              asserting real evidence should exist here) get NO_DATA
              honestly rather than a silent green (B5/N2, 2026-08-26 audit
              -- a required assurance-gates step read PASS every run against
              a directory .gitignore guarantees is always empty, because the
              gate had no way to say anything other than PASS or FAIL for
              "nothing to check").

The gate FAILS CLOSED: a missing or unparseable oracle file is FAIL. A
missing OR EMPTY trace directory is NO_DATA (PASS by default, unless
`--require-trace-dir` is set, in which case it is a non-PASS NO_DATA,
exit 2).

Usage
    python3 scripts/check_evidence_staleness.py
    python3 scripts/check_evidence_staleness.py --max-age-days 7
    python3 scripts/check_evidence_staleness.py --trace-dir scripts/icc_traces
    python3 scripts/check_evidence_staleness.py --format json
    python3 scripts/check_evidence_staleness.py --self-test

Exit status is 0 on PASS and 1 on FAIL.

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ORACLES = os.path.join(REPO_ROOT, ".icc", "completion-oracles.yaml")
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "evidence_staleness_gate.jsonl"
PROBE_ID = "evidence_staleness_clean"
DEFAULT_MAX_AGE_DAYS = 14.0


class StalenessGateError(Exception):
    """The oracle file could not be read or parsed at all."""


def _load_yaml(path: str) -> object:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise StalenessGateError(
            "PyYAML is required to check evidence staleness (pip install pyyaml)"
        ) from exc

    if not os.path.isfile(path):
        raise StalenessGateError(f"oracle file not found at {path} (the gate fails closed)")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        raise StalenessGateError(f"oracle file at {path} is not parseable: {exc}") from exc


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _criteria_of(oracle: dict) -> list:
    raw = oracle.get("requires")
    if raw is None:
        raw = oracle.get("criteria")
    if not isinstance(raw, list):
        return []
    return [c for c in raw if isinstance(c, dict)]


def high_severity_runtime_event_criteria(data: dict) -> list[dict]:
    """Every `severity: high` `runtime_event` criterion across every target."""

    out: list[dict] = []
    for oracle in _as_list(data.get("oracles")):
        if not isinstance(oracle, dict):
            continue
        oracle_name = str(oracle.get("name") or oracle.get("target") or "<unnamed>")
        for criterion in _criteria_of(oracle):
            if criterion.get("severity") != "high":
                continue
            if "runtime_event" not in criterion and criterion.get("kind") != "runtime_event":
                continue
            raw = criterion.get("runtime_event")
            payload = raw if isinstance(raw, dict) else {}
            event_kinds = set(str(k) for k in _as_list(payload.get("event_kinds") or criterion.get("event_kinds")))
            if not event_kinds:
                continue
            out.append({
                "oracle": oracle_name,
                "label": str(criterion.get("label") or "<unlabeled>"),
                "event_kinds": event_kinds,
                "event_names": set(str(n) for n in _as_list(payload.get("event_names") or criterion.get("event_names"))),
                "event_values": set(str(v) for v in _as_list(payload.get("event_values") or criterion.get("event_values"))),
            })
    return out


def _load_trace_records(trace_dir: str) -> list[dict]:
    """Every JSON record from every `*.jsonl` file in `trace_dir`, each
    tagged with `_file` and `_file_mtime` for the file-mtime fallback."""

    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
        try:
            file_mtime = os.path.getmtime(path)
        except OSError:
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    record["_file"] = path
                    record["_file_mtime"] = file_mtime
                    records.append(record)
        except OSError:
            continue
    return records


def _record_age_days(record: dict, now: float) -> tuple[float, str]:
    """(age_days, source) -- source is "record_timestamp" or "file_mtime"."""

    stamp = record.get("timestamp")
    if isinstance(stamp, (int, float)):
        return max(0.0, (now - float(stamp)) / 86400.0), "record_timestamp"
    return max(0.0, (now - float(record["_file_mtime"])) / 86400.0), "file_mtime"


def _record_matches(record: dict, criterion: dict) -> bool:
    if str(record.get("kind")) not in criterion["event_kinds"]:
        return False
    if criterion["event_names"] and str(record.get("name")) not in criterion["event_names"]:
        return False
    if criterion["event_values"]:
        value = record.get("value")
        if str(value) not in criterion["event_values"]:
            return False
    return True


def check(
    data: object,
    trace_dir: str,
    max_age_days: float,
    now: float | None = None,
    require_trace_dir: bool = False,
) -> dict:
    """Never raises; returns a report."""

    now = time.time() if now is None else now

    if not isinstance(data, dict):
        return {"passed": False, "status": "FAIL", "errors": ["oracle document is not a mapping"], "criteria": []}

    criteria = high_severity_runtime_event_criteria(data)

    if not os.path.isdir(trace_dir):
        if require_trace_dir:
            return {
                "passed": False,
                "status": "NO_DATA",
                "errors": [f"trace directory {trace_dir} does not exist (--require-trace-dir set)"],
                "criteria": [],
            }
        return {
            "passed": True,
            "status": "NO_DATA",
            "errors": [],
            "criteria": [],
            "note": f"trace directory {trace_dir} does not exist -- nothing to check "
                    "(absence-of-evidence is the severity gate's job, not this one's)",
        }

    records = _load_trace_records(trace_dir)

    rows: list[dict] = []
    stale: list[dict] = []
    for criterion in criteria:
        matches = [r for r in records if _record_matches(r, criterion)]
        if not matches:
            continue  # no evidence at all: out of scope for staleness
        ages = [(_record_age_days(r, now), r) for r in matches]
        (best_age, source), newest_record = min(ages, key=lambda pair: pair[0][0])
        row = {
            "oracle": criterion["oracle"],
            "label": criterion["label"],
            "age_days": round(best_age, 2),
            "age_source": source,
            "trace_file": newest_record.get("_file"),
            "is_stale": best_age > max_age_days,
        }
        rows.append(row)
        if row["is_stale"]:
            stale.append(row)

    errors = [
        f"{row['oracle']!r} / {row['label']!r}: newest matching evidence is "
        f"{row['age_days']:.1f} days old (source: {row['age_source']}, "
        f"{row['trace_file']}), past the {max_age_days:g}-day window"
        for row in stale
    ]

    # Three-way verdict (B5/N2, 2026-08-26 audit). A trace directory that
    # EXISTS but holds zero records matching any severity:high criterion
    # (the shape a CI job with no build/run step in it — e.g. a pure-Python
    # assurance job — structurally always produces) used to read PASS
    # identically to a directory holding real, fresh evidence: "0 criteria
    # checked, 0 stale" and "12 criteria checked, 0 stale" both had
    # `errors == []`. That let a required CI step go green forever without
    # ever having graded anything. NO_DATA is now distinct from PASS: it
    # still does not FAIL by default (missing evidence is the severity
    # gate's job — see the docstring), but `--require-trace-dir` callers
    # (a CI step asserting "this directory is supposed to have evidence in
    # it") get an honest non-PASS verdict instead of a silent one, exactly
    # the `required_context_consistency_clean` PASS/FAIL/NO_DATA precedent
    # `scripts/check_required_context_consistency.py` already established.
    if stale:
        status = "FAIL"
        passed = False
    elif not rows:
        status = "NO_DATA"
        passed = not require_trace_dir
        if require_trace_dir:
            errors = [
                f"trace directory {trace_dir} exists but contains no evidence matching "
                "any severity:high criterion (--require-trace-dir set): nothing was graded"
            ]
    else:
        status = "PASS"
        passed = True

    return {
        "passed": passed,
        "status": status,
        "errors": errors,
        "criteria": rows,
        "max_age_days": max_age_days,
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

_SELFTEST_ORACLES = """
oracles:
  - name: selftest-staleness
    requires:
      - runtime_event:
          event_kinds: [selftest_kind]
          event_names: ["selftest_probe"]
          event_values: ["PASS"]
        severity: high
        label: self-test staleness probe
        action: ./scripts/run_selftest.sh
"""


def _write_trace(trace_dir: str, filename: str, record: dict, mtime: float | None = None) -> None:
    path = os.path.join(trace_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def self_test() -> bool:
    now = time.time()
    cases: list[tuple[str, bool]] = []

    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-evidence-staleness-") as tmp_dir:
        oracle_path = os.path.join(tmp_dir, "oracles.yaml")
        with open(oracle_path, "w", encoding="utf-8") as handle:
            handle.write(_SELFTEST_ORACLES)
        data = _load_yaml(oracle_path)

        # Case 1: fresh trace (mtime = now) -> PASS.
        fresh_dir = os.path.join(tmp_dir, "fresh")
        os.makedirs(fresh_dir)
        _write_trace(
            fresh_dir, "selftest.jsonl",
            {"kind": "selftest_kind", "name": "selftest_probe", "value": "PASS"},
            mtime=now,
        )
        result = check(data, fresh_dir, max_age_days=14.0, now=now)
        cases.append(("fresh_trace_passes", result["passed"] is True))

        # Case 2: backdated trace (mtime = 20 days ago, past the 14-day
        # default window) -> FAIL. This is the actual "prove it can go RED"
        # requirement: a trace file whose mtime is deliberately set into the
        # past via os.utime, exactly reproducing what a stale GPU/stress
        # trace nobody re-ran in three weeks looks like on disk.
        stale_dir = os.path.join(tmp_dir, "stale")
        os.makedirs(stale_dir)
        backdated = now - 20 * 86400.0
        _write_trace(
            stale_dir, "selftest.jsonl",
            {"kind": "selftest_kind", "name": "selftest_probe", "value": "PASS"},
            mtime=backdated,
        )
        result = check(data, stale_dir, max_age_days=14.0, now=now)
        cases.append(("backdated_trace_fails", result["passed"] is False and len(result["errors"]) == 1))

        # Case 3: explicit record timestamp overrides file mtime, and a
        # stale explicit timestamp is caught even with a fresh file mtime.
        ts_dir = os.path.join(tmp_dir, "explicit_timestamp")
        os.makedirs(ts_dir)
        _write_trace(
            ts_dir, "selftest.jsonl",
            {"kind": "selftest_kind", "name": "selftest_probe", "value": "PASS", "timestamp": backdated},
            mtime=now,
        )
        result = check(data, ts_dir, max_age_days=14.0, now=now)
        cases.append(("explicit_stale_timestamp_beats_fresh_mtime", result["passed"] is False))

        # Case 4: no matching evidence at all, --require-trace-dir NOT set
        # -> PASS/NO_DATA (absence is the severity gate's job, not this
        # one's, for an ordinary/local invocation).
        empty_dir = os.path.join(tmp_dir, "empty")
        os.makedirs(empty_dir)
        result = check(data, empty_dir, max_age_days=14.0, now=now)
        cases.append(("no_evidence_is_not_this_gates_job", result["passed"] is True and not result["criteria"]
                      and result["status"] == "NO_DATA"))

        # Case 5: missing trace dir -> PASS unless --require-trace-dir.
        missing_dir = os.path.join(tmp_dir, "does-not-exist")
        result = check(data, missing_dir, max_age_days=14.0, now=now)
        cases.append(("missing_trace_dir_passes_by_default", result["passed"] is True and result["status"] == "NO_DATA"))
        result = check(data, missing_dir, max_age_days=14.0, now=now, require_trace_dir=True)
        cases.append(("missing_trace_dir_fails_when_required", result["passed"] is False and result["status"] == "NO_DATA"))

        # Case 6 (B5/N2, 2026-08-26 audit): a trace directory that EXISTS but
        # holds zero records matching any severity:high criterion is the
        # SHAPE the assurance-gates CI job produced every single run (a
        # gitignored, always-empty scripts/icc_traces/) -- and the gate used
        # to read that identically to genuine "checked N criteria, none
        # stale" PASS. With --require-trace-dir this must now be a
        # distinguishable, non-silent NO_DATA verdict, not PASS.
        existing_empty_dir = os.path.join(tmp_dir, "existing_empty")
        os.makedirs(existing_empty_dir)
        result = check(data, existing_empty_dir, max_age_days=14.0, now=now, require_trace_dir=True)
        cases.append(("existing_but_empty_trace_dir_is_no_data_when_required",
                      result["passed"] is False and result["status"] == "NO_DATA"))

        # Case 7: the SAME existing-but-empty directory populated with one
        # real, fresh record -> a genuine PASS (status flips from NO_DATA to
        # PASS the moment there is something to actually grade).
        _write_trace(
            existing_empty_dir, "selftest.jsonl",
            {"kind": "selftest_kind", "name": "selftest_probe", "value": "PASS"},
            mtime=now,
        )
        result = check(data, existing_empty_dir, max_age_days=14.0, now=now, require_trace_dir=True)
        cases.append(("populated_dir_is_a_real_pass_when_required",
                      result["passed"] is True and result["status"] == "PASS" and len(result["criteria"]) == 1))

    all_ok = True
    print("check_evidence_staleness.py self-test:")
    for name, ok in cases:
        all_ok = all_ok and ok
        verdict = "OK" if ok else "GATE IS BROKEN"
        print(f"  [{verdict}] {name}")

    if all_ok:
        print("self-test: PASS -- the gate goes RED on backdated evidence and does not "
              "false-positive on fresh, absent, or missing-directory cases")
    else:
        print("self-test: FAIL -- the staleness gate did not discriminate correctly", file=sys.stderr)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oracles", default=os.environ.get("ESHKOL_ORACLE_FILE", DEFAULT_ORACLES))
    parser.add_argument("--trace-dir", default=os.environ.get("ESHKOL_TRACE_DIR", DEFAULT_TRACE_DIR))
    parser.add_argument(
        "--max-age-days", type=float,
        default=float(os.environ.get("ESHKOL_EVIDENCE_MAX_AGE_DAYS", DEFAULT_MAX_AGE_DAYS)),
    )
    parser.add_argument("--require-trace-dir", action="store_true")
    parser.add_argument("--emit-trace-dir", default=DEFAULT_TRACE_DIR,
                         help="where this gate's own PASS/FAIL trace is written (default: --trace-dir's usual home)")
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    try:
        data = _load_yaml(args.oracles)
    except StalenessGateError as exc:
        snippet = f"oracle file unusable: {exc}"
        if not args.no_trace:
            emit_trace(args.emit_trace_dir, "FAIL", snippet)
        if args.format == "json":
            print(json.dumps({"status": "FAIL", "passed": False, "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: FAIL -- {exc}", file=sys.stderr)
        return 1

    result = check(data, args.trace_dir, args.max_age_days, require_trace_dir=args.require_trace_dir)
    status = result["status"]
    if status == "PASS":
        checked = len(result.get("criteria", []))
        snippet = result.get("note") or f"{checked} criteria had matching evidence, none past {args.max_age_days:g} days"
    elif status == "NO_DATA":
        snippet = result.get("note") or (
            result["errors"][0] if result["errors"] else "no severity:high criterion has any matching evidence"
        )
    else:
        snippet = f"{len(result['errors'])} stale criterion/criteria: " + "; ".join(result["errors"][:5])

    if not args.no_trace:
        emit_trace(args.emit_trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  oracle file    : {args.oracles}")
        print(f"  trace dir      : {args.trace_dir}")
        print(f"  max age (days) : {args.max_age_days:g}")
        if result.get("note"):
            print(f"  note           : {result['note']}")
        if result.get("criteria"):
            print(f"  criteria with evidence : {len(result['criteria'])}")
            for row in result["criteria"]:
                marker = "  <-- STALE" if row["is_stale"] else ""
                print(f"    - {row['oracle']:<28} {row['age_days']:>7.2f}d ({row['age_source']}){marker}")
        if result["errors"]:
            print("  ERRORS:")
            for error in result["errors"]:
                print(f"    - {error}")
        if status == "NO_DATA":
            print("  NO_DATA is not a pass: nothing was graded.", file=sys.stderr)

    if status == "FAIL":
        return 1
    if status == "NO_DATA" and not result["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
