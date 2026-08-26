#!/usr/bin/env python3
"""Release gate: no `.icc/completion-oracles.yaml` target can read `ready`
(or any non-`blocked` status) while every one of its criteria has zero
evidence.

Motivating incident: the 2026-08-25 architectural audit (F19) measured
`icc readiness --target gpu-execution` returning `Status: ready  Score: 94`
on a host that had never run a single GPU kernel and a trace directory with
no `gpu_execution` record anywhere in it. The mechanism is ICC's own,
unmodified and correct in isolation: `completion_oracle_criteria.py`'s
`_status_for_missing(severity)` returns `FAIL` only when a criterion's
declared `severity` is `high`; `medium` and `low` degrade a zero-evidence
miss to `WARN`. `readiness_service.py`'s status rule then only forces
`blocked` when at least one oracle criterion is `FAIL` (or a `high`-severity
graph gap exists, or a runtime check FAILed) --  a target whose criteria are
ALL `medium`/`low` can clear `score >= 90` on pure absence and read `ready`.

That is not a bug in ICC's grader -- it is a bug in *this repo's* severity
declarations. Proof: the four Ozaki targets in the same file, scored on the
same host with the same absent evidence, correctly return `blocked` because
every one of their criteria is `severity: high`. `gpu-execution` was the
only oracle target in the file with a `requires` list that has *zero*
`high`-severity entries.

This gate re-derives ICC's own status arithmetic (`_status_for_missing`,
and the `blocked`/`ready`/`incomplete` rule in `readiness_service.py`'s
`build_readiness_payload`) locally, in this repo, over
`.icc/completion-oracles.yaml`, and asks one question per target: "if
every criterion in this target had zero evidence right now, would ICC's
status rule still call it `blocked`?" A target is a FALSE-GREEN RISK when
the answer is no -- i.e. the target has no `high`-severity criterion, so
total absence of evidence cannot force `blocked` and the target is free to
read `ready` (if the approximated score clears 90) or `incomplete`
(if it does not, which is still not the `blocked` a correctness gate with
zero evidence should read).

The predicted score here is an APPROXIMATION: it replays only the
oracle-criteria terms of ICC's score formula (`-8` per criterion `FAIL`,
`-3` per criterion `WARN`), not the cross-cutting graph-gap / liveness /
stubbed-path / runtime-check terms that also feed a real `icc readiness`
run and vary repo-registration to repo-registration (measured: the F19
audit's registration read `gpu-execution` at score 94, a fresh registration
of the same file in this script's own worktree read it at 90 -- both
`ready`, same defect, different incidental score). The STATUS classification
(would zero evidence force `blocked`?) does not depend on those terms and is
exact; only the printed score is a lower-bound estimate, and is labelled as
such.

Checks
    For every oracle target in the file:
      - criteria count, severity histogram
      - predicted oracle-criteria-only score and status if EVERY criterion
        in the target had zero evidence right now
      - `false_green_risk`: true iff that predicted status is not `blocked`
        (i.e. the target has no `high`-severity criterion, so it cannot be
        forced blocked purely by absent evidence)

Grading
    PASS  no target is a false_green_risk (every target has at least one
          `high`-severity criterion, so total absence of evidence forces
          `blocked` for every target in the file).
    FAIL  one or more targets have zero `high`-severity criteria.

The gate FAILS CLOSED: a missing or unparseable oracle file is FAIL.

Usage
    python3 scripts/audit_oracle_false_green.py
    python3 scripts/audit_oracle_false_green.py --oracles path/to/file.yaml
    python3 scripts/audit_oracle_false_green.py --format json
    python3 scripts/audit_oracle_false_green.py --self-test

Exit status is 0 on PASS and 1 on FAIL.

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ORACLES = os.path.join(REPO_ROOT, ".icc", "completion-oracles.yaml")
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "oracle_false_green_gate.jsonl"
PROBE_ID = "oracle_false_green_clean"

VALID_SEVERITIES = ("high", "medium", "low")

# Mirrors completion_oracle_criteria._status_for_missing exactly: FAIL only
# for `high`; everything else (including a missing/invalid severity, which
# ICC's own normalizer defaults to `medium`) degrades to WARN.
def _status_for_missing(severity: str) -> str:
    return "FAIL" if severity == "high" else "WARN"


class OracleAuditError(Exception):
    """The oracle file could not be read or parsed at all."""


def _load_yaml(path: str) -> object:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise OracleAuditError(
            "PyYAML is required to audit the oracle file (pip install pyyaml)"
        ) from exc

    if not os.path.isfile(path):
        raise OracleAuditError(f"oracle file not found at {path} (the gate fails closed)")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        raise OracleAuditError(f"oracle file at {path} is not parseable: {exc}") from exc


def _criteria_of(oracle: dict) -> list:
    raw = oracle.get("requires")
    if raw is None:
        raw = oracle.get("criteria")
    if not isinstance(raw, list):
        return []
    return [c for c in raw if isinstance(c, dict)]


def _severity_of(criterion: dict) -> str:
    severity = criterion.get("severity")
    return severity if severity in VALID_SEVERITIES else "medium"


def _label_of(criterion: dict) -> str:
    return str(criterion.get("label") or criterion.get("id") or "<unlabeled criterion>")


def audit_target(oracle: dict) -> dict:
    """Predict the target's status/score if every criterion had zero evidence."""

    name = str(oracle.get("name") or oracle.get("target") or "<unnamed>")
    criteria = _criteria_of(oracle)

    histogram = {"high": 0, "medium": 0, "low": 0}
    non_high: list[dict] = []
    for criterion in criteria:
        severity = _severity_of(criterion)
        histogram[severity] += 1
        if severity != "high":
            non_high.append({
                "label": _label_of(criterion),
                "severity": severity,
                "kind": criterion.get("kind") or next(
                    (k for k in criterion if k not in
                     ("id", "label", "severity", "action", "recommended_action")),
                    "unknown",
                ),
            })

    fail_count = histogram["high"]
    warn_count = histogram["medium"] + histogram["low"]
    predicted_score = max(0, 100 - 8 * fail_count - 3 * warn_count)
    if fail_count > 0:
        predicted_status = "blocked"
    elif predicted_score >= 90:
        predicted_status = "ready"
    else:
        predicted_status = "incomplete"

    false_green_risk = predicted_status != "blocked"

    return {
        "name": name,
        "criteria_count": len(criteria),
        "severity_histogram": histogram,
        "predicted_score_on_zero_evidence": predicted_score,
        "predicted_status_on_zero_evidence": predicted_status,
        "false_green_risk": false_green_risk,
        "non_high_criteria": non_high,
    }


def audit(data: object) -> dict:
    """Audit a parsed oracle document. Never raises; returns a report."""

    if not isinstance(data, dict):
        return {"passed": False, "errors": ["oracle document is not a mapping"], "targets": []}

    oracles_raw = data.get("oracles")
    if not isinstance(oracles_raw, list):
        return {"passed": False, "errors": ["oracle document has no top-level `oracles` list"], "targets": []}

    targets = [audit_target(o) for o in oracles_raw if isinstance(o, dict)]
    risky = [t for t in targets if t["false_green_risk"]]
    passed = not risky
    errors = [
        f"target {t['name']!r} has zero `severity: high` criteria "
        f"({t['criteria_count']} criteria, all medium/low) -- it predicts "
        f"`{t['predicted_status_on_zero_evidence']}` on TOTAL ABSENCE of evidence"
        for t in risky
    ]
    return {"passed": passed, "errors": errors, "targets": targets, "risky_targets": [t["name"] for t in risky]}


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

# Reproduces the F19 shape exactly: one criterion, severity medium, nothing
# else in the target -- must be flagged as a false-green risk.
_RISKY_ORACLES = """
oracles:
  - name: selftest-gpu-like
    requires:
      - runtime_event:
          event_kinds: [gpu_execution]
          event_names: ["gpu_execution_gate"]
          event_values: ["PASS"]
        severity: medium
        label: GPU tensor ops match the CPU reference within tolerance
        action: ./tests/gpu/gpu_correctness_gate.sh
"""

# The fixed shape: same criterion, severity high -- must NOT be flagged.
_FIXED_ORACLES = """
oracles:
  - name: selftest-gpu-like-fixed
    requires:
      - runtime_event:
          event_kinds: [gpu_execution]
          event_names: ["gpu_execution_gate"]
          event_values: ["PASS"]
        severity: high
        label: GPU tensor ops match the CPU reference within tolerance
        action: ./tests/gpu/gpu_correctness_gate.sh
"""

# A multi-criterion target where every criterion is medium/low -- also a
# risk, and score-wise would predict `incomplete` rather than `ready` at 4+
# criteria, but `incomplete` is still not `blocked` so it must still flag.
_RISKY_MULTI_ORACLES = """
oracles:
  - name: selftest-multi-medium
    requires:
      - test_evidence: true
        severity: medium
        label: fixture A indexed
        action: ctest -R a
      - test_evidence: true
        severity: medium
        label: fixture B indexed
        action: ctest -R b
      - test_evidence: true
        severity: low
        label: fixture C indexed
        action: ctest -R c
      - test_evidence: true
        severity: low
        label: fixture D indexed
        action: ctest -R d
"""

# A target with a mix, at least one high -- must NOT be flagged, since one
# high-severity miss already forces `blocked` on zero evidence.
_MIXED_SAFE_ORACLES = """
oracles:
  - name: selftest-mixed-safe
    requires:
      - runtime_event:
          event_kinds: [eshkol_smoke]
        severity: high
        label: the load-bearing correctness claim
        action: ./scripts/run_selftest.sh
      - test_evidence: true
        severity: medium
        label: an explicitly advisory bookkeeping criterion
        action: ctest -R advisory
"""


def _run_fixture(name: str, text: str, tmp_dir: str) -> dict:
    path = os.path.join(tmp_dir, name + ".yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    data = _load_yaml(path)
    return audit(data)


def self_test() -> bool:
    cases = [
        ("risky_single_medium (reproduces F19's gpu-execution shape)", _RISKY_ORACLES, False),
        ("fixed_single_high", _FIXED_ORACLES, True),
        ("risky_multi_medium_low", _RISKY_MULTI_ORACLES, False),
        ("mixed_one_high_is_safe", _MIXED_SAFE_ORACLES, True),
    ]

    all_ok = True
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-oracle-false-green-") as tmp_dir:
        print("audit_oracle_false_green.py self-test:")
        for name, text, expect_passed in cases:
            result = _run_fixture(name, text, tmp_dir)
            ok = result["passed"] == expect_passed
            all_ok = all_ok and ok
            verdict = "OK" if ok else "GATE IS BROKEN"
            print(f"  [{verdict}] {name}: expected passed={expect_passed}, got passed={result['passed']}")
            for t in result["targets"]:
                print(
                    f"           {t['name']}: {t['criteria_count']} criteria "
                    f"{t['severity_histogram']} -> predicted "
                    f"{t['predicted_status_on_zero_evidence']}/{t['predicted_score_on_zero_evidence']} "
                    f"false_green_risk={t['false_green_risk']}"
                )

    if all_ok:
        print("self-test: PASS -- the detector flags every zero-high-severity target "
              "and clears every target with at least one high-severity criterion")
    else:
        print("self-test: FAIL -- the detector did not discriminate risky input from safe input", file=sys.stderr)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oracles", default=os.environ.get("ESHKOL_ORACLE_FILE", DEFAULT_ORACLES))
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    try:
        data = _load_yaml(args.oracles)
    except OracleAuditError as exc:
        snippet = f"oracle file unusable: {exc}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "FAIL", snippet)
        if args.format == "json":
            print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: FAIL -- {exc}", file=sys.stderr)
        return 1

    result = audit(data)
    status = "PASS" if result["passed"] else "FAIL"
    if result["passed"]:
        snippet = f"{len(result['targets'])} targets, all carry >=1 severity:high criterion -- none can go ready on zero evidence"
    else:
        snippet = f"{len(result['risky_targets'])} false-green-risk target(s): " + ", ".join(result["risky_targets"])

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  oracle file : {args.oracles}")
        print(f"  targets     : {len(result['targets'])}")
        print()
        header = f"{'target':<32} {'crit':>4}  {'high':>4} {'med':>4} {'low':>4}   {'pred.status':<11} {'pred.score':>10}  risk"
        print(header)
        print("-" * len(header))
        for t in sorted(result["targets"], key=lambda t: (not t["false_green_risk"], t["name"])):
            h = t["severity_histogram"]
            marker = "  <-- FALSE-GREEN RISK" if t["false_green_risk"] else ""
            print(
                f"{t['name']:<32} {t['criteria_count']:>4}  {h['high']:>4} {h['medium']:>4} {h['low']:>4}   "
                f"{t['predicted_status_on_zero_evidence']:<11} {t['predicted_score_on_zero_evidence']:>10}{marker}"
            )
        if result["errors"]:
            print("\n  ERRORS:")
            for error in result["errors"]:
                print(f"    - {error}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
