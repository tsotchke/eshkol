#!/usr/bin/env python3
"""Release gate: `.icc/completion-oracles.yaml` parses and every criterion has
the shape ICC's completion-oracle grader needs to actually grade it.

Motivating incident: PR #429 added a new `higher_order_shadowing_oracle`
criterion but dropped the previous entry's `action:` field and the new
entry's `- runtime_event:` list-item opener. The new criterion's keys landed
INSIDE the previous criterion's `runtime_event` mapping — a well-formed-
looking edit that in fact broke the YAML parse (`ParserError: while parsing a
block mapping expected <block end>, but found <block mapping start>`). ICC's
own loader (`completion_oracle_config.load_completion_oracle_configs`)
already rejects an unparseable file outright and reports 0 criteria loaded;
what it CANNOT do from outside this repo is fail a PR before merge, because
nothing ran it. `icc readiness` was left blocked on an unparseable oracle
file (score 85) until commit c8449f8a restored the missing lines by hand —
after the breakage had already landed on the branch that fed the release
gate.

The sharper failure mode this gate is built to catch is not "the file fails
to parse" — that one is at least loud. It's the one where the file DOES
parse (YAML is very permissive) but a chunk of criteria collapses into
something the grader treats as fewer, or different, criteria than the file's
author wrote: a `runtime_event:` with no `event_kinds`, two compact keys on
one criterion, a duplicated criterion id shadowing an earlier one. Every one
of those keeps `icc readiness` running to completion and printing a verdict
— just a verdict computed over less evidence than the author thinks they
wired in. That is a readiness score of 100 computed by grading 2 of 43
criteria while silently dropping the other 41: vacuous green, worse than a
loud failure because it looks identical to a real pass.

This gate re-derives ICC's own per-criterion required-key rules (read
directly from `completion_oracle_config._normalize_criterion` in the ICC
source at /Users/tyr/Desktop/infinite_context_coder/scripts/, since ICC is a
private tool this repo does not vendor or depend on at build time) for the
criterion kinds this file actually uses today: `runtime_event`,
`no_contract_gap`, `no_stubbed_paths` and `test_evidence`. Other recognised
kinds are accepted structurally without a deep per-kind check; extend
`REQUIRED_KEYS_BY_KIND` here the day this file adopts one of them.

Checks
    (a) The file parses as YAML and its top level is `{oracles: [...]}`.
    (b) Every oracle has a `name` and a non-empty `requires` (or `criteria`)
        list, and no two oracles share a `name` or an alias token (either
        would make target resolution pick one arbitrarily).
    (c) Every criterion is a mapping with exactly one way to tell what kind
        it is (an explicit `kind:` field, XOR exactly one compact key), a
        `label`, a `severity` in {high, medium, low}, and an `action` — the
        convention every criterion in this file already follows.
    (d) The required payload key(s) for that criterion's kind are present
        and non-empty (`event_kinds` for `runtime_event`, `gap_kinds` for
        `no_contract_gap`, `contract_kinds` for `contract_kind`, `name` for
        `runtime_check`).
    (e) No two criteria in the same oracle declare the same explicit `id`
        (an id one criterion's edit was meant for landing on another).
    (f) DECLARED vs VALID criteria per oracle: the count of criteria the file
        says an oracle has, before normalisation, versus the count that
        actually validated. These are always printed, even on PASS, because
        a gap between them — as opposed to a gate failure — is exactly the
        "graded 2 of 43" shape: a human scanning green output would not
        otherwise notice that the file quietly lost criteria.

Grading
    PASS  the file parses, and every oracle/criterion satisfies (b)-(e).
    FAIL  a parse error, or any oracle/criterion violates (b)-(e). declared
          != valid for any oracle is ALSO a hard FAIL, not just a printed
          count — a criterion that fails to validate is a criterion ICC will
          never grade, silently.

The gate FAILS CLOSED: a missing or unparseable oracle file is FAIL.

Usage
    python3 scripts/check_oracle_schema.py
    python3 scripts/check_oracle_schema.py --oracles path/to/completion-oracles.yaml
    python3 scripts/check_oracle_schema.py --format json
    python3 scripts/check_oracle_schema.py --self-test

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
TRACE_BASENAME = "oracle_schema_gate.jsonl"
PROBE_ID = "oracle_schema_clean"

# The compact keys ICC's loader recognises (completion_oracles.COMPACT_CRITERION_KEYS)
# plus the full kind vocabulary (completion_oracles.VALID_CRITERION_KINDS), so a
# criterion using a kind this gate does not deep-validate is still accepted
# rather than rejected as "unknown".
COMPACT_KEYS = (
    "runtime_check",
    "runtime_event",
    "contract_kind",
    "runtime_or_contract",
    "no_contract_gap",
    "no_stubbed_paths",
    "test_evidence",
    "doc_truth",
)
KNOWN_KINDS = frozenset(COMPACT_KEYS) | frozenset({
    "model_verified",
    "capability_exercised",
    "audit_pattern_clean",
    "physics_bounds",
    "execution_source",
    "performance_budget",
})
VALID_SEVERITIES = ("high", "medium", "low")

# Required payload key per kind, and how to read it out of the compact or
# explicit form. `None` means "no additional required key beyond the kind
# itself" (test_evidence, no_stubbed_paths, doc_truth all default their
# payload when absent, per ICC's _normalize_criterion).
REQUIRED_KEYS_BY_KIND: dict[str, str | None] = {
    "runtime_check": "name",
    "runtime_event": "event_kinds",
    "contract_kind": "contract_kinds",
    "runtime_or_contract": None,  # event_kinds OR contract_kinds; checked specially
    "no_contract_gap": "gap_kinds",
    "no_stubbed_paths": None,
    "test_evidence": None,
    "doc_truth": None,
}


class OracleSchemaError(Exception):
    """The oracle file could not be read or parsed at all."""


def _load_yaml(path: str) -> object:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise OracleSchemaError(
            "PyYAML is required to check the oracle file (pip install pyyaml)"
        ) from exc

    if not os.path.isfile(path):
        raise OracleSchemaError(f"oracle file not found at {path} (the gate fails closed)")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        raise OracleSchemaError(f"oracle file at {path} is not parseable: {exc}") from exc


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item is not None]
    return [value]


def _payload_value(criterion: dict, kind: str, plural_key: str) -> list:
    """Read a kind's payload out of either the compact or the explicit form."""

    if kind in criterion and plural_key not in criterion:
        # Compact form: `runtime_event: eshkol_smoke` or `runtime_event: {...}`.
        raw = criterion[kind]
        if isinstance(raw, dict):
            return _as_list(raw.get(plural_key))
        return _as_list(raw)
    return _as_list(criterion.get(plural_key))


def _classify_criterion(raw: dict, oracle_name: str, position: int) -> tuple[dict | None, list[str]]:
    """Normalise one criterion. Returns (info, errors). info is None if the
    criterion could not even be classified (errors will be non-empty)."""

    label_for_error = f"oracle {oracle_name!r} criterion #{position}"
    if raw.get("id"):
        label_for_error += f" (id={raw['id']!r})"

    if not isinstance(raw, dict):
        return None, [f"{label_for_error}: is not a mapping"]

    errors: list[str] = []

    for key in ("id", "kind", "label", "severity", "action"):
        if key in raw and not isinstance(raw[key], str):
            errors.append(f"{label_for_error}: field {key!r} must be a string")

    if errors:
        return None, errors

    if "kind" in raw:
        kind = raw["kind"]
    else:
        present_compact = [key for key in COMPACT_KEYS if key in raw]
        if len(present_compact) == 0:
            return None, [f"{label_for_error}: has no `kind` field and no compact requirement key "
                          f"(expected one of {', '.join(COMPACT_KEYS)})"]
        if len(present_compact) > 1:
            return None, [f"{label_for_error}: has multiple compact requirement keys: "
                          f"{', '.join(present_compact)} (ambiguous — which one is the kind?)"]
        kind = present_compact[0]

    if kind not in KNOWN_KINDS:
        return None, [f"{label_for_error}: has unrecognised kind {kind!r}"]

    if not raw.get("label"):
        errors.append(f"{label_for_error}: missing `label`")
    severity = raw.get("severity")
    if not severity:
        errors.append(f"{label_for_error}: missing `severity`")
    elif severity not in VALID_SEVERITIES:
        errors.append(f"{label_for_error}: severity {severity!r} not one of {'/'.join(VALID_SEVERITIES)}")
    if not raw.get("action") and not raw.get("recommended_action"):
        errors.append(f"{label_for_error}: missing `action`")

    if kind == "runtime_or_contract":
        event_kinds = _payload_value(raw, kind, "event_kinds")
        contract_kinds = _payload_value(raw, kind, "contract_kinds")
        if not event_kinds and not contract_kinds:
            errors.append(f"{label_for_error}: runtime_or_contract requires event_kinds or contract_kinds")
    else:
        required_key = REQUIRED_KEYS_BY_KIND.get(kind)
        if required_key:
            values = _payload_value(raw, kind, required_key)
            if not values:
                errors.append(f"{label_for_error}: {kind} requires non-empty `{required_key}`")

    if errors:
        return None, errors

    return {"kind": kind, "id": raw.get("id")}, []


def check(data: object) -> dict:
    """Validate a parsed oracle document. Never raises; returns a report."""

    errors: list[str] = []

    if not isinstance(data, dict):
        return {"passed": False, "errors": ["oracle document is not a mapping"], "oracles": []}

    oracles_raw = data.get("oracles")
    if not isinstance(oracles_raw, list):
        return {"passed": False, "errors": ["oracle document has no top-level `oracles` list"], "oracles": []}

    oracle_reports: list[dict] = []
    seen_name_tokens: dict[str, str] = {}  # casefolded token -> owning oracle name

    for oindex, oracle in enumerate(oracles_raw, start=1):
        if not isinstance(oracle, dict):
            errors.append(f"oracle #{oindex} is not a mapping")
            continue

        name = oracle.get("name") or oracle.get("target")
        if not name:
            errors.append(f"oracle #{oindex} is missing required key `name`")
            name = f"<unnamed #{oindex}>"

        tokens = [str(name)] + [str(a) for a in _as_list(oracle.get("aliases"))]
        for token in tokens:
            folded = token.casefold()
            owner = seen_name_tokens.get(folded)
            if owner is not None and owner != name:
                errors.append(
                    f"oracle {name!r} shares name/alias token {token!r} with oracle {owner!r} "
                    "(target resolution would pick one arbitrarily)"
                )
            else:
                seen_name_tokens[folded] = name

        raw_criteria = oracle.get("requires")
        if raw_criteria is None:
            raw_criteria = oracle.get("criteria")
        if not isinstance(raw_criteria, list):
            errors.append(f"oracle {name!r} has no `requires`/`criteria` list")
            raw_criteria = []
        declared_count = len(raw_criteria)
        if declared_count == 0:
            errors.append(f"oracle {name!r} has zero criteria — it would pass vacuously on anything")

        valid_count = 0
        seen_ids: dict[str, int] = {}
        for cindex, raw_criterion in enumerate(raw_criteria, start=1):
            info, crit_errors = _classify_criterion(raw_criterion, str(name), cindex)
            if crit_errors:
                errors.extend(crit_errors)
                continue
            valid_count += 1
            explicit_id = info.get("id") if info else None
            if explicit_id:
                if explicit_id in seen_ids:
                    errors.append(
                        f"oracle {name!r} has duplicate criterion id {explicit_id!r} "
                        f"(criteria #{seen_ids[explicit_id]} and #{cindex})"
                    )
                else:
                    seen_ids[explicit_id] = cindex

        if valid_count != declared_count:
            errors.append(
                f"oracle {name!r}: declared {declared_count} criteria but only {valid_count} "
                "validated — the rest would be silently dropped by the grader"
            )

        oracle_reports.append({
            "name": str(name),
            "declared_criteria": declared_count,
            "valid_criteria": valid_count,
        })

    passed = not errors
    return {"passed": passed, "errors": errors, "oracles": oracle_reports}


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

_GOOD_ORACLES = """
oracles:
  - name: selftest-oracle
    aliases: [selftest]
    requires:
      - runtime_event:
          event_kinds: [eshkol_smoke]
          event_names: ["selftest_probe"]
          event_values: ["PASS"]
        severity: high
        label: self-test probe passes
        action: ./scripts/run_selftest.sh
      - test_evidence: true
        severity: medium
        label: self-test fixtures are indexed
        action: ctest -R selftest
"""

# Reproduces the #429 class of defect: a criterion's keys land inside the
# previous criterion's mapping because the list-item opener and the previous
# entry's closing field were both dropped. YAML rejects this outright.
_MALFORMED_YAML = """
oracles:
  - name: selftest-oracle
    requires:
      - runtime_event:
          event_kinds: [eshkol_smoke]
        severity: high
        label: first criterion, missing its action AND the next opener
          event_names: ["stray_key_inside_previous_mapping"]
        severity: high
        label: second criterion never got its own list item
        action: ./scripts/run_selftest.sh
"""

# Parses fine, but the runtime_event criterion has no event_kinds: this is
# the "parses but grades nothing" shape the gate exists to catch, since a
# parser alone accepts this file without complaint.
_MISSING_REQUIRED_KEY_ORACLES = """
oracles:
  - name: selftest-oracle
    requires:
      - runtime_event:
          event_names: ["no_event_kinds_here"]
        severity: high
        label: runtime_event with no event_kinds
        action: ./scripts/run_selftest.sh
"""

_DUPLICATE_CRITERION_ID_ORACLES = """
oracles:
  - name: selftest-oracle
    requires:
      - id: probe-a
        runtime_event:
          event_kinds: [eshkol_smoke]
        severity: high
        label: first probe
        action: ./scripts/run_selftest.sh
      - id: probe-a
        runtime_event:
          event_kinds: [eshkol_smoke]
        severity: high
        label: second probe, id collides with the first
        action: ./scripts/run_selftest.sh
"""


def _run_fixture(name: str, text: str, tmp_dir: str) -> tuple[bool, str]:
    path = os.path.join(tmp_dir, name + ".yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    try:
        data = _load_yaml(path)
    except OracleSchemaError as exc:
        return False, f"parse error (expected for a malformed-YAML fixture): {exc}"
    result = check(data)
    return result["passed"], "; ".join(result["errors"]) or "no errors"


def self_test() -> bool:
    cases = [
        ("malformed_yaml", _MALFORMED_YAML, False),
        ("missing_required_key", _MISSING_REQUIRED_KEY_ORACLES, False),
        ("duplicate_criterion_id", _DUPLICATE_CRITERION_ID_ORACLES, False),
        ("well_formed", _GOOD_ORACLES, True),
    ]

    all_ok = True
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-oracle-schema-") as tmp_dir:
        print("check_oracle_schema.py self-test:")
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
        result = check(data)
    except OracleSchemaError as exc:
        snippet = f"oracle file unusable: {exc}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "FAIL", snippet)
        if args.format == "json":
            print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: FAIL — {exc}", file=sys.stderr)
        return 1

    total_declared = sum(o["declared_criteria"] for o in result["oracles"])
    total_valid = sum(o["valid_criteria"] for o in result["oracles"])
    status = "PASS" if result["passed"] else "FAIL"
    if result["passed"]:
        snippet = (
            f"{len(result['oracles'])} oracles, {total_valid}/{total_declared} criteria "
            "declared and graded, no schema errors"
        )
    else:
        snippet = f"{len(result['errors'])} schema error(s): " + "; ".join(result["errors"][:5])

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, "total_declared_criteria": total_declared,
                           "total_valid_criteria": total_valid, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  oracle file : {args.oracles}")
        print(f"  oracles     : {len(result['oracles'])}")
        print(f"  criteria    : {total_valid}/{total_declared} graded (declared/valid — a human's vacuity check)")
        for o in result["oracles"]:
            marker = "" if o["valid_criteria"] == o["declared_criteria"] else "  <-- MISMATCH"
            print(f"    - {o['name']:<40} {o['valid_criteria']:>3}/{o['declared_criteria']:<3}{marker}")
        if result["errors"]:
            print("  ERRORS:")
            for error in result["errors"]:
                print(f"    - {error}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
