#!/usr/bin/env python3
"""Pure validation for release-record facts and the current release notes."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re


class ContractError(ValueError):
    """A release record or notes document violates the publication contract."""


_TAG = re.compile(r"^v1\.3\.(?:[5-9]|[1-9][0-9]+)-evolve$")
_SEMVER = re.compile(r"^v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_STATUSES = {"RELEASE CANDIDATE", "PREPARED FOR PUBLICATION", "SHIPPED"}
_TOTALS = ("ctest_total", "vm_parity_total")
_PENDING_MARKER = "RELEASE_EVIDENCE_PENDING"
_PENDING_WORDING = re.compile(
    r"\b(?:pending|unrecorded|awaits?\s+(?:the\s+)?(?:full\s+)?(?:evidence|battery|verification)|"
    r"verification\s+(?:is\s+)?pending|evidence\s+(?:is\s+)?pending)\b", re.I
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _valid_tag(value: object, *, current: bool) -> bool:
    if not isinstance(value, str):
        return False
    return bool((_TAG if current else _SEMVER).fullmatch(value))


def _date(value: object) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ContractError("release_date must be a canonical YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ContractError("release_date must be a real calendar date") from exc
    if parsed.isoformat() != value:
        raise ContractError("release_date must be canonical YYYY-MM-DD")
    return parsed


def load_record(path, strict=False):
    """Load and type-check the release record; preparation may retain null totals."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            record = json.load(stream, object_pairs_hook=_unique_object)
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read release record: {exc}") from exc
    return validate_record(record, strict)


def validate_record(record, strict=False):
    """Type-check supplied objects too; callers cannot bypass loading checks."""
    if not isinstance(record, dict):
        raise ContractError("release record must be a JSON object")
    required = {"schema", "tag", "previous_tag", "release_date", "status", *_TOTALS}
    missing = sorted(required - record.keys())
    if missing:
        raise ContractError("release record is missing required keys: " + ", ".join(missing))
    if record["schema"] != "eshkol.release-record.v1":
        raise ContractError("unsupported release record schema")
    if not _valid_tag(record["tag"], current=True):
        raise ContractError("tag must match the supported v1.3.<5+>-evolve grammar")
    if not _valid_tag(record["previous_tag"], current=False):
        raise ContractError("previous_tag must be a valid semantic version tag")
    if record["previous_tag"] == record["tag"]:
        raise ContractError("previous_tag must precede tag")
    _date(record["release_date"])
    if not isinstance(record["status"], str) or record["status"] not in _STATUSES:
        raise ContractError("status must be a recognized release status")
    for key in _TOTALS:
        value = record[key]
        if value is None and not strict:
            continue
        if type(value) is not int or value <= 0:
            raise ContractError(f"{key} must be a positive integer" + (" in strict mode" if strict else " or null"))
    if strict:
        if record["status"] != "PREPARED FOR PUBLICATION":
            raise ContractError("strict publication requires PREPARED FOR PUBLICATION status")
        placeholders = sorted(key for key in record if "placeholder" in key.lower())
        if placeholders:
            raise ContractError("strict publication rejects placeholder keys: " + ", ".join(placeholders))
    if "_comment" in record and not isinstance(record["_comment"], str):
        raise ContractError("_comment must be text")
    if "sources" in record and (not isinstance(record["sources"], dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in record["sources"].items())):
        raise ContractError("sources must map text keys to text descriptions")
    for key in ("surface_total", "builtins_total", "ctest_required_total", "ctest_optional_skipped"):
        if key in record and (type(record[key]) is not int or record[key] < 0):
            raise ContractError(f"{key} must be a nonnegative integer")
    return record


def _current_notes_section(text: str) -> str:
    if not isinstance(text, str):
        raise ContractError("release notes must be text")
    return text.split("\n---\n", 1)[0]


def _date_from_notes(line: str) -> str | None:
    match = re.fullmatch(r"\s*\*\*(?:Planned|Intended) release date:\*\*\s*(.+?)\.?\s*", line)
    if not match:
        return None
    value = match.group(1)
    # Accept either the record's ISO date or a human date optionally prefixed
    # by its weekday, while verifying the weekday when one is supplied.
    try:
        parsed = date.fromisoformat(value)
        return parsed.isoformat()
    except ValueError:
        pass
    human = re.fullmatch(r"(?:(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+)?([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})", value)
    if not human:
        return None
    months = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
    weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    try:
        parsed = date(int(human.group(4)), months.index(human.group(2)) + 1, int(human.group(3)))
    except ValueError:
        return None
    if human.group(1) and weekdays[parsed.weekday()] != human.group(1):
        return None
    return parsed.isoformat()


def validate_notes(text, record, tag, role="preparation"):
    """Validate the current notes section against record facts and role."""
    if role not in {"preparation", "candidate-proof", "tag-publication"}:
        raise ContractError("unknown publication role")
    validate_record(record, strict=role != "preparation")
    section = _current_notes_section(text)
    if len(re.findall(rf"(?m)^# Eshkol {re.escape(tag)} — Release Notes$", text)) != 1:
        raise ContractError("duplicate or absent current-tag release heading")
    lines = section.splitlines()
    if not lines or lines[0] != f"# Eshkol {tag} — Release Notes":
        raise ContractError("release-notes heading does not match the requested tag")
    if tag != record.get("tag"):
        raise ContractError("requested tag does not match the release record")
    if len([line for line in lines if line.startswith("# ")]) != 1:
        raise ContractError("current notes must contain exactly one release heading")
    statuses = [m.group(1) for line in lines if (m := re.fullmatch(r"\s*\*\*Status:\*\*\s*(.*?)\.?\s*", line))]
    date_lines = [line for line in lines if re.match(r"\s*\*\*(?:Planned|Intended) release date:", line, re.I)]
    dates = [_date_from_notes(line) for line in date_lines]
    if len(statuses) != 1 or statuses[0] != record.get("status"):
        raise ContractError("current release notes must contain one matching status line")
    if len(dates) != 1 or dates[0] != record.get("release_date"):
        raise ContractError("current release notes must contain one matching planned release date")
    if role != "preparation":
        if record.get("status") != "PREPARED FOR PUBLICATION":
            raise ContractError("strict notes validation requires PREPARED FOR PUBLICATION intent")
        if _PENDING_MARKER in section or _PENDING_WORDING.search(section):
            raise ContractError("strict notes validation rejects pending or unrecorded evidence wording")
        if "RELEASE CANDIDATE" in section:
            raise ContractError("strict notes validation rejects RELEASE CANDIDATE wording")
        count_text = re.sub(r"(?<=\d),(?=\d)", "", section.replace("**", ""))
        for key in _TOTALS:
            total = record.get(key)
            if type(total) is not int or total <= 0:
                raise ContractError(f"strict notes validation requires a complete {key}")
            label = "CTest" if key == "ctest_total" else "VM parity(?: differential)?"
            if key == "ctest_total" and record.get("ctest_optional_skipped", 0):
                required = record.get("ctest_required_total")
                skipped = record["ctest_optional_skipped"]
                if type(required) is not int or required <= 0 or skipped <= 0 or required + skipped > total:
                    raise ContractError("invalid explicit configured/required/optional facts")
                explicit = f"CTest configured {total}, required {required}/{required}, optional skipped {skipped}"
                if explicit not in section or re.search(r"\bCTest\s+\d+/\d+", section, re.I):
                    raise ContractError("optional skip must be rendered separately, never as N/N PASS")
                continue
            claims = re.findall(rf"\b{label}\s+(\d+)/(\d+)\b", count_text, re.I)
            claims += re.findall(rf"\b(\d+)/(\d+)\s+{label}\b", count_text, re.I)
            if not claims or any(claim != (str(total), str(total)) for claim in claims):
                raise ContractError(f"current release notes must state {label} {total}/{total}")
    return section.rstrip() + "\n"

# The evidence contract is deliberately independent of ICC/network/build calls.
import argparse
import hashlib
import subprocess
import sys
import xml.etree.ElementTree as ET


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"unreadable JSON evidence {path}: {exc}") from exc


def sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ContractError(f"unreadable evidence {path}: {exc}") from exc


def evidence_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ContractError("evidence path must be relative")
    if Path(root).is_symlink():
        raise ContractError("evidence root cannot be a symlink")
    root = Path(root).resolve()
    candidate = root / relative
    if ".." in Path(relative).parts or any(p.is_symlink() for p in [candidate, *candidate.parents] if p != root.parent):
        raise ContractError("evidence paths cannot escape or use symlinks")
    try:
        candidate.resolve().relative_to(root)
    except ValueError as exc:
        raise ContractError("evidence path escapes bundle") from exc
    if not candidate.is_file():
        raise ContractError(f"evidence file missing: {relative}")
    return candidate


def source_snapshot(workspace):
    def git(*args):
        result = subprocess.run(["git", "-C", str(workspace), *args], capture_output=True, text=True)
        if result.returncode:
            raise ContractError("cannot obtain actual source identity")
        return result.stdout.strip()
    return {"sha": git("rev-parse", "HEAD"), "clean": not git("status", "--porcelain", "--untracked-files=no")}


def ctest_summary(text):
    matches = re.findall(r"(?m)^\s*(\d+)% tests passed(?:, (\d+) tests failed)? out of (\d+)\s*$", text)
    if len(matches) != 1:
        raise ContractError("CTest requires exactly one complete raw summary")
    percent, failed, total = matches[0]
    if not failed and percent != "100":
        raise ContractError("failure count absent from non-green CTest summary")
    percent, failed, total = int(percent), int(failed or 0), int(total)
    if total <= 0 or failed > total or abs(percent - 100 * (total - failed) / total) >= 1:
        raise ContractError("contradictory CTest summary")
    return total, failed


def ctest_raw_outcomes(text, total):
    """Require the producer's complete unique per-test console verdicts."""
    pattern = re.compile(r"^\s*(\d+)/(\d+)\s+Test\s+#\s*(\d+):\s+(\S+)\s+\.{2,}\s*(.+?)\s+([0-9]+(?:\.[0-9]+)?)\s+sec\s*$")
    outcomes, indices = {}, set()
    lines = iter(text.splitlines())
    for line in lines:
        # CTest attaches starred failures directly to the dotted leader and
        # prints a trailing newline inside PASS_REGULAR_EXPRESSION as two
        # console lines. Join only its exact closing-bracket/duration shape;
        # arbitrary test output must never supply a missing verdict.
        if not pattern.fullmatch(line) and re.match(r"^\s*\d+/\d+\s+Test\s+#", line) and re.search(
                r"\*+Failed\s+Required regular expression not found\. Regex=\[[^\n]*$", line):
            continuation = next(lines, "")
            if not re.fullmatch(r"\]\s+[0-9]+(?:\.[0-9]+)?\s+sec\s*", continuation):
                raise ContractError("truncated CTest regular-expression failure")
            line += continuation
        match = pattern.fullmatch(line)
        if not match:
            continue
        progress, denominator, test_index, name, status, duration = match.groups()
        index = int(test_index)
        if int(denominator) != total or not 1 <= int(progress) <= total or not 1 <= index <= total or index in indices or name in outcomes:
            raise ContractError("duplicate or contradictory raw CTest identity/denominator")
        indices.add(index)
        status = status.strip().lstrip("*")
        if status == "Passed":
            verdict = "passed"
        elif status in {"Skipped", "Not Run", "Disabled"}:
            verdict = "skipped"
        elif re.search(r"\btimeout\b", status, re.I):
            verdict = "infra"
        elif re.match(r"(?:Failed|Exception:|Subprocess aborted|SEGFAULT|ILLEGAL)", status, re.I):
            verdict = "failed"
        else:
            raise ContractError(f"unknown raw CTest verdict: {status}")
        outcomes[name] = verdict
    if len(outcomes) != total or indices != set(range(1, total + 1)):
        raise ContractError("raw CTest outcomes missing or truncated")
    return outcomes


def parity_summary(text):
    matches = re.findall(r"(?m)^vm-parity: (\d+) passed, (\d+) failed, (\d+) infra \(no verdict\)\s*$", text)
    if len(matches) != 1:
        raise ContractError("parity requires exactly one full raw summary")
    return tuple(map(int, matches[0]))


def junit_outcomes(path):
    try:
        cases = list(ET.parse(path).getroot().iter("testcase"))
    except (OSError, ET.ParseError) as exc:
        raise ContractError(f"invalid CTest JUnit: {exc}") from exc
    outcomes = {}
    for case in cases:
        name = case.get("name")
        if not name or name in outcomes:
            raise ContractError("missing or duplicate JUnit test identity")
        status = case.get("status", "")
        failures = [node for node in case if node.tag in {"failure", "error"}]
        if case.find("skipped") is not None or status in {"skipped", "notrun", "disabled"}:
            verdict = "skipped"
        elif failures or status in {"fail", "failed", "error"}:
            detail = " ".join(" ".join(node.itertext()) + " " + node.get("message", "") for node in failures)
            verdict = "infra" if re.search(r"\btimeout\b", detail, re.I) else "failed"
        elif status in {"", "run", "passed", "pass"}:
            verdict = "passed"
        else:
            raise ContractError(f"unknown JUnit status for {name}: {status}")
        outcomes[name] = verdict
    if not outcomes:
        raise ContractError("empty CTest JUnit")
    return outcomes


def trace_events(path):
    events = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = json.loads(line, object_pairs_hook=_unique_object)
                if not isinstance(event, dict):
                    raise ContractError("trace event must be an object")
                events.append(event)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid producer trace: {exc}") from exc
    return events


def measurement_facts(root, producer):
    """Derive counts from raw evidence, never from committed totals."""
    root = Path(root)
    text = (root / "raw.log").read_text(encoding="utf-8")
    events = trace_events(root / "trace.jsonl")
    if producer == "run_ctest_gate":
        total, raw_failed = ctest_summary(text)
        inventory = read_json(root / "inventory.json")
        tests = inventory.get("tests") if isinstance(inventory, dict) else None
        if not isinstance(tests, list) or not tests:
            raise ContractError("missing configured CTest inventory")
        names = [test.get("name") for test in tests if isinstance(test, dict)]
        if len(names) != len(tests) or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ContractError("invalid configured CTest identities")
        outcomes = junit_outcomes(root / "junit.xml")
        raw_outcomes = ctest_raw_outcomes(text, total)
        if raw_outcomes != outcomes:
            raise ContractError("raw CTest identities/outcomes disagree with JUnit")
        for name, verdict in outcomes.items():
            matching = [e for e in events if e.get("kind") == "ctest" and e.get("name") == "ctest_" + name]
            expected = {"passed": "PASS", "failed": "FAIL", "infra": "INFRA", "skipped": "SKIP"}[verdict]
            if len(matching) != 1 or matching[0].get("value") != expected:
                raise ContractError("raw CTest/JUnit outcomes disagree with unique per-test trace")
        if set(names) != set(outcomes) or total != len(names):
            raise ContractError("CTest inventory, JUnit and raw denominator disagree")
        counts = {key: sum(v == key for v in outcomes.values()) for key in ("passed", "failed", "infra", "skipped")}
        if raw_failed != counts["failed"] + counts["infra"]:
            raise ContractError("raw CTest failures disagree with JUnit")
        policy = read_json(root / "optional-policy.json")
        if not isinstance(policy, dict) or policy.get("schema") != "eshkol.release-optional-ctest.v1" or not isinstance(policy.get("exclusions"), dict):
            raise ContractError("invalid source-versioned optional policy")
        exclusions = policy["exclusions"]
        if any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in exclusions.items()) or not set(exclusions) <= set(names):
            raise ContractError("invalid optional exclusion identity or justification")
        skipped = sorted(n for n, v in outcomes.items() if v == "skipped")
        if not set(skipped) <= set(exclusions):
            raise ContractError("required or undeclared configured test skipped")
        green = [e for e in events if e.get("kind") == "ctest" and e.get("name") == "ctest_suite_green"]
        scans = [e for e in events if e.get("kind") == "ctest" and e.get("name") == "ctest_self_verdict_scan"]
        if len(green) != 1 or green[0].get("value") != "PASS" or len(scans) != 1 or scans[0].get("value") != "PASS":
            raise ContractError("CTest gate and self-verdict scan must both pass exactly once")
        if any(e.get("value") in {"FAIL", "ABSENT", "SHRUNK", "INFRA"} for e in events if e.get("kind") == "ctest"):
            raise ContractError("required CTest pillar or test did not pass")
        return {"total": total, **counts, "required_total": total - len(exclusions),
                "required_passed": sum(v == "passed" for n, v in outcomes.items() if n not in exclusions),
                "optional_skipped": len(skipped), "optional_skipped_inventory": skipped}
    if producer != "run_vm_parity":
        raise ContractError("unknown measurement producer")
    passed, failed, infra = parity_summary(text)
    inventory = read_json(root / "inventory.json")
    if not isinstance(inventory, dict) or set(inventory) != {"corpus", "found", "oos", "fatal"}:
        raise ContractError("full parity configured inventory required")
    required_names = {"vm_gap_canonicalization", "vm_parity_audit", "vm_parity_self_verdict_scan"}
    for category, files in inventory.items():
        if not isinstance(files, list) or any(not isinstance(name, str) or not re.fullmatch(r"[^/]+\.esk", name) for name in files) or len(set(files)) != len(files):
            raise ContractError("invalid parity configured inventory")
        if category != "found" and not files:
            raise ContractError("empty required parity stage inventory")
        for filename in files:
            stem = filename[:-4]
            names = {f"corpus_{stem}_vmsrc", f"corpus_{stem}_vmeskb"} if category == "corpus" else {f"fatal_{stem}_native", f"fatal_{stem}_vm"} if category == "fatal" else {f"{category}_{stem}"}
            required_names.update(names)
    reports = [e for e in events if e.get("kind") == "vm_parity" and e.get("name") not in {"vm_parity_gate", "vm_dispatch_native"}]
    names = [e.get("name") for e in reports]
    if len(set(names)) != len(names) or set(names) != required_names:
        raise ContractError("parity configured inventory does not match executed source/serialized stages")
    if (sum(e.get("value") == "PASS" for e in reports), sum(e.get("value") == "FAIL" for e in reports), sum(e.get("value") == "INFRA" for e in reports)) != (passed, failed, infra):
        raise ContractError("parity trace outcomes disagree with raw summary")
    reported = re.findall(r"(?m)^(PASSED|FAILED|INFRA)\s+tests/vm_parity[^\n]*$", text)
    if (reported.count("PASSED"), reported.count("FAILED"), reported.count("INFRA")) != (passed, failed, infra):
        raise ContractError("parity raw outcomes disagree with final summary")
    gates = [e for e in events if e.get("kind") == "vm_parity" and e.get("name") == "vm_parity_gate"]
    scope = [e for e in events if e.get("kind") == "release_measurement_scope"]
    flags = scope[0].get("value") if len(scope) == 1 else None
    full_scope = (isinstance(flags, dict) and set(flags) == {"do_eskb", "audit_only"}
                  and type(flags["do_eskb"]) is int and flags["do_eskb"] == 1
                  and type(flags["audit_only"]) is int and flags["audit_only"] == 0)
    if len(gates) != 1 or gates[0].get("value") != "PASS" or not full_scope:
        raise ContractError("parity requires full serialized-bytecode scope and one PASS gate")
    for stage in ("stage 1:", "stage 2:", "stage 3:", "stage 4:"):
        if stage not in text:
            raise ContractError("parity raw log is missing a required stage")
    total = passed + failed + infra
    return {"total": total, "passed": passed, "failed": failed, "infra": infra, "skipped": 0,
            "required_total": total, "required_passed": passed, "optional_skipped": 0, "optional_skipped_inventory": []}


def write_measurement(root, producer, workspace, start, cohort, phase_id, target, exit_code, argv, ctest_exit_code=None):
    root = Path(root)
    before = read_json(start)
    after = source_snapshot(workspace)
    receipt = {"schema": "eshkol.release-measurement.v1", "producer": producer,
               "source_sha": after["sha"], "source_clean": before == after and after["clean"] is True,
               "phase_id": phase_id, "target": target, "build_cohort_sha256": sha256(cohort),
               "mode": "full" if not argv else "partial", "argv": argv, "exit_code": exit_code,
               "ctest_exit_code": ctest_exit_code}
    if producer == "run_vm_parity":
        inventory = {category: sorted(path.name for path in (Path(workspace) / "tests/vm_parity" / category).glob("*.esk"))
                     for category in ("corpus", "found", "oos", "fatal")}
        (root / "inventory.json").write_text(json.dumps(inventory, sort_keys=True) + "\n")
    try:
        if producer == "run_ctest_gate" and sha256(root / "optional-policy.json") != sha256(Path(workspace) / "tests/coverage/release_optional_ctest.json"):
            raise ContractError("copied optional policy differs from actual source")
        receipt.update(measurement_facts(root, producer))
    except (ContractError, OSError, ValueError) as exc:
        receipt.update({field: None for field in ("total", "passed", "failed", "infra", "skipped", "required_total", "required_passed", "optional_skipped", "optional_skipped_inventory")})
        receipt["validation_errors"] = [str(exc)]
    (root / "source-end.json").write_text(json.dumps(after, sort_keys=True) + "\n")
    files = {"raw_log": "raw.log", "trace": "trace.jsonl", "source_start": "source-start.json", "source_end": "source-end.json", "exit_receipt": "exit.json"}
    if producer == "run_ctest_gate":
        files.update(configured_inventory="inventory.json", junit="junit.xml", optional_policy="optional-policy.json")
    else:
        # The VM configured scope is the trace of named checks actually reported.
        files.update(configured_inventory="inventory.json")
    for field, name in files.items():
        receipt[field + "_relative_path"] = name
        receipt[field + "_sha256"] = sha256(root / name) if (root / name).is_file() else None
    (root / "measurement.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def _hash(value, length=64):
    return isinstance(value, str) and bool(re.fullmatch(rf"[0-9a-f]{{{length}}}", value))


def load_measurements(bundle, sha, target, phase_id=None, cohort=None):
    bundle = Path(bundle)
    manifest = read_json(bundle / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != "eshkol.release-measurements.v1" or manifest.get("receipts") != ["ctest/measurement.json", "vm/measurement.json"]:
        raise ContractError("full CTest and VM measurement manifest required")
    selected = {}
    for relative in manifest["receipts"]:
        path = evidence_path(bundle, relative)
        if not isinstance(manifest.get("receipt_sha256"), dict) or sha256(path) != manifest["receipt_sha256"].get(relative):
            raise ContractError("tampered measurement receipt/manifest")
        receipt = read_json(path)
        if not isinstance(receipt, dict) or receipt.get("schema") != "eshkol.release-measurement.v1":
            raise ContractError("invalid measurement schema")
        producer = "run_ctest_gate" if relative.startswith("ctest/") else "run_vm_parity"
        if receipt.get("producer") != producer or receipt.get("mode") != "full" or receipt.get("argv") != []:
            raise ContractError("filtered or partial producer cannot certify publication")
        if receipt.get("source_sha") != sha or not _hash(sha, 40) or receipt.get("source_clean") is not True or receipt.get("target") != target:
            raise ContractError("measurement source/target mismatch or dirty source")
        if not isinstance(receipt.get("phase_id"), str) or not receipt["phase_id"] or not _hash(receipt.get("build_cohort_sha256")):
            raise ContractError("invalid phase or build cohort")
        if phase_id is not None and receipt["phase_id"] != phase_id or cohort is not None and receipt["build_cohort_sha256"] != cohort:
            raise ContractError("measurement phase/build cohort mismatch")
        for field in ("exit_code", "total", "passed", "failed", "infra", "skipped", "required_total", "required_passed", "optional_skipped"):
            if type(receipt.get(field)) is not int or receipt[field] < 0:
                raise ContractError(f"measurement {field} must be a nonnegative integer")
        if receipt["exit_code"] != 0 or receipt["total"] <= 0 or receipt["failed"] or receipt["infra"] or receipt["required_passed"] != receipt["required_total"]:
            raise ContractError("measurement has unresolved required outcomes")
        if producer == "run_ctest_gate" and (type(receipt.get("ctest_exit_code")) is not int or receipt["ctest_exit_code"] != 0):
            raise ContractError("nonzero actual CTest exit")
        fields = ("raw_log", "trace", "configured_inventory", "junit", "optional_policy") if producer == "run_ctest_gate" else ("raw_log", "trace", "configured_inventory")
        fields += ("source_start", "source_end", "exit_receipt")
        canonical_files = {"raw_log": "raw.log", "trace": "trace.jsonl", "configured_inventory": "inventory.json",
                           "junit": "junit.xml", "optional_policy": "optional-policy.json", "source_start": "source-start.json",
                           "source_end": "source-end.json", "exit_receipt": "exit.json"}
        for field in fields:
            if receipt.get(field + "_relative_path") != canonical_files[field]:
                raise ContractError(f"{producer} {field} must bind the consumed canonical file")
            evidence = evidence_path(path.parent, receipt.get(field + "_relative_path"))
            if not _hash(receipt.get(field + "_sha256")) or sha256(evidence) != receipt[field + "_sha256"]:
                raise ContractError(f"tampered {producer} {field}")
        start = read_json(path.parent / "source-start.json")
        end = read_json(path.parent / "source-end.json")
        if start != end or end != {"sha": sha, "clean": True} or not isinstance(end, dict) or end.get("clean") is not True or start.get("clean") is not True:
            raise ContractError("source changed during measurement")
        actual_exit = read_json(path.parent / "exit.json")
        if not isinstance(actual_exit, dict) or any(type(actual_exit.get(k)) is not int for k in (("gate_exit_code", "ctest_exit_code") if producer == "run_ctest_gate" else ("exit_code",))):
            raise ContractError("actual exit counters must be typed integers")
        if producer == "run_ctest_gate":
            if actual_exit != {"gate_exit_code": receipt["exit_code"], "ctest_exit_code": receipt["ctest_exit_code"], "argv": receipt["argv"]}:
                raise ContractError("actual CTest exit/argv receipt mismatch")
        elif actual_exit != {"exit_code": receipt["exit_code"]}:
            raise ContractError("actual parity exit receipt mismatch")
        derived = measurement_facts(path.parent, producer)
        if any(receipt.get(k) != v or type(receipt.get(k)) is not type(v) for k, v in derived.items()):
            raise ContractError("measurement counters contradict raw evidence")
        selected[producer] = receipt
    receipts = list(selected.values())
    if len({r["phase_id"] for r in receipts}) != 1 or len({r["build_cohort_sha256"] for r in receipts}) != 1:
        raise ContractError("measurements come from different phases/build cohorts")
    configured_cohort = read_json(bundle / "build-cohort.json")
    expected_source = {"git_head": sha, "tracked_state_sha256": hashlib.sha256(b"").hexdigest()}
    if not isinstance(configured_cohort, dict) or configured_cohort.get("__source__") != expected_source:
        raise ContractError("build cohort source is dirty or mismatched")
    if sha256(bundle / "build-cohort.json") != receipts[0]["build_cohort_sha256"]:
        raise ContractError("measurement build-cohort manifest mismatch")
    return selected


def metadata_facts(record_path, notes_path, bundle, sha, target, role):
    record = load_record(record_path, strict=True)
    notes = validate_notes(Path(notes_path).read_text(encoding="utf-8"), record, target, role)
    measurements = load_measurements(bundle, sha, target)
    ctest, vm = measurements["run_ctest_gate"], measurements["run_vm_parity"]
    state = read_json(evidence_path(bundle, "phase-state.json"))
    manifest = read_json(Path(bundle) / "manifest.json")
    if (not isinstance(state, dict) or state.get("schema") != "eshkol.release-evidence-phases.v1"
            or state.get("head") != sha or state.get("phase_id") != ctest["phase_id"]
            or state.get("completed") != ["baseline", "smoke", "final-evidence"]
            or state.get("coverage_completed") is not True
            or sha256(Path(bundle) / "phase-state.json") != manifest.get("phase_state_sha256")):
        raise ContractError("normal complete baseline/smoke/final-evidence phase proof required")
    source_policy = Path(notes_path).parent / "tests/coverage/release_optional_ctest.json"
    if sha256(source_policy) != ctest["optional_policy_sha256"]:
        raise ContractError("optional policy must match the actual publication source")
    if ctest["total"] != record["ctest_total"] or vm["total"] != record["vm_parity_total"]:
        raise ContractError("committed totals disagree with full measured totals")
    if ctest["optional_skipped"]:
        if record.get("ctest_optional_skipped") != ctest["optional_skipped"] or record.get("ctest_required_total") != ctest["required_total"]:
            raise ContractError("optional configured skips must have explicit record facts")
    elif ctest["passed"] != ctest["total"] or vm["passed"] != vm["total"]:
        raise ContractError("complete N/N measurements required")
    return {"record_sha256": sha256(record_path), "notes_sha256": hashlib.sha256(notes.encode()).hexdigest(),
            "measurement_manifest_sha256": sha256(Path(bundle) / "manifest.json"),
            "phase_id": ctest["phase_id"], "build_cohort_sha256": ctest["build_cohort_sha256"],
            "metadata_validated": True}


def validate_proof(proof, expected):
    if not isinstance(proof, dict) or proof.get("schema") != "eshkol.release-readiness.v2":
        raise ContractError("publication requires a v2 readiness proof")
    if proof.get("qualification_mode") != "normal" or type(proof.get("waiver_count")) is not int or proof["waiver_count"] != 0 or proof.get("metadata_validated") is not True:
        raise ContractError("normal complete no-waiver qualification required")
    if proof.get("status") != "ready" or type(proof.get("score")) is not int or proof["score"] != 100:
        raise ContractError("readiness proof must be typed ready/100")
    for field in ("sha", "target", "role", "run_id", "run_attempt", "phase_id", "build_cohort_sha256", "record_sha256", "notes_sha256", "measurement_manifest_sha256"):
        if field not in expected or type(proof.get(field)) is not type(expected[field]) or proof[field] != expected[field]:
            raise ContractError(f"readiness proof {field} mismatch")
    if proof["role"] not in {"candidate-proof", "tag-publication"} or not _hash(proof["sha"], 40):
        raise ContractError("invalid proof role/source")
    if any(type(proof[k]) is not int or proof[k] <= 0 for k in ("run_id", "run_attempt")):
        raise ContractError("proof run identity must be positive integers")
    if any(not _hash(proof[k]) for k in ("record_sha256", "notes_sha256", "measurement_manifest_sha256", "build_cohort_sha256")):
        raise ContractError("invalid proof hash")


def validate_publication(record_path, notes_path, bundle, proof, expected_identity):
    expected = dict(expected_identity)
    expected.update(metadata_facts(record_path, notes_path, bundle, expected["sha"], expected["target"], expected["role"]))
    validate_proof(proof, expected)
    return validate_notes(Path(notes_path).read_text(), load_record(record_path, strict=True), expected["target"], expected["role"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("source", "measure", "measurements", "metadata", "proof"))
    parser.add_argument("--workspace", default=".")
    for field in ("output", "root", "start", "cohort", "phase-id", "target", "sha", "record", "notes", "verdict"):
        parser.add_argument("--" + field)
    parser.add_argument("--producer", choices=("run_ctest_gate", "run_vm_parity"))
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--ctest-exit-code", type=int)
    parser.add_argument("--argv-json", default="[]")
    parser.add_argument("--role", choices=("candidate-proof", "tag-publication"))
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--run-attempt", type=int)
    args = parser.parse_args()
    try:
        if args.action == "source":
            result = source_snapshot(args.workspace)
        elif args.action == "measure":
            result = write_measurement(args.root, args.producer, args.workspace, args.start, args.cohort, args.phase_id, args.target, args.exit_code, json.loads(args.argv_json), args.ctest_exit_code)
        elif args.action == "measurements":
            load_measurements(args.root, args.sha, args.target, args.phase_id, sha256(args.cohort) if args.cohort else None)
            result = {"validated": True}
        else:
            result = metadata_facts(args.record, args.notes, args.root, args.sha, args.target, args.role)
            if args.action == "proof":
                verdict = read_json(args.verdict)
                if not isinstance(verdict, dict):
                    raise ContractError("ICC verdict must be an object")
                score = verdict.get("score", verdict.get("readiness"))
                if verdict.get("target") != args.target or verdict.get("status") != "ready" or type(score) not in (int, float) or score != 100:
                    raise ContractError("ICC verdict must be exact target ready/100")
                result.update(schema="eshkol.release-readiness.v2", sha=args.sha, target=args.target, role=args.role,
                              run_id=args.run_id, run_attempt=args.run_attempt, status="ready", score=100,
                              qualification_mode="normal", waiver_count=0)
                validate_proof(result, result)
        if args.action == "measure" and (result.get("validation_errors") or result["source_clean"] is not True or result["exit_code"] != 0):
            raise ContractError("producer receipt retained with incomplete/failed measurement")
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"publication contract {args.action}: PASS")
        return 0
    except (ContractError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"publication contract {args.action}: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
