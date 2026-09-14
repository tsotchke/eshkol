#!/usr/bin/env python3
"""Turn a focused release CTest JUnit report into fail-closed test receipts."""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path


REQUIRED = (
    "qubit_linearity_engine_parity_gate",
    "closure_upvalue_capacity_overflow_gate",
    "abi_layout_pin_test",
    "v1_3_quoted_datum_kinds_runtime_smoke",
    "python_bindings_capsule_lifetime",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--ctest-exit-code", type=int, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    cases: dict[str, list[ET.Element]] = {name: [] for name in REQUIRED}
    try:
        root = ET.parse(args.junit).getroot()
    except (OSError, ET.ParseError) as exc:
        root = None
        errors.append(f"CTest JUnit report unavailable or malformed: {exc}")
    if root is not None:
        for case in root.iter("testcase"):
            name = case.attrib.get("name", "")
            if name in cases:
                cases[name].append(case)

    records = []
    for name in REQUIRED:
        matches = cases[name]
        passed = len(matches) == 1
        summary = ""
        if not matches:
            summary = "required CTest did not execute (missing from JUnit report)"
        elif len(matches) != 1:
            summary = f"required CTest appeared {len(matches)} times; expected exactly once"
        else:
            case = matches[0]
            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")
            status = case.attrib.get("status", "")
            passed = failure is None and error is None and skipped is None and status not in {"notrun", "failed"}
            summary = "CTest passed" if passed else "CTest did not pass"
            if failure is not None or error is not None or skipped is not None or status in {"notrun", "failed"}:
                detail_node = failure if failure is not None else error if error is not None else skipped
                detail = " ".join(detail_node.itertext()).strip() if detail_node is not None else ""
                reasons = [value for value in (detail[:500], f"status={status}" if status in {"notrun", "failed"} else "") if value]
                summary += ": " + ("; ".join(reasons) if reasons else "failure marker present")
        if not passed:
            errors.append(f"{name}: {summary}")
        records.append({
            "kind": "test_result",
            "name": f"ctest::{name}",
            "value": {"passed": passed, "summary": summary},
            "timestamp": int(time.time()),
        })

    if args.ctest_exit_code != 0:
        errors.append(f"ctest exited {args.ctest_exit_code}")
        records.append({
            "kind": "test_result",
            "name": "ctest::v1_3_required_suite",
            "value": {"passed": False, "summary": f"ctest exited {args.ctest_exit_code}"},
            "timestamp": int(time.time()),
        })
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    with args.trace.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    for error in errors:
        print(f"release CTest evidence: {error}")
    passed_count = sum(bool(record["value"]["passed"]) for record in records if record["name"].startswith("ctest::") and record["name"] != "ctest::v1_3_required_suite")
    print(f"release CTest evidence: {passed_count}/{len(REQUIRED)} required tests passed")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
