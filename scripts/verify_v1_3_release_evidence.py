#!/usr/bin/env python3
"""Fail closed unless every v1.3.5 criterion has fresh release-recipe evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


TEST_ACTIONS = {
    "BUILD_DIR=build ./tests/memory/vm_region_flat_rss_test.sh": "vm_region_flat_rss",
    "BUILD_DIR=build ./tests/memory/vm_region_evac_subtype_coverage_test.sh": "vm_region_evac_subtype_coverage",
    "cd build && cmake --build . && ctest -R qubit_linearity_engine_parity_gate": "ctest::qubit_linearity_engine_parity_gate",
    "./scripts/run_ad_exactness_gate.sh": "ad_exactness_gate",
    "./scripts/run_dense_tensor_ad_gate.sh": "dense_tensor_ad_gate",
    "BUILD_DIR=build ./scripts/run_tco_tests.sh": "release_action::run_tco_tests",
    "cd build && cmake --build . && ctest -R closure_upvalue_capacity_overflow_gate": "ctest::closure_upvalue_capacity_overflow_gate",
    "python3 scripts/abi_header_inventory.py && cd build && cmake --build . && ctest -R abi_layout_pin_test": "ctest::abi_layout_pin_test",
    "ASAN_OPTIONS=detect_leaks=1 LSAN_OPTIONS= ./scripts/run_sanitizer_fuzz.sh --quick": "sanitizer_fuzz_quick",
    "BUILD_DIR=build ./scripts/run_control_flow_tests.sh": "release_action::run_control_flow_tests",
    "python3 scripts/check_self_verdicts.py --self-test": "self_verdict_gate_self_test",
    "ctest -R v1_3_quoted_datum_kinds_runtime_smoke": "ctest::v1_3_quoted_datum_kinds_runtime_smoke",
    "cd build && cmake --build . && ctest -R python_bindings_capsule_lifetime": "ctest::python_bindings_capsule_lifetime",
    "python3 scripts/check_test_coverage.py": "test_coverage_inventory",
}


def load_events(trace_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(f"cannot read {path}: {exc}") from exc
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(record, dict):
                record["_trace_path"] = str(path)
                records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--trace-dir", type=Path, required=True)
    args = parser.parse_args()
    oracle_file = args.repo_root / ".icc" / "completion-oracles.yaml"
    data = yaml.safe_load(oracle_file.read_text(encoding="utf-8"))
    oracle = next((item for item in data.get("oracles", []) if item.get("name") == "v1.3.5-evolve"), None)
    if oracle is None:
        raise SystemExit("v1.3.5-evolve completion oracle is absent")
    criteria = oracle.get("requires", [])
    if len(criteria) != 37:
        raise SystemExit(f"expected 37 authored criteria; found {len(criteria)}")

    records = load_events(args.trace_dir)
    errors: list[str] = []
    checked_runtime = checked_tests = 0
    cohort = [r for r in records if r.get("kind") == "release_build_cohort" and r.get("name") == "release_build_cohort_clean"]
    if len(cohort) != 1 or cohort[0].get("value") != "PASS":
        errors.append("main compiler/runtime fingerprint cohort did not produce exactly one PASS")
    for index, criterion in enumerate(criteria, 1):
        if "runtime_event" in criterion:
            payload = criterion["runtime_event"]
            kinds = payload.get("event_kinds", [])
            names = payload.get("event_names", [])
            values = payload.get("event_values", [])
            for name in names:
                matched = [r for r in records if r.get("kind") in kinds and r.get("name") == name]
                if len(matched) != 1:
                    errors.append(f"criterion {index} {name}: expected one current producer event, found {len(matched)}")
                elif matched[0].get("value") not in values:
                    errors.append(f"criterion {index} {name}: producer value is {matched[0].get('value')!r}, expected one of {values}")
                checked_runtime += 1
        elif "test_evidence" in criterion:
            action = criterion.get("action", "")
            receipt_name = TEST_ACTIONS.get(action)
            if not receipt_name:
                errors.append(f"criterion {index}: unmapped test_evidence action: {action}")
                continue
            matched = [r for r in records if r.get("kind") == "test_result" and r.get("name") == receipt_name]
            passed = len(matched) == 1 and isinstance(matched[0].get("value"), dict) and matched[0]["value"].get("passed") is True
            if not passed:
                errors.append(f"criterion {index} {receipt_name}: expected exactly one passing test_result, found {len(matched)}")
            checked_tests += 1
        elif "no_stubbed_paths" in criterion:
            # ICC evaluates this criterion directly against the bound source.
            continue
        else:
            errors.append(f"criterion {index}: unsupported criterion shape")

    if errors:
        for error in errors:
            print(f"release evidence verification: FAIL: {error}")
        return 1
    print(f"release evidence verification: PASS ({checked_runtime} runtime events, {checked_tests} named test actions, 37 criteria)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
