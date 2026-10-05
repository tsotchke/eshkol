#!/usr/bin/env python3
"""Require genuine receipts before advancing between release evidence phases."""

from __future__ import annotations

import argparse
import json
import os
from release_publication_contract import ContractError, load_measurements, source_snapshot, read_json, sha256
from pathlib import Path


PHASE_REQUIREMENTS = {
    "baseline": (
        ("runtime_event", "language_surface_coverage", "PASS"),
        ("language_coverage_prereq", "core_suite", "PASS"),
        ("vm_parity", "vm_parity_gate", "PASS"),
        ("ctest", "ctest_suite_green", "PASS"),
    ),
    "smoke": (
        ("eshkol_smoke", "language_surface_coverage_floor", "PASS"),
        ("eshkol_smoke", "pipe_symbol_oracle", "PASS"),
        ("eshkol_smoke", "ad_carrier_model_clean", "PASS"),
        ("eshkol_smoke", "eskm_model_fuzz_smoke", "PASS"),
        ("runtime_event", "engine_semantic_parity_threshold", "PASS"),
    ),
}


def records(trace_dir: Path) -> list[dict]:
    result = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(event, dict):
                result.append(event)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=tuple(PHASE_REQUIREMENTS))
    parser.add_argument("--trace-dir", type=Path, required=True)
    args = parser.parse_args()
    events = records(args.trace_dir)
    errors = []
    for kind, name, value in PHASE_REQUIREMENTS[args.phase]:
        matched = [event for event in events if event.get("kind") == kind and event.get("name") == name]
        if len(matched) != 1 or matched[0].get("value") != value:
            actual = [event.get("value") for event in matched]
            errors.append(f"{kind}:{name}: expected one {value}, found {actual}")
    try:
        state = read_json(args.trace_dir / "release-phase-state.json")
        load_measurements(args.trace_dir / "publication", source_snapshot(Path(__file__).resolve().parents[1])["sha"],
                          os.environ.get("RELEASE_TARGET", "v1.3.6-evolve"), state.get("phase_id"),
                          sha256(args.trace_dir / "release-build-cohort.json"))
    except (ContractError, OSError, ValueError) as exc:
        errors.append(str(exc))
    if errors:
        for error in errors:
            print(f"release {args.phase} phase receipt: FAIL: {error}")
        return 1
    print(f"release {args.phase} phase receipts: PASS ({len(PHASE_REQUIREMENTS[args.phase])} current producer events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
