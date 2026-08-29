#!/usr/bin/env python3
"""Check the measured engine-parity counts against their allowed thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE = ROOT / "scripts" / "icc_traces" / "engine_parity_coverage.jsonl"
TRACE_NAME = "engine_semantic_parity_threshold"


def latest_event(path: Path) -> dict | None:
    if not path.is_file():
        return None
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("name") == "engine_semantic_parity":
            found = event
    return found


def grade(event: dict | None, max_new_divergent: int = 0) -> tuple[bool, str]:
    if not isinstance(event, dict):
        return False, "engine_semantic_parity event is missing"
    threshold = event.get("threshold")
    if not isinstance(threshold, dict):
        return False, "engine_semantic_parity event has no structured threshold payload"
    required = (
        "differential_fraction", "minimum_differential_fraction",
        "new_divergent_programs", "regressed_programs",
    )
    if any(key not in threshold for key in required):
        return False, "engine_semantic_parity threshold payload is incomplete"
    try:
        fraction = float(threshold["differential_fraction"])
        floor = float(threshold["minimum_differential_fraction"])
        new_divergent = int(threshold["new_divergent_programs"])
        regressions = int(threshold["regressed_programs"])
    except (TypeError, ValueError):
        return False, "engine_semantic_parity threshold payload has invalid types"
    if fraction < floor:
        return False, "differential coverage %.4f is below allowed floor %.4f" % (fraction, floor)
    if new_divergent > max_new_divergent:
        return False, "%d new divergent program(s) exceeds allowed count %d" % (
            new_divergent, max_new_divergent)
    if regressions:
        return False, "%d previously-agreeing program(s) regressed on the VM" % regressions
    if event.get("value") != "PASS":
        return False, "engine_semantic_parity event reports %r" % event.get("value")
    return True, "coverage %.2f%% >= %.2f%%; new divergences %d <= %d" % (
        fraction * 100.0, floor * 100.0, new_divergent, max_new_divergent)


def emit(path: Path, status: str, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "kind": "runtime_event",
            "name": TRACE_NAME,
            "value": status,
            "snippet": detail,
            "confidence": 1.0,
        }) + "\n")


def self_test() -> bool:
    good = {"name": "engine_semantic_parity", "value": "PASS", "threshold": {
        "differential_fraction": 0.50,
        "minimum_differential_fraction": 0.50,
        "new_divergent_programs": 0,
        "regressed_programs": 0,
    }}
    low = {**good, "threshold": {**good["threshold"], "differential_fraction": 0.49}}
    new = {**good, "threshold": {**good["threshold"], "new_divergent_programs": 1}}
    return grade(good)[0] and not grade(low)[0] and not grade(new)[0] and not grade(None)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--max-new-divergent", type=int, default=0)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        ok = self_test()
        print("check_engine_parity_threshold.py self-test: %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    ok, detail = grade(latest_event(args.trace_file), args.max_new_divergent)
    status = "PASS" if ok else "FAIL"
    print("engine_semantic_parity_threshold: %s -- %s" % (status, detail))
    if not args.no_trace:
        emit(args.trace_file.parent / "engine_semantic_parity_threshold.jsonl", status, detail)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
