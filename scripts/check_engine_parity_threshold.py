#!/usr/bin/env python3
"""Check the measured engine-parity counts against their allowed thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE = ROOT / "scripts" / "icc_traces" / "engine_parity_coverage.jsonl"
TRACE_NAME = "engine_semantic_parity_threshold"

# Absorbs float round-trip noise: a floor recorded as round(fraction, 4) can
# round UP past the exact fraction it was measured from (e.g. 0.323467...
# rounds to a stored 0.3235, which is fractionally ABOVE 0.323467...).
# 1e-9 is nine orders of magnitude below the smallest real coverage change
# on a corpus this size (~1/1137 =~ 0.00088), so it never masks a genuine
# regression.
EPS = 1e-9


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

    # High-risk axis + its achievability ceiling are optional for backward
    # compatibility with older trace events, but when present they are
    # validated BEFORE the ordinary floor comparisons below: a baseline whose
    # recorded high-risk floor exceeds what this exact corpus could ever
    # measure is not a run that failed, it is a MALFORMED baseline (the
    # literal `high_risk_differential_floor: 1.0` this gate exists to catch
    # never actually observed a run — see run_engine_parity_coverage.py's
    # --update-baseline, which now writes the measured fraction instead).
    hr_fraction = threshold.get("high_risk_differential_fraction")
    hr_floor = threshold.get("high_risk_minimum_differential_fraction")
    if hr_fraction is not None and hr_floor is not None:
        try:
            hr_fraction = float(hr_fraction)
            hr_floor = float(hr_floor)
        except (TypeError, ValueError):
            return False, "engine_semantic_parity high-risk threshold fields have invalid types"
        hr_ceiling = threshold.get("high_risk_ceiling_fraction")
        if hr_ceiling is not None:
            try:
                hr_ceiling = float(hr_ceiling)
            except (TypeError, ValueError):
                return False, "engine_semantic_parity high-risk ceiling field has an invalid type"
            if hr_floor > hr_ceiling + EPS:
                return False, (
                    "baseline is malformed: high-risk floor %.4f exceeds this "
                    "corpus's own ceiling %.4f (the most high-risk differential "
                    "evidence any run of this corpus could ever produce) -- no "
                    "run can pass this; fix the baseline, not the run"
                    % (hr_floor, hr_ceiling))
        if hr_fraction < hr_floor - EPS:
            return False, (
                "high-risk differential coverage %.4f is below allowed floor %.4f"
                % (hr_fraction, hr_floor))

    if fraction < floor - EPS:
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
        "high_risk_differential_fraction": 0.3235,
        "high_risk_minimum_differential_fraction": 0.3235,
        "high_risk_ceiling_fraction": 0.3615,
    }}
    low = {**good, "threshold": {**good["threshold"], "differential_fraction": 0.49}}
    new = {**good, "threshold": {**good["threshold"], "new_divergent_programs": 1}}
    checks = [
        ("good passes", grade(good)[0]),
        ("overall coverage below its floor fails", not grade(low)[0]),
        ("a new divergence fails", not grade(new)[0]),
        ("a missing event fails", not grade(None)[0]),
    ]

    # A run measuring exactly at the recorded high-risk floor passes -- this
    # is what --update-baseline just wrote (32.35% recorded from a 32.35%
    # measurement), the case the never-measured 100% literal could never be.
    hr_at_floor = good
    checks.append(("run at the recorded high-risk floor passes", grade(hr_at_floor)[0]))

    # A run whose high-risk coverage falls below the RECORDED measurement
    # fails, same as the overall floor -- this is the actual ratchet: it
    # fails only on a regression from what was once measured, never on
    # distance from an unmeasured aspiration.
    hr_below = {**good, "value": "FAIL", "threshold": {
        **good["threshold"], "high_risk_differential_fraction": 0.30}}
    checks.append(("high-risk coverage below its floor fails", not grade(hr_below)[0]))

    # The defect this whole file exists to fix: a baseline recording a
    # literal high-risk floor (1.0) above what the corpus could EVER
    # measure (its own ceiling, 0.3615) is REJECTED as a malformed baseline
    # with a clear, actionable message -- never graded as an ordinary "the
    # run fell short" failure, because no run ever could clear it.
    hr_malformed_ok, hr_malformed_msg = grade({**good, "threshold": {
        **good["threshold"], "high_risk_minimum_differential_fraction": 1.0}})
    checks.append(("floor above the corpus ceiling is rejected as malformed",
                   not hr_malformed_ok))
    checks.append(("the malformed-baseline message names the defect",
                   "malformed" in hr_malformed_msg.lower()
                   and "ceiling" in hr_malformed_msg.lower()))

    ok = all(result for _, result in checks)
    if not ok:
        for label, result in checks:
            if not result:
                print("check_engine_parity_threshold.py self-test: FAILED -- %s" % label)
    return ok


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
