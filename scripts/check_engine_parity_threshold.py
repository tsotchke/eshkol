#!/usr/bin/env python3
"""Check the measured engine-parity counts against their allowed thresholds."""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE = ROOT / "scripts" / "icc_traces" / "engine_parity_coverage.jsonl"
TRACE_NAME = "engine_semantic_parity_threshold"

# Every coverage fraction and floor is compared EXACTLY, with no epsilon.
#
# A fraction is carried as its integer counts: `<key>_numerator` and
# `<key>_denominator` beside the float `<key>`. Comparisons use those counts
# through fractions.Fraction, which is integer cross-multiplication. A payload
# from before the counts existed carries only the float; it is compared as the
# exact binary value it holds (Fraction(float)). JSON round-trips a Python
# float exactly, so neither path needs a tolerance.
#
# There used to be an EPS of 1e-9 here, with a comment claiming it absorbed a
# floor recorded as round(fraction, 4) rounding UP past its own measurement.
# It never could: 4-place rounding moves a value by up to 5e-5, four orders of
# magnitude more than 1e-9. The recorded high-risk floor 0.3277, rounded up
# from the measured 155/473 = 0.327695..., therefore failed against the very
# measurement it was recorded from. The floor is now recorded exactly, so the
# comparison has nothing to absorb.


def exact_fraction(payload: dict, key: str) -> Fraction | None:
    """The exact value of fraction `key` in `payload`, or None when absent.

    Integer counts win when present; the float beside them must agree with
    them exactly, so a hand edit of one without the other is rejected rather
    than silently graded on whichever field a reader happens to prefer.
    Raises ValueError for a malformed or inconsistent fraction.
    """
    numerator = payload.get(key + "_numerator")
    denominator = payload.get(key + "_denominator")
    value = payload.get(key)
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("%s must be a number, not %r" % (key, value))
        if not math.isfinite(value):
            raise ValueError("%s must be finite, not %r" % (key, value))
    if numerator is None and denominator is None:
        return None if value is None else Fraction(value)
    if type(numerator) is not int or type(denominator) is not int:
        raise ValueError("%s_numerator and %s_denominator must both be integers"
                         % (key, key))
    if denominator <= 0 or not 0 <= numerator <= denominator:
        raise ValueError("%s counts %r/%r are not a fraction in [0, 1]"
                         % (key, numerator, denominator))
    if value is not None and value != numerator / denominator:
        raise ValueError("%s %r disagrees with its own counts %d/%d"
                         % (key, value, numerator, denominator))
    return Fraction(numerator, denominator)


def record_fraction(payload: dict, key: str, numerator: int, denominator: int) -> None:
    """Record `numerator/denominator` under `key` at full precision.

    Writes the integer counts (the value every reader grades on) and the
    float they divide to (exact to the last bit, for readers and older
    tooling). Never rounds: a recorded floor must never exceed the
    measurement it was recorded from.
    """
    payload[key] = numerator / denominator
    payload[key + "_numerator"] = numerator
    payload[key + "_denominator"] = denominator


def describe(payload: dict, key: str) -> str:
    numerator = payload.get(key + "_numerator")
    denominator = payload.get(key + "_denominator")
    percent = 100.0 * float(payload[key])
    if type(numerator) is int and type(denominator) is int:
        return "%d/%d (%.2f%%)" % (numerator, denominator, percent)
    return "%.2f%%" % percent


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
        fraction = exact_fraction(threshold, "differential_fraction")
        floor = exact_fraction(threshold, "minimum_differential_fraction")
        if fraction is None or floor is None:
            raise ValueError("differential fraction and floor must be numbers")
        new_divergent = int(threshold["new_divergent_programs"])
        regressions = int(threshold["regressed_programs"])
    except (TypeError, ValueError) as exc:
        return False, "engine_semantic_parity threshold payload has invalid fields: %s" % exc

    # High-risk axis + its achievability ceiling are optional for backward
    # compatibility with older trace events, but when present they are
    # validated BEFORE the ordinary floor comparisons below: a baseline whose
    # recorded high-risk floor exceeds what this exact corpus could ever
    # measure is not a run that failed, it is a MALFORMED baseline (the
    # literal `high_risk_differential_floor: 1.0` this gate exists to catch
    # never actually observed a run — see run_engine_parity_coverage.py's
    # --update-baseline, which writes the measured fraction instead).
    try:
        hr_fraction = exact_fraction(threshold, "high_risk_differential_fraction")
        hr_floor = exact_fraction(threshold, "high_risk_minimum_differential_fraction")
        hr_ceiling = exact_fraction(threshold, "high_risk_ceiling_fraction")
    except ValueError as exc:
        return False, "engine_semantic_parity high-risk threshold fields are invalid: %s" % exc
    if hr_fraction is not None and hr_floor is not None:
        if hr_ceiling is not None and hr_floor > hr_ceiling:
            return False, (
                "baseline is malformed: high-risk floor %s exceeds this "
                "corpus's own ceiling %s (the most high-risk differential "
                "evidence any run of this corpus could ever produce) -- no "
                "run can pass this; fix the baseline, not the run"
                % (describe(threshold, "high_risk_minimum_differential_fraction"),
                   describe(threshold, "high_risk_ceiling_fraction")))
        if hr_fraction < hr_floor:
            return False, (
                "high-risk differential coverage %s is below allowed floor %s"
                % (describe(threshold, "high_risk_differential_fraction"),
                   describe(threshold, "high_risk_minimum_differential_fraction")))

    if fraction < floor:
        return False, "differential coverage %s is below allowed floor %s" % (
            describe(threshold, "differential_fraction"),
            describe(threshold, "minimum_differential_fraction"))
    if new_divergent > max_new_divergent:
        return False, "%d new divergent program(s) exceeds allowed count %d" % (
            new_divergent, max_new_divergent)
    if regressions:
        return False, "%d previously-agreeing program(s) regressed on the VM" % regressions
    if event.get("value") != "PASS":
        return False, "engine_semantic_parity event reports %r" % event.get("value")
    return True, "coverage %s >= %s; new divergences %d <= %d" % (
        describe(threshold, "differential_fraction"),
        describe(threshold, "minimum_differential_fraction"),
        new_divergent, max_new_divergent)


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


def threshold_from_counts(credited: int, surface: int, floor_credited: int,
                          floor_surface: int, hr_credited: int, hr_surface: int,
                          hr_floor_credited: int, hr_floor_surface: int,
                          hr_ceiling: int) -> dict:
    """A threshold payload shaped exactly as run_engine_parity_coverage.py emits it."""
    threshold = {"new_divergent_programs": 0, "regressed_programs": 0}
    record_fraction(threshold, "differential_fraction", credited, surface)
    record_fraction(threshold, "minimum_differential_fraction", floor_credited, floor_surface)
    record_fraction(threshold, "high_risk_differential_fraction", hr_credited, hr_surface)
    record_fraction(threshold, "high_risk_minimum_differential_fraction",
                    hr_floor_credited, hr_floor_surface)
    record_fraction(threshold, "high_risk_ceiling_fraction", hr_ceiling, hr_surface)
    return threshold


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
    # is what --update-baseline writes (a floor recorded from the measurement
    # itself), the case the never-measured 100% literal could never be.
    hr_at_floor = good
    checks.append(("run at the recorded high-risk floor passes", grade(hr_at_floor)[0]))

    # A run whose high-risk coverage falls below the RECORDED measurement
    # fails, same as the overall floor -- this is the actual ratchet: it
    # fails only on a regression from what was once measured, never on
    # distance from an unmeasured aspiration.
    hr_below = {**good, "value": "FAIL", "threshold": {
        **good["threshold"], "high_risk_differential_fraction": 0.30}}
    checks.append(("high-risk coverage below its floor fails", not grade(hr_below)[0]))

    # A baseline recording a literal high-risk floor (1.0) above what the
    # corpus could EVER measure (its own ceiling, 0.3615) is REJECTED as a
    # malformed baseline with a clear, actionable message -- never graded as
    # an ordinary "the run fell short" failure, because no run ever could
    # clear it.
    hr_malformed_ok, hr_malformed_msg = grade({**good, "threshold": {
        **good["threshold"], "high_risk_minimum_differential_fraction": 1.0}})
    checks.append(("floor above the corpus ceiling is rejected as malformed",
                   not hr_malformed_ok))
    checks.append(("the malformed-baseline message names the defect",
                   "malformed" in hr_malformed_msg.lower()
                   and "ceiling" in hr_malformed_msg.lower()))

    # The measurement that exposed the rounded floor: high-risk coverage
    # 155/473 = 0.327695..., recorded as round(0.327695..., 4) = 0.3277 and
    # then graded "32.77% fell below the recorded floor 32.77%". A floor
    # recorded from a measurement must pass that same measurement, and one
    # construct fewer must fail. Both floors here are recorded by
    # record_fraction(), the writer run_engine_parity_coverage.py uses for
    # the baseline, so this covers the recording path and not only grade().
    baseline: dict = {}
    record_fraction(baseline, "differential_floor", 321, 1139)
    record_fraction(baseline, "high_risk_differential_floor", 155, 473)
    recorded_hr = exact_fraction(json.loads(json.dumps(baseline)),
                                 "high_risk_differential_floor")
    recorded = exact_fraction(json.loads(json.dumps(baseline)), "differential_floor")
    checks.append(("a recorded floor never exceeds its own measurement",
                   recorded_hr == Fraction(155, 473) and recorded == Fraction(321, 1139)))

    def measured(credited: int, hr_credited: int) -> dict:
        event = {"name": "engine_semantic_parity", "value": "PASS",
                 "threshold": threshold_from_counts(
                     credited, 1139, recorded.numerator, recorded.denominator,
                     hr_credited, 473, recorded_hr.numerator, recorded_hr.denominator,
                     171)}
        return json.loads(json.dumps(event))

    checks.append(("155/473 passes against the floor recorded from 155/473",
                   grade(measured(321, 155))[0]))
    checks.append(("154/473 fails against the floor recorded from 155/473",
                   not grade(measured(321, 154))[0]))
    checks.append(("320/1139 fails against the floor recorded from 321/1139",
                   not grade(measured(320, 155))[0]))

    # Counts are what get graded, so a float that disagrees with them is a
    # malformed payload, not a second opinion to pick between.
    inconsistent = measured(321, 155)
    inconsistent["threshold"]["high_risk_minimum_differential_fraction"] = 0.3277
    checks.append(("a float that disagrees with its counts is rejected",
                   not grade(inconsistent)[0]))

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
