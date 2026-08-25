#!/usr/bin/env python3
"""Release gate: no artifact from a run the harness counted as PASSING may
contain a self-reported failure marker.

WHY THIS EXISTS (docs/design/FLAW_DETECTION_ROADMAP.md, D-05 TESTS THAT
ANNOUNCE THEIR OWN FAILURE)

`tests/parser/test_function_shadowing.esk` printed `FAIL: Expected 12` on the
VM, out loud, on a green baseline, for months (SW-24, closed by #429) — the
program's own verdict said FAIL and the harness that ran it still counted the
run as a pass. Twenty of the `scripts/run_*_tests.sh` suites already scan
their OWN captured output for exactly this contradiction (see
`scripts/lib/test_isolation.sh`'s `eshkol_test_output_has_failure`, and
`run_all_tests.sh`'s "Harness contradiction" check at its own tail). Both are
correct and both stay in place — this gate does not replace them.

What neither of those provides is a REUSABLE, INDEPENDENTLY TESTABLE tool
that can be pointed at any captured artifact from any harness — including the
lanes the roadmap calls out by name as gaps: the VM and wasm lanes
(`run_vm_parity.sh` grades by `cmp -s native.out vmsrc.out` alone, so two
engines printing the identical FAIL line still compares equal and passes),
and the CTest suite (`ctest`'s own PASS/FAIL verdict is derived from exit
status, not from scanning `<system-out>` for a self-reported contradiction).

THREE WAYS TO FEED IT EVIDENCE

  --junit FILE      A CTest/JUnit XML file (e.g. from
                     `ctest --output-junit FILE`, exactly what
                     scripts/run_ctest_gate.sh already produces). For every
                     <testcase> CTest counted as passing (no <failure>/
                     <error> child, and no fail-shaped `status` attribute),
                     scan its captured <system-out>/<system-err> for a
                     self-reported failure marker.

  --manifest FILE   A TSV file, one row per captured artifact:
                         VERDICT<TAB>path/to/captured.log[<TAB>label]
                     VERDICT is PASS or FAIL, as the harness that produced
                     the log itself concluded. Rows marked FAIL are recorded
                     but never flagged (they are SUPPOSED to contain failure
                     text). This is the integration point for any harness:
                     write a manifest line next to each captured log.

  --pair VERDICT:PATH   One ad hoc artifact/verdict pair on the command line,
                     repeatable, for quick wiring without a manifest file.

A contradiction is: a PASS-graded artifact whose text contains an unambiguous
self-reported failure marker (FAIL, FAILED, FAILURE, FAILS, MISMATCH,
DIVERGENCE by default — see --extra-pattern to add more, e.g. bare ERROR,
which is deliberately NOT in the default set because several error-handling
tests in this tree print "ERROR:" as their EXPECTED output; the same
reasoning test_isolation.sh's ESHKOL_TEST_FAILURE_REGEX already documents).
Noise that merely MENTIONS a failure token without reporting one — a summary
line stating a zero count ("Failed: 0"), or a decorative banner title
("=== DEBUG MINIMAL FAIL TEST ===") — is filtered out first, using the same
rules `eshkol_test_filter_verdict_noise` uses, so this gate does not cry wolf
on the same false positives that motivated that shared helper.

Grading
    PASS  no PASS-graded artifact contains a self-reported failure marker
          (including the case where no input is given at all — nothing to
          contradict).
    FAIL  at least one PASS-graded artifact contains one.

Usage
    python3 scripts/check_self_verdicts.py --junit build/ctest-junit.xml
    python3 scripts/check_self_verdicts.py --manifest evidence/verdicts.tsv
    python3 scripts/check_self_verdicts.py --pair PASS:some.log --pair FAIL:other.log
    python3 scripts/check_self_verdicts.py --format json
    python3 scripts/check_self_verdicts.py --self-test

Exit status is 0 on PASS and 1 on FAIL (including under --self-test, where it
reports whether the gate itself is capable of failing).

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "self_verdict_gate.jsonl"
PROBE_ID = "self_verdict_clean"

# ─────────────────────────── ported failure-marker logic ───────────────────
#
# Mirrors scripts/lib/test_isolation.sh's ESHKOL_TEST_FAILURE_REGEX /
# ESHKOL_TEST_ZERO_FAILURE_REGEX / ESHKOL_TEST_TITLE_DECORATION, extended with
# MISMATCH and DIVERGENCE per the roadmap's requested marker set. Kept as a
# faithful port (same exclusions, same reasoning) rather than a rewrite, so
# the two implementations cannot silently drift into disagreeing about what
# counts as a failure.

_FAILURE_WORDS = r"(?:FAIL|FAILED|FAILURE|FAILS|MISMATCH|MISMATCHES|DIVERGENCE|DIVERGENCES)"
FAILURE_REGEX = re.compile(
    r"(^|[^A-Za-z0-9_])" + _FAILURE_WORDS + r"([^A-Za-z0-9_]|$)"
    r"|Failed:\s*[1-9]"
    r"|Failures:\s*[1-9]"
    r"|^\s*✗"  # ✗
    r"|Assertion failed"
    r"|Segmentation fault"
    r"|Bus error"
    r"|Abort trap"
    r"|fatal signal",
    re.MULTILINE,
)

_ZERO_FAILURE_REGEX = re.compile(
    r"(?:FAIL|FAILED|FAILURES?|FAILS|MISMATCHES?|DIVERGENCES?)\s*[:=]?\s*0(?:[^0-9]|$)"
)

# A run of 3+ of = * # - at BOTH ends is a decorative banner. Only bare FAIL
# is discounted this way (see module docstring / test_isolation.sh's own
# comment): FAILED/FAILURE/FAILS/MISMATCH/DIVERGENCE in a banner is still a
# verdict.
_TITLE_DECORATION = re.compile(r"^\s*[=*#-][=*#-][=*#-].*[=*#-][=*#-][=*#-]\s*$")
_TITLE_SAFE_WORDS = re.compile(r"FAILED|FAILURE|FAILS|MISMATCH|DIVERGENCE")


def _filter_verdict_noise(text: str) -> str:
    kept = []
    for line in text.splitlines():
        if _ZERO_FAILURE_REGEX.search(line):
            continue
        if _TITLE_DECORATION.match(line) and not _TITLE_SAFE_WORDS.search(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def offending_lines(text: str, extra_pattern: str | None = None, limit: int = 10) -> list[str]:
    """Lines in `text` that report a genuine self-reported failure, after
    dropping zero-count summaries and decorative titles. Empty list = clean."""
    if not text:
        return []
    filtered = _filter_verdict_noise(text)
    pattern = FAILURE_REGEX
    if extra_pattern:
        pattern = re.compile(pattern.pattern + "|" + extra_pattern, re.MULTILINE)
    hits = []
    for line in filtered.splitlines():
        if pattern.search(line):
            hits.append(line.strip())
            if len(hits) >= limit:
                break
    return hits


def has_failure(text: str, extra_pattern: str | None = None) -> bool:
    return bool(offending_lines(text, extra_pattern, limit=1))


# ───────────────────────────── evidence sources ────────────────────────────


class Contradiction:
    def __init__(self, source: str, name: str, offending: list[str]):
        self.source = source
        self.name = name
        self.offending = offending

    def describe(self) -> str:
        sample = "; ".join(self.offending[:3])
        return f"[{self.source}] {self.name!r} was graded PASS but its output contains: {sample}"


def _ctest_verdict_failed(case: ET.Element) -> bool:
    status = case.get("status") or ""
    return (
        case.find("failure") is not None
        or case.find("error") is not None
        or status in ("fail", "failed", "notrun", "error")
    )


def scan_junit(path: str, extra_pattern: str | None) -> tuple[list[Contradiction], int]:
    """Returns (contradictions, testcases_examined). Raises on unparseable XML —
    the caller decides whether that is fatal for the whole run."""
    root = ET.parse(path).getroot()
    contradictions: list[Contradiction] = []
    examined = 0
    for case in root.iter("testcase"):
        name = case.get("name") or ""
        if not name:
            continue
        status = case.get("status") or ""
        if case.find("skipped") is not None or status == "skipped":
            continue
        examined += 1
        if _ctest_verdict_failed(case):
            continue  # already graded FAIL — not a contradiction to check
        text_parts = []
        out_el = case.find("system-out")
        err_el = case.find("system-err")
        if out_el is not None and out_el.text:
            text_parts.append(out_el.text)
        if err_el is not None and err_el.text:
            text_parts.append(err_el.text)
        combined = "\n".join(text_parts)
        hits = offending_lines(combined, extra_pattern)
        if hits:
            contradictions.append(Contradiction(f"junit:{os.path.basename(path)}", name, hits))
    return contradictions, examined


def scan_manifest(path: str, extra_pattern: str | None) -> tuple[list[Contradiction], int]:
    contradictions: list[Contradiction] = []
    examined = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                raise ValueError(f"{path}:{lineno}: expected VERDICT<TAB>path[<TAB>label], got: {line!r}")
            verdict, artifact_path = fields[0].strip(), fields[1].strip()
            label = fields[2].strip() if len(fields) > 2 else artifact_path
            if verdict not in ("PASS", "FAIL"):
                raise ValueError(f"{path}:{lineno}: verdict must be PASS or FAIL, got {verdict!r}")
            examined += 1
            if verdict == "FAIL":
                continue
            if not os.path.isfile(artifact_path):
                contradictions.append(
                    Contradiction(f"manifest:{os.path.basename(path)}", label,
                                  [f"artifact does not exist: {artifact_path}"])
                )
                continue
            with open(artifact_path, "r", encoding="utf-8", errors="replace") as afh:
                text = afh.read()
            hits = offending_lines(text, extra_pattern)
            if hits:
                contradictions.append(Contradiction(f"manifest:{os.path.basename(path)}", label, hits))
    return contradictions, examined


def scan_pair(verdict: str, artifact_path: str, extra_pattern: str | None) -> tuple[list[Contradiction], int]:
    if verdict not in ("PASS", "FAIL"):
        raise ValueError(f"--pair verdict must be PASS or FAIL, got {verdict!r}")
    if verdict == "FAIL":
        return [], 1
    if not os.path.isfile(artifact_path):
        return [Contradiction("pair", artifact_path, [f"artifact does not exist: {artifact_path}"])], 1
    with open(artifact_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    hits = offending_lines(text, extra_pattern)
    if hits:
        return [Contradiction("pair", artifact_path, hits)], 1
    return [], 1


# ───────────────────────────── trace emission ──────────────────────────────


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


# ───────────────────────────────── self-test ────────────────────────────────

_JUNIT_CONTRADICTION = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="ctest" tests="2">
    <testcase name="legit_failure_test" status="run">
      <failure message="nonzero exit"/>
      <system-out>FAIL: Expected 12, got 7</system-out>
    </testcase>
    <testcase name="silently_wrong_test" status="run">
      <system-out>Running...
FAIL: Expected 12, got 7 (VM fast-path bug)
exit 0</system-out>
    </testcase>
  </testsuite>
</testsuites>
"""

_JUNIT_CLEAN = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="ctest" tests="2">
    <testcase name="legit_failure_test" status="run">
      <failure message="nonzero exit"/>
      <system-out>FAIL: Expected 12, got 7</system-out>
    </testcase>
    <testcase name="clean_pass_test" status="run">
      <system-out>Total: 17, Passed: 17, Failed: 0
=== DEBUG MINIMAL FAIL TEST ===
All good.</system-out>
    </testcase>
  </testsuite>
</testsuites>
"""


def _write(tmp_dir: str, name: str, content: str) -> str:
    path = os.path.join(tmp_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def self_test() -> bool:
    all_ok = True
    print("check_self_verdicts.py self-test:")

    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-self-verdicts-") as tmp_dir:
        # --- JUnit: a PASS-graded testcase whose own output says FAIL -> must FAIL.
        junit_bad = _write(tmp_dir, "bad.xml", _JUNIT_CONTRADICTION)
        contradictions, examined = scan_junit(junit_bad, None)
        ok = len(contradictions) == 1 and examined == 2
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] junit_contradiction: "
              f"expected 1 contradiction (the legit failure must NOT double-count), got {len(contradictions)}")

        # --- JUnit: only the legitimately-failed case mentions FAIL; the PASS
        #     case has a zero-count summary and a decorative banner -> clean.
        junit_good = _write(tmp_dir, "good.xml", _JUNIT_CLEAN)
        contradictions2, examined2 = scan_junit(junit_good, None)
        ok2 = len(contradictions2) == 0 and examined2 == 2
        all_ok = all_ok and ok2
        print(f"  [{'OK' if ok2 else 'GATE IS BROKEN'}] junit_clean_with_noise: "
              f"expected 0 contradictions (zero-count + banner must not trip), got {len(contradictions2)}")

        # --- manifest: PASS row pointing at a log containing DIVERGENCE -> FAIL.
        div_log = _write(tmp_dir, "divergence.log", "native: 42\nvm: 41\nDIVERGENCE: values differ\n")
        clean_log = _write(tmp_dir, "clean.log", "Failed: 0\nAll 60 checks green.\n")
        real_fail_log = _write(tmp_dir, "real_fail.log", "MISMATCH: expected 3.14 got 2.71\n")
        manifest_bad = _write(tmp_dir, "manifest_bad.tsv",
                               f"PASS\t{div_log}\tvm-lane\nPASS\t{clean_log}\tclean-lane\n")
        contradictions3, examined3 = scan_manifest(manifest_bad, None)
        ok3 = len(contradictions3) == 1 and examined3 == 2 and contradictions3[0].name == "vm-lane"
        all_ok = all_ok and ok3
        print(f"  [{'OK' if ok3 else 'GATE IS BROKEN'}] manifest_divergence: "
              f"expected 1 contradiction naming vm-lane, got {[c.name for c in contradictions3]}")

        # --- manifest: the row with the real failure text is honestly marked
        #     FAIL, so it must NOT be flagged (FAIL rows are never contradictions).
        manifest_honest = _write(tmp_dir, "manifest_honest.tsv",
                                  f"FAIL\t{real_fail_log}\thonest-fail\nPASS\t{clean_log}\tclean-lane\n")
        contradictions4, examined4 = scan_manifest(manifest_honest, None)
        ok4 = len(contradictions4) == 0 and examined4 == 2
        all_ok = all_ok and ok4
        print(f"  [{'OK' if ok4 else 'GATE IS BROKEN'}] manifest_honest_fail_not_flagged: "
              f"expected 0 contradictions, got {len(contradictions4)}")

        # --- manifest: a PASS row pointing at a missing artifact -> FAIL.
        manifest_missing = _write(tmp_dir, "manifest_missing.tsv",
                                   f"PASS\t{os.path.join(tmp_dir, 'does_not_exist.log')}\tghost\n")
        contradictions5, _ = scan_manifest(manifest_missing, None)
        ok5 = len(contradictions5) == 1
        all_ok = all_ok and ok5
        print(f"  [{'OK' if ok5 else 'GATE IS BROKEN'}] manifest_missing_artifact: "
              f"expected 1 contradiction, got {len(contradictions5)}")

        # --- ad hoc pair, clean -> no contradiction.
        contradictions6, _ = scan_pair("PASS", clean_log, None)
        ok6 = len(contradictions6) == 0
        all_ok = all_ok and ok6
        print(f"  [{'OK' if ok6 else 'GATE IS BROKEN'}] pair_clean: expected 0, got {len(contradictions6)}")

        # --- ad hoc pair, contradictory -> contradiction.
        contradictions7, _ = scan_pair("PASS", div_log, None)
        ok7 = len(contradictions7) == 1
        all_ok = all_ok and ok7
        print(f"  [{'OK' if ok7 else 'GATE IS BROKEN'}] pair_contradiction: expected 1, got {len(contradictions7)}")

    if all_ok:
        print("self-test: PASS — the gate fails on every broken fixture and passes the well-formed one")
    else:
        print("self-test: FAIL — the gate did not discriminate broken input from good input", file=sys.stderr)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--junit", action="append", default=[], help="CTest/JUnit XML file (repeatable)")
    parser.add_argument("--manifest", action="append", default=[], help="VERDICT<TAB>path[<TAB>label] TSV file (repeatable)")
    parser.add_argument("--pair", action="append", default=[], help="VERDICT:PATH ad hoc pair (repeatable)")
    parser.add_argument("--extra-pattern", default=None, help="extra regex OR-ed into the failure marker set")
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    contradictions: list[Contradiction] = []
    examined = 0
    read_errors: list[str] = []

    for path in args.junit:
        try:
            c, n = scan_junit(path, args.extra_pattern)
            contradictions.extend(c)
            examined += n
        except (OSError, ET.ParseError) as exc:
            read_errors.append(f"junit {path}: {exc}")

    for path in args.manifest:
        try:
            c, n = scan_manifest(path, args.extra_pattern)
            contradictions.extend(c)
            examined += n
        except (OSError, ValueError) as exc:
            read_errors.append(f"manifest {path}: {exc}")

    for spec in args.pair:
        if ":" not in spec:
            read_errors.append(f"--pair must be VERDICT:PATH, got {spec!r}")
            continue
        verdict, artifact_path = spec.split(":", 1)
        try:
            c, n = scan_pair(verdict, artifact_path, args.extra_pattern)
            contradictions.extend(c)
            examined += n
        except (OSError, ValueError) as exc:
            read_errors.append(f"pair {spec}: {exc}")

    # A read error (unparseable JUnit, malformed manifest) is itself a FAIL —
    # fail closed rather than silently skipping evidence this gate was told
    # to check.
    passed = not contradictions and not read_errors

    if passed:
        snippet = f"{examined} artifact(s) examined across {len(args.junit) + len(args.manifest) + len(args.pair)} source(s), no contradiction"
    else:
        details = [c.describe() for c in contradictions] + read_errors
        snippet = f"{len(details)} problem(s): " + "; ".join(details[:5])

    if not args.no_trace:
        emit_trace(args.trace_dir, "PASS" if passed else "FAIL", snippet)

    if args.format == "json":
        print(json.dumps({
            "passed": passed,
            "examined": examined,
            "contradictions": [{"source": c.source, "name": c.name, "offending": c.offending} for c in contradictions],
            "read_errors": read_errors,
        }, indent=2))
    else:
        print(f"{PROBE_ID}: {'PASS' if passed else 'FAIL'}")
        print(f"  artifacts examined : {examined}")
        if contradictions:
            print("  CONTRADICTIONS (graded PASS, but self-reports failure):")
            for c in contradictions:
                print(f"    - {c.describe()}")
        if read_errors:
            print("  READ ERRORS:")
            for e in read_errors:
                print(f"    - {e}")
        if examined == 0 and not args.junit and not args.manifest and not args.pair:
            print("  (no --junit/--manifest/--pair given — nothing to scan)")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
