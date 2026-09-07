#!/usr/bin/env python3
"""Release gate: every registered doc's surface/builtin-count claim must
match the machine-verified numbers, or the gate fails.

Motivating incident (doc-truth audit 2026-08-26, finding N4): the language
surface gate's own denominator moved from 1,106 to 1,107 constructs, and the
doc set had JUST finished a hand reconciliation pass to 1,106 THE SAME DAY —
every doc agreed with every other doc, and every one of them was wrong within
24 hours, because "reconcile the docs by hand" is a one-shot fix that rots the
moment the gate moves again. The same audit found a *second*, independent
class of the same defect: `docs/FEATURE_MATRIX.md` cited "1,058 in ADR-0011
S2.1" as the source of an old number -- ADR-0011 is the guest-collector
adapter and contains no such section. A hand-written citation can be wrong
in exactly the same silent way a hand-copied number can.

This script closes both failure modes by making "the docs agree with the
gate" a property CI checks by re-deriving each doc's claimed number from its
own text and comparing it against the canonical value read fresh from the
machine sources below, every run, rather than trusting that a previous
reconciliation pass is still standing.

Canonical sources (never hand-edited numbers -- read from the repo's own
generated/gated files):
    tests/coverage/coverage_policy.json  -> baseline_surface_total
        the enforced floor: the number `scripts/language_coverage.py`
        actually gates on.
    tests/coverage/language_surface.json -> counts.builtins_total
        the deterministic builtin count `scripts/gen_language_surface.py`
        derives from the BUILTINS[] tables directly.

Registered docs (the CLOSED set this gate checks -- adding a new doc that
states one of these numbers means adding it here deliberately, the same
discipline `check_required_context_consistency.py` applies to required
status contexts):
    README.md
    docs/FEATURE_MATRIX.md
    .icc/architecture-model.yaml
    docs/reference/*/INDEX.md (ad, agent, benchmarks, language, runtime,
        stdlib, tensors -- present today with no numeric claim in most of
        them; registered so a claim added later is checked from day one
        rather than needing a second incident to notice it should have
        been)
    docs/COMPILER_ROADMAP.md, docs/TEST_COVERAGE.md -- their surrounding
        CTest/SICP/parity figures are dated measurements pinned to a past
        cut (see the exclusion note below), but each also states an
        "executable language coverage N/N" clause that cross-references
        FEATURE_MATRIX.md as "the canonical surface count" -- present tense,
        no commit pinned, and exactly the clause that drifted (1,106 and
        1,091 respectively, both silently wrong per the 2026-08-28 audit).
        Registered for that clause specifically.
    docs/API_REFERENCE.md, docs/COMPLETE_LANGUAGE_SPECIFICATION.md,
        docs/ESHKOL_LANGUAGE_GUIDE.md, docs/ESHKOL_QUICK_REFERENCE.md --
        each stated "555+ built-in functions" (or "555+ builtins") as the
        current count against an actual 1,042, invisibly, because none of
        the four were registered.

Deliberately NOT registered: CHANGELOG.md, RELEASE_NOTES.md, ANNOUNCEMENT.md,
ROADMAP.md, docs/TESTING.md, press/*. Every occurrence the audit found in
those files is a dated claim pinned to a specific past release commit
("measured on the v1.3.4-evolve cut", "remeasured 2026-08-25 against
4bf871a0") with its own evidence citation -- correcting those to today's
numbers would misrepresent them as having been measured on a commit they
were not. Only docs that assert the CURRENT surface/builtin count, with no
commit pinned, belong in this registry.

Extraction, not a stale-value blocklist: for each registered doc this gate
runs a small set of regexes tuned to the phrasings these docs actually use
("N-construct", "N built-in functions", "N builtins", "language coverage
N/N", "surface_total = N", "N constructs including", "N is the enforced
floor") and compares WHATEVER NUMBER IS FOUND against the canonical value.
A blocklist of previously-wrong numbers would only catch reversion to an
already-known mistake; extracting the live claim and diffing it against the
gate's own denominator catches the next drift too, which is the point --
this makes the *class* of defect (docs silently outliving the number they
quote) impossible to reintroduce silently, not just this specific instance
of it.

CTest / VM-parity counts: the task that motivated this gate also asked for
ctest and VM-parity totals to be reconciled the same way. Unlike the surface
counts, this repository has no committed machine source for either (both are
produced only by actually running the suite), so this gate accepts optional
`--ctest-log` / `--parity-log` paths to an evidence file (ctest's own stdout,
or `scripts/run_vm_parity.sh`'s stdout) and, when given, checks the doc
claims against them too. Without a log path, those two checks are skipped
(reported, not silently ignored) rather than failing the whole gate on
evidence this repository does not commit.

Modes / exit status
    PASS      0   canonical machine sources were read successfully, and
                  every extracted doc claim matches them (ctest/parity
                  claims are also checked if a log was supplied).
    FAIL      1   canonical sources were read, but at least one registered
                  doc contains a mismatching number, OR a registered doc is
                  missing entirely (the registry itself has drifted).
    NO_DATA   2   the canonical machine sources themselves could not be
                  read at all -- nothing was verified. Distinct from PASS
                  so a caller cannot mistake "we never checked" for "we
                  checked and it's fine."

Usage
    python3 scripts/check_surface_counts.py
    python3 scripts/check_surface_counts.py --ctest-log build/ctest.log \\
        --parity-log build/vm_parity.log
    python3 scripts/check_surface_counts.py --format json
    python3 scripts/check_surface_counts.py --self-test

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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "surface_counts_gate.jsonl"
PROBE_ID = "surface_counts_consistent"

COVERAGE_POLICY_PATH = os.path.join(REPO_ROOT, "tests", "coverage", "coverage_policy.json")
LANGUAGE_SURFACE_PATH = os.path.join(REPO_ROOT, "tests", "coverage", "language_surface.json")

# The closed set of docs this gate checks. See the module docstring for why
# each file is (or is deliberately not) here.
REGISTERED_DOCS = [
    "README.md",
    "docs/FEATURE_MATRIX.md",
    ".icc/architecture-model.yaml",
    "docs/reference/ad/INDEX.md",
    "docs/reference/agent/INDEX.md",
    "docs/reference/benchmarks/INDEX.md",
    "docs/reference/language/INDEX.md",
    "docs/reference/runtime/INDEX.md",
    "docs/reference/stdlib/INDEX.md",
    "docs/reference/tensors/INDEX.md",
    # BI-20 (v1.3.5 docs audit, 2026-08-28): these six were the exact
    # failure mode this gate exists to close -- each states a live surface
    # or builtin-count claim with no commit pinned, and each had silently
    # drifted (docs/COMPILER_ROADMAP.md said 1,106, docs/TEST_COVERAGE.md
    # said 1,091, and all four language docs said "555+" against an actual
    # 1,042) while `check_surface_counts.py` reported "all 10 registered
    # docs agree" -- because none of the six were registered.
    "docs/COMPILER_ROADMAP.md",
    "docs/TEST_COVERAGE.md",
    "docs/API_REFERENCE.md",
    "docs/COMPLETE_LANGUAGE_SPECIFICATION.md",
    "docs/ESHKOL_LANGUAGE_GUIDE.md",
    "docs/ESHKOL_QUICK_REFERENCE.md",
]

# Each pattern has exactly one capturing group unless noted; a pattern with
# two groups (the "N/N" coverage-fraction phrasing) requires BOTH captured
# numbers to equal the canonical value.
SURFACE_TOTAL_PATTERNS = [
    re.compile(r"([0-9]{1,3}(?:,[0-9]{3})*)-construct\b"),
    re.compile(r"declared language surface is \*{0,2}([0-9,]+)\*{0,2} constructs"),
    re.compile(r"floor of\s+([0-9,]+) declared constructs"),
    re.compile(r"language coverage \*{0,2}([0-9,]+)/([0-9,]+)\*{0,2}"),
    re.compile(r"surface_total`?\s*[=:]\s*\*{0,2}([0-9,]+)\*{0,2}"),
    re.compile(r"\(([0-9,]+) constructs including"),
    re.compile(r"([0-9,]+)\s+is the enforced floor"),
    re.compile(r"baseline_surface_total[\"']?\s*[:=]\s*\*{0,2}([0-9,]+)\*{0,2}"),
]

BUILTINS_TOTAL_PATTERNS = [
    re.compile(r"\*{0,2}([0-9,]+) built-in functions\*{0,2}"),
    re.compile(r"([0-9,]+) builtins across"),
    re.compile(r"special forms,\s*([0-9,]+) builtins"),
    re.compile(r"\(([0-9,]+)\s*builtins \+ [0-9,]+ special forms"),
    re.compile(r"builtins_total[\"']?\s*[:=]\s*\*{0,2}([0-9,]+)\*{0,2}"),
]

CTEST_PATTERNS = [
    re.compile(r"CTest \*{0,2}([0-9,]+)/([0-9,]+)\*{0,2}"),
]

PARITY_PATTERNS = [
    re.compile(r"VM parity differential \*{0,2}([0-9,]+)/([0-9,]+)\*{0,2}"),
]

# ctest's own summary omits the "N tests failed" clause entirely when the
# failure count is zero ("100% tests passed out of N"), and includes it only
# when at least one test failed ("87% tests passed, 3 tests failed out of
# N") -- both forms are accepted so the log format's happy path is not
# mistaken for "unparseable".
CTEST_LOG_TOTAL_RE = re.compile(
    r"[0-9]+% tests passed(?:, [0-9]+ tests failed)? out of ([0-9]+)")
PARITY_LOG_RE = re.compile(r"vm-parity:\s*([0-9]+) passed,\s*([0-9]+) failed")


class SourceError(Exception):
    """A canonical machine source could not be read (gate fails closed)."""


def _to_int(token: str) -> int:
    return int(token.replace(",", ""))


def load_canonical_surface_total() -> int:
    if not os.path.isfile(COVERAGE_POLICY_PATH):
        raise SourceError(f"canonical source not found: {COVERAGE_POLICY_PATH}")
    try:
        with open(COVERAGE_POLICY_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise SourceError(f"{COVERAGE_POLICY_PATH} is not valid JSON: {exc}") from exc
    value = data.get("baseline_surface_total")
    if not isinstance(value, int):
        raise SourceError(f"{COVERAGE_POLICY_PATH} has no integer baseline_surface_total")
    return value


def load_canonical_builtins_total() -> int:
    if not os.path.isfile(LANGUAGE_SURFACE_PATH):
        raise SourceError(f"canonical source not found: {LANGUAGE_SURFACE_PATH}")
    try:
        with open(LANGUAGE_SURFACE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise SourceError(f"{LANGUAGE_SURFACE_PATH} is not valid JSON: {exc}") from exc
    counts = data.get("counts") if isinstance(data, dict) else None
    value = counts.get("builtins_total") if isinstance(counts, dict) else None
    if not isinstance(value, int):
        raise SourceError(f"{LANGUAGE_SURFACE_PATH} has no integer counts.builtins_total")
    return value


def parse_ctest_log(path: str) -> int | None:
    """Total tests run, from ctest's own summary line. None if unparseable."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None
    match = CTEST_LOG_TOTAL_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def parse_parity_log(path: str) -> int | None:
    """Total cases (passed + failed), from run_vm_parity.sh's summary line."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None
    match = PARITY_LOG_RE.search(text)
    if not match:
        return None
    return int(match.group(1)) + int(match.group(2))


def _scan(text: str, patterns: list[re.Pattern], canonical: int, quantity: str,
          doc_rel: str) -> list[dict]:
    findings = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            for group in match.groups():
                if group is None:
                    continue
                value = _to_int(group)
                if value != canonical:
                    line_no = text.count("\n", 0, match.start()) + 1
                    findings.append({
                        "doc": doc_rel,
                        "quantity": quantity,
                        "line": line_no,
                        "found": value,
                        "expected": canonical,
                        "snippet": match.group(0),
                    })
    return findings


def check_doc(doc_rel: str, surface_total: int, builtins_total: int,
              ctest_total: int | None, parity_total: int | None) -> tuple[list[dict], list[str]]:
    path = os.path.join(REPO_ROOT, doc_rel)
    if not os.path.isfile(path):
        return ([{
            "doc": doc_rel, "quantity": "registry", "line": 0, "found": None,
            "expected": None, "snippet": "registered doc does not exist",
        }], [])

    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    findings = []
    findings += _scan(text, SURFACE_TOTAL_PATTERNS, surface_total, "surface_total", doc_rel)
    findings += _scan(text, BUILTINS_TOTAL_PATTERNS, builtins_total, "builtins_total", doc_rel)

    notes = []
    if ctest_total is not None:
        findings += _scan(text, CTEST_PATTERNS, ctest_total, "ctest_total", doc_rel)
    elif any(p.search(text) for p in CTEST_PATTERNS):
        notes.append(f"{doc_rel}: cites a CTest N/N figure but no --ctest-log was given; not graded")

    if parity_total is not None:
        findings += _scan(text, PARITY_PATTERNS, parity_total, "parity_total", doc_rel)
    elif any(p.search(text) for p in PARITY_PATTERNS):
        notes.append(f"{doc_rel}: cites a VM-parity N/N figure but no --parity-log was given; not graded")

    return findings, notes


def run_gate(ctest_log: str | None, parity_log: str | None) -> dict:
    surface_total = load_canonical_surface_total()
    builtins_total = load_canonical_builtins_total()

    ctest_total = parse_ctest_log(ctest_log) if ctest_log else None
    parity_total = parse_parity_log(parity_log) if parity_log else None

    all_findings: list[dict] = []
    all_notes: list[str] = []
    for doc_rel in REGISTERED_DOCS:
        findings, notes = check_doc(doc_rel, surface_total, builtins_total,
                                     ctest_total, parity_total)
        all_findings.extend(findings)
        all_notes.extend(notes)

    return {
        "surface_total": surface_total,
        "builtins_total": builtins_total,
        "ctest_total": ctest_total,
        "parity_total": parity_total,
        "findings": all_findings,
        "notes": all_notes,
        "passed": not all_findings,
    }


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


# ───────────────────────── self-test ─────────────────────────

def self_test() -> bool:
    all_ok = True
    print("check_surface_counts.py self-test:")

    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-surface-gate-") as tmp_dir:
        policy_path = os.path.join(tmp_dir, "coverage_policy.json")
        manifest_path = os.path.join(tmp_dir, "language_surface.json")
        with open(policy_path, "w", encoding="utf-8") as handle:
            json.dump({"baseline_surface_total": 1107}, handle)
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({"counts": {"builtins_total": 1041}}, handle)

        global COVERAGE_POLICY_PATH, LANGUAGE_SURFACE_PATH, REGISTERED_DOCS
        real_policy, real_manifest, real_docs = COVERAGE_POLICY_PATH, LANGUAGE_SURFACE_PATH, REGISTERED_DOCS
        COVERAGE_POLICY_PATH, LANGUAGE_SURFACE_PATH = policy_path, manifest_path

        cases = [
            ("green_doc_matches_canonical",
             "the declared language surface is **1,107** constructs, "
             "**1,041 built-in functions**.\n", True),
            ("red_stale_surface_number",
             "the declared language surface is **1,106** constructs.\n", False),
            ("red_stale_builtins_number",
             "**1,040 built-in functions** in this release.\n", False),
            ("red_construct_suffix_stale",
             "a 1,106-construct canonical language surface.\n", False),
            ("green_no_claim_at_all",
             "this doc says nothing about the surface count.\n", True),
        ]
        for name, doc_text, expect_pass in cases:
            doc_path = os.path.join(tmp_dir, "DOC.md")
            with open(doc_path, "w", encoding="utf-8") as handle:
                handle.write(doc_text)
            REGISTERED_DOCS = [os.path.relpath(doc_path, REPO_ROOT)]
            result = run_gate(ctest_log=None, parity_log=None)
            ok = result["passed"] == expect_pass
            all_ok = all_ok and ok
            verdict = "OK" if ok else "GATE IS BROKEN"
            detail = "PASS" if result["passed"] else "; ".join(
                f"{f['doc']}:{f['line']} {f['quantity']}={f['found']} (expected {f['expected']})"
                for f in result["findings"])
            print(f"  [{verdict}] {name}: expected passed={expect_pass}, got passed={result['passed']}")
            print(f"           {detail}")

        # Missing registered doc must FAIL, not silently skip.
        REGISTERED_DOCS = ["does/not/exist.md"]
        result = run_gate(ctest_log=None, parity_log=None)
        ok = result["passed"] is False
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] missing_registered_doc_fails: "
              f"passed={result['passed']}")

        # Optional ctest-log path: a mismatching log must FAIL when supplied,
        # and be silently skipped (with a note) when not.
        doc_path = os.path.join(tmp_dir, "CTEST_DOC.md")
        with open(doc_path, "w", encoding="utf-8") as handle:
            handle.write("CTest **198/198** and nothing else.\n")
        REGISTERED_DOCS = [os.path.relpath(doc_path, REPO_ROOT)]

        result_no_log = run_gate(ctest_log=None, parity_log=None)
        ok = result_no_log["passed"] and any("not graded" in n for n in result_no_log["notes"])
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] ctest_claim_ungraded_without_log: "
              f"passed={result_no_log['passed']}, notes={result_no_log['notes']}")

        mismatched_log = os.path.join(tmp_dir, "ctest.log")
        with open(mismatched_log, "w", encoding="utf-8") as handle:
            handle.write("100% tests passed out of 200\n")
        result_bad_log = run_gate(ctest_log=mismatched_log, parity_log=None)
        ok = result_bad_log["passed"] is False
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] ctest_claim_checked_against_log: "
              f"passed={result_bad_log['passed']}")

        # The real-world format: ctest omits "N tests failed" entirely when
        # nothing failed. Must still parse (this is the format that shipped
        # broken on the first cut of this gate — 0 failures read as
        # unparseable and silently skipped the check).
        matched_log = os.path.join(tmp_dir, "ctest_ok.log")
        with open(matched_log, "w", encoding="utf-8") as handle:
            handle.write("100% tests passed out of 198\n")
        result_good_log = run_gate(ctest_log=matched_log, parity_log=None)
        ok = result_good_log["passed"] is True and result_good_log["ctest_total"] == 198
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] ctest_claim_matches_log_zero_failures: "
              f"passed={result_good_log['passed']}, ctest_total={result_good_log['ctest_total']}")

        # The with-failures format must also parse.
        some_failed_log = os.path.join(tmp_dir, "ctest_some_failed.log")
        with open(some_failed_log, "w", encoding="utf-8") as handle:
            handle.write("98% tests passed, 4 tests failed out of 198\n")
        result_some_failed = run_gate(ctest_log=some_failed_log, parity_log=None)
        ok = result_some_failed["ctest_total"] == 198
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] ctest_log_with_failures_parses: "
              f"ctest_total={result_some_failed['ctest_total']}")

        # NO_DATA path: canonical source unreadable.
        COVERAGE_POLICY_PATH = os.path.join(tmp_dir, "does-not-exist.json")
        no_data_raised = False
        try:
            load_canonical_surface_total()
        except SourceError:
            no_data_raised = True
        ok = no_data_raised
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] no_data_when_canonical_source_missing: "
              f"raised={no_data_raised}")

        COVERAGE_POLICY_PATH, LANGUAGE_SURFACE_PATH, REGISTERED_DOCS = real_policy, real_manifest, real_docs

    if all_ok:
        print("self-test: PASS — matching claims pass, any stale claim in any registered "
              "doc fails, a missing registered doc fails, ctest/parity claims are graded only "
              "when a log is supplied, and an unreadable canonical source is distinguishable "
              "from a clean pass")
    else:
        print("self-test: FAIL — the gate did not behave as specified", file=sys.stderr)
    return all_ok


# ───────────────────────── CLI ─────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ctest-log", default=None,
                         help="path to a ctest run's captured stdout; if given, CTest N/N "
                              "claims in registered docs are graded against it")
    parser.add_argument("--parity-log", default=None,
                         help="path to scripts/run_vm_parity.sh's captured stdout; if given, "
                              "VM parity N/N claims in registered docs are graded against it")
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    try:
        result = run_gate(args.ctest_log, args.parity_log)
    except SourceError as exc:
        snippet = f"NO_DATA: {exc}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "NO_DATA", snippet)
        if args.format == "json":
            print(json.dumps({"status": "NO_DATA", "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: NO_DATA — {exc}", file=sys.stderr)
            print("NO_DATA is not a pass: nothing was verified.", file=sys.stderr)
        return 2

    status = "PASS" if result["passed"] else "FAIL"
    if result["passed"]:
        snippet = (f"surface_total={result['surface_total']} "
                   f"builtins_total={result['builtins_total']} — every registered doc agrees")
    else:
        snippet = f"{len(result['findings'])} mismatch(es): " + "; ".join(
            f"{f['doc']}:{f['line']} {f['quantity']}={f['found']} (expected {f['expected']})"
            for f in result["findings"][:5]
        )

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  canonical surface_total  : {result['surface_total']}"
              f" (tests/coverage/coverage_policy.json)")
        print(f"  canonical builtins_total : {result['builtins_total']}"
              f" (tests/coverage/language_surface.json)")
        if result["ctest_total"] is not None:
            print(f"  canonical ctest_total    : {result['ctest_total']} ({args.ctest_log})")
        if result["parity_total"] is not None:
            print(f"  canonical parity_total   : {result['parity_total']} ({args.parity_log})")
        for note in result["notes"]:
            print(f"  note: {note}")
        if result["findings"]:
            print("  MISMATCHES:")
            for f in result["findings"]:
                print(f"    - {f['doc']}:{f['line']} [{f['quantity']}] "
                      f"found {f['found']!r}, expected {f['expected']!r} — {f['snippet']!r}")
        else:
            print(f"  all {len(REGISTERED_DOCS)} registered docs agree with the machine sources")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
