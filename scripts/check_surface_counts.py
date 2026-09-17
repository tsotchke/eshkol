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
ctest and VM-parity totals to be reconciled the same way. Both are produced
only by actually running the suite, so this gate accepts optional
`--ctest-log` / `--parity-log` paths to an evidence file (ctest's own stdout,
or `scripts/run_vm_parity.sh`'s stdout) and, when given, checks the doc
claims against them. The committed copy of that evidence is the release
record (below): without a log the claims are graded against the record, a log
that disagrees with the record fails, and a total the record does not carry
yet is reported rather than silently ignored.

Release record (tests/coverage/release_record.json): the release date, the
release status and the two evidence totals were restated by hand across a
dozen release-facing documents with no machine source at all, so a change
of date or status meant finding every restatement again. The record is now
their one source. For each
document in RELEASE_DOCS, inside the scope that describes the current
release, the gate extracts every statement of the release date (changelog
heading, "Release date:" line, "<tag> ... SHIPPED <date>", "<tag> (<date>)")
and compares it with the record, checks a stated weekday against the
calendar, requires the ladder rows and the roadmap heading for the tag to
carry the record's status label, refuses pre-release wording once the record
says SHIPPED, requires the anchor documents to state the date at all (so a
claim cannot be deleted to pass), and requires every
`<!-- release-record:KEY -->` span to equal the record's rendering. The
generated site pages under site/static/content are graded as text, so a
stale mirror fails too. `--sync` rewrites every graded claim from the
record; `--require-complete` fails while the record still has a null total.

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
    python3 scripts/check_surface_counts.py --sync              # rewrite from the release record
    python3 scripts/check_surface_counts.py --require-complete  # no null total in the record
    python3 scripts/check_surface_counts.py --self-test

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import datetime
import html
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
    re.compile(r"VM parity(?: differential)? \*{0,2}([0-9,]+)/([0-9,]+)\*{0,2}"),
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


# ───────────────────────── release record ─────────────────────────
#
# The surface and builtin totals above have a generated manifest as their
# machine source. Four release facts did not, and each was restated by hand
# across a dozen documents: the release date, the release status, the CTest
# total and the VM-parity total. tests/coverage/release_record.json is their
# one committed source; everything below grades the release-facing documents
# against it, and `--sync` rewrites the graded claims from it.

RELEASE_RECORD_PATH = os.path.join(REPO_ROOT, "tests", "coverage", "release_record.json")
RELEASE_STATUS_SHIPPED = "SHIPPED"
RELEASE_STATUSES = ("RELEASE CANDIDATE", RELEASE_STATUS_SHIPPED)

# Documents that speak about the CURRENT release. Unlike REGISTERED_DOCS these
# include the dated narrative files, so each is graded only inside the scope
# that describes the current release (see release_scope).
RELEASE_DOCS = [
    "README.md",
    "RELEASE_NOTES.md",
    "CHANGELOG.md",
    "ANNOUNCEMENT.md",
    "ROADMAP.md",
    "CONTRIBUTING.md",
    "docs/README.md",
    "docs/COMPILER_ROADMAP.md",
    "docs/COMPLETE_LANGUAGE_SPECIFICATION.md",
    "docs/FEATURE_MATRIX.md",
    "docs/KNOWN_ISSUES.md",
    "docs/TESTING.md",
    "docs/TEST_COVERAGE.md",
    "press/ESHKOL_DESCRIPTION_COPY.md",
    "press/ESHKOL_PRESS_INFORMATION_SHEET.md",
]

# Documents that must state the release date at least once. Without this a
# date claim could be deleted to make the gate pass.
RELEASE_DATE_ANCHORS = ["README.md", "RELEASE_NOTES.md", "CHANGELOG.md", "ROADMAP.md"]

# Generated HTML mirrors of release documents (scripts/build-site-content.sh).
# They are graded as text so a stale mirror cannot ship a superseded date or
# status; they are never rewritten by --sync, only regenerated.
SITE_MIRROR_DIR = os.path.join("site", "static", "content")

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_ISO_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
_LONG_DATE = r"(?:(?P<weekday>[A-Z][a-z]+day),\s+)?(?P<long>(?:%s)\s+[0-9]{1,2},\s+[0-9]{4})" % "|".join(MONTHS)
_DAY_FIRST_DATE = r"(?P<dayfirst>[0-9]{1,2}\s+(?:%s)\s+[0-9]{4})" % "|".join(MONTHS)

# Pre-release status vocabulary. A status is a vocabulary rather than a
# number, so it cannot be extracted and diffed the way a count is; what can be
# checked is that none of the wording that describes an unreleased cut
# survives once the record says SHIPPED.
PRE_RELEASE_PHRASES = [
    re.compile(r"refreshed\s+(?:release\s+)?candidate", re.IGNORECASE),
    re.compile(r"(?:previous|earlier)[\s-]+(?:v[0-9][0-9A-Za-z.\-]*\s+)?candidate", re.IGNORECASE),
    re.compile(r"verification\s+(?:is\s+|remains\s+|and\s+publication\s+are\s+)?(?:pending|in\s+progress)", re.IGNORECASE),
    re.compile(r"planned\s+release\s+date", re.IGNORECASE),
    re.compile(r"planned\s+(?:for|release\s+on)\s+[A-Z][a-z]+day", re.IGNORECASE),
    re.compile(r"pending\s+remeasurement", re.IGNORECASE),
    re.compile(r"has\s+not\s+yet\s+rerun", re.IGNORECASE),
    re.compile(r"do\s+not\s+infer\s+a\s+readiness", re.IGNORECASE),
    re.compile(r"awaiting\s+(?:its\s+final|open|hardening)", re.IGNORECASE),
    re.compile(r"release\s+battery\s+remains\s+pending", re.IGNORECASE),
    re.compile(r"must\s+be\s+re(?:run|measured)\s+on\s+the\s+refreshed", re.IGNORECASE),
    re.compile(r"RELEASE CANDIDATE"),
    re.compile(r"\*\*Status\*{0,2}:?\*{0,2}:?[^\n]*\bcandidate\b", re.IGNORECASE),
]

# `<!-- release-record:KEY -->rendered text<!-- /release-record -->`
# An HTML comment is invisible in rendered Markdown and in a published release
# body. The text between the markers is owned by the record: the gate fails if
# it differs from the rendering below, and --sync rewrites it. The narrative
# release documents quote many dated totals from earlier cuts ("VM parity
# 109/109" in a changelog entry is correct as written), so a release total
# there is identified by its marker rather than guessed from its phrasing.
# The present-tense REGISTERED_DOCS are still graded by phrasing as well.
RECORD_SPAN_RE = re.compile(
    r"<!-- release-record:(?P<key>[a-z-]+) -->(?P<body>.*?)<!-- /release-record -->", re.DOTALL)

def load_release_record() -> dict:
    if not os.path.isfile(RELEASE_RECORD_PATH):
        raise SourceError(f"canonical source not found: {RELEASE_RECORD_PATH}")
    try:
        with open(RELEASE_RECORD_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise SourceError(f"{RELEASE_RECORD_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SourceError(f"{RELEASE_RECORD_PATH} is not a JSON object")
    tag = data.get("tag")
    if not isinstance(tag, str) or not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9]+)?", tag):
        raise SourceError(f"{RELEASE_RECORD_PATH} has no well-formed tag")
    try:
        release_date = datetime.date.fromisoformat(str(data.get("release_date")))
    except ValueError as exc:
        raise SourceError(f"{RELEASE_RECORD_PATH} has no ISO release_date: {exc}") from exc
    status = data.get("status")
    if status not in RELEASE_STATUSES:
        raise SourceError(f"{RELEASE_RECORD_PATH} status must be one of {RELEASE_STATUSES}")
    totals = {}
    for key in ("ctest_total", "vm_parity_total"):
        value = data.get(key)
        if value is not None and (type(value) is not int or value <= 0):
            raise SourceError(f"{RELEASE_RECORD_PATH} {key} must be a positive integer or null")
        totals[key] = value
    return {"tag": tag, "version": tag[1:], "date": release_date, "status": status, **totals}


def iso_date(value: datetime.date) -> str:
    return value.isoformat()


def long_date(value: datetime.date) -> str:
    return f"{MONTHS[value.month - 1]} {value.day}, {value.year}"


def weekday_name(value: datetime.date) -> str:
    return WEEKDAYS[value.weekday()]


def parse_long_date(text: str) -> datetime.date | None:
    match = re.fullmatch(r"([A-Z][a-z]+)\s+([0-9]{1,2}),\s+([0-9]{4})", text.strip())
    if not match or match.group(1) not in MONTHS:
        return None
    try:
        return datetime.date(int(match.group(3)), MONTHS.index(match.group(1)) + 1, int(match.group(2)))
    except ValueError:
        return None


def parse_day_first_date(text: str) -> datetime.date | None:
    match = re.fullmatch(r"([0-9]{1,2})\s+([A-Z][a-z]+)\s+([0-9]{4})", text.strip())
    if not match or match.group(2) not in MONTHS:
        return None
    try:
        return datetime.date(int(match.group(3)), MONTHS.index(match.group(2)) + 1, int(match.group(1)))
    except ValueError:
        return None


def render_record_span(key: str, record: dict) -> str | None:
    """Text a `release-record:KEY` span must carry. None for an unknown key."""
    ctest, parity = record["ctest_total"], record["vm_parity_total"]
    if key == "ctest":
        return f"CTest **{ctest:,}/{ctest:,}**" if ctest else "the full CTest suite"
    if key == "ctest-cell":
        return f"{ctest:,}/{ctest:,} tests" if ctest else "every registered test"
    if key == "vm-parity":
        return (f"VM parity differential **{parity:,}/{parity:,}**" if parity
                else "the VM parity differential")
    if key == "vm-parity-figure":
        return f"**{parity:,}/{parity:,}**" if parity else "green"
    return None


def release_scope(doc_rel: str, text: str, record: dict) -> tuple[int, int]:
    """[start, end) of the part of a document that describes the current release.

    CHANGELOG.md: the current version's section. RELEASE_NOTES.md and
    ANNOUNCEMENT.md: everything above the first horizontal rule, below which
    earlier releases are archived verbatim. Every other document: all of it.
    """
    base = os.path.basename(doc_rel)
    if base == "CHANGELOG.md":
        head = re.search(r"^## \[%s\]" % re.escape(record["version"]), text, re.MULTILINE)
        if not head:
            return (0, 0)
        tail = re.search(r"^## \[", text[head.end():], re.MULTILINE)
        return (head.start(), head.end() + tail.start() if tail else len(text))
    if base in ("RELEASE_NOTES.md", "ANNOUNCEMENT.md"):
        cut = text.find("\n---\n")
        return (0, cut if cut >= 0 else len(text))
    return (0, len(text))


def _release_date_claims(scope: str, record: dict) -> list[dict]:
    """Every statement of the current release's date found in `scope`.

    Each claim: start/end offsets into scope, the stated date (or None if it
    could not be read), the stated weekday (or None), and the replacement text
    that would make it agree with the record.
    """
    tag, version, date = re.escape(record["tag"]), re.escape(record["version"]), record["date"]
    claims: list[dict] = []

    def record_claim(match: re.Match, group: str | int, stated: datetime.date | None,
            weekday: str | None, replacement: str) -> None:
        claims.append({"start": match.start(group), "end": match.end(group), "stated": stated,
                       "weekday": weekday, "replacement": replacement,
                       "snippet": match.group(0)[:120]})

    def iso(token: str) -> datetime.date | None:
        try:
            return datetime.date.fromisoformat(token)
        except ValueError:
            return None

    # Keep-a-Changelog heading: `## [1.3.5-evolve] - 2026-09-22`
    for m in re.finditer(r"\[%s\] - (%s)" % (version, _ISO_DATE), scope):
        record_claim(m, 1, iso(m.group(1)), None, iso_date(date))
    # `Release date: Tuesday, September 22, 2026`, with any bold/colon layout.
    for m in re.finditer(r"[Rr]elease date\*{0,2}:?\*{0,2}\s+(?P<whole>%s)" % _LONG_DATE, scope):
        record_claim(m, "whole", parse_long_date(m.group("long")), m.group("weekday"),
            f"{weekday_name(date)}, {long_date(date)}")
    # `| Release date | 22 September 2026 |`, the press-sheet fact table.
    for m in re.finditer(r"[Rr]elease date\W{1,6}%s" % _DAY_FIRST_DATE, scope):
        record_claim(m, "dayfirst", parse_day_first_date(m.group("dayfirst")), None,
            f"{date.day} {MONTHS[date.month - 1]} {date.year}")
    # `<tag> ... shipped|released [on] <date>` with no other version in between.
    verb = r"(?:SHIPPED|[Ss]hipped|[Ss]hips|[Rr]eleased)(?:\s+on)?\s+"
    gap = r"(?:(?!v[0-9]+\.[0-9]+)[^\n]){0,200}?"
    for m in re.finditer(tag + gap + verb + r"(?P<iso>%s)" % _ISO_DATE, scope):
        record_claim(m, "iso", iso(m.group("iso")), None, iso_date(date))
    # `<tag> (2026-09-22)`, the form a "last shipped release" line uses.
    for m in re.finditer(tag + r"\*{0,2}`?\s+\((?P<iso>%s)" % _ISO_DATE, scope):
        record_claim(m, "iso", iso(m.group("iso")), None, iso_date(date))
    for m in re.finditer(tag + gap + verb + r"(?P<whole>%s)" % _LONG_DATE, scope):
        replacement = long_date(date)
        if m.group("weekday"):
            replacement = f"{weekday_name(date)}, {replacement}"
        record_claim(m, "whole", parse_long_date(m.group("long")), m.group("weekday"), replacement)
    return claims


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def check_release_doc(doc_rel: str, record: dict, *, mirror: bool = False) -> tuple[list[dict], list[dict]]:
    """Grade one release-facing document. Returns (findings, edits)."""
    path = os.path.join(REPO_ROOT, doc_rel)
    if not os.path.isfile(path):
        return ([{"doc": doc_rel, "quantity": "registry", "line": 0, "found": None,
                  "expected": None, "snippet": "registered release doc does not exist"}], [])
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    if mirror:
        # Generated HTML: grade the visible text. Offsets are meaningless after
        # tag stripping, so findings carry line 0 and no edit is ever proposed.
        scope_text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)))
        start = 0
    else:
        start, end = release_scope(doc_rel, text, record)
        scope_text = text[start:end]

    findings: list[dict] = []
    edits: list[dict] = []

    def finding(quantity: str, offset: int, found, expected, snippet: str) -> None:
        findings.append({"doc": doc_rel, "quantity": quantity,
                         "line": 0 if mirror else _line_of(text, start + offset),
                         "found": found, "expected": expected, "snippet": snippet})

    # 1. release date
    claims = _release_date_claims(scope_text, record)
    for claim in claims:
        stated, wrong = claim["stated"], False
        if stated != record["date"]:
            finding("release_date", claim["start"], str(stated), iso_date(record["date"]), claim["snippet"])
            wrong = True
        elif claim["weekday"] and claim["weekday"] != weekday_name(record["date"]):
            finding("release_weekday", claim["start"], claim["weekday"],
                    weekday_name(record["date"]), claim["snippet"])
            wrong = True
        if wrong and not mirror:
            edits.append({"doc": doc_rel, "start": start + claim["start"],
                          "end": start + claim["end"], "text": claim["replacement"]})
    if doc_rel in RELEASE_DATE_ANCHORS and not claims:
        finding("release_date", 0, None, iso_date(record["date"]),
                "document must state the release date of " + record["tag"])

    # 2. status: a table row or heading for the tag carries the record's label,
    #    and no pre-release wording survives a SHIPPED record.
    if not mirror:
        tag = re.escape(record["tag"])
        for m in re.finditer(r"^\|\s*\*{0,2}%s\*{0,2}\s*\|.*$" % tag, scope_text, re.MULTILINE):
            row = m.group(0)
            if record["status"] not in row or iso_date(record["date"]) not in row:
                finding("release_status", m.start(), row[:80],
                        f"{record['status']} {iso_date(record['date'])}", row[:120])
        for m in re.finditer(r"^## %s\b.* - ([A-Z][A-Z ]+)$" % tag, scope_text, re.MULTILINE):
            if m.group(1) != record["status"]:
                finding("release_status", m.start(1), m.group(1), record["status"], m.group(0)[:120])
                edits.append({"doc": doc_rel, "start": start + m.start(1),
                              "end": start + m.end(1), "text": record["status"]})
    if record["status"] == RELEASE_STATUS_SHIPPED:
        for pattern in PRE_RELEASE_PHRASES:
            for m in pattern.finditer(scope_text):
                finding("release_status", m.start(), m.group(0), "no pre-release wording",
                        scope_text[max(0, m.start() - 40):m.end() + 40].replace("\n", " "))

    # 3. record-owned spans and the evidence totals
    if not mirror:
        for m in RECORD_SPAN_RE.finditer(scope_text):
            expected = render_record_span(m.group("key"), record)
            if expected is None:
                finding("record_span", m.start(), m.group("key"), "a known release-record key", m.group(0)[:120])
            elif m.group("body") != expected:
                finding("record_span", m.start("body"), m.group("body"), expected, m.group(0)[:160])
                edits.append({"doc": doc_rel, "start": start + m.start("body"),
                              "end": start + m.end("body"), "text": expected})
    return findings, edits


def site_mirrors() -> list[str]:
    """Generated HTML pages whose Markdown source is a registered release doc."""
    sources = {os.path.splitext(os.path.basename(d))[0].lower() for d in RELEASE_DOCS}
    root = os.path.join(REPO_ROOT, SITE_MIRROR_DIR)
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(SITE_MIRROR_DIR, name).replace(os.sep, "/")
                  for name in os.listdir(root)
                  if name.endswith(".html") and os.path.splitext(name)[0] in sources)


def apply_edits(edits: list[dict]) -> list[str]:
    """Rewrite graded claims from the record. Returns the documents changed."""
    changed = []
    by_doc: dict[str, list[dict]] = {}
    for edit in edits:
        by_doc.setdefault(edit["doc"], []).append(edit)
    for doc_rel, doc_edits in sorted(by_doc.items()):
        path = os.path.join(REPO_ROOT, doc_rel)
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        seen = set()
        for edit in sorted(doc_edits, key=lambda e: e["start"], reverse=True):
            if (edit["start"], edit["end"]) in seen:
                continue
            seen.add((edit["start"], edit["end"]))
            text = text[:edit["start"]] + edit["text"] + text[edit["end"]:]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        changed.append(doc_rel)
    return changed



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


def run_gate(ctest_log: str | None, parity_log: str | None,
             require_complete: bool = False) -> dict:
    surface_total = load_canonical_surface_total()
    builtins_total = load_canonical_builtins_total()
    record = load_release_record()

    log_ctest = parse_ctest_log(ctest_log) if ctest_log else None
    log_parity = parse_parity_log(parity_log) if parity_log else None

    all_findings: list[dict] = []
    all_notes: list[str] = []
    all_edits: list[dict] = []

    # A supplied log is evidence; the record is the committed copy of it. When
    # both exist they must agree, so a record can never outlive its evidence.
    for key, measured, label in (("ctest_total", log_ctest, ctest_log),
                                 ("vm_parity_total", log_parity, parity_log)):
        if measured is not None and record[key] is not None and measured != record[key]:
            all_findings.append({
                "doc": os.path.relpath(RELEASE_RECORD_PATH, REPO_ROOT), "quantity": key, "line": 0,
                "found": record[key], "expected": measured,
                "snippet": f"release record disagrees with the evidence log {label}"})
        if record[key] is None:
            message = (f"release record has no {key}; documents render that claim without a number")
            if require_complete:
                all_findings.append({
                    "doc": os.path.relpath(RELEASE_RECORD_PATH, REPO_ROOT), "quantity": key,
                    "line": 0, "found": None, "expected": "a recorded total",
                    "snippet": message})
            else:
                all_notes.append(message)

    ctest_total = log_ctest if log_ctest is not None else record["ctest_total"]
    parity_total = log_parity if log_parity is not None else record["vm_parity_total"]

    for doc_rel in RELEASE_DOCS:
        findings, edits = check_release_doc(doc_rel, record)
        all_findings.extend(findings)
        all_edits.extend(edits)
    for doc_rel in site_mirrors():
        findings, _ = check_release_doc(doc_rel, record, mirror=True)
        for item in findings:
            item["snippet"] += " (generated page: rerun scripts/build-site-content.sh)"
        all_findings.extend(findings)
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
        "release": {"tag": record["tag"], "date": iso_date(record["date"]),
                    "status": record["status"]},
        "edits": all_edits,
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
        global RELEASE_RECORD_PATH, RELEASE_DOCS, RELEASE_DATE_ANCHORS, SITE_MIRROR_DIR
        real_policy, real_manifest, real_docs = COVERAGE_POLICY_PATH, LANGUAGE_SURFACE_PATH, REGISTERED_DOCS
        real_release = (RELEASE_RECORD_PATH, RELEASE_DOCS, RELEASE_DATE_ANCHORS, SITE_MIRROR_DIR)
        COVERAGE_POLICY_PATH, LANGUAGE_SURFACE_PATH = policy_path, manifest_path
        # The count fixtures below must not depend on the real release docs.
        RELEASE_RECORD_PATH = os.path.join(tmp_dir, "release_record.json")
        RELEASE_DOCS, RELEASE_DATE_ANCHORS = [], []
        SITE_MIRROR_DIR = os.path.relpath(os.path.join(tmp_dir, "no-site"), REPO_ROOT)

        def write_record(**overrides) -> None:
            record = {"tag": "v9.8.7-test", "release_date": "2031-03-04", "status": "SHIPPED",
                      "ctest_total": None, "vm_parity_total": None}
            record.update(overrides)
            with open(RELEASE_RECORD_PATH, "w", encoding="utf-8") as handle:
                json.dump(record, handle)

        write_record()

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

        COVERAGE_POLICY_PATH, LANGUAGE_SURFACE_PATH = policy_path, manifest_path
        REGISTERED_DOCS = []

        # ── release record: date, status, record-owned spans, totals ──
        # 2031-03-04 is a Tuesday.
        release_cases = [
            ("green_release_date_and_status",
             "**Release date:** Tuesday, March 4, 2031.\n\n"
             "| **v9.8.7-test** | 2031-03-04 | Theme | **SHIPPED 2031-03-04.** text |\n", {}, True),
            ("red_stale_release_date",
             "**Release date:** Tuesday, February 25, 2031.\n", {}, False),
            ("red_wrong_weekday",
             "**Release date:** Monday, March 4, 2031.\n", {}, False),
            ("red_changelog_heading_date",
             "## [9.8.7-test] - 2031-02-25\n", {}, False),
            ("red_shipped_date_after_tag",
             "v9.8.7-test SHIPPED 2031-02-25 with a parser.\n"
             "**Release date:** Tuesday, March 4, 2031.\n", {}, False),
            ("red_day_first_fact_table_date",
             "**Release date:** Tuesday, March 4, 2031.\n| Release date | 25 February 2031 |\n", {}, False),
            ("green_day_first_fact_table_date",
             "| Release date | 4 March 2031 |\n", {}, True),
            ("red_parenthesized_date_after_tag",
             "**Last shipped release**: v9.8.7-test (2031-02-25).\n"
             "**Release date:** Tuesday, March 4, 2031.\n", {}, False),
            ("green_other_release_date_is_not_ours",
             "v9.8.7-test follows v9.8.6-test, SHIPPED 2031-01-01.\n"
             "**Release date:** Tuesday, March 4, 2031.\n", {}, True),
            ("red_pre_release_wording_after_shipped",
             "**Release date:** Tuesday, March 4, 2031.\n"
             "The refreshed candidate awaits its battery; verification is pending.\n", {}, False),
            ("green_pre_release_wording_while_candidate",
             "**Release date:** Tuesday, March 4, 2031.\n"
             "The refreshed candidate awaits its battery.\n", {"status": "RELEASE CANDIDATE"}, True),
            ("red_status_row_not_shipped",
             "**Release date:** Tuesday, March 4, 2031.\n"
             "| **v9.8.7-test** | 2031-03-04 | Theme | In flight |\n", {}, False),
            ("red_missing_anchor_date",
             "This document never says when v9.8.7-test shipped.\n", {}, False),
            ("green_span_without_total",
             "**Release date:** Tuesday, March 4, 2031.\n"
             "<!-- release-record:ctest -->the full CTest suite<!-- /release-record -->\n",
             {}, True),
            ("red_span_stale_after_total_recorded",
             "**Release date:** Tuesday, March 4, 2031.\n"
             "<!-- release-record:ctest -->the full CTest suite<!-- /release-record -->\n",
             {"ctest_total": 1234}, False),
            ("green_span_with_total",
             "**Release date:** Tuesday, March 4, 2031.\n"
             "<!-- release-record:ctest -->CTest **1,234/1,234**<!-- /release-record -->\n",
             {"ctest_total": 1234}, True),
            ("red_unknown_span_key",
             "**Release date:** Tuesday, March 4, 2031.\n"
             "<!-- release-record:nonsense -->x<!-- /release-record -->\n", {}, False),
            ("red_parity_total_disagrees_with_record",
             "**Release date:** Tuesday, March 4, 2031.\n<!-- release-record:vm-parity -->"
             "VM parity differential **338/338**<!-- /release-record -->.\n",
             {"vm_parity_total": 340}, False),
            ("green_parity_total_matches_record",
             "**Release date:** Tuesday, March 4, 2031.\n<!-- release-record:vm-parity -->"
             "VM parity differential **340/340**<!-- /release-record -->.\n",
             {"vm_parity_total": 340}, True),
            ("green_dated_total_from_an_earlier_cut_is_not_ours",
             "**Release date:** Tuesday, March 4, 2031.\nAn earlier entry: VM parity 109/109.\n",
             {"vm_parity_total": 340}, True),
        ]
        release_doc = os.path.join(tmp_dir, "RELEASE_DOC.md")
        release_rel = os.path.relpath(release_doc, REPO_ROOT)
        RELEASE_DOCS, RELEASE_DATE_ANCHORS = [release_rel], [release_rel]
        for name, doc_text, overrides, expect_pass in release_cases:
            write_record(**overrides)
            with open(release_doc, "w", encoding="utf-8") as handle:
                handle.write(doc_text)
            result = run_gate(ctest_log=None, parity_log=None)
            ok = result["passed"] == expect_pass
            all_ok = all_ok and ok
            detail = "PASS" if result["passed"] else "; ".join(
                f"{f['quantity']}={f['found']!r} (expected {f['expected']!r})"
                for f in result["findings"])
            print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] {name}: expected passed={expect_pass}, "
                  f"got passed={result['passed']}")
            print(f"           {detail}")

        # --sync is the repair for every graded claim: after applying the
        # edits the gate proposes, the same document must pass.
        write_record(ctest_total=1234, vm_parity_total=340)
        with open(release_doc, "w", encoding="utf-8") as handle:
            handle.write("## [9.8.7-test] - 2031-02-25\n"
                         "**Release date:** Monday, February 25, 2031.\n"
                         "<!-- release-record:ctest -->the full CTest suite<!-- /release-record -->\n"
                         "<!-- release-record:vm-parity-figure -->**338/338**<!-- /release-record -->\n")
        before = run_gate(ctest_log=None, parity_log=None)
        apply_edits(before["edits"])
        after = run_gate(ctest_log=None, parity_log=None)
        ok = (not before["passed"]) and after["passed"]
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] sync_repairs_every_graded_claim: "
              f"before={before['passed']}, after={after['passed']}")

        # A null total is a note by default and a failure under --require-complete.
        write_record()
        with open(release_doc, "w", encoding="utf-8") as handle:
            handle.write("**Release date:** Tuesday, March 4, 2031.\n")
        relaxed = run_gate(ctest_log=None, parity_log=None)
        strict = run_gate(ctest_log=None, parity_log=None, require_complete=True)
        ok = relaxed["passed"] and not strict["passed"]
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] null_total_fails_only_when_completeness_required: "
              f"relaxed={relaxed['passed']}, strict={strict['passed']}")

        # The record may not outlive its evidence: a log that disagrees fails.
        write_record(ctest_total=1234)
        disagreeing_log = os.path.join(tmp_dir, "ctest_disagrees.log")
        with open(disagreeing_log, "w", encoding="utf-8") as handle:
            handle.write("100% tests passed out of 1240\n")
        result = run_gate(ctest_log=disagreeing_log, parity_log=None)
        ok = result["passed"] is False
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] record_disagreeing_with_evidence_log_fails: "
              f"passed={result['passed']}")

        # A generated HTML mirror is graded as text.
        mirror_dir = os.path.join(tmp_dir, "site-content")
        os.makedirs(mirror_dir)
        SITE_MIRROR_DIR = os.path.relpath(mirror_dir, REPO_ROOT)
        write_record()
        with open(os.path.join(mirror_dir, "release_doc.html"), "w", encoding="utf-8") as handle:
            handle.write("<p><strong>Release date:</strong> Tuesday,\nFebruary 25, 2031.</p>\n")
        result = run_gate(ctest_log=None, parity_log=None)
        ok = result["passed"] is False and any(f["doc"].endswith(".html") for f in result["findings"])
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] stale_generated_mirror_fails: "
              f"passed={result['passed']}")

        COVERAGE_POLICY_PATH, LANGUAGE_SURFACE_PATH, REGISTERED_DOCS = real_policy, real_manifest, real_docs
        RELEASE_RECORD_PATH, RELEASE_DOCS, RELEASE_DATE_ANCHORS, SITE_MIRROR_DIR = real_release

    if all_ok:
        print("self-test: PASS — matching claims pass, any stale claim in any registered "
              "doc fails, a missing registered doc fails, ctest/parity claims are graded "
              "against a supplied log or the release record, release date, status and "
              "record-owned spans are held to the release record and repaired by --sync, and "
              "an unreadable canonical source is distinguishable from a clean pass")
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
    parser.add_argument("--sync", action="store_true",
                         help="rewrite every graded release-record claim (release date, status "
                              "label, record-owned spans, CTest and VM-parity totals) from "
                              "tests/coverage/release_record.json, then grade")
    parser.add_argument("--require-complete", action="store_true",
                         help="fail while the release record still has a null total")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    try:
        if args.sync:
            changed = apply_edits(run_gate(args.ctest_log, args.parity_log)["edits"])
            for doc_rel in changed:
                print(f"synced from the release record: {doc_rel}")
        result = run_gate(args.ctest_log, args.parity_log, args.require_complete)
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
                   f"builtins_total={result['builtins_total']} "
                   f"release={result['release']['tag']}@{result['release']['date']} "
                   f"— every registered doc agrees")
    else:
        snippet = f"{len(result['findings'])} mismatch(es): " + "; ".join(
            f"{f['doc']}:{f['line']} {f['quantity']}={f['found']} (expected {f['expected']})"
            for f in result["findings"][:5]
        )

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status,
                          **{k: v for k, v in result.items() if k != "edits"}}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        release = result["release"]
        print(f"  canonical release        : {release['tag']} {release['status']} {release['date']}"
              f" (tests/coverage/release_record.json)")
        print(f"  canonical surface_total  : {result['surface_total']}"
              f" (tests/coverage/coverage_policy.json)")
        print(f"  canonical builtins_total : {result['builtins_total']}"
              f" (tests/coverage/language_surface.json)")
        record_rel = "tests/coverage/release_record.json"
        if result["ctest_total"] is not None:
            print(f"  canonical ctest_total    : {result['ctest_total']} ({args.ctest_log or record_rel})")
        if result["parity_total"] is not None:
            print(f"  canonical parity_total   : {result['parity_total']} ({args.parity_log or record_rel})")
        for note in result["notes"]:
            print(f"  note: {note}")
        if result["findings"]:
            print("  MISMATCHES:")
            for f in result["findings"]:
                print(f"    - {f['doc']}:{f['line']} [{f['quantity']}] "
                      f"found {f['found']!r}, expected {f['expected']!r} — {f['snippet']!r}")
        else:
            print(f"  all {len(REGISTERED_DOCS)} registered docs agree with the machine sources")
            print(f"  all {len(RELEASE_DOCS)} release docs and {len(site_mirrors())} generated pages "
                  f"agree with the release record")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
