#!/usr/bin/env python3
"""Release gate: the changelog must account for every pull request merged in
the release range, or the gate fails.

Motivating incident (v1.3.5-evolve): the documentation sweep for the release
ran on 2026-09-11, and dozens of pull requests merged after it. The changelog
described few of them, and nothing noticed, because no gate connected "a pull
request merged" with "the release section accounts for it". Every other
release fact had a machine source by then -- the surface counts, the release
date, the evidence totals -- but the list of what actually changed was still
reconciled by hand, once, and a one-shot reconciliation is wrong the moment
the next pull request lands.

The rule. Every merged pull request number in the release range must have
exactly one home:

  1. a `#N` reference inside the changelog section for the release
     (`## [<version>]`, where the version is the release record's tag without
     its leading `v`) or inside `## [Unreleased]`; or
  2. an entry in the checked-in ledger
     tests/coverage/changelog_no_user_facing_change.json, with a class from a
     closed set and a specific reason why a user cannot observe the change.

Anything else is an unaccounted pull request and the gate fails, printing the
number, its commit subject and the two ways to resolve it. A reference in an
OLDER release section does not count: that section describes a different
release.

Sources (nothing here is a hand-copied number):
    tests/coverage/release_record.json -> tag, previous_tag
        the single source for release facts. The range is
        `<previous_tag>..HEAD`.
    git history -> the pull request numbers
        derived from commit subjects only, with no network access: a squash
        subject ending in `(#N)` and a merge subject
        `Merge pull request #N ...`. A `#N` that appears mid-subject
        ("rework of #391 (#412)") is a cross-reference, not the merged
        pull request, and is ignored.
    CHANGELOG.md -> the `#N` references of the two graded sections
    tests/coverage/changelog_no_user_facing_change.json -> the ledger

The ledger is graded as strictly as the changelog, because an unexamined
ledger is just a second place to hide a pull request. An entry fails when its
class is not in the closed set, when its reason is shorter than 25 characters
or is a generic phrase, when its pull request is listed twice, when its pull
request is not in the range (stale), when its pull request is ALSO referenced
in the graded changelog sections (redundant: one home per pull request), when
the ledger's `release` differs from the record's tag (a ledger carried over
from the previous release), or when the entries are not sorted by pull
request number (so diffs stay reviewable).

The pull request under review. A pull request's own number is not in the
history until it merges, so a pull request that accounts for itself in advance
-- the right moment to do it -- would see its own ledger entry graded stale.
`--pending-pr N` names a number that is about to join the range: a ledger
entry for it is not stale, and with `--require-pending` it must already have a
home, exactly like a merged pull request. CI passes the number of the pull
request it is grading; a push or merge-queue run has none and grades the
history alone.

Fail closed. The gate needs the full history and the previous release tag. If
git is unavailable, the previous tag does not resolve, the checkout is shallow
or the tag is not an ancestor of the graded commit, the range cannot be walked
and the gate exits NO_DATA rather than passing over an empty list. A range
that yields zero pull requests fails for the same reason: a release with no
merged pull requests is far less likely than a range that was computed wrong.

Modes / exit status
    PASS      0   the range was walked and every pull request has one home.
    FAIL      1   at least one pull request is unaccounted, or the ledger
                  breaks one of its rules.
    NO_DATA   2   the release record, the changelog, the ledger or the git
                  history could not be read -- nothing was verified. Distinct
                  from PASS so a caller cannot mistake "we never checked" for
                  "we checked and it is fine".

Usage
    python3 scripts/check_changelog_completeness.py
    python3 scripts/check_changelog_completeness.py --no-trace
    python3 scripts/check_changelog_completeness.py --format json
    python3 scripts/check_changelog_completeness.py --range v1.3.4-evolve..HEAD
    python3 scripts/check_changelog_completeness.py --pending-pr 712
    python3 scripts/check_changelog_completeness.py --pending-pr 712 --require-pending
    python3 scripts/check_changelog_completeness.py --self-test

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "changelog_completeness_gate.jsonl"
PROBE_ID = "changelog_accounts_for_every_pr"

RELEASE_RECORD_PATH = os.path.join(REPO_ROOT, "tests", "coverage", "release_record.json")
CHANGELOG_PATH = os.path.join(REPO_ROOT, "CHANGELOG.md")
LEDGER_PATH = os.path.join(REPO_ROOT, "tests", "coverage", "changelog_no_user_facing_change.json")
LEDGER_REL = "tests/coverage/changelog_no_user_facing_change.json"

LEDGER_SCHEMA = "eshkol.changelog-no-user-facing-change.v1"

# The closed set of reasons a merged pull request may have no changelog entry.
# Adding a class is a deliberate edit here, reviewed like any other gate change.
LEDGER_CLASSES = {
    "ci": "workflow, runner or lane wiring; nothing in the shipped tree changes",
    "test-only": "adds or repairs tests, fixtures or harnesses only",
    "docs-only": "refreshes generated or internal documentation claims; no new user guidance",
    "release-machinery": "release scripts, evidence recipes, readiness or packaging-gate plumbing",
    "build-internal": "internal refactor or build plumbing with no behaviour change",
    "merge-integration": "integration or branch-synchronisation pull request; its content is "
                         "accounted for by the pull requests it carries",
    "reverted-or-superseded": "reverted, or wholly replaced by a later pull request, inside the "
                              "same release range",
}

MIN_REASON_LENGTH = 25
# Compared after lowercasing and dropping everything but letters, digits and
# single spaces, so punctuation and hyphenation do not defeat the check.
GENERIC_REASONS = {
    "no user facing change",
    "no user facing changes",
    "no user facing change in this pr",
    "no user facing change in this pull request",
    "this pr has no user facing change",
    "this pull request has no user facing change",
    "there is no user facing change",
    "not a user facing change",
    "internal change with no user facing effect",
    "internal change only no user facing change",
    "internal only no user facing change",
    "nothing user facing changed here",
    "does not affect users in any way",
    "no changelog entry needed for this",
    "no changelog entry is needed here",
    "see the pull request description",
}

TAG_RE = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9]+)?")
SQUASH_SUBJECT_RE = re.compile(r"\(#([0-9]+)\)\s*$")
MERGE_SUBJECT_RE = re.compile(r"^Merge pull request #([0-9]+)\b")
REFERENCE_RE = re.compile(r"(?<![0-9A-Za-z&])#([0-9]+)\b")
SECTION_HEADING_RE = re.compile(r"^## \[(?P<name>[^\]]+)\]", re.MULTILINE)


class SourceError(Exception):
    """A source could not be read, so nothing was verified (gate fails closed)."""


# ───────────────────────── sources ─────────────────────────

def load_release_record(path: str) -> dict:
    if not os.path.isfile(path):
        raise SourceError(f"release record not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise SourceError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SourceError(f"{path} is not a JSON object")
    return validate_release_record(data, path)


def validate_release_record(data: dict, path: str = "release record") -> dict:
    tag = data.get("tag")
    if not isinstance(tag, str) or not TAG_RE.fullmatch(tag):
        raise SourceError(f"{path} has no well-formed tag")
    previous = data.get("previous_tag")
    if not isinstance(previous, str) or not TAG_RE.fullmatch(previous):
        raise SourceError(
            f"{path} has no well-formed previous_tag; add the tag of the release before {tag} "
            f"(for example \"previous_tag\": \"v1.3.4-evolve\") so the release range can be derived")
    if previous == tag:
        raise SourceError(f"{path} previous_tag equals tag ({tag}); the release range would be empty")
    return {"tag": tag, "version": tag[1:], "previous_tag": previous}


def read_text(path: str, what: str) -> str:
    if not os.path.isfile(path):
        raise SourceError(f"{what} not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def load_ledger(path: str) -> dict:
    text = read_text(path, "ledger")
    try:
        data = json.loads(text)
    except Exception as exc:
        raise SourceError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SourceError(f"{path} is not a JSON object")
    return data


def run_git(args: list[str]) -> tuple[int, str, str]:
    """Run git in the repository root. Raises OSError when git is not installed."""
    proc = subprocess.run(["git", "-C", REPO_ROOT, *args], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout, proc.stderr


FETCH_HINT = ("fetch the full history and the release tags, then run the gate again: "
              "`git fetch --unshallow --tags` in a shallow clone, `git fetch --tags` otherwise "
              "(in GitHub Actions, check out with `fetch-depth: 0`)")


def collect_range_subjects(range_spec: str, runner=run_git) -> list[tuple[str, str]]:
    """Return (sha, subject) for every commit in the range, newest first.

    Fails closed: any condition under which the range cannot be trusted raises
    SourceError instead of returning a short or empty list.
    """
    if ".." not in range_spec or "..." in range_spec:
        raise SourceError(f"range {range_spec!r} is not of the form <previous_tag>..<commit>")
    base, _, tip = range_spec.partition("..")
    tip = tip or "HEAD"
    if not base:
        raise SourceError(f"range {range_spec!r} has no starting tag")

    def git(args: list[str]) -> tuple[int, str, str]:
        try:
            return runner(args)
        except OSError as exc:
            raise SourceError(f"git is unavailable ({exc}); the release range cannot be walked. "
                              f"Run the gate in a git checkout with git installed") from exc

    code, out, err = git(["rev-parse", "--is-shallow-repository"])
    if code != 0:
        raise SourceError(f"not a usable git checkout ({err.strip() or 'git rev-parse failed'}); "
                          f"the release range cannot be walked")
    if out.strip() == "true":
        raise SourceError(f"the checkout is shallow, so {range_spec} cannot be walked completely; "
                          + FETCH_HINT)
    for ref, what in ((base, "previous release tag"), (tip, "graded commit")):
        code, _, _ = git(["rev-parse", "--verify", "--quiet", ref + "^{commit}"])
        if code != 0:
            raise SourceError(f"{what} {ref!r} does not resolve to a commit; " + FETCH_HINT)
    code, _, _ = git(["merge-base", "--is-ancestor", base, tip])
    if code != 0:
        raise SourceError(f"{base} is not an ancestor of {tip}, so {range_spec} is not the list of "
                          f"commits merged since that release; check previous_tag in "
                          f"tests/coverage/release_record.json, or " + FETCH_HINT)
    code, out, err = git(["log", "--format=%H%x09%s", f"{base}..{tip}"])
    if code != 0:
        raise SourceError(f"git log {range_spec} failed: {err.strip()}")
    commits = []
    for line in out.splitlines():
        sha, _, subject = line.partition("\t")
        if sha:
            commits.append((sha, subject))
    return commits


# ───────────────────────── pure grading ─────────────────────────

def parse_pr_number(subject: str) -> int | None:
    """The pull request a commit subject merges, or None.

    Only the two shapes the forge itself writes are recognised; a `#N`
    elsewhere in the subject is a cross-reference.
    """
    match = MERGE_SUBJECT_RE.match(subject) or SQUASH_SUBJECT_RE.search(subject)
    return int(match.group(1)) if match else None


def prs_in_commits(commits: list[tuple[str, str]]) -> dict[int, dict]:
    """Map each merged pull request number to the newest commit that carries it."""
    found: dict[int, dict] = {}
    for sha, subject in commits:
        number = parse_pr_number(subject)
        if number is not None and number not in found:
            found[number] = {"sha": sha, "subject": subject}
    return found


def changelog_sections(text: str) -> dict[str, str]:
    headings = list(SECTION_HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        # A duplicated heading is graded as one section rather than letting
        # the second copy shadow the first.
        sections[heading.group("name")] = sections.get(heading.group("name"), "") + text[heading.start():end]
    return sections


def changelog_references(text: str, version: str) -> tuple[dict[int, str], list[str]]:
    """`#N` references of the release section and of [Unreleased], with the
    section each was found in, plus problems with the changelog itself."""
    sections = changelog_sections(text)
    problems = []
    if version not in sections:
        problems.append(f"CHANGELOG has no `## [{version}]` section; the release record's tag is "
                        f"v{version}, so the changelog must have a section for it")
    references: dict[int, str] = {}
    for name in (version, "Unreleased"):
        for match in REFERENCE_RE.finditer(sections.get(name, "")):
            references.setdefault(int(match.group(1)), name)
    return references, problems


def normalise_reason(reason: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", reason.lower()).split())


def validate_ledger(ledger: dict, tag: str, range_prs: set[int],
                    references: dict[int, str]) -> tuple[set[int], list[str]]:
    """Return the ledgered pull request numbers and every rule the ledger breaks."""
    problems: list[str] = []
    if ledger.get("schema") != LEDGER_SCHEMA:
        problems.append(f"ledger schema is {ledger.get('schema')!r}, expected {LEDGER_SCHEMA!r}")
    if ledger.get("release") != tag:
        problems.append(f"ledger release is {ledger.get('release')!r} but the release record's tag is "
                        f"{tag!r}; a ledger is per release -- set \"release\": \"{tag}\" and delete the "
                        f"entries the gate then reports as stale")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        problems.append("ledger has no \"entries\" list")
        return set(), problems

    seen: set[int] = set()
    order: list[int] = []
    for index, entry in enumerate(entries):
        where = f"entries[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} is not an object")
            continue
        number = entry.get("pr")
        if type(number) is not int or number <= 0:
            problems.append(f"{where} has no positive integer \"pr\"")
            continue
        where = f"#{number}"
        order.append(number)
        if number in seen:
            problems.append(f"{where} is listed more than once; keep one entry")
            continue
        seen.add(number)
        unknown = sorted(set(entry) - {"pr", "class", "reason"})
        if unknown:
            problems.append(f"{where} has unknown key(s) {', '.join(unknown)}; an entry is "
                            f"pr, class and reason")
        klass = entry.get("class")
        if klass not in LEDGER_CLASSES:
            problems.append(f"{where} has class {klass!r}; use one of: "
                            + ", ".join(sorted(LEDGER_CLASSES)))
        reason = entry.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < MIN_REASON_LENGTH:
            problems.append(f"{where} needs a specific reason of at least {MIN_REASON_LENGTH} "
                            f"characters saying what the pull request changed and why a user "
                            f"cannot observe it")
        elif normalise_reason(reason) in GENERIC_REASONS or normalise_reason(reason) == normalise_reason(str(klass)):
            problems.append(f"{where} has a generic reason ({reason.strip()!r}); say what the pull "
                            f"request changed and why a user cannot observe it")
        if number not in range_prs:
            problems.append(f"{where} is stale: no commit for it in the release range (and it is not "
                            f"the pull request under review, --pending-pr); delete the entry")
        if number in references:
            problems.append(f"{where} is redundant: it is also referenced in the changelog's "
                            f"[{references[number]}] section; one home per pull request -- delete the "
                            f"ledger entry, or remove the reference if the change is not user-facing")
    if order != sorted(order):
        problems.append("ledger entries are not sorted by pull request number; sort them so diffs "
                        "stay reviewable")
    return seen, problems


def grade(commits: list[tuple[str, str]], changelog_text: str, ledger: dict, record: dict,
          range_spec: str, pending: tuple[int, ...] = (), require_pending: bool = False) -> dict:
    prs = prs_in_commits(commits)
    references, problems = changelog_references(changelog_text, record["version"])
    if not prs:
        problems.append(f"the range {range_spec} yields zero merged pull requests ({len(commits)} "
                        f"commit(s) walked); a release range without a single pull request means the "
                        f"range is wrong, not that there is nothing to account for -- check "
                        f"previous_tag in tests/coverage/release_record.json")
    # A pending pull request is not merged yet: its ledger entry is not stale,
    # and it is required to have a home only when the caller says so. The
    # zero-pull-request check above deliberately ignores it.
    allowed = set(prs) | set(pending)
    required = dict(prs)
    if require_pending:
        for number in pending:
            required.setdefault(number, {"sha": "-" * 9, "subject": "(the pull request under review)"})
    ledgered, ledger_problems = validate_ledger(ledger, record["tag"], allowed, references)
    unaccounted = [
        {"pr": number, **required[number]}
        for number in sorted(required)
        if number not in references and number not in ledgered
    ]
    return {
        "release": record["tag"],
        "range": range_spec,
        "commits": len(commits),
        "pull_requests": len(prs),
        "referenced": sum(1 for number in prs if number in references),
        "ledgered": sum(1 for number in prs if number in ledgered and number not in references),
        "unaccounted": unaccounted,
        "problems": problems + ledger_problems,
        "passed": not unaccounted and not problems and not ledger_problems,
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
    """Hermetic: in-memory fixtures only, no git and no network."""
    print("check_changelog_completeness.py self-test:")
    all_ok = True

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal all_ok
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] {name}" + (f": {detail}" if detail else ""))

    # Subject parsing.
    subject_cases = [
        ("squash_subject", "fix(parser): keep line numbers (#412)", 412),
        ("squash_subject_trailing_space", "fix(parser): keep line numbers (#412) ", 412),
        ("merge_subject", "Merge pull request #433 from someone/topic", 433),
        ("mid_subject_reference_is_not_the_pr", "rework of #391 (#412)", 412),
        ("mid_subject_reference_same_number", "rework of #391 (#391)", 391),
        ("reference_only_mid_subject", "follow-up to #391 without a number", None),
        ("parenthesised_pr_word_is_not_a_merge", "docs(api): regenerate after rebase (PR #559)", None),
        ("plain_commit", "docs: refresh stale source line counts", None),
        ("quoted_revert_is_not_a_merge", "Revert \"feat: something (#588)\"", None),
        ("revert_with_its_own_pr", "Revert \"feat: something (#588)\" (#601)", 601),
    ]
    for name, subject, expected in subject_cases:
        got = parse_pr_number(subject)
        check(f"subject_{name}", got == expected, f"{subject!r} -> {got!r}")

    record = validate_release_record({"tag": "v9.8.7-test", "previous_tag": "v9.8.6-test"})
    range_spec = "v9.8.6-test..HEAD"
    commits = [
        ("c" * 40, "feat: third thing (#103)"),
        ("b" * 40, "Merge pull request #102 from someone/second"),
        ("a" * 40, "fix: first thing (#101)"),
        ("9" * 40, "docs: a direct commit with no pull request"),
    ]
    changelog = (
        "# Changelog\n\n## [Unreleased]\n\n## [9.8.7-test] - 2031-03-04\n\n### Added\n"
        "- **Third thing** does a thing. (#103)\n- **Second thing**, see #102.\n\n"
        "## [9.8.6-test] - 2031-01-01\n\n### Fixed\n- **Old thing** (#55)\n"
    )
    good_reason = "Only rewires the nightly lane's cache key; no shipped file changes."

    def ledger(*entries, release="v9.8.7-test", schema=LEDGER_SCHEMA) -> dict:
        return {"schema": schema, "release": release, "entries": list(entries)}

    def entry(number: int, klass: str = "ci", reason: str = good_reason) -> dict:
        return {"pr": number, "class": klass, "reason": reason}

    def run(name: str, expect_pass: bool, *, commits_=None, changelog_=None, ledger_=None,
            expect_unaccounted: list[int] | None = None, expect_problem: str | None = None) -> None:
        result = grade(commits if commits_ is None else commits_,
                       changelog if changelog_ is None else changelog_,
                       ledger(entry(101)) if ledger_ is None else ledger_, record, range_spec)
        ok = result["passed"] is expect_pass
        if expect_unaccounted is not None:
            ok = ok and [item["pr"] for item in result["unaccounted"]] == expect_unaccounted
        if expect_problem is not None:
            ok = ok and any(expect_problem in problem for problem in result["problems"])
        check(name, ok, f"passed={result['passed']} unaccounted="
                        f"{[item['pr'] for item in result['unaccounted']]} "
                        f"problems={len(result['problems'])}")

    run("green_referenced_and_ledgered", True, expect_unaccounted=[])
    run("red_unaccounted_pr", False, ledger_=ledger(), expect_unaccounted=[101])
    run("green_ledgered_pr_accounts_for_it", True, ledger_=ledger(entry(101, "test-only")))
    run("green_referenced_pr_accounts_for_it", True, ledger_=ledger(),
        changelog_=changelog.replace("see #102.", "see #102 and #101."))
    run("green_unreleased_reference_counts", True, ledger_=ledger(),
        changelog_=changelog.replace("## [Unreleased]\n", "## [Unreleased]\n- **First thing** (#101)\n"))
    run("red_reference_in_older_release_does_not_count", False, ledger_=ledger(),
        changelog_=changelog.replace("(#55)", "(#55, #101)"), expect_unaccounted=[101])
    run("red_longer_number_is_not_a_reference", False, ledger_=ledger(),
        changelog_=changelog.replace("see #102.", "see #102 and #1010."), expect_unaccounted=[101])
    run("red_missing_release_section", False,
        changelog_=changelog.replace("## [9.8.7-test]", "## [9.8.8-test]"),
        expect_problem="has no `## [9.8.7-test]` section")
    run("red_stale_ledger_entry", False, ledger_=ledger(entry(101), entry(999)),
        expect_problem="#999 is stale")
    run("red_redundant_ledger_entry", False, ledger_=ledger(entry(101), entry(103)),
        expect_problem="#103 is redundant")
    run("red_unknown_class", False, ledger_=ledger(entry(101, "misc")),
        expect_problem="has class 'misc'")
    run("red_short_reason", False, ledger_=ledger(entry(101, reason="CI only.")),
        expect_problem="needs a specific reason")
    run("red_generic_reason", False,
        ledger_=ledger(entry(101, reason="No user-facing change in this PR.")),
        expect_problem="generic reason")
    run("red_duplicate_entry", False, ledger_=ledger(entry(101), entry(101)),
        expect_problem="listed more than once")
    extra = commits + [("8" * 40, "test: another harness (#100)")]
    run("green_two_sorted_entries", True, commits_=extra, ledger_=ledger(entry(100), entry(101)))
    run("red_unsorted_entries", False, commits_=extra, ledger_=ledger(entry(101), entry(100)),
        expect_problem="not sorted")
    run("red_release_mismatch", False, ledger_=ledger(entry(101), release="v9.8.6-test"),
        expect_problem="ledger release is")
    run("red_schema_mismatch", False, ledger_=ledger(entry(101), schema="something.else"),
        expect_problem="ledger schema is")
    run("red_unknown_entry_key", False,
        ledger_=ledger({**entry(101), "note": "extra"}), expect_problem="unknown key")
    # The pull request under review is not in the history yet.
    def run_pending(name: str, expect_pass: bool, ledger_: dict, *, require: bool,
                    expect_unaccounted: list[int] | None = None) -> None:
        result = grade(commits, changelog, ledger_, record, range_spec,
                       pending=(104,), require_pending=require)
        ok = result["passed"] is expect_pass
        if expect_unaccounted is not None:
            ok = ok and [item["pr"] for item in result["unaccounted"]] == expect_unaccounted
        check(name, ok, f"passed={result['passed']} unaccounted="
                        f"{[item['pr'] for item in result['unaccounted']]} "
                        f"problems={len(result['problems'])}")

    run("red_ledger_entry_for_unmerged_pr_is_stale_without_pending", False,
        ledger_=ledger(entry(101), entry(104)), expect_problem="#104 is stale")
    run_pending("green_pending_pr_may_ledger_itself", True, ledger(entry(101), entry(104)),
                require=False)
    run_pending("green_pending_pr_is_optional_by_default", True, ledger(entry(101)), require=False)
    run_pending("red_required_pending_pr_unaccounted", False, ledger(entry(101)), require=True,
                expect_unaccounted=[104])
    run_pending("green_required_pending_pr_ledgered", True, ledger(entry(101), entry(104)),
                require=True)
    result = grade([("9" * 40, "docs: a direct commit")], changelog, ledger(entry(104)), record,
                   range_spec, pending=(104,), require_pending=True)
    check("red_pending_pr_does_not_rescue_a_zero_pr_range", result["passed"] is False,
          f"passed={result['passed']}")

    run("red_zero_pr_range", False, commits_=[("9" * 40, "docs: a direct commit")], ledger_=ledger(),
        expect_problem="yields zero merged pull requests")
    run("red_empty_range", False, commits_=[], ledger_=ledger(),
        expect_problem="yields zero merged pull requests")

    # The release record must carry a usable previous_tag.
    for name, data in (
        ("record_without_previous_tag", {"tag": "v9.8.7-test"}),
        ("record_previous_tag_equals_tag", {"tag": "v9.8.7-test", "previous_tag": "v9.8.7-test"}),
        ("record_malformed_previous_tag", {"tag": "v9.8.7-test", "previous_tag": "last release"}),
    ):
        try:
            validate_release_record(data)
            check(f"red_{name}", False, "accepted")
        except SourceError as exc:
            check(f"red_{name}", True, str(exc)[:60])

    # Fail closed: the range walk never returns a short list quietly.
    log_output = "".join(f"{sha}\t{subject}\n" for sha, subject in commits)

    def fake_git(*, shallow="false", missing=(), ancestor=True, log_code=0, not_a_repo=False):
        def runner(args: list[str]) -> tuple[int, str, str]:
            if args[:2] == ["rev-parse", "--is-shallow-repository"]:
                return (128, "", "fatal: not a git repository") if not_a_repo else (0, shallow + "\n", "")
            if args[:1] == ["rev-parse"]:
                return (1 if args[-1].split("^")[0] in missing else 0), "", ""
            if args[:1] == ["merge-base"]:
                return (0 if ancestor else 1), "", ""
            if args[:1] == ["log"]:
                return log_code, (log_output if log_code == 0 else ""), "fatal: bad revision"
            return 1, "", "unexpected git call"
        return runner

    def no_git(args: list[str]):
        raise FileNotFoundError("git")

    try:
        walked = collect_range_subjects(range_spec, runner=fake_git())
        check("green_full_history_is_walked", walked == commits, f"{len(walked)} commits")
    except SourceError as exc:
        check("green_full_history_is_walked", False, str(exc))
    for name, runner, needle in (
        ("git_unavailable", no_git, "git is unavailable"),
        ("not_a_git_checkout", fake_git(not_a_repo=True), "not a usable git checkout"),
        ("shallow_checkout", fake_git(shallow="true"), "shallow"),
        ("previous_tag_does_not_resolve", fake_git(missing=("v9.8.6-test",)), "does not resolve"),
        ("graded_commit_does_not_resolve", fake_git(missing=("HEAD",)), "does not resolve"),
        ("previous_tag_not_an_ancestor", fake_git(ancestor=False), "not an ancestor"),
        ("git_log_fails", fake_git(log_code=128), "git log"),
    ):
        try:
            collect_range_subjects(range_spec, runner=runner)
            check(f"red_{name}_fails_closed", False, "returned a list")
        except SourceError as exc:
            check(f"red_{name}_fails_closed", needle in str(exc), str(exc)[:60])
    for bad_range in ("HEAD", "a...b", "..HEAD"):
        try:
            collect_range_subjects(bad_range, runner=fake_git())
            check(f"red_malformed_range_{bad_range}", False, "returned a list")
        except SourceError:
            check(f"red_malformed_range_{bad_range}", True)

    if all_ok:
        print("self-test: PASS -- a referenced or ledgered pull request is accounted for, an "
              "unaccounted one fails, an [Unreleased] reference counts and an older release's does "
              "not, every ledger rule (class, reason, duplicate, stale, redundant, order, release, "
              "schema) fails red, both merge subject shapes parse and a mid-subject #N does not, "
              "a pull request under review may account for itself in advance, and a range that "
              "cannot be walked or yields no pull requests fails closed")
    else:
        print("self-test: FAIL -- the gate did not behave as specified", file=sys.stderr)
    return all_ok


# ───────────────────────── CLI ─────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--range", default=None, dest="range_spec",
                        help="git range to grade (default: <previous_tag>..HEAD from the release record)")
    parser.add_argument("--pending-pr", type=int, action="append", default=[], metavar="N",
                        help="a pull request about to join the range (the one under review): its "
                             "ledger entry is not stale; repeatable")
    parser.add_argument("--require-pending", action="store_true",
                        help="a --pending-pr must already be accounted for, like a merged one")
    parser.add_argument("--changelog", default=CHANGELOG_PATH)
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--record", default=RELEASE_RECORD_PATH)
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1
    if any(number <= 0 for number in args.pending_pr):
        parser.error("--pending-pr takes a positive pull request number")

    try:
        record = load_release_record(args.record)
        range_spec = args.range_spec or f"{record['previous_tag']}..HEAD"
        changelog_text = read_text(args.changelog, "changelog")
        ledger = load_ledger(args.ledger)
        commits = collect_range_subjects(range_spec)
        result = grade(commits, changelog_text, ledger, record, range_spec,
                       tuple(args.pending_pr), args.require_pending)
    except SourceError as exc:
        snippet = f"NO_DATA: {exc}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "NO_DATA", snippet)
        if args.format == "json":
            print(json.dumps({"status": "NO_DATA", "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: NO_DATA -- {exc}", file=sys.stderr)
            print("NO_DATA is not a pass: nothing was verified.", file=sys.stderr)
        return 2

    status = "PASS" if result["passed"] else "FAIL"
    if result["passed"]:
        snippet = (f"release={result['release']} range={result['range']} "
                   f"pull_requests={result['pull_requests']} referenced={result['referenced']} "
                   f"ledgered={result['ledgered']} -- every pull request is accounted for")
    else:
        snippet = (f"{len(result['unaccounted'])} unaccounted pull request(s): "
                   + ", ".join(f"#{item['pr']}" for item in result["unaccounted"][:40])
                   + f"; {len(result['problems'])} other problem(s): "
                   + "; ".join(result["problems"][:5]))
    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
        return 0 if result["passed"] else 1

    print(f"{PROBE_ID}: {status}")
    print(f"  release        : {result['release']} (tests/coverage/release_record.json)")
    print(f"  range          : {result['range']} ({result['commits']} commits)")
    print(f"  pull requests  : {result['pull_requests']}")
    print(f"  in changelog   : {result['referenced']} "
          f"(`#N` in ## [{record['version']}] or ## [Unreleased])")
    print(f"  in ledger      : {result['ledgered']} ({LEDGER_REL})")
    if result["problems"]:
        print("  PROBLEMS:")
        for problem in result["problems"]:
            print(f"    - {problem}")
    if result["unaccounted"]:
        print(f"  UNACCOUNTED PULL REQUESTS ({len(result['unaccounted'])}):")
        for item in result["unaccounted"]:
            print(f"    - #{item['pr']}  {item['sha'][:9]}  {item['subject']}")
        print("  Resolve each one in exactly one of two ways:")
        print(f"    1. a user can observe the change (language, stdlib, runtime, tooling, build,")
        print(f"       packaging, user-facing docs, examples, benchmarks): describe it under Added,")
        print(f"       Changed or Fixed in the `## [{record['version']}]` section of CHANGELOG.md and end")
        print(f"       the entry with `(#N)`; if an entry already describes it, append the reference.")
        print(f"    2. a user cannot observe it: add {{\"pr\": N, \"class\": ..., \"reason\": ...}} to")
        print(f"       {LEDGER_REL}, sorted by number, with a class")
        print(f"       from {{{', '.join(sorted(LEDGER_CLASSES))}}}")
        print(f"       and a reason that says what changed and why it is not user-facing.")
    if result["passed"]:
        print("  every merged pull request in the range has exactly one home")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
