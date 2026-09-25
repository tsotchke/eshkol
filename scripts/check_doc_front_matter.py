#!/usr/bin/env python3
"""Release gate: a documentation page that declares what it is must declare it
correctly, keep to the present tense if it says it is current, and never
present a dated page as current -- and the number of pages that declare
nothing may only go down.

Motivating incident (v1.3.5-evolve documentation pass, 2026-09-17): the
documentation set mixes four kinds of page under one directory tree with
nothing to tell them apart -- reference pages that must describe the tree as
it is, dated audit reports that must never be edited, designs that a later
design replaced, and release narrative. A reader following a link from a
current guide landed on a superseded design with no warning, reference pages
carried "new in" and "recently" sentences that were true for one release and
wrong for every release after it, and an audit of which pages were still
authoritative had to be done by reading them. The owner adopted a front-matter
block so that a page states its own kind and status; a block nobody checks
decays exactly like the prose it was meant to fix, so this gate checks it.

Scope. Every tracked `*.md` under docs/ plus the root project pages
(README.md, CHANGELOG.md, RELEASE_NOTES.md, ANNOUNCEMENT.md, ROADMAP.md,
CONTRIBUTING.md, SECURITY.md), excluding docs/api/** (generated). The list
comes from `git ls-files` (tracked, plus untracked files git does not ignore,
so a page is graded before its first commit); without git the tree is walked.

Four rules.

  1. Ratchet. A page WITHOUT front matter is left alone but counted. The count
     is recorded in tests/coverage/doc_front_matter_baseline.json. A count
     above the baseline fails: a new page must carry the block. A count below
     it fails too, until the baseline is lowered with `--update-baseline`, so
     the recorded number can only go down and cannot creep back.

  2. Schema. A page WITH front matter (the file's first line is `---`, and a
     later `---` line closes the block) is parsed by the small strict parser
     below -- the block is flat `key: value` pairs plus one list, `sources:`,
     written as a block list or as the inline empty list `[]`; ` #` starts a
     comment. The keys are exactly kind, status, owner-area, since and
     sources, plus `superseded-by`, which is required when and only when the
     status is `superseded`. kind, status and owner-area come from the closed
     sets below; since is a release tag; every sources entry and superseded-by
     is a repo-relative path that exists. Unknown keys, duplicate keys and an
     unterminated block fail, and the first non-blank line after the block
     must be a markdown H1. A page of kind `report` is a dated record: its
     status is `historical` or `superseded`, never `current`.

  3. Evergreen. A page of kind tutorial, guide, reference or explanation with
     status current describes what is true now. Outside fenced code blocks
     these phrases fail, with file:line: "new in v", "what's new", "recently",
     "refreshed candidate", "release candidate", "this release adds", "in this
     release". Release narrative belongs in the changelog and the release
     notes. A line is exempt when it, or the line above it, carries
     `<!-- evergreen: allow <reason> -->`; every exemption is counted and
     printed, so an exemption is visible rather than silent.

  4. Links. In a page whose status is current, a relative markdown link to a
     page whose own front matter says historical or superseded must say so on
     the same line, with the word "historical", "superseded", "dated" or
     "archived". A page without front matter is not judged, as a source or as
     a target.

Why a hand-written parser. The block is a restricted shape on purpose, the
gate must run where nothing is installed, and a general YAML loader accepts
spellings (anchors, flow mappings, multi-line scalars, `status: superseded-by:
x` read as a string) that this schema wants refused with a sentence saying
what to write instead.

Modes / exit status
    PASS      0   every page with front matter is valid, and the count of
                  pages without it equals the baseline.
    FAIL      1   a page breaks a rule, the count differs from the baseline in
                  either direction, or the baseline cannot be read.
    NO_DATA   2   no documentation page was found at all -- nothing was
                  verified, which is not a pass.

Usage
    python3 scripts/check_doc_front_matter.py
    python3 scripts/check_doc_front_matter.py --no-trace
    python3 scripts/check_doc_front_matter.py --format json
    python3 scripts/check_doc_front_matter.py --update-baseline
    python3 scripts/check_doc_front_matter.py --self-test
    python3 scripts/check_doc_front_matter.py --self-test --self-test-dir DIR

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "doc_front_matter_gate.jsonl"
PROBE_ID = "doc_front_matter_valid"

BASELINE_REL = "tests/coverage/doc_front_matter_baseline.json"
BASELINE_SCHEMA = "eshkol.doc-front-matter-baseline.v1"
BASELINE_COMMENT = (
    "Number of documentation pages in the scope of scripts/check_doc_front_matter.py that carry "
    "no front-matter block (kind, status, owner-area, since, sources). The gate fails when the "
    "live count is above this number (a new page must carry the block) and when it is below it "
    "(lower the number with `python3 scripts/check_doc_front_matter.py --update-baseline`), so "
    "the recorded number can only go down. Never raise it by hand.")

ROOT_PAGES = ("README.md", "CHANGELOG.md", "RELEASE_NOTES.md", "ANNOUNCEMENT.md", "ROADMAP.md",
              "CONTRIBUTING.md", "SECURITY.md")
DOCS_DIR = "docs"
EXCLUDED_PREFIXES = ("docs/api/",)

KINDS = ("tutorial", "guide", "reference", "explanation", "project", "report")
STATUSES = ("current", "historical", "superseded")
OWNER_AREAS = ("language", "types", "ad", "tensors", "stdlib", "runtime", "vm", "memory", "build",
               "platform", "web", "gpu", "quantum", "agent", "testing", "release", "docs", "project")
REQUIRED_KEYS = ("kind", "status", "owner-area", "since", "sources")
OPTIONAL_KEYS = ("superseded-by",)
EVERGREEN_KINDS = ("tutorial", "guide", "reference", "explanation")
NOT_CURRENT = ("historical", "superseded")

SINCE_RE = re.compile(r"^v\d+\.\d+\.\d+(-[a-z0-9]+)?$")
KEY_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):(?:[ \t]+(.*))?$")
LIST_ITEM_RE = re.compile(r"^[ \t]+-[ \t]+(.*)$")
H1_RE = re.compile(r"^# \S")
FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
ALLOW_RE = re.compile(r"<!--\s*evergreen:\s*allow\s+\S.*?-->")
EVERGREEN_PHRASES = ("new in v", "what's new", "what’s new", "recently", "refreshed candidate",
                     "release candidate", "this release adds", "in this release")
INLINE_LINK_RE = re.compile(r"\[[^\]\n]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE_LINK_RE = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*<?([^\s>]+)>?")
LINK_LABEL_WORDS_RE = re.compile(r"\b(historical|superseded|dated|archived)\b", re.IGNORECASE)
SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


# ───────────────────────── scope ─────────────────────────

def in_scope(rel: str) -> bool:
    rel = rel.replace(os.sep, "/")
    if not rel.endswith(".md"):
        return False
    if rel in ROOT_PAGES:
        return True
    if not rel.startswith(DOCS_DIR + "/"):
        return False
    return not any(rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def list_pages(root: str, use_git: bool = True) -> list[str]:
    """Repo-relative, sorted paths of every page in scope that exists on disk."""
    candidates: list[str] | None = None
    if use_git:
        try:
            proc = subprocess.run(
                ["git", "-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard",
                 "--", DOCS_DIR, *ROOT_PAGES],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode == 0:
                candidates = [p for p in proc.stdout.decode("utf-8", "replace").split("\0") if p]
        except OSError:
            candidates = None
    if candidates is None:
        candidates = [name for name in ROOT_PAGES]
        for base, dirs, files in os.walk(os.path.join(root, DOCS_DIR)):
            dirs.sort()
            for name in files:
                candidates.append(os.path.relpath(os.path.join(base, name), root).replace(os.sep, "/"))
    return sorted({rel for rel in candidates
                   if in_scope(rel) and os.path.isfile(os.path.join(root, rel))})


# ───────────────────────── front matter ─────────────────────────

def strip_comment(value: str) -> str:
    """Drop a ` #` comment and surrounding space; a `#` inside a word stays."""
    match = re.search(r"(^|[ \t])#", value)
    if match:
        value = value[:match.start()]
    return value.strip()


def unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def split_front_matter(text: str) -> tuple[list[tuple[int, str]] | None, int, str | None]:
    """Return (block lines with 1-based numbers, index of the first body line, error).

    (None, 0, None) means the page has no front matter at all.
    """
    lines = text.split("\n")
    if not lines or lines[0].lstrip("﻿").rstrip() != "---":
        return None, 0, None
    for index in range(1, len(lines)):
        if lines[index].rstrip() == "---":
            return [(n + 1, lines[n]) for n in range(1, index)], index + 1, None
    return [], len(lines), ("front matter is not terminated: the file opens with `---` but no later "
                            "`---` line closes the block")


def parse_front_matter(block: list[tuple[int, str]]) -> tuple[dict, list[tuple[int, str]]]:
    """Parse the restricted shape. Returns (mapping, [(line, problem)])."""
    data: dict = {}
    problems: list[tuple[int, str]] = []
    open_list: str | None = None
    for number, raw in block:
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = LIST_ITEM_RE.match(line)
        if item:
            if open_list is None:
                problems.append((number, "a list item appears outside `sources:`; only `sources` is a "
                                         "list, written as `sources:` followed by `  - path` lines"))
                continue
            value = unquote(strip_comment(item.group(1)))
            if not value:
                problems.append((number, "empty list item under `sources:`"))
            else:
                data[open_list].append(value)
            continue
        if line[0] in " \t":
            problems.append((number, f"unexpected indented line {line.strip()!r}; the block is flat "
                                     f"`key: value` pairs plus the `sources:` list"))
            continue
        open_list = None
        match = KEY_LINE_RE.match(line)
        if not match:
            problems.append((number, f"not a `key: value` line: {line.strip()!r}"))
            continue
        key, value = match.group(1), unquote(strip_comment(match.group(2) or ""))
        if key in data:
            problems.append((number, f"duplicate key `{key}`"))
            continue
        if key not in REQUIRED_KEYS + OPTIONAL_KEYS:
            problems.append((number, f"unknown key `{key}`; the keys are "
                                     f"{', '.join(REQUIRED_KEYS)} and, for a superseded page, "
                                     f"superseded-by"))
            continue
        if key == "sources":
            if value == "[]":
                data[key] = []
            elif value == "":
                data[key] = []
                open_list = key
            else:
                data[key] = []
                problems.append((number, "`sources` is a list: write `sources: []` when there are none, "
                                         "or `sources:` followed by one `  - path` line per source"))
            continue
        if value == "":
            problems.append((number, f"`{key}` has no value"))
        data[key] = value
    return data, problems


def path_problem(root: str, value: str, what: str) -> str | None:
    if SCHEME_RE.match(value) or value.startswith("/") or value.startswith("~") or "\\" in value:
        return f"{what} {value!r} must be a repo-relative path (no URL, no leading `/`, no `\\`)"
    normal = posixpath.normpath(value)
    if normal == ".." or normal.startswith("../"):
        return f"{what} {value!r} leaves the repository"
    if not os.path.exists(os.path.join(root, normal)):
        return f"{what} {value!r} does not exist in the repository"
    return None


def validate_front_matter(root: str, data: dict, first_line: int) -> list[tuple[int, str]]:
    problems: list[tuple[int, str]] = []

    def add(message: str) -> None:
        problems.append((first_line, message))

    for key in REQUIRED_KEYS:
        if key not in data:
            add(f"missing key `{key}`")
    kind, status = data.get("kind"), data.get("status")
    if "kind" in data and kind not in KINDS:
        add(f"kind {kind!r} is not one of: {', '.join(KINDS)}")
    if "status" in data and status not in STATUSES:
        hint = ""
        if isinstance(status, str) and status.startswith("superseded"):
            hint = ("; write `status: superseded` and, on the next line, `superseded-by: <path>` "
                    "as a separate key")
        add(f"status {status!r} is not one of: {', '.join(STATUSES)}{hint}")
    if "owner-area" in data and data["owner-area"] not in OWNER_AREAS:
        add(f"owner-area {data['owner-area']!r} is not one of: {', '.join(OWNER_AREAS)}")
    if "since" in data and not SINCE_RE.match(data["since"] or ""):
        add(f"since {data['since']!r} is not a release tag such as v1.3.5 or v1.3.5-evolve")
    for source in data.get("sources") or []:
        problem = path_problem(root, source, "sources entry")
        if problem:
            add(problem)
    if status == "superseded" and "superseded-by" not in data:
        add("status is superseded, so `superseded-by: <repo-relative path>` is required")
    if "superseded-by" in data:
        if status != "superseded":
            add("`superseded-by` is only allowed when status is superseded")
        elif data["superseded-by"]:
            problem = path_problem(root, data["superseded-by"], "superseded-by")
            if problem:
                add(problem)
    if kind == "report" and status == "current":
        add("a page of kind report is a dated record: its status is historical or superseded, "
            "never current")
    return problems


# ───────────────────────── body rules ─────────────────────────

def body_lines_outside_fences(lines: list[str], start: int):
    """Yield (index, line) for body lines that are not inside a fenced code block."""
    fence: str | None = None
    for index in range(start, len(lines)):
        line = lines[index]
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker[0] * 3
            elif marker.startswith(fence):
                fence = None
            continue
        if fence is None:
            yield index, line


def evergreen_findings(lines: list[str], start: int) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return (violations, exemptions), each as (1-based line, text)."""
    violations, exemptions = [], []
    for index, line in body_lines_outside_fences(lines, start):
        lowered = line.lower()
        hits = [phrase for phrase in EVERGREEN_PHRASES if phrase in lowered]
        if not hits:
            continue
        allowed = ALLOW_RE.search(line) or (index > 0 and ALLOW_RE.search(lines[index - 1]))
        phrase = hits[0].replace("’", "'")
        if allowed:
            exemptions.append((index + 1, f"\"{phrase}\" allowed: {allowed.group(0)}"))
        else:
            violations.append((index + 1, f"release narrative in a current page: \"{phrase}\" -- say "
                                          f"what is true now (use \"(since vX.Y.Z)\" for provenance), "
                                          f"move the sentence to the changelog or release notes, or "
                                          f"mark the line `<!-- evergreen: allow <reason> -->`"))
    return violations, exemptions


def link_targets(lines: list[str], start: int):
    """Yield (1-based line, line text, raw target) for each relative markdown link."""
    for index, line in body_lines_outside_fences(lines, start):
        targets = [m.group(1) for m in INLINE_LINK_RE.finditer(line)]
        reference = REFERENCE_LINK_RE.match(line)
        if reference:
            targets.append(reference.group(1))
        for target in targets:
            if target.startswith("#") or SCHEME_RE.match(target) or target.startswith("//"):
                continue
            yield index + 1, line, target


def resolve_link(page_rel: str, target: str) -> str:
    target = target.split("#", 1)[0].split("?", 1)[0]
    if target.startswith("/"):
        return posixpath.normpath(target.lstrip("/"))
    return posixpath.normpath(posixpath.join(posixpath.dirname(page_rel), target))


# ───────────────────────── grading ─────────────────────────

def load_baseline(path: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("schema") != BASELINE_SCHEMA:
        raise ValueError(f"baseline schema is {data.get('schema') if isinstance(data, dict) else None!r}, "
                         f"expected {BASELINE_SCHEMA!r}")
    count = data.get("pages_without_front_matter")
    if type(count) is not int or count < 0:
        raise ValueError("pages_without_front_matter must be a non-negative integer")
    return count


def write_baseline(path: str, count: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"schema": BASELINE_SCHEMA, "_comment": BASELINE_COMMENT,
                   "pages_without_front_matter": count}, handle, indent=2)
        handle.write("\n")


def scan(root: str, use_git: bool = True) -> dict:
    """Grade every page; the ratchet is applied by the caller."""
    pages = list_pages(root, use_git)
    findings: list[dict] = []
    exemptions: list[dict] = []
    unmarked: list[str] = []
    parsed: dict[str, dict] = {}
    bodies: dict[str, tuple[list[str], int]] = {}

    def finding(page: str, line: int, rule: str, message: str) -> None:
        findings.append({"page": page, "line": line, "rule": rule, "message": message})

    for page in pages:
        with open(os.path.join(root, page), "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        block, body_start, error = split_front_matter(text)
        if block is None:
            unmarked.append(page)
            continue
        if error:
            finding(page, 1, "schema", error)
            parsed[page] = {}
            continue
        data, problems = parse_front_matter(block)
        problems += validate_front_matter(root, data, 1)
        lines = text.split("\n")
        first_body = next((i for i in range(body_start, len(lines)) if lines[i].strip()), None)
        if first_body is None or not H1_RE.match(lines[first_body]):
            problems.append(((first_body or body_start) + 1,
                             "the first non-blank line after the front matter must be the page's "
                             "markdown H1 (`# Title`)"))
        for line, message in problems:
            finding(page, line, "schema", message)
        parsed[page] = data
        bodies[page] = (lines, body_start)

    for page, (lines, body_start) in bodies.items():
        data = parsed[page]
        if data.get("status") != "current":
            continue
        if data.get("kind") in EVERGREEN_KINDS:
            violations, allowed = evergreen_findings(lines, body_start)
            for line, message in violations:
                finding(page, line, "evergreen", message)
            for line, message in allowed:
                exemptions.append({"page": page, "line": line, "message": message})
        for line, text, target in link_targets(lines, body_start):
            resolved = resolve_link(page, target)
            status = parsed.get(resolved, {}).get("status")
            if status in NOT_CURRENT and not LINK_LABEL_WORDS_RE.search(text):
                finding(page, line, "link",
                        f"links to {resolved}, whose front matter says {status}, without saying so: put "
                        f"\"historical\", \"superseded\", \"dated\" or \"archived\" on the same line, or "
                        f"link to the current page instead")

    by_directory: dict[str, int] = {}
    for page in unmarked:
        directory = posixpath.dirname(page) or "."
        by_directory[directory] = by_directory.get(directory, 0) + 1
    top = sorted(by_directory.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "pages": len(pages),
        "with_front_matter": len(pages) - len(unmarked),
        "without_front_matter": len(unmarked),
        "unmarked_pages": unmarked,
        "top_unmarked_directories": [{"directory": d, "pages": n} for d, n in top],
        "findings": sorted(findings, key=lambda f: (f["page"], f["line"])),
        "exemptions": exemptions,
    }


def ratchet_problems(count: int, baseline_path: str) -> tuple[int | None, list[str]]:
    rel = os.path.relpath(baseline_path, REPO_ROOT).replace(os.sep, "/")
    try:
        recorded = load_baseline(baseline_path)
    except Exception as exc:
        return None, [f"cannot read the baseline {rel}: {exc}; create it with --update-baseline and "
                      f"review the number"]
    if count > recorded:
        return recorded, [f"{count} page(s) have no front matter, the baseline allows {recorded}: a new "
                          f"or renamed page must start with the front-matter block (kind, status, "
                          f"owner-area, since, sources); the baseline is never raised"]
    if count < recorded:
        return recorded, [f"{count} page(s) have no front matter, the baseline still says {recorded}: "
                          f"lower it with `python3 scripts/check_doc_front_matter.py --update-baseline` "
                          f"so the count cannot creep back"]
    return recorded, []


def run_gate(root: str, baseline_path: str, use_git: bool = True) -> dict:
    result = scan(root, use_git)
    recorded, problems = ratchet_problems(result["without_front_matter"], baseline_path)
    result["baseline"] = recorded
    result["ratchet_problems"] = problems
    result["passed"] = not result["findings"] and not problems
    return result


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

GOOD_BLOCK = ("---\nkind: guide\nstatus: current   # a comment\nowner-area: types\nsince: v9.8.7\n"
              "sources:\n  - lib/source.c\n  - lib\n---\n")


def self_test(parent_dir: str | None) -> bool:
    """Hermetic: fixture trees in a fresh temporary directory, no git, no network."""
    print("check_doc_front_matter.py self-test:")
    all_ok = True
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
        work = tempfile.mkdtemp(prefix="doc-front-matter-selftest-", dir=parent_dir)
    else:
        work = tempfile.mkdtemp(prefix="doc-front-matter-selftest-")
    counter = [0]

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal all_ok
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] {name}" + (f": {detail}" if detail else ""))

    def tree(files: dict[str, str], baseline: int | None) -> tuple[str, str]:
        counter[0] += 1
        root = os.path.join(work, f"case{counter[0]:03d}")
        files = {"lib/source.c": "int x;\n", **files}
        for rel, text in files.items():
            path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        baseline_path = os.path.join(root, BASELINE_REL)
        if baseline is not None:
            write_baseline(baseline_path, baseline)
        return root, baseline_path

    def run(name: str, files: dict[str, str], expect_pass: bool, *, baseline: int | None = 0,
            needle: str | None = None, rule: str | None = None, exemptions: int | None = None) -> dict:
        root, baseline_path = tree(files, baseline)
        result = run_gate(root, baseline_path, use_git=False)
        messages = [f["message"] for f in result["findings"]] + result["ratchet_problems"]
        ok = result["passed"] is expect_pass
        if needle is not None:
            ok = ok and any(needle in message for message in messages)
        if rule is not None:
            ok = ok and any(f["rule"] == rule for f in result["findings"])
        if exemptions is not None:
            ok = ok and len(result["exemptions"]) == exemptions
        check(name, ok, f"passed={result['passed']} findings={len(result['findings'])} "
                        f"ratchet={len(result['ratchet_problems'])}")
        return result

    def page(block: str = GOOD_BLOCK, body: str = "# Title\n\nText.\n") -> str:
        return block + body

    try:
        # Scope.
        scope_cases = [("docs/a.md", True), ("docs/deep/er/b.md", True), ("README.md", True),
                       ("SECURITY.md", True), ("docs/api/generated.md", False), ("docs/a.txt", False),
                       ("lib/README.md", False), ("OTHER.md", False)]
        check("scope_filter", all(in_scope(rel) is want for rel, want in scope_cases))
        result = run("green_generated_and_foreign_pages_are_out_of_scope",
                     {"docs/a.md": page(), "docs/api/x.md": "no block\n", "lib/README.md": "no block\n"},
                     True)
        check("scope_counts_only_pages_in_scope", result["pages"] == 1, f"pages={result['pages']}")

        # Ratchet, both directions.
        unmarked = {"docs/old.md": "# Old\n", "README.md": "# Readme\n", "docs/a.md": page()}
        run("green_unmarked_count_equals_baseline", unmarked, True, baseline=2)
        run("red_unmarked_count_above_baseline", unmarked, False, baseline=1, needle="the baseline allows 1")
        run("red_unmarked_count_below_baseline", unmarked, False, baseline=3, needle="still says 3")
        run("red_missing_baseline", unmarked, False, baseline=None, needle="cannot read the baseline")
        root, baseline_path = tree(unmarked, 2)
        with open(baseline_path, "w", encoding="utf-8") as handle:
            json.dump({"schema": "something.else", "pages_without_front_matter": 2}, handle)
        result = run_gate(root, baseline_path, use_git=False)
        check("red_baseline_schema_mismatch", result["passed"] is False)
        result = run("green_top_directories_reported",
                     {"docs/x/a.md": "# A\n", "docs/x/b.md": "# B\n", "docs/y/c.md": "# C\n"}, True,
                     baseline=3)
        check("top_directory_order", result["top_unmarked_directories"][:2] ==
              [{"directory": "docs/x", "pages": 2}, {"directory": "docs/y", "pages": 1}])

        # Schema.
        run("green_valid_block", {"docs/a.md": page()}, True)
        run("green_inline_empty_sources",
            {"docs/a.md": page(GOOD_BLOCK.replace("sources:\n  - lib/source.c\n  - lib\n", "sources: []\n"))},
            True)
        run("green_superseded_with_target",
            {"docs/new.md": page(),
             "docs/a.md": page(GOOD_BLOCK.replace("status: current   # a comment",
                                                  "status: superseded\nsuperseded-by: docs/new.md"))},
            True)
        run("green_historical_report",
            {"docs/a.md": page(GOOD_BLOCK.replace("kind: guide", "kind: report")
                               .replace("status: current   # a comment", "status: historical"))}, True)
        schema_red = [
            ("unterminated_block", "---\nkind: guide\n# Title\n", "not terminated"),
            ("unknown_key", GOOD_BLOCK.replace("since:", "author: someone\nsince:") + "# T\n", "unknown key"),
            ("duplicate_key", GOOD_BLOCK.replace("since:", "kind: guide\nsince:") + "# T\n", "duplicate key"),
            ("missing_key", GOOD_BLOCK.replace("owner-area: types\n", "") + "# T\n", "missing key `owner-area`"),
            ("bad_kind", GOOD_BLOCK.replace("kind: guide", "kind: howto") + "# T\n", "kind 'howto'"),
            ("bad_status", GOOD_BLOCK.replace("status: current", "status: draft") + "# T\n", "status 'draft'"),
            ("status_superseded_by_on_one_line",
             GOOD_BLOCK.replace("status: current   # a comment", "status: superseded-by: docs/new.md") + "# T\n",
             "as a separate key"),
            ("bad_owner_area", GOOD_BLOCK.replace("owner-area: types", "owner-area: compiler") + "# T\n",
             "owner-area 'compiler'"),
            ("bad_since", GOOD_BLOCK.replace("since: v9.8.7", "since: 9.8") + "# T\n", "not a release tag"),
            ("missing_source_path", GOOD_BLOCK.replace("lib/source.c", "lib/absent.c") + "# T\n",
             "does not exist"),
            ("absolute_source_path", GOOD_BLOCK.replace("lib/source.c", "/etc/hosts") + "# T\n",
             "repo-relative"),
            ("source_path_leaves_repo", GOOD_BLOCK.replace("lib/source.c", "../outside") + "# T\n",
             "leaves the repository"),
            ("inline_sources_list", GOOD_BLOCK.replace("sources:\n  - lib/source.c\n  - lib\n",
                                                       "sources: [lib/source.c]\n") + "# T\n",
             "`sources` is a list"),
            ("list_item_outside_sources", GOOD_BLOCK.replace("kind: guide\n", "kind: guide\n  - stray\n") + "# T\n",
             "outside `sources:`"),
            ("not_a_key_value_line", GOOD_BLOCK.replace("since: v9.8.7", "since v9.8.7") + "# T\n",
             "not a `key: value` line"),
            ("superseded_without_target",
             GOOD_BLOCK.replace("status: current   # a comment", "status: superseded") + "# T\n",
             "is required"),
            ("superseded_by_on_current_page",
             GOOD_BLOCK.replace("since:", "superseded-by: lib/source.c\nsince:") + "# T\n", "only allowed"),
            ("superseded_by_missing_path",
             GOOD_BLOCK.replace("status: current   # a comment",
                                "status: superseded\nsuperseded-by: docs/absent.md") + "# T\n",
             "does not exist"),
            ("report_cannot_be_current", GOOD_BLOCK.replace("kind: guide", "kind: report") + "# T\n",
             "never current"),
            ("first_line_after_block_is_not_h1", GOOD_BLOCK + "\nSome text first.\n\n# Title\n", "markdown H1"),
            ("h2_is_not_h1", GOOD_BLOCK + "## Title\n", "markdown H1"),
        ]
        for name, text, needle in schema_red:
            run("red_" + name, {"docs/a.md": text}, False, needle=needle, rule="schema")

        # Evergreen.
        narrative = "# T\n\nThis is new in v9.8 and was recently fixed.\n"
        run("red_evergreen_phrase_in_current_guide", {"docs/a.md": page(body=narrative)}, False,
            needle="release narrative", rule="evergreen")
        for phrase in ("New in v1.2", "What's New", "Recently", "the refreshed candidate",
                       "Release Candidate", "this release adds", "In this release"):
            run("red_evergreen_" + re.sub(r"[^a-z]+", "_", phrase.lower()).strip("_"),
                {"docs/a.md": page(body=f"# T\n\nText: {phrase} here.\n")}, False, rule="evergreen")
        run("green_evergreen_phrase_inside_code_fence",
            {"docs/a.md": page(body="# T\n\n```\nrecently\n```\n\n~~~sh\nnew in v1\n~~~\n")}, True)
        run("green_evergreen_allowed_on_same_line",
            {"docs/a.md": page(body="# T\n\nA recently used list. <!-- evergreen: allow LRU term -->\n")},
            True, exemptions=1)
        run("green_evergreen_allowed_on_previous_line",
            {"docs/a.md": page(body="# T\n\n<!-- evergreen: allow LRU term -->\nA recently used list.\n")},
            True, exemptions=1)
        run("red_evergreen_allow_without_reason",
            {"docs/a.md": page(body="# T\n\nA recently used list. <!-- evergreen: allow -->\n")}, False,
            rule="evergreen")
        run("red_evergreen_allow_two_lines_up_does_not_count",
            {"docs/a.md": page(body="# T\n\n<!-- evergreen: allow LRU term -->\n\nA recently used list.\n")},
            False, rule="evergreen")
        run("green_evergreen_not_enforced_on_project_pages",
            {"docs/a.md": page(GOOD_BLOCK.replace("kind: guide", "kind: project"), narrative)}, True)
        run("green_evergreen_not_enforced_on_historical_pages",
            {"docs/a.md": page(GOOD_BLOCK.replace("status: current   # a comment", "status: historical"),
                               narrative)}, True)
        run("green_evergreen_not_enforced_without_front_matter", {"docs/a.md": narrative}, True, baseline=1)

        # Links.
        old = page(GOOD_BLOCK.replace("status: current   # a comment", "status: historical"))
        gone = page(GOOD_BLOCK.replace("status: current   # a comment",
                                       "status: superseded\nsuperseded-by: docs/a.md"))
        run("red_link_to_historical_page_unlabelled",
            {"docs/a.md": page(body="# T\n\nSee [the audit](sub/old.md#part).\n"), "docs/sub/old.md": old},
            False, needle="whose front matter says historical", rule="link")
        run("red_link_to_superseded_page_unlabelled",
            {"docs/a.md": page(body="# T\n\nSee [the design](gone.md).\n"), "docs/gone.md": gone}, False,
            needle="whose front matter says superseded", rule="link")
        run("red_reference_style_link_unlabelled",
            {"docs/a.md": page(body="# T\n\nSee [the audit][1].\n\n[1]: sub/old.md\n"), "docs/sub/old.md": old},
            False, rule="link")
        run("red_updated_is_not_the_word_dated",
            {"docs/a.md": page(body="# T\n\nSee the updated [audit](sub/old.md).\n"), "docs/sub/old.md": old},
            False, rule="link")
        for word in ("historical", "superseded", "dated", "Archived"):
            run("green_link_labelled_" + word.lower(),
                {"docs/a.md": page(body=f"# T\n\nSee the {word} [audit](sub/old.md).\n"),
                 "docs/sub/old.md": old}, True)
        run("green_link_from_parent_directory_resolves",
            {"docs/sub/a.md": page(body="# T\n\nSee the dated [audit](../old.md).\n"), "docs/old.md": old}, True)
        run("green_link_to_current_page", {"docs/a.md": page(body="# T\n\nSee [b](b.md).\n"), "docs/b.md": page()},
            True)
        run("green_link_to_page_without_front_matter_is_not_judged",
            {"docs/a.md": page(body="# T\n\nSee [b](b.md).\n"), "docs/b.md": "# B\n"}, True, baseline=1)
        run("green_external_and_anchor_links_are_not_judged",
            {"docs/a.md": page(body="# T\n\n[x](https://example.org/old.md) [y](#part) [z](mailto:a@b.c)\n"),
             "docs/old.md": old}, True)
        run("green_link_inside_code_fence_is_not_judged",
            {"docs/a.md": page(body="# T\n\n```\n[audit](old.md)\n```\n"), "docs/old.md": old}, True)
        run("green_historical_page_may_link_anywhere",
            {"docs/old.md": old, "docs/older.md": page(
                GOOD_BLOCK.replace("status: current   # a comment", "status: historical"),
                "# T\n\nSee [old](old.md).\n")}, True)

        # --update-baseline writes the live count and refuses to raise it.
        root, baseline_path = tree({"docs/x.md": "# X\n", "docs/y.md": "# Y\n"}, 5)
        code = update_baseline(root, baseline_path, use_git=False, allow_increase=False, out=None)
        check("update_baseline_lowers", code == 0 and load_baseline(baseline_path) == 2)
        write_baseline(baseline_path, 1)
        code = update_baseline(root, baseline_path, use_git=False, allow_increase=False, out=None)
        check("update_baseline_refuses_to_raise", code == 1 and load_baseline(baseline_path) == 1)
        code = update_baseline(root, baseline_path, use_git=False, allow_increase=True, out=None)
        check("update_baseline_raises_only_with_allow_increase", code == 0 and load_baseline(baseline_path) == 2)

        # No pages at all is not a pass.
        root, baseline_path = tree({}, 0)
        check("no_pages_is_no_data", scan(root, use_git=False)["pages"] == 0)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if all_ok:
        print("self-test: PASS -- the ratchet fails in both directions and on an unreadable baseline, "
              "every schema rule fails red and a valid block passes, release narrative fails only in "
              "current evergreen pages outside code fences and an explicit exemption is counted, an "
              "unlabelled link from a current page to a historical or superseded page fails, and "
              "--update-baseline never raises the number unasked")
    else:
        print("self-test: FAIL -- the gate did not behave as specified", file=sys.stderr)
    return all_ok


# ───────────────────────── CLI ─────────────────────────

def update_baseline(root: str, baseline_path: str, use_git: bool, allow_increase: bool, out=sys.stdout) -> int:
    count = scan(root, use_git)["without_front_matter"]
    try:
        previous = load_baseline(baseline_path)
    except Exception:
        previous = None
    if previous is not None and count > previous and not allow_increase:
        if out:
            print(f"refusing to raise the baseline from {previous} to {count}: a new page must carry "
                  f"front matter. Pass --allow-increase only for a reviewed scope change.", file=out)
        return 1
    write_baseline(baseline_path, count)
    if out:
        print(f"baseline written: {os.path.relpath(baseline_path, root)} "
              f"(pages_without_front_matter={count}, was {previous})", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=REPO_ROOT, help="repository root to grade (default: this repo)")
    parser.add_argument("--baseline", default=None, help=f"baseline file (default: <root>/{BASELINE_REL})")
    parser.add_argument("--update-baseline", action="store_true",
                        help="record the live count of pages without front matter; refuses to raise it")
    parser.add_argument("--allow-increase", action="store_true",
                        help="with --update-baseline: allow the recorded count to go up (reviewed "
                             "scope change only)")
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    parser.add_argument("--self-test-dir", default=None,
                        help="directory under which the self-test creates (and removes) its fixture "
                             "tree; default: the platform temporary directory (honours TMPDIR)")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test(args.self_test_dir) else 1

    root = os.path.abspath(args.root)
    baseline_path = args.baseline or os.path.join(root, BASELINE_REL)
    if args.update_baseline:
        return update_baseline(root, baseline_path, True, args.allow_increase)

    result = run_gate(root, baseline_path)
    if result["pages"] == 0:
        snippet = f"NO_DATA: no documentation page found under {root}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "NO_DATA", snippet)
        if args.format == "json":
            print(json.dumps({"status": "NO_DATA", "error": snippet}, indent=2))
        else:
            print(f"{PROBE_ID}: {snippet}", file=sys.stderr)
            print("NO_DATA is not a pass: nothing was verified.", file=sys.stderr)
        return 2

    status = "PASS" if result["passed"] else "FAIL"
    if result["passed"]:
        snippet = (f"pages={result['pages']} with_front_matter={result['with_front_matter']} "
                   f"without={result['without_front_matter']} (baseline {result['baseline']}) "
                   f"exemptions={len(result['exemptions'])} -- every declared page is valid")
    else:
        snippet = (f"{len(result['findings'])} finding(s), {len(result['ratchet_problems'])} ratchet "
                   f"problem(s): " + "; ".join(
                       [f"{f['page']}:{f['line']} [{f['rule']}]" for f in result["findings"][:12]]
                       + result["ratchet_problems"]))
    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **{k: v for k, v in result.items() if k != "unmarked_pages"}},
                         indent=2))
        return 0 if result["passed"] else 1

    print(f"{PROBE_ID}: {status}")
    print(f"  pages in scope           : {result['pages']} (docs/**/*.md except docs/api/, plus "
          f"{len(ROOT_PAGES)} root pages)")
    print(f"  with front matter        : {result['with_front_matter']}")
    print(f"  without front matter     : {result['without_front_matter']} "
          f"(baseline {result['baseline']}, {BASELINE_REL})")
    if result["top_unmarked_directories"]:
        print("  most unmarked pages, by directory:")
        for row in result["top_unmarked_directories"]:
            print(f"    {row['pages']:5d}  {row['directory']}")
    print(f"  evergreen exemptions     : {len(result['exemptions'])}")
    for item in result["exemptions"]:
        print(f"    - {item['page']}:{item['line']} {item['message']}")
    if result["ratchet_problems"]:
        print("  RATCHET:")
        for problem in result["ratchet_problems"]:
            print(f"    - {problem}")
    if result["findings"]:
        print(f"  FINDINGS ({len(result['findings'])}):")
        for f in result["findings"]:
            print(f"    - {f['page']}:{f['line']} [{f['rule']}] {f['message']}")
    if result["passed"]:
        print("  every page that declares front matter is valid, and the unmarked count matches the baseline")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
