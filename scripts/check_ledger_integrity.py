#!/usr/bin/env python3
"""Release gate: the silent-wrong flaw ledger is a well-formed ledger.

`.icc/silent-wrong-ledger.yaml` is hand-edited by every branch that opens or
closes a flaw, and it is merged like any other text file: two branches that
each add an entry near the same place produce a textual merge, not a content
merge, and git happily accepts it even when the result is unusable as a
ledger. This gate is the check that a bare `git merge` cannot perform.

It answers three questions a YAML parser alone does not:

  1. Does the file parse at all? A stray `- id:` at the wrong indent, an
     unclosed quote, or a duplicated block key breaks every consumer
     downstream (gate_no_silent_wrong.py, the completion-oracle criteria that
     read its trace) with no signal beyond a stack trace nobody was watching
     for.

  2. Does every id appear exactly once ACROSS THE WHOLE LEDGER? A textual
     merge does not conflict when two branches each independently pick the
     same next-free id — both edits "apply" cleanly and the file still
     parses. The result is two entries answering to one name: a lookup finds
     whichever one comes first, edits meant for one land on the other, and a
     human auditing "is SW-42 closed" gets an arbitrary answer. This is not
     hypothetical: SW-33 was independently allocated three times, SW-35 twice
     and SW-42 twice, on stacked branches that never conflicted with each
     other. The ledger's own `renumbered_from` fields record how each
     collision was eventually resolved by hand; this gate exists so the next
     one is caught before merge instead of after.

  3. Does every entry carry the minimum shape a reader (or gate_no_silent_
     wrong.py) needs to make sense of it: an `id`, a `bucket` that is one of
     the buckets this file itself declares, a `status` from the vocabulary
     the gate understands, a `title`, and — for any entry whose status is not
     `open` — at least one closure-evidence field naming what closed it
     (`closed_at`, `evidence`, `fixed_by`, `resolution` or `guarded_by`).
     "Believed fixed" with nothing to point at is not evidence; the ledger's
     own header states this as INVARIANT 1 and 3.

  4. (BI-23, v1.3.5 docs audit, 2026-08-28 -- ADVISORY, does not affect
     grading) Is `closed_at`, where present, actually the kind of evidence
     INVARIANT 3 asks for -- "closed by measurement at a named SHA"? Two
     ways it silently is not: `closed_at` names a BRANCH rather than a
     commit (a moving pointer, not a fixed point in history), or names a
     real commit that is not an ancestor of master (a squash-merge
     artifact -- the entry's own PR merged, but this exact SHA did not).
     `git log --all` does not help a reader notice either case, and neither
     breaks the structural checks above. This is reported as WARNINGS with
     counts (`closure_provenance` in the result / --format json output),
     never as a FAIL this release -- see `closure_provenance_warnings`.

Grading
    PASS  the file parses, has no duplicate id anywhere in `entries`, and
          every entry satisfies the minimum shape above. (Closure
          provenance warnings, item 4, do NOT affect this verdict.)
    FAIL  a parse error, any duplicate id, or any entry missing a required
          field / carrying a bucket or status this gate does not recognise /
          closed-like with no closure evidence.

The gate FAILS CLOSED: a missing or unparseable ledger is FAIL, never a silent
pass, for the same reason gate_no_silent_wrong.py fails closed — absence of
the ledger is not evidence of an absence of defects in it.

Usage
    python3 scripts/check_ledger_integrity.py
    python3 scripts/check_ledger_integrity.py --ledger path/to/ledger.yaml
    python3 scripts/check_ledger_integrity.py --repo-root . --master-ref origin/master
    python3 scripts/check_ledger_integrity.py --no-provenance-check
    python3 scripts/check_ledger_integrity.py --format json
    python3 scripts/check_ledger_integrity.py --self-test

Exit status is 0 on PASS and 1 on FAIL (including under --self-test, where it
reports whether the gate itself is capable of failing).

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LEDGER = os.path.join(REPO_ROOT, ".icc", "silent-wrong-ledger.yaml")
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "ledger_integrity_gate.jsonl"
PROBE_ID = "ledger_integrity_clean"
DEFAULT_MASTER_REF = "origin/master"

KNOWN_STATUSES = ("open", "closed", "fixed", "guarded")
TERMINAL_STATUSES = ("closed", "fixed", "guarded")
CLOSURE_EVIDENCE_FIELDS = ("closed_at", "evidence", "fixed_by", "resolution", "guarded_by")
REQUIRED_ENTRY_FIELDS = ("id", "bucket", "status", "title")

# BI-23 (v1.3.5 docs audit, 2026-08-28): a git commit SHA, short or full.
SHA_LIKE_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)


class LedgerIntegrityError(Exception):
    """The ledger could not be read or parsed at all."""


def _load_yaml(path: str) -> object:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LedgerIntegrityError(
            "PyYAML is required to check the ledger (pip install pyyaml)"
        ) from exc

    if not os.path.isfile(path):
        raise LedgerIntegrityError(f"ledger not found at {path} (the gate fails closed)")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        raise LedgerIntegrityError(f"ledger at {path} is not parseable: {exc}") from exc


def _resolve_ref(repo_root: str, ref: str) -> str | None:
    """The commit SHA `ref` resolves to in `repo_root`, or None if it does
    not resolve (no such ref, not a git repo, git not installed, etc.) --
    every failure mode here is "cannot check", never an exception."""
    try:
        proc = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "--verify", "--quiet", ref + "^{commit}"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def closure_provenance_warnings(entries: list, repo_root: str, master_ref: str = DEFAULT_MASTER_REF) -> dict:
    """BI-23 (v1.3.5 docs audit, 2026-08-28): the ledger's own invariant 3
    requires a terminal-status entry to be closed "by measurement at a named
    SHA". Two ways that invariant is violated in practice, neither of which
    breaks YAML structure or the required-field check above, so neither was
    ever caught:

      - `closed_at` names a BRANCH, not a commit -- a branch is a moving
        pointer; the entry's evidence silently drifts out from under it the
        next time that branch advances (or is deleted after merge).
      - `closed_at` names a real commit that is a squash-merge artifact --
        it exists in the object database but was never actually integrated
        into `master`, so "closed at <sha>" cannot be verified by walking
        master's own history.

    This is advisory, not a structural defect: unlike a duplicate id or a
    missing field, it does not make the ledger unusable, and the maintainer
    may have a good reason (a squash-merge commit is real work, just not an
    ancestor). So this WARNS (`warnings`, counts) rather than failing the
    gate -- deliberately not added to `errors`/`passed` -- for this release.
    Returns counts even when nothing could be checked (no repo_root git
    checkout, unresolvable master_ref) so a caller can tell "checked, zero
    problems" apart from "could not check anything".
    """
    warnings: list[str] = []
    branch_like = 0
    non_ancestor = 0
    unresolved = 0
    master_sha = _resolve_ref(repo_root, master_ref)

    for raw in entries:
        if not isinstance(raw, dict):
            continue
        status = raw.get("status")
        closed_at = raw.get("closed_at")
        if status not in TERMINAL_STATUSES or not closed_at:
            continue
        entry_id = raw.get("id", "?")
        closed_at_s = str(closed_at)

        if not SHA_LIKE_RE.match(closed_at_s):
            branch_like += 1
            warnings.append(
                f"{entry_id}: closed_at={closed_at_s!r} looks like a branch name, not a commit SHA "
                "(a branch is a moving pointer -- record the SHA it pointed to when the entry closed)"
            )
            continue

        if master_sha is None:
            unresolved += 1
            continue  # nothing to compare against; not a defect finding either way

        entry_sha = _resolve_ref(repo_root, closed_at_s)
        if entry_sha is None:
            non_ancestor += 1
            warnings.append(f"{entry_id}: closed_at={closed_at_s} does not resolve to a known commit here")
            continue

        try:
            proc = subprocess.run(
                ["git", "-C", repo_root, "merge-base", "--is-ancestor", entry_sha, master_sha],
                capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            unresolved += 1
            continue

        if proc.returncode == 1:
            non_ancestor += 1
            warnings.append(
                f"{entry_id}: closed_at={closed_at_s} is not an ancestor of {master_ref} "
                f"({master_sha[:8]}) -- likely a squash-merge artifact"
            )
        elif proc.returncode not in (0, 1):
            unresolved += 1

    return {
        "warnings": warnings,
        "branch_like_count": branch_like,
        "non_ancestor_count": non_ancestor,
        "unresolved_count": unresolved,
        "master_ref": master_ref,
        "master_sha": master_sha,
    }


def check(
    data: object,
    repo_root: str = REPO_ROOT,
    master_ref: str = DEFAULT_MASTER_REF,
    check_provenance: bool = True,
) -> dict:
    """Validate a parsed ledger document. Never raises; returns a report."""

    errors: list[str] = []

    if not isinstance(data, dict):
        return {"passed": False, "errors": ["ledger document is not a mapping"],
                "entry_count": 0, "duplicate_ids": {}, "bucket_counts": {},
                "closure_provenance": None}

    buckets = data.get("buckets")
    declared_buckets = set(buckets.keys()) if isinstance(buckets, dict) else set()
    if not declared_buckets:
        errors.append("ledger has no `buckets` mapping (or it is empty) — nothing to validate entry buckets against")

    entries = data.get("entries")
    if not isinstance(entries, list):
        errors.append("ledger has no `entries` list")
        entries = []

    seen_ids: dict[str, int] = {}
    duplicate_ids: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}

    for position, raw in enumerate(entries, start=1):
        if not isinstance(raw, dict):
            errors.append(f"entry #{position} is not a mapping")
            continue

        label = f"entry #{position}"
        entry_id = raw.get("id")
        if entry_id:
            label = f"entry #{position} ({entry_id!r})"
            key = str(entry_id)
            if key in seen_ids:
                duplicate_ids[key] = duplicate_ids.get(key, seen_ids[key]) + 1
            else:
                seen_ids[key] = 1

        missing = [f for f in REQUIRED_ENTRY_FIELDS if not raw.get(f)]
        if missing:
            errors.append(f"{label} is missing required field(s): {', '.join(missing)}")
            # Still keep validating what IS present below — a missing id
            # should not hide an unrelated missing-evidence problem.

        bucket = raw.get("bucket")
        if bucket:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            if declared_buckets and bucket not in declared_buckets:
                errors.append(f"{label} has bucket {bucket!r}, which is not declared in `buckets`")

        status = raw.get("status")
        if status is not None and status not in KNOWN_STATUSES:
            errors.append(
                f"{label} has status {status!r}, expected one of {'/'.join(KNOWN_STATUSES)}"
            )
        elif status in TERMINAL_STATUSES:
            if not any(raw.get(field) for field in CLOSURE_EVIDENCE_FIELDS):
                errors.append(
                    f"{label} has status {status!r} but no closure evidence "
                    f"({'/'.join(CLOSURE_EVIDENCE_FIELDS)})"
                )

    # Duplicate ids that only ever collide with themselves are recorded once
    # above (seen_ids counts the FIRST occurrence, duplicate_ids the rest);
    # normalise to "how many times each id appears" for reporting.
    occurrence_counts: dict[str, int] = {}
    for raw in entries:
        if isinstance(raw, dict) and raw.get("id"):
            key = str(raw["id"])
            occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
    true_duplicates = {k: v for k, v in occurrence_counts.items() if v > 1}
    for entry_id, count in sorted(true_duplicates.items()):
        errors.append(f"id {entry_id!r} is used by {count} entries (must be unique across the whole ledger)")

    closure_provenance = closure_provenance_warnings(entries, repo_root, master_ref) if check_provenance else None

    passed = not errors  # closure_provenance is advisory: it never affects `passed`.
    return {
        "passed": passed,
        "errors": errors,
        "entry_count": len(entries),
        "duplicate_ids": true_duplicates,
        "bucket_counts": bucket_counts,
        "closure_provenance": closure_provenance,
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


# ───────────────────────────── self-test ─────────────────────────────
#
# "A gate that cannot fail is not a gate." Each fixture below feeds the gate
# deliberately-broken input and asserts it grades FAIL; one well-formed
# fixture asserts the gate does NOT grade every input FAIL regardless of
# content (a gate that always fails is exactly as useless as one that never
# does).

_GOOD_LEDGER = """
schema: eshkol.silent_wrong_ledger.v1
buckets:
  SILENT-WRONG: "wrong value, no diagnostic, exit 0"
entries:
  - id: SW-SELFTEST-01
    bucket: SILENT-WRONG
    status: closed
    title: "self-test entry, closed with evidence"
    closed_at: "deadbeef"
  - id: SW-SELFTEST-02
    bucket: SILENT-WRONG
    status: open
    title: "self-test entry, open"
"""

_MALFORMED_YAML = """
schema: eshkol.silent_wrong_ledger.v1
entries:
  - id: SW-X
    bucket: SILENT-WRONG
      status: open
    title: "broken indentation opens a block-mapping the parser cannot close"
"""

_DUPLICATE_ID_LEDGER = """
schema: eshkol.silent_wrong_ledger.v1
buckets:
  SILENT-WRONG: "wrong value, no diagnostic, exit 0"
entries:
  - id: SW-42
    bucket: SILENT-WRONG
    status: closed
    title: "first SW-42, allocated on one branch"
    closed_at: "aaaaaaaa"
  - id: SW-42
    bucket: SILENT-WRONG
    status: open
    title: "second SW-42, allocated independently on another branch"
"""

_MISSING_FIELD_LEDGER = """
schema: eshkol.silent_wrong_ledger.v1
buckets:
  SILENT-WRONG: "wrong value, no diagnostic, exit 0"
entries:
  - id: SW-INCOMPLETE
    status: open
    title: "entry with no bucket at all"
  - id: SW-UNEVIDENCED
    bucket: SILENT-WRONG
    status: closed
    title: "closed with nothing pointing at what closed it"
"""


def _run_fixture(name: str, text: str, tmp_dir: str) -> tuple[bool, str]:
    path = os.path.join(tmp_dir, name + ".yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    try:
        data = _load_yaml(path)
    except LedgerIntegrityError as exc:
        return False, f"parse error (expected for a malformed-YAML fixture): {exc}"
    # check_provenance=False: these fixtures exercise structural validity
    # only; closure_provenance_warnings has its own dedicated self-test
    # fixtures below, against a throwaway git repo.
    result = check(data, check_provenance=False)
    return result["passed"], "; ".join(result["errors"]) or "no errors"


def self_test() -> bool:
    """Run the gate against fixtures with known-bad and known-good shape.

    Fixtures live in a temp directory INSIDE the repo (never /tmp) so the
    gate is exercised against real files on disk exactly as it runs in CI,
    and the directory is removed before this function returns.
    """

    cases = [
        ("malformed_yaml", _MALFORMED_YAML, False),
        ("duplicate_id", _DUPLICATE_ID_LEDGER, False),
        ("missing_field", _MISSING_FIELD_LEDGER, False),
        ("well_formed", _GOOD_LEDGER, True),
    ]

    all_ok = True
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-ledger-integrity-") as tmp_dir:
        print("check_ledger_integrity.py self-test:")
        for name, text, expect_pass in cases:
            passed, detail = _run_fixture(name, text, tmp_dir)
            ok = passed == expect_pass
            all_ok = all_ok and ok
            verdict = "OK" if ok else "GATE IS BROKEN"
            print(f"  [{verdict}] {name}: expected passed={expect_pass}, got passed={passed}")
            print(f"           {detail}")

    # BI-23 (2026-08-28): closure_provenance_warnings, exercised against a
    # real throwaway git repo (never /tmp) since "is an ancestor of master"
    # is a git-history question a YAML fixture alone cannot pose. This is
    # advisory (WARN), so its own self-test checks the WARNINGS/counts it
    # produces, not `passed` -- a provenance problem must never fail this
    # gate this release.
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-ledger-provenance-") as repo_dir:
        def _git(*args: str) -> None:
            subprocess.run(["git", "-C", repo_dir, *args], check=True, capture_output=True, text=True)

        _git("init", "--quiet", "-b", "main")
        _git("config", "user.email", "selftest@example.invalid")
        _git("config", "user.name", "selftest")
        with open(os.path.join(repo_dir, "f.txt"), "w", encoding="utf-8") as f:
            f.write("one\n")
        _git("add", "f.txt")
        _git("commit", "--quiet", "-m", "on-master")
        on_master_sha = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        _git("branch", "--quiet", "master")  # what _resolve_ref("origin/master") will be faked as below

        # A commit that exists but is NOT reachable from master (an orphan
        # branch, standing in for a squash-merge artifact).
        _git("checkout", "--quiet", "--orphan", "orphan-branch")
        with open(os.path.join(repo_dir, "g.txt"), "w", encoding="utf-8") as f:
            f.write("two\n")
        _git("add", "g.txt")
        _git("commit", "--quiet", "-m", "off-master")
        off_master_sha = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()

        entries = [
            {"id": "P-ON-MASTER", "status": "closed", "closed_at": on_master_sha},
            {"id": "P-OFF-MASTER", "status": "closed", "closed_at": off_master_sha},
            {"id": "P-BRANCH-NAME", "status": "closed", "closed_at": "feat/some-branch"},
            {"id": "P-OPEN-IGNORED", "status": "open", "closed_at": "feat/irrelevant-not-terminal"},
            {"id": "P-NO-EVIDENCE", "status": "guarded", "guarded_by": "a runtime check"},  # no closed_at at all
        ]
        # "master" here is the local `master` branch created above, standing
        # in for the real gate's default of "origin/master" — the function
        # takes any ref, it does not hardcode "origin/".
        report = closure_provenance_warnings(entries, repo_dir, master_ref="master")

        prov_cases = [
            ("on_master_no_warning", not any("P-ON-MASTER" in w for w in report["warnings"])),
            ("off_master_warns_non_ancestor", any("P-OFF-MASTER" in w and "not an ancestor" in w
                                                   for w in report["warnings"])),
            ("branch_name_warns", any("P-BRANCH-NAME" in w and "branch name" in w for w in report["warnings"])),
            ("open_status_ignored", not any("P-OPEN-IGNORED" in w for w in report["warnings"])),
            ("no_closed_at_ignored", not any("P-NO-EVIDENCE" in w for w in report["warnings"])),
            ("counts_match", report["branch_like_count"] == 1 and report["non_ancestor_count"] == 1),
            ("master_ref_resolved", report["master_sha"] is not None),
        ]
        for name, ok in prov_cases:
            all_ok = all_ok and ok
            verdict = "OK" if ok else "GATE IS BROKEN"
            print(f"  [{verdict}] closure_provenance/{name}")
        if not all_ok:
            print(f"           warnings={report['warnings']}", file=sys.stderr)

        # An unresolvable master_ref must degrade to "ancestry not checked",
        # not a crash and not a false claim that the non-ancestor commit is
        # fine -- but the branch-name check does not depend on master_ref
        # at all, so it still fires.
        empty_report = closure_provenance_warnings(entries, repo_dir, master_ref="no-such-ref-anywhere")
        no_ref_ok = (empty_report["master_sha"] is None
                     and empty_report["branch_like_count"] == 1
                     and empty_report["non_ancestor_count"] == 0
                     and not any("P-OFF-MASTER" in w for w in empty_report["warnings"]))
        all_ok = all_ok and no_ref_ok
        print(f"  [{'OK' if no_ref_ok else 'GATE IS BROKEN'}] closure_provenance/unresolvable_master_ref_degrades_gracefully")

    if all_ok:
        print("self-test: PASS — the gate fails on every broken fixture and passes the well-formed one")
    else:
        print("self-test: FAIL — the gate did not discriminate broken input from good input", file=sys.stderr)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", default=os.environ.get("ESHKOL_FLAW_LEDGER", DEFAULT_LEDGER))
    parser.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(
        os.environ.get("ESHKOL_FLAW_LEDGER", DEFAULT_LEDGER))),
        help="git checkout to resolve closed_at SHAs and --master-ref against (default: this ledger's repo)")
    parser.add_argument("--master-ref", default=DEFAULT_MASTER_REF,
                         help="git ref every closed_at SHA is expected to be an ancestor of (BI-23)")
    parser.add_argument("--no-provenance-check", action="store_true",
                         help="skip the closed_at branch-name/non-ancestor WARN check (BI-23) entirely")
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    try:
        data = _load_yaml(args.ledger)
        result = check(data, repo_root=args.repo_root, master_ref=args.master_ref,
                       check_provenance=not args.no_provenance_check)
    except LedgerIntegrityError as exc:
        snippet = f"ledger unusable: {exc}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "FAIL", snippet)
        if args.format == "json":
            print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: FAIL — {exc}", file=sys.stderr)
        return 1

    status = "PASS" if result["passed"] else "FAIL"
    if result["passed"]:
        snippet = (
            f"{result['entry_count']} entries, no duplicate ids, "
            f"all required fields present across {len(result['bucket_counts'])} buckets"
        )
    else:
        snippet = f"{len(result['errors'])} integrity error(s): " + "; ".join(result["errors"][:5])

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  ledger      : {args.ledger}")
        print(f"  entries     : {result['entry_count']}")
        for bucket, count in sorted(result.get("bucket_counts", {}).items()):
            print(f"  {bucket:<24}: {count}")
        if result["duplicate_ids"]:
            print("  DUPLICATE IDS:")
            for entry_id, count in sorted(result["duplicate_ids"].items()):
                print(f"    - {entry_id}  used by {count} entries")
        if result["errors"]:
            print("  ERRORS:")
            for error in result["errors"]:
                print(f"    - {error}")
        prov = result.get("closure_provenance")
        if prov:
            if prov["master_sha"] is None:
                print(f"  closure provenance: NOT CHECKED ({prov['master_ref']} did not resolve here)")
            else:
                print(f"  closure provenance vs {prov['master_ref']} ({prov['master_sha'][:8]}): "
                      f"{prov['branch_like_count']} branch-name closed_at, "
                      f"{prov['non_ancestor_count']} non-ancestor closed_at, "
                      f"{prov['unresolved_count']} unresolved (WARN only, does not fail this gate)")
            if prov["warnings"]:
                print("  CLOSURE PROVENANCE WARNINGS (advisory, BI-23):")
                for warning in prov["warnings"]:
                    print(f"    - {warning}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
