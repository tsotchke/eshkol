#!/usr/bin/env python3
"""Release gate: `.icc/doc-claims-allowlist.yaml` claims to be "consumed by
the doc-typed-claims-residual release-oracle criterion", enforcing the
identity `current_wrong_count == len(allowlist entries) + open build-item
count, with zero unexplained remainder`. As of the 2026-08-26 audit (N5),
that criterion existed nowhere: the string `doc-typed-claims-residual`
appeared only in the allowlist file's own header comment, never as an
actual criterion id, script, or wired check anywhere else in the repo. The
allowlist itself had also drifted: it was generated on 2026-08-14 against
an older tree, and by 2026-08-26 `icc doc-typed-claims` measured 101 wrong
claims against 74 allowlisted ones -- 27 genuinely unexplained. This script
is the criterion the allowlist's header always claimed existed, plus the
allowlist was regenerated in the same change (27 stale doc line-count
mentions corrected in the doc text itself; 8 claims that had merely drifted
line number as newer content was prepended above them, re-hashed in place).

BI-19 (v1.3.5 docs audit, 2026-08-28): re-hashing by hand was not a fix, it
was a recurring cost -- the same five dated CHANGELOG/RELEASE_NOTES entries
needed it again on the very next release, and the one after that, for no
reason but new content landing above them and shifting their line number.
`claim_id` (sha1 of path+line+value, computed inside `icc`) is fundamentally
NOT stable under that kind of drift. This gate now ALSO matches a wrong
claim against the allowlist by `stable_claim_key` -- (path, claim_type,
subject, value), with no line in it -- whenever an allowlist entry carries
those fields (see `stable_claim_key` / `allowlisted_stable_keys`). Exact
`claim_id` matching is tried first and kept working unchanged, so an entry
that has not been enriched with claim_type/subject/value yet still matches
by id alone during the transition; a fresh `claim_id` from a later `icc` run
against the very same claim also still matches, by definition, on both keys
at once.

What this checks
    Given a `icc doc-typed-claims --format json` result (see
    `--doc-typed-claims-json`, or `--icc-bin`/`--repo` to run it directly):
      wrong_ids        = claim_ids with status == "wrong"
      allowlisted_ids   = claim_ids named in .icc/doc-claims-allowlist.yaml
      allowlisted_keys  = stable_claim_key(...) of every allowlist entry that
                          carries path/claim_type/subject/value
      open_build_ids    = residual_claim_ids named by an OPEN DOC-DEBT ledger
                          entry in .icc/silent-wrong-ledger.yaml (a maintainer
                          may explicitly track a real, not-yet-fixed doc
                          defect as ongoing work instead of allowlisting a
                          claim the detector could otherwise ground) -- matched
                          by claim_id only; the ledger's own schema is out of
                          this gate's scope to change
      a wrong claim is EXPLAINED if its claim_id is in allowlisted_ids or
                          open_build_ids, OR its own stable_claim_key(...) is
                          in allowlisted_keys
      unexplained       = wrong claims that are not explained
      stale             = an allowlist/ledger entry whose claim_id no longer
                          names a current wrong claim AND (for an allowlist
                          entry) whose stable key no longer matches one
                          either -- not a failure, but worth surfacing so the
                          allowlist does not silently accumulate dead entries
                          forever

Grading (PASS / FAIL / NO_DATA)
    PASS      unexplained is empty (the identity holds exactly).
    FAIL      unexplained is non-empty: a wrong claim was found that this
              repo's own bookkeeping does not account for.
    NO_DATA   no `icc doc-typed-claims` result could be obtained at all
              (no --doc-typed-claims-json given and no resolvable ICC
              binary/repo) -- distinct from PASS; a required step must not
              read green on missing input (same NO_DATA discipline as
              check_required_context_consistency.py / the B5/N2 fix to
              check_evidence_staleness.py). Exit 2.

A `stale` entry does not fail the gate (an allowlist that is ahead of the
detector, e.g. after a doc fix lands, is not a defect), but is reported so
`.icc/doc-claims-allowlist.yaml` can be pruned by hand.

Usage
    python3 scripts/check_doc_claims_residual.py --doc-typed-claims-json out.json
    python3 scripts/check_doc_claims_residual.py --icc-bin /path/to/icc --repo eshkol
    python3 scripts/check_doc_claims_residual.py --self-test

Exit status: 0 PASS, 1 FAIL, 2 NO_DATA.

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ALLOWLIST = os.path.join(REPO_ROOT, ".icc", "doc-claims-allowlist.yaml")
DEFAULT_LEDGER = os.path.join(REPO_ROOT, ".icc", "silent-wrong-ledger.yaml")
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "doc_claims_residual_gate.jsonl"
PROBE_ID = "doc_claims_residual_clean"


class ResidualGateError(Exception):
    """The allowlist or ledger could not be read or parsed at all."""


def _load_yaml(path: str, what: str) -> object:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ResidualGateError("PyYAML is required (pip install pyyaml)") from exc

    if not os.path.isfile(path):
        raise ResidualGateError(f"{what} not found at {path} (the gate fails closed)")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        raise ResidualGateError(f"{what} at {path} is not parseable: {exc}") from exc


def stable_claim_key(path: object, claim_type: object, subject: object, value: object) -> tuple | None:
    """A line-independent identity for a claim: (path, claim_type, subject,
    value). `claim_id` (sha1 of path+line+value, computed inside `icc`) is
    NOT line-independent -- a dated CHANGELOG/RELEASE_NOTES/ROADMAP entry
    re-hashes every time later content is prepended above it and shifts its
    line number, with no change in what the claim actually says. This key
    is what the allowlist should really be tracking; `claim_id` is kept
    alongside it purely for cross-referencing a specific `icc` run.

    Returns None when any component is missing -- callers must treat that
    as "no stable key available", never as a wildcard match.
    """
    if path is None or claim_type is None or subject is None or value is None:
        return None
    if not isinstance(path, str) or not isinstance(claim_type, str) or not isinstance(subject, str):
        return None
    return (path, claim_type, subject, str(value))


def allowlisted_claim_ids(allowlist_path: str) -> dict[str, dict]:
    data = _load_yaml(allowlist_path, "doc-claims allowlist")
    if not isinstance(data, dict):
        raise ResidualGateError("doc-claims allowlist is not a mapping")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ResidualGateError("doc-claims allowlist has no `entries` list")
    out: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cid = entry.get("claim_id")
        if isinstance(cid, str) and cid:
            out[cid] = entry
    return out


def allowlisted_stable_keys(allowlist_path: str) -> dict[tuple, str]:
    """stable_claim_key(...) -> claim_id, for every entry that carries the
    path/claim_type/subject/value fields a stable key needs. An entry
    missing any of those fields (e.g. a hand-written fixture, or a legacy
    entry predating this field set) simply contributes no stable key --
    it is still matched by exact `claim_id`, unchanged."""
    data = _load_yaml(allowlist_path, "doc-claims allowlist")
    if not isinstance(data, dict):
        raise ResidualGateError("doc-claims allowlist is not a mapping")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ResidualGateError("doc-claims allowlist has no `entries` list")
    out: dict[tuple, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cid = entry.get("claim_id")
        key = stable_claim_key(entry.get("path"), entry.get("claim_type"), entry.get("subject"), entry.get("value"))
        if key is not None and isinstance(cid, str) and cid:
            out[key] = cid
    return out


def open_build_item_claim_ids(ledger_path: str) -> dict[str, dict]:
    """`residual_claim_ids` named by any OPEN DOC-DEBT ledger entry -- a
    maintainer-tracked real doc defect that is not (yet) allowlisted. Absent
    ledger or absent DOC-DEBT bucket is not an error: it just means zero
    open build items explain anything, which is the common case."""

    try:
        data = _load_yaml(ledger_path, "silent-wrong ledger")
    except ResidualGateError:
        return {}
    if not isinstance(data, dict):
        return {}
    entries = data.get("entries")
    if not isinstance(entries, list):
        return {}
    out: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("bucket") != "DOC-DEBT" or entry.get("status") != "open":
            continue
        for cid in entry.get("residual_claim_ids") or []:
            if isinstance(cid, str) and cid:
                out[cid] = entry
    return out


def wrong_claim_ids_from_doc_typed_claims(payload: dict) -> dict[str, dict]:
    claims = payload.get("claims")
    if not isinstance(claims, list):
        raise ResidualGateError(
            "doc-typed-claims payload has no `claims` list -- re-run with "
            "--claim-limit/--finding-limit high enough that claims_truncated is false"
        )
    if payload.get("claims_truncated"):
        raise ResidualGateError(
            "doc-typed-claims payload has claims_truncated=true -- re-run "
            "icc doc-typed-claims with a higher --claim-limit"
        )
    out: dict[str, dict] = {}
    for claim in claims:
        if isinstance(claim, dict) and claim.get("status") == "wrong":
            cid = claim.get("claim_id")
            if isinstance(cid, str) and cid:
                out[cid] = claim
    return out


def run_icc_doc_typed_claims(icc_bin: str, repo: str) -> dict:
    cmd = [
        icc_bin, "doc-typed-claims",
        "--repo", repo,
        "--claim-limit", "50000",
        "--finding-limit", "5000",
        "--format", "json",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResidualGateError(f"failed to run {' '.join(cmd)}: {exc}") from exc
    if proc.returncode != 0:
        raise ResidualGateError(
            f"{' '.join(cmd)} exited {proc.returncode}: {proc.stderr.strip()[:500]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ResidualGateError(f"doc-typed-claims did not print JSON: {exc}") from exc


def check(
    allowlist_path: str,
    ledger_path: str,
    doc_typed_claims_payload: dict | None,
) -> dict:
    """Never raises; returns a report with a `status` of PASS/FAIL/NO_DATA."""

    if doc_typed_claims_payload is None:
        return {
            "passed": False,
            "status": "NO_DATA",
            "errors": ["no doc-typed-claims result available (no --doc-typed-claims-json "
                       "and no resolvable --icc-bin/--repo)"],
            "unexplained": [], "stale": [], "wrong_count": None,
            "allowlist_count": None, "open_build_item_count": None,
        }

    allow = allowlisted_claim_ids(allowlist_path)
    allow_stable = allowlisted_stable_keys(allowlist_path)
    open_items = open_build_item_claim_ids(ledger_path)
    wrong = wrong_claim_ids_from_doc_typed_claims(doc_typed_claims_payload)

    # BI-19 (v1.3.5 docs audit, 2026-08-28): `claim_id` is sha1(path+line+
    # value) computed inside `icc`, so a dated CHANGELOG/RELEASE_NOTES/
    # ROADMAP entry re-hashes on every release simply because later content
    # was prepended above it -- the third release running the same five
    # entries needed re-hashing by hand. A wrong claim is now ALSO explained
    # when its line-independent stable_claim_key(...) matches an allowlist
    # entry, so a pure line shift with no change in path/claim_type/subject/
    # value stops being churn. Exact claim_id matching is kept (and tried
    # first) as the migration path: an allowlist entry that has not yet
    # been enriched with claim_type/subject/value still matches by id alone.
    wrong_stable = {
        cid: stable_claim_key(
            claim.get("location", {}).get("path") if isinstance(claim.get("location"), dict) else None,
            claim.get("claim_type"), claim.get("subject"), claim.get("value"),
        )
        for cid, claim in wrong.items()
    }

    def _explained(cid: str) -> bool:
        if cid in allow or cid in open_items:
            return True
        key = wrong_stable.get(cid)
        return key is not None and key in allow_stable

    unexplained = sorted(cid for cid in wrong if not _explained(cid))

    # `stale`: an allowlist/ledger entry that no longer corresponds to any
    # CURRENT wrong claim, by id NOR by stable key -- true dead weight, safe
    # to prune. An entry whose id has drifted but whose stable key still
    # matches a current wrong claim is NOT stale; it is doing its job.
    live_stable_keys = {k for k in wrong_stable.values() if k is not None}
    explained_ids = set(allow) | set(open_items)

    def _still_relevant(cid: str) -> bool:
        if cid in wrong:
            return True
        entry = allow.get(cid)
        key = stable_claim_key(entry.get("path"), entry.get("claim_type"), entry.get("subject"),
                                entry.get("value")) if entry else None
        return key is not None and key in live_stable_keys

    stale = sorted(cid for cid in explained_ids if not _still_relevant(cid))

    errors = [
        f"{cid}: {wrong[cid]['location']['path']}:{wrong[cid]['location']['line']} "
        f"({wrong[cid].get('claim_type')} {wrong[cid].get('subject')}={wrong[cid].get('value')}) "
        "is wrong but neither allowlisted nor an open DOC-DEBT build item"
        for cid in unexplained
    ]

    return {
        "passed": not errors,
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "unexplained": unexplained,
        "stale": stale,
        "wrong_count": len(wrong),
        "allowlist_count": len(allow),
        "open_build_item_count": len(open_items),
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

def _fake_payload(wrong_ids: list[str]) -> dict:
    return {
        "claims_truncated": False,
        "claims": [
            {
                "claim_id": cid, "status": "wrong", "claim_type": "numeric",
                "subject": "fake.cpp", "value": 1,
                "location": {"path": "FAKE.md", "line": 1},
            }
            for cid in wrong_ids
        ],
    }


def self_test() -> bool:
    cases: list[tuple[str, bool]] = []

    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-doc-claims-residual-") as tmp:
        allow_path = os.path.join(tmp, "allowlist.yaml")
        ledger_path = os.path.join(tmp, "ledger.yaml")

        with open(allow_path, "w", encoding="utf-8") as f:
            f.write("entries:\n- claim_id: aaa\n  path: X\n  line: 1\n  disposition: EXEMPT-HISTORICAL\n"
                    "- claim_id: bbb\n  path: Y\n  line: 2\n  disposition: DETECTOR-MISATTRIBUTION\n")
        with open(ledger_path, "w", encoding="utf-8") as f:
            f.write("entries:\n- id: DD-99\n  bucket: DOC-DEBT\n  status: open\n"
                    "  residual_claim_ids: [ccc]\n"
                    "- id: DD-98\n  bucket: DOC-DEBT\n  status: closed\n"
                    "  residual_claim_ids: [ddd]\n")

        # Case 1: every wrong id is explained (allowlist + open build item) -> PASS.
        result = check(allow_path, ledger_path, _fake_payload(["aaa", "bbb", "ccc"]))
        cases.append(("fully_explained_passes", result["passed"] is True and result["status"] == "PASS"
                      and not result["unexplained"]))

        # Case 2 (the actual regression this gate exists for): a wrong id with
        # no allowlist entry and no open build item -> FAIL, and it must NAME
        # the offending claim.
        result = check(allow_path, ledger_path, _fake_payload(["aaa", "bbb", "zzz"]))
        cases.append(("unexplained_wrong_claim_fails", result["passed"] is False
                      and result["status"] == "FAIL" and result["unexplained"] == ["zzz"]))

        # Case 3: a CLOSED DOC-DEBT entry's residual_claim_ids do NOT count as
        # explaining anything -- only OPEN entries may excuse a wrong claim.
        result = check(allow_path, ledger_path, _fake_payload(["aaa", "bbb", "ddd"]))
        cases.append(("closed_ledger_entry_does_not_explain", result["passed"] is False
                      and result["unexplained"] == ["ddd"]))

        # Case 4: an allowlisted id that is no longer wrong is reported as
        # `stale`, not a failure.
        result = check(allow_path, ledger_path, _fake_payload(["aaa"]))
        cases.append(("stale_allowlist_entry_is_not_a_failure", result["passed"] is True
                      and "bbb" in result["stale"]))

        # Case 5: no doc-typed-claims payload at all -> NO_DATA, not PASS.
        result = check(allow_path, ledger_path, None)
        cases.append(("missing_payload_is_no_data_not_pass", result["passed"] is False
                      and result["status"] == "NO_DATA"))

        # BI-19 (2026-08-28): the whole point of stable_claim_key -- an
        # allowlist entry enriched with path/claim_type/subject/value
        # explains a wrong claim whose claim_id and line have both moved,
        # as long as the (path, claim_type, subject, value) tuple is
        # unchanged. "eee" here is a DIFFERENT claim_id than any allowlist
        # entry's claim_id (simulating the exact re-hash-on-every-release
        # churn this fix retires); it must still be explained.
        stable_allow_path = os.path.join(tmp, "allowlist_stable.yaml")
        with open(stable_allow_path, "w", encoding="utf-8") as f:
            f.write(
                "entries:\n"
                "- claim_id: old_id_at_a_stale_line\n"
                "  path: CHANGELOG.md\n"
                "  line: 42\n"
                "  claim_type: numeric\n"
                "  subject: some_file.cpp\n"
                "  value: 1234\n"
                "  disposition: EXEMPT-HISTORICAL\n"
            )

        def _fake_payload_full(claims: list[dict]) -> dict:
            return {"claims_truncated": False, "claims": claims}

        drifted_claim = {
            "claim_id": "eee", "status": "wrong", "claim_type": "numeric",
            "subject": "some_file.cpp", "value": 1234,
            "location": {"path": "CHANGELOG.md", "line": 917},  # line drifted, everything else didn't
        }
        result = check(stable_allow_path, ledger_path, _fake_payload_full([drifted_claim]))
        cases.append(("line_drifted_claim_explained_by_stable_key", result["passed"] is True
                      and result["status"] == "PASS" and not result["unexplained"]))

        # A claim with the SAME path/type/subject but a DIFFERENT value is a
        # genuinely different claim and must NOT be explained by the stable
        # key of an unrelated one.
        different_value_claim = dict(drifted_claim, claim_id="fff", value=9999)
        result = check(stable_allow_path, ledger_path, _fake_payload_full([different_value_claim]))
        cases.append(("different_value_not_explained_by_stable_key", result["passed"] is False
                      and result["unexplained"] == ["fff"]))

        # Once no wrong claim's stable key matches the entry at all, the
        # entry is genuinely `stale` -- id AND stable key both dead. (An
        # empty ledger here, not the shared one above, which still has its
        # own open DOC-DEBT item that would independently show up stale too.)
        empty_ledger_path = os.path.join(tmp, "empty_ledger.yaml")
        with open(empty_ledger_path, "w", encoding="utf-8") as f:
            f.write("entries: []\n")
        result = check(stable_allow_path, empty_ledger_path, _fake_payload_full([]))
        cases.append(("fully_dead_entry_is_stale", result["passed"] is True
                      and result["stale"] == ["old_id_at_a_stale_line"]))

        # Case 6: a truncated claims list is a hard error (FAIL), never a
        # silent undercount.
        truncated = _fake_payload(["aaa"])
        truncated["claims_truncated"] = True
        raised = False
        try:
            check(allow_path, ledger_path, truncated)
        except ResidualGateError:
            raised = True
        cases.append(("truncated_claims_payload_is_rejected", raised))

    all_ok = True
    print("check_doc_claims_residual.py self-test:")
    for name, ok in cases:
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] {name}")
    if all_ok:
        print("self-test: PASS -- the gate goes RED on an unexplained wrong claim and does not "
              "false-positive on explained, stale, or missing-payload cases")
    else:
        print("self-test: FAIL -- the residual gate did not discriminate correctly", file=sys.stderr)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allowlist", default=os.environ.get("ESHKOL_DOC_CLAIMS_ALLOWLIST", DEFAULT_ALLOWLIST))
    parser.add_argument("--ledger", default=os.environ.get("ESHKOL_SILENT_WRONG_LEDGER", DEFAULT_LEDGER))
    parser.add_argument("--doc-typed-claims-json", default=None,
                         help="path to a pre-generated `icc doc-typed-claims --format json` result")
    parser.add_argument("--icc-bin", default=os.environ.get("ICC_BIN"),
                         help="ICC binary to invoke directly when --doc-typed-claims-json is not given")
    parser.add_argument("--repo", default=os.environ.get("ICC_REPO"),
                         help="ICC repo registration name for --icc-bin")
    parser.add_argument("--trace-dir", default=os.environ.get("ESHKOL_TRACE_DIR", DEFAULT_TRACE_DIR))
    parser.add_argument("--emit-trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    try:
        payload: dict | None = None
        if args.doc_typed_claims_json:
            with open(args.doc_typed_claims_json, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        elif args.icc_bin and args.repo and os.path.isfile(args.icc_bin) and os.access(args.icc_bin, os.X_OK):
            payload = run_icc_doc_typed_claims(args.icc_bin, args.repo)

        result = check(args.allowlist, args.ledger, payload)
    except ResidualGateError as exc:
        snippet = f"gate error: {exc}"
        if not args.no_trace:
            emit_trace(args.emit_trace_dir, "FAIL", snippet)
        if args.format == "json":
            print(json.dumps({"status": "FAIL", "passed": False, "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: FAIL -- {exc}", file=sys.stderr)
        return 1

    status = result["status"]
    if status == "PASS":
        snippet = (f"{result['wrong_count']} wrong claims, all explained "
                   f"({result['allowlist_count']} allowlisted + {result['open_build_item_count']} "
                   f"open build items); {len(result['stale'])} stale entries")
    elif status == "NO_DATA":
        snippet = result["errors"][0] if result["errors"] else "no doc-typed-claims result available"
    else:
        snippet = f"{len(result['unexplained'])} unexplained wrong claim(s): " + "; ".join(result["errors"][:5])

    if not args.no_trace:
        emit_trace(args.emit_trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({**result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  allowlist      : {args.allowlist}")
        print(f"  ledger         : {args.ledger}")
        if result.get("wrong_count") is not None:
            print(f"  wrong claims   : {result['wrong_count']}")
            print(f"  allowlisted    : {result['allowlist_count']}")
            print(f"  open build items: {result['open_build_item_count']}")
            if result["stale"]:
                print(f"  stale (no longer wrong, safe to prune): {len(result['stale'])}")
                for cid in result["stale"]:
                    print(f"    - {cid}")
        if result["errors"]:
            print("  ERRORS:")
            for error in result["errors"]:
                print(f"    - {error}")
        if status == "NO_DATA":
            print("  NO_DATA is not a pass: nothing was graded.", file=sys.stderr)

    if status == "FAIL":
        return 1
    if status == "NO_DATA":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
