#!/usr/bin/env python3
"""Release gate: every branch-protection required status context is
REPORTABLE on every PR shape this repo's CI can produce.

Motivating incident: `.github/workflows/ci.yml`'s `docs-only-required-
context-stubs` job (added by PR #455) exists because GitHub Actions does
not expand a skipped matrix job's `${{ matrix.name }}` into per-leg check
runs — a skipped `unix-matrix`/`windows-matrix` leg reports exactly one
check run named with the literal, unresolved string `"${{ matrix.name }}"`,
which matches no required context. PR #455's stub job worked around this
by running its OWN matrix, over the literal names of the 7 contexts that
were required at the time, whenever `docs_only == 'true'`. Branch
protection later grew 5 more matrix-derived required contexts
(windows-arm64-lite, macos-arm64-xla, macos-x64-xla, macos-arm64-lite,
macos-x64-lite) and nobody extended the stub matrix to match — the same
permanent-block failure PR #455 fixed, reopened for those 5, silently:
`docs-only-required-context-stubs` and `.github/branches/master/
protection`'s required list are two lists a human keeps in sync by hand,
with nothing that failed a build when they drifted. With
`enforce_admins: true` now set, a drift like this makes a docs-only PR
permanently unmergeable by ANYONE, not just non-admins.

This gate is that missing consistency check. It computes, by parsing the
actual workflow YAML (never by regexing job text), the set of status
contexts that are REPORTABLE on every PR shape:

    required_contexts SUBSET-OF  unconditional_contexts
                                  UNION stub_matrix_contexts
                                  UNION (skippable_contexts INTERSECT stub_matrix_contexts)

  - `unconditional_contexts`: jobs whose `if:` never depends on the
    docs-only predicate (includes jobs from a workflow, such as
    identity-guard.yml, that has no docs-only concept at all — "a workflow
    with no docs paths-ignore" is exactly a workflow where every job
    classifies as unconditional here).
  - `skippable_contexts`: jobs whose `if:` is gated on
    `needs.changes.outputs.docs_only == 'false'` — these do not run, and
    so cannot report, on a docs-only PR UNLESS a stub also covers the name.
  - `stub_matrix_contexts`: the name list of whichever job's `if:` is
    gated on `docs_only == 'true'` (the stub job) — it runs ONLY on
    docs-only PRs, so it complements the skippable set exactly when the
    two name sets match.

A required context that is in `skippable_contexts` but NOT stubbed is
unreportable on a docs-only PR. A required context that appears in NEITHER
set at all is unreportable on ANY PR shape — a name nothing in CI could
ever emit, which is at least as bad (a required context that can never
even go green once, on any diff).

Where the required set comes from
    LIVE      `GET /repos/{repo}/branches/{branch}/protection` via a
              GITHUB_TOKEN/GH_TOKEN in the environment (or `--token`), or
              via the `gh` CLI if it is already authenticated. Tried in
              that order; either failure falls through to the next method
              rather than crashing.
    FALLBACK  a committed snapshot (`.icc/required-status-contexts.json`)
              used when no token and no authenticated `gh` are available.
              This is NOT re-verified against live branch protection by
              this script — it is only as fresh as whoever last updated it
              by hand after a `gh api .../protection` change. FALLBACK mode
              says so on every run.
    NO_DATA   neither LIVE nor FALLBACK produced a required-context list
              at all (no token, no `gh`, and no readable fallback file).
              This is NOT a PASS: nothing has been verified. NO_DATA exits
              2, distinct from PASS (0) and FAIL (1), specifically so a
              caller cannot mistake "we never checked" for "we checked and
              it's fine" — the exact vacuous-assurance shape this gate
              itself was written to close elsewhere in this file's history.

Grading
    PASS      LIVE or FALLBACK produced a required-context list, and every
              context in it is reportable per the SUBSET-OF rule above.
    FAIL      LIVE or FALLBACK produced a required-context list, and at
              least one context in it is not reportable — OR a workflow
              file is missing/unparseable (fails closed, same as this
              repo's other gates).
    NO_DATA   no required-context list could be obtained at all. Exit 2.

Usage
    python3 scripts/check_required_context_consistency.py
    python3 scripts/check_required_context_consistency.py --offline
    python3 scripts/check_required_context_consistency.py --format json
    python3 scripts/check_required_context_consistency.py --self-test

Exit status: 0 PASS, 1 FAIL, 2 NO_DATA.

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WORKFLOWS = [
    os.path.join(REPO_ROOT, ".github", "workflows", "ci.yml"),
    os.path.join(REPO_ROOT, ".github", "workflows", "identity-guard.yml"),
]
DEFAULT_FALLBACK_FILE = os.path.join(REPO_ROOT, ".icc", "required-status-contexts.json")
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
TRACE_BASENAME = "required_context_consistency_gate.jsonl"
PROBE_ID = "required_context_consistency_clean"

DEFAULT_REPO = "tsotchke/eshkol"
DEFAULT_BRANCH = "master"

DOCS_ONLY_FALSE_RE = re.compile(r"docs_only\s*==\s*'false'")
DOCS_ONLY_TRUE_RE = re.compile(r"docs_only\s*==\s*'true'")
MATRIX_NAME_TEMPLATE = "${{ matrix.name }}"


class ConsistencyError(Exception):
    """A workflow file could not be read or parsed at all (fails closed)."""


class LiveFetchError(Exception):
    """The live branch-protection API (or `gh`) could not be reached/used."""


# ───────────────────────── workflow parsing ─────────────────────────

def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ConsistencyError("PyYAML is required (pip install pyyaml)") from exc

    if not os.path.isfile(path):
        raise ConsistencyError(f"workflow file not found at {path} (the gate fails closed)")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:
        raise ConsistencyError(f"workflow file at {path} is not parseable: {exc}") from exc
    if not isinstance(data, dict):
        raise ConsistencyError(f"workflow file at {path} does not parse to a mapping")
    return data


def _job_context_names(job: dict) -> tuple[list[str], bool]:
    """The literal, resolvable check-run name(s) a job produces, and whether
    they came from a matrix expansion (`${{ matrix.name }}`) as opposed to a
    single static name.

    The matrix/static distinction matters because a SKIPPED job's own
    check-run report satisfies branch protection only when GitHub can
    resolve a concrete name for it. PR #455 proved this empirically both
    ways: a skipped STATIC-named job (`wasm-execute-diff`) reports one
    correctly-named SKIPPED check run that DOES satisfy its required
    context with no stub needed; a skipped MATRIX job instead collapses to
    one check run named with the literal, unresolved string
    `"${{ matrix.name }}"`, which matches no required context at all — this
    is exactly why `docs-only-required-context-stubs` exists. Conflating
    the two would make this gate demand stub coverage for a static job that
    never needed it (or, worse, silently accept a matrix job that does).

    Returns ([], False) when the name cannot be statically resolved (any
    unresolved `${{ ... }}` expression other than the exact
    `${{ matrix.name }}` pattern this repo's matrices use) — such jobs are
    invisible to this gate rather than causing an error, since a required
    context can never legally be an unresolvable template string anyway.
    """

    name = job.get("name")
    if name is None:
        return [], False
    if not isinstance(name, str):
        return [], False

    if name.strip() == MATRIX_NAME_TEMPLATE:
        strategy = job.get("strategy") or {}
        matrix = strategy.get("matrix") or {}
        if isinstance(matrix, dict) and isinstance(matrix.get("name"), list):
            return [str(n) for n in matrix["name"] if isinstance(n, (str, int, float))], True
        include = matrix.get("include") if isinstance(matrix, dict) else None
        if isinstance(include, list):
            return [str(item["name"]) for item in include
                    if isinstance(item, dict) and "name" in item], True
        return [], True

    if "${{" in name:
        # A template this gate does not know how to resolve statically
        # (e.g. `${{ matrix.arch }}` in a job name). Never matches a real,
        # concrete required-context string, so it contributes nothing.
        return [], False

    return [name], False


def classify_workflow(doc: dict) -> dict:
    """Bucket every job in one workflow document by its docs-only exposure.

    Returns {"unconditional", "skippable_matrix", "skippable_static", "stub"}
    (each a set of context names). A workflow with no `docs_only` reference
    anywhere (e.g. identity-guard.yml) puts every job's names into
    `unconditional` — that IS "a workflow with no docs paths-ignore", just
    derived per job rather than assumed for the whole file.
    """

    jobs = doc.get("jobs")
    if not isinstance(jobs, dict):
        raise ConsistencyError("workflow document has no top-level `jobs` mapping")

    unconditional: set[str] = set()
    skippable_matrix: set[str] = set()
    skippable_static: set[str] = set()
    stub: set[str] = set()

    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        names, is_matrix = _job_context_names(job)
        if not names:
            # Fall back to the job id itself when no `name:` is given —
            # that is what GitHub reports for an unnamed job. Never a
            # matrix job in that case (a matrix job's rendered `name:` is
            # what carries `${{ matrix.name }}`).
            if job.get("name") is None:
                names = [job_id]
        if not names:
            continue

        cond = job.get("if")
        cond_str = cond if isinstance(cond, str) else ""

        if DOCS_ONLY_TRUE_RE.search(cond_str):
            stub.update(names)
        elif DOCS_ONLY_FALSE_RE.search(cond_str):
            if is_matrix:
                skippable_matrix.update(names)
            else:
                # A skipped STATIC-named job reports one correctly-named
                # SKIPPED check run, which satisfies branch protection on
                # its own (PR #455, measured) — no stub needed.
                skippable_static.update(names)
        else:
            unconditional.update(names)

    return {
        "unconditional": unconditional,
        "skippable_matrix": skippable_matrix,
        "skippable_static": skippable_static,
        "stub": stub,
    }


def compute_reportable(workflow_paths: list[str]) -> dict:
    """Merge the classification of every workflow file into one report."""

    unconditional: set[str] = set()
    skippable_matrix: set[str] = set()
    skippable_static: set[str] = set()
    stub: set[str] = set()

    for path in workflow_paths:
        doc = _load_yaml(path)
        classified = classify_workflow(doc)
        unconditional |= classified["unconditional"]
        skippable_matrix |= classified["skippable_matrix"]
        skippable_static |= classified["skippable_static"]
        stub |= classified["stub"]

    reportable_everywhere = unconditional | skippable_static | (skippable_matrix & stub)
    all_known = unconditional | skippable_matrix | skippable_static | stub

    return {
        "unconditional": unconditional,
        "skippable_matrix": skippable_matrix,
        "skippable_static": skippable_static,
        "stub": stub,
        "reportable_everywhere": reportable_everywhere,
        "all_known": all_known,
    }


# ───────────────────────── required-context sourcing ─────────────────────────

def fetch_required_via_http(repo: str, branch: str, token: str, timeout: float = 10.0) -> list[str]:
    url = f"https://api.github.com/repos/{repo}/branches/{branch}/protection"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "eshkol-required-context-consistency-gate",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LiveFetchError(f"GitHub API returned HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise LiveFetchError(f"could not reach GitHub API ({url}): {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise LiveFetchError(f"unexpected error querying {url}: {exc}") from exc

    try:
        contexts = payload["required_status_checks"]["contexts"]
    except (KeyError, TypeError) as exc:
        raise LiveFetchError(
            f"GitHub API response for {url} has no required_status_checks.contexts"
        ) from exc
    if not isinstance(contexts, list):
        raise LiveFetchError(f"required_status_checks.contexts at {url} is not a list")
    return [str(c) for c in contexts]


def fetch_required_via_gh_cli(repo: str, branch: str, timeout: float = 15.0) -> list[str]:
    gh = shutil.which("gh")
    if not gh:
        raise LiveFetchError("`gh` CLI not found on PATH")
    try:
        auth = subprocess.run([gh, "auth", "status"], capture_output=True, timeout=timeout)
    except Exception as exc:
        raise LiveFetchError(f"`gh auth status` could not run: {exc}") from exc
    if auth.returncode != 0:
        raise LiveFetchError("`gh` is installed but not authenticated")

    try:
        result = subprocess.run(
            [gh, "api", f"repos/{repo}/branches/{branch}/protection",
             "--jq", ".required_status_checks.contexts"],
            capture_output=True, timeout=timeout, text=True,
        )
    except Exception as exc:
        raise LiveFetchError(f"`gh api` could not run: {exc}") from exc
    if result.returncode != 0:
        raise LiveFetchError(f"`gh api` exited {result.returncode}: {result.stderr.strip()[:500]}")
    try:
        contexts = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LiveFetchError(f"`gh api` output was not JSON: {exc}") from exc
    if not isinstance(contexts, list):
        raise LiveFetchError("`gh api` output was not a JSON list")
    return [str(c) for c in contexts]


def load_fallback(path: str) -> list[str]:
    if not os.path.isfile(path):
        raise ConsistencyError(f"fallback contexts file not found at {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise ConsistencyError(f"fallback contexts file at {path} is not valid JSON: {exc}") from exc
    contexts = data.get("contexts") if isinstance(data, dict) else None
    if not isinstance(contexts, list) or not contexts:
        raise ConsistencyError(f"fallback contexts file at {path} has no non-empty `contexts` list")
    return [str(c) for c in contexts]


def determine_required_contexts(args: argparse.Namespace) -> tuple[str, list[str] | None, list[str]]:
    """Returns (mode, contexts_or_None, notes). mode in LIVE_HTTP / LIVE_GH /
    FALLBACK / NO_DATA. contexts is None only for NO_DATA."""

    notes: list[str] = []

    if args.offline:
        notes.append("--offline requested: skipping any network/gh attempt")
    else:
        token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            try:
                contexts = fetch_required_via_http(args.repo, args.branch, token)
                notes.append(f"LIVE via GitHub API (token present), repo={args.repo} branch={args.branch}")
                return "LIVE_HTTP", contexts, notes
            except LiveFetchError as exc:
                notes.append(f"LIVE via GitHub API failed, falling back: {exc}")
        else:
            notes.append("no GITHUB_TOKEN/GH_TOKEN in environment and no --token given")

        try:
            contexts = fetch_required_via_gh_cli(args.repo, args.branch)
            notes.append(f"LIVE via authenticated `gh` CLI, repo={args.repo} branch={args.branch}")
            return "LIVE_GH", contexts, notes
        except LiveFetchError as exc:
            notes.append(f"LIVE via `gh` CLI unavailable, falling back: {exc}")

    try:
        contexts = load_fallback(args.fallback_file)
        notes.append(
            f"FALLBACK: read committed snapshot {args.fallback_file} — NOT re-verified against "
            "live branch protection; stale if that snapshot was not updated by hand after the "
            "last `gh api .../protection` change"
        )
        return "FALLBACK", contexts, notes
    except ConsistencyError as exc:
        notes.append(f"FALLBACK unavailable: {exc}")

    notes.append("no LIVE source and no usable FALLBACK file — required-context set is UNKNOWN")
    return "NO_DATA", None, notes


# ───────────────────────── grading ─────────────────────────

def check(required_contexts: list[str], reportable: dict) -> dict:
    reportable_everywhere = reportable["reportable_everywhere"]
    all_known = reportable["all_known"]

    missing: list[dict] = []
    for context in required_contexts:
        if context in reportable_everywhere:
            continue
        if context in reportable["skippable_matrix"]:
            reason = (
                "produced by a MATRIX job that is skipped on docs-only PRs (its `if:` requires "
                "docs_only == 'false'); a skipped matrix job reports one check run under the "
                "literal unresolved '${{ matrix.name }}' string, not this name, and this name is "
                "NOT covered by the docs-only stub matrix"
            )
            reason_kind = "NOT_STUBBED"
        elif context in all_known:
            # In `stub` only, or some other combination that never yields
            # a real, always-on report — surfaced generically rather than
            # asserted incorrectly.
            reason = "only ever reported by the docs-only stub job, never by a real job on a code PR"
            reason_kind = "STUB_ONLY"
        else:
            reason = "no job in any checked workflow ever reports a context with this name"
            reason_kind = "UNKNOWN_TO_ANY_WORKFLOW"
        missing.append({"context": context, "reason": reason, "reason_kind": reason_kind})

    passed = not missing
    return {
        "passed": passed,
        "missing": missing,
        "required_count": len(required_contexts),
        "reportable_count": len(reportable_everywhere),
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

# A minimal, well-formed pair of fixture workflows: one unconditional job
# ("guard", standing in for identity-guard.yml), one skippable matrix job
# whose names are ALL covered by the stub job, and the stub job itself.
_GOOD_CI_YML = """
jobs:
  changes:
    name: changes
    if: "github.event_name != 'schedule'"
    runs-on: ubuntu-22.04
  docs-only-required-context-stubs:
    name: ${{ matrix.name }}
    if: "github.event_name != 'schedule' && needs.changes.outputs.docs_only == 'true'"
    strategy:
      matrix:
        name: [alpha, beta]
  unix-matrix:
    name: ${{ matrix.name }}
    if: "github.event_name != 'schedule' && needs.changes.outputs.docs_only == 'false'"
    strategy:
      matrix:
        name: [alpha, beta, gamma-lite]
"""
_GOOD_IDENTITY_YML = """
jobs:
  guard:
    runs-on: ubuntu-latest
"""
_GOOD_REQUIRED = ["guard", "alpha", "beta"]

# A static-named (non-matrix) job that is skipped on docs-only PRs, exactly
# like the real `wasm-execute-diff` — mirrors PR #455's measured finding
# that its own SKIPPED report already satisfies its required context, with
# NO stub matrix entry and no other coverage. Must PASS.
_STATIC_SKIPPABLE_CI_YML = """
jobs:
  changes:
    name: changes
    if: "github.event_name != 'schedule'"
    runs-on: ubuntu-22.04
  wasm-execute-diff:
    name: wasm-execute-diff
    if: "github.event_name != 'schedule' && needs.changes.outputs.docs_only == 'false'"
    runs-on: ubuntu-22.04
"""
_STATIC_SKIPPABLE_REQUIRED = ["guard", "wasm-execute-diff"]

# Red case (a): reproduces the real defect this gate exists to catch — a
# name (`beta`) is required and still skippable, but was dropped from the
# stub matrix (mirrors deleting a name from docs-only-required-context-
# stubs while branch protection still requires it).
_MISSING_STUB_CI_YML = """
jobs:
  changes:
    name: changes
    if: "github.event_name != 'schedule'"
    runs-on: ubuntu-22.04
  docs-only-required-context-stubs:
    name: ${{ matrix.name }}
    if: "github.event_name != 'schedule' && needs.changes.outputs.docs_only == 'true'"
    strategy:
      matrix:
        name: [alpha]
  unix-matrix:
    name: ${{ matrix.name }}
    if: "github.event_name != 'schedule' && needs.changes.outputs.docs_only == 'false'"
    strategy:
      matrix:
        name: [alpha, beta]
"""
_MISSING_STUB_REQUIRED = ["guard", "alpha", "beta"]

# Red case (b): a required context that no workflow, anywhere, could ever
# emit (mirrors an admin adding a required context that names no job).
_UNKNOWN_REQUIRED = ["guard", "alpha", "beta", "totally-imaginary-context"]

_MALFORMED_YAML = """
jobs:
  changes:
    name: changes
    if: "github.event_name != 'schedule'
    runs-on: ubuntu-22.04
"""


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _run_case(tmp_dir: str, ci_text: str, identity_text: str, required: list[str]) -> tuple[bool, str]:
    ci_path = os.path.join(tmp_dir, "ci.yml")
    identity_path = os.path.join(tmp_dir, "identity-guard.yml")
    _write(ci_path, ci_text)
    _write(identity_path, identity_text)
    try:
        reportable = compute_reportable([ci_path, identity_path])
    except ConsistencyError as exc:
        return False, f"parse error (expected for a malformed-YAML fixture): {exc}"
    result = check(required, reportable)
    if result["passed"]:
        detail = f"all {result['required_count']} required contexts are reportable"
    else:
        detail = "; ".join(f"{m['context']!r} ({m['reason_kind']})" for m in result["missing"])
    return result["passed"], detail


def self_test() -> bool:
    all_ok = True
    print("check_required_context_consistency.py self-test:")

    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-context-gate-") as tmp_dir:
        cases = [
            ("well_formed", _GOOD_CI_YML, _GOOD_IDENTITY_YML, _GOOD_REQUIRED, True),
            ("static_skippable_needs_no_stub", _STATIC_SKIPPABLE_CI_YML, _GOOD_IDENTITY_YML,
             _STATIC_SKIPPABLE_REQUIRED, True),
            ("red_a_stub_entry_removed", _MISSING_STUB_CI_YML, _GOOD_IDENTITY_YML,
             _MISSING_STUB_REQUIRED, False),
            ("red_b_unknown_required_context", _GOOD_CI_YML, _GOOD_IDENTITY_YML,
             _UNKNOWN_REQUIRED, False),
            ("malformed_yaml", _MALFORMED_YAML, _GOOD_IDENTITY_YML, _GOOD_REQUIRED, False),
        ]
        for name, ci_text, identity_text, required, expect_pass in cases:
            passed, detail = _run_case(tmp_dir, ci_text, identity_text, required)
            ok = passed == expect_pass
            all_ok = all_ok and ok
            verdict = "OK" if ok else "GATE IS BROKEN"
            print(f"  [{verdict}] {name}: expected passed={expect_pass}, got passed={passed}")
            print(f"           {detail}")

    # Mode-selection self-test: NO_DATA must be reachable and distinguishable
    # from FALLBACK, without ever touching the network.
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-context-gate-mode-") as tmp_dir:
        missing_fallback = os.path.join(tmp_dir, "does-not-exist.json")
        offline_args = argparse.Namespace(
            offline=True, token=None, repo=DEFAULT_REPO, branch=DEFAULT_BRANCH,
            fallback_file=missing_fallback,
        )
        mode, contexts, _notes = determine_required_contexts(offline_args)
        if mode == "NO_DATA" and contexts is None:
            print("  [OK] mode_selection_no_data: --offline + missing fallback file -> NO_DATA, contexts=None")
        else:
            print(f"  [GATE IS BROKEN] mode_selection_no_data: expected NO_DATA/None, got {mode}/{contexts}")
            all_ok = False

        present_fallback = os.path.join(tmp_dir, "fallback.json")
        with open(present_fallback, "w", encoding="utf-8") as handle:
            json.dump({"contexts": ["guard", "alpha"]}, handle)
        offline_args.fallback_file = present_fallback
        mode, contexts, _notes = determine_required_contexts(offline_args)
        if mode == "FALLBACK" and contexts == ["guard", "alpha"]:
            print("  [OK] mode_selection_fallback: --offline + present fallback file -> FALLBACK, contexts read back")
        else:
            print(f"  [GATE IS BROKEN] mode_selection_fallback: expected FALLBACK/['guard','alpha'], got {mode}/{contexts}")
            all_ok = False

    if all_ok:
        print("self-test: PASS — the gate fails on every broken fixture, passes the well-formed one, "
              "and NO_DATA is reachable and distinct from FALLBACK/PASS")
    else:
        print("self-test: FAIL — the gate did not behave as specified", file=sys.stderr)
    return all_ok


# ───────────────────────── CLI ─────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workflow", action="append", dest="workflows",
                         help="workflow YAML to include (repeatable); default is ci.yml + identity-guard.yml")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--token", default=None, help="GitHub token; else $GITHUB_TOKEN / $GH_TOKEN")
    parser.add_argument("--fallback-file", default=DEFAULT_FALLBACK_FILE)
    parser.add_argument("--offline", action="store_true",
                         help="never attempt network/gh; go straight to the fallback file (or NO_DATA)")
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    workflows = args.workflows or DEFAULT_WORKFLOWS

    mode, required_contexts, notes = determine_required_contexts(args)

    if mode == "NO_DATA":
        snippet = "NO_DATA: " + "; ".join(notes)
        if not args.no_trace:
            emit_trace(args.trace_dir, "NO_DATA", snippet)
        if args.format == "json":
            print(json.dumps({"status": "NO_DATA", "notes": notes}, indent=2))
        else:
            print(f"{PROBE_ID}: NO_DATA", file=sys.stderr)
            for note in notes:
                print(f"  - {note}", file=sys.stderr)
            print("NO_DATA is not a pass: nothing was verified.", file=sys.stderr)
        return 2

    try:
        reportable = compute_reportable(workflows)
    except ConsistencyError as exc:
        snippet = f"workflow(s) unusable: {exc}"
        if not args.no_trace:
            emit_trace(args.trace_dir, "FAIL", snippet)
        if args.format == "json":
            print(json.dumps({"status": "FAIL", "mode": mode, "error": str(exc)}, indent=2))
        else:
            print(f"{PROBE_ID}: FAIL — {exc}", file=sys.stderr)
        return 1

    result = check(required_contexts, reportable)
    status = "PASS" if result["passed"] else "FAIL"

    if result["passed"]:
        snippet = (
            f"[{mode}] {result['required_count']} required contexts, all reportable on every "
            f"PR shape ({result['reportable_count']} contexts reportable overall)"
        )
    else:
        snippet = f"[{mode}] {len(result['missing'])} required context(s) unreportable: " + "; ".join(
            f"{m['context']!r} ({m['reason_kind']})" for m in result["missing"][:5]
        )

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, "mode": mode, "notes": notes, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}  [mode={mode}]")
        for note in notes:
            print(f"  note: {note}")
        print(f"  required contexts : {result['required_count']}")
        print(f"  reportable always : {result['reportable_count']}")
        if result["missing"]:
            print("  UNREPORTABLE REQUIRED CONTEXTS:")
            for m in result["missing"]:
                print(f"    - {m['context']!r}: {m['reason']} [{m['reason_kind']}]")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
