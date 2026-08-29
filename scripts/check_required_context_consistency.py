#!/usr/bin/env python3
"""Release gate: every INTENDED branch-protection required status context
is REPORTABLE on every PR shape this repo's CI can produce.

Motivating incident: `.github/workflows/ci.yml`'s `docs-only-required-
context-stubs` job (added by PR #455) exists because a `unix-matrix` /
`windows-matrix` leg is `needs: changes` gated and its `if:` depends on
`docs_only`. On a docs-only PR those jobs never instantiate at all, so
their per-leg contexts (`linux-x64-xla`, `macos-arm64-lite`, ...) are
ABSENT from the head SHA's check-run set entirely — not reported under a
wrong name, simply never created. Branch protection cannot resolve a
required context that no check run ever reports for, so it blocks forever
with no timeout. PR #455's stub job works around this by running its OWN
matrix, over the literal names that needed it at the time, whenever
`docs_only == 'true'`, so a check run under each real name still exists.

Branch protection later grew 5 more matrix-derived required contexts
(windows-arm64-lite, macos-arm64-xla, macos-x64-xla, macos-arm64-lite,
macos-x64-lite) and nobody extended the stub matrix to match — the same
permanent-block failure PR #455 fixed, reopened for those 5, silently:
`docs-only-required-context-stubs` and branch protection's required list
are two lists a human keeps in sync by hand, with nothing that failed a
build when they drift. With `enforce_admins: true` set, a drift like this
makes a docs-only PR permanently unmergeable by ANYONE, not just
non-admins — confirmed directly: a real docs-only PR was MERGEABLE/BLOCKED
with those 5 required, and MERGEABLE/CLEAN the moment they were removed
from branch protection, no other change, no new CI run. Branch protection
has (as of this gate's introduction) been temporarily narrowed to exclude
those 5 while this fix ships; this gate's job is to certify the FULL
intended set is reportable so they can be safely restored, not merely to
agree with whatever is required at this exact moment.

A context that IS emitted, but with a SKIPPED conclusion, is a different
and unproblematic case: a job that is skipped via a STATIC (non-matrix)
`if:` still reports one check run under its own real name with conclusion
`skipped`, and branch protection treats `skipped` as SATISFYING a required
context (confirmed directly: `wasm-execute-diff`, a static-named job
gated the same way as the matrix jobs above, stayed required throughout
and never blocked anything). The two look superficially identical in the
YAML ("skipped when docs_only") but have opposite consequences, which is
exactly why this gate must distinguish them rather than treating "gated on
docs_only" as one bucket:

    a required context is satisfiable iff it is
      (a) emitted unconditionally, or
      (b) emitted by a job skipped via a STATIC `if:` (reports `skipped`,
          which satisfies branch protection), or
      (c) present in the docs-only stub matrix (covers a MATRIX job's
          per-leg names, which are otherwise ABSENT — never reported under
          any name at all — when the job is skipped).

A required context produced only by a skipped MATRIX job, uncovered by (c),
is case (a)-and-(b)-and-(c) failing simultaneously: ABSENT on a docs-only
PR. A required context that matches no job's name in ANY checked workflow
is unreportable on every PR shape — a name nothing in CI could ever emit
at all, which is at least as bad.

This gate computes the above by parsing the actual workflow YAML (never by
regexing job text):

    required_contexts SUBSET-OF  unconditional_contexts
                                  UNION static_skipped_contexts
                                  UNION (matrix_skipped_contexts INTERSECT stub_matrix_contexts)

Which required-context set is graded, and why LIVE alone is not enough
    A gate that only ever graded the CURRENTLY live required-context set
    would, right now, read branch protection's temporarily-narrowed 11
    contexts as complete and say nothing about the 5 that are missing
    specifically because this gate's own fix has not landed yet — the
    exact "checked nothing, reported success" shape this whole file exists
    to prevent, just one level up. So grading is always done against a
    committed TARGET file (`.icc/required-status-contexts.json`), which
    records the INTENDED full required-context set independent of branch
    protection's state at any given moment. LIVE branch protection (via a
    GITHUB_TOKEN/GH_TOKEN, or an authenticated `gh` CLI) is fetched
    best-effort and folded into the graded set too (the union of target
    and live — so an ungoverned addition to live that nobody added to the
    target file is *also* checked, not silently skipped), and the
difference between the two is always reported, never silently
resolved one way. Update the target file by hand whenever the intended
policy changes; this script does not write it.

The target may intentionally carry a required-context candidate before the
administrator adds it to live branch protection. `linux-x64-debug` is such a
candidate: it is graded now so its docs-only stub coverage is proven before
the one-click branch-protection change.

Modes
    TARGET_ONLY        `--offline`, or no token/`gh` available: grades the
                        committed target file alone.
    TARGET_PLUS_LIVE    both the target file and a live fetch succeeded:
                        grades their union; reports contexts that differ
                        between the two (current-vs-intended visibility).
    LIVE_ONLY_NO_TARGET the target file is missing/unreadable but a live
                        fetch succeeded: grades live alone, with a loud
                        warning that the committed intended-policy record
                        is itself missing.
    NO_DATA             neither the target file nor a live fetch produced
                        anything to grade. This is NOT a PASS: nothing has
                        been verified. NO_DATA exits 2, distinct from PASS
                        (0) and FAIL (1), specifically so a caller cannot
                        mistake "we never checked" for "we checked and
                        it's fine."

Grading
    PASS      a graded context set was obtained (any mode but NO_DATA),
              and every context in it is reportable per the SUBSET-OF rule
              above.
    FAIL      a graded context set was obtained and at least one context
              in it is not reportable — OR a workflow file is
              missing/unparseable (fails closed, same as this repo's other
              gates).
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
DEFAULT_TARGET_FILE = os.path.join(REPO_ROOT, ".icc", "required-status-contexts.json")
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


def load_target(path: str) -> list[str]:
    """The committed INTENDED required-context set — the source of truth
    this gate grades against, independent of whatever branch protection
    currently requires (which may be temporarily narrower mid-rollout, or
    could in principle drift wider without anyone updating this file)."""

    if not os.path.isfile(path):
        raise ConsistencyError(f"target contexts file not found at {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise ConsistencyError(f"target contexts file at {path} is not valid JSON: {exc}") from exc
    contexts = data.get("contexts") if isinstance(data, dict) else None
    if not isinstance(contexts, list) or not contexts:
        raise ConsistencyError(f"target contexts file at {path} has no non-empty `contexts` list")
    return [str(c) for c in contexts]


def fetch_live(args: argparse.Namespace) -> tuple[list[str] | None, list[str]]:
    """Best-effort LIVE branch-protection fetch. Returns (contexts_or_None,
    notes). Never raises — a failure is just a note, since LIVE is always
    optional (the target file alone is sufficient to grade)."""

    notes: list[str] = []
    if args.offline:
        notes.append("--offline requested: skipping any network/gh attempt")
        return None, notes

    token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        try:
            contexts = fetch_required_via_http(args.repo, args.branch, token)
            notes.append(f"LIVE via GitHub API (token present), repo={args.repo} branch={args.branch}")
            return contexts, notes
        except LiveFetchError as exc:
            notes.append(f"LIVE via GitHub API failed: {exc}")
    else:
        notes.append("no GITHUB_TOKEN/GH_TOKEN in environment and no --token given")

    try:
        contexts = fetch_required_via_gh_cli(args.repo, args.branch)
        notes.append(f"LIVE via authenticated `gh` CLI, repo={args.repo} branch={args.branch}")
        return contexts, notes
    except LiveFetchError as exc:
        notes.append(f"LIVE via `gh` CLI unavailable: {exc}")

    return None, notes


def determine_grading_contexts(args: argparse.Namespace) -> dict:
    """Returns {"mode", "target_contexts", "live_contexts",
    "effective_contexts", "notes"}. `effective_contexts` (what actually
    gets graded) is None only for NO_DATA.

    Grading is ALWAYS anchored on the committed target file when it is
    available — never on live alone — specifically so a live set that is
    temporarily (or permanently, by mistake) narrower than intended cannot
    make this gate agree that nothing is missing. When live also succeeds,
    the graded set is target UNION live, so an ungoverned addition on the
    live side gets checked too rather than silently ignored.
    """

    notes: list[str] = []

    target_contexts: list[str] | None = None
    try:
        target_contexts = load_target(args.target_file)
    except ConsistencyError as exc:
        notes.append(f"target file unavailable: {exc}")

    live_contexts, live_notes = fetch_live(args)
    notes.extend(live_notes)

    if target_contexts is not None and live_contexts is not None:
        effective = sorted(set(target_contexts) | set(live_contexts))
        live_only = sorted(set(live_contexts) - set(target_contexts))
        target_only = sorted(set(target_contexts) - set(live_contexts))
        if live_only:
            notes.append(
                f"LIVE requires {len(live_only)} context(s) not in the target file "
                f"(update {args.target_file}): {live_only}"
            )
        if target_only:
            notes.append(
                f"target file intends {len(target_only)} context(s) live does not currently "
                f"require (expected mid-rollout; restore in branch protection once this gate "
                f"is green): {target_only}"
            )
        if not live_only and not target_only:
            notes.append("LIVE and the target file agree exactly")
        return {"mode": "TARGET_PLUS_LIVE", "target_contexts": target_contexts,
                "live_contexts": live_contexts, "effective_contexts": effective, "notes": notes}

    if target_contexts is not None:
        return {"mode": "TARGET_ONLY", "target_contexts": target_contexts,
                "live_contexts": None, "effective_contexts": target_contexts, "notes": notes}

    if live_contexts is not None:
        notes.append(
            "grading LIVE ONLY: the committed target/intended-policy file is missing or "
            "unreadable, so only what branch protection currently requires is being checked, "
            "not the full intended policy"
        )
        return {"mode": "LIVE_ONLY_NO_TARGET", "target_contexts": None,
                "live_contexts": live_contexts, "effective_contexts": live_contexts, "notes": notes}

    notes.append("no target file and no LIVE source — required-context set is UNKNOWN")
    return {"mode": "NO_DATA", "target_contexts": None, "live_contexts": None,
            "effective_contexts": None, "notes": notes}


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
                "ABSENT entirely on a docs-only PR: produced by a MATRIX job whose `if:` requires "
                "docs_only == 'false', and a skipped matrix job never instantiates its per-leg "
                "names (they collapse to one check run under the literal unresolved "
                "'${{ matrix.name }}' string) — this name is NOT covered by the docs-only stub "
                "matrix, so no check run bearing it is ever created on that PR shape"
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
    # from TARGET_ONLY, without ever touching the network.
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-context-gate-mode-") as tmp_dir:
        missing_target = os.path.join(tmp_dir, "does-not-exist.json")
        offline_args = argparse.Namespace(
            offline=True, token=None, repo=DEFAULT_REPO, branch=DEFAULT_BRANCH,
            target_file=missing_target,
        )
        grading = determine_grading_contexts(offline_args)
        if grading["mode"] == "NO_DATA" and grading["effective_contexts"] is None:
            print("  [OK] mode_selection_no_data: --offline + missing target file -> NO_DATA, contexts=None")
        else:
            print(f"  [GATE IS BROKEN] mode_selection_no_data: expected NO_DATA/None, got "
                  f"{grading['mode']}/{grading['effective_contexts']}")
            all_ok = False

        present_target = os.path.join(tmp_dir, "target.json")
        with open(present_target, "w", encoding="utf-8") as handle:
            json.dump({"contexts": ["guard", "alpha"]}, handle)
        offline_args.target_file = present_target
        grading = determine_grading_contexts(offline_args)
        if grading["mode"] == "TARGET_ONLY" and grading["effective_contexts"] == ["guard", "alpha"]:
            print("  [OK] mode_selection_target_only: --offline + present target file -> TARGET_ONLY, contexts read back")
        else:
            print(f"  [GATE IS BROKEN] mode_selection_target_only: expected TARGET_ONLY/['guard','alpha'], got "
                  f"{grading['mode']}/{grading['effective_contexts']}")
            all_ok = False

        # THE decisive regression case for this incident: branch protection
        # was measured live-narrower than intended (11 required, temporarily
        # missing 5 matrix-derived contexts; the debug candidate is also
        # target-only) specifically BECAUSE this gate's
        # own fix had not shipped yet. A gate that graded live alone would
        # have read that narrowed set as complete and said nothing. Simulate
        # "live" by union-ing the target file with a narrower set by hand
        # (rather than a real network call) and confirm the STILL-required,
        # still-unstubbed name in the target file is caught even though a
        # live-only view would have missed it entirely.
        target_says_beta_required = os.path.join(tmp_dir, "target_with_beta.json")
        with open(target_says_beta_required, "w", encoding="utf-8") as handle:
            json.dump({"contexts": _MISSING_STUB_REQUIRED}, handle)  # ["guard", "alpha", "beta"]
        live_narrowed_without_beta = ["guard", "alpha"]  # what a rolled-back live set would show
        effective = sorted(set(_MISSING_STUB_REQUIRED) | set(live_narrowed_without_beta))
        try:
            ci_path = os.path.join(tmp_dir, "narrow_ci.yml")
            identity_path = os.path.join(tmp_dir, "narrow_identity.yml")
            _write(ci_path, _MISSING_STUB_CI_YML)
            _write(identity_path, _GOOD_IDENTITY_YML)
            reportable = compute_reportable([ci_path, identity_path])
            result_live_only = check(live_narrowed_without_beta, reportable)
            result_target_union = check(effective, reportable)
        except ConsistencyError as exc:
            print(f"  [GATE IS BROKEN] narrowed_live_would_have_hidden_gap: fixture setup failed: {exc}")
            all_ok = False
        else:
            if result_live_only["passed"] and not result_target_union["passed"]:
                print("  [OK] narrowed_live_would_have_hidden_gap: grading live's narrowed set alone would "
                      "wrongly PASS ('beta' never checked); grading target UNION live correctly FAILS on it")
            else:
                print(f"  [GATE IS BROKEN] narrowed_live_would_have_hidden_gap: expected live-only PASS "
                      f"(false confidence) and union FAIL, got live-only passed="
                      f"{result_live_only['passed']}, union passed={result_target_union['passed']}")
                all_ok = False

    if all_ok:
        print("self-test: PASS — the gate fails on every broken fixture, passes the well-formed ones, "
              "NO_DATA is reachable and distinct from TARGET_ONLY/PASS, and grading target UNION live "
              "catches a gap a narrowed live-only view would hide")
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
    parser.add_argument("--target-file", default=DEFAULT_TARGET_FILE,
                         help="committed INTENDED required-context list; always graded when present")
    parser.add_argument("--offline", action="store_true",
                         help="never attempt network/gh; grade the target file alone (or NO_DATA)")
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    workflows = args.workflows or DEFAULT_WORKFLOWS

    grading = determine_grading_contexts(args)
    mode, required_contexts, notes = grading["mode"], grading["effective_contexts"], grading["notes"]

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
            f"[{mode}] {result['required_count']} graded contexts, all reportable on every "
            f"PR shape ({result['reportable_count']} contexts reportable overall)"
        )
    else:
        snippet = f"[{mode}] {len(result['missing'])} graded context(s) unreportable: " + "; ".join(
            f"{m['context']!r} ({m['reason_kind']})" for m in result["missing"][:5]
        )

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **grading, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}  [mode={mode}]")
        for note in notes:
            print(f"  note: {note}")
        print(f"  graded contexts   : {result['required_count']}")
        print(f"  reportable always : {result['reportable_count']}")
        if result["missing"]:
            print("  UNREPORTABLE REQUIRED CONTEXTS:")
            for m in result["missing"]:
                print(f"    - {m['context']!r}: {m['reason']} [{m['reason_kind']}]")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
