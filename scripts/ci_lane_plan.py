#!/usr/bin/env python3
"""Decide WHICH CI lanes a change actually needs, from the lanes themselves.

`scripts/ci_change_class.py` answers "how much could this change possibly
affect" (docs / non-build / tests-only / full). For the two inert classes
the answer is already actionable -- skip the heavy matrix, let
`docs-only-required-context-stubs` report the required contexts. For
`tests-only` it was, until this script existed, purely informational: a
one-line edit to a single `.esk` test still ran all 25 cross-platform
lanes, six of them on hosted macOS runners, which are this repo's
throughput bottleneck.

This script turns that class into a lane decision. It never contains a
hand-written lane list: every lane name, every lane's runner, and every
lane's capability flags are re-derived from `.github/workflows/ci.yml`
itself on each run, so a lane added, renamed or removed there is picked up
with nothing to keep in sync. (The same discipline
`scripts/check_required_context_consistency.py` and
`scripts/ci_change_class.py` already apply: derive from the real artifact,
never maintain a second copy of it.)

Lane selection for `tests-only`
    Every rule below is a statement about what the changed files can
    reach, and each is derived from lane metadata rather than typed out:

      * DIRECT REFERENCE -- a changed file whose exact path appears in a
        lane's `test_command` (or in an aux job's `run:` block) activates
        that lane. This is how an edit to `scripts/run_gpu_tests.sh`
        reaches exactly the lanes that run it.
      * GPU tests -- a change under a `tests/` directory named after a
        capability flag (`tests/gpu`) activates the lanes declaring that
        capability (`gpu_enabled: 'ON'`), and nothing else.
      * XLA tests -- likewise for `tests/xla` and `xla_enabled: 'ON'`.
      * VM-parity tests -- `tests/vm_parity` is executed by the
        `pillars-fast` job (`scripts/run_vm_parity.sh`), which is not
        matrix-gated and runs on every non-inert PR regardless. One
        portable lane is added alongside it so the corpus is also
        exercised through a real compiler build.
      * ANYTHING ELSE under `tests/` -- the cheapest portable ("lite")
        lane of each OS family, in the order `ci.yml` declares them:
        the first lane per family with `xla_enabled: 'OFF'`,
        `gpu_enabled: 'OFF'`, no sanitizer flags and the default build
        directory. Today that resolves to linux-x64-lite,
        macos-arm64-lite and windows-arm64-lite -- one per OS, each
        running the full `scripts/run_all_tests.sh` suite.

    Every lane NOT selected still instantiates as a matrix leg and still
    reports a check run under its own name (its steps are gated on the
    `LANE_ACTIVE` job env this script's output feeds). That is
    load-bearing: a lane whose job never instantiates at all reports
    NOTHING, and a required status context with no check run blocks a PR
    forever -- the failure mode
    `scripts/check_required_context_consistency.py` exists to prevent.
    Skipping steps rather than skipping jobs keeps the entire required-
    context set intact by construction, for every class.

Output contract
    A JSON object on stdout:

      {"impact": "<class>",
       "active": "ALL" | "|lane|lane|...|" | "",
       "active_lanes": [...], "skipped_lanes": [...],
       "aux_jobs": [...], "skipped_aux_jobs": [...],
       "rules": ["<why each lane was selected>", ...],
       "summary": "<one line>",
       "derivation": {...}}

    `active` is the exact string `.github/workflows/ci.yml` consumes.
    "ALL" means "every lane active" and is the fail-safe value -- the
    workflow treats it, and any value it cannot make sense of, as
    "run everything".

Usage
    git diff --name-only origin/master...HEAD \\
        | python3 scripts/ci_lane_plan.py --impact tests-only
    python3 scripts/ci_lane_plan.py --impact full
    python3 scripts/ci_lane_plan.py --impact tests-only tests/gpu/x.esk
    python3 scripts/ci_lane_plan.py --self-test

Exit status: 0 on a successful plan or a passing `--self-test`; 1 on a
usage error, an unparseable workflow, or a failing `--self-test`.

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

MATRIX_NAME_TEMPLATE = "${{ matrix.name }}"
DOCS_ONLY_FALSE_RE = re.compile(r"docs_only\s*==\s*'false'")
DOCS_ONLY_TRUE_RE = re.compile(r"docs_only\s*==\s*'true'")

# Classes for which the heavy matrix is skipped at the JOB level (the
# `docs_only == 'true'` path), so no lane is active and the stub job
# reports the required contexts.
INERT_CLASSES = ("docs", "non-build", "equivalent")
# Classes that run every lane.
FULL_CLASSES = ("full",)
CLASS_TESTS_ONLY = "tests-only"
KNOWN_CLASSES = INERT_CLASSES + FULL_CLASSES + (CLASS_TESTS_ONLY,)

ALL_LANES = "ALL"

# `tests/<dir>` -> the lane capability flag that directory's tests need.
# Both halves are real: the directory is checked against the repository's
# own tests tree, and the flag against the lane definitions in ci.yml.
CAPABILITY_DIRS = {
    "gpu": "gpu_enabled",
    "xla": "xla_enabled",
}
# `tests/<dir>` -> a non-matrix job that already executes that corpus.
# `pillars-fast` runs scripts/run_vm_parity.sh unconditionally on every
# non-inert PR, so a vm_parity change is covered there; one portable lane
# is added so it is also exercised against a real compiler build.
CORPUS_COVERED_ELSEWHERE = {
    "vm_parity": "pillars-fast",
}


class LanePlanError(Exception):
    """The workflow could not be read well enough to plan anything."""


# ─────────────────────────── ci.yml (line) parsing ───────────────────────────
#
# Deliberately dependency-free. This script runs in the `changes` job,
# which is the single point of failure for the whole workflow and is kept
# to a bare checkout plus stock `python3` -- exactly like
# `scripts/ci_change_class.py`, which parses the same workflow files the
# same way and for the same reason. The structures parsed here are the
# three the file actually uses, and `--self-test` asserts the parse
# against the real, committed `ci.yml` rather than fixtures alone.

def _split_jobs(text: str) -> dict[str, list[str]]:
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip() == "jobs:")
    except StopIteration as exc:
        raise LanePlanError("workflow has no top-level `jobs:` block") from exc

    jobs: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[start + 1:]:
        if line and not line.startswith(" ") and not line.startswith("#"):
            break  # a new top-level key ends the jobs block
        match = re.match(r"^  ([A-Za-z0-9_.\-]+):\s*$", line)
        if match:
            current = match.group(1)
            jobs[current] = []
            continue
        if current is not None:
            jobs[current].append(line)
    if not jobs:
        raise LanePlanError("workflow's `jobs:` block contained no jobs")
    return jobs


def _job_scalar(block: list[str], key: str) -> str | None:
    """A job-level scalar, inline or as a folded/literal block.

    `if:` conditions in this workflow are long enough that some are written
    as `if: >-` over several lines; reading only the first line would see
    the fold indicator instead of the condition and silently mis-bucket the
    job, so the continuation is joined here.
    """

    pattern = re.compile(rf"^    {re.escape(key)}:\s*(.*)$")
    for index, line in enumerate(block):
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1).strip()
        if value not in (">", ">-", ">+", "|", "|-", "|+"):
            return _unquote(value)
        parts: list[str] = []
        for continuation in block[index + 1:]:
            if not continuation.strip():
                continue
            if not continuation.startswith("      "):
                break
            parts.append(continuation.strip())
        return " ".join(parts)
    return None


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _matrix_block(block: list[str]) -> list[str]:
    """The lines under `strategy: matrix:` for one job, or []."""

    out: list[str] = []
    inside = False
    for line in block:
        if re.match(r"^      matrix:\s*$", line):
            inside = True
            continue
        if inside:
            if line.strip() and not line.startswith("        "):
                break
            out.append(line)
    return out


def _parse_include_legs(matrix_lines: list[str]) -> list[dict[str, str]]:
    """`include:` entries as flat {key: scalar} dicts.

    Handles the two scalar shapes this workflow uses: an inline value and
    a folded block scalar (`>-`), which `linux-x64-debug`'s multi-command
    `test_command` needs.
    """

    legs: list[dict[str, str]] = []
    inside = False
    current: dict[str, str] | None = None
    pending_key: str | None = None
    pending: list[str] = []

    def flush_pending() -> None:
        nonlocal pending_key, pending
        if current is not None and pending_key is not None:
            current[pending_key] = " ".join(part.strip() for part in pending).strip()
        pending_key = None
        pending = []

    for line in matrix_lines:
        if re.match(r"^        include:\s*$", line):
            inside = True
            continue
        if not inside:
            continue
        if line.strip() and not line.startswith("          "):
            break

        entry = re.match(r"^          - ([A-Za-z0-9_]+):\s*(.*)$", line)
        if entry:
            flush_pending()
            if current is not None:
                legs.append(current)
            current = {}
            key, value = entry.group(1), entry.group(2)
            if value.strip() in (">", ">-", "|", "|-"):
                pending_key = key
            else:
                current[key] = _unquote(value)
            continue

        field = re.match(r"^            ([A-Za-z0-9_]+):\s*(.*)$", line)
        if field and current is not None:
            flush_pending()
            key, value = field.group(1), field.group(2)
            if value.strip() in (">", ">-", "|", "|-"):
                pending_key = key
            else:
                current[key] = _unquote(value)
            continue

        if pending_key is not None and line.strip() and not line.lstrip().startswith("#"):
            pending.append(line)

    flush_pending()
    if current is not None:
        legs.append(current)
    return [leg for leg in legs if "name" in leg]


def _parse_name_list(matrix_lines: list[str]) -> list[str]:
    """A plain `name:` sequence (the stub job's matrix)."""

    names: list[str] = []
    inside = False
    for line in matrix_lines:
        if re.match(r"^        name:\s*$", line):
            inside = True
            continue
        if not inside:
            continue
        if line.strip() and not line.startswith("          "):
            break
        entry = re.match(r"^          - ([A-Za-z0-9_.\-]+)\s*$", line)
        if entry:
            names.append(entry.group(1))
    return names


def load_workflow(path: Path | None = None) -> dict:
    """Everything this script needs to know about ci.yml, derived."""

    workflow_path = path or CI_WORKFLOW
    try:
        text = workflow_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LanePlanError(f"cannot read {workflow_path}: {exc}") from exc

    jobs = _split_jobs(text)

    lanes: list[dict[str, str]] = []
    lane_job_ids: list[str] = []
    stub_names: list[str] = []
    aux_jobs: list[dict[str, str]] = []
    job_text: dict[str, str] = {}

    for job_id, block in jobs.items():
        job_text[job_id] = "\n".join(block)
        name = _job_scalar(block, "name")
        cond = _job_scalar(block, "if") or ""
        runs_on = _job_scalar(block, "runs-on") or ""
        matrix_lines = _matrix_block(block)

        if name is not None and name.strip() == MATRIX_NAME_TEMPLATE:
            if DOCS_ONLY_TRUE_RE.search(cond):
                stub_names.extend(_parse_name_list(matrix_lines))
            elif DOCS_ONLY_FALSE_RE.search(cond):
                legs = _parse_include_legs(matrix_lines)
                if legs:
                    lane_job_ids.append(job_id)
                for leg in legs:
                    leg = dict(leg)
                    leg["_job"] = job_id
                    lanes.append(leg)
            continue

        # A statically-named heavy job on a hosted macOS runner. These are
        # advisory (never required contexts), and hosted macOS runners are
        # the scarce resource, so they are planned alongside the lanes.
        if DOCS_ONLY_FALSE_RE.search(cond) and runs_on.startswith("macos"):
            aux_jobs.append({"job": job_id, "name": name or job_id,
                             "runner": runs_on})

    if not lanes:
        raise LanePlanError(
            "no matrix lanes found in the workflow -- refusing to plan a "
            "reduction from an empty lane set")

    return {
        "path": str(workflow_path),
        "lanes": lanes,
        "lane_job_ids": lane_job_ids,
        "stub_names": stub_names,
        "aux_jobs": aux_jobs,
        "job_text": job_text,
    }


# ───────────────────────────────── planning ─────────────────────────────────

def _os_family(runner: str) -> str:
    if runner.startswith("ubuntu"):
        return "linux"
    if runner.startswith("macos"):
        return "macos"
    if runner.startswith("windows"):
        return "windows"
    return "other"


def _is_lite(leg: dict[str, str]) -> bool:
    return (
        leg.get("xla_enabled") == "OFF"
        and leg.get("gpu_enabled") == "OFF"
        and not leg.get("sanitizer_flags")
        and leg.get("build_dir") == "build"
    )


def lite_lane_per_family(lanes: list[dict[str, str]]) -> dict[str, str]:
    """First portable lane of each OS family, in declaration order."""

    chosen: dict[str, str] = {}
    for leg in lanes:
        if not _is_lite(leg):
            continue
        family = _os_family(leg.get("runner", ""))
        chosen.setdefault(family, leg["name"])
    return chosen


def _result(impact: str, active_lanes: list[str], skipped_lanes: list[str],
            aux_active: list[str], aux_skipped: list[str],
            rules: list[str], workflow: dict,
            active: str | None = None) -> dict:
    """Assemble the output document, including the `active` string ci.yml reads."""

    if active is None:
        tokens = active_lanes + aux_active
        active = "|" + "|".join(tokens) + "|" if tokens else ""
    if active_lanes and not skipped_lanes and active != "":
        active = ALL_LANES
    summary = (
        f"impact {impact}: {len(active_lanes)}/"
        f"{len(active_lanes) + len(skipped_lanes)} matrix lanes active"
        + (f", aux {', '.join(aux_active)}" if aux_active else ", no aux job")
    )
    return {
        "impact": impact,
        "active": active,
        "active_lanes": active_lanes,
        "skipped_lanes": skipped_lanes,
        "aux_jobs": aux_active,
        "skipped_aux_jobs": aux_skipped,
        "rules": rules,
        "summary": summary,
        "derivation": {
            "workflow": workflow["path"],
            "lane_jobs": workflow["lane_job_ids"],
            "lanes_declared": [leg["name"] for leg in workflow["lanes"]],
            "stub_matrix": workflow["stub_names"],
            "aux_jobs_declared": [aux["job"] for aux in workflow["aux_jobs"]],
        },
    }


def plan(impact: str, changed_files: list[str], workflow: dict) -> dict:
    lanes = workflow["lanes"]
    lane_names = [leg["name"] for leg in lanes]
    aux_names = [aux["job"] for aux in workflow["aux_jobs"]]

    if impact in INERT_CLASSES:
        return _result(
            impact, [], lane_names, [], aux_names,
            ["no lane runs: the matrix jobs are skipped at the job level for "
             f"impact '{impact}'; docs-only-required-context-stubs reports "
             "every required context"],
            workflow, active="")

    if impact not in KNOWN_CLASSES or impact in FULL_CLASSES:
        reason = (f"impact '{impact}' runs every lane"
                  if impact in FULL_CLASSES
                  else f"unrecognised impact '{impact}' -- running every lane "
                       "(the conservative answer)")
        return _result(impact, lane_names, [], aux_names, [], [reason],
                       workflow, active=ALL_LANES)

    # ---- tests-only ---------------------------------------------------
    selected: set[str] = set()
    selected_aux: set[str] = set()
    rules: list[str] = []

    lite = lite_lane_per_family(lanes)
    by_name = {leg["name"]: leg for leg in lanes}

    generic_test_change = False
    for path in changed_files:
        parts = path.split("/")
        matched_rule = False

        # DIRECT REFERENCE -- a lane or aux job that literally names this file.
        for leg in lanes:
            command = " ".join(
                value for key, value in leg.items()
                if key in ("test_command", "test_mode"))
            if path and path in command:
                selected.add(leg["name"])
                rules.append(f"{path}: named by lane {leg['name']}'s test command")
                matched_rule = True
        for aux in workflow["aux_jobs"]:
            if path and path in workflow["job_text"].get(aux["job"], ""):
                selected_aux.add(aux["job"])
                rules.append(f"{path}: named by the {aux['job']} job")
                matched_rule = True

        if len(parts) >= 2 and parts[0] == "tests":
            group = parts[1]
            if group in CAPABILITY_DIRS:
                flag = CAPABILITY_DIRS[group]
                capable = [leg["name"] for leg in lanes if leg.get(flag) == "ON"]
                if capable:
                    selected.update(capable)
                    rules.append(
                        f"{path}: tests/{group} needs {flag}=ON -- "
                        f"{', '.join(sorted(capable))}")
                    matched_rule = True
            elif group in CORPUS_COVERED_ELSEWHERE:
                job = CORPUS_COVERED_ELSEWHERE[group]
                portable = lite.get("linux") or (lane_names[0] if lane_names else None)
                if portable:
                    selected.add(portable)
                rules.append(
                    f"{path}: tests/{group} is executed by the {job} job "
                    f"(not matrix-gated); {portable} added for a real "
                    "compiler build")
                matched_rule = True
            else:
                generic_test_change = True
                matched_rule = True

        if not matched_rule:
            # A tests-only file this script cannot place (a test script CI
            # runs, most often). Treat it like a generic test change rather
            # than assuming it is covered.
            generic_test_change = True

    if generic_test_change:
        trio = [name for _, name in sorted(lite.items()) if name in by_name]
        selected.update(trio)
        rules.append(
            "generic tests/ change: the first portable lane of each OS family "
            f"-- {', '.join(sorted(trio))}")

    if not selected:
        selected.update(name for name in lite.values())
        rules.append("no rule matched: falling back to the portable lane of "
                     "each OS family")

    active = [name for name in lane_names if name in selected]
    skipped = [name for name in lane_names if name not in selected]
    aux_active = [name for name in aux_names if name in selected_aux]
    aux_skipped = [name for name in aux_names if name not in selected_aux]
    return _result(impact, active, skipped, aux_active, aux_skipped, rules,
                   workflow)


# ───────────────────────────────── self-test ─────────────────────────────────

def self_test() -> bool:
    print("ci_lane_plan.py self-test:")
    ok = True

    try:
        workflow = load_workflow()
    except LanePlanError as exc:
        print(f"  FAIL parse_real_ci_yml: {exc}")
        print("  RESULT: FAIL")
        return False

    lane_names = [leg["name"] for leg in workflow["lanes"]]
    print(f"  INFO parsed {len(lane_names)} lanes from {workflow['path']}: "
          f"{', '.join(lane_names)}")

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        if condition:
            print(f"  PASS {name}")
        else:
            ok = False
            print(f"  FAIL {name}{': ' + detail if detail else ''}")

    # The parse must find real lanes, real stub names, and real aux jobs.
    check("lanes_parsed", len(lane_names) >= 10,
          f"only {len(lane_names)} lanes parsed")
    check("lane_flags_parsed",
          any(leg.get("gpu_enabled") == "ON" for leg in workflow["lanes"])
          and any(leg.get("xla_enabled") == "ON" for leg in workflow["lanes"]),
          "no lane declared gpu_enabled/xla_enabled -- the flag parse is broken")
    check("stub_matrix_parsed", len(workflow["stub_names"]) >= 10,
          f"stub matrix parsed as {workflow['stub_names']}")
    check("aux_jobs_parsed", len(workflow["aux_jobs"]) >= 1,
          "no hosted-macOS aux job found")

    # Every stub name must be a real lane name: the stub job exists only to
    # stand in for lanes, so a stub entry naming nothing is drift.
    unknown_stubs = [n for n in workflow["stub_names"] if n not in lane_names]
    check("stub_names_are_real_lanes", not unknown_stubs,
          f"stub entries naming no lane: {unknown_stubs}")

    lite = lite_lane_per_family(workflow["lanes"])
    check("one_portable_lane_per_os_family",
          {"linux", "macos", "windows"} <= set(lite),
          f"resolved portable lanes: {lite}")

    # ---- class behaviour ------------------------------------------------
    for inert in INERT_CLASSES:
        result = plan(inert, ["docs/README.md"], workflow)
        check(f"inert_{inert}_activates_no_lane",
              result["active"] == "" and not result["active_lanes"],
              f"got {result['active']!r}")

    full = plan("full", ["lib/backend/llvm_codegen.cpp"], workflow)
    check("full_runs_every_lane",
          full["active"] == ALL_LANES and not full["skipped_lanes"],
          f"got {full['active']!r} with skipped={full['skipped_lanes']}")

    unknown = plan("something-new", ["whatever"], workflow)
    check("unknown_class_falls_back_to_every_lane",
          unknown["active"] == ALL_LANES,
          f"got {unknown['active']!r}")

    gpu = plan(CLASS_TESTS_ONLY, ["tests/gpu/gpu_diagnostic_test.esk"], workflow)
    gpu_lanes = [leg["name"] for leg in workflow["lanes"]
                 if leg.get("gpu_enabled") == "ON"]
    check("gpu_tests_select_exactly_the_gpu_lanes",
          sorted(gpu["active_lanes"]) == sorted(gpu_lanes),
          f"got {gpu['active_lanes']}, expected {gpu_lanes}")
    check("gpu_tests_skip_the_portable_lanes",
          all(name not in gpu["active_lanes"] for name in lite.values()),
          f"got {gpu['active_lanes']}")

    generic = plan(CLASS_TESTS_ONLY, ["tests/lists/append_test.esk"], workflow)
    check("generic_tests_select_one_lane_per_os_family",
          sorted(generic["active_lanes"]) == sorted(lite.values()),
          f"got {generic['active_lanes']}, expected {sorted(lite.values())}")
    check("generic_tests_skip_the_rest",
          len(generic["skipped_lanes"]) == len(lane_names) - len(lite),
          f"got {generic['skipped_lanes']}")

    parity = plan(CLASS_TESTS_ONLY, ["tests/vm_parity/PARITY.tsv"], workflow)
    check("vm_parity_selects_one_portable_lane",
          parity["active_lanes"] == [lite["linux"]],
          f"got {parity['active_lanes']}")

    mixed = plan(CLASS_TESTS_ONLY,
                 ["tests/gpu/gpu_diagnostic_test.esk",
                  "tests/lists/append_test.esk"], workflow)
    check("mixed_change_unions_both_rules",
          set(gpu_lanes) <= set(mixed["active_lanes"])
          and set(lite.values()) <= set(mixed["active_lanes"]),
          f"got {mixed['active_lanes']}")

    # A changed file a lane's test_command literally names must reach that
    # lane even though it is not under tests/.
    referenced = None
    for leg in workflow["lanes"]:
        command = leg.get("test_command", "")
        found = re.search(r"(?:\./)?(scripts/[A-Za-z0-9_./-]+\.sh)", command)
        if found:
            referenced = (found.group(1), leg["name"])
            break
    if referenced is None:
        check("direct_reference_activates_the_lane", False,
              "no lane test_command names a scripts/*.sh file")
    else:
        script, lane = referenced
        direct = plan(CLASS_TESTS_ONLY, [script], workflow)
        check("direct_reference_activates_the_lane",
              lane in direct["active_lanes"],
              f"{script} did not activate {lane}: {direct['active_lanes']}")

    # The whole point of step-gating rather than job-gating: on EVERY
    # class, every lane the stub matrix stands in for is either active or
    # still instantiated (and therefore still reports its own context).
    for cls in KNOWN_CLASSES:
        result = plan(cls, ["tests/lists/append_test.esk"], workflow)
        if cls in INERT_CLASSES:
            covered = set(workflow["stub_names"])
        else:
            covered = set(result["active_lanes"]) | set(result["skipped_lanes"])
        missing = [n for n in workflow["stub_names"] if n not in covered]
        check(f"required_contexts_reportable_for_{cls}", not missing,
              f"unreportable: {missing}")

    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decide which CI lanes a change actually needs.")
    parser.add_argument("--impact", help="build-impact class from "
                                         "scripts/ci_change_class.py")
    parser.add_argument("--workflow", default=None,
                        help="workflow file to derive lanes from "
                             "(default: .github/workflows/ci.yml)")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("files", nargs="*",
                        help="changed files (default: read from stdin)")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    if not args.impact:
        parser.error("--impact is required (or pass --self-test)")

    files = list(args.files)
    if not files and not sys.stdin.isatty():
        files = [line.strip() for line in sys.stdin.read().splitlines()
                 if line.strip()]

    try:
        workflow = load_workflow(Path(args.workflow) if args.workflow else None)
    except LanePlanError as exc:
        print(f"ci_lane_plan: {exc}", file=sys.stderr)
        return 1

    result = plan(args.impact, files, workflow)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    print(result["summary"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
