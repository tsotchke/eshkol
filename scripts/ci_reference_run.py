#!/usr/bin/env python3
"""Fail-closed verification of the latest successful CI run and its jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This is the complete set of 28 successful jobs in the approved full CI
# reference. Treat any workflow expansion or display-name change as unknown
# until the reuse contract and its fixture are reviewed together.
EXPECTED_PASSED_JOBS = frozenset({
    "surface-manifest",
    "changes",
    "mesh-gate-advisory (advisory, no-op)",
    "assurance-gates",
    "rosette-oracle (advisory)",
    "prefetch-windows-llvm-archives (x64)",
    "hosted experimental allocation",
    "quantum-macos (advisory)",
    "pillars-fast",
    "bench-smoke (not a measurement)",
    "prefetch-windows-llvm-archives (arm64)",
    "wasm-execute-diff",
    "linux-x64-cuda",
    "macos-x64-lite",
    "linux-arm64-lite",
    "linux-x64-xla",
    "macos-arm64-xla",
    "linux-x64-lite",
    "macos-x64-xla",
    "linux-x64-debug",
    "linux-arm64-xla",
    "macos-arm64-lite",
    "linux-arm64-cuda",
    "linux-x64-asan-ubsan",
    "windows-arm64-xla",
    "windows-x64-cuda",
    "windows-arm64-lite",
    "pillars-readiness",
})
ALLOWED_SKIPS = frozenset({
    "matrix.name",
    "llvm-sdk-cache-keepalive (${{ matrix.arch }})",
})
EXPECTED_JOB_NAMES = EXPECTED_PASSED_JOBS | ALLOWED_SKIPS


def _collect(payload: object, key: str) -> tuple[list[dict], str | None]:
    """Collect all gh --paginate --slurp pages without dropping bad rows."""
    pages = payload if isinstance(payload, list) else [payload]
    if not pages:
        return [], "API response contains no pages"
    rows: list[dict] = []
    totals: set[int] = set()
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get(key), list):
            return [], f"API page is missing its {key!r} list"
        if any(not isinstance(row, dict) for row in page[key]):
            return [], f"API page contains a malformed {key[:-1]} record"
        rows.extend(page[key])
        total = page.get("total_count")
        if not isinstance(total, int) or total < 0:
            return [], "API page is missing a valid total_count"
        totals.add(total)
    if len(totals) != 1 or next(iter(totals)) != len(rows):
        return [], f"API pagination is incomplete (reported {sorted(totals)}, collected {len(rows)})"
    return rows, None


def latest_run(payload: object, head_sha: str, workflow_name: str) -> dict | None:
    runs, error = _collect(payload, "workflow_runs")
    if error:
        return None
    matches = [run for run in runs
               if run.get("head_sha") == head_sha and run.get("name") == workflow_name]
    if not matches:
        return None
    # Reruns keep run_number and increment run_attempt. A later workflow run
    # has a higher run_number, so a previous green run cannot mask a newer
    # failed or cancelled one for the same SHA.
    try:
        return max(matches, key=lambda run: (int(run["run_number"]),
                                             int(run["run_attempt"])))
    except (KeyError, TypeError, ValueError):
        return None


def verify(run: dict | None, jobs_payload: object, head_sha: str,
           workflow_name: str) -> tuple[bool, str]:
    if run is None:
        return False, "no run of the reference workflow exists for this SHA"
    if run.get("head_sha") != head_sha or run.get("name") != workflow_name:
        return False, "latest workflow run does not match the requested workflow and SHA"
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        return False, "latest workflow run is not completed successfully"
    jobs, error = _collect(jobs_payload, "jobs")
    if error:
        return False, error
    if len(jobs) != len(EXPECTED_JOB_NAMES):
        return False, f"expected {len(EXPECTED_JOB_NAMES)} complete job records, found {len(jobs)}"
    names = [job.get("name") for job in jobs]
    if any(not isinstance(name, str) or not name for name in names):
        return False, "job record has no valid name"
    if len(names) != len(set(names)):
        return False, "job records contain duplicate names"
    ids = [job.get("id") for job in jobs]
    if any(not isinstance(job_id, int) for job_id in ids) or len(ids) != len(set(ids)):
        return False, "job records contain missing or duplicate IDs"
    if set(names) != EXPECTED_JOB_NAMES:
        missing = sorted(EXPECTED_JOB_NAMES - set(names))
        unexpected = sorted(set(names) - EXPECTED_JOB_NAMES)
        return False, f"job set is incomplete or unexpected (missing={missing}, unexpected={unexpected})"
    attempt = run.get("run_attempt")
    if not isinstance(attempt, int) or attempt < 1:
        return False, "workflow run has no valid attempt"
    if not any(job.get("run_attempt") == attempt for job in jobs):
        return False, "latest job response contains no job from the current attempt"
    for job in jobs:
        name = job["name"]
        if job.get("run_id") != run.get("id") or job.get("head_sha") != head_sha:
            return False, f"job {name!r} is from a different workflow run or commit"
        # filter=latest retains successful jobs from earlier attempts when
        # only a failed job is rerun. They remain measured passes of this
        # same run and commit; future or invalid attempt numbers do not.
        job_attempt = job.get("run_attempt")
        if not isinstance(job_attempt, int) or not 1 <= job_attempt <= attempt:
            return False, f"job {name!r} has an invalid workflow attempt"
        if job.get("status") != "completed":
            return False, f"job {name!r} is incomplete ({job.get('status')!r})"
        conclusion = job.get("conclusion")
        if name in EXPECTED_PASSED_JOBS and conclusion == "success":
            continue
        if name in ALLOWED_SKIPS and conclusion == "skipped":
            continue
        return False, f"job {name!r} did not pass ({conclusion!r})"
    return True, "latest workflow attempt passed all 28 required and advisory jobs"


def _load(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-json", required=True)
    parser.add_argument("--jobs-json")
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--workflow-name", required=True)
    args = parser.parse_args(argv)
    run = latest_run(_load(args.runs_json), args.head_sha, args.workflow_name)
    if run is None or not args.jobs_json:
        result = {"verified": False, "run_id": run.get("id") if run else None,
                  "run_attempt": run.get("run_attempt") if run else None,
                  "reason": "no matching run" if run is None else
                            "latest run metadata selected; job verification required"}
    else:
        verified, reason = verify(run, _load(args.jobs_json), args.head_sha,
                                  args.workflow_name)
        result = {"verified": verified, "run_id": run.get("id"),
                  "run_attempt": run.get("run_attempt"), "reason": reason}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
