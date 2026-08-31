#!/usr/bin/env python3
"""Pin the temporary hosted release-train workflow policy."""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
MANUAL_ONLY = (
    "ci-mesh.yml",
    "adversarial-nightly.yml",
    "pillars-nightly.yml",
)


def event_keys(text: str) -> set[str]:
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if line == "on:"), None)
    if start is None:
        raise ValueError("missing top-level on block")
    keys: set[str] = set()
    for line in lines[start + 1 :]:
        if line and not line.startswith((" ", "#")):
            break
        match = re.match(r"^  ([A-Za-z_]+):", line)
        if match:
            keys.add(match.group(1))
    return keys


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> int:
    for name in MANUAL_ONLY:
        text = (WORKFLOWS / name).read_text()
        events = event_keys(text)
        if events != {"workflow_dispatch"}:
            fail(f"{name} must remain manual-only; found triggers {sorted(events)}")

    ci = (WORKFLOWS / "ci.yml").read_text()
    ci_events = event_keys(ci)
    if ci_events != {"push", "pull_request", "merge_group", "workflow_dispatch"}:
        fail(f"ci.yml must gate the hosted release train; found triggers {sorted(ci_events)}")
    if "reason:" not in ci or "Why hosted fallback is necessary" not in ci:
        fail("ci.yml hosted fallback must require a recorded dispatch reason")

    release = (WORKFLOWS / "release.yml").read_text()
    if "runs-on: ${{ matrix.runner }}" not in release:
        fail("release matrices must use the hosted release-train runner map")
    for required_runner in ("ubuntu-22.04-arm", "macos-15-intel", "windows-11-arm"):
        if required_runner not in release:
            fail(f"release.yml lost required hosted platform {required_runner}")

    local_grid = (WORKFLOWS / "local-grid.yml").read_text()
    required = (
        "workflow_dispatch:",
        "statuses: write",
        "runs-on: [self-hosted, macOS, ARM64, eshkol, grid-controller]",
        "/Users/tyr/EshkolGrid/controller/run_from_github_actions.sh",
    )
    for needle in required:
        if needle not in local_grid:
            fail(f"local-grid.yml lost required trusted-controller contract: {needle}")
    if "actions/checkout" in local_grid:
        fail("the trusted grid controller must never check out pull-request code")

    print("CI release policy: PASS (hosted PR/release gate; heavy nightlies manual)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"CI cost policy: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
