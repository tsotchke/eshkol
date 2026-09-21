#!/usr/bin/env python3
"""Bind split release evidence phases to one source revision and workflow run."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


PHASES = ("baseline", "smoke", "final-evidence")
MARKABLE_PHASES = PHASES


def head_at(repo_root: Path) -> str:
    return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


def read_state(path: Path) -> dict:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"release phase state is missing or invalid: {exc}") from exc
    if not isinstance(state, dict):
        raise RuntimeError("release phase state must be a JSON object")
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("begin", "require", "mark"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--phase", choices=PHASES)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    head = head_at(repo_root)

    if args.action == "begin":
        state = {"schema": "eshkol.release-evidence-phases.v1", "head": head,
                 "phase_id": args.phase_id, "completed": []}
        args.state.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.state.parent,
                                         prefix="release-phase-state.", delete=False) as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temp = Path(handle.name)
        temp.replace(args.state)
        print(f"release evidence phase state started for {head} / {args.phase_id}")
        return 0

    try:
        state = read_state(args.state)
    except RuntimeError as exc:
        print(f"release phase state: FAIL: {exc}")
        return 1
    if state.get("schema") != "eshkol.release-evidence-phases.v1":
        print("release phase state: FAIL: unexpected schema")
        return 1
    if state.get("head") != head or state.get("phase_id") != args.phase_id:
        print("release phase state: FAIL: HEAD or workflow run/attempt does not match this phase")
        return 1
    completed = state.get("completed", [])
    if not isinstance(completed, list) or any(phase not in MARKABLE_PHASES for phase in completed):
        print("release phase state: FAIL: invalid completed-phase list")
        return 1

    if args.action == "require":
        index = PHASES.index(args.phase)
        missing = [phase for phase in PHASES[:index + 1] if phase not in completed]
        if missing:
            print("release phase state: FAIL: prior phase(s) incomplete: " + ", ".join(missing))
            return 1
        print(f"release phase state: PASS (prior phases complete for {args.phase})")
        return 0

    if args.phase not in MARKABLE_PHASES:
        print("release phase state: FAIL: unknown producer phase")
        return 1
    next_phase = MARKABLE_PHASES[len(completed)] if len(completed) < len(MARKABLE_PHASES) else None
    if args.phase != next_phase:
        print(f"release phase state: FAIL: expected next phase {next_phase}, received {args.phase}")
        return 1
    if args.phase in completed:
        print(f"release phase state: FAIL: {args.phase} is already recorded")
        return 1
    completed.append(args.phase)
    state["completed"] = completed
    args.state.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"release phase state: recorded {args.phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
