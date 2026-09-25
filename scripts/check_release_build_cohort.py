#!/usr/bin/env python3
"""Pin the main compiler/runtime artifacts across release evidence production."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


ARTIFACTS = (
    "eshkol-run",
    "eshkol-vm-standalone-test",
    "stdlib.o",
    "stdlib.bc",
    "libeshkol-runtime.a",
    "libeshkol-static.a",
    "libeshkol-agent-ffi.a",
)


def snapshot(build_dir: Path) -> dict[str, dict[str, object]]:
    result = {}
    for name in ARTIFACTS:
        path = build_dir / name
        if not path.is_file():
            raise RuntimeError(f"required main-build artifact missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[name] = {"sha256": digest, "size": path.stat().st_size}
    return result


def source_state(build_dir: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(build_dir), *args], check=True,
                                capture_output=True, text=True)
        return result.stdout

    head = git("rev-parse", "HEAD").strip()
    tracked_state = git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_head": head,
        "tracked_state_sha256": hashlib.sha256(tracked_state.encode("utf-8")).hexdigest(),
    }


def emit(trace: Path, value: str, detail: str) -> None:
    trace.parent.mkdir(parents=True, exist_ok=True)
    with trace.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "kind": "release_build_cohort",
            "name": "release_build_cohort_clean",
            "value": value,
            "snippet": detail,
            "timestamp": time.time(),
            "confidence": 1.0,
        }, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "check", "verify"))
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    try:
        current = snapshot(args.build_dir)
        current["__source__"] = source_state(args.build_dir)
        if args.mode == "capture":
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"release build cohort captured: {len(current)} main artifacts")
            return 0
        before = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError) as exc:
        emit(args.trace, "FAIL", str(exc))
        print(f"release build cohort: FAIL: {exc}")
        return 1
    changed = [name for name in ARTIFACTS if before.get(name) != current.get(name)]
    before_source = before.get("__source__") or {}
    current_source = current["__source__"]
    if before_source.get("git_head") != current_source.get("git_head"):
        changed.append("source revision (git HEAD)")
    if before_source.get("tracked_state_sha256") != current_source.get("tracked_state_sha256"):
        changed.append("tracked worktree state")
    if before_source != current_source and not changed:
        changed.append("source state")
    if changed:
        detail = "main compiler/runtime artifacts changed during release evidence: " + ", ".join(changed)
        if args.mode == "verify":
            emit(args.trace, "FAIL", detail)
        print(f"release build cohort: FAIL: {detail}")
        return 1
    if args.mode == "check":
        print(f"release build cohort: unchanged ({len(current)} main artifacts)")
        return 0
    emit(args.trace, "PASS", f"{len(current)} main compiler/runtime artifacts unchanged")
    print(f"release build cohort: PASS ({len(current)} main artifacts unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
