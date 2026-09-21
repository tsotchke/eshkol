#!/usr/bin/env python3
"""Move prior release evidence aside without leaving it visible to ICC."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    trace_dir = args.trace_dir.resolve()
    archive_root = args.archive_root.resolve()
    if os.path.commonpath((str(trace_dir), str(archive_root))) == str(trace_dir):
        raise SystemExit("archive root must be outside the active ICC trace directory")
    if not trace_dir.is_dir():
        print("release trace archive: no previous trace directory")
        return 0
    prior = sorted(trace_dir.rglob("*.jsonl"))
    for basename in ("release-build-cohort.json", "release-phase-state.json"):
        manifest = trace_dir / basename
        if manifest.is_file():
            prior.append(manifest)
    if not prior:
        print("release trace archive: no previous cohort to archive")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = archive_root / f"{stamp}-{os.getpid()}"
    destination.mkdir(parents=True, exist_ok=False)
    for source in prior:
        target = destination / source.relative_to(trace_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
    print(f"release trace archive: moved {len(prior)} prior files to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
