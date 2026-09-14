#!/usr/bin/env python3
"""Choose the fail-closed base commit for CI's incoming disclosure scan."""

from __future__ import annotations

import argparse
import subprocess
import sys


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def require_commit(rev: str) -> str:
    return git("rev-parse", "--verify", f"{rev}^{{commit}}")


def choose_base(
    *, event: str, head: str, ref: str, default_branch: str,
    before: str = "", merge_group_base: str = "",
) -> str:
    """Return the commit before the incoming range, resolving all refs strictly."""
    head_sha = require_commit(head)

    if event == "merge_group":
        if not merge_group_base:
            raise ValueError("merge_group event has no base SHA")
        return require_commit(merge_group_base)

    if event == "push":
        if not before:
            raise ValueError("push event has no before SHA")
        return require_commit(before)

    if event == "workflow_dispatch":
        default_ref = f"refs/heads/{default_branch}"
        if ref == default_ref:
            # Manual runs on the default branch commonly validate a squash
            # release cut; its parent captures the full cut as one diff.
            return require_commit(f"{head_sha}^")

        if not default_branch:
            raise ValueError(f"invalid repository default branch: {default_branch!r}")
        remote_ref = f"refs/remotes/origin/{default_branch}"
        git("fetch", "--no-tags", "origin", f"{default_ref}:{remote_ref}")
        return require_commit(git("merge-base", head_sha, remote_ref))

    raise ValueError(f"unsupported event for incoming disclosure scan: {event!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--default-branch", required=True)
    parser.add_argument("--before", default="")
    parser.add_argument("--merge-group-base", default="")
    args = parser.parse_args()
    try:
        print(choose_base(
            event=args.event,
            head=args.head,
            ref=args.ref,
            default_branch=args.default_branch,
            before=args.before,
            merge_group_base=args.merge_group_base,
        ))
    except (subprocess.CalledProcessError, ValueError) as exc:
        detail = getattr(exc, "stderr", "")
        if isinstance(exc, subprocess.CalledProcessError):
            detail = detail.strip()
        print(f"disclosure-gate could not select a base commit: {exc} {detail}".rstrip(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
