#!/usr/bin/env python3
"""Decide whether a PR head is CONTENT-IDENTICAL to an already-verified commit.

Motivating case: a branch is rebased, re-cut onto a fresh base, or
cherry-picked into a new PR after its original PR was closed. The commit
SHA changes, so every required status context has to be produced again for
the new head -- the full cross-platform matrix runs a second time over a
change whose content nothing has altered. On this repo the macOS runner
pool is the throughput bottleneck, so those re-runs are the single largest
avoidable cost in the queue.

This script answers ONE narrow question, in terms git itself can settle:

    is <head> the same CONTENT as <reference>?

and it answers it two ways, in decreasing strength:

  tree   `git rev-parse <sha>^{tree}` is equal for both commits. The two
         commits have byte-identical working trees. Nothing a build or a
         test could observe differs -- this is the strongest possible
         statement and it is completely independent of history shape,
         author, message, parentage or base.

  patch  `git diff <merge-base(base, sha)> <sha> | git patch-id --stable`
         is equal for both commits. The two commits introduce the same
         change relative to their own bases. This is what `git rebase`
         and `git cherry` use to recognise "already applied" commits, and
         `--stable` makes the id independent of diff ordering. Weaker than
         tree identity -- the two BASES may differ, so the resulting trees
         can differ -- which is why it is reported as its own `kind` and
         why the caller is expected to pair it with proof that the
         reference itself passed CI on a base that is an ancestor of the
         current one.

  none   neither holds. The two commits are not known to be equivalent.
         This is also the answer whenever anything could not be computed
         (an object that does not resolve, an empty diff, a git failure):
         "I could not tell" and "they differ" collapse to the same
         CONSERVATIVE outcome, because the only thing a caller may do with
         a `none` verdict is run the full verification.

What this script deliberately does NOT do
    It never consults GitHub, never reads a PR label, and never decides
    whether the reference commit was actually verified. Equivalence to a
    commit that never passed CI is worth nothing, so the caller
    (`.github/workflows/ci.yml`'s `changes` job) must independently
    confirm that the reference SHA has a completed, successful `CI`
    workflow run before acting on a `true` verdict here. Keeping the two
    halves separate is the point: this half is pure git and is fully
    testable offline; that half is pure GitHub API and cannot be faked by
    anything in the diff.

Usage
    python3 scripts/ci_equivalent_head.py --head <sha> --reference <sha>
    python3 scripts/ci_equivalent_head.py --head HEAD --reference abc123 \\
        --base-ref origin/master
    python3 scripts/ci_equivalent_head.py --head <sha> --reference <sha> \\
        --fetch-remote origin
    python3 scripts/ci_equivalent_head.py --self-test

Output: one JSON object on stdout --

    {"equivalent": bool,
     "kind": "tree" | "patch" | "none",
     "reason": "<human sentence>",
     "head": "<resolved sha or null>",
     "reference": "<resolved sha or null>",
     "checks": {"tree": {...}, "patch": {...}}}

Exit status: 0 whenever a verdict was produced (equivalent or not), 1 on a
usage error or a failing `--self-test`. A `none` verdict is NOT an error
exit -- the caller's fall-through path is a normal, expected outcome.

Scratch space: `--self-test` builds throwaway git repositories under
`<repo>/.scratch/` (never a system temp directory, which this machine
wipes on reboot) and removes them again on the way out.

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = REPO_ROOT / ".scratch"

KIND_TREE = "tree"
KIND_PATCH = "patch"
KIND_NONE = "none"


def _git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _rev_parse(rev: str, cwd: Path) -> str | None:
    code, out, _ = _git(["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], cwd)
    if code != 0:
        return None
    resolved = out.strip()
    return resolved or None


def _tree_of(rev: str, cwd: Path) -> str | None:
    code, out, _ = _git(["rev-parse", "--verify", "--quiet", f"{rev}^{{tree}}"], cwd)
    if code != 0:
        return None
    resolved = out.strip()
    return resolved or None


def _merge_base(a: str, b: str, cwd: Path) -> str | None:
    code, out, _ = _git(["merge-base", a, b], cwd)
    if code != 0:
        return None
    resolved = out.strip()
    return resolved or None


def _patch_id(base: str, tip: str, cwd: Path) -> str | None:
    """The `--stable` patch id of `base..tip`, or None if there isn't one.

    An EMPTY diff has no patch id (git prints nothing), and two empty
    diffs must never be called "the same patch" -- that would make every
    no-op commit equivalent to every other one. None here therefore means
    "no usable evidence", and the caller treats it as `none`.
    """

    diff = subprocess.run(
        ["git", "diff", "--full-index", "--binary", base, tip],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if diff.returncode != 0 or not diff.stdout.strip():
        return None
    ident = subprocess.run(
        ["git", "patch-id", "--stable"],
        cwd=str(cwd),
        input=diff.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ident.returncode != 0:
        return None
    text = ident.stdout.decode("utf-8", "replace").strip()
    if not text:
        return None
    return text.split()[0]


def _try_fetch(remote: str, sha: str, cwd: Path) -> None:
    """Best-effort: make `sha` resolvable in a shallow/partial checkout.

    Never raises. A fetch that fails simply leaves the object missing,
    which the caller reports as `none` -- the conservative answer.
    """

    _git(["fetch", "--no-tags", "--quiet", remote, sha], cwd)


def evaluate(
    head: str,
    reference: str,
    repo_dir: Path,
    base_ref: str = "origin/master",
    fetch_remote: str | None = None,
) -> dict:
    """Compute the equivalence verdict. Never raises on git-level problems."""

    checks: dict[str, dict] = {}

    head_sha = _rev_parse(head, repo_dir)
    if head_sha is None and fetch_remote:
        _try_fetch(fetch_remote, head, repo_dir)
        head_sha = _rev_parse(head, repo_dir)

    reference_sha = _rev_parse(reference, repo_dir)
    if reference_sha is None and fetch_remote:
        _try_fetch(fetch_remote, reference, repo_dir)
        reference_sha = _rev_parse(reference, repo_dir)

    if head_sha is None or reference_sha is None:
        missing = []
        if head_sha is None:
            missing.append(f"head {head!r}")
        if reference_sha is None:
            missing.append(f"reference {reference!r}")
        return {
            "equivalent": False,
            "kind": KIND_NONE,
            "reason": (
                "could not resolve " + " and ".join(missing)
                + " in this checkout; treating the two commits as different"
            ),
            "head": head_sha,
            "reference": reference_sha,
            "checks": checks,
        }

    if head_sha == reference_sha:
        # Same commit. Trivially the same tree; report it as tree identity
        # rather than inventing a third kind.
        checks["tree"] = {"head": _tree_of(head_sha, repo_dir),
                          "reference": _tree_of(reference_sha, repo_dir),
                          "equal": True}
        return {
            "equivalent": True,
            "kind": KIND_TREE,
            "reason": f"head and reference are the same commit ({head_sha})",
            "head": head_sha,
            "reference": reference_sha,
            "checks": checks,
        }

    head_tree = _tree_of(head_sha, repo_dir)
    reference_tree = _tree_of(reference_sha, repo_dir)
    checks["tree"] = {
        "head": head_tree,
        "reference": reference_tree,
        "equal": bool(head_tree and reference_tree and head_tree == reference_tree),
    }
    if checks["tree"]["equal"]:
        return {
            "equivalent": True,
            "kind": KIND_TREE,
            "reason": (
                f"tree identity: {head_sha} and {reference_sha} have the same "
                f"tree {head_tree} -- the two checkouts are byte-identical"
            ),
            "head": head_sha,
            "reference": reference_sha,
            "checks": checks,
        }

    head_base = _merge_base(base_ref, head_sha, repo_dir)
    reference_base = _merge_base(base_ref, reference_sha, repo_dir)
    head_patch = _patch_id(head_base, head_sha, repo_dir) if head_base else None
    reference_patch = (
        _patch_id(reference_base, reference_sha, repo_dir) if reference_base else None
    )
    checks["patch"] = {
        "base_ref": base_ref,
        "head_merge_base": head_base,
        "reference_merge_base": reference_base,
        "head_patch_id": head_patch,
        "reference_patch_id": reference_patch,
        "equal": bool(head_patch and reference_patch and head_patch == reference_patch),
    }
    if checks["patch"]["equal"]:
        return {
            "equivalent": True,
            "kind": KIND_PATCH,
            "reason": (
                f"patch identity: {head_sha} and {reference_sha} both introduce "
                f"patch-id {head_patch} relative to their own merge-base with "
                f"{base_ref}"
            ),
            "head": head_sha,
            "reference": reference_sha,
            "checks": checks,
        }

    detail = "trees differ"
    if head_patch is None or reference_patch is None:
        detail += " and no usable patch-id could be computed for both commits"
    else:
        detail += " and the two patch-ids differ"
    return {
        "equivalent": False,
        "kind": KIND_NONE,
        "reason": (
            f"{head_sha} is not equivalent to {reference_sha}: {detail}"
        ),
        "head": head_sha,
        "reference": reference_sha,
        "checks": checks,
    }


# ───────────────────────────────── self-test ─────────────────────────────────

def _run(args: list[str], cwd: Path, env: dict | None = None) -> None:
    proc = subprocess.run(
        args, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{proc.stdout}")


def _fixture_env() -> dict:
    env = dict(os.environ)
    env.update({
        "GIT_AUTHOR_NAME": "selftest",
        "GIT_AUTHOR_EMAIL": "selftest@example.invalid",
        "GIT_COMMITTER_NAME": "selftest",
        "GIT_COMMITTER_EMAIL": "selftest@example.invalid",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    })
    return env


def _build_fixture(root: Path) -> dict:
    """A throwaway repo containing all three cases this script must separate.

        master:   b0 -- b1 -- b2            (b2 is the current base tip)
        rebased:  b2 -- R      (the same change as O, re-cut onto b2)
        original: b1 -- O      (verified earlier, on the older base)
        copy:     b1 -- C      (identical content to O, different commit)
        other:    b2 -- X      (a genuinely different change)

    `C` vs `O` proves TREE identity across distinct commits; `R` vs `O`
    proves PATCH identity across different bases; `X` vs `O` is the
    negative case.
    """

    env = _fixture_env()
    _run(["git", "init", "--quiet", "--initial-branch=master", "."], root, env)

    def commit(message: str) -> str:
        _run(["git", "add", "-A"], root, env)
        _run(["git", "commit", "--quiet", "--no-gpg-sign", "-m", message], root, env)
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root),
            stdout=subprocess.PIPE, text=True, env=env,
        ).stdout.strip()

    (root / "base.txt").write_text("base 0\n")
    commit("b0")
    (root / "base.txt").write_text("base 1\n")
    b1 = commit("b1")
    (root / "base.txt").write_text("base 2\n")
    b2 = commit("b2")

    # The originally verified head, cut from b1.
    _run(["git", "checkout", "--quiet", "-b", "original", b1], root, env)
    (root / "feature.txt").write_text("feature payload\n")
    original = commit("feature")

    # A byte-identical re-commit of the same content, also on b1.
    _run(["git", "checkout", "--quiet", "-b", "copy", b1], root, env)
    (root / "feature.txt").write_text("feature payload\n")
    copy = commit("feature (recommitted, different message metadata)")

    # The same change rebased onto the newer base b2.
    _run(["git", "checkout", "--quiet", "-b", "rebased", b2], root, env)
    (root / "feature.txt").write_text("feature payload\n")
    rebased = commit("feature")

    # A genuinely different change on the same base.
    _run(["git", "checkout", "--quiet", "-b", "other", b2], root, env)
    (root / "feature.txt").write_text("a different payload entirely\n")
    other = commit("other feature")

    _run(["git", "checkout", "--quiet", "master"], root, env)
    return {"b1": b1, "b2": b2, "original": original, "copy": copy,
            "rebased": rebased, "other": other}


def self_test(scratch_dir: Path | None = None) -> bool:
    print("ci_equivalent_head.py self-test:")
    root_dir = scratch_dir or SCRATCH_ROOT
    root_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="ci_equivalent_head_", dir=str(root_dir)))
    ok = True
    try:
        repo = work / "repo"
        repo.mkdir()
        shas = _build_fixture(repo)

        cases = [
            # name, head, reference, expect_equivalent, expect_kind
            ("tree_identity_across_distinct_commits",
             shas["copy"], shas["original"], True, KIND_TREE),
            ("patch_identity_after_rebase_onto_newer_base",
             shas["rebased"], shas["original"], True, KIND_PATCH),
            ("negative_a_different_change_is_not_equivalent",
             shas["other"], shas["original"], False, KIND_NONE),
            ("negative_an_unresolvable_reference_is_not_equivalent",
             shas["rebased"], "0" * 40, False, KIND_NONE),
            ("same_commit_is_tree_identical",
             shas["original"], shas["original"], True, KIND_TREE),
        ]

        for name, head, reference, want_equivalent, want_kind in cases:
            verdict = evaluate(head, reference, repo, base_ref="master")
            got = (verdict["equivalent"], verdict["kind"])
            want = (want_equivalent, want_kind)
            if got == want:
                print(f"  PASS {name}: {verdict['kind']} "
                      f"(equivalent={verdict['equivalent']})")
            else:
                ok = False
                print(f"  FAIL {name}: expected {want}, got {got} "
                      f"-- {verdict['reason']}")

        # The patch-identity case must NOT be reachable by tree identity --
        # otherwise the "patch" case above would be silently proving the
        # "tree" path a second time instead of the code it names.
        rebased_tree = _tree_of(shas["rebased"], repo)
        original_tree = _tree_of(shas["original"], repo)
        if rebased_tree == original_tree:
            ok = False
            print("  FAIL fixture_integrity: the rebased and original commits "
                  "share a tree, so the patch-identity case never exercised "
                  "the patch-id path")
        else:
            print("  PASS fixture_integrity: the rebased commit really does "
                  "have a different tree than the original")
    except Exception as exc:  # noqa: BLE001 - a self-test reports, never crashes
        ok = False
        print(f"  FAIL self-test raised: {exc}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decide whether a PR head is content-identical to an "
                    "already-verified commit.")
    parser.add_argument("--head", help="the PR head commit-ish")
    parser.add_argument("--reference", help="the already-verified commit-ish")
    parser.add_argument("--base-ref", default="origin/master",
                        help="branch both commits are measured against for "
                             "patch identity (default: origin/master)")
    parser.add_argument("--repo-dir", default=".",
                        help="git repository to evaluate in (default: cwd)")
    parser.add_argument("--fetch-remote", default=None,
                        help="best-effort `git fetch <remote> <sha>` when an "
                             "object does not resolve locally")
    parser.add_argument("--self-test", action="store_true",
                        help="prove both equivalence kinds and a negative case")
    parser.add_argument("--scratch-dir", default=None,
                        help="where --self-test builds its throwaway repos "
                             "(default: <repo>/.scratch)")
    args = parser.parse_args(argv)

    if args.self_test:
        scratch = Path(args.scratch_dir).resolve() if args.scratch_dir else None
        return 0 if self_test(scratch) else 1

    if not args.head or not args.reference:
        parser.error("--head and --reference are both required "
                     "(or pass --self-test)")

    verdict = evaluate(
        args.head,
        args.reference,
        Path(args.repo_dir).resolve(),
        base_ref=args.base_ref,
        fetch_remote=args.fetch_remote,
    )
    json.dump(verdict, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    print(verdict["reason"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
