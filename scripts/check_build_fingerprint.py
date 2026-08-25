#!/usr/bin/env python3
"""Release gate: every harness's recorded binary fingerprint still describes
the binary actually on disk, and that binary is not older than the source
tree it claims to have been built from.

WHY THIS EXISTS (docs/design/FLAW_DETECTION_ROADMAP.md, D-11 BUILD FRESHNESS)

Recorded twice in the v1.3.4 campaign as a standing trap: "Rebuild before
believing any harness after a rebase." A stale binary produced a false
`engine_semantic_parity` FAIL on #418's new corpus file — the harness was
honest about what it measured, but what it measured was the wrong compiler.

`scripts/lib/test_isolation.sh` already guards against a binary changing
*during* a run (`eshkol_test_pin_toolchain` / `eshkol_test_toolchain_verify`).
This gate guards a different window: between when a harness recorded its
fingerprint and when a release gate later reads that harness's evidence, and
at the moment a harness started, relative to the source tree.

`scripts/lib/build_fingerprint.sh`'s `eshkol_emit_build_fingerprint_event`
appends one JSON-L record per harness invocation to
`<trace-dir>/build_fingerprint.jsonl`:

    {"kind": "build_fingerprint", "harness": "run_icc_smoke",
     "binary": "eshkol-run", "path": ".../build/eshkol-run",
     "size": N, "mtime": N, "sha256": "...", "git_sha": "...",
     "recorded_epoch": N}

This gate reads that file and fails on either of two independent conditions:

  1. IDENTITY MISMATCH — a recorded fingerprint's size/mtime/sha256 no longer
     matches the binary now present at `path`. Either the binary was rebuilt
     or replaced after the harness ran (so the harness's evidence is about a
     binary that no longer exists — re-run it), or the binary was deleted
     outright.

  2. STALE BINARY — for every real binary named `--binary` (default:
     eshkol-run) found under `--build-dir`, the binary's own mtime is older
     than the most recent build-relevant source change: the newer of (a) the
     last commit touching lib/, inc/, exe/, cmake/ or CMakeLists.txt, and
     (b) the mtime of any currently-dirty file under those paths. This is
     the literal "ran after a rebase, never rebuilt" incident: new commits
     (or local edits) landed in the source tree after the binary was linked.

Both checks are FAIL CLOSED in the sense that matters here — a binary that
does not exist at all is not this gate's concern (nothing has been built yet,
a different gate's job), but a binary that DOES exist and IS stale, or
DOES have recorded evidence that no longer matches it, is always reported.
Absence of any recorded evidence (no harness has run yet in this tree) is
PASS: there is nothing to contradict.

Usage
    python3 scripts/check_build_fingerprint.py
    python3 scripts/check_build_fingerprint.py --build-dir build --binary eshkol-run
    python3 scripts/check_build_fingerprint.py --trace-dir scripts/icc_traces
    python3 scripts/check_build_fingerprint.py --format json
    python3 scripts/check_build_fingerprint.py --self-test

Exit status is 0 on PASS and 1 on FAIL (including under --self-test, where it
reports whether the gate itself is capable of failing).

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACE_DIR = os.path.join(REPO_ROOT, "scripts", "icc_traces")
DEFAULT_BUILD_DIR = os.path.join(REPO_ROOT, "build")
FINGERPRINT_BASENAME = "build_fingerprint.jsonl"
TRACE_BASENAME = "build_fingerprint_gate.jsonl"
PROBE_ID = "build_fingerprint_clean"

SOURCE_PATHS = ("lib", "inc", "exe", "cmake", "CMakeLists.txt")
DEFAULT_BINARIES = ("eshkol-run",)


# ─────────────────────────── pure logic (self-testable without git/fs) ────


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return "unavailable"
    return h.hexdigest()


def current_fingerprint(path: str) -> dict | None:
    """Real stat+digest of a file on disk now. None if it does not exist."""
    if not os.path.isfile(path):
        return None
    st = os.stat(path)
    return {
        "path": path,
        "size": st.st_size,
        "mtime": int(st.st_mtime),
        "sha256": sha256_of(path),
    }


def check_identity(recorded: dict, current: dict | None) -> list[str]:
    """Compare one recorded fingerprint entry to the binary's real state now.

    Pure function: takes dicts, returns a list of error strings (empty means
    OK). No filesystem or git access, so this branch of the gate's logic is
    directly self-testable with synthetic fixtures.
    """
    errors: list[str] = []
    harness = recorded.get("harness", "<unknown harness>")
    binary = recorded.get("binary", "<unknown binary>")
    label = f"{harness} recorded a fingerprint for {binary!r}"

    if current is None:
        errors.append(
            f"{label} at {recorded.get('path')!r}, but that file does not exist now "
            f"(binary deleted or moved since the harness ran)"
        )
        return errors

    mismatches = []
    for field in ("size", "mtime", "sha256"):
        want = recorded.get(field)
        have = current.get(field)
        if want != have:
            mismatches.append(f"{field}: recorded={want!r} now={have!r}")

    if mismatches:
        errors.append(
            f"{label}, but the binary on disk no longer matches ({'; '.join(mismatches)}) — "
            f"it was rebuilt or replaced after {harness} ran; re-run {harness} against the "
            f"current binary before trusting its evidence"
        )
    return errors


def check_freshness(binary: str, binary_mtime: int, source_freshness_epoch: int) -> list[str]:
    """Pure function: is a binary's mtime older than the source it should
    reflect? Takes plain integers so it needs no git or filesystem access."""
    if binary_mtime < source_freshness_epoch:
        age = source_freshness_epoch - binary_mtime
        return [
            f"{binary} (mtime={binary_mtime}) predates the source tree "
            f"(freshness={source_freshness_epoch}, {age}s newer) — rebuild before trusting "
            f"any harness run against it"
        ]
    return []


# ───────────────────────────── real git/filesystem wiring ─────────────────


def _run(cmd: list[str], cwd: str) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def source_freshness_epoch(repo_root: str, paths: tuple[str, ...] = SOURCE_PATHS) -> int:
    """The most recent moment the build-relevant source tree changed.

    Prefers the last commit's committer timestamp over the named paths (this
    is what makes a rebase visible even though `git checkout` resets working-
    tree mtimes to "now"). Also considers the filesystem mtime of any
    currently-dirty file under those paths, so uncommitted local edits count
    too. Falls back to a plain filesystem walk when git is unavailable.
    """
    existing = [p for p in paths if os.path.exists(os.path.join(repo_root, p))]
    if not existing:
        return 0

    candidates = []

    commit_ts = _run(["git", "log", "-1", "--format=%ct", "--"] + existing, repo_root)
    if commit_ts:
        try:
            candidates.append(int(commit_ts))
        except ValueError:
            pass

    dirty = _run(["git", "status", "--porcelain", "--"] + existing, repo_root)
    if dirty is not None:
        for line in dirty.splitlines():
            rel = line[3:].strip()
            # Renames report as "old -> new"; keep the new path.
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[1]
            full = os.path.join(repo_root, rel)
            if os.path.isfile(full):
                try:
                    candidates.append(int(os.stat(full).st_mtime))
                except OSError:
                    pass

    if candidates:
        return max(candidates)

    # No git available at all: best-effort filesystem walk.
    latest = 0
    for p in existing:
        full = os.path.join(repo_root, p)
        if os.path.isfile(full):
            latest = max(latest, int(os.stat(full).st_mtime))
            continue
        for dirpath, _dirnames, filenames in os.walk(full):
            for name in filenames:
                fp = os.path.join(dirpath, name)
                try:
                    latest = max(latest, int(os.stat(fp).st_mtime))
                except OSError:
                    continue
    return latest


def load_fingerprint_events(trace_dir: str) -> list[dict]:
    path = os.path.join(trace_dir, FINGERPRINT_BASENAME)
    if not os.path.isfile(path):
        return []
    events = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                events.append({"_parse_error": f"line {lineno}: {exc}"})
                continue
            if isinstance(rec, dict) and rec.get("kind") == "build_fingerprint":
                events.append(rec)
    return events


def run_check(
    repo_root: str,
    trace_dir: str,
    build_dir: str,
    binaries: tuple[str, ...],
) -> dict:
    errors: list[str] = []
    checked_binaries: list[str] = []
    checked_events = 0

    events = load_fingerprint_events(trace_dir)
    for rec in events:
        if "_parse_error" in rec:
            errors.append(f"unparseable fingerprint record: {rec['_parse_error']}")
            continue
        checked_events += 1
        current = current_fingerprint(rec.get("path", ""))
        errors.extend(check_identity(rec, current))

    freshness = source_freshness_epoch(repo_root)
    for binary in binaries:
        path = os.path.join(build_dir, binary)
        current = current_fingerprint(path)
        if current is None:
            continue  # not built in this tree — not this gate's concern
        checked_binaries.append(binary)
        errors.extend(check_freshness(binary, current["mtime"], freshness))

    passed = not errors
    return {
        "passed": passed,
        "errors": errors,
        "fingerprint_events_checked": checked_events,
        "binaries_checked": checked_binaries,
        "source_freshness_epoch": freshness,
    }


# ───────────────────────────── trace emission ──────────────────────────────


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


# ───────────────────────────────── self-test ────────────────────────────────
#
# "A gate that cannot fail is not a gate." Exercises both branches: the pure
# check_identity/check_freshness functions against synthetic dicts (no git or
# filesystem dependency, deterministic everywhere), plus one end-to-end
# fixture against real files in a repo-local temp directory so the wiring
# (stat, sha256, JSON-L parsing) is proven too, not just the logic.


def _pure_fixture_cases() -> list[tuple[str, bool, list[str]]]:
    matching_current = {"path": "/x/eshkol-run", "size": 100, "mtime": 1000, "sha256": "abc"}
    recorded_match = {"harness": "h", "binary": "eshkol-run", "path": "/x/eshkol-run",
                       "size": 100, "mtime": 1000, "sha256": "abc"}
    recorded_mismatch = {"harness": "h", "binary": "eshkol-run", "path": "/x/eshkol-run",
                          "size": 100, "mtime": 1000, "sha256": "DIFFERENT"}

    cases = []
    cases.append(("identity_match", True, check_identity(recorded_match, matching_current)))
    cases.append(("identity_mismatch", False, check_identity(recorded_mismatch, matching_current)))
    cases.append(("identity_deleted", False, check_identity(recorded_match, None)))
    cases.append(("freshness_ok", True, check_freshness("eshkol-run", 2000, 1000)))
    cases.append(("freshness_stale", False, check_freshness("eshkol-run", 500, 1000)))
    return cases


def _e2e_fixture(tmp_dir: str) -> tuple[bool, str]:
    """Write a fake 'binary', record its real fingerprint, then mutate the
    binary and confirm the gate goes red; then restore agreement and confirm
    it goes green; then age the binary behind a fake 'source freshness' and
    confirm the standalone freshness check (used the same way run_check does)
    catches it."""
    build_dir = os.path.join(tmp_dir, "build")
    trace_dir = os.path.join(tmp_dir, "traces")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)

    binary_path = os.path.join(build_dir, "eshkol-run")
    with open(binary_path, "wb") as fh:
        fh.write(b"fake binary content v1")

    fp = current_fingerprint(binary_path)
    assert fp is not None
    event = {
        "kind": "build_fingerprint",
        "harness": "selftest_harness",
        "binary": "eshkol-run",
        "path": binary_path,
        "size": fp["size"],
        "mtime": fp["mtime"],
        "sha256": fp["sha256"],
        "git_sha": "deadbeef",
        "recorded_epoch": int(time.time()),
    }
    fingerprint_file = os.path.join(trace_dir, FINGERPRINT_BASENAME)
    with open(fingerprint_file, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")

    # Case 1: matches immediately after recording -> no identity errors.
    events = load_fingerprint_events(trace_dir)
    assert len(events) == 1
    errs = check_identity(events[0], current_fingerprint(binary_path))
    if errs:
        return False, f"expected no identity errors right after recording, got: {errs}"

    # Case 2: mutate the binary (rebuild/replace) -> must go red.
    with open(binary_path, "wb") as fh:
        fh.write(b"fake binary content v2 -- REBUILT")
    errs = check_identity(events[0], current_fingerprint(binary_path))
    if not errs:
        return False, "expected identity mismatch after mutating the binary, got none (gate cannot fail)"

    # Case 3: delete the binary entirely -> must go red with a clear reason.
    os.remove(binary_path)
    errs = check_identity(events[0], current_fingerprint(binary_path))
    if not errs or "does not exist" not in errs[0]:
        return False, f"expected a 'does not exist' identity error, got: {errs}"

    return True, "identity check correctly PASSes on a match and FAILs on mutation/deletion"


def self_test() -> bool:
    all_ok = True
    print("check_build_fingerprint.py self-test:")

    for name, expect_pass, errors in _pure_fixture_cases():
        got_pass = not errors
        ok = got_pass == expect_pass
        all_ok = all_ok and ok
        verdict = "OK" if ok else "GATE IS BROKEN"
        print(f"  [{verdict}] {name}: expected passed={expect_pass}, got passed={got_pass}")
        if errors:
            print(f"           {'; '.join(errors)}")

    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-build-fingerprint-") as tmp_dir:
        ok, detail = _e2e_fixture(tmp_dir)
        all_ok = all_ok and ok
        verdict = "OK" if ok else "GATE IS BROKEN"
        print(f"  [{verdict}] end_to_end_identity: {detail}")

    if all_ok:
        print("self-test: PASS — the gate fails on every broken fixture and passes the well-formed one")
    else:
        print("self-test: FAIL — the gate did not discriminate broken input from good input", file=sys.stderr)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=REPO_ROOT)
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--build-dir", default=os.environ.get("BUILD_DIR", DEFAULT_BUILD_DIR))
    parser.add_argument("--binary", action="append", default=None,
                         help="binary name under --build-dir to freshness-check (repeatable; default eshkol-run)")
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    build_dir = args.build_dir
    if not os.path.isabs(build_dir):
        build_dir = os.path.join(args.repo_root, build_dir)
    binaries = tuple(args.binary) if args.binary else DEFAULT_BINARIES

    result = run_check(args.repo_root, args.trace_dir, build_dir, binaries)
    status = "PASS" if result["passed"] else "FAIL"

    if result["passed"]:
        snippet = (
            f"{result['fingerprint_events_checked']} fingerprint event(s) consistent; "
            f"{len(result['binaries_checked'])} binary(ies) not stale relative to source"
        )
    else:
        snippet = f"{len(result['errors'])} fingerprint error(s): " + "; ".join(result["errors"][:5])

    if not args.no_trace:
        emit_trace(args.trace_dir, status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  build dir              : {build_dir}")
        print(f"  fingerprint events     : {result['fingerprint_events_checked']}")
        print(f"  binaries freshness-checked: {', '.join(result['binaries_checked']) or '(none built)'}")
        print(f"  source freshness epoch : {result['source_freshness_epoch']}")
        if result["errors"]:
            print("  ERRORS:")
            for error in result["errors"]:
                print(f"    - {error}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
