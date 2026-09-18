#!/usr/bin/env python3
"""No shipped artifact carries a build-host path (ADR-0021).

A compiler records where code came from: in diagnostics, in the location
substrate, and in string constants the backend embeds so the runtime can name
a source location at error time. If the recorded spelling is the absolute host
path, the build machine's directory layout — a home directory, a user name, a
worktree name — travels inside every artifact built from that source, and two
builds of the same source stop being byte-identical.

`inc/eshkol/frontend/source_paths.h` normalizes every recorded path at one
place. This gate is the falsifier for that: it reads shipped artifacts as
bytes and fails on any home-directory path, so a producer that starts
embedding a host path again is caught in the artifact rather than in review.

It needs no build: by default it scans the artifacts checked into the tree
(the site WebAssembly modules). A caller that has just built something —
a packaged binary, an AOT output — passes it with --artifact.

Exit 0 when clean, 1 with one line per finding.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Artifacts that ship from the tree as built files.
DEFAULT_ARTIFACT_GLOBS = ("site/static/*.wasm",)

# A home-directory path in an artifact: the user name is the disclosure, so the
# pattern needs the component after the home root to be a real name.
PATTERNS: tuple[tuple[str, "re.Pattern[bytes]"], ...] = (
    ("unix-home-users", re.compile(rb"/Users/[A-Za-z0-9._-]{1,32}/")),
    ("unix-home", re.compile(rb"/home/[A-Za-z0-9._-]{1,32}/")),
    ("windows-home", re.compile(rb"[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9._ -]{1,32}\\")),
)

CONTEXT = 24


class Finding(tuple):
    __slots__ = ()

    def __new__(cls, artifact: str, pattern: str, excerpt: str):
        return super().__new__(cls, (artifact, pattern, excerpt))

    artifact = property(lambda self: self[0])
    pattern = property(lambda self: self[1])
    excerpt = property(lambda self: self[2])

    def __str__(self) -> str:
        return f"{self.artifact}: {self.pattern}: {self.excerpt}"


def scan_bytes(artifact: str, blob: bytes) -> list[Finding]:
    findings: list[Finding] = []
    for name, pattern in PATTERNS:
        for match in pattern.finditer(blob):
            start = max(0, match.start() - 4)
            end = min(len(blob), match.end() + CONTEXT)
            excerpt = blob[start:end].decode("utf-8", "replace").replace("\n", " ")
            findings.append(Finding(artifact, name, excerpt))
    return findings


def resolve_artifacts(root: Path, extra: list[str]) -> list[Path]:
    paths: list[Path] = []
    for glob in DEFAULT_ARTIFACT_GLOBS:
        paths.extend(sorted(root.glob(glob)))
    for item in extra:
        paths.append(Path(item))
    return paths


def scan_artifacts(root: Path = REPO_ROOT, extra: list[str] | None = None) -> tuple[list[Finding], list[str]]:
    """Returns (findings, scanned artifact names)."""
    findings: list[Finding] = []
    scanned: list[str] = []
    for path in resolve_artifacts(root, extra or []):
        if not path.is_file():
            continue
        name = os.path.relpath(path, root) if path.is_relative_to(root) else str(path)
        scanned.append(name)
        findings.extend(scan_bytes(name, path.read_bytes()))
    return findings, scanned


def self_test() -> int:
    failures = 0
    planted = [
        b"\x00R/Users/someone/Desktop/eshkol/lib/core/ad/interval.esk\x00",
        b"\x00/home/builder/work/eshkol/lib/stdlib.esk\x00",
        b"C:\\Users\\builder\\eshkol\\lib\\stdlib.esk",
    ]
    for blob in planted:
        if not scan_bytes("self-test", blob):
            print(f"self-test: planted host path not caught: {blob!r}")
            failures += 1
    clean = [
        b"\x00Rlib/core/ad/interval.esk\x00",
        b"\x00<unknown>\x00examples/hello.esk\x00",
        b"/usr/lib/eshkol/stdlib.o",
    ]
    for blob in clean:
        found = scan_bytes("self-test", blob)
        if found:
            print(f"self-test: false positive: {found[0]}")
            failures += 1
    if failures:
        print(f"FAIL: artifact host-path gate self-test ({failures})")
        return 1
    print(f"PASS: artifact host-path gate self-test ({len(planted)} planted, {len(clean)} clean)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact", action="append", default=[],
                        help="extra file to scan (repeatable), e.g. a binary a test just built")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    findings, scanned = scan_artifacts(args.root, args.artifact)
    if not scanned:
        print("FAIL: artifact host-path gate scanned nothing (no artifact matched); "
              "a gate whose subject never ran is not a gate")
        return 1
    if findings:
        for finding in findings:
            print(f"  {finding}")
        print(f"FAIL: artifact host-path gate ({len(findings)} finding(s) in "
              f"{len(scanned)} artifact(s)); see ADR-0021")
        return 1
    print(f"PASS: artifact host-path gate ({len(scanned)} artifact(s) clean: {', '.join(scanned)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
