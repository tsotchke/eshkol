#!/usr/bin/env python3
"""Bind a release tag and its ICC oracle to the checked-in project identity."""
import argparse
from pathlib import Path
import re


def project_identity():
    cmake = (Path(__file__).resolve().parents[1] / "CMakeLists.txt").read_text()
    versions = re.findall(r'^project\("Eshkol" VERSION (\d+\.\d+\.\d+) LANGUAGES C CXX\)$', cmake, re.M)
    names = re.findall(r'^set\(ESHKOL_RELEASE_NAME "([a-z][a-z0-9-]*)"\)$', cmake, re.M)
    if len(versions) != 1 or len(names) != 1:
        raise ValueError("expected one project version and one release name")
    version, name = versions[0], names[0]
    return f"v{version}-{name}", f"v{version.rsplit('.', 1)[0]}-{name}"


def validate_target(value: str) -> str:
    _, target = project_identity()
    if value != target:
        raise ValueError(f"expected readiness target {target}, got {value!r}")
    return target


def target_for_tag(tag: str) -> str:
    expected, target = project_identity()
    if tag != expected:
        raise ValueError(f"expected release tag {expected}, got {tag!r}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-version", action="store_true", help="print the configured package version")
    parser.add_argument("--tag")
    parser.add_argument("--target")
    args = parser.parse_args()
    try:
        tag, target = project_identity()
        if args.tag is not None:
            target_for_tag(args.tag)
        if args.target is not None:
            validate_target(args.target)
        print(tag[1:] if args.package_version else target)
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
