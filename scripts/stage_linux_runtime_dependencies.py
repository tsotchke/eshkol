#!/usr/bin/env python3
"""Stage the relocatable Linux image-codec runtime closure for a release.

Linux release binaries use the platform libpng/libjpeg/libwebp APIs.  A binary
package must therefore carry the exact shared objects it was linked against,
including the codec-side zlib dependency, rather than assuming the target host
has matching runtime and ``-dev`` packages.  This script copies regular files
(never symlinks), creates stable link-time aliases for Eshkol's AOT linker,
records hashes/package versions, and stages the exact distribution copyright
files for the redistributed objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import NoReturn


RUNTIME_SUBDIR = Path("lib/eshkol/runtime-deps")
REQUIRED_RUNPATH = "$ORIGIN/../lib/eshkol/runtime-deps"

FAMILY_PATTERNS = {
    "libpng": re.compile(r"^libpng(?:16)?\.so(?:\..+)?$"),
    "libjpeg": re.compile(r"^libjpeg\.so(?:\..+)?$"),
    "libwebp": re.compile(r"^libwebp\.so(?:\..+)?$"),
    "libsharpyuv": re.compile(r"^libsharpyuv\.so(?:\..+)?$"),
    "zlib": re.compile(r"^libz\.so(?:\..+)?$"),
}
REQUIRED_FAMILIES = ("libpng", "libjpeg", "libwebp", "zlib")
LINK_ALIASES = {
    "libpng": ("libpng.so", "libpng16.so"),
    "libjpeg": ("libjpeg.so",),
    "libwebp": ("libwebp.so",),
    "libsharpyuv": ("libsharpyuv.so",),
    "zlib": ("libz.so",),
}
LICENSE_OUTPUTS = {
    "libpng": "linux-libpng-copyright.txt",
    "libjpeg": "linux-libjpeg-copyright.txt",
    "libwebp": "linux-libwebp-copyright.txt",
    "libsharpyuv": "linux-libsharpyuv-copyright.txt",
    "zlib": "linux-zlib-copyright.txt",
}


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], description: str) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        fail(
            f"{description} failed with status {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def parse_ldd_output(output: str) -> dict[str, Path]:
    """Return SONAME -> resolved regular-file path from trusted ``ldd`` output."""
    dependencies: dict[str, Path] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if "=> not found" in line:
            fail(f"unresolved shared-library dependency: {line}")
        match = re.match(r"^(\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-fA-F]+\)$", line)
        if not match:
            continue
        soname, resolved = match.groups()
        if resolved.startswith("/"):
            dependencies[soname] = Path(resolved)
    return dependencies


def family_for_soname(soname: str) -> str | None:
    for family, pattern in FAMILY_PATTERNS.items():
        if pattern.fullmatch(soname):
            return family
    return None


def _copy_regular(source: Path, target: Path) -> None:
    resolved = source.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size == 0:
        fail(f"runtime dependency is missing or empty: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            fail(f"runtime dependency destination is not a regular file: {target}")
        if sha256(target) != sha256(resolved):
            fail(f"conflicting runtime dependency payload: {target.name}")
        return
    shutil.copy2(resolved, target, follow_symlinks=True)
    if target.is_symlink() or not target.is_file() or target.stat().st_size == 0:
        fail(f"failed to stage regular runtime dependency: {target}")


def _package_for_file(path: Path) -> tuple[str, str]:
    owner_output = ""
    owner_error = ""
    candidates = (path, path.resolve(strict=True))
    for candidate in dict.fromkeys(str(item) for item in candidates):
        result = subprocess.run(
            ["dpkg-query", "-S", candidate],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            owner_output = result.stdout
            break
        owner_error = result.stderr.strip()
    if not owner_output:
        fail(f"no Debian package owns runtime dependency {path}: {owner_error}")
    owner_line = next((line for line in owner_output.splitlines() if ": " in line), "")
    if not owner_line:
        fail(f"no Debian package owns runtime dependency: {path}")
    package = owner_line.split(": ", 1)[0]
    version = _run(
        ["dpkg-query", "-W", "-f=${Version}", package],
        f"resolve Debian package version for {package}",
    ).strip()
    if not version:
        fail(f"Debian package has no version: {package}")
    return package, version


def _package_copyright(package: str) -> Path:
    listing = _run(
        ["dpkg-query", "-L", package],
        f"list Debian package files for {package}",
    )
    for line in listing.splitlines():
        candidate = Path(line)
        if candidate.name == "copyright" and candidate.exists():
            resolved = candidate.resolve(strict=True)
            if resolved.is_file() and resolved.stat().st_size > 0:
                return resolved
    fail(f"Debian package copyright file is unavailable: {package}")


def _ensure_packaged_runpath(binary: Path) -> None:
    existing = _run(
        ["patchelf", "--print-rpath", str(binary)],
        f"read existing RUNPATH for {binary}",
    ).strip()
    entries = [entry for entry in existing.split(":") if entry and entry != REQUIRED_RUNPATH]
    runpath = ":".join((REQUIRED_RUNPATH, *entries))
    _run(
        ["patchelf", "--set-rpath", runpath, str(binary)],
        f"set relocatable RUNPATH on {binary}",
    )
    dynamic = _run(["readelf", "-d", str(binary)], f"inspect RUNPATH for {binary}")
    runpath_match = re.search(r"\((?:RUNPATH|RPATH)\).*\[(.*?)\]", dynamic)
    if not runpath_match or runpath_match.group(1).split(":")[0] != REQUIRED_RUNPATH:
        fail(
            f"packaged binary does not prefer relocatable codec RUNPATH {REQUIRED_RUNPATH}: "
            f"{binary}"
        )


def _set_origin_runpath(library: Path) -> None:
    _run(
        ["patchelf", "--set-rpath", "$ORIGIN", str(library)],
        f"set relocatable RUNPATH on {library}",
    )
    dynamic = _run(["readelf", "-d", str(library)], f"inspect RUNPATH for {library}")
    if "$ORIGIN" not in dynamic:
        fail(f"staged runtime dependency lacks a relative RUNPATH: {library}")


def stage_runtime_dependencies(build_dir: Path, package_dir: Path) -> dict[str, object]:
    build_dir = build_dir.resolve()
    package_dir = package_dir.resolve()
    if not build_dir.is_dir():
        fail(f"Linux release build directory is unavailable: {build_dir}")
    binaries = (package_dir / "bin" / "eshkol-run", package_dir / "bin" / "eshkol-repl")
    for binary in binaries:
        if binary.is_symlink() or not binary.is_file() or binary.stat().st_size == 0:
            fail(f"packaged Linux executable is missing, empty, or symlinked: {binary}")
        _ensure_packaged_runpath(binary)

    all_dependencies: dict[str, Path] = {}
    for binary in binaries:
        all_dependencies.update(parse_ldd_output(_run(["ldd", str(binary)], f"ldd {binary}")))

    selected: dict[str, tuple[str, Path]] = {}
    for soname, source in sorted(all_dependencies.items()):
        family = family_for_soname(soname)
        if family is not None:
            selected[soname] = (family, source)

    present_families = {family for family, _ in selected.values()}
    missing = sorted(set(REQUIRED_FAMILIES) - present_families)
    if missing:
        fail(f"required Linux image-codec dependency families are absent: {', '.join(missing)}")

    dependency_dir = package_dir / RUNTIME_SUBDIR
    dependency_dir.mkdir(parents=True, exist_ok=True)
    package_metadata: dict[str, tuple[str, str]] = {}
    manifest_files: dict[str, dict[str, str]] = {}

    for soname, (family, source) in selected.items():
        destination = dependency_dir / soname
        _copy_regular(source, destination)
        package, version = _package_for_file(source)
        package_metadata.setdefault(family, (package, version))
        manifest_files[soname] = {
            "family": family,
            "sha256": sha256(destination),
            "source_package": package,
            "source_version": version,
        }

    for family in sorted(present_families):
        source_name = next(name for name, (candidate, _) in selected.items() if candidate == family)
        source = dependency_dir / source_name
        for alias in LINK_ALIASES[family]:
            target = dependency_dir / alias
            _copy_regular(source, target)
            manifest_files.setdefault(
                alias,
                {
                    "family": family,
                    "sha256": sha256(target),
                    "source_package": package_metadata[family][0],
                    "source_version": package_metadata[family][1],
                },
            )

    # DT_RUNPATH is not inherited from eshkol-run by indirect dependencies.
    # Give every redistributed codec object its own $ORIGIN lookup so libpng's
    # zlib and newer libwebp's libsharpyuv dependencies remain inside the
    # package as well.  Hashes below intentionally describe the patched,
    # shipped bytes rather than the unmodified distro files.
    for name in sorted(manifest_files):
        path = dependency_dir / name
        _set_origin_runpath(path)
        manifest_files[name]["sha256"] = sha256(path)

    licenses_dir = package_dir / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    staged_licenses: dict[str, dict[str, str]] = {}
    for family, output_name in LICENSE_OUTPUTS.items():
        if family not in package_metadata:
            continue
        package, version = package_metadata[family]
        source = _package_copyright(package)
        target = licenses_dir / output_name
        _copy_regular(source, target)
        staged_licenses[output_name] = {
            "family": family,
            "sha256": sha256(target),
            "source_package": package,
            "source_version": version,
        }

    manifest: dict[str, object] = {
        "schema_version": 1,
        "runtime_subdir": RUNTIME_SUBDIR.as_posix(),
        "required_runpath": REQUIRED_RUNPATH,
        "files": dict(sorted(manifest_files.items())),
        "licenses": dict(sorted(staged_licenses.items())),
    }
    manifest_path = dependency_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--package-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = stage_runtime_dependencies(args.build_dir, args.package_dir)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS: staged relocatable Linux image-codec runtime closure "
        f"({len(manifest['files'])} files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
