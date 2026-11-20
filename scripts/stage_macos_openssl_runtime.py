#!/usr/bin/env python3
"""Stage relocatable macOS OpenSSL runtime and AOT link closure."""
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
OPENSSL_DYLIBS = ("libssl.3.dylib", "libcrypto.3.dylib")
OPENSSL_ARCHIVES = ("libssl.a", "libcrypto.a")
LLVM_DYLIBS = ("libLLVM.dylib",)
ZSTD_DYLIBS = ("libzstd.1.dylib",)
SYSTEM_PREFIXES = ("/System/Library/", "/usr/lib/")


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def canonical_package_root(path: Path) -> Path:
    """Resolve OS temp-directory aliases while refusing a symlinked package root."""
    absolute = path.absolute()
    if absolute.is_symlink():
        fail(f"refusing symlinked package staging root: {absolute}")
    try:
        return absolute.resolve(strict=True)
    except OSError as exc:
        fail(f"package staging root is unavailable: {absolute}: {exc}")


def require_no_symlink_descendants(package: Path, path: Path) -> Path:
    """Refuse symlinks below the canonical package root, including the target."""
    package = package.resolve(strict=True)
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(package)
    except ValueError:
        fail(f"staging path escapes package root: {absolute} (root {package})")
    current = package
    for component in relative.parts:
        current /= component
        try:
            current.lstat()
        except FileNotFoundError:
            continue
        if current.is_symlink():
            fail(f"refusing symlinked staging path component: {current}")
    return current


def run(args: list[str]) -> str:
    p = subprocess.run(args, text=True, capture_output=True, check=False)
    if p.returncode:
        fail(f"{' '.join(args)} failed ({p.returncode}): {p.stderr.strip()}")
    return p.stdout


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular_source(prefix: Path, name: str) -> Path:
    path = prefix / "lib" / name
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        fail(f"required OpenSSL payload is missing: {path}: {exc}")
    if not resolved.is_file() or resolved.stat().st_size == 0:
        fail(f"required OpenSSL payload is empty or not a regular file: {path}")
    if not resolved.is_relative_to(prefix.resolve()):
        fail(f"OpenSSL payload escapes selected formula prefix: {path} -> {resolved}")
    return resolved


def license_source(prefix: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        path = prefix / candidate
        if path.is_file() and not path.is_symlink() and path.stat().st_size:
            return path
    fail(f"dependency license not found under formula prefix {prefix}")


def copy_replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            fail(f"refusing to replace non-regular package payload: {target}")
        target.chmod(0o644)
        target.unlink()
    shutil.copy2(source, target)


def install_id(path: Path) -> str:
    lines = run(["otool", "-D", str(path)]).splitlines()
    return lines[1].strip() if len(lines) > 1 else ""


def dylib_refs(path: Path) -> list[str]:
    lines = run(["otool", "-L", str(path)]).splitlines()[1:]
    refs = []
    for line in lines:
        ref = line.strip().split(" (compatibility version", 1)[0]
        if ref:
            refs.append(ref)
    return refs


def add_rpath(path: Path, rpath: str) -> None:
    load_commands = run(["otool", "-l", str(path)])
    if rpath in load_commands:
        return
    run(["install_name_tool", "-add_rpath", rpath, str(path)])


def normalize_rpaths(path: Path, allowed: str) -> None:
    commands = run(["otool", "-l", str(path)])
    for match in re.finditer(r"cmd LC_RPATH\s+cmdsize \d+\s+path (\S+) \(offset \d+\)", commands):
        rpath = match.group(1)
        if rpath != allowed:
            run(["install_name_tool", "-delete_rpath", rpath, str(path)])


def stage(openssl_prefix: Path, llvm_prefix: Path, zstd_prefix: Path,
          package: Path, versions: dict[str, str]) -> dict:
    package = canonical_package_root(package)
    prefixes = {
        "openssl@3": openssl_prefix.resolve(strict=True),
        "llvm@21": llvm_prefix.resolve(strict=True),
        "zstd": zstd_prefix.resolve(strict=True),
    }
    if not package.is_dir():
        fail(f"package staging root is not an existing directory: {package}")
    for formula, prefix in prefixes.items():
        if package == prefix or package in prefix.parents or prefix in package.parents:
            fail(f"package staging root overlaps {formula} provider prefix: {package} / {prefix}")
    out = package / RUNTIME_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    require_no_symlink_descendants(package, out)
    files: dict[str, dict[str, str]] = {}
    payloads = (
        tuple(("openssl@3", name) for name in (*OPENSSL_DYLIBS, *OPENSSL_ARCHIVES))
        + tuple(("llvm@21", name) for name in LLVM_DYLIBS)
        + tuple(("zstd", name) for name in ZSTD_DYLIBS)
    )
    for formula, name in payloads:
        source = regular_source(prefixes[formula], name)
        target = out / name
        copy_replace(source, target)
        files[name] = {"sha256": sha256(target), "source_formula": formula,
                       "source_version": versions[formula]}

    # Rewrite only the staged package image. Preserve system libraries and fail
    # closed on any additional Homebrew or other absolute third-party closure.
    package_binaries = (package / "bin/eshkol-run", package / "bin/eshkol-repl")
    require_no_symlink_descendants(package, package / "bin")
    dylib_rewrites = {
        "libssl.3.dylib": "libssl.3.dylib",
        "libcrypto.3.dylib": "libcrypto.3.dylib",
        "libLLVM.dylib": "libLLVM.dylib",
        "libzstd.1.dylib": "libzstd.1.dylib",
    }
    for binary in package_binaries:
        if not binary.is_file() or binary.is_symlink():
            fail(f"packaged macOS executable is absent or symlinked: {binary}")
        for old in dylib_refs(binary):
            base = Path(old).name
            if base in dylib_rewrites:
                run(["install_name_tool", "-change", old, f"@rpath/{dylib_rewrites[base]}", str(binary)])
            elif old.startswith("/") and not old.startswith(SYSTEM_PREFIXES):
                fail(f"unsupported absolute third-party dependency in {binary.name}: {old}")
        package_rpath = "@executable_path/../lib/eshkol/runtime-deps"
        normalize_rpaths(binary, package_rpath)
        add_rpath(binary, package_rpath)

    all_dylibs = (*OPENSSL_DYLIBS, *LLVM_DYLIBS, *ZSTD_DYLIBS)
    for name in all_dylibs:
        dylib = out / name
        own_id = install_id(dylib)
        for old in dylib_refs(dylib):
            if old == own_id:
                continue
            base = Path(old).name
            if base in dylib_rewrites:
                run(["install_name_tool", "-change", old, f"@rpath/{dylib_rewrites[base]}", str(dylib)])
            elif old.startswith("@rpath/"):
                fail(f"unsupported @rpath dependency in staged OpenSSL library {name}: {old}")
            elif old.startswith("/") and not old.startswith(SYSTEM_PREFIXES):
                fail(f"unsupported absolute third-party dependency in {name}: {old}")
        if own_id:
            run(["install_name_tool", "-id", f"@rpath/{name}", str(dylib)])
        normalize_rpaths(dylib, "@loader_path")
        add_rpath(dylib, "@loader_path")

    # CMake's LLVM config names the aggregate as -lLLVM-21. Provide that
    # link-time filename while retaining the canonical @rpath/libLLVM.dylib ID.
    llvm_alias = out / "libLLVM-21.dylib"
    copy_replace(out / "libLLVM.dylib", llvm_alias)
    files[llvm_alias.name] = {
        "sha256": sha256(llvm_alias), "source_formula": "llvm@21",
        "source_version": versions["llvm@21"],
    }
    # install_name_tool invalidates upstream signatures. Re-seal only the
    # package-local runtime copies with an ad-hoc signature for dyld validation.
    for name in (*all_dylibs, llvm_alias.name):
        signed_path = out / name
        run(["codesign", "--force", "--sign", "-", str(signed_path)])
        files[name]["sha256"] = sha256(signed_path)

    license_specs = {
        "openssl-LICENSE.txt": ("openssl@3", ("LICENSE.txt", "share/doc/openssl@3/LICENSE.txt", "share/doc/openssl/LICENSE.txt", "share/doc/openssl@3/LICENSE")),
        "llvm-LICENSE.TXT": ("llvm@21", ("LICENSE.TXT", "share/doc/llvm/LICENSE.TXT", "share/doc/llvm/LICENSE.txt", "share/doc/llvm/LICENSE")),
        "zstd-LICENSE": ("zstd", ("LICENSE", "LICENSE.txt", "share/doc/zstd/LICENSE", "share/doc/zstd/LICENSE.txt")),
    }
    license_metadata: dict[str, dict[str, str]] = {}
    license_dir = package / "licenses"
    license_dir.mkdir(parents=True, exist_ok=True)
    require_no_symlink_descendants(package, license_dir)
    for output_name, (formula, candidates) in license_specs.items():
        source = license_source(prefixes[formula], candidates)
        target = license_dir / output_name
        copy_replace(source, target)
        license_metadata[output_name] = {
            "sha256": sha256(target), "source_formula": formula,
            "source_version": versions[formula],
        }
    metadata = {
        "schema_version": 2,
        "runtime_subdir": RUNTIME_SUBDIR.as_posix(),
        "providers": versions,
        "files": files,
        "licenses": license_metadata,
    }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openssl-prefix", required=True, type=Path)
    parser.add_argument("--llvm-prefix", required=True, type=Path)
    parser.add_argument("--zstd-prefix", required=True, type=Path)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--openssl-version", required=True)
    parser.add_argument("--llvm-version", required=True)
    parser.add_argument("--zstd-version", required=True)
    args = parser.parse_args()
    try:
        versions = {"openssl@3": args.openssl_version, "llvm@21": args.llvm_version,
                    "zstd": args.zstd_version}
        metadata = stage(args.openssl_prefix, args.llvm_prefix, args.zstd_prefix,
                         args.package_dir, versions)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: staged {len(metadata['files'])} macOS runtime files from OpenSSL, LLVM, and zstd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
