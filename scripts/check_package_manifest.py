#!/usr/bin/env python3
"""Release gate: a staged package directory matches the declared package manifest.

ADR-0010 gap A10 ("no packaging manifest"): the release workflow
(.github/workflows/release.yml) stages a release archive by an imperative
list of `cp`/`Copy-Item` lines, with no single declared source of truth for
what the finished archive must contain. A copy line silently deleted during
a refactor, or a new build artifact added to CMakeLists.txt but never added
to the packaging step, would ship a release that is missing a file with
nothing in CI to say so — `scripts/verify_release_package.py` smoke-tests
that the package it was HANDED still runs a program, which cannot notice a
file the packaging step never staged in the first place.

This gate reads `.icc/package-manifest.yaml` (the declared package surface)
and checks a STAGED package directory against it, before that directory is
tar'd/zipped. It answers a different question than the smoke test: not "does
this package run a program" but "does this package contain everything it is
declared to contain".

Checks performed, per manifest category:
  - binaries            : bin/<stem>[.exe on Windows] exists and is non-empty
  - stdlib_artifacts     : <stem> exists and is non-empty under every declared dir
  - static_libraries     : <prefix><stem><ext> exists and is non-empty under every
                           declared dir, with the platform-correct prefix/extension
                           (posix "lib" prefix + .a, or no prefix + .lib on Windows)
  - stdlib_sources        : declared `path` entries exist; a `mirror_glob` entry
                           is checked as a full path-set mirror between the
                           repo's real source glob and the staged destination —
                           not a bare count — so a single dropped or renamed
                           module fails the gate even when the total count
                           still looks plausible
  - docs                 : declared `path` entries exist

Every entry may restrict itself to a `platforms` list; entries default to
"all" and are skipped entirely (neither required nor forbidden) on a
platform they do not name.

Usage
    python3 scripts/check_package_manifest.py --package-dir <staged dir> --platform linux
    python3 scripts/check_package_manifest.py --self-test

Exit status is 0 on PASS and 1 on FAIL (including under --self-test, where it
reports whether the gate itself is capable of failing).

Copyright (C) tsotchke
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import platform as _platform
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MANIFEST = REPO_ROOT / ".icc" / "package-manifest.yaml"
DEFAULT_TRACE_DIR = REPO_ROOT / "scripts" / "icc_traces"
TRACE_BASENAME = "package_manifest_gate.jsonl"
PROBE_ID = "package_manifest_complete"

PLATFORMS = ("linux", "macos", "windows")


class PackageManifestError(Exception):
    """The manifest could not be read or parsed at all."""


def _load_yaml(path: Path) -> object:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PackageManifestError(
            "PyYAML is required to check the package manifest (pip install pyyaml)"
        ) from exc

    if not path.is_file():
        raise PackageManifestError(f"manifest not found at {path} (the gate fails closed)")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        raise PackageManifestError(f"manifest at {path} is not parseable: {exc}") from exc


def detect_platform() -> str:
    system = _platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    return "linux"


def _platform_allowed(entry: dict, platform_name: str) -> bool:
    platforms = entry.get("platforms", "all")
    if platforms == "all":
        return True
    if isinstance(platforms, list):
        return platform_name in platforms
    return True


def _check_leaf(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"{label}: missing file {path}")
    elif path.stat().st_size == 0:
        errors.append(f"{label}: {path} exists but is empty")


def _check_binaries(entries: list, package_dir: Path, platform_name: str, errors: list[str], checked: list[int]) -> None:
    for entry in entries:
        if not _platform_allowed(entry, platform_name):
            continue
        if not entry.get("required", True):
            continue
        stem = entry["stem"]
        ext = entry.get("windows_ext", "") if platform_name == "windows" else ""
        filename = f"{stem}{ext}"
        rel_dir = entry.get("dir", "bin")
        checked[0] += 1
        _check_leaf(package_dir / rel_dir / filename, f"binary {stem!r}", errors)


def _check_stdlib_artifacts(entries: list, package_dir: Path, platform_name: str, errors: list[str], checked: list[int]) -> None:
    for entry in entries:
        if not _platform_allowed(entry, platform_name):
            continue
        if not entry.get("required", True):
            continue
        stem = entry["stem"]
        for rel_dir in entry.get("dirs", ["lib"]):
            checked[0] += 1
            _check_leaf(package_dir / rel_dir / stem, f"stdlib artifact {stem!r} ({rel_dir})", errors)


def _static_library_filename(stem: str, posix_prefix: bool, platform_name: str) -> str:
    if platform_name == "windows":
        return f"{stem}.lib"
    prefix = "lib" if posix_prefix else ""
    return f"{prefix}{stem}.a"


def _check_static_libraries(entries: list, package_dir: Path, platform_name: str, errors: list[str], checked: list[int]) -> None:
    for entry in entries:
        if not _platform_allowed(entry, platform_name):
            continue
        if not entry.get("required", True):
            continue
        stem = entry["stem"]
        posix_prefix = bool(entry.get("posix_prefix", False))
        filename = _static_library_filename(stem, posix_prefix, platform_name)
        for rel_dir in entry.get("dirs", ["lib"]):
            checked[0] += 1
            _check_leaf(package_dir / rel_dir / filename, f"static library {stem!r} ({rel_dir})", errors)


def _check_stdlib_sources(entries: list, package_dir: Path, repo_root: Path, platform_name: str, errors: list[str], checked: list[int]) -> None:
    for entry in entries:
        if not _platform_allowed(entry, platform_name):
            continue
        if not entry.get("required", True):
            continue
        if "path" in entry:
            checked[0] += 1
            _check_leaf(package_dir / entry["path"], f"stdlib source {entry['path']!r}", errors)
            continue

        mirror_glob = entry.get("mirror_glob")
        if not mirror_glob:
            errors.append(f"stdlib_sources entry has neither `path` nor `mirror_glob`: {entry!r}")
            continue

        dest_prefix = entry.get("dest_prefix", "")
        min_count = int(entry.get("min_count", 1))
        source_files = sorted(repo_root.glob(mirror_glob))
        source_files = [p for p in source_files if p.is_file()]
        checked[0] += 1

        if len(source_files) < min_count:
            errors.append(
                f"mirror source {mirror_glob!r} under {repo_root} matched only "
                f"{len(source_files)} file(s), expected at least {min_count} — "
                "the glob itself looks broken, not just the mirror"
            )
            continue

        glob_root = repo_root / mirror_glob.split("*")[0].rstrip("/")
        missing: list[str] = []
        for source_file in source_files:
            rel = source_file.relative_to(glob_root) if glob_root.is_dir() else source_file.name
            dest = package_dir / dest_prefix / rel
            if not dest.is_file():
                missing.append(str(rel))

        if missing:
            shown = ", ".join(missing[:8])
            more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
            errors.append(
                f"mirror {mirror_glob!r} -> {dest_prefix!r}: {len(missing)} of "
                f"{len(source_files)} source file(s) not mirrored into the package: {shown}{more}"
            )


def _check_docs(entries: list, package_dir: Path, platform_name: str, errors: list[str], checked: list[int]) -> None:
    for entry in entries:
        if not _platform_allowed(entry, platform_name):
            continue
        if not entry.get("required", True):
            continue
        checked[0] += 1
        _check_leaf(package_dir / entry["path"], f"doc {entry['path']!r}", errors)


def evaluate(manifest: dict, package_dir: Path, repo_root: Path, platform_name: str) -> dict:
    """Validate a staged package directory against a parsed manifest. Never raises."""

    errors: list[str] = []
    checked = [0]

    if not isinstance(manifest, dict) or "package_surface" not in manifest:
        return {
            "passed": False,
            "errors": ["manifest document has no top-level `package_surface` mapping"],
            "checked_count": 0,
        }

    surface = manifest["package_surface"]
    if not isinstance(surface, dict):
        return {"passed": False, "errors": ["`package_surface` is not a mapping"], "checked_count": 0}

    if not package_dir.is_dir():
        errors.append(f"package directory not found: {package_dir} (the gate fails closed)")
        return {"passed": False, "errors": errors, "checked_count": 0}

    _check_binaries(surface.get("binaries", []), package_dir, platform_name, errors, checked)
    _check_stdlib_artifacts(surface.get("stdlib_artifacts", []), package_dir, platform_name, errors, checked)
    _check_static_libraries(surface.get("static_libraries", []), package_dir, platform_name, errors, checked)
    _check_stdlib_sources(surface.get("stdlib_sources", []), package_dir, repo_root, platform_name, errors, checked)
    _check_docs(surface.get("docs", []), package_dir, platform_name, errors, checked)

    return {"passed": not errors, "errors": errors, "checked_count": checked[0]}


def emit_trace(trace_dir: Path, status: str, snippet: str) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / TRACE_BASENAME
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


# ───────────────────────────── self-test ─────────────────────────────
#
# "A gate that cannot fail is not a gate." Builds a tiny synthetic manifest
# and a tiny synthetic repo/package layout entirely under REPO_ROOT (never
# /tmp), and asserts the gate grades each fixture the way it must.

_SELFTEST_MANIFEST = """
version: 1
package_surface:
  binaries:
    - stem: fake-run
      dir: bin
      windows_ext: ".exe"
      required: true
  static_libraries:
    - stem: fake-runtime
      posix_prefix: true
      dirs: [lib]
      required: true
  stdlib_sources:
    - mirror_glob: "srclib/**/*.esk"
      dest_prefix: share/lib
      required: true
      min_count: 2
  docs:
    - path: README.md
      required: true
"""


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_good_fixture(root: Path, platform_name: str) -> tuple[Path, Path]:
    repo_root = root / "repo"
    package_dir = root / "package"
    _write(repo_root / "srclib" / "a.esk")
    _write(repo_root / "srclib" / "nested" / "b.esk")

    ext = ".exe" if platform_name == "windows" else ""
    _write(package_dir / "bin" / f"fake-run{ext}")

    lib_name = "fake-runtime.lib" if platform_name == "windows" else "libfake-runtime.a"
    _write(package_dir / "lib" / lib_name)

    _write(package_dir / "share" / "lib" / "a.esk")
    _write(package_dir / "share" / "lib" / "nested" / "b.esk")
    _write(package_dir / "README.md")
    return repo_root, package_dir


def self_test() -> bool:
    import yaml  # type: ignore

    all_ok = True
    with tempfile.TemporaryDirectory(dir=REPO_ROOT, prefix=".selftest-package-manifest-") as tmp:
        tmp_root = Path(tmp)
        manifest = yaml.safe_load(_SELFTEST_MANIFEST)
        platform_name = "linux"

        print("check_package_manifest.py self-test:")

        # 1. Well-formed fixture -> must PASS.
        repo_root, package_dir = _build_good_fixture(tmp_root / "good", platform_name)
        result = evaluate(manifest, package_dir, repo_root, platform_name)
        ok = result["passed"] is True
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] well_formed: expected passed=True, got {result['passed']}")
        if not ok:
            print(f"           {result['errors']}")

        # 2. Missing binary -> must FAIL.
        repo_root, package_dir = _build_good_fixture(tmp_root / "missing_binary", platform_name)
        (package_dir / "bin" / "fake-run").unlink()
        result = evaluate(manifest, package_dir, repo_root, platform_name)
        ok = result["passed"] is False and any("binary" in e for e in result["errors"])
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] missing_binary: expected passed=False (binary error), got {result}")

        # 3. Mirror drops one source file -> must FAIL, naming the file.
        repo_root, package_dir = _build_good_fixture(tmp_root / "missing_mirror", platform_name)
        (package_dir / "share" / "lib" / "nested" / "b.esk").unlink()
        result = evaluate(manifest, package_dir, repo_root, platform_name)
        ok = result["passed"] is False and any("nested" in e and "b.esk" in e for e in result["errors"])
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] missing_mirror_file: expected passed=False naming nested/b.esk, got {result}")

        # 4. Wrong platform naming (posix .a shipped where the gate is told
        #    the package is for Windows) -> must FAIL, proving the
        #    prefix/extension logic actually discriminates, not just presence.
        repo_root, package_dir = _build_good_fixture(tmp_root / "wrong_platform", "linux")
        result = evaluate(manifest, package_dir, repo_root, "windows")
        ok = result["passed"] is False
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] wrong_platform_ext: expected passed=False, got passed={result['passed']}")

        # 5. Malformed manifest document -> must raise / be reported unusable.
        try:
            bad = yaml.safe_load("package_surface: [not, a, mapping]")
            result = evaluate(bad, package_dir, repo_root, platform_name)
            ok = result["passed"] is False
        except Exception:
            ok = True
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'GATE IS BROKEN'}] malformed_manifest: expected passed=False")

    if all_ok:
        print("self-test: PASS — the gate fails on every broken fixture and passes the well-formed one")
    else:
        print("self-test: FAIL — the gate did not discriminate broken input from good input", file=sys.stderr)
    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--package-dir", default=None, help="staged package directory to verify")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--platform", choices=PLATFORMS, default=None, help="defaults to auto-detected host platform")
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    parser.add_argument("--no-trace", action="store_true", help="grade only, write no trace")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--self-test", action="store_true", help="run built-in red/green fixtures and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    if not args.package_dir:
        print("error: --package-dir is required (unless --self-test)", file=sys.stderr)
        return 2

    platform_name = args.platform or detect_platform()

    try:
        manifest = _load_yaml(Path(args.manifest))
    except PackageManifestError as exc:
        snippet = f"manifest unusable: {exc}"
        if not args.no_trace:
            emit_trace(Path(args.trace_dir), "FAIL", snippet)
        print(f"{PROBE_ID}: FAIL — {exc}", file=sys.stderr)
        return 1

    result = evaluate(manifest, Path(args.package_dir), Path(args.repo_root), platform_name)

    status = "PASS" if result["passed"] else "FAIL"
    if result["passed"]:
        snippet = f"{result['checked_count']} manifest entries checked against {args.package_dir} ({platform_name}), all present"
    else:
        snippet = f"{len(result['errors'])} manifest violation(s): " + "; ".join(result["errors"][:5])

    if not args.no_trace:
        emit_trace(Path(args.trace_dir), status, snippet)

    if args.format == "json":
        print(json.dumps({"status": status, **result}, indent=2))
    else:
        print(f"{PROBE_ID}: {status}")
        print(f"  manifest    : {args.manifest}")
        print(f"  package-dir : {args.package_dir}")
        print(f"  platform    : {platform_name}")
        print(f"  checked     : {result['checked_count']} manifest entries")
        if result["errors"]:
            print("  VIOLATIONS:")
            for error in result["errors"]:
                print(f"    - {error}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
