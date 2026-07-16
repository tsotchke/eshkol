#!/usr/bin/env python3
"""Verify that the public website matches the authoritative release matrix."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1.3.3-evolve"


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"{label} is missing {needle!r}")


def release_assets(workflow: str) -> list[str]:
    match = re.search(
        r"Validate Release Asset Set.*?expected=\(\n(?P<body>.*?)\n\s*\)",
        workflow,
        re.DOTALL,
    )
    if not match:
        fail("could not find the release workflow's expected asset array")

    assets = re.findall(r'"(eshkol-\$\{RELEASE_TAG\}-[^\"]+)"', match.group("body"))
    if not assets:
        fail("release workflow expected asset array is empty")
    return [asset.replace("${RELEASE_TAG}", VERSION) for asset in assets]


def matrix_name(asset: str) -> str:
    prefix = f"eshkol-{VERSION}-"
    if not asset.startswith(prefix):
        fail(f"unexpected release asset name: {asset}")
    name = asset[len(prefix) :]
    for suffix in (".tar.gz", ".zip"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    fail(f"unexpected release asset extension: {asset}")
    raise AssertionError("unreachable")


def main() -> int:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    site_source = (ROOT / "site/src/main.esk").read_text(encoding="utf-8")
    index = (ROOT / "site/static/index.html").read_text(encoding="utf-8")
    announcement = (ROOT / "ANNOUNCEMENT.md").read_text(encoding="utf-8")
    announcement_html = (ROOT / "site/static/content/announcement.html").read_text(
        encoding="utf-8"
    )
    announcement_html_text = re.sub(
        r"\s+", " ", re.sub(r"<[^>]+>", "", announcement_html)
    )
    site_wasm = (ROOT / "site/static/eshkol-site.wasm").read_bytes()

    assets = release_assets(workflow)
    if len(assets) != 15:
        fail(f"expected 15 platform packages, release workflow declares {len(assets)}")
    if len(set(assets)) != len(assets):
        fail("release workflow contains duplicate package names")
    if any("windows-arm64-cuda" in asset for asset in assets):
        fail("release workflow advertises unsupported Windows ARM64 CUDA")

    for asset in assets:
        require(site_source, matrix_name(asset), "site release matrix")

    require(
        site_source,
        "15 platform packages plus SHA256SUMS.txt",
        "site download summary",
    )
    require(site_source, "16-file release payload", "site download summary")
    require(site_source, '"releases-container"', "site release loader target")
    require(site_source, '"release-status"', "site publication status target")
    require(site_source, "Unsupported", "Windows ARM64 CUDA matrix cell")
    require(
        site_source,
        "(define route-buf (make-string 256 #\\nul))",
        "mutable pathname router buffer",
    )
    if "16 pre-built binaries" in site_source:
        fail("site still claims that all 16 release files are binaries")
    if (
        "(let ((make-badge (lambda" in site_source
        or "(let ((make-link-card (lambda" in site_source
    ):
        fail("site route rendering still uses unsupported local WASM helper closures")

    require(index, VERSION, "site metadata and release loader")
    require(index, "softwareVersion", "site structured metadata")
    require(index, "expectedTag", "site GitHub release loader")
    require(index, "escapeHtml", "site GitHub release loader")

    require(announcement, "1,057/1,057", "release announcement")
    require(
        announcement,
        "15 platform packages plus `SHA256SUMS.txt`",
        "release announcement",
    )
    require(
        announcement,
        "not a claim of complete backend parity",
        "VM parity disclosure",
    )
    require(announcement_html_text, "1,057/1,057", "generated announcement HTML")
    require(
        announcement_html_text,
        "15 platform packages plus SHA256SUMS.txt",
        "generated announcement HTML",
    )

    for needle in (
        VERSION.encode(),
        b"Fifteen platform packages plus SHA256SUMS.txt",
        b"16-file release payload",
        b"Verified VM subset",
        b"Install Eshkol on macOS, Linux, Windows",
        b"windows-arm64-lite",
        b"windows-arm64-xla",
    ):
        if needle not in site_wasm:
            fail(f"committed site WASM is missing {needle.decode()!r}; rebuild the site")
    if b"16 pre-built binaries" in site_wasm:
        fail("committed site WASM still contains the stale 16-binary claim")

    print(
        "PASS: website matches 15 platform packages + SHA256SUMS.txt, "
        "the publication loader is present, and VM parity wording is scoped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
