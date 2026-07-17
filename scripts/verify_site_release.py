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


def read_varuint(data: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return result, i


def wasm_export_param_counts(wasm: bytes) -> dict[str, int]:
    """Map exported function names to their parameter counts."""
    i = 8
    func_types: list[int] = []
    type_params: list[int] = []
    imported_funcs = 0
    exports: dict[str, int] = {}
    while i < len(wasm):
        section_id = wasm[i]
        i += 1
        size, i = read_varuint(wasm, i)
        end = i + size
        j = i
        if section_id == 1:  # type section
            count, j = read_varuint(wasm, j)
            for _ in range(count):
                if wasm[j] != 0x60:
                    fail("unexpected wasm type-section entry")
                j += 1
                nparams, j = read_varuint(wasm, j)
                j += nparams
                nresults, j = read_varuint(wasm, j)
                j += nresults
                type_params.append(nparams)
        elif section_id == 2:  # import section
            count, j = read_varuint(wasm, j)
            for _ in range(count):
                mlen, j = read_varuint(wasm, j)
                j += mlen
                nlen, j = read_varuint(wasm, j)
                j += nlen
                kind = wasm[j]
                j += 1
                if kind == 0:
                    _, j = read_varuint(wasm, j)
                    imported_funcs += 1
                elif kind == 1:
                    j += 1
                    flags, j = read_varuint(wasm, j)
                    _, j = read_varuint(wasm, j)
                    if flags & 1:
                        _, j = read_varuint(wasm, j)
                elif kind == 2:
                    flags, j = read_varuint(wasm, j)
                    _, j = read_varuint(wasm, j)
                    if flags & 1:
                        _, j = read_varuint(wasm, j)
                elif kind == 3:
                    j += 2
        elif section_id == 3:  # function section
            count, j = read_varuint(wasm, j)
            for _ in range(count):
                type_idx, j = read_varuint(wasm, j)
                func_types.append(type_idx)
        elif section_id == 7:  # export section
            count, j = read_varuint(wasm, j)
            for _ in range(count):
                nlen, j = read_varuint(wasm, j)
                name = wasm[j : j + nlen].decode()
                j += nlen
                kind = wasm[j]
                j += 1
                idx, j = read_varuint(wasm, j)
                if kind == 0:
                    local_idx = idx - imported_funcs
                    if 0 <= local_idx < len(func_types):
                        exports[name] = type_params[func_types[local_idx]]
        i = end
    return exports


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

    # The exported scheme_main must take no parameters. The underlying Scheme
    # main returns a tagged value that the wasm ABI demotes to an sret
    # out-pointer; exporting that demoted signature let JS glue calling
    # scheme_main(0) write the return value through address 0 and corrupt the
    # first data globals (the SPA router broke on the second navigation).
    exports = wasm_export_param_counts(site_wasm)
    if "scheme_main" not in exports:
        fail("committed site WASM does not export scheme_main")
    if exports["scheme_main"] != 0:
        fail(
            "exported scheme_main takes "
            f"{exports['scheme_main']} parameter(s); it must be a zero-argument "
            "shim so JS re-entry cannot alias linear memory"
        )

    # The playground statistics must match the committed artifacts so the
    # public numbers cannot silently go stale.
    stat_claims = {
        slot: int(value)
        for value, slot in re.findall(
            r'\(create-text "div" "(\d+)KB" (s\d)\)', site_source
        )
    }
    for label, slot, artifact in (
        ("Site WASM", "s1", "site/static/eshkol-site.wasm"),
        ("VM WASM", "s2", "site/static/eshkol-vm.wasm"),
    ):
        actual_kb = round((ROOT / artifact).stat().st_size / 1000)
        claimed = stat_claims.get(slot)
        if claimed is None:
            fail(f"could not find the {label} statistic in site/src/main.esk")
        if abs(claimed - actual_kb) > 1:
            fail(
                f"{label} statistic says {claimed}KB but {artifact} is "
                f"{actual_kb}KB; update the playground stats"
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
