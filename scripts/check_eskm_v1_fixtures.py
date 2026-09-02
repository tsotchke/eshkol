#!/usr/bin/env python3
"""Verify the immutable ESKM v1 compatibility corpus without a build.

Checks the manifest/corpus bijection, sizes, SHA-256 digests, CRC-32 footers,
wire structure, expected rejection category, and exact metadata/payload bits.
Use --self-test to prove that a checksummed payload drift is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "tests/core/fixtures/eskm-v1/manifest.json"
HEX_DIGITS = frozenset("0123456789abcdef")
MAX_FIXTURE_BYTES = 64 * 1024
EXPECTED_PROVENANCE = {
    "producer_tag": "v1.2.4",
    "tag_object": "4e07f166a7a0da28d24c78fb1c1af4258c4c1845",
    "producer_commit": "b98dc8b32399de739a037e9fa0a470bf0426eca9",
    "producer_source": "lib/core/model_io.cpp",
    "producer_source_sha256": "53e10801ef30701705e643462418bde4649dfc975f0c8731f7762dc8527de889",
    "producer_function": "write_checkpoint",
}


class FormatError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def take(self, size: int, what: str) -> bytes:
        if size < 0 or size > len(self.data) - self.offset:
            raise FormatError("truncated", f"truncated while reading {what}")
        value = self.data[self.offset : self.offset + size]
        self.offset += size
        return value

    def u8(self, what: str) -> int:
        return self.take(1, what)[0]

    def u32(self, what: str) -> int:
        return struct.unpack("<I", self.take(4, what))[0]

    def u64(self, what: str) -> int:
        return struct.unpack("<Q", self.take(8, what))[0]


def parse_eskm(data: bytes) -> dict[str, Any]:
    if len(data) < 20:
        raise FormatError("truncated", "file is shorter than header plus footer")

    payload = data[:-4]
    stored_crc = struct.unpack("<I", data[-4:])[0]
    computed_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if stored_crc != computed_crc:
        raise FormatError(
            "crc32", f"stored CRC {stored_crc:08x} != computed {computed_crc:08x}"
        )

    reader = Reader(payload)
    if reader.take(4, "magic") != b"ESKM":
        raise FormatError("magic", "magic is not ESKM")
    version = reader.u32("version")
    if version != 1:
        raise FormatError("version", f"unsupported version {version}")
    record_count = reader.u32("record count")
    flags = reader.u32("flags")
    if flags != 0:
        raise FormatError("flags", f"reserved flags are nonzero: {flags:#x}")

    if record_count > (len(payload) - reader.offset) // 9:
        raise FormatError("record-count", "record count cannot fit in the payload")

    records: list[dict[str, Any]] = []
    for record_index in range(record_count):
        name_len = reader.u32(f"record {record_index} name length")
        name = reader.take(name_len, f"record {record_index} name")
        rank = reader.u32(f"record {record_index} rank")
        if rank > (len(payload) - reader.offset) // 8:
            raise FormatError("rank", f"record {record_index} dimensions cannot fit")
        dimensions = [reader.u64(f"record {record_index} dimension") for _ in range(rank)]
        dtype = reader.u8(f"record {record_index} dtype")
        if dtype != 0:
            raise FormatError("dtype", f"record {record_index} has unsupported dtype {dtype}")

        element_count = 1
        for dimension in dimensions:
            if dimension == 0:
                element_count = 0
                break
            if element_count > 0xFFFFFFFFFFFFFFFF // dimension:
                raise FormatError("shape-overflow", f"record {record_index} shape overflows u64")
            element_count *= dimension
        if element_count > (len(payload) - reader.offset) // 8:
            raise FormatError("element-count", f"record {record_index} elements cannot fit")
        elements = [reader.u64(f"record {record_index} element") for _ in range(element_count)]
        records.append(
            {
                "name_hex": name.hex(),
                "dimensions": dimensions,
                "dtype": dtype,
                "elements_hex": [f"{element:016x}" for element in elements],
            }
        )

    if reader.offset != len(payload):
        raise FormatError("trailing-data", f"{len(payload) - reader.offset} unconsumed byte(s)")
    return {"flags": flags, "crc32": f"{stored_crc:08x}", "records": records}


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in HEX_DIGITS for character in value)
    )


def _is_int(value: object, minimum: int = 0, maximum: int | None = None) -> bool:
    return (
        type(value) is int
        and value >= minimum
        and (maximum is None or value <= maximum)
    )


def validate_expected(label: str, expected: object) -> list[str]:
    if not isinstance(expected, dict) or type(expected.get("accept")) is not bool:
        return [f"{label}: expected.accept must be boolean"]
    if not expected["accept"]:
        return [] if isinstance(expected.get("error"), str) and expected["error"] else [
            f"{label}: rejected fixture must name a non-empty error category"
        ]

    errors: list[str] = []
    if not _is_int(expected.get("flags")) or expected["flags"] != 0:
        errors.append(f"{label}: expected.flags must be integer 0")
    if not _is_hex(expected.get("crc32"), 8):
        errors.append(f"{label}: expected.crc32 must be eight lowercase hex digits")
    records = expected.get("records")
    if not isinstance(records, list):
        return errors + [f"{label}: expected.records must be an array"]
    for index, record in enumerate(records):
        record_label = f"{label} record {index}"
        if not isinstance(record, dict):
            errors.append(f"{record_label}: record must be an object")
            continue
        name_hex = record.get("name_hex")
        if not isinstance(name_hex, str) or len(name_hex) % 2 or not all(
            character in HEX_DIGITS for character in name_hex
        ):
            errors.append(f"{record_label}: name_hex must contain complete lowercase bytes")
        dimensions = record.get("dimensions")
        if not isinstance(dimensions, list) or not all(
            _is_int(dimension, maximum=0xFFFFFFFFFFFFFFFF) for dimension in dimensions
        ):
            errors.append(f"{record_label}: dimensions must be u64 integers")
        if not _is_int(record.get("dtype")) or record["dtype"] != 0:
            errors.append(f"{record_label}: dtype must be integer 0")
        elements = record.get("elements_hex")
        if not isinstance(elements, list) or not all(_is_hex(element, 16) for element in elements):
            errors.append(f"{record_label}: elements_hex must contain 16-digit lowercase words")
    return errors


def read_small_fixture(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(MAX_FIXTURE_BYTES + 1)
    if len(data) > MAX_FIXTURE_BYTES:
        raise ValueError(f"exceeds the {MAX_FIXTURE_BYTES}-byte fixture limit")
    return data


def apply_mutation(source: bytes, mutation: object) -> bytes:
    if not isinstance(mutation, dict) or type(mutation.get("recompute_crc32")) is not bool:
        raise ValueError("mutation must be an object with boolean recompute_crc32")

    operation = mutation.get("operation")
    result = bytearray(source)
    if operation in ("replace", "insert"):
        offset = mutation.get("offset")
        data_hex = mutation.get("data_hex")
        if not _is_int(offset, maximum=len(result)):
            raise ValueError(f"{operation} offset is outside the source")
        if not isinstance(data_hex, str) or not data_hex or len(data_hex) % 2 or not all(
            character in HEX_DIGITS for character in data_hex
        ):
            raise ValueError(f"{operation} data_hex must contain lowercase bytes")
        replacement = bytes.fromhex(data_hex)
        if operation == "replace":
            if offset + len(replacement) > len(result):
                raise ValueError("replacement extends beyond the source")
            result[offset : offset + len(replacement)] = replacement
        else:
            result[offset:offset] = replacement
    elif operation == "xor":
        offset = mutation.get("offset")
        mask = mutation.get("mask")
        if not _is_int(offset, maximum=len(result) - 1):
            raise ValueError("xor offset is outside the source")
        if not _is_int(mask, minimum=1, maximum=0xFF):
            raise ValueError("xor mask must be an integer from 1 through 255")
        result[offset] ^= mask
    elif operation == "truncate":
        size = mutation.get("size")
        if not _is_int(size, maximum=len(result) - 1):
            raise ValueError("truncate size must be smaller than the source")
        del result[size:]
    else:
        raise ValueError(f"unsupported mutation operation: {operation!r}")

    if mutation["recompute_crc32"]:
        if len(result) < 4:
            raise ValueError("cannot recompute CRC-32 without a footer")
        result[-4:] = struct.pack("<I", zlib.crc32(result[:-4]) & 0xFFFFFFFF)
    return bytes(result)


def check_manifest(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"manifest is unavailable or invalid JSON: {exc}"]

    if not isinstance(manifest, dict):
        return ["manifest root must be an object"]
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        errors.append("schema_version must be 1")
    if (
        type(manifest.get("format")) is not str
        or manifest["format"] != "ESKM"
        or type(manifest.get("format_version")) is not int
        or manifest["format_version"] != 1
    ):
        errors.append("format must identify ESKM version 1")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    else:
        for field, expected_value in EXPECTED_PROVENANCE.items():
            if type(provenance.get(field)) is not str or provenance[field] != expected_value:
                errors.append(f"provenance {field} does not match the pinned v1.2.4 producer")

    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        return errors + ["fixtures must be a non-empty array"]

    corpus_dir = manifest_path.parent
    listed: set[str] = set()
    fixture_data: dict[str, bytes] = {}
    origin_kinds: dict[str, object] = {}
    derived_sources: list[tuple[str, str, object]] = []
    for index, fixture in enumerate(fixtures):
        label = f"fixture {index}"
        if not isinstance(fixture, dict):
            errors.append(f"{label} must be an object")
            continue
        filename = fixture.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".eskm"):
            errors.append(f"{label} has an unsafe or invalid file name")
            continue
        label = filename
        if filename in listed:
            errors.append(f"duplicate fixture file: {filename}")
            continue
        listed.add(filename)

        path = corpus_dir / filename
        try:
            data = read_small_fixture(path)
        except (OSError, ValueError) as exc:
            errors.append(f"{label}: cannot read: {exc}")
            continue
        fixture_data[filename] = data
        declared_size = fixture.get("size")
        if not _is_int(declared_size, maximum=MAX_FIXTURE_BYTES):
            errors.append(f"{label}: size must be an integer within the fixture limit")
        elif declared_size != len(data):
            errors.append(f"{label}: size mismatch (manifest {fixture.get('size')}, actual {len(data)})")
        digest = hashlib.sha256(data).hexdigest()
        if not _is_hex(fixture.get("sha256"), 64):
            errors.append(f"{label}: sha256 must be a lowercase SHA-256 digest")
        if fixture.get("sha256") != digest:
            errors.append(f"{label}: SHA-256 mismatch")

        expected = fixture.get("expected")
        expected_errors = validate_expected(label, expected)
        if expected_errors:
            errors.extend(expected_errors)
            continue
        origin = fixture.get("origin")
        if not isinstance(origin, dict):
            errors.append(f"{label}: origin must be an object")
        else:
            origin_kinds[filename] = origin.get("kind")
            if expected["accept"] and origin.get("kind") != "historical-writer":
                errors.append(f"{label}: accepted fixture must identify the historical writer")
            elif not expected["accept"] and (
                origin.get("kind") != "derived-malformed"
                or not isinstance(origin.get("source"), str)
                or not isinstance(origin.get("mutation"), dict)
            ):
                errors.append(f"{label}: rejected fixture must identify its source and mutation")
            elif not expected["accept"]:
                derived_sources.append((label, origin["source"], origin["mutation"]))
        try:
            actual = parse_eskm(data)
        except FormatError as exc:
            if expected["accept"]:
                errors.append(f"{label}: expected acceptance, got {exc.code}: {exc}")
            elif expected.get("error") != exc.code:
                errors.append(
                    f"{label}: expected rejection {expected.get('error')!r}, got {exc.code!r}"
                )
        else:
            if not expected["accept"]:
                errors.append(f"{label}: expected rejection {expected.get('error')!r}, but parsed")
            else:
                declared = {key: expected.get(key) for key in ("flags", "crc32", "records")}
                if actual != declared:
                    errors.append(f"{label}: parsed metadata or payload differs from manifest")

    present = {path.name for path in corpus_dir.glob("*.eskm") if path.is_file()}
    for filename in sorted(listed - present):
        errors.append(f"missing listed fixture: {filename}")
    for filename in sorted(present - listed):
        errors.append(f"unlisted fixture: {filename}")
    for filename, source, mutation in derived_sources:
        if source not in listed or source == filename:
            errors.append(f"{filename}: malformed origin source is not another listed fixture")
        elif source not in fixture_data:
            errors.append(f"{filename}: malformed origin source could not be read")
        elif origin_kinds.get(source) != "historical-writer":
            errors.append(f"{filename}: malformed origin source is not historical-writer output")
        else:
            try:
                reconstructed = apply_mutation(fixture_data[source], mutation)
            except ValueError as exc:
                errors.append(f"{filename}: invalid mutation: {exc}")
            else:
                if reconstructed != fixture_data.get(filename):
                    errors.append(f"{filename}: bytes do not match the declared mutation")
    return errors


def self_test() -> bool:
    baseline_errors = check_manifest(DEFAULT_MANIFEST)
    if baseline_errors:
        print("  [FAIL] committed corpus is not a valid green baseline")
        for error in baseline_errors:
            print(f"    - {error}")
        return False
    print("  [OK] committed corpus passes")

    with tempfile.TemporaryDirectory(prefix="eskm-v1-selftest-") as temporary:
        corpus = Path(temporary) / "eskm-v1"
        shutil.copytree(DEFAULT_MANIFEST.parent, corpus)
        manifest_path = corpus / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next(item for item in manifest["fixtures"] if item["file"] == "scalar.eskm")
        fixture_path = corpus / entry["file"]
        data = bytearray(fixture_path.read_bytes())
        data[25] ^= 0x01
        data[-4:] = struct.pack("<I", zlib.crc32(data[:-4]) & 0xFFFFFFFF)
        fixture_path.write_bytes(data)
        entry["sha256"] = hashlib.sha256(data).hexdigest()
        entry["expected"]["crc32"] = f"{struct.unpack('<I', data[-4:])[0]:08x}"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        negative_errors = check_manifest(manifest_path)
        if not any("parsed metadata or payload differs" in error for error in negative_errors):
            print("  [FAIL] checksummed payload drift was not rejected")
            return False
    print("  [OK] checksummed payload drift is rejected")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print("check_eskm_v1_fixtures.py self-test:")
        passed = self_test()
        print(f"self-test: {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1

    errors = check_manifest(args.manifest.resolve())
    if errors:
        print("ESKM v1 fixtures: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    accepted = sum(item["expected"]["accept"] for item in manifest["fixtures"])
    rejected = len(manifest["fixtures"]) - accepted
    print(f"ESKM v1 fixtures: PASS ({accepted} accepted, {rejected} rejected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
