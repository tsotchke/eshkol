#!/usr/bin/env python3
"""Check experimental v2 goldens independently using only struct and zlib.

This is a bounded fixture consistency check, not a public v2 reader or writer.
The manifest declares metadata, raw IEEE-754 bits, and extension ordering.
Reconstructing every byte pins those declarations as well as hashes and CRCs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import zlib

ROOT = Path(__file__).resolve().parent.parent
DIRECTORY = ROOT / "tests/core/fixtures/eskm-v2"
EMPTY_HEX = "45534b4d02000000000000000000000000000000000000000570e8c3"


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def reconstruct(expected: dict) -> bytes:
    assert expected["status"] == "ok"
    assert expected["required_features"] == 0
    extensions = bytearray()
    for tlv in expected["extensions"]:
        assert 0 < tlv["type"] <= 0xFFFFFFFF
        if tlv["type"] == 1:
            entries = tlv["annotations"]
            payload = bytearray(u32(len(entries)))
            keys = set()
            for entry in entries:
                key = bytes.fromhex(entry["key_hex"])
                value = bytes.fromhex(entry["value_hex"])
                assert key and key not in keys
                keys.add(key)
                payload += u32(len(key)) + u32(len(value)) + key + value
        else:
            payload = bytes.fromhex(tlv["payload_hex"])
        extensions += u32(tlv["type"]) + u32(len(payload)) + payload
    records = expected["records"]
    data = bytearray(b"ESKM" + u32(2) + u32(len(records)) + u32(0))
    data += u64(len(extensions)) + extensions
    for record in records:
        name = bytes.fromhex(record["name_hex"])
        dimensions = record["dimensions"]
        count = 1
        for dimension in dimensions:
            count *= dimension
            assert count <= 0xFFFFFFFFFFFFFFFF
        assert count == len(record["elements_hex"])
        assert record["dtype"] == 0
        data += u32(len(name)) + name + u32(len(dimensions))
        data += b"".join(u64(dimension) for dimension in dimensions)
        data += bytes([record["dtype"]])
        data += b"".join(u64(int(bits, 16)) for bits in record["elements_hex"])
    crc = zlib.crc32(data)
    assert f"{crc:08x}" == expected["crc32"], "metadata CRC mismatch"
    return bytes(data + u32(crc))


def check(directory: Path, manifest: dict) -> None:
    assert manifest["schema_version"] == 1
    assert (manifest["format"], manifest["format_version"]) == ("ESKM", 2)
    assert manifest["provenance"]["kind"] == "independent specification construction"
    fixtures = manifest["fixtures"]
    names = [fixture["file"] for fixture in fixtures]
    assert len(names) == len(set(names))
    assert set(names) == {path.name for path in directory.glob("*.eskm")}
    for fixture in fixtures:
        name = fixture["file"]
        assert Path(name).name == name and name.endswith(".eskm")
        wire = (directory / name).read_bytes()
        assert len(wire) == fixture["size"] <= 65536, f"{name}: size mismatch"
        assert hashlib.sha256(wire).hexdigest() == fixture["sha256"], f"{name}: digest mismatch"
        assert zlib.crc32(wire[:-4]) == struct.unpack("<I", wire[-4:])[0], f"{name}: CRC mismatch"
        assert wire == reconstruct(fixture["expected"]), f"{name}: metadata/bytes mismatch"
    assert (directory / "empty.eskm").read_bytes().hex() == EMPTY_HEX


def self_test(directory: Path, manifest: dict) -> None:
    mutations = (
        lambda m: m["fixtures"][1]["expected"]["records"][0].update(name_hex="7700fe"),
        lambda m: m["fixtures"][1]["expected"]["records"][0]["elements_hex"].__setitem__(5, "7ff8000000001235"),
        lambda m: m["fixtures"][0]["expected"].update(status="malformed"),
        lambda m: m["fixtures"][0].update(sha256="0" * 64),
        lambda m: m["fixtures"].pop(),
    )
    for mutate in mutations:
        changed = copy.deepcopy(manifest)
        mutate(changed)
        try:
            check(directory, changed)
        except AssertionError:
            continue
        raise AssertionError("negative control unexpectedly passed")
    # A wrong payload paired with its recomputed CRC must still disagree with
    # the checked-in byte stream; checking CRC alone would miss this control.
    changed = copy.deepcopy(manifest["fixtures"][1]["expected"])
    changed["records"][0]["elements_hex"][0] = "0000000000000001"
    original = (directory / "raw-bits.eskm").read_bytes()
    altered = bytearray(original[:-4])
    altered[-48] ^= 1
    changed["crc32"] = f"{zlib.crc32(altered):08x}"
    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["fixtures"][1]["expected"] = changed
    try:
        check(directory, changed_manifest)
    except AssertionError:
        pass
    else:
        raise AssertionError("CRC-correct metadata drift unexpectedly passed")


def main() -> None:
    if not __debug__:
        raise SystemExit("Fixture checking requires assertions; remove -O/PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fixtures", type=Path, default=DIRECTORY)
    parser.add_argument("--test-executable", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    check(args.fixtures, manifest)
    if args.self_test:
        self_test(args.fixtures, manifest)
    if args.test_executable:
        for mode in ("status", "offset", "metadata"):
            completed = subprocess.run(
                [str(args.test_executable.resolve()), str(args.fixtures.resolve()),
                 f"--negative-control={mode}"], capture_output=True, text=True, timeout=30)
            marker = f"negative control rejected: {mode}\n"
            if completed.returncode != 1 or completed.stdout != marker or completed.stderr:
                raise SystemExit(f"{mode} negative control did not fail as intended: "
                                 f"rc={completed.returncode}, stdout={completed.stdout!r}, "
                                 f"stderr={completed.stderr!r}")
        print("ESKM v2 C negative controls: wrong status, offset, and metadata rejected")
    print(f"ESKM v2 fixtures: {len(manifest['fixtures'])} exact-byte goldens passed"
          + ("; negative controls passed" if args.self_test else ""))


if __name__ == "__main__":
    main()
