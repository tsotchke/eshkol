#!/usr/bin/env python3
"""Exercise standalone ESKB admission with tiny CRC-valid containers."""
import pathlib
import struct
import subprocess
import sys
import tempfile
import zlib


def leb(value):
    out = bytearray()
    while True:
        byte = value & 0x7f
        value >>= 7
        if value:
            byte |= 0x80
        out.append(byte)
        if not value:
            return bytes(out)


def instruction(opcode, operand=0):
    zigzag = (operand << 1) ^ (operand >> 31)
    return bytes((opcode,)) + leb(zigzag)


def int_constant(value):
    return bytes((1,)) + struct.pack("<q", value)


def float_constant(value):
    return bytes((2,)) + struct.pack("<d", value)


def string_constant(value):
    encoded = value.encode("ascii")
    return bytes((6,)) + leb(len(encoded)) + encoded


def eskb(instructions, constants=(), advertised_count=None):
    # Tiny single-function containers; all count-controlled sections stay
    # small so malformed inputs cannot trigger large allocations.
    code = bytearray(leb(1))
    code += leb(4) + b"main"
    code += b"\0" + leb(0) + leb(0)
    code += leb(len(instructions))
    for opcode, operand in instructions:
        code += instruction(opcode, operand)
    constant_section = bytearray(leb(
        len(constants) if advertised_count is None else advertised_count))
    constant_section += b"".join(constants)
    payload = bytearray(leb(2))
    payload += bytes((0,)) + leb(len(constant_section))
    payload += bytes((1,)) + leb(len(code))
    payload += constant_section + code
    header = struct.pack("<IIII", 0x45534b42, 2, 1,
                         zlib.crc32(payload) & 0xffffffff)
    return header + payload


def run(binary, path):
    try:
        return subprocess.run([binary, str(path)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"{binary} hung on {path.name}") from exc


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: standalone_eskb_admission_test.py VM [VM ...]")
    with tempfile.TemporaryDirectory(prefix="eshkol-eskb-admission-") as tmp:
        root = pathlib.Path(tmp)
        cases = {
            "valid": eskb([(36, 0)]),
            "valid-constant-pool": eskb([(1, 0), (5, 0), (36, 0)],
                                        [int_constant(42)]),
            "valid-closure-constant": eskb([(24, 0), (5, 0), (36, 0)],
                                            [int_constant(0)]),
            "valid-closure-long-constant": eskb(
                [(67, 0), (68, 0), (5, 0), (36, 0)], [int_constant(0)]),
            "invalid-opcode": eskb([(255, 0)]),
            "invalid-constant-index": eskb([(1, 1)]),
            # The marker would be printed if the malformed closure passed
            # module admission and execution began.
            "invalid-closure-constant-type": eskb(
                [(24, 0), (5, 0), (1, 1), (35, 0), (36, 0)],
                [float_constant(0.0), string_constant("ADMISSION_MARKER")]),
            "invalid-closure-constant-index": eskb(
                [(24, 1), (5, 0), (1, 0), (35, 0), (36, 0)],
                [int_constant(0), string_constant("ADMISSION_MARKER")]),
            "invalid-closure-long-constant-type": eskb(
                [(67, 0), (68, 0), (5, 0), (1, 1), (35, 0), (36, 0)],
                [float_constant(0.0), string_constant("ADMISSION_MARKER")]),
            "invalid-closure-long-constant-index": eskb(
                [(67, 1), (68, 0), (5, 0), (1, 0), (35, 0), (36, 0)],
                [int_constant(0), string_constant("ADMISSION_MARKER")]),
            # Advertised pool records are intentionally tiny. These exercise
            # EOF and count/entry mismatches without approaching the profile
            # ceiling, whose parser allocation occurs before profile checks.
            "truncated-constant-entry": eskb([(36, 0)], [bytes((1, 0, 0, 0))]),
            "advertised-constant-count-mismatch": eskb(
                [(36, 0)], [int_constant(7)], advertised_count=2),
            # OP_VEC_CREATE (39) uses -(count + 1) for numeric tensors.
            # Both signed extremes must fail admission before allocation;
            # decoding INT32_MIN must also avoid signed negation overflow.
            "invalid-vector-count-int32-max": eskb([(39, (1 << 31) - 1)]),
            "invalid-tensor-count-int32-min": eskb([(39, -(1 << 31))]),
            "truncated": eskb([(36, 0)])[:-1],
        }
        paths = {}
        for name, data in cases.items():
            if len(data) > 1024 * 1024:
                raise AssertionError(f"fixture exceeds 1 MiB: {name}")
            paths[name] = root / f"{name}.eskb"
            paths[name].write_bytes(data)
        for binary in sys.argv[1:]:
            for name in ("valid", "valid-constant-pool", "valid-closure-constant",
                         "valid-closure-long-constant"):
                valid = run(binary, paths[name])
                if valid.returncode != 0:
                    raise AssertionError(f"valid ESKB {name} rejected by {binary}: "
                                         f"{valid.stderr.decode(errors='replace')}")
            for name in cases.keys() - {"valid"}:
                if name.startswith("valid-"):
                    continue
                result = run(binary, paths[name])
                if result.returncode != 1:
                    raise AssertionError(f"{binary} failed unsafely on {name}: "
                                         f"return code {result.returncode}")
                diagnostic = result.stderr.lower()
                if not any(word in diagnostic for word in
                           (b"invalid", b"opcode", b"crc32", b"failed to load",
                            b"error")):
                    raise AssertionError(f"{binary} gave no bounded diagnostic "
                                         f"for {name}: {result.stderr!r}")
                if b"ADMISSION_MARKER" in result.stdout + result.stderr:
                    raise AssertionError(f"{binary} executed rejected fixture {name}")
    print("PASS: standalone ESKB admission rejects malformed operands and constant pools")


if __name__ == "__main__":
    main()
