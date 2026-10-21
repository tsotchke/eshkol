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


def eskb(opcode, operand=0):
    # One constant-free function, starting at pc 0.  Halt (36) is the valid
    # control; callers replace it with an invalid opcode or OP_CONST index.
    code = bytearray(leb(1))       # one function
    code += leb(4) + b"main"       # function name
    code += b"\0"                  # parameters
    code += leb(0) + leb(0)         # locals, upvalues
    code += leb(1)                  # code length
    zigzag = (operand << 1) ^ (operand >> 31)
    code += bytes((opcode,)) + leb(zigzag)
    constants = leb(0)
    payload = bytearray(leb(2))
    payload += bytes((0,)) + leb(len(constants))
    payload += bytes((1,)) + leb(len(code))
    payload += constants + code
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
            "valid": eskb(36),
            "invalid-opcode": eskb(255),
            "invalid-constant-index": eskb(1, 1),
            "truncated": eskb(36)[:-1],
        }
        paths = {}
        for name, data in cases.items():
            paths[name] = root / f"{name}.eskb"
            paths[name].write_bytes(data)
        for binary in sys.argv[1:]:
            valid = run(binary, paths["valid"])
            if valid.returncode != 0:
                raise AssertionError(f"valid ESKB rejected by {binary}: "
                                     f"{valid.stderr.decode(errors='replace')}")
            for name in ("invalid-opcode", "invalid-constant-index", "truncated"):
                result = run(binary, paths[name])
                if result.returncode <= 0:
                    raise AssertionError(f"{binary} failed unsafely on {name}: "
                                         f"return code {result.returncode}")
                diagnostic = result.stderr.lower()
                if not any(word in diagnostic for word in
                           (b"invalid", b"opcode", b"crc32", b"failed to load")):
                    raise AssertionError(f"{binary} gave no bounded diagnostic "
                                         f"for {name}: {result.stderr!r}")
    print("PASS: standalone ESKB admission rejects invalid opcode and operands")


if __name__ == "__main__":
    main()
