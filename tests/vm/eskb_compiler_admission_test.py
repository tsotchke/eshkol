#!/usr/bin/env python3
"""Keep compiler-produced tensor-literal ESKB modules admissible."""
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import zlib


def read_leb(data, offset):
    value = shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7f) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("truncated ESKB LEB128")


def corrupt_first_opcode(path):
    data = bytearray(path.read_bytes())
    payload = data[16:]
    section_count, cursor = read_leb(payload, 0)
    sections = []
    for _ in range(section_count):
        section_id = payload[cursor]
        cursor += 1
        section_size, cursor = read_leb(payload, cursor)
        sections.append((section_id, section_size))

    section_offset = cursor
    for section_id, section_size in sections:
        if section_id == 1:  # code section
            code = section_offset
            function_count, code = read_leb(payload, code)
            if function_count == 0:
                raise ValueError("compiler emitted no ESKB functions")
            name_size, code = read_leb(payload, code)
            code += name_size + 1  # function name and parameter count
            _, code = read_leb(payload, code)  # locals
            _, code = read_leb(payload, code)  # upvalues
            instruction_count, code = read_leb(payload, code)
            if instruction_count == 0:
                raise ValueError("compiler emitted an empty entry function")
            data[16 + code] = 255
            struct.pack_into("<I", data, 12, zlib.crc32(data[16:]) & 0xffffffff)
            path.write_bytes(data)
            return
        section_offset += section_size
    raise ValueError("compiler output has no ESKB code section")


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: eskb_compiler_admission_test.py RUNNER VM [VM ...]")
    root = pathlib.Path(__file__).resolve().parents[2]
    runner = pathlib.Path(sys.argv[1]).resolve()
    vms = [pathlib.Path(arg).resolve() for arg in sys.argv[2:]]
    source = root / "tests/v1_2_edge_cases/msgpack_test.esk"
    env = dict(os.environ, ESHKOL_PATH=str(root / "lib"),
               ESHKOL_LIB_DIR=str(runner.parent), ESHKOL_JIT_CACHE="0",
               ESHKOL_VM_NO_DISASM="1")

    with tempfile.TemporaryDirectory(prefix="eshkol-eskb-compiler-admission-") as tmp:
        module = pathlib.Path(tmp) / "msgpack_test.eskb"
        emitted = subprocess.run(
            [str(runner), "--profile", "hosted-vm", "--emit-eskb", str(module),
             str(source), "-L" + str(runner.parent)],
            cwd=root, env=env, capture_output=True, text=True, timeout=180)
        if emitted.returncode != 0 or not module.is_file():
            raise AssertionError(f"compiler failed to emit ESKB:\n{emitted.stderr}")
        valid_module = module.read_bytes()

        for vm in vms:
            module.write_bytes(valid_module)
            valid = subprocess.run([str(vm), str(module)], cwd=root, env=env,
                                   capture_output=True, text=True, timeout=30)
            if (valid.returncode != 0 or "Passed: 42" not in valid.stdout or
                    "Failed: 0" not in valid.stdout):
                raise AssertionError(f"valid compiler ESKB rejected by {vm}:\n"
                                     f"{valid.stdout}\n{valid.stderr}")

            corrupt_first_opcode(module)
            invalid = subprocess.run([str(vm), str(module)], cwd=root, env=env,
                                     capture_output=True, text=True, timeout=10)
            if invalid.returncode <= 0 or "invalid" not in invalid.stderr.lower():
                raise AssertionError(f"malformed compiler ESKB was not rejected by {vm}:\n"
                                     f"{invalid.stdout}\n{invalid.stderr}")

    print("PASS: compiler-produced tensor-literal ESKB admission and invalid-opcode rejection")


if __name__ == "__main__":
    main()
