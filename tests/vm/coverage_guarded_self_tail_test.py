#!/usr/bin/env python3
"""Compile a traced guarded self-tail call and verify both VM dispatches."""
import hashlib
import os
import pathlib
import struct
import subprocess
import sys
import tempfile


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


def read_string(data, offset):
    size, offset = read_leb(data, offset)
    return data[offset:offset + size].decode("utf-8", "replace"), offset + size


def fnv31(name):
    value = 2166136261
    for byte in name.encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value & 0x7fffffff


def instructions_by_function(module):
    if len(module) < 16 or struct.unpack_from("<I", module)[0] != 0x45534b42:
        raise ValueError("not an ESKB module")
    payload = module[16:]
    section_count, offset = read_leb(payload, 0)
    sections = []
    for _ in range(section_count):
        section_id = payload[offset]
        offset += 1
        size, offset = read_leb(payload, offset)
        sections.append((section_id, size))
    code = None
    for section_id, size in sections:
        if section_id == 1:
            code = payload[offset:offset + size]
            break
        offset += size
    if code is None:
        raise ValueError("ESKB has no CODE section")

    offset = 0
    function_count, offset = read_leb(code, offset)
    functions = []
    for _ in range(function_count):
        name, offset = read_string(code, offset)
        offset += 1  # parameter count
        _, offset = read_leb(code, offset)  # locals
        _, offset = read_leb(code, offset)  # upvalues
        code_len, offset = read_leb(code, offset)
        instructions = []
        for _ in range(code_len):
            opcode = code[offset]
            offset += 1
            zigzag, offset = read_leb(code, offset)
            operand = -(zigzag // 2) - 1 if zigzag & 1 else zigzag // 2
            instructions.append((opcode, operand))
        functions.append((name, instructions))
    return functions


def check_trace_adjacency(module):
    self_hash = fnv31("coverage-guard-loop")
    alias_hash = fnv31("coverage-guard-loop-alias")
    if self_hash == alias_hash:
        raise AssertionError("self-call and alias hashes unexpectedly collide")
    guarded_self_tail = False
    marker_count = 0
    for function_name, instructions in instructions_by_function(module):
        for pc, (opcode, operand) in enumerate(instructions):
            if opcode != 65:  # OP_LANGUAGE_COVERAGE_CALL
                continue
            marker_count += 1
            successor = instructions[pc + 1] if pc + 1 < len(instructions) else None
            if successor is None or successor[0] not in (25, 26, 70):
                raise AssertionError(
                    f"{function_name} PC {pc}: coverage marker has invalid successor {successor}")
            if (operand == self_hash and pc > 0 and instructions[pc - 1][0] == 58 and
                    successor[0] in (26, 70)):
                guarded_self_tail = True
    if marker_count == 0:
        raise AssertionError("compiler emitted no language coverage call markers")
    if not guarded_self_tail:
        raise AssertionError("guarded self-tail marker was not placed after handler cleanup")
    return self_hash, alias_hash, marker_count


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: coverage_guarded_self_tail_test.py RUNNER VM VM SOURCE")
    repo = pathlib.Path(__file__).resolve().parents[2]
    runner = pathlib.Path(sys.argv[1]).resolve()
    vms = [pathlib.Path(sys.argv[2]).resolve(), pathlib.Path(sys.argv[3]).resolve()]
    source = pathlib.Path(sys.argv[4]).resolve()
    base_env = dict(os.environ, ESHKOL_PATH=str(repo / "lib"),
                    ESHKOL_LIB_DIR=str(runner.parent), ESHKOL_JIT_CACHE="0",
                    ESHKOL_VM_NO_DISASM="1")

    with tempfile.TemporaryDirectory(prefix="eshkol-coverage-guarded-tail-") as tmp:
        root = pathlib.Path(tmp)
        module = root / "coverage_guarded_self_tail_regression.eskb"
        compile_trace = root / "trace-compile"
        compile_trace.mkdir()
        compile_env = dict(base_env,
                           ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR=str(compile_trace))
        compiled = subprocess.run(
            [str(runner), "--profile", "hosted-vm", "--emit-eskb", str(module),
             str(source)], cwd=repo, env=compile_env, capture_output=True,
            text=True, timeout=45)
        if compiled.returncode != 0 or not module.is_file():
            raise AssertionError(f"runner failed to emit fixture ESKB:\n"
                                 f"{compiled.stdout}\n{compiled.stderr}")

        module_bytes = module.read_bytes()
        self_hash, alias_hash, marker_count = check_trace_adjacency(module_bytes)
        module_hash = hashlib.sha256(module_bytes).hexdigest()
        for index, vm in enumerate(vms):
            trace_dir = root / f"trace-vm-{index}"
            trace_dir.mkdir()
            env = dict(base_env,
                       ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR=str(trace_dir))
            result = subprocess.run([str(vm), str(module)], cwd=repo, env=env,
                                    capture_output=True, text=True, timeout=30)
            expected = "PASS: coverage guarded self-tail result 0"
            if (result.returncode != 0 or expected not in result.stdout or
                    "FAIL:" in result.stdout or "ERROR:" in result.stderr):
                raise AssertionError(f"VM failed fixture: {vm}\n"
                                     f"rc={result.returncode}\n"
                                     f"stdout:\n{result.stdout}\n"
                                     f"stderr:\n{result.stderr}")
            rows = []
            for trace in trace_dir.glob("language-coverage-*.tsv"):
                rows.extend(line.split("\t") for line in trace.read_text().splitlines())
            call_hashes = {row[4] for row in rows
                           if len(row) >= 6 and row[0] == "V" and row[5] == "@call"}
            if str(self_hash) not in call_hashes:
                raise AssertionError(f"{vm} did not emit self-call coverage hash {self_hash}")
            if str(alias_hash) not in call_hashes:
                raise AssertionError(f"{vm} did not emit alias call coverage hash {alias_hash}")

    print("PASS: guarded self-tail marker admission and runtime coverage "
          f"(markers={marker_count}, self={self_hash}, alias={alias_hash}, "
          f"module_sha256={module_hash})")


if __name__ == "__main__":
    main()
